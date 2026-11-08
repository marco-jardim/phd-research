"""Cliente HTTP mínimo (stdlib) com limite de taxa, retentativas e lista de hosts proibidos.

Somente fontes legais e abertas. Bibliotecas-sombra (Anna's Archive, Sci-Hub, LibGen
e afins) nunca são requisitadas: a verificação ocorre na URL inicial e em cada redirecionamento.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

FORBIDDEN_HOST_MARKERS = (
    "annas-archive",
    "annas-blog",
    "anna-archive",
    "sci-hub",
    "scihub",
    "sci-net",
    "libgen",
    "library.lol",
    "z-lib",
    "zlibrary",
    "z-library",
    "1lib.",
    "b-ok.",
    "bookfi",
    "pdfdrive",
)
MAX_BYTES = 40 * 1024 * 1024
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504, 529}
NOT_FOUND_STATUS = {400, 401, 403, 404, 405, 406, 410, 451}


class HttpError(Exception):
    """Falha de rede ou de servidor após as retentativas."""


class ForbiddenHostError(HttpError):
    """Tentativa de acessar um host proibido (bibliotecas-sombra)."""


def host_of(url: str) -> str:
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def is_forbidden_host(url: str) -> bool:
    host = host_of(url)
    return any(marker in host for marker in FORBIDDEN_HOST_MARKERS)


def assert_allowed(url: str) -> None:
    if is_forbidden_host(url):
        raise ForbiddenHostError(f"host proibido: {host_of(url)}")


@dataclass
class HttpResponse:
    status: int
    body: bytes
    content_type: str
    url: str

    def text(self) -> str:
        charset = "utf-8"
        for part in self.content_type.split(";")[1:]:
            name, _sep, value = part.strip().partition("=")
            if name.lower() == "charset" and value:
                charset = value.strip().strip('"')
        try:
            return self.body.decode(charset, errors="replace")
        except LookupError:
            return self.body.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.body.decode("utf-8"))


class _GuardedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        assert_allowed(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HttpClient:
    """GET com User-Agent com mailto, intervalo mínimo por host e backoff."""

    def __init__(
        self,
        mailto: str = "",
        *,
        min_interval: float = 1.0,
        retries: int = 3,
        timeout: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.mailto = mailto
        self.min_interval = min_interval
        self.retries = retries
        self.timeout = timeout
        self._sleep = sleep
        self._clock = clock
        self._last: dict[str, float] = {}
        self.last_status: int | None = None  # status da última resposta recebida (para diagnóstico)
        self._opener = urllib.request.build_opener(_GuardedRedirect)

    @property
    def user_agent(self) -> str:
        contact = f" (mailto:{self.mailto})" if self.mailto else ""
        return f"phd-research-atestado/1.0{contact}"

    def _throttle(self, host: str) -> None:
        last = self._last.get(host)
        if last is not None:
            wait = self.min_interval - (self._clock() - last)
            if wait > 0:
                self._sleep(wait)
        self._last[host] = self._clock()

    def _fetch(self, url: str, headers: dict[str, str]) -> tuple[int, bytes, str, str]:
        """Uma requisição; devolve (status, corpo, content-type, URL final)."""
        request = urllib.request.Request(url, headers=headers)
        try:
            with self._opener.open(request, timeout=self.timeout) as resp:
                body = resp.read(MAX_BYTES + 1)
                return resp.status, body, resp.headers.get("Content-Type", ""), resp.geturl()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(2048), exc.headers.get("Content-Type", ""), url

    def get(
        self, url: str, params: dict[str, str] | None = None, *, accept: str | None = None
    ) -> HttpResponse | None:
        """Resposta 200, ou None para 404 e similares. Levanta HttpError se esgotar as tentativas."""
        if params:
            url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        assert_allowed(url)
        headers = {"User-Agent": self.user_agent}
        if accept:
            headers["Accept"] = accept
        last_problem = "sem tentativa"
        self.last_status = None
        for attempt in range(self.retries + 1):
            self._throttle(host_of(url))
            try:
                status, body, ctype, final_url = self._fetch(url, headers)
            except OSError as exc:
                last_problem = f"erro de rede: {type(exc).__name__}"
            else:
                assert_allowed(final_url)
                self.last_status = status
                if status == 200:
                    if len(body) > MAX_BYTES:
                        raise HttpError(f"resposta acima do limite de {MAX_BYTES} bytes")
                    return HttpResponse(status, body, ctype, final_url)
                if status in NOT_FOUND_STATUS:
                    return None
                last_problem = f"HTTP {status}"
                if status not in RETRY_STATUS:
                    raise HttpError(f"{last_problem} em {host_of(url)}")
            if attempt < self.retries:
                self._sleep(2.0 * 2**attempt)
        raise HttpError(f"{last_problem} em {host_of(url)} após {self.retries + 1} tentativas")

    def get_json(self, url: str, params: dict[str, str] | None = None):
        resp = self.get(url, params, accept="application/json")
        if resp is None:
            return None
        try:
            return resp.json()
        except ValueError as exc:
            raise HttpError(f"JSON inválido de {host_of(url)}") from exc
