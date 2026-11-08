"""Levantamento de termos consagrados na produção da banca e da orientadora.

Objetivo: decidir, por conceito, uma forma consagrada (inglesa ou portuguesa) e,
por sigla, uma expansão, a partir do que a orientadora e os membros da banca
escrevem em seus próprios trabalhos, e comparar com o uso na tese.

Etapas (``--etapa``):

* ``autores``: resolve os identificadores OpenAlex dos cinco autores
  (``autores.json``; ``--autores-ids`` sobrescreve).
* ``corpus``: lista as obras (relacionadas a linkage, mais as ``--n-gerais``
  mais citadas de cada autor), obtém o texto em acesso aberto e grava um JSON
  por documento em ``corpus/``.
* ``contar``: conta variantes de conceitos e siglas no corpus e na tese e grava
  ``documentos.csv``, ``contagens.csv``, ``definicoes.jsonl``,
  ``tese_contagens.csv`` e ``resumo.md``.
* ``tudo`` (padrão): as três em sequência.

Texto completo apenas de fontes abertas, em ordem: SciELO (DOI 10.1590),
Europe PMC (``fullTextXML``), melhor localização aberta do Unpaywall (PDF via
``fitz`` se instalado) e, na falta delas, o resumo (sinalizado). Bibliotecas
sombra (Anna's Archive, Sci-Hub, LibGen e afins) nunca são consultadas: todo
endereço, inclusive de redirecionamento, passa por ``verificar_host``.

Saídas em ``--out`` (padrão ``.local/termos``). Os textos completos ficam só
sob ``.local/``; as saídas trazem trechos de até 240 caracteres. Todas as
propostas do resumo são PROPOSTA: a decisão é da autora.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urlencode, urljoin, urlparse

import requests
import yaml

OPENALEX = "https://api.openalex.org"
EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
UNPAYWALL = "https://api.unpaywall.org/v2"

# (chave, nome completo, rótulo de exibição)
AUTORES: list[tuple[str, str, str]] = [
    ("rejane", "Rejane Sobrino Pinheiro", "Pinheiro (orientadora)"),
    ("claudia", "Claudia Medina Coeli", "Coeli"),
    ("kenneth", "Kenneth Rochel de Camargo Jr.", "Camargo Jr."),
    ("valeria", "Valéria Menezes Bastos", "Bastos"),
    ("myrian", "Myrian Christina de Aragão Costa", "Costa"),
]
ROTULOS = {chave: rotulo for chave, _nome, rotulo in AUTORES}

TERMOS_LINKAGE = [
    "linkage",
    "record linkage",
    "relacionamento de registros",
    "relacionamento de bases",
    "relacionamento probabilistico",
    "pareamento",
    "vinculacao",
    "reclink",
    "openreclink",
    "blocagem",
    "deduplicacao",
    "deduplication",
]

TERMOS_INSTITUICAO = (
    "federal do rio de janeiro",
    "ufrj",
    "fiocruz",
    "oswaldo cruz",
    "saude coletiva",
    "iesc",
    "escola nacional de saude publica",
    "ensp",
    "estado do rio de janeiro",
    "uerj",
    "universidade federal fluminense",
)

SHADOW_TOKENS = (
    "annas-archive",
    "annasarchive",
    "sci-hub",
    "scihub",
    "sci.hub",
    "libgen",
    "z-lib",
    "zlibrary",
    "1lib",
    "b-ok",
    "bookfi",
    "library.lol",
)

MIN_CHARS = 1500
MAX_SNIPPET = 240
LADO_SNIPPET = 120
EXCLUIR_SEMPRE = {"tese_rev_diff.tex"}
EXCLUIR_CONTAGEM = {"siglas.tex", "glossario.tex"}
NAO_TEX = {".pgf", ".pdf", ".png", ".jpg", ".jpeg", ".eps", ".svg", ".tikz"}
SELECT_OBRAS = (
    "id,doi,title,publication_year,language,cited_by_count,abstract_inverted_index,"
    "ids,primary_location,best_oa_location,open_access,type"
)
POR_PAGINA = 200
MAX_PAGINAS_AUTOR = 10
TIPOS_IGNORADOS = {"erratum", "paratext", "retraction"}


# ---------------------------------------------------------------------------
# Normalização de texto
# ---------------------------------------------------------------------------
_HIFENS = set("-\u2010\u2011\u2012\u2013\u2014\u2015\u2212_")


def dobrar(texto: str) -> tuple[str, list[int]]:
    """Texto sem acentos, com hífens e espaços reduzidos a um espaço.

    Mantém a caixa. Retorna o texto dobrado e, para cada caractere dele, o
    índice do caractere correspondente em ``texto``.
    """
    saida: list[str] = []
    mapa: list[int] = []
    ultimo_espaco = True
    for i, ch in enumerate(texto):
        if ch in _HIFENS or ch.isspace():
            if not ultimo_espaco:
                saida.append(" ")
                mapa.append(i)
                ultimo_espaco = True
            continue
        for c in unicodedata.normalize("NFKD", ch):
            if unicodedata.combining(c):
                continue
            saida.append(c)
            mapa.append(i)
        ultimo_espaco = False
    return "".join(saida), mapa


def sem_acentos(texto: str) -> str:
    """Minúsculas e sem acentos (para comparações simples)."""
    return dobrar(texto)[0].lower()


def recortar(texto: str, a: int, b: int, lado: int = LADO_SNIPPET, teto: int = MAX_SNIPPET) -> str:
    """Trecho de ``texto`` em volta de [a, b), com no máximo ``teto`` caracteres."""
    lado = max(0, min(lado, (teto - (b - a)) // 2))
    trecho = texto[max(0, a - lado) : b + lado]
    return " ".join(trecho.split())[:teto]


_PT = {
    "de", "da", "do", "das", "dos", "em", "para", "e", "sobre", "por", "com", "na", "no",
    "nos", "nas", "uma", "um", "que", "se", "os", "as", "ao", "entre", "sistema", "saude",
}
_EN = {
    "of", "the", "for", "and", "in", "on", "to", "with", "by", "is", "are", "was", "were",
    "that", "this", "from", "an", "under", "between", "system", "health", "using",
}


def adivinhar_lingua(texto: str) -> str:
    """``pt``, ``en`` ou ``?``, por palavras funcionais e terminações."""
    pt = en = 0
    for t in re.findall(r"[a-z]+", sem_acentos(texto)):
        if t in _PT:
            pt += 1
        if t in _EN:
            en += 1
        if t.endswith(("cao", "coes", "dade", "mente", "ncia")):
            pt += 1
        if t.endswith(("tion", "tions", "ity", "ing", "ment")):
            en += 1
    if pt > en:
        return "pt"
    if en > pt:
        return "en"
    return "?"


def contar_palavras(texto: str) -> int:
    return len(texto.split())


def escrever_atomico(caminho: Path, dados: str | bytes) -> None:
    """Grava via arquivo temporário e ``os.replace`` com novas tentativas (Windows)."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_name(caminho.name + ".tmp")
    if isinstance(dados, str):
        tmp.write_text(dados, encoding="utf-8", newline="\n")
    else:
        tmp.write_bytes(dados)
    for tentativa in range(6):
        try:
            os.replace(tmp, caminho)
            return
        except PermissionError:
            if tentativa == 5:
                raise
            time.sleep(0.2 * (tentativa + 1))


# ---------------------------------------------------------------------------
# HTTP: cache, ritmo, tentativas e bloqueio de bibliotecas sombra
# ---------------------------------------------------------------------------
class FetchError(RuntimeError):
    """Falha ao obter um endereço depois das tentativas."""


class ShadowHostError(FetchError):
    """Endereço de biblioteca sombra: nunca consultado."""


def verificar_host(url: str) -> None:
    """Levanta ``ShadowHostError`` se o endereço for de biblioteca sombra."""
    partes = urlparse(url)
    if partes.scheme not in ("http", "https"):
        raise FetchError(f"esquema não permitido: {url}")
    host = (partes.hostname or "").lower()
    for token in SHADOW_TOKENS:
        if token in host:
            raise ShadowHostError(f"host bloqueado (biblioteca sombra): {host}")


@dataclass
class Resp:
    status: int
    content: bytes
    content_type: str = ""
    url: str = ""
    retry_after: float = 0.0

    @property
    def text(self) -> str:
        m = re.search(r"charset=([\w\-]+)", self.content_type, flags=re.I)
        enc = m.group(1) if m else "utf-8"
        try:
            return self.content.decode(enc, errors="replace")
        except LookupError:
            return self.content.decode("utf-8", errors="replace")

    def json(self) -> dict:
        return json.loads(self.text)


class Http:
    """Cliente HTTP educado: ritmo mínimo, novas tentativas com recuo e cache em disco."""

    def __init__(
        self,
        cache_dir: Path,
        session: object | None = None,
        intervalo: float = 1.0,
        tentativas: int = 4,
        recuo: float = 2.0,
        timeout: float = 40.0,
        user_agent: str = "phd-research-termos/1.0",
        dormir=time.sleep,
        relogio=time.monotonic,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.session = session if session is not None else requests.Session()
        self.intervalo = intervalo
        self.tentativas = tentativas
        self.recuo = recuo
        self.timeout = timeout
        self.user_agent = user_agent
        self.dormir = dormir
        self.relogio = relogio
        self._ultimo = None

    @staticmethod
    def _chave(url: str, params: dict | None) -> str:
        base = url + "?" + urlencode(sorted((params or {}).items()))
        return hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]

    def _ler_cache(self, chave: str) -> Resp | None:
        meta = self.cache_dir / f"{chave}.json"
        corpo = self.cache_dir / f"{chave}.bin"
        if not (meta.is_file() and corpo.is_file()):
            return None
        info = json.loads(meta.read_text(encoding="utf-8"))
        return Resp(info["status"], corpo.read_bytes(), info.get("content_type", ""), info.get("url", ""))

    def _gravar_cache(self, chave: str, resp: Resp) -> None:
        escrever_atomico(self.cache_dir / f"{chave}.bin", resp.content)
        meta = {"status": resp.status, "content_type": resp.content_type, "url": resp.url}
        escrever_atomico(self.cache_dir / f"{chave}.json", json.dumps(meta, ensure_ascii=False))

    def _esperar(self) -> None:
        agora = self.relogio()
        if self._ultimo is not None:
            falta = self.intervalo - (agora - self._ultimo)
            if falta > 0:
                self.dormir(falta)
        self._ultimo = self.relogio()

    def _uma_vez(self, url: str, params: dict | None, headers: dict | None) -> Resp:
        cab = {"User-Agent": self.user_agent}
        cab.update(headers or {})
        atual, p = url, params
        for _ in range(6):
            verificar_host(atual)
            r = self.session.get(atual, params=p, headers=cab, timeout=self.timeout, allow_redirects=False)
            destino = r.headers.get("Location") if hasattr(r, "headers") else None
            if r.status_code in (301, 302, 303, 307, 308) and destino:
                atual, p = urljoin(atual, destino), None
                continue
            try:
                espera = float(r.headers.get("Retry-After", 0))
            except (TypeError, ValueError):
                espera = 0.0
            return Resp(r.status_code, r.content, r.headers.get("Content-Type", ""), atual, espera)
        raise FetchError(f"redirecionamentos demais: {url}")

    def get(self, url: str, params: dict | None = None, headers: dict | None = None) -> Resp:
        verificar_host(url)
        chave = self._chave(url, params)
        em_cache = self._ler_cache(chave)
        if em_cache is not None:
            return em_cache
        motivo = ""
        for tentativa in range(self.tentativas):
            self._esperar()
            try:
                resp = self._uma_vez(url, params, headers)
            except requests.RequestException as exc:
                motivo = f"{type(exc).__name__}: {exc}"
                self.dormir(self.recuo**tentativa)
                continue
            if resp.status in (429, 500, 502, 503, 504):
                motivo = f"HTTP {resp.status}"
                self.dormir(min(max(resp.retry_after, self.recuo**tentativa), 60.0))
                continue
            if resp.status in (200, 404, 410):
                self._gravar_cache(chave, resp)
            return resp
        raise FetchError(f"{url}: {motivo or 'sem resposta'}")


# ---------------------------------------------------------------------------
# Autores (OpenAlex)
# ---------------------------------------------------------------------------
def id_curto(valor: str | None) -> str:
    """``https://openalex.org/A123`` -> ``A123``."""
    return (valor or "").rstrip("/").rsplit("/", 1)[-1]


_PARTICULAS = {"de", "da", "do", "das", "dos", "e", "del", "di", "jr", "junior", "filho", "neto"}


def _tokens(nome: str) -> list[str]:
    return re.findall(r"[a-z]+", sem_acentos(nome))


def _significativos(nome: str) -> list[str]:
    return [t for t in _tokens(nome) if t not in _PARTICULAS and len(t) > 1]


def compatibilidade_nome(consulta: str, nomes: list[str]) -> float:
    """Fração dos nomes significativos da consulta presente no melhor nome do candidato.

    Exige primeiro e último nome da consulta; 0 se nenhum nome do candidato os tiver.
    """
    q = _significativos(consulta)
    if len(q) < 2:
        return 0.0
    melhor = 0.0
    for nome in nomes:
        c = set(_tokens(nome))
        if q[0] in c and q[-1] in c:
            melhor = max(melhor, len(set(q) & c) / len(q))
    return melhor


def instituicoes_do_autor(cand: dict) -> list[dict]:
    insts: list[dict] = []
    for inst in cand.get("last_known_institutions") or []:
        insts.append(inst)
    li = cand.get("last_known_institution")
    if li:
        insts.append(li)
    for af in cand.get("affiliations") or []:
        if af.get("institution"):
            insts.append(af["institution"])
    return insts


def pontuar_candidato(consulta: str, cand: dict) -> dict:
    """Pontuação de um candidato a autor: nome compatível, instituição brasileira, produção."""
    nomes = [cand.get("display_name") or ""] + list(cand.get("display_name_alternatives") or [])
    compat = compatibilidade_nome(consulta, nomes)
    insts = instituicoes_do_autor(cand)
    nomes_inst = sorted({i.get("display_name", "") for i in insts if i.get("display_name")})
    texto_inst = " | ".join(sem_acentos(n) for n in nomes_inst)
    chave = any(t in texto_inst for t in TERMOS_INSTITUICAO)
    brasil = any((i.get("country_code") or "").upper() == "BR" for i in insts)
    pontos = 0.0
    if compat > 0:
        pontos = compat * 10 + (5 if chave else 0) + (2 if brasil else 0)
        pontos += math.log10((cand.get("works_count") or 0) + 1)
    return {
        "id": id_curto(cand.get("id")),
        "nome": cand.get("display_name") or "",
        "works_count": cand.get("works_count") or 0,
        "instituicoes": nomes_inst,
        "compatibilidade": round(compat, 3),
        "pontuacao": round(pontos, 4),
    }


def escolher_autor(consulta: str, candidatos: list[dict]) -> tuple[str | None, list[dict]]:
    """Escolhe o candidato de maior pontuação (nome compatível obrigatório).

    Retorna (id escolhido ou None, candidatos pontuados em ordem decrescente).
    """
    vistos: dict[str, dict] = {}
    for cand in candidatos:
        pont = pontuar_candidato(consulta, cand)
        if pont["id"] and pont["id"] not in vistos:
            vistos[pont["id"]] = pont
    ordenados = sorted(vistos.values(), key=lambda p: (-p["pontuacao"], p["id"]))
    elegiveis = [p for p in ordenados if p["compatibilidade"] > 0]
    return (elegiveis[0]["id"] if elegiveis else None), ordenados


_RE_EMAIL = re.compile(r"^[^@\s\"'<>]+@[^@\s\"'<>]+\.[^@\s\"'<>]+$")


def email_valido(email: str | None) -> str | None:
    """E-mail aparado, ou ``None`` se vazio ou de aparência inválida (ex.: ``n/d``)."""
    e = (email or "").strip().strip("\"'").strip()
    return e if _RE_EMAIL.match(e) else None


def montar_user_agent(email: str | None) -> str:
    """User-Agent com ``mailto`` apenas quando há e-mail configurado."""
    base = "phd-research-termos/1.0"
    e = email_valido(email)
    return f"{base} (mailto:{e})" if e else base


def _params_openalex(email: str | None, **extra: object) -> dict:
    params = {k: v for k, v in extra.items()}
    e = email_valido(email)
    if e:
        params["mailto"] = e
    return params


def _trecho_corpo(resp: "Resp", teto: int = 300) -> str:
    return " ".join(resp.text.split())[:teto]


def buscar_candidatos_autor(http: Http, nome: str, email: str | None) -> list[dict]:
    q = _significativos(nome)
    consultas = [nome]
    if len(q) >= 2:
        consultas.append(f"{q[0]} {q[-1]}")
    achados: list[dict] = []
    for consulta in consultas:
        resp = http.get(f"{OPENALEX}/authors", _params_openalex(email, search=consulta, **{"per-page": 25}))
        if resp.status != 200:
            continue
        achados.extend(resp.json().get("results", []))
    return achados


def resolver_autores(
    http: Http, email: str | None, overrides: dict[str, list[str]] | None = None
) -> dict:
    """Resolve o(s) identificador(es) OpenAlex de cada autor; ``overrides`` tem precedência."""
    overrides = overrides or {}
    escolhidos: dict[str, list[str]] = {}
    candidatos: dict[str, list[dict]] = {}
    for chave, nome, _rotulo in AUTORES:
        achados = buscar_candidatos_autor(http, nome, email)
        escolhido, pontuados = escolher_autor(nome, achados)
        candidatos[chave] = pontuados[:10]
        if chave in overrides:
            escolhidos[chave] = list(overrides[chave])
        elif escolhido:
            escolhidos[chave] = [escolhido]
        else:
            escolhidos[chave] = []
    return {
        "nota": "Edite 'escolhidos' para corrigir; um autor pode ter mais de um perfil no OpenAlex.",
        "escolhidos": escolhidos,
        "candidatos": candidatos,
    }


def interpretar_overrides(valores: list[str] | None) -> dict[str, list[str]]:
    """``claudia=A123+A456`` (também aceita URLs do OpenAlex e vírgulas) -> dict."""
    saida: dict[str, list[str]] = {}
    for bruto in valores or []:
        for item in bruto.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"use chave=ID (ex.: claudia=A123): {item}")
            chave, ids = item.split("=", 1)
            chave = chave.strip().lower()
            if chave not in ROTULOS:
                raise ValueError(f"autor desconhecido: {chave} (válidos: {', '.join(ROTULOS)})")
            saida[chave] = [id_curto(x.strip()) for x in ids.split("+") if x.strip()]
    return saida


# ---------------------------------------------------------------------------
# Obras (OpenAlex)
# ---------------------------------------------------------------------------
def reconstruir_resumo(indice: dict | None) -> str:
    if not indice:
        return ""
    posicoes: list[tuple[int, str]] = []
    for palavra, lugares in indice.items():
        for p in lugares:
            posicoes.append((p, palavra))
    return " ".join(p for _pos, p in sorted(posicoes))


def normalizar_doi(doi: str | None) -> str:
    d = (doi or "").strip()
    d = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", d, flags=re.I)
    return d.lower()


def normalizar_obra(w: dict) -> dict:
    """Campos usados do registro de obra do OpenAlex."""
    primaria = w.get("primary_location") or {}
    fonte = primaria.get("source") or {}
    melhor = w.get("best_oa_location") or {}
    ids = w.get("ids") or {}
    pmcid = ""
    m = re.search(r"PMC\d+", ids.get("pmcid") or "")
    if m:
        pmcid = m.group(0)
    urls = [u for u in (melhor.get("pdf_url"), melhor.get("landing_page_url")) if u]
    return {
        "id": id_curto(w.get("id")),
        "doi": normalizar_doi(w.get("doi")),
        "titulo": " ".join((w.get("title") or "").split()),
        "ano": w.get("publication_year"),
        "venue": fonte.get("display_name") or "",
        "lingua": w.get("language") or "",
        "citacoes": w.get("cited_by_count") or 0,
        "resumo": reconstruir_resumo(w.get("abstract_inverted_index")),
        "tipo_obra": w.get("type") or "",
        "pmcid": pmcid,
        "urls_oa": urls,
    }


def eh_obra_linkage(titulo: str, resumo: str) -> bool:
    """Título ou resumo com algum termo relacionado a linkage (sem acentos, início de palavra)."""
    texto = sem_acentos(f"{titulo} {resumo}")
    for termo in TERMOS_LINKAGE:
        if re.search(r"(?<![a-z0-9])" + re.escape(termo), texto):
            return True
    return False


def listar_obras_autor(http: Http, ids: list[str], email: str | None, max_paginas: int = 25) -> list[dict]:
    """Todas as obras dos perfis ``ids`` (paginação por cursor)."""
    if not ids:
        return []
    filtro = "author.id:" + "|".join(ids)
    chave_api = (os.environ.get("OPENALEX_API_KEY") or "").strip()
    obras: list[dict] = []
    # Paginação básica (page=N): cursor= exige chave de API no OpenAlex.
    for pagina in range(1, min(max_paginas, MAX_PAGINAS_AUTOR) + 1):
        params = _params_openalex(
            email, filter=filtro, select=SELECT_OBRAS, page=pagina, **{"per-page": POR_PAGINA}
        )
        if chave_api:
            params["api_key"] = chave_api
        resp = http.get(f"{OPENALEX}/works", params)
        if resp.status in (401, 403):
            # Recusa de autenticação: tenta de novo sem os parâmetros opcionais (mailto, depois select).
            semi = {k: v for k, v in params.items() if k != "mailto"}
            semis = {k: v for k, v in semi.items() if k != "select"}
            for alt in (semi, semis):
                if alt == params:
                    continue
                resp = http.get(f"{OPENALEX}/works", alt)
                if resp.status not in (401, 403):
                    break
        if resp.status != 200:
            raise FetchError(
                f"OpenAlex /works devolveu HTTP {resp.status} para {filtro}: {_trecho_corpo(resp)}"
            )
        dados = resp.json()
        for w in dados.get("results", []):
            obra = normalizar_obra(w)
            if obra["tipo_obra"] in TIPOS_IGNORADOS or not obra["id"]:
                continue
            obras.append(obra)
        resultados = dados.get("results") or []
        total = (dados.get("meta") or {}).get("count")
        if len(resultados) < POR_PAGINA or (isinstance(total, int) and pagina * POR_PAGINA >= total):
            break
    return obras


def selecionar_obras(obras_por_autor: dict[str, list[dict]], n_gerais: int) -> list[dict]:
    """Obras de linkage de cada autor mais as ``n_gerais`` mais citadas das demais; sem duplicatas.

    O campo ``autores`` de cada obra lista todos os autores da banca em cujas listas ela aparece.
    """
    autores_da_obra: dict[str, set[str]] = defaultdict(set)
    base: dict[str, dict] = {}
    for chave, obras in obras_por_autor.items():
        for o in obras:
            autores_da_obra[o["id"]].add(chave)
            base.setdefault(o["id"], o)
    escolhidas: dict[str, str] = {}
    for chave in obras_por_autor:
        obras = obras_por_autor[chave]
        for o in obras:
            if eh_obra_linkage(o["titulo"], o["resumo"]):
                escolhidas[o["id"]] = "linkage"
        gerais = [o for o in obras if not eh_obra_linkage(o["titulo"], o["resumo"])]
        gerais.sort(key=lambda o: (-o["citacoes"], o["id"]))
        for o in gerais[: max(0, n_gerais)]:
            escolhidas.setdefault(o["id"], "geral")
    saida = []
    for oid in sorted(escolhidas):
        obra = dict(base[oid])
        obra["tipo_selecao"] = escolhidas[oid]
        obra["autores"] = sorted(autores_da_obra[oid])
        saida.append(obra)
    return saida


# ---------------------------------------------------------------------------
# Texto completo em acesso aberto
# ---------------------------------------------------------------------------
_RE_REFERENCIAS = re.compile(
    r"^\s*(?:\d+[.)]?\s*)?(?:referencias(?:\s+bibliograficas)?|references?|bibliografia|"
    r"bibliography|literatura\s+citada)\s*:?\s*$",
    flags=re.I,
)


def cortar_referencias(texto: str) -> str:
    """Corta o texto no último título de referências situado depois de 30% do texto."""
    corte = None
    pos = 0
    for linha in texto.split("\n"):
        if _RE_REFERENCIAS.match(sem_acentos(linha)) and pos >= 0.3 * len(texto):
            corte = pos
        pos += len(linha) + 1
    return texto[:corte].rstrip() if corte is not None else texto


_VAZIOS = {"br", "hr", "img", "meta", "link", "input", "area", "base", "col", "embed", "source", "track", "wbr"}
_PULAR = {"script", "style", "noscript", "nav", "header", "footer", "aside", "button", "svg", "select", "head"}
_BLOCOS = {
    "p", "div", "section", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table",
    "article", "blockquote", "ul", "ol", "figcaption", "caption", "main", "hr",
}


class _No:
    __slots__ = ("tag", "attrs", "filhos", "pai")

    def __init__(self, tag: str, attrs: dict, pai: "_No | None") -> None:
        self.tag = tag
        self.attrs = attrs
        self.filhos: list = []
        self.pai = pai


class _Arvore(HTMLParser):
    """Árvore mínima e tolerante a HTML malformado."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.raiz = _No("[raiz]", {}, None)
        self.atual = self.raiz

    def handle_starttag(self, tag, attrs):
        no = _No(tag, {k: (v or "") for k, v in attrs}, self.atual)
        self.atual.filhos.append(no)
        if tag not in _VAZIOS:
            self.atual = no

    def handle_startendtag(self, tag, attrs):
        self.atual.filhos.append(_No(tag, {k: (v or "") for k, v in attrs}, self.atual))

    def handle_endtag(self, tag):
        no = self.atual
        while no is not None and no.tag != tag:
            no = no.pai
        if no is not None and no is not self.raiz:
            self.atual = no.pai

    def handle_data(self, data):
        self.atual.filhos.append(data)


def _texto_no(no: _No) -> str:
    partes: list[str] = []
    pilha: list = [no]
    while pilha:
        item = pilha.pop()
        if isinstance(item, str):
            partes.append(item)
            continue
        if item.tag in _PULAR:
            continue
        if item.tag in _BLOCOS:
            partes.append("\n")
            pilha.append("\n")
        pilha.extend(reversed(item.filhos))
    linhas = [" ".join(linha.split()) for linha in "".join(partes).split("\n")]
    return "\n".join(linha for linha in linhas if linha)


def _achar(raiz: _No, pred) -> _No | None:
    fila = deque([raiz])
    while fila:
        no = fila.popleft()
        if no is not raiz and pred(no):
            return no
        fila.extend(f for f in no.filhos if isinstance(f, _No))
    return None


_CLASSES_CORPO = {"articletxt", "articletext", "article-body", "articlebody", "article__body", "fulltext", "full-text"}
_IDS_CORPO = {"article-body", "articletext", "standalonearticle"}


def _seletores_corpo():
    def por_classe(no: _No) -> bool:
        classes = set((no.attrs.get("class") or "").lower().split())
        return bool(classes & _CLASSES_CORPO) or (no.attrs.get("id") or "").lower() in _IDS_CORPO

    return [
        lambda no: (no.attrs.get("itemprop") or "").lower() == "articlebody",
        por_classe,
        lambda no: no.tag == "article",
        lambda no: no.tag == "main",
        lambda no: no.tag == "body",
    ]


def extrair_html(html: str, minimo: int = 200) -> str:
    """Texto do corpo do artigo em HTML (SciELO e similares), sem referências."""
    arvore = _Arvore()
    arvore.feed(html)
    arvore.close()
    for seletor in _seletores_corpo():
        no = _achar(arvore.raiz, seletor)
        if no is None:
            continue
        texto = _texto_no(no)
        if len(texto) >= minimo:
            return cortar_referencias(texto)
    return ""


def extrair_europepmc(xml: bytes | str) -> str:
    """Texto do ``<body>`` do XML de texto completo do Europe PMC (sem lista de referências)."""
    dados = xml.encode("utf-8") if isinstance(xml, str) else xml
    try:
        raiz = ET.fromstring(dados)
    except ET.ParseError:
        return ""

    def nome(el) -> str:
        return el.tag.rsplit("}", 1)[-1] if isinstance(el.tag, str) else ""

    corpo = next((el for el in raiz.iter() if nome(el) == "body"), None)
    if corpo is None:
        return ""
    linhas: list[str] = []

    def visitar(el) -> None:
        n = nome(el)
        if n in {"ref-list", "fn-group", "ack"}:
            return
        if n in {"p", "title"}:
            t = " ".join("".join(el.itertext()).split())
            if t:
                linhas.append(t)
            return
        for filho in el:
            visitar(filho)

    visitar(corpo)
    return "\n".join(linhas)


def extrair_pdf(conteudo: bytes) -> str:
    """Texto de PDF via ``fitz`` (PyMuPDF), se instalado; senão string vazia."""
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError:
        return ""
    try:
        with fitz.open(stream=conteudo, filetype="pdf") as doc:
            texto = "\n".join(pagina.get_text() for pagina in doc)
    except (RuntimeError, ValueError):
        return ""
    return re.sub(r"(\w)-\n(\w)", r"\1\2", texto)


def texto_scielo(http: Http, doi: str) -> str:
    resp = http.get(f"https://doi.org/{quote(doi, safe='/')}", headers={"Accept": "text/html"})
    if resp.status != 200:
        return ""
    return extrair_html(resp.text, minimo=MIN_CHARS)


def pmcid_europepmc(http: Http, doi: str) -> str:
    resp = http.get(
        f"{EUROPEPMC}/search",
        {"query": f'DOI:"{doi}"', "format": "json", "resultType": "lite", "pageSize": 1},
    )
    if resp.status != 200:
        return ""
    for r in resp.json().get("resultList", {}).get("result", []):
        if r.get("pmcid") and r.get("isOpenAccess") == "Y":
            return r["pmcid"]
    return ""


def texto_europepmc(http: Http, pmcid: str) -> str:
    resp = http.get(f"{EUROPEPMC}/{pmcid}/fullTextXML")
    if resp.status != 200:
        return ""
    return cortar_referencias(extrair_europepmc(resp.content))


def urls_unpaywall(http: Http, doi: str, email: str) -> list[str]:
    resp = http.get(f"{UNPAYWALL}/{quote(doi, safe='/')}", {"email": email})
    if resp.status != 200:
        return []
    dados = resp.json()
    locais = [dados.get("best_oa_location")] + list(dados.get("oa_locations") or [])
    urls: list[str] = []
    for loc in locais:
        if not loc:
            continue
        for campo in ("url_for_pdf", "url_for_landing_page", "url"):
            u = loc.get(campo)
            if u and u not in urls:
                urls.append(u)
    return urls


def texto_de_urls(http: Http, urls: list[str], avisos: list[str], limite: int = 4) -> str:
    for url in urls[:limite]:
        try:
            resp = http.get(url)
        except FetchError as exc:
            avisos.append(f"url ignorada: {exc}")
            continue
        if resp.status != 200:
            continue
        tipo = resp.content_type.lower()
        if resp.content[:5] == b"%PDF" or "pdf" in tipo:
            texto = cortar_referencias(extrair_pdf(resp.content))
        elif "html" in tipo or resp.content.lstrip()[:1] == b"<":
            texto = extrair_html(resp.text, minimo=MIN_CHARS)
        else:
            continue
        if len(texto) >= MIN_CHARS:
            return texto
    return ""


def obter_texto(http: Http, obra: dict, email: str | None) -> tuple[str, str, list[str]]:
    """Texto da obra, fonte e avisos. Apenas fontes abertas; resumo como último recurso."""
    avisos: list[str] = []
    doi = obra.get("doi") or ""

    def tentar(nome: str, funcao) -> str:
        try:
            return funcao()
        except FetchError as exc:
            avisos.append(f"{nome}: {exc}")
        except (ValueError, KeyError) as exc:
            avisos.append(f"{nome}: resposta inesperada ({type(exc).__name__})")
        return ""

    if doi.startswith("10.1590/"):
        texto = tentar("scielo", lambda: texto_scielo(http, doi))
        if len(texto) >= MIN_CHARS:
            return texto, "scielo", avisos
    pmcid = obra.get("pmcid") or ""
    if not pmcid and doi:
        pmcid = tentar("europepmc_busca", lambda: pmcid_europepmc(http, doi))
    if pmcid:
        texto = tentar("europepmc", lambda: texto_europepmc(http, pmcid))
        if len(texto) >= MIN_CHARS:
            return texto, "europepmc", avisos
    if doi and email:
        urls = tentar("unpaywall", lambda: urls_unpaywall(http, doi, email)) or []
        texto = texto_de_urls(http, urls, avisos)
        if texto:
            return texto, "unpaywall", avisos
    elif obra.get("urls_oa"):
        texto = texto_de_urls(http, list(obra["urls_oa"]), avisos)
        if texto:
            return texto, "openalex_oa", avisos
    if obra.get("resumo"):
        return obra["resumo"], "resumo", avisos
    return "", "nenhum", avisos


def construir_corpus(
    http: Http,
    obras: list[dict],
    corpus_dir: Path,
    email: str | None,
    refazer: bool = False,
    log=lambda msg: None,
) -> list[Path]:
    """Grava um JSON por obra em ``corpus_dir``; obras já gravadas são mantidas."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    gravados: list[Path] = []
    for i, obra in enumerate(obras, start=1):
        caminho = corpus_dir / f"{obra['id']}.json"
        if caminho.is_file() and not refazer:
            gravados.append(caminho)
            continue
        texto, fonte, avisos = obter_texto(http, obra, email)
        lingua = obra.get("lingua") or adivinhar_lingua(" ".join(texto.split()[:3000]))
        doc = {
            "id": obra["id"],
            "doi": obra.get("doi", ""),
            "ano": obra.get("ano"),
            "titulo": obra.get("titulo", ""),
            "venue": obra.get("venue", ""),
            "lingua": lingua,
            "autores": obra.get("autores", []),
            "tipo_selecao": obra.get("tipo_selecao", ""),
            "fonte_texto": fonte,
            "avisos": avisos,
            "texto": texto,
        }
        escrever_atomico(caminho, json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True))
        gravados.append(caminho)
        log(f"[{i}/{len(obras)}] {obra['id']} {fonte} ({contar_palavras(texto)} palavras)")
    return gravados


def carregar_corpus(corpus_dir: Path) -> list[dict]:
    docs = []
    for caminho in sorted(Path(corpus_dir).glob("*.json")):
        doc = json.loads(caminho.read_text(encoding="utf-8"))
        doc["n_palavras"] = contar_palavras(doc.get("texto", ""))
        docs.append(doc)
    docs.sort(key=lambda d: (d.get("ano") or 9999, d["id"]))
    return docs


# ---------------------------------------------------------------------------
# Conceitos e variantes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Variante:
    conceito: str
    forma: str
    lingua: str
    sensivel: bool
    regex: re.Pattern
    exato: re.Pattern


@dataclass
class Conceito:
    nome: str
    guia: str
    notas: str
    indices: list[int] = field(default_factory=list)


def _padrao_palavra(palavra: str, sensivel: bool) -> str:
    if sensivel:
        return re.escape(palavra)
    p = palavra.lower()
    if len(p) > 3 and p.endswith("ao"):
        return re.escape(p[:-2]) + "(?:ao|oes|aes)"
    if len(p) > 3 and p.endswith("y"):
        return re.escape(p[:-1]) + "(?:y|ies)"
    if len(p) >= 3:
        return re.escape(p) + "(?:s|es)?"
    return re.escape(p)


def eh_sensivel(forma: str) -> bool:
    """Formas só em maiúsculas (VPP, F1) são comparadas com distinção de caixa."""
    return any(c.isupper() for c in forma) and not any(c.islower() for c in forma)


def criar_variante(conceito: str, forma: str, lingua: str) -> Variante:
    sensivel = eh_sensivel(forma)
    dobrada = dobrar(forma)[0].strip()
    corpo = " ".join(_padrao_palavra(p, sensivel) for p in dobrada.split(" "))
    flags = 0 if sensivel else re.IGNORECASE
    return Variante(
        conceito,
        forma,
        lingua,
        sensivel,
        re.compile(r"(?<![A-Za-z0-9])" + corpo + r"(?![A-Za-z0-9])", flags),
        re.compile(corpo, flags),
    )


def carregar_candidatos(caminho: Path) -> tuple[list[Conceito], list[Variante]]:
    """Lê o YAML de candidatos; devolve conceitos e a lista plana de variantes."""
    bruto = yaml.safe_load(Path(caminho).read_text(encoding="utf-8")) or []
    conceitos: list[Conceito] = []
    variantes: list[Variante] = []
    nomes: set[str] = set()
    for item in bruto:
        nome = str(item["conceito"]).strip()
        if nome in nomes:
            raise ValueError(f"conceito repetido: {nome}")
        nomes.add(nome)
        c = Conceito(nome, str(item.get("guia") or ""), str(item.get("notas") or ""))
        vistas: set[str] = set()
        for v in item["variantes"]:
            forma, lingua = str(v["forma"]).strip(), str(v["lingua"]).strip().lower()
            if lingua not in ("pt", "en"):
                raise ValueError(f"lingua inválida em {nome}/{forma}: {lingua}")
            chave = dobrar(forma)[0].strip().lower() if not eh_sensivel(forma) else dobrar(forma)[0].strip()
            if chave in vistas:
                continue
            vistas.add(chave)
            c.indices.append(len(variantes))
            variantes.append(criar_variante(nome, forma, lingua))
        conceitos.append(c)
    return conceitos, variantes


# ---------------------------------------------------------------------------
# Contagem de variantes (maior casamento primeiro, sem sobreposição)
# ---------------------------------------------------------------------------
@dataclass
class Analise:
    texto: str
    dobrado: str
    mapa: list[int]
    casamentos: list[tuple[int, int, int]]  # (início, fim, índice da variante) no texto dobrado

    def contagens(self) -> Counter:
        return Counter(vi for _a, _b, vi in self.casamentos)

    def original(self, a: int, b: int) -> tuple[int, int]:
        return self.mapa[a], self.mapa[b - 1] + 1

    def trecho(self, a: int, b: int) -> str:
        oa, ob = self.original(a, b)
        return recortar(self.texto, oa, ob)


def analisar(texto: str, variantes: list[Variante]) -> Analise:
    dobrado, mapa = dobrar(texto)
    candidatos: list[tuple[int, int, int, int]] = []
    for vi, v in enumerate(variantes):
        for m in v.regex.finditer(dobrado):
            candidatos.append((m.end() - m.start(), m.start(), vi, m.end()))
    candidatos.sort(key=lambda c: (-c[0], c[1], c[2]))
    ocupado = bytearray(len(dobrado))
    aceitos: list[tuple[int, int, int]] = []
    for tam, a, vi, b in candidatos:
        if any(ocupado[a:b]):
            continue
        ocupado[a:b] = b"\x01" * tam
        aceitos.append((a, b, vi))
    aceitos.sort()
    return Analise(texto, dobrado, mapa, aceitos)


_RE_PAREN = re.compile(r"\s*\(\s*([^()]{1,100}?)\s*\)")
_RE_PREFIXO = re.compile(
    r"^(?:(?:em|do|in)\s+(?:ingl[eê]s|portugu[eê]s|english|portuguese)|ingl[eê]s|portugu[eê]s|english|portuguese|"
    r"tamb[eé]m\s+(?:denominado|chamado|conhecido(?:\s+como)?)|tambem\s+(?:denominado|chamado|conhecido(?:\s+como)?)|"
    r"denominado|chamado|conhecido\s+como|"
    r"known\s+as|also\s+(?:called|known\s+as)|ou|or|i\s*e|e\s*g|sigla|cf)\s*[:,.]?\s+",
    flags=re.I,
)


def _limpar_interno(interno: str) -> str:
    interno = interno.strip(" \"'\u201c\u201d\u2018\u2019*.,;:")
    for _ in range(2):
        novo = _RE_PREFIXO.sub("", interno, count=1)
        if novo == interno:
            break
        interno = novo.strip(" \"'\u201c\u201d\u2018\u2019*.,;:")
    return interno


def definicoes_de_variantes(analise: Analise, variantes: list[Variante], origem: str) -> list[dict]:
    """Casos ``X (Y)`` em que X e Y são variantes diferentes do mesmo conceito."""
    por_conceito: dict[str, list[int]] = defaultdict(list)
    for vi, v in enumerate(variantes):
        por_conceito[v.conceito].append(vi)
    achados: list[dict] = []
    for a, b, vi in analise.casamentos:
        m = _RE_PAREN.match(analise.dobrado, b)
        if not m:
            continue
        interno = _limpar_interno(m.group(1))
        if not interno:
            continue
        v1 = variantes[vi]
        melhor = None
        for vj in por_conceito[v1.conceito]:
            if vj == vi:
                continue
            if variantes[vj].exato.fullmatch(interno):
                melhor = vj
                break
        if melhor is None:
            continue
        v2 = variantes[melhor]
        oa, ob = analise.original(a, m.end())
        achados.append(
            {
                "origem": origem,
                "tipo": "variantes",
                "conceito": v1.conceito,
                "principal": v1.forma,
                "lingua_principal": v1.lingua,
                "entre_parenteses": v2.forma,
                "lingua_parenteses": v2.lingua,
                "trecho": recortar(analise.texto, oa, ob),
            }
        )
    return achados


# ---------------------------------------------------------------------------
# Siglas
# ---------------------------------------------------------------------------
_PARADA = {
    "de", "da", "do", "das", "dos", "e", "em", "of", "the", "for", "and", "in", "on", "to",
    "para", "por", "sobre", "with", "by", "a", "o", "as", "os", "na", "no",
}
_RE_SIGLA_ENTRE = re.compile(r"\(\s*([A-Z][A-Za-z0-9\-]{1,14})\s*\)")
_RE_SIGLA_ABRE = re.compile(r"(?<![\w\-])([A-Z][A-Za-z0-9\-]{1,14})\s*\(\s*([^()]{4,140}?)\s*\)")


def _letras_sigla(sigla: str) -> list[str]:
    return [c for c in sigla if c.isalpha() and c.isupper()]


def _iniciais(token: str) -> list[str]:
    """Inicial do token e maiúsculas internas (``OpenRecLink`` -> O, R, L), sem acentos."""
    letras = [unicodedata.normalize("NFKD", c)[0] for c in token if c.isalpha()]
    if not letras:
        return []
    return [letras[0].upper()] + [c for c in letras[1:] if c.isupper()]


def _casa_sigla(tokens: list[str], letras: list[str]) -> bool:
    if not tokens or len(letras) < 2:
        return False
    ini = _iniciais(tokens[0])
    if not ini or ini[0] != letras[0]:
        return False
    sequencia = iter(c for t in tokens for c in _iniciais(t))
    return all(c in sequencia for c in letras)


def _limpar_token(token: str) -> str:
    return token.strip(" \"'\u201c\u201d\u2018\u2019*,;:")


def _expansao_antes(texto: str, pos: int, letras: list[str]) -> list[str] | None:
    antes = texto[max(0, pos - 220) : pos]
    corte = max(antes.rfind("("), antes.rfind(")"))
    if corte >= 0:
        antes = antes[corte + 1 :]
    tokens: list[str] = []
    for tok in reversed(re.findall(r"\S+", antes)):
        if tokens and tok[-1] in ".;:!?":
            break
        tokens.append(tok)
        if len(tokens) >= len(letras) + 6:
            break
    tokens.reverse()
    for k in range(1, len(tokens) + 1):
        janela = tokens[-k:]
        if _casa_sigla(janela, letras):
            return [_limpar_token(t) for t in janela]
    return None


def detectar_siglas(texto: str, origem: str) -> list[dict]:
    """Definições ``Expansão (SIGLA)`` e ``SIGLA (Expansão)`` cujas iniciais conferem."""
    achados: list[dict] = []
    for m in _RE_SIGLA_ENTRE.finditer(texto):
        sigla = m.group(1)
        letras = _letras_sigla(sigla)
        if len(letras) < 2:
            continue
        janela = _expansao_antes(texto, m.start(), letras)
        if janela is None:
            continue
        expansao = " ".join(janela)
        achados.append(
            {
                "origem": origem,
                "tipo": "sigla",
                "ordem": "expansao_sigla",
                "sigla": sigla,
                "expansao": expansao,
                "lingua_expansao": adivinhar_lingua(expansao),
                "trecho": recortar(texto, m.start() - len(expansao) - 1, m.end()),
            }
        )
    for m in _RE_SIGLA_ABRE.finditer(texto):
        sigla = m.group(1)
        letras = _letras_sigla(sigla)
        if len(letras) < 2:
            continue
        interno = _limpar_interno(m.group(2))
        tokens = [_limpar_token(t) for t in interno.split()]
        tokens = [t for t in tokens if t]
        if not tokens or len(tokens) > len(letras) + 6 or not _casa_sigla(tokens, letras):
            continue
        expansao = " ".join(tokens)
        achados.append(
            {
                "origem": origem,
                "tipo": "sigla",
                "ordem": "sigla_expansao",
                "sigla": sigla,
                "expansao": expansao,
                "lingua_expansao": adivinhar_lingua(expansao),
                "trecho": recortar(texto, m.start(), m.end()),
            }
        )
    return achados


def compilar_siglas(siglas) -> re.Pattern | None:
    ordenadas = sorted(set(siglas), key=lambda s: (-len(s), s))
    if not ordenadas:
        return None
    alternativas = "|".join(re.escape(s) for s in ordenadas)
    return re.compile(r"(?<![\w\-])(" + alternativas + r")s?(?![\w\-])")


def contar_siglas(texto: str, padrao: re.Pattern | None) -> Counter:
    if padrao is None:
        return Counter()
    return Counter(m.group(1) for m in padrao.finditer(texto))


def ler_siglas_tex(caminho: Path) -> list[dict]:
    """Siglas e expansões de ``siglas.tex``: [{sigla, expansao_en, expansao_pt, expansao}]."""
    bruto = Path(caminho).read_text(encoding="utf-8-sig")
    codigo = "\n".join(linha_sem_comentario(l) for l in bruto.splitlines())
    ini = codigo.find("\\begin{longtable}")
    fim = codigo.find("\\end{longtable}")
    if ini < 0 or fim < 0:
        return []
    corpo = codigo[ini + len("\\begin{longtable}") : fim]
    corpo = corpo.lstrip()
    if corpo.startswith("{"):
        e = _fecha_chave(corpo, 0)
        corpo = corpo[e + 1 :] if e >= 0 else corpo
    saidas: list[dict] = []
    for linha in re.split(r"\\\\(?:\[[^\]]*\])?", corpo):
        if "&" not in linha:
            continue
        sigla_bruta, resto = linha.split("&", 1)
        sigla = latex_para_texto(sigla_bruta).strip()
        resto = resto.strip()
        if not sigla or not resto:
            continue
        en, pt = "", ""
        if resto.startswith("\\textit{"):
            e = _fecha_chave(resto, len("\\textit"))
            if e > 0:
                en = latex_para_texto(resto[len("\\textit{") : e]).strip()
                pt = latex_para_texto(resto[e + 1 :]).strip().lstrip(":").strip()
            else:
                pt = latex_para_texto(resto)
        else:
            pt = latex_para_texto(resto)
        saidas.append(
            {"sigla": sigla, "expansao_en": en, "expansao_pt": pt, "expansao": latex_para_texto(resto)}
        )
    return saidas


# ---------------------------------------------------------------------------
# Leitura da tese (LaTeX -> texto)
# ---------------------------------------------------------------------------
def linha_sem_comentario(linha: str) -> str:
    """Remove o comentário LaTeX (``%`` não escapado) de uma linha."""
    barras = 0
    for i, ch in enumerate(linha):
        if ch == "\\":
            barras += 1
            continue
        if ch == "%" and barras % 2 == 0:
            return linha[:i]
        barras = 0
    return linha


def _fecha_chave(s: str, i: int) -> int:
    """Índice da chave que fecha a ``{`` em ``s[i]``; -1 se não fechar."""
    prof = 0
    j = i
    while j < len(s):
        ch = s[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "{":
            prof += 1
        elif ch == "}":
            prof -= 1
            if prof == 0:
                return j
        j += 1
    return -1


_ACENTOS = {"'": "\u0301", "`": "\u0300", "^": "\u0302", "~": "\u0303", '"': "\u0308"}
_RE_ACENTO = re.compile(r"\\(['`^~\"])\s*(?:\{\s*\\?([A-Za-z])\s*\}|\\?([A-Za-z]))")
_RE_CEDILHA = re.compile(r"\\c\s*(?:\{\s*([A-Za-z])\s*\}|\s+([A-Za-z]))")


def acentos_latex(s: str) -> str:
    """``\\c{c}\\~{a}`` -> ``ç ã`` (Unicode NFC)."""

    def acento(m: re.Match) -> str:
        letra = m.group(2) or m.group(3)
        return letra + _ACENTOS[m.group(1)]

    def cedilha(m: re.Match) -> str:
        return (m.group(1) or m.group(2)) + "\u0327"

    s = _RE_ACENTO.sub(acento, s)
    s = _RE_CEDILHA.sub(cedilha, s)
    s = re.sub(r"\\i(?![A-Za-z])", "i", s)
    return unicodedata.normalize("NFC", s)


# nome do comando -> (número de grupos {..} consumidos, índice do grupo mantido ou None)
_COMANDOS = {
    "path": (1, None), "url": (1, None), "label": (1, None), "ref": (1, None), "autoref": (1, None),
    "pageref": (1, None), "cref": (1, None), "eqref": (1, None), "nameref": (1, None),
    "includegraphics": (1, None), "bibliography": (1, None), "bibliographystyle": (1, None),
    "addbibresource": (1, None), "vspace": (1, None), "hspace": (1, None), "setlength": (2, None),
    "index": (1, None), "cline": (1, None), "input": (1, None), "include": (1, None),
    "color": (1, None), "cellcolor": (1, None), "rowcolor": (1, None),
    "textcolor": (2, 1), "multicolumn": (3, 2), "multirow": (3, 2), "href": (2, 1),
    "resizebox": (3, 2), "scalebox": (2, 1), "captionof": (2, 1),
    "cite": (1, None), "citeonline": (1, None), "citep": (1, None), "citet": (1, None),
    "citeauthor": (1, None), "citeauthoronline": (1, None), "citeyear": (1, None),
    "citeyearpar": (1, None), "parencite": (1, None), "textcite": (1, None), "autocite": (1, None),
    "apud": (2, None), "apudonline": (2, None),
}
_TABULAR_LIKE = {"tabular", "longtable", "tabularx", "tabulary", "tabular*", "minipage", "longtabu"}
_RE_COMANDO = re.compile(r"\\([A-Za-z]+)(\*?)")
_COMANDOS_ESTRUTURAIS = {
    "item", "par", "newline", "quad", "qquad", "hfill", "vfill", "noindent", "medskip", "bigskip",
    "smallskip", "centering", "hline", "toprule", "midrule", "bottomrule", "endhead", "endfirsthead",
    "endfoot", "endlastfoot", "clearpage", "newpage", "chapter", "section", "subsection",
    "subsubsection", "paragraph", "caption", "footnote",
}
_AMBIENTES_MATEMATICOS = ("equation", "align", "gather", "multline", "eqnarray", "displaymath", "math", "flalign", "alignat")


def _pular_opcionais(s: str, j: int) -> int:
    while True:
        k = j
        while k < len(s) and s[k] in " \t":
            k += 1
        if k < len(s) and s[k] == "[":
            depth = 0
            m = k
            while m < len(s):
                if s[m] == "[":
                    depth += 1
                elif s[m] == "]":
                    depth -= 1
                    if depth == 0:
                        break
                m += 1
            if m >= len(s):
                return j
            j = m + 1
            continue
        return j


def _ler_grupo(s: str, j: int) -> tuple[str, int] | None:
    k = j
    while k < len(s) and s[k] in " \t\n":
        k += 1
    if k < len(s) and s[k] == "{":
        e = _fecha_chave(s, k)
        if e >= 0:
            return s[k + 1 : e], e + 1
    return None


def _processar_comandos(texto: str) -> str:
    saida: list[str] = []
    i = 0
    while True:
        m = _RE_COMANDO.search(texto, i)
        if not m:
            saida.append(texto[i:])
            break
        saida.append(texto[i : m.start()])
        nome = m.group(1)
        fim = m.end()
        if nome in ("begin", "end"):
            g = _ler_grupo(texto, fim)
            if g is None:
                i = fim
                continue
            ambiente, k = g[0].strip(), g[1]
            if nome == "begin" and ambiente in _TABULAR_LIKE:
                k = _pular_opcionais(texto, k)
                g2 = _ler_grupo(texto, k)
                if g2 is not None:
                    k = g2[1]
            saida.append(" ")
            i = k
            continue
        especificacao = _COMANDOS.get(nome)
        if especificacao is None:
            saida.append(texto[m.start() : m.end()])
            i = fim
            continue
        n_grupos, mantido = especificacao
        k = _pular_opcionais(texto, fim)
        grupos: list[str] = []
        for idx in range(n_grupos):
            g = _ler_grupo(texto, k)
            if g is None:
                break
            grupos.append(g[0])
            k = g[1] if idx == n_grupos - 1 else _pular_opcionais(texto, g[1])
        if mantido is not None and mantido < len(grupos):
            saida.append(_processar_comandos(grupos[mantido]))
        else:
            saida.append(" ")
        i = k
    return "".join(saida)


def latex_para_texto(bruto: str) -> str:
    """Prosa de um trecho LaTeX: sem comentários, matemática, ``\\path``, ``\\cite`` e comandos."""
    t = "\n".join(linha_sem_comentario(l) for l in bruto.splitlines())
    t = re.sub(r"\\\\(?:\[[^\]\n]*\])?", "\n", t)
    t = (
        t.replace("\\{", "\ue001")
        .replace("\\}", "\ue002")
        .replace("\\&", "\ue003")
        .replace("\\%", "\ue004")
        .replace("\\$", "\ue005")
        .replace("\\_", "_")
        .replace("\\#", "#")
    )
    for amb in _AMBIENTES_MATEMATICOS:
        t = re.sub(rf"\\begin\{{{amb}\*?\}}.*?\\end\{{{amb}\*?\}}", " ", t, flags=re.S)
    t = re.sub(r"\$\$.*?\$\$", " ", t, flags=re.S)
    t = re.sub(r"\\\[.*?\\\]", " ", t, flags=re.S)
    t = re.sub(r"\\\(.*?\\\)", " ", t, flags=re.S)
    t = re.sub(r"\$[^$]{0,600}\$", " ", t, flags=re.S)
    t = acentos_latex(t)
    t = t.replace("\\textordfeminine", "\u00aa").replace("\\textordmasculine", "\u00ba")
    t = t.replace("\\ldots", "\u2026").replace("\\dots", "\u2026")
    t = _processar_comandos(t)
    t = _RE_COMANDO.sub(lambda m: " " if m.group(1) in _COMANDOS_ESTRUTURAIS else "", t)
    t = t.replace("&", " | ").replace("~", " ").replace("``", '"').replace("''", '"')
    t = t.replace("{", "").replace("}", "").replace("$", "").replace("\\", " ")
    t = (
        t.replace("\ue001", "{")
        .replace("\ue002", "}")
        .replace("\ue003", "&")
        .replace("\ue004", "%")
        .replace("\ue005", "$")
    )
    return " ".join(t.split())


_RE_INCLUDE = re.compile(r"\\(?:input|include)\s*\{([^}]*)\}")


def _achatar(caminho: Path, raiz: Path, excluir: set[str], pilha: tuple[Path, ...], avisos: list[str], arquivos: list[str]) -> str:
    try:
        bruto = caminho.read_text(encoding="utf-8-sig")
    except OSError as exc:
        avisos.append(f"não foi possível ler {caminho}: {exc}")
        return ""
    arquivos.append(caminho.name)
    pilha = pilha + (caminho.resolve(),)
    partes: list[str] = []
    for linha in bruto.splitlines():
        codigo = linha_sem_comentario(linha)
        pos = 0
        for m in _RE_INCLUDE.finditer(codigo):
            partes.append(codigo[pos : m.start()])
            partes.append(_incluir(m.group(1), raiz, excluir, pilha, avisos, arquivos))
            pos = m.end()
        partes.append(codigo[pos:] + "\n")
    return "".join(partes)


def _incluir(alvo: str, raiz: Path, excluir: set[str], pilha, avisos: list[str], arquivos: list[str]) -> str:
    nome = alvo.strip()
    if not nome:
        return ""
    p = Path(nome)
    if p.suffix.lower() in NAO_TEX or "figures" in p.parts:
        return ""
    if p.suffix.lower() != ".tex":
        p = Path(nome + ".tex")
    if p.name in EXCLUIR_SEMPRE or p.name in excluir:
        return "\n"
    caminho = raiz / p
    if caminho.resolve() in pilha:
        avisos.append(f"inclusão cíclica ignorada: {alvo}")
        return ""
    if not caminho.is_file():
        avisos.append(f"arquivo incluído não encontrado: {alvo}")
        return ""
    return _achatar(caminho, raiz, excluir, pilha, avisos, arquivos)


def ler_tese(tex: Path, excluir: set[str] | None = None) -> tuple[str, list[str], list[str]]:
    """Prosa da tese seguindo ``\\input``/``\\include`` a partir de ``tex``.

    Retorna (texto, arquivos lidos, avisos). ``tese_rev_diff.tex`` é sempre ignorado;
    ``excluir`` (padrão: siglas.tex e glossario.tex) são arquivos de definições.
    """
    excluir = set(EXCLUIR_CONTAGEM if excluir is None else excluir)
    tex = Path(tex).resolve()
    avisos: list[str] = []
    arquivos: list[str] = []
    achatado = _achatar(tex, tex.parent, excluir, (), avisos, arquivos)
    ini = achatado.find("\\begin{document}")
    if ini >= 0:
        achatado = achatado[ini + len("\\begin{document}") :]
    fim = achatado.find("\\end{document}")
    if fim >= 0:
        achatado = achatado[:fim]
    return latex_para_texto(achatado), arquivos, avisos


# ---------------------------------------------------------------------------
# Proposta de forma padrão
# ---------------------------------------------------------------------------
def propor_forma(estatisticas: list[dict]) -> dict:
    """Escolhe a forma padrão (PROPOSTA).

    ``estatisticas``: itens com ``forma``, ``lingua``, ``docs`` e ``total``. Regra:
    (1) a forma usada em mais documentos; (2) em empate, a forma em português;
    (3) persistindo o empate, a de mais ocorrências e, por fim, a ordem alfabética.
    Forma inglesa proposta deve ser mantida em itálico.
    """
    usados = [e for e in estatisticas if e["docs"] > 0]
    if not usados:
        return {"forma": None, "regra": "sem_uso", "italico": False}
    maximo = max(e["docs"] for e in usados)
    topo = [e for e in usados if e["docs"] == maximo]
    if len(topo) == 1:
        escolhida, regra = topo[0], "maioria"
    else:
        pt = [e for e in topo if e["lingua"] == "pt"]
        base = pt if pt else topo
        regra = "empate_portugues" if pt else "empate_total"
        escolhida = sorted(base, key=lambda e: (-e["total"], e["forma"]))[0]
    return {"forma": escolhida["forma"], "regra": regra, "italico": escolhida["lingua"] == "en"}


REGRAS = {
    "maioria": "(1) forma usada em mais documentos da banca e da orientadora",
    "empate_portugues": "(2) empate em documentos, desempatado pela forma em português",
    "empate_total": "(2) empate em documentos sem forma em português, desempatado pelo total de ocorrências",
    "sem_uso": "sem uso no corpus",
}


# ---------------------------------------------------------------------------
# Execução da contagem e saídas
# ---------------------------------------------------------------------------
def _escrever_csv(caminho: Path, colunas: list[str], linhas: list[list]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(colunas)
        w.writerows(linhas)


def _escrever_jsonl(caminho: Path, registros: list[dict]) -> None:
    linhas = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in registros]
    caminho.write_text("\n".join(linhas) + ("\n" if linhas else ""), encoding="utf-8", newline="\n")


def _rotulos(chaves) -> str:
    return ", ".join(ROTULOS.get(c, c) for c in sorted(chaves))


def executar_contagem(
    out_dir: Path,
    candidatos: Path,
    tese_tex: Path | None,
    siglas_tex: Path | None,
    so_texto_completo: bool = False,
) -> dict:
    """Conta variantes e siglas no corpus e na tese; grava as saídas e o resumo."""
    out_dir = Path(out_dir)
    conceitos, variantes = carregar_candidatos(candidatos)
    docs = carregar_corpus(out_dir / "corpus")
    if so_texto_completo:
        docs = [d for d in docs if d.get("fonte_texto") not in ("resumo", "nenhum")]

    siglas_conhecidas = ler_siglas_tex(siglas_tex) if siglas_tex and Path(siglas_tex).is_file() else []

    # --- corpus: variantes
    contagens_doc: dict[str, Counter] = {}
    definicoes: list[dict] = []
    exemplos: dict[tuple[str, str], tuple[str, str, object, str]] = {}
    for d in docs:
        an = analisar(d.get("texto", ""), variantes)
        contagens_doc[d["id"]] = an.contagens()
        definicoes.extend(definicoes_de_variantes(an, variantes, d["id"]))
        vistos_autor: set[tuple[str, str]] = set()
        for a, b, vi in an.casamentos:
            conceito = variantes[vi].conceito
            for autor in d.get("autores", []):
                if (autor, conceito) in exemplos or (autor, conceito) in vistos_autor:
                    continue
                vistos_autor.add((autor, conceito))
                exemplos[(autor, conceito)] = (variantes[vi].forma, d["id"], d.get("ano"), an.trecho(a, b))
        definicoes.extend(detectar_siglas(d.get("texto", ""), d["id"]))

    # --- tese
    tese_texto, tese_arquivos, tese_avisos = ("", [], [])
    if tese_tex and Path(tese_tex).is_file():
        tese_texto, tese_arquivos, tese_avisos = ler_tese(Path(tese_tex))
    an_tese = analisar(tese_texto, variantes)
    contagem_tese = an_tese.contagens()
    definicoes.extend(definicoes_de_variantes(an_tese, variantes, "tese"))
    definicoes_tese_siglas = detectar_siglas(tese_texto, "tese")
    definicoes.extend(definicoes_tese_siglas)

    # --- siglas
    detectadas_corpus = {d["sigla"] for d in definicoes if d["tipo"] == "sigla" and d["origem"] != "tese"}
    detectadas_tese = {d["sigla"] for d in definicoes_tese_siglas}
    universo = {s["sigla"] for s in siglas_conhecidas} | detectadas_corpus | detectadas_tese
    padrao_siglas = compilar_siglas(universo)
    siglas_doc = {d["id"]: contar_siglas(d.get("texto", ""), padrao_siglas) for d in docs}
    siglas_tese = contar_siglas(tese_texto, padrao_siglas)

    # --- saídas
    docs_meta = [
        [
            ";".join(ROTULOS.get(a, a) for a in d.get("autores", [])),
            d["id"],
            d.get("doi", ""),
            d.get("ano") or "",
            d.get("titulo", ""),
            d.get("venue", ""),
            d.get("lingua", ""),
            d.get("fonte_texto", ""),
            d["n_palavras"],
            d.get("tipo_selecao", ""),
        ]
        for d in docs
    ]
    _escrever_csv(
        out_dir / "documentos.csv",
        ["autores", "openalex_id", "doi", "ano", "titulo", "venue", "lingua", "fonte_texto", "n_palavras", "tipo_selecao"],
        docs_meta,
    )
    linhas = []
    for d in docs:
        for vi, n in contagens_doc[d["id"]].items():
            v = variantes[vi]
            linhas.append([v.conceito, v.forma, v.lingua, d["id"], n])
        for sigla, n in siglas_doc[d["id"]].items():
            linhas.append([f"sigla:{sigla}", sigla, "sigla", d["id"], n])
    linhas.sort(key=lambda r: (r[0], r[1], r[3]))
    _escrever_csv(out_dir / "contagens.csv", ["conceito", "variante", "lingua", "documento", "contagem"], linhas)

    linhas_t = [[variantes[vi].conceito, variantes[vi].forma, variantes[vi].lingua, n] for vi, n in contagem_tese.items()]
    linhas_t += [[f"sigla:{s}", s, "sigla", n] for s, n in siglas_tese.items()]
    linhas_t.sort(key=lambda r: (r[0], r[1]))
    _escrever_csv(out_dir / "tese_contagens.csv", ["conceito", "variante", "lingua", "contagem"], linhas_t)

    definicoes.sort(
        key=lambda r: (
            r["origem"], r["tipo"], r.get("conceito", r.get("sigla", "")),
            r.get("principal", r.get("expansao", "")), r["trecho"],
        )
    )
    _escrever_jsonl(out_dir / "definicoes.jsonl", definicoes)

    resumo = montar_resumo(
        conceitos, variantes, docs, contagens_doc, exemplos, definicoes, contagem_tese,
        siglas_conhecidas, siglas_doc, siglas_tese, tese_arquivos, tese_avisos, out_dir,
    )
    (out_dir / "resumo.md").write_text(resumo, encoding="utf-8", newline="\n")
    return {
        "docs": docs,
        "contagens_doc": contagens_doc,
        "contagem_tese": contagem_tese,
        "definicoes": definicoes,
        "siglas_doc": siglas_doc,
        "siglas_tese": siglas_tese,
        "tese_avisos": tese_avisos,
    }


def estatisticas_por_variante(variantes, docs, contagens_doc) -> dict[int, dict]:
    est: dict[int, dict] = {i: {"docs": set(), "autores": set(), "total": 0} for i in range(len(variantes))}
    for d in docs:
        for vi, n in contagens_doc[d["id"]].items():
            est[vi]["docs"].add(d["id"])
            est[vi]["autores"].update(d.get("autores", []))
            est[vi]["total"] += n
    return est


def montar_resumo(
    conceitos, variantes, docs, contagens_doc, exemplos, definicoes, contagem_tese,
    siglas_conhecidas, siglas_doc, siglas_tese, tese_arquivos, tese_avisos, out_dir,
) -> str:
    est = estatisticas_por_variante(variantes, docs, contagens_doc)
    fontes = Counter(d.get("fonte_texto", "") for d in docs)
    L: list[str] = [
        "# Termos consagrados na banca e na orientadora: levantamento e propostas",
        "",
        "Gerado por `scripts/termos_consagrados.py`. Todas as formas indicadas como PROPOSTA são "
        "sugestões para decisão da autora; nada foi alterado na tese.",
        "",
        "## Corpus",
        "",
        f"- Documentos: {len(docs)}",
        "- Fonte do texto: " + (", ".join(f"{k or '(vazio)'} {v}" for k, v in sorted(fontes.items())) or "nenhuma"),
        "- Documentos só com resumo (`resumo`) ou sem texto (`nenhum`) pesam pouco nas contagens; "
        "use `--so-texto-completo` para excluí-los.",
        f"- Arquivos da tese lidos: {len(tese_arquivos)} (siglas.tex e glossario.tex ficam fora da "
        "contagem por serem arquivos de definições; tese_rev_diff.tex é ignorado)",
        "",
        "Regra da proposta: (1) forma usada em mais documentos da banca e da orientadora; "
        "(2) em caso de empate, a forma em português; (3) forma inglesa consagrada fica em itálico "
        "(por exemplo, `\\textit{linkage}`). Persistindo o empate, vale o total de ocorrências.",
        "",
        "## Conceitos",
        "",
    ]
    sem_uso: list[str] = []
    for c in conceitos:
        vis = c.indices
        n_docs_conceito = len({d for vi in vis for d in est[vi]["docs"]})
        uso_tese = {vi: contagem_tese.get(vi, 0) for vi in vis}
        if n_docs_conceito == 0 and not any(uso_tese.values()):
            sem_uso.append(c.nome)
            continue
        L += [f"### {c.nome}", ""]
        if c.guia:
            L += [f"Guia de estilo atual: {c.guia}.", ""]
        if c.notas:
            L += [c.notas, ""]
        L += [
            "| Forma | Língua | Banca/orientadora: documentos | Banca/orientadora: autores | "
            "Banca/orientadora: ocorrências | Tese: ocorrências |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        estat_prop = []
        for vi in vis:
            v, e = variantes[vi], est[vi]
            L.append(
                f"| {v.forma} | {v.lingua} | {len(e['docs'])} | {len(e['autores'])} | {e['total']} | {uso_tese[vi]} |"
            )
            estat_prop.append({"forma": v.forma, "lingua": v.lingua, "docs": len(e["docs"]), "total": e["total"]})
        L.append("")
        defs = Counter(
            (d["principal"], d["entre_parenteses"]) for d in definicoes
            if d["tipo"] == "variantes" and d["conceito"] == c.nome and d["origem"] != "tese"
        )
        if defs:
            L.append("Formas apresentadas uma com a outra entre parênteses na banca/orientadora:")
            L += [f"- {p} ({q}): {n}" for (p, q), n in sorted(defs.items())]
            L.append("")
        defs_tese = Counter(
            (d["principal"], d["entre_parenteses"]) for d in definicoes
            if d["tipo"] == "variantes" and d["conceito"] == c.nome and d["origem"] == "tese"
        )
        if defs_tese:
            L.append("Formas apresentadas uma com a outra entre parênteses na tese:")
            L += [f"- {p} ({q}): {n}" for (p, q), n in sorted(defs_tese.items())]
            L.append("")
        ex_autores = [(a, exemplos[(a, c.nome)]) for a in ROTULOS if (a, c.nome) in exemplos]
        if ex_autores:
            L.append("Primeiro uso por autor (trecho):")
            for a, (forma, doc_id, ano, trecho) in ex_autores:
                L.append(f"- {ROTULOS[a]}, {doc_id} ({ano or 's.d.'}), forma \"{forma}\": \"{trecho}\"")
            L.append("")
        prop = propor_forma(estat_prop)
        if prop["forma"]:
            marca = " (em itálico, estrangeirismo consagrado)" if prop["italico"] else ""
            L.append(f"**PROPOSTA:** {prop['forma']}{marca}. Regra aplicada: {REGRAS[prop['regra']]}.")
            if c.guia and sem_acentos(c.guia) != sem_acentos(prop["forma"]):
                L.append(f"Divergência com o guia de estilo atual ({c.guia}): decisão da autora.")
        else:
            L.append("**PROPOSTA:** sem proposta; nenhuma variante aparece no corpus. Manter o guia de estilo.")
        usadas_tese = [variantes[vi].forma for vi in vis if uso_tese[vi] > 0]
        if len(usadas_tese) >= 2:
            L.append(
                "**ALERTA:** a tese usa 2 ou mais formas para o conceito: "
                + "; ".join(f"{variantes[vi].forma} ({uso_tese[vi]})" for vi in vis if uso_tese[vi] > 0)
                + "."
            )
        L.append("")
    if sem_uso:
        L += ["### Conceitos sem ocorrência no corpus nem na tese", ""]
        L += [f"- {n}" for n in sem_uso]
        L.append("")

    L += _resumo_siglas(siglas_conhecidas, docs, siglas_doc, siglas_tese, definicoes)
    if tese_avisos:
        L += ["## Avisos da leitura da tese", ""] + [f"- {a}" for a in tese_avisos] + [""]
    return "\n".join(L).rstrip("\n") + "\n"


def _resumo_siglas(siglas_conhecidas, docs, siglas_doc, siglas_tese, definicoes) -> list[str]:
    conhecidas = {s["sigla"]: s for s in siglas_conhecidas}
    uso_corpus: dict[str, dict] = defaultdict(lambda: {"docs": set(), "autores": set(), "total": 0})
    for d in docs:
        for s, n in siglas_doc[d["id"]].items():
            uso_corpus[s]["docs"].add(d["id"])
            uso_corpus[s]["autores"].update(d.get("autores", []))
            uso_corpus[s]["total"] += n
    exp_corpus: dict[str, dict[str, dict]] = defaultdict(dict)
    exp_tese: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for df in definicoes:
        if df["tipo"] != "sigla":
            continue
        chave = sem_acentos(df["expansao"])
        if df["origem"] == "tese":
            exp_tese[df["sigla"]][df["expansao"]] += 1
            continue
        item = exp_corpus[df["sigla"]].setdefault(
            chave, {"forma": df["expansao"], "lingua": df["lingua_expansao"], "docs": set(), "total": 0}
        )
        item["docs"].add(df["origem"])
        item["total"] += 1
    siglas = sorted(set(conhecidas) | set(uso_corpus) | set(exp_corpus) | set(exp_tese) | set(siglas_tese))
    L = ["## Siglas", ""]
    L += [
        "| Sigla | Corpus: documentos | Corpus: autores | Corpus: ocorrências | Tese: ocorrências | "
        "Expansões na banca/orientadora | Expansões na tese | Lista de siglas da tese | PROPOSTA |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    alertas: list[str] = []
    for s in siglas:
        u = uso_corpus.get(s)
        n_corpus = u["total"] if u else 0
        n_tese = siglas_tese.get(s, 0)
        if not (n_corpus or n_tese or exp_corpus.get(s) or exp_tese.get(s)):
            continue
        ec = exp_corpus.get(s, {})
        et = exp_tese.get(s, {})
        estat = [
            {"forma": e["forma"], "lingua": e["lingua"] if e["lingua"] != "?" else "en", "docs": len(e["docs"]), "total": e["total"]}
            for e in ec.values()
        ]
        prop = propor_forma(estat)
        lista = conhecidas.get(s)
        texto_lista = (lista["expansao"] if lista else "") or ""
        if prop["forma"]:
            marca = " (itálico)" if prop["italico"] else ""
            proposta = f"{prop['forma']}{marca}; {REGRAS[prop['regra']]}"
        elif lista:
            proposta = "sem evidência no corpus; manter a expansão da lista de siglas"
        else:
            proposta = "sem evidência no corpus; sem expansão registrada"
        exp_c_txt = "; ".join(
            f"{e['forma']} ({len(e['docs'])} doc.)" for e in sorted(ec.values(), key=lambda x: (-len(x["docs"]), x["forma"]))
        )
        exp_t_txt = "; ".join(f"{k} ({v})" for k, v in sorted(et.items()))
        L.append(
            f"| {s} | {len(u['docs']) if u else 0} | {len(u['autores']) if u else 0} | {n_corpus} | {n_tese} | "
            f"{exp_c_txt or '-'} | {exp_t_txt or '-'} | {texto_lista.replace('|', '/')[:100] or '-'} | {proposta} |"
        )
        if len({sem_acentos(k) for k in et}) >= 2:
            alertas.append(f"{s}: a tese apresenta expansões diferentes ({'; '.join(sorted(et))})")
        if len(ec) >= 2:
            alertas.append(f"{s}: a banca/orientadora usa expansões diferentes ({'; '.join(e['forma'] for e in ec.values())})")
    L.append("")
    if alertas:
        L += ["### Alertas de siglas", ""] + [f"- {a}" for a in sorted(alertas)] + [""]
    return L


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------
def email_do_ambiente() -> str | None:
    return email_valido(os.environ.get("UNPAYWALL_EMAIL")) or email_valido(os.environ.get("CROSSREF_MAILTO"))


def etapa_autores(http: Http, out_dir: Path, email: str | None, overrides: dict, refazer: bool) -> dict[str, list[str]]:
    caminho = out_dir / "autores.json"
    if caminho.is_file() and not refazer and not overrides:
        return json.loads(caminho.read_text(encoding="utf-8"))["escolhidos"]
    dados = resolver_autores(http, email, overrides)
    escrever_atomico(caminho, json.dumps(dados, ensure_ascii=False, indent=1, sort_keys=True))
    return dados["escolhidos"]


def etapa_corpus(http: Http, out_dir: Path, escolhidos: dict[str, list[str]], n_gerais: int, email: str | None, refazer: bool) -> list[Path]:
    obras_por_autor: dict[str, list[dict]] = {}
    for chave, _nome, _rot in AUTORES:
        obras_por_autor[chave] = listar_obras_autor(http, escolhidos.get(chave, []), email)
    obras = selecionar_obras(obras_por_autor, n_gerais)
    print(f"obras selecionadas: {len(obras)}", file=sys.stderr)
    return construir_corpus(
        http, obras, out_dir / "corpus", email, refazer, log=lambda m: print(m, file=sys.stderr)
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Termos e siglas consagrados na banca e na orientadora.")
    ap.add_argument("--etapa", choices=["autores", "corpus", "contar", "tudo"], default="tudo")
    ap.add_argument("--out", default=".local/termos", help="diretório de saída")
    ap.add_argument("--candidatos", default="scripts/termos_consagrados_candidatos.yaml")
    ap.add_argument("--tese", default="text/latex/tese.tex")
    ap.add_argument("--siglas", default="text/latex/siglas.tex")
    ap.add_argument("--autores-ids", action="append", help="chave=ID[+ID], ex.: claudia=A123 (repetível)")
    ap.add_argument("--n-gerais", type=int, default=8, help="obras gerais mais citadas por autor")
    ap.add_argument("--refazer", action="store_true", help="baixar de novo obras e autores já gravados")
    ap.add_argument("--so-texto-completo", action="store_true", help="ignorar documentos só com resumo")
    ap.add_argument("--intervalo", type=float, default=1.0, help="segundos entre requisições")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    email = email_do_ambiente()
    if args.etapa in ("autores", "corpus", "tudo"):
        if not email:
            print("aviso: UNPAYWALL_EMAIL/CROSSREF_MAILTO ausente; Unpaywall não será consultado", file=sys.stderr)
        http = Http(out_dir / "cache", intervalo=args.intervalo, user_agent=montar_user_agent(email))
        try:
            overrides = interpretar_overrides(args.autores_ids)
        except ValueError as exc:
            print(f"erro: {exc}", file=sys.stderr)
            return 2
        escolhidos = etapa_autores(http, out_dir, email, overrides, args.refazer)
        print("autores: " + "; ".join(f"{k}={'+'.join(v) or '?'}" for k, v in sorted(escolhidos.items())), file=sys.stderr)
        if args.etapa in ("corpus", "tudo"):
            try:
                etapa_corpus(http, out_dir, escolhidos, args.n_gerais, email, args.refazer)
            except FetchError as exc:
                print(f"erro ao montar o corpus: {exc}", file=sys.stderr)
                return 1
    if args.etapa in ("contar", "tudo"):
        r = executar_contagem(
            out_dir, Path(args.candidatos), Path(args.tese), Path(args.siglas), args.so_texto_completo
        )
        for aviso in r["tese_avisos"]:
            print(f"aviso: {aviso}", file=sys.stderr)
        print(f"documentos: {len(r['docs'])}; definições: {len(r['definicoes'])}; saída: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
