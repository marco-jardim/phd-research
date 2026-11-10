"""Lista única de hosts de bibliotecas-sombra, compartilhada pelos coletores de texto.

Nenhum script do repositório pode requisitar esses endereços (nem por redirecionamento).
A comparação é feita por rótulos do nome do host, não por substring, para tratar
subdomínios (``pt.annas-archive.se``) e variações de domínio de topo (``libgen.is``,
``libgen.rs``) sem bloquear hosts legítimos que apenas contenham o texto.
"""

from __future__ import annotations

from urllib.parse import urlsplit

# Nome (um ou mais rótulos) seguido de qualquer domínio de topo: ``annas-archive.*``.
# Variantes de grafia que já apareciam nas listas antigas ficam junto.
NOMES_COM_TLD_VARIAVEL: tuple[str, ...] = (
    "annas-archive",
    "annasarchive",
    "anna-archive",
    "annas-blog",
    "sci-hub",
    "scihub",
    "sci.hub",
    "sci-net",
    "libgen",
    "booksc",
    "singlelogin",
    "z-lib",
    "z-library",
    "zlibrary",
    "1lib",
    "b-ok",
    "bookfi",
    "pdfdrive",
)

# Domínios exatos; qualquer subdomínio deles também é bloqueado.
DOMINIOS_EXATOS: tuple[str, ...] = (
    "gen.lib.rus.ec",
    "library.lol",
    "libstc.cc",
    "dokumen.pub",
)


def _rotulos(host: str) -> list[str]:
    return [r for r in host.strip().lower().rstrip(".").split(".") if r]


def _contem_sequencia(rotulos: list[str], nome: list[str]) -> bool:
    """O ``nome`` aparece em ``rotulos`` com ao menos um rótulo (o domínio de topo) depois dele."""
    n = len(nome)
    for i in range(len(rotulos) - n):
        trecho = rotulos[i : i + n]
        if trecho[:-1] != nome[:-1]:
            continue
        ultimo = trecho[-1]
        # ``z-lib`` casa ``z-lib`` e ``z-lib-global``, mas não ``zlib`` nem ``mylibgen``.
        if ultimo == nome[-1] or ultimo.startswith(nome[-1] + "-"):
            return True
    return False


def host_bloqueado(host: str) -> bool:
    """Verdadeiro se ``host`` (sem esquema nem porta) for de biblioteca-sombra."""
    rotulos = _rotulos(host)
    if not rotulos:
        return False
    for dominio in DOMINIOS_EXATOS:
        alvo = dominio.split(".")
        if rotulos[-len(alvo) :] == alvo:
            return True
    return any(_contem_sequencia(rotulos, nome.split(".")) for nome in NOMES_COM_TLD_VARIAVEL)


def nome_do_host(url: str) -> str:
    """Host em minúsculas de uma URL; vazio se a URL for inválida."""
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def url_bloqueada(url: str) -> bool:
    """Verdadeiro se o host da URL for de biblioteca-sombra."""
    return host_bloqueado(nome_do_host(url))
