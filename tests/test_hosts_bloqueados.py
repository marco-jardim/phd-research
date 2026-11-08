"""Lista compartilhada de hosts de bibliotecas-sombra e seu uso pelos coletores."""

import pytest

from scripts import hosts_bloqueados as hb
from scripts.atestado import http as net

BLOQUEADOS = [
    "annas-archive.org",
    "annas-archive.se",
    "pt.annas-archive.li",
    "sci-hub.se",
    "www.sci-hub.ru",
    "sci.hub.st",
    "libgen.is",
    "libgen.rs",
    "www.libgen.li",
    "gen.lib.rus.ec",
    "mirror.gen.lib.rus.ec",
    "library.lol",
    "www.library.lol",
    "libstc.cc",
    "booksc.org",
    "booksc.xyz",
    "singlelogin.re",
    "singlelogin.zlibrary.to",
    "z-lib.org",
    "z-lib.fm",
    "zlibrary.to",
    "zlibrary-global.se",
    "1lib.sk",
    "b-ok.cc",
    "b-ok.global",
    "dokumen.pub",
    "www.dokumen.pub",
    "LibGen.IS",
    "libgen.is.",
]

LIBERADOS = [
    "www.scielo.br",
    "doi.org",
    "europepmc.org",
    "api.openalex.org",
    "library.example.org",
    "notlibrary.lol",
    "mylibgen.org",
    "zlib.net",
    "dokumen.pub.example.org",
    "libstc.cc.example.org",
    "",
]


@pytest.mark.parametrize("host", BLOQUEADOS)
def test_host_bloqueado(host):
    assert hb.host_bloqueado(host)
    assert hb.url_bloqueada(f"https://{host}/x?y=1")


@pytest.mark.parametrize("host", LIBERADOS)
def test_host_liberado(host):
    assert not hb.host_bloqueado(host)


def test_url_invalida_nao_levanta():
    assert hb.url_bloqueada("http://[invalido/x") is False
    assert hb.url_bloqueada("") is False


def test_nome_na_trilha_da_url_nao_bloqueia():
    assert not hb.url_bloqueada("https://www.scielo.br/libgen/sci-hub/annas-archive")


@pytest.mark.parametrize("host", BLOQUEADOS[:-2])
def test_cliente_atestado_usa_a_lista_compartilhada(host):
    assert net.is_forbidden_host(f"https://{host}/x")
    with pytest.raises(net.ForbiddenHostError):
        net.assert_allowed(f"https://{host}/x")


def test_cliente_atestado_nao_tem_lista_local():
    assert not hasattr(net, "FORBIDDEN_HOST_MARKERS")


def test_termos_consagrados_usa_a_lista_compartilhada():
    from scripts import termos_consagrados as tc

    assert not hasattr(tc, "SHADOW_TOKENS")
    assert tc.host_bloqueado is hb.host_bloqueado
    with pytest.raises(tc.ShadowHostError):
        tc.verificar_host("https://booksc.org/x")
