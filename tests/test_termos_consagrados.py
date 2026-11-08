"""Testes de scripts/termos_consagrados.py: todo HTTP é simulado, fixtures em tmp_path."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest
import requests

from scripts import termos_consagrados as tc
from scripts.hosts_bloqueados import url_bloqueada

RAIZ = Path(__file__).resolve().parents[1]
YAML_REAL = RAIZ / "scripts" / "termos_consagrados_candidatos.yaml"


# ---------------------------------------------------------------------------
# Dublês de HTTP
# ---------------------------------------------------------------------------
class FakeResp:
    def __init__(self, status=200, content=b"", headers=None):
        self.status_code = status
        self.content = content
        self.headers = headers or {}


def resp_json(obj, status=200):
    return FakeResp(status, json.dumps(obj).encode("utf-8"), {"Content-Type": "application/json"})


def resp_html(html, status=200):
    return FakeResp(status, html.encode("utf-8"), {"Content-Type": "text/html; charset=utf-8"})


class FakeSession:
    """Sessão que responde por URL (sem parâmetros) e registra tudo o que foi pedido."""

    def __init__(self, respostas=None):
        self.respostas = respostas or {}
        self.calls: list[str] = []

    def get(self, url, params=None, headers=None, timeout=None, allow_redirects=True):
        self.calls.append(url)
        r = self.respostas.get(url)
        if isinstance(r, list):
            r = r.pop(0)
        elif callable(r):
            r = r()
        return r if r is not None else FakeResp(404)


def novo_http(tmp_path, respostas=None, dormir=None):
    sessao = FakeSession(respostas)
    http = tc.Http(
        tmp_path / "cache", session=sessao, intervalo=0, recuo=2, dormir=dormir or (lambda s: None)
    )
    return http, sessao


def var(forma, conceito="c", lingua="pt"):
    return tc.criar_variante(conceito, forma, lingua)


# ---------------------------------------------------------------------------
# Seleção de autores
# ---------------------------------------------------------------------------
UFRJ = {"display_name": "Universidade Federal do Rio de Janeiro", "country_code": "BR"}


def cand(oid, nome, works=10, insts=None, alt=None):
    return {
        "id": f"https://openalex.org/{oid}",
        "display_name": nome,
        "display_name_alternatives": alt or [],
        "works_count": works,
        "last_known_institutions": insts or [],
    }


def test_escolher_autor_prefere_nome_completo_e_instituicao_brasileira():
    candidatos = [
        cand("A7", "Claudia Coeli", works=900, insts=[{"display_name": "Univ X", "country_code": "US"}]),
        cand("A1", "Claudia Medina Coeli", works=120, insts=[UFRJ], alt=["Cláudia Coeli"]),
        cand("A3", "Maria Silva", works=5000, insts=[UFRJ]),
    ]
    escolhido, pontuados = tc.escolher_autor("Claudia Medina Coeli", candidatos)
    assert escolhido == "A1"
    assert [p["id"] for p in pontuados][0] == "A1"
    inelegivel = next(p for p in pontuados if p["id"] == "A3")
    assert inelegivel["compatibilidade"] == 0


def test_escolher_autor_desempata_por_brasil_antes_de_producao():
    candidatos = [
        cand("A10", "Valéria Menezes Bastos", works=300, insts=[{"display_name": "Univ Y", "country_code": "PT"}]),
        cand("A11", "Valéria Menezes Bastos", works=40, insts=[{"display_name": "Fundação Oswaldo Cruz", "country_code": "BR"}]),
    ]
    escolhido, _ = tc.escolher_autor("Valéria Menezes Bastos", candidatos)
    assert escolhido == "A11"


def test_escolher_autor_sem_candidato_compativel():
    escolhido, pontuados = tc.escolher_autor("Rejane Sobrino Pinheiro", [cand("A1", "João Pinheiro")])
    assert escolhido is None
    assert pontuados[0]["compatibilidade"] == 0


def test_resolver_autores_com_override_e_candidatos_gravaveis(tmp_path):
    resultados = [
        cand("A1", "Rejane Sobrino Pinheiro", 80, [UFRJ]),
        cand("A2", "Claudia Medina Coeli", 120, [UFRJ], alt=["Cláudia Coeli"]),
        cand("A3", "Kenneth Camargo", 200, [UFRJ], alt=["Kenneth Rochel de Camargo Jr."]),
        cand("A4", "Valéria Menezes Bastos", 60, [UFRJ]),
        cand("A5", "Myrian Christina de Aragão Costa", 70, [UFRJ]),
        cand("A6", "Myrian Costa", 900, [{"display_name": "Univ Z", "country_code": "US"}]),
    ]
    http, _ = novo_http(tmp_path, {f"{tc.OPENALEX}/authors": lambda: resp_json({"results": resultados})})
    dados = tc.resolver_autores(http, None, overrides={"kenneth": ["A99"]})
    assert dados["escolhidos"] == {
        "rejane": ["A1"],
        "claudia": ["A2"],
        "kenneth": ["A99"],
        "valeria": ["A4"],
        "myrian": ["A5"],
    }
    assert dados["candidatos"]["myrian"][0]["id"] == "A5"
    json.dumps(dados)


def test_interpretar_overrides():
    r = tc.interpretar_overrides(["claudia=https://openalex.org/A123+A456", "kenneth=A9"])
    assert r == {"claudia": ["A123", "A456"], "kenneth": ["A9"]}
    with pytest.raises(ValueError):
        tc.interpretar_overrides(["fulano=A1"])
    with pytest.raises(ValueError):
        tc.interpretar_overrides(["A1"])


# ---------------------------------------------------------------------------
# Obras: filtro de linkage, normalização, paginação e seleção
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "titulo,resumo",
    [
        ("Probabilistic record linkage of tuberculosis notifications", ""),
        ("Estudo de coorte", "Utilizou relacionamento de registros entre bases."),
        ("VINCULAÇÃO de bases de dados do SUS", ""),
        ("Deduplicação de registros", ""),
        ("Uso do OpenRecLink", ""),
        ("Avaliação de pareamento", ""),
        ("Passos de blocagem", ""),
        ("Data deduplication in registries", ""),
    ],
)
def test_filtro_linkage_positivos(titulo, resumo):
    assert tc.eh_obra_linkage(titulo, resumo)


@pytest.mark.parametrize(
    "titulo,resumo",
    [
        ("Epidemiology of tuberculosis in Brazil", "We estimated incidence and mortality."),
        ("Sensibilidade e VPP de testes diagnósticos", ""),
        ("", ""),
    ],
)
def test_filtro_linkage_negativos(titulo, resumo):
    assert not tc.eh_obra_linkage(titulo, resumo)


def obra_openalex(n, titulo, cit=0, tipo="article", resumo_idx=None, doi=None):
    return {
        "id": f"https://openalex.org/W{n}",
        "doi": doi,
        "title": titulo,
        "publication_year": 2000 + n,
        "language": "pt",
        "cited_by_count": cit,
        "abstract_inverted_index": resumo_idx,
        "ids": {"pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456"} if n == 1 else {},
        "primary_location": {"source": {"display_name": "Cad Saude Publica"}},
        "best_oa_location": {"pdf_url": None, "landing_page_url": "https://x.org/1"} if n == 1 else None,
        "type": tipo,
    }


def test_listar_obras_pagina_normaliza_e_filtra_tipos(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "POR_PAGINA", 2)
    p1 = {
        "meta": {"count": 3},
        "results": [
            obra_openalex(1, "Record linkage", 5, resumo_idx={"Estudo": [0], "de": [1], "linkage": [2]},
                          doi="https://doi.org/10.1590/ABC"),
            obra_openalex(2, "Erratum", 1, tipo="erratum"),
        ],
    }
    p2 = {"meta": {"count": 3}, "results": [obra_openalex(3, "Outra", 9)]}
    http, sessao = novo_http(tmp_path, {f"{tc.OPENALEX}/works": [resp_json(p1), resp_json(p2)]})
    obras = tc.listar_obras_autor(http, ["A1"], None)
    assert [o["id"] for o in obras] == ["W1", "W3"]
    w1 = obras[0]
    assert w1["doi"] == "10.1590/abc"
    assert w1["pmcid"] == "PMC123456"
    assert w1["resumo"] == "Estudo de linkage"
    assert w1["urls_oa"] == ["https://x.org/1"]
    assert len(sessao.calls) == 2


class SessaoGravadora:
    """Sessão que registra (url, params, headers) e responde em sequência."""

    def __init__(self, respostas):
        self.respostas = list(respostas)
        self.pedidos: list[tuple[str, dict, dict]] = []

    def get(self, url, params=None, headers=None, timeout=None, allow_redirects=True):
        self.pedidos.append((url, dict(params or {}), dict(headers or {})))
        return self.respostas.pop(0)


@pytest.mark.parametrize("valor", [None, "", "  ", "n/d", "sem-arroba"])
def test_mailto_e_user_agent_so_com_email_valido(tmp_path, valor):
    assert tc.email_valido(valor) is None
    assert "mailto" not in tc._params_openalex(valor, search="x")
    assert "mailto" not in tc.montar_user_agent(valor)
    sessao = SessaoGravadora([resp_json({"meta": {"count": 0}, "results": []})])
    http = tc.Http(tmp_path / "cache", session=sessao, intervalo=0, user_agent=tc.montar_user_agent(valor))
    tc.listar_obras_autor(http, ["A5079601916"], valor)
    _url, params, cab = sessao.pedidos[0]
    assert "mailto" not in params
    assert "mailto" not in cab["User-Agent"]
    assert params["filter"] == "author.id:A5079601916"


def _pagina(ids):
    return resp_json({"meta": {"count": 5}, "results": [obra_openalex(n, f"T{n}", n) for n in ids]})


def test_works_paginacao_basica_sem_cursor(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.setattr(tc, "POR_PAGINA", 2)
    sessao = SessaoGravadora([_pagina([1, 2]), _pagina([3, 4]), _pagina([5])])
    http = tc.Http(tmp_path / "cache", session=sessao, intervalo=0)
    obras = tc.listar_obras_autor(http, ["A1"], None)
    assert [o["id"] for o in obras] == ["W1", "W2", "W3", "W4", "W5"]
    assert [p[1]["page"] for p in sessao.pedidos] == [1, 2, 3]
    assert all("cursor" not in p[1] and p[1]["per-page"] == 2 for p in sessao.pedidos)
    assert all("api_key" not in p[1] for p in sessao.pedidos)


def test_works_para_ao_atingir_meta_count(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "POR_PAGINA", 2)
    cheia = resp_json({"meta": {"count": 2}, "results": [obra_openalex(1, "A"), obra_openalex(2, "B")]})
    sessao = SessaoGravadora([cheia])
    http = tc.Http(tmp_path / "cache", session=sessao, intervalo=0)
    assert len(tc.listar_obras_autor(http, ["A1"], None)) == 2
    assert len(sessao.pedidos) == 1


def test_works_limite_de_dez_paginas_por_autor(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "POR_PAGINA", 1)
    paginas = [resp_json({"meta": {"count": 999}, "results": [obra_openalex(n, f"T{n}")]}) for n in range(1, 30)]
    sessao = SessaoGravadora(paginas)
    http = tc.Http(tmp_path / "cache", session=sessao, intervalo=0)
    tc.listar_obras_autor(http, ["A1"], None)
    assert len(sessao.pedidos) == 10


def test_works_api_key_so_quando_variavel_definida(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "  chave-teste  ")
    sessao = SessaoGravadora([_pagina([1])])
    http = tc.Http(tmp_path / "cache", session=sessao, intervalo=0)
    tc.listar_obras_autor(http, ["A1"], None)
    params = sessao.pedidos[0][1]
    assert params["api_key"] == "chave-teste" and "cursor" not in params and params["page"] == 1

    monkeypatch.setenv("OPENALEX_API_KEY", "   ")
    sessao2 = SessaoGravadora([_pagina([1])])
    http2 = tc.Http(tmp_path / "cache2", session=sessao2, intervalo=0)
    tc.listar_obras_autor(http2, ["A1"], None)
    assert "api_key" not in sessao2.pedidos[0][1]


def test_mailto_enviado_quando_ha_email_valido():
    assert tc.email_valido(" a@b.org ") == "a@b.org"
    assert tc._params_openalex("a@b.org", search="x")["mailto"] == "a@b.org"
    assert tc.montar_user_agent("a@b.org").endswith("(mailto:a@b.org)")


def test_api_key_nunca_aparece_em_erros_nem_em_cache(tmp_path, monkeypatch, capsys):
    segredo = "segredo-xyz-123"
    monkeypatch.setenv("OPENALEX_API_KEY", segredo)

    class SessaoQueFalha:
        def get(self, url, params=None, headers=None, timeout=None, allow_redirects=True):
            raise requests.ConnectionError(
                f"HTTPSConnectionPool(host='api.openalex.org'): Max retries exceeded with url: "
                f"/works?filter=author.id:A1&page=1&api_key={segredo} (Caused by timeout)"
            )

    http = tc.Http(tmp_path / "falha", session=SessaoQueFalha(), intervalo=0, dormir=lambda s: None)
    with pytest.raises(tc.FetchError) as exc:
        tc.listar_obras_autor(http, ["A1"], None)
    assert segredo not in str(exc.value) and "api_key=***" in str(exc.value)

    # Corpo de erro que repete a chave (401) também sai redigido.
    eco = f"invalid api_key={segredo}".encode()
    sessao401 = SessaoGravadora([FakeResp(401, eco)] * 3)
    http401 = tc.Http(tmp_path / "c401", session=sessao401, intervalo=0, dormir=lambda s: None)
    with pytest.raises(tc.FetchError) as exc401:
        tc.listar_obras_autor(http401, ["A1"], None)
    assert segredo not in str(exc401.value)

    # Resposta 200 que repete a chave não a grava no cache; o nome do arquivo não depende dela.
    corpo = {"meta": {"count": 1, "note": f"api_key={segredo}"}, "results": [obra_openalex(3, "Outra", 9)]}
    cache = tmp_path / "ok"
    http_ok = tc.Http(cache, session=SessaoGravadora([resp_json(corpo)]), intervalo=0)
    tc.listar_obras_autor(http_ok, ["A1"], None)
    arquivos = list(cache.iterdir())
    assert arquivos
    assert all(segredo.encode() not in a.read_bytes() and segredo not in a.name for a in arquivos)

    saida = capsys.readouterr()
    assert segredo not in saida.out and segredo not in saida.err


def test_redigir_troca_valor_do_parametro_api_key():
    assert tc.redigir("GET /w?api_key=abc&x=1") == "GET /w?api_key=***&x=1"
    assert tc.redigir("API-KEY=abc") == "API-KEY=***"


def test_works_401_repete_sem_mailto_e_depois_sem_select(tmp_path):
    vazio = resp_json({"meta": {"count": 1}, "results": [obra_openalex(3, "Outra", 9)]})
    sessao = SessaoGravadora([FakeResp(401, b'{"error":"x"}'), FakeResp(401, b'{"error":"y"}'), vazio])
    http = tc.Http(tmp_path / "cache", session=sessao, intervalo=0, dormir=lambda s: None)
    obras = tc.listar_obras_autor(http, ["A1"], "a@b.org")
    assert [o["id"] for o in obras] == ["W3"]
    assert "mailto" in sessao.pedidos[0][1] and "select" in sessao.pedidos[0][1]
    assert "mailto" not in sessao.pedidos[1][1] and "select" in sessao.pedidos[1][1]
    assert "mailto" not in sessao.pedidos[2][1] and "select" not in sessao.pedidos[2][1]


def test_works_401_persistente_inclui_corpo_na_mensagem(tmp_path):
    corpo = ("Chave de API inválida " * 40).encode("utf-8")
    sessao = SessaoGravadora([FakeResp(401, corpo)] * 3)
    http = tc.Http(tmp_path / "cache", session=sessao, intervalo=0, dormir=lambda s: None)
    with pytest.raises(tc.FetchError) as exc:
        tc.listar_obras_autor(http, ["A5079601916"], None)
    msg = str(exc.value)
    assert "HTTP 401" in msg and "author.id:A5079601916" in msg and "Chave de API inválida" in msg
    assert len(msg.split(": ", 1)[1]) < 400
    assert len(sessao.pedidos) == 2  # sem mailto: só a variante sem select é nova


def obra(i, titulo, cit):
    return {"id": i, "titulo": titulo, "resumo": "", "citacoes": cit}


def test_selecionar_obras_linkage_mais_gerais_sem_duplicatas():
    comum = obra("W1", "Record linkage em tuberculose", 3)
    por_autor = {
        "claudia": [comum, obra("W2", "A", 100), obra("W3", "B", 50), obra("W4", "C", 10)],
        "kenneth": [dict(comum), obra("W5", "D", 5)],
    }
    sel = tc.selecionar_obras(por_autor, n_gerais=2)
    assert [o["id"] for o in sel] == ["W1", "W2", "W3", "W5"]
    por_id = {o["id"]: o for o in sel}
    assert por_id["W1"]["autores"] == ["claudia", "kenneth"]
    assert por_id["W1"]["tipo_selecao"] == "linkage"
    assert por_id["W2"]["tipo_selecao"] == "geral"
    assert por_id["W2"]["autores"] == ["claudia"]


# ---------------------------------------------------------------------------
# Extração de texto
# ---------------------------------------------------------------------------
CORPO = " ".join(f"Parágrafo {i} sobre o relacionamento de registros e a sensibilidade." for i in range(1, 6))

HTML_SCIELO = f"""<!DOCTYPE html>
<html><head><title>Artigo</title><script>var x = "SCRIPT_NAO";</script></head>
<body>
<nav>MENU_NAO <a href="/">Início</a></nav>
<header>CABECALHO_NAO</header>
<div class="content">
 <div class="articleTxt">
  <h1>Título do artigo</h1>
  <p>{CORPO}</p>
  <p>Segundo bloco com pareamento probabilístico &amp; blocagem.</p>
  <h2>Referências</h2>
  <ul><li>Fulano F. Um artigo antigo. 1999.</li><li>Beltrano B. Outro. 2001.</li></ul>
 </div>
</div>
<footer>RODAPE_NAO</footer>
</body></html>"""


def test_extrair_html_scielo_so_corpo_sem_referencias():
    texto = tc.extrair_html(HTML_SCIELO)
    assert "Parágrafo 3 sobre o relacionamento de registros" in texto
    assert "pareamento probabilístico & blocagem" in texto
    for fora in ("SCRIPT_NAO", "MENU_NAO", "CABECALHO_NAO", "RODAPE_NAO", "Fulano", "Beltrano"):
        assert fora not in texto
    assert not texto.rstrip().endswith("Referências")


def test_extrair_html_sem_corpo_suficiente_devolve_vazio():
    assert tc.extrair_html("<html><body><p>curto</p></body></html>") == ""


def test_cortar_referencias_variantes_de_titulo():
    corpo = "Introdução\n" + ("texto do artigo. " * 40 + "\n")
    for titulo in ("Referências", "REFERÊNCIAS BIBLIOGRÁFICAS", "References", "Bibliografia", "2. Referencias:"):
        cortado = tc.cortar_referencias(f"{corpo}{titulo}\nAutor A. Titulo. 2000.\nAutor B. Outro. 2001.\n")
        assert "Autor A" not in cortado
        assert "texto do artigo" in cortado


def test_cortar_referencias_ignora_titulo_no_inicio():
    texto = "Referências\n" + "texto do artigo. " * 80
    assert tc.cortar_referencias(texto) == texto


def test_cortar_referencias_sem_titulo_mantem_texto():
    texto = "Texto sem lista.\nMais texto.\n"
    assert tc.cortar_referencias(texto) == texto


def test_extrair_europepmc_body_sem_lista_de_referencias():
    xml = """<?xml version="1.0"?>
<article><front><article-meta><title-group><article-title>T</article-title></title-group></article-meta></front>
<body><sec><title>Introduction</title><p>Probabilistic <italic>record linkage</italic> is common.</p></sec>
<sec><title>Methods</title><p>We used blocking.</p></sec></body>
<back><ref-list><ref><mixed-citation>REF_NAO</mixed-citation></ref></ref-list></back></article>"""
    texto = tc.extrair_europepmc(xml)
    assert "Probabilistic record linkage is common." in texto
    assert "We used blocking." in texto
    assert "REF_NAO" not in texto
    assert tc.extrair_europepmc("<article><nao fechado>") == ""


# ---------------------------------------------------------------------------
# Bibliotecas sombra
# ---------------------------------------------------------------------------
HOSTS_SOMBRA = [
    "https://annas-archive.org/md5/abc",
    "https://booksc.org/x",
    "https://singlelogin.re/x",
    "https://gen.lib.rus.ec/book",
    "https://library.lol/main/abc",
    "https://libstc.cc/x",
    "https://dokumen.pub/x.html",
    "https://b-ok.cc/x",
    "https://pt.annas-archive.se/x",
    "https://sci-hub.se/10.1/abc",
    "https://www.sci-hub.ru/abc",
    "https://libgen.is/book",
    "https://libgen.rs/x",
    "https://z-lib.org/x",
    "https://singlelogin.zlibrary.to/x",
]


@pytest.mark.parametrize("url", HOSTS_SOMBRA)
def test_verificar_host_bloqueia_bibliotecas_sombra(url):
    with pytest.raises(tc.ShadowHostError):
        tc.verificar_host(url)


@pytest.mark.parametrize("url", ["https://www.scielo.br/j/csp/a/x/", "https://doi.org/10.1/abc", "https://europepmc.org/x"])
def test_verificar_host_libera_fontes_abertas(url):
    tc.verificar_host(url)


def test_http_nao_chama_sessao_para_host_sombra(tmp_path):
    http, sessao = novo_http(tmp_path)
    with pytest.raises(tc.ShadowHostError):
        http.get("https://annas-archive.org/md5/abc")
    assert sessao.calls == []


def test_http_nao_segue_redirecionamento_para_host_sombra(tmp_path):
    http, sessao = novo_http(
        tmp_path,
        {"https://doi.org/10.1590/x": FakeResp(302, headers={"Location": "https://sci-hub.se/10.1590/x"})},
    )
    with pytest.raises(tc.ShadowHostError):
        http.get("https://doi.org/10.1590/x")
    assert sessao.calls == ["https://doi.org/10.1590/x"]


TEXTO_LONGO = "Texto de teste sobre linkage e sensibilidade. " * 80


def test_obter_texto_nunca_pede_host_sombra_e_usa_localizacao_aberta(tmp_path):
    unpaywall = {
        "best_oa_location": {
            "url_for_pdf": "https://annas-archive.org/x.pdf",
            "url_for_landing_page": "https://sci-hub.se/10.1234/abc",
        },
        "oa_locations": [{"url_for_landing_page": "https://repo.example.org/paper.html"}],
    }
    http, sessao = novo_http(
        tmp_path,
        {
            f"{tc.EUROPEPMC}/search": resp_json({"resultList": {"result": []}}),
            f"{tc.UNPAYWALL}/10.1234/abc": resp_json(unpaywall),
            "https://repo.example.org/paper.html": resp_html(f"<html><body><article><p>{TEXTO_LONGO}</p></article></body></html>"),
        },
    )
    texto, fonte, avisos = tc.obter_texto(http, {"doi": "10.1234/abc", "pmcid": "", "urls_oa": [], "resumo": "abs"}, "a@b.c")
    assert fonte == "unpaywall"
    assert "linkage e sensibilidade" in texto
    assert "https://repo.example.org/paper.html" in sessao.calls
    for url in sessao.calls:
        assert not url_bloqueada(url), url
    assert any("bloqueado" in a for a in avisos)


def test_obter_texto_scielo_via_doi_com_redirecionamento(tmp_path):
    html = f"<html><body><div class='articleTxt'><p>{TEXTO_LONGO}</p><h2>Referências</h2><p>REF_NAO</p></div></body></html>"
    http, sessao = novo_http(
        tmp_path,
        {
            "https://doi.org/10.1590/abc": FakeResp(302, headers={"Location": "https://www.scielo.br/j/csp/a/abc/"}),
            "https://www.scielo.br/j/csp/a/abc/": resp_html(html),
        },
    )
    texto, fonte, _ = tc.obter_texto(http, {"doi": "10.1590/abc", "pmcid": "", "resumo": ""}, None)
    assert fonte == "scielo"
    assert "REF_NAO" not in texto
    assert sessao.calls == ["https://doi.org/10.1590/abc", "https://www.scielo.br/j/csp/a/abc/"]


def test_obter_texto_europepmc_por_pmcid(tmp_path):
    paragrafos = "".join(f"<p>{TEXTO_LONGO[:400]}</p>" for _ in range(5))
    xml = f"<article><body><sec><title>Intro</title>{paragrafos}</sec></body></article>"
    http, _ = novo_http(tmp_path, {f"{tc.EUROPEPMC}/PMC1/fullTextXML": FakeResp(200, xml.encode(), {"Content-Type": "text/xml"})})
    texto, fonte, _ = tc.obter_texto(http, {"doi": "", "pmcid": "PMC1", "resumo": ""}, None)
    assert fonte == "europepmc"
    assert len(texto) >= tc.MIN_CHARS


def test_obter_texto_cai_para_resumo_e_depois_nenhum(tmp_path):
    http, _ = novo_http(tmp_path)
    texto, fonte, _ = tc.obter_texto(http, {"doi": "10.9/x", "pmcid": "", "resumo": "Só o resumo."}, None)
    assert (texto, fonte) == ("Só o resumo.", "resumo")
    texto, fonte, _ = tc.obter_texto(http, {"doi": "", "pmcid": "", "resumo": ""}, None)
    assert (texto, fonte) == ("", "nenhum")


def test_codigo_nao_contem_url_de_biblioteca_sombra():
    codigo = (RAIZ / "scripts" / "termos_consagrados.py").read_text(encoding="utf-8")
    assert re.search(r"https?://\S*(?:annas-archive|sci-hub|libgen|z-lib)", codigo) is None


def test_http_tentativas_com_recuo_e_cache(tmp_path):
    esperas: list[float] = []
    http, sessao = novo_http(
        tmp_path,
        {"https://api.example.org/x": [FakeResp(503), resp_json({"a": 1})]},
        dormir=esperas.append,
    )
    r = http.get("https://api.example.org/x")
    assert r.status == 200 and r.json() == {"a": 1}
    assert esperas == [1.0]
    assert http.get("https://api.example.org/x").json() == {"a": 1}
    assert len(sessao.calls) == 2


def test_http_falha_depois_das_tentativas(tmp_path):
    http, sessao = novo_http(tmp_path, {"https://api.example.org/y": lambda: FakeResp(500)})
    with pytest.raises(tc.FetchError):
        http.get("https://api.example.org/y")
    assert len(sessao.calls) == http.tentativas


def test_construir_corpus_grava_json_e_retoma(tmp_path, monkeypatch):
    chamadas: list[str] = []

    def falso(http, o, email):
        chamadas.append(o["id"])
        return "Texto de teste. " * 10, "scielo", []

    monkeypatch.setattr(tc, "obter_texto", falso)
    obras = [
        {"id": "W1", "doi": "10.1/a", "ano": 2000, "titulo": "T", "venue": "V", "lingua": "pt",
         "autores": ["claudia"], "tipo_selecao": "linkage", "resumo": ""}
    ]
    tc.construir_corpus(None, obras, tmp_path / "corpus", None)
    tc.construir_corpus(None, obras, tmp_path / "corpus", None)
    assert chamadas == ["W1"]
    doc = json.loads((tmp_path / "corpus" / "W1.json").read_text(encoding="utf-8"))
    assert doc["fonte_texto"] == "scielo" and doc["autores"] == ["claudia"] and doc["texto"].startswith("Texto")


# ---------------------------------------------------------------------------
# Contagem: maior casamento, acentos, caixa
# ---------------------------------------------------------------------------
def contagem_por_forma(texto, variantes):
    an = tc.analisar(texto, variantes)
    return {variantes[vi].forma: n for vi, n in an.contagens().items()}, an


def test_maior_casamento_sem_sobreposicao():
    variantes = [
        var("linkage", "linkage", "en"),
        var("record linkage", "linkage", "en"),
        var("probabilistic linkage", "prob", "en"),
    ]
    contagens, an = contagem_por_forma("Probabilistic linkage, record linkage e linkage.", variantes)
    assert contagens == {"probabilistic linkage": 1, "record linkage": 1, "linkage": 1}
    spans = sorted((a, b) for a, b, _ in an.casamentos)
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))


def test_acentos_caixa_plural_e_limite_de_palavra():
    c, _ = contagem_por_forma("Vinculação, VINCULACAO e vinculações.", [var("vinculação")])
    assert c == {"vinculação": 3}
    c, _ = contagem_por_forma("Falsos positivos e um Falso Positivo.", [var("falso positivo")])
    assert c == {"falso positivo": 2}
    c, _ = contagem_por_forma("padrão ouro e padrão-ouro", [var("padrão-ouro")])
    assert c == {"padrão-ouro": 2}
    c, _ = contagem_por_forma("matching, rematch, Match, matches", [var("match", lingua="en")])
    assert c == {"match": 2}


def test_forma_toda_em_maiusculas_distingue_caixa():
    c, _ = contagem_por_forma("O VPP e o vpp; VPPs; (VPP).", [var("VPP")])
    assert c == {"VPP": 2}


def test_mesma_forma_em_dois_conceitos_conta_uma_vez():
    variantes = [var("corte", "ponto de corte"), var("corte", "outro")]
    c, _ = contagem_por_forma("O corte foi 7.", variantes)
    assert c == {"corte": 1}


def test_recortar_respeita_o_teto():
    texto = "x" * 500 + " termo " + "y" * 500
    a = texto.index("termo")
    assert len(tc.recortar(texto, a, a + 5)) <= tc.MAX_SNIPPET
    assert "termo" in tc.recortar(texto, a, a + 5)
    assert len(tc.recortar(texto, 0, 400)) <= tc.MAX_SNIPPET


# ---------------------------------------------------------------------------
# Definições entre parênteses
# ---------------------------------------------------------------------------
YAML_MINI = """
- conceito: ml
  variantes:
    - {forma: aprendizado de máquina, lingua: pt}
    - {forma: machine learning, lingua: en}
- conceito: linkage
  variantes:
    - {forma: linkage, lingua: en}
    - {forma: record linkage, lingua: en}
    - {forma: relacionamento de registros, lingua: pt}
"""


def test_definicoes_parenteticas_entre_variantes(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text(YAML_MINI, encoding="utf-8")
    _, variantes = tc.carregar_candidatos(f)
    texto = (
        "O aprendizado de máquina (machine learning) é usado. "
        "Já o machine learning (aprendizado de máquina) também. "
        "O linkage (também denominado relacionamento de registros) é comum. "
        "O aprendizado de máquina (ML) não conta. "
        "Usa-se linkage (em inglês, record linkage) aqui."
    )
    an = tc.analisar(texto, variantes)
    achados = tc.definicoes_de_variantes(an, variantes, "W1")
    pares = {(d["conceito"], d["principal"], d["entre_parenteses"]) for d in achados}
    assert pares == {
        ("ml", "aprendizado de máquina", "machine learning"),
        ("ml", "machine learning", "aprendizado de máquina"),
        ("linkage", "linkage", "relacionamento de registros"),
        ("linkage", "linkage", "record linkage"),
    }
    assert all(len(d["trecho"]) <= tc.MAX_SNIPPET and d["origem"] == "W1" for d in achados)
    ml = next(d for d in achados if d["principal"] == "aprendizado de máquina")
    assert (ml["lingua_principal"], ml["lingua_parenteses"]) == ("pt", "en")


# ---------------------------------------------------------------------------
# Siglas
# ---------------------------------------------------------------------------
def test_detectar_siglas_expansao_seguida_de_sigla_e_inversa():
    texto = (
        "O Sistema Único de Saúde (SUS) atende. "
        "Usa-se Knowledge Discovery in Databases (KDD) hoje. "
        "Também KDD (Knowledge Discovery in Databases) aparece. "
        "O Valor Preditivo Positivo (VPP) foi alto. "
        "O Comparador de Registros (CdR) calcula escores. "
        "A taxa observada (AB) não é sigla. "
        "Veja (Figura 3) e Tabela (Resumo)."
    )
    achados = tc.detectar_siglas(texto, "W9")
    pares = {(d["sigla"], d["expansao"], d["ordem"]) for d in achados}
    assert ("SUS", "Sistema Único de Saúde", "expansao_sigla") in pares
    assert ("KDD", "Knowledge Discovery in Databases", "expansao_sigla") in pares
    assert ("KDD", "Knowledge Discovery in Databases", "sigla_expansao") in pares
    assert ("VPP", "Valor Preditivo Positivo", "expansao_sigla") in pares
    assert ("CdR", "Comparador de Registros", "expansao_sigla") in pares
    assert {p[0] for p in pares} == {"SUS", "KDD", "VPP", "CdR"}
    assert all(len(d["trecho"]) <= tc.MAX_SNIPPET for d in achados)
    kdd = next(d for d in achados if d["sigla"] == "KDD")
    assert kdd["lingua_expansao"] == "en"


def test_contar_siglas_distingue_caixa_e_hifen():
    padrao = tc.compilar_siglas(["AUC", "AUC-ROC", "SUS", "LLM"])
    c = tc.contar_siglas("AUC e AUC-ROC; o SUS, sus, SUS-RJ e dois LLMs (AUC).", padrao)
    assert c == {"AUC": 2, "AUC-ROC": 1, "SUS": 1, "LLM": 1}


SIGLAS_TEX = r"""\chapter*{Lista}
\begin{longtable}{@{}l@{\hspace{1.5em}}p{12cm}@{}}
SUS      & Sistema \'{U}nico de Sa\'{u}de \\[6pt]
KDD      & \textit{Knowledge Discovery in Databases}: descoberta de conhecimento em bases de dados \\[6pt]
VPP & Valor Preditivo Positivo \\
\end{longtable}
"""


def test_ler_siglas_tex(tmp_path):
    f = tmp_path / "siglas.tex"
    f.write_text(SIGLAS_TEX, encoding="utf-8")
    siglas = {s["sigla"]: s for s in tc.ler_siglas_tex(f)}
    assert list(siglas) == ["SUS", "KDD", "VPP"]
    assert siglas["SUS"]["expansao_pt"] == "Sistema Único de Saúde"
    assert siglas["KDD"]["expansao_en"] == "Knowledge Discovery in Databases"
    assert siglas["KDD"]["expansao_pt"] == "descoberta de conhecimento em bases de dados"


# ---------------------------------------------------------------------------
# Leitura da tese
# ---------------------------------------------------------------------------
def montar_tese(raiz: Path) -> Path:
    (raiz / "tables").mkdir()
    (raiz / "tese.tex").write_text(
        "\\documentclass{abntex2}\n\\newcommand{\\x}{PREAMBULO_NAO}\n\\begin{document}\n"
        "\\input{cap1.tex}\n"
        "%\\input{cap2.tex}\n"
        "\\input{tables/t1}\n"
        "\\input{tese_rev_diff.tex}\n"
        "\\input{siglas.tex}\n"
        "\\input{figures/fig.pgf}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    (raiz / "cap1.tex").write_text(
        "\\chapter{Introdução}\n"
        "O \\textit{linkage} probabilístico (\\textit{probabilistic linkage}) é usado~\\cite{Fellegi1969}. % blocagem comentada\n"
        "Arquivo em \\path{data/linkage_resultado.csv} e vincula\\c{c}\\~{a}o de registros. "
        "Valor $m = 0{,}9$ e $$ \\text{pareamento} $$.\n"
        "\\begin{equation} x = \\text{matching} \\end{equation}\n"
        "Uma frase com 50\\% de acerto.\n",
        encoding="utf-8",
    )
    (raiz / "cap2.tex").write_text("COMENTADO_NAO\n", encoding="utf-8")
    (raiz / "tables" / "t1.tex").write_text(
        "\\begin{tabular}{ll}\nTABELACONTEUDO & valor \\\\\n\\end{tabular}\n", encoding="utf-8"
    )
    (raiz / "tese_rev_diff.tex").write_text("ZZDIFF_NAO\n", encoding="utf-8")
    (raiz / "siglas.tex").write_text(SIGLAS_TEX + "SIGLAEXCLUIDA_NAO\n", encoding="utf-8")
    return raiz / "tese.tex"


def test_ler_tese_segue_include_e_limpa_latex(tmp_path):
    texto, arquivos, avisos = tc.ler_tese(montar_tese(tmp_path))
    assert "linkage probabilístico (probabilistic linkage) é usado" in texto
    assert "vinculação de registros" in texto
    assert "50% de acerto" in texto
    assert "TABELACONTEUDO | valor" in texto
    for fora in (
        "Fellegi", "blocagem comentada", "linkage_resultado", "pareamento", "matching",
        "COMENTADO_NAO", "ZZDIFF_NAO", "SIGLAEXCLUIDA_NAO", "PREAMBULO_NAO", "0,9",
    ):
        assert fora not in texto
    assert "ll" not in texto.split()
    assert sorted(arquivos) == ["cap1.tex", "t1.tex", "tese.tex"]
    assert avisos == []


def test_ler_tese_pode_incluir_siglas_se_nao_excluido(tmp_path):
    texto, _, _ = tc.ler_tese(montar_tese(tmp_path), excluir=set())
    assert "SIGLAEXCLUIDA_NAO" in texto
    assert "ZZDIFF_NAO" not in texto  # tese_rev_diff.tex é sempre ignorado


# ---------------------------------------------------------------------------
# Proposta
# ---------------------------------------------------------------------------
def est(forma, lingua, docs, total=0):
    return {"forma": forma, "lingua": lingua, "docs": docs, "total": total}


def test_propor_forma_maioria_de_documentos():
    p = tc.propor_forma([est("linkage", "en", 5, 10), est("relacionamento de registros", "pt", 2, 90)])
    assert (p["forma"], p["regra"], p["italico"]) == ("linkage", "maioria", True)


def test_propor_forma_empate_fica_com_portugues():
    p = tc.propor_forma([est("recall", "en", 3, 50), est("sensibilidade", "pt", 3, 5), est("revocação", "pt", 1, 1)])
    assert (p["forma"], p["regra"], p["italico"]) == ("sensibilidade", "empate_portugues", False)


def test_propor_forma_empate_entre_portugues_usa_total_e_sem_portugues_usa_total():
    p = tc.propor_forma([est("a", "pt", 2, 3), est("b", "pt", 2, 9), est("c", "en", 2, 99)])
    assert (p["forma"], p["regra"]) == ("b", "empate_portugues")
    p = tc.propor_forma([est("x", "en", 2, 3), est("y", "en", 2, 9)])
    assert (p["forma"], p["regra"], p["italico"]) == ("y", "empate_total", True)


def test_propor_forma_sem_uso():
    p = tc.propor_forma([est("a", "pt", 0), est("b", "en", 0)])
    assert p["forma"] is None and p["regra"] == "sem_uso"


# ---------------------------------------------------------------------------
# Arquivo de candidatos real
# ---------------------------------------------------------------------------
FORMAS_OBRIGATORIAS = [
    "linkage", "record linkage", "relacionamento de registros", "vinculação de registros", "pareamento",
    "blocking", "blocagem", "match", "par verdadeiro", "vínculo", "non-match", "não par", "par falso",
    "clerical review", "revisão manual", "revisão clerical", "gold standard", "padrão-ouro",
    "padrão de referência", "sensitivity", "sensibilidade", "PPV", "VPP", "valor preditivo positivo",
    "precision", "precisão", "recall", "revocação", "specificity", "especificidade", "F-measure", "F1",
    "medida F", "threshold", "limiar", "ponto de corte", "score", "escore", "pontuação", "peso",
    "machine learning", "aprendizado de máquina", "aprendizagem de máquina", "deduplication",
    "deduplicação", "false positive", "falso positivo", "false negative", "falso negativo",
    "probabilistic linkage", "linkage probabilístico", "relacionamento probabilístico",
    "deterministic linkage", "linkage determinístico", "string comparator", "comparador de strings",
    "phonetic code", "código fonético", "soundex", "m-probability", "u-probability", "dataset",
    "base de dados", "banco de dados", "feature", "atributo", "variável preditora", "random forest",
    "floresta aleatória", "cutoff", "corte", "pipeline", "fluxo",
]


def test_yaml_real_carrega_e_cobre_formas_obrigatorias():
    conceitos, variantes = tc.carregar_candidatos(YAML_REAL)
    assert len({c.nome for c in conceitos}) == len(conceitos)
    assert {v.lingua for v in variantes} == {"pt", "en"}
    presentes = {tc.dobrar(v.forma)[0].strip().lower() for v in variantes}
    faltando = [f for f in FORMAS_OBRIGATORIAS if tc.dobrar(f)[0].strip().lower() not in presentes]
    assert faltando == []


def test_carregar_candidatos_rejeita_lingua_invalida_e_conceito_repetido(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("- conceito: a\n  variantes:\n    - {forma: x, lingua: fr}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        tc.carregar_candidatos(f)
    f.write_text(
        "- conceito: a\n  variantes: [{forma: x, lingua: pt}]\n- conceito: a\n  variantes: [{forma: y, lingua: pt}]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        tc.carregar_candidatos(f)


# ---------------------------------------------------------------------------
# Execução completa (sem rede)
# ---------------------------------------------------------------------------
YAML_INTEGRACAO = """
- conceito: linkage
  guia: linkage
  variantes:
    - {forma: linkage, lingua: en}
    - {forma: record linkage, lingua: en}
    - {forma: relacionamento de registros, lingua: pt}
- conceito: linkage probabilístico
  variantes:
    - {forma: linkage probabilístico, lingua: pt}
- conceito: sensibilidade
  variantes:
    - {forma: sensibilidade, lingua: pt}
    - {forma: recall, lingua: en}
- conceito: valor preditivo positivo
  variantes:
    - {forma: VPP, lingua: pt}
    - {forma: PPV, lingua: en}
- conceito: sem uso em lugar nenhum
  variantes:
    - {forma: zebra, lingua: pt}
"""


def doc_json(i, autores, ano, texto, fonte="scielo"):
    return {
        "id": i, "doi": f"10.1/{i}", "ano": ano, "titulo": f"Título {i}", "venue": "Rev", "lingua": "pt",
        "autores": autores, "tipo_selecao": "linkage", "fonte_texto": fonte, "avisos": [], "texto": texto,
    }


def preparar_execucao(tmp_path):
    out = tmp_path / "termos"
    (out / "corpus").mkdir(parents=True)
    docs = [
        doc_json("W1", ["claudia"], 2010, "O linkage probabilístico e o linkage foram usados. A sensibilidade e o VPP foram altos."),
        doc_json("W2", ["kenneth"], 2012, "O relacionamento de registros (record linkage) foi aplicado. O relacionamento de registros mostrou sensibilidade e VPP."),
        doc_json("W3", ["rejane", "claudia"], 2014, "Resumo sobre recall e linkage.", fonte="resumo"),
    ]
    for d in docs:
        (out / "corpus" / f"{d['id']}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    cand = tmp_path / "cand.yaml"
    cand.write_text(YAML_INTEGRACAO, encoding="utf-8")
    raiz_tese = tmp_path / "tese"
    raiz_tese.mkdir()
    (raiz_tese / "tese.tex").write_text(
        "\\begin{document}\n\\input{cap1.tex}\n\\end{document}\n", encoding="utf-8"
    )
    (raiz_tese / "cap1.tex").write_text(
        "O linkage e o relacionamento de registros. O recall e a sensibilidade. "
        "Uso de VPP e PPV. O Sistema Único de Saúde (SUS) é grande.\n",
        encoding="utf-8",
    )
    siglas = tmp_path / "siglas.tex"
    siglas.write_text(SIGLAS_TEX, encoding="utf-8")
    return out, cand, raiz_tese / "tese.tex", siglas


def ler_csv(caminho):
    with caminho.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_executar_contagem_gera_saidas_e_proposta(tmp_path):
    out, cand, tese, siglas = preparar_execucao(tmp_path)
    tc.executar_contagem(out, cand, tese, siglas)

    docs = ler_csv(out / "documentos.csv")
    assert [d["openalex_id"] for d in docs] == ["W1", "W2", "W3"]
    assert docs[2]["fonte_texto"] == "resumo"
    assert docs[2]["autores"] == "Pinheiro (orientadora);Coeli"
    assert int(docs[0]["n_palavras"]) == len(json.loads((out / "corpus" / "W1.json").read_text(encoding="utf-8"))["texto"].split())

    cont = ler_csv(out / "contagens.csv")
    achado = {(c["conceito"], c["variante"], c["lingua"], c["documento"]): int(c["contagem"]) for c in cont}
    assert achado[("linkage", "linkage", "en", "W1")] == 1
    assert achado[("linkage probabilístico", "linkage probabilístico", "pt", "W1")] == 1
    assert achado[("linkage", "relacionamento de registros", "pt", "W2")] == 2
    assert achado[("sigla:VPP", "VPP", "sigla", "W2")] == 1
    chaves = [(c["conceito"], c["variante"], c["documento"]) for c in cont]
    assert chaves == sorted(chaves)

    tese_c = {(c["conceito"], c["variante"]): int(c["contagem"]) for c in ler_csv(out / "tese_contagens.csv")}
    assert tese_c[("linkage", "linkage")] == 1
    assert tese_c[("linkage", "relacionamento de registros")] == 1
    assert tese_c[("sensibilidade", "recall")] == 1
    assert tese_c[("valor preditivo positivo", "PPV")] == 1
    assert tese_c[("sigla:SUS", "SUS")] == 1

    defs = [json.loads(l) for l in (out / "definicoes.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(len(d["trecho"]) <= tc.MAX_SNIPPET for d in defs)
    assert any(
        d["origem"] == "W2" and d["tipo"] == "variantes"
        and (d["principal"], d["entre_parenteses"]) == ("relacionamento de registros", "record linkage")
        for d in defs
    )
    assert any(d["origem"] == "tese" and d["tipo"] == "sigla" and d["sigla"] == "SUS" for d in defs)

    resumo = (out / "resumo.md").read_text(encoding="utf-8")
    assert "### linkage\n" in resumo
    assert "**PROPOSTA:** linkage (em itálico, estrangeirismo consagrado)" in resumo
    assert "**PROPOSTA:** sensibilidade." in resumo
    assert "**ALERTA:** a tese usa 2 ou mais formas" in resumo
    assert "sem uso em lugar nenhum" in resumo
    assert "| Forma | Língua | Banca/orientadora: documentos" in resumo
    assert "## Siglas" in resumo
    assert "—" not in resumo and "–" not in resumo


def test_executar_contagem_e_deterministica_e_so_texto_completo(tmp_path):
    out, cand, tese, siglas = preparar_execucao(tmp_path)
    nomes = ["documentos.csv", "contagens.csv", "tese_contagens.csv", "definicoes.jsonl", "resumo.md"]
    tc.executar_contagem(out, cand, tese, siglas)
    antes = {n: (out / n).read_bytes() for n in nomes}
    tc.executar_contagem(out, cand, tese, siglas)
    assert {n: (out / n).read_bytes() for n in nomes} == antes

    tc.executar_contagem(out, cand, tese, siglas, so_texto_completo=True)
    assert [d["openalex_id"] for d in ler_csv(out / "documentos.csv")] == ["W1", "W2"]


def test_main_etapa_contar_sem_rede(tmp_path, capsys):
    out, cand, tese, siglas = preparar_execucao(tmp_path)
    rc = tc.main([
        "--etapa", "contar", "--out", str(out), "--candidatos", str(cand),
        "--tese", str(tese), "--siglas", str(siglas),
    ])
    assert rc == 0
    assert (out / "resumo.md").is_file()
    assert "documentos: 3" in capsys.readouterr().out
