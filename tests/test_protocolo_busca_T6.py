"""Testes minimos do protocolo de busca T6: parse do PubMed e geracao do CSV com fixtures pequenas."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import protocolo_busca_T6 as coleta  # noqa: E402
import protocolo_busca_T6_tabelas as tab  # noqa: E402

XML_PUBMED = b"""<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle>
  <MedlineCitation><PMID>111</PMID>
   <Article>
    <Journal><Title>Cad Saude Publica</Title><JournalIssue><PubDate><Year>2019</Year></PubDate></JournalIssue></Journal>
    <ArticleTitle>Probabilistic <i>record linkage</i> in Brazil</ArticleTitle>
    <AuthorList><Author><LastName>Silva</LastName></Author></AuthorList>
    <AffiliationInfo><Affiliation>Instituto X, Rio de Janeiro, BRAZIL.</Affiliation></AffiliationInfo>
   </Article>
  </MedlineCitation>
  <PubmedData><ArticleIdList>
   <ArticleId IdType="pubmed">111</ArticleId><ArticleId IdType="doi">10.1/abc</ArticleId>
  </ArticleIdList>
  <ReferenceList><Reference><ArticleIdList><ArticleId IdType="doi">10.9/referencia</ArticleId></ArticleIdList></Reference></ReferenceList>
  </PubmedData>
 </PubmedArticle>
</PubmedArticleSet>"""


def _grava(caminho: Path, obj: dict) -> Path:
    caminho.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return caminho


def test_pubmed_parse_doi_do_artigo_e_afiliacao_sem_diferenciar_caixa() -> None:
    (reg,) = coleta.pubmed_parse(XML_PUBMED)
    assert reg["pmid"] == "111"
    assert reg["doi"] == "10.1/abc"  # nao o DOI da lista de referencias
    assert reg["titulo"] == "Probabilistic record linkage in Brazil"
    assert reg["afiliacao_brasil"] is True


def test_openalex_registro_e_resumo_invertido() -> None:
    w = {
        "id": "https://openalex.org/W1",
        "doi": "https://doi.org/10.1/x",
        "title": "T",
        "publication_year": 2024,
        "authorships": [{"author": {"display_name": "A B"}}],
        "primary_location": {"source": {"display_name": "J"}},
        "type": "article",
    }
    r = coleta.openalex_registro(w)
    assert r["id"] == "W1" and r["doi"] == "10.1/x" and r["primeiro_autor"] == "A B"
    assert coleta.resumo_de({"machine": [1], "learning": [2], "Brazil": [0]}) == "Brazil machine learning"
    assert coleta.resumo_de(None) is None


def test_gerar_csv_colunas_contagens_e_decisoes(tmp_path: Path) -> None:
    pm = _grava(tmp_path / "pm.json", {
        "data_busca": "2026-11-03",
        "consultas": {
            "P1": {"string_exata": "q1", "descricao": "d", "filtros": "f", "n": 2, "pmids": ["111", "222"]},
            "P4": {"string_exata": "q4", "descricao": "d", "filtros": "f", "n": 1, "pmids": ["333"]},
        },
        "registros": {
            "111": {"pmid": "111", "ano": "2019", "revista": "J", "primeiro_autor": "Silva", "titulo": "Record linkage de obitos", "doi": "10.1/a", "afiliacao_brasil": True},
            "222": {"pmid": "222", "ano": "2020", "revista": "J", "primeiro_autor": "Souza", "titulo": "Estudo sem tema", "doi": "", "afiliacao_brasil": True},
            "333": {"pmid": "333", "ano": "2021", "revista": "J", "primeiro_autor": "Lee", "titulo": "Record linkage with ML", "doi": "", "afiliacao_brasil": False},
        },
    })
    oa = _grava(tmp_path / "oa.json", {
        "data_busca": "2026-11-03",
        "consultas": {"O1": {"string_exata": "q", "descricao": "d", "modo": "search", "filtros": "f", "url_primeira_pagina": "http://x", "n_paginas": 1, "n": 3, "ids": ["W1", "W2", "W3"]}},
        "registros": {
            "W1": {"id": "W1", "doi": "10.1038/s41598-026-73928-3", "titulo": "Machine learning-based record linkage", "ano": 2026, "primeiro_autor": "Lages", "periodico": "Sci Rep", "tipo": "article"},
            "W2": {"id": "W2", "doi": "", "titulo": "Tema sem relacao", "ano": 2020, "primeiro_autor": "", "periodico": "", "tipo": "article"},
            "W3": {"id": "W3", "doi": "", "titulo": "Entity resolution for products", "ano": 2021, "primeiro_autor": "", "periodico": "", "tipo": "article", "resumo_disponivel": True, "resumo_brasil": False, "resumo_ml": True},
        },
    })
    man = _grava(tmp_path / "man.json", {
        "data_busca": "2026-11-03",
        "consultas": [{"base": "SciELO", "consulta": "S1", "string_exata": "s", "filtros": "f", "n": 0, "nota": "n"},
                      {"base": "Scopus", "consulta": "-", "nota": "nao executada"}],
        "registros": [{"base": "SciELO", "consulta": "S2", "primeiro_autor": "X", "ano": "2015", "titulo": "T", "decisao": "excluido", "motivo": "m"}],
    })
    saida = tmp_path / "out.csv"
    res = tab.gerar_csv(pm, oa, man, saida)
    rows = list(csv.DictReader(saida.open(encoding="utf-8")))
    assert list(rows[0].keys()) == tab.COLUNAS
    assert res["linhas"] == len(rows)
    consultas = {r["consulta"]: r for r in rows if r["etapa"] == "consulta" and r["base"] == "PubMed"}
    assert consultas["P1"]["n_recuperado"] == "2" and consultas["P1"]["data_busca"] == "2026-11-03"
    assert consultas["P1"]["string_exata"] == "q1"
    # contagens nunca ficam em registro_id
    assert all(not r["registro_id"].isdigit() for r in rows if r["etapa"] == "consulta")
    tri = {r["registro_id"]: r for r in rows if r["etapa"] == "triagem"}
    assert tri["PMID 111"]["motivo_ou_nota"] == tab.SEM_DECISAO  # tem termo de linkage e nao ha decisao
    assert tri["PMID 222"]["motivo_ou_nota"] == tab.MOTIVOS["titulo"]
    assert tri["PMID 333"]["motivo_ou_nota"] == tab.MOTIVOS["internacional"]
    assert tri["W1"]["decisao"] == "incluido"
    assert tri["W2"]["motivo_ou_nota"] == tab.MOTIVOS["titulo"]
    assert tri["W3"]["motivo_ou_nota"] == tab.MOTIVOS["sem_brasil"]
    assert res["sem_decisao"] == 1
