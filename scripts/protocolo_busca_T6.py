#!/usr/bin/env python3
"""Protocolo de busca bibliografica (T6): coleta no PubMed e no OpenAlex e geracao do CSV.

Executa as cadeias do protocolo no PubMed (E-utilities) e no OpenAlex (API),
grava os brutos (somente identificadores e metadados bibliograficos, sem
resumos e sem dados pessoais) em

    data/protocolo_busca_T6_pubmed.json
    data/protocolo_busca_T6_openalex.json

e regera data/protocolo_busca_T6.csv a partir desses JSONs e do registro manual
das consultas feitas por navegador (data/protocolo_busca_T6_manual.json:
SciELO e BVS/LILACS, que recusam requisicoes diretas por desafio de JavaScript).

Uso:
    python scripts/protocolo_busca_T6.py            # coleta e regera o CSV
    python scripts/protocolo_busca_T6.py --offline  # regera o CSV a partir dos JSONs

A decisao de triagem de cada registro esta codificada em
scripts/protocolo_busca_T6_tabelas.py (leitura de titulo; resumo lido nos
registros mais proximos do criterio).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from urllib.parse import urlencode

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocolo_busca_T6_tabelas import LINK_RE, gerar_csv  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
OPENALEX = "https://api.openalex.org/works"
CONTATO = "phd-research-T6 (mailto:marquinho.jardim@gmail.com)"
ANO_INI, ANO_FIM = 2010, 2026
PER_PAGE = 200

# --- cadeias -------------------------------------------------------------------------------
PRINCIPAL = (
    '("record linkage" OR "relacionamento de registros" OR "pareamento probabilístico") '
    'AND ("machine learning" OR "aprendizado de máquina" OR supervised) '
    "AND (Brasil OR Brazil OR SIM OR SINAN OR DATASUS)"
)
# Ampliada (A1): acrescenta variantes de nomenclatura do linkage e do aprendizado supervisionado.
AMPLIADA = (
    '("record linkage" OR "data linkage" OR "relacionamento de registros" OR "relacionamento probabilístico" '
    'OR "linkage probabilístico" OR "pareamento de registros" OR "pareamento probabilístico") '
    'AND ("machine learning" OR "aprendizado de máquina" OR supervised OR supervisionado) '
    "AND (Brasil OR Brazil OR Brazilian OR SIM OR SINAN OR DATASUS)"
)
# Somente os termos novos, em forma curta (a API do OpenAlex limita o numero de operadores booleanos).
NOVOS_TERMOS = (
    '("data linkage" OR "relacionamento probabilístico" OR "linkage probabilístico" OR "pareamento de registros") '
    'AND ("machine learning" OR "aprendizado de máquina" OR supervised OR supervisionado) '
    "AND (Brasil OR Brazil OR Brazilian)"
)

PUBMED = {
    "P1": (PRINCIPAL, "cadeia principal, sem rótulos de campo (mapeamento automático de termos)"),
    "P2": (
        '("record linkage"[tiab] OR "relacionamento de registros"[tiab] OR "pareamento probabilístico"[tiab]) '
        'AND ("machine learning"[tiab] OR "aprendizado de máquina"[tiab] OR supervised[tiab]) '
        "AND (Brasil[tiab] OR Brazil[tiab] OR Brazilian[tiab] OR SIM[tiab] OR SINAN[tiab] OR DATASUS[tiab])",
        "cadeia principal restrita a título e resumo [tiab]",
    ),
    "P3": (
        '("record linkage"[tiab] OR "data linkage"[tiab]) AND ("machine learning"[tiab] OR supervised[tiab]) '
        "AND (Brazil[ad] OR Brasil[ad])",
        "linkage e aprendizado supervisionado em título e resumo, afiliação Brazil ou Brasil [ad]",
    ),
    "P4": (
        '("record linkage"[tiab] OR "data linkage"[tiab]) AND ("machine learning"[tiab] OR supervised[tiab])',
        "mesmo bloco sem restrição de país (referência internacional)",
    ),
    "P5": (AMPLIADA, "cadeia ampliada (A1), sem rótulos de campo"),
    "P6": (
        '("record linkage"[tiab] OR "data linkage"[tiab] OR "relacionamento de registros"[tiab] '
        'OR "relacionamento probabilístico"[tiab] OR "linkage probabilístico"[tiab] OR "pareamento de registros"[tiab] '
        'OR "pareamento probabilístico"[tiab]) '
        'AND ("machine learning"[tiab] OR "aprendizado de máquina"[tiab] OR supervised[tiab] OR supervisionado[tiab]) '
        "AND (Brasil[tiab] OR Brazil[tiab] OR Brazilian[tiab] OR SIM[tiab] OR SINAN[tiab] OR DATASUS[tiab])",
        "cadeia ampliada (A1) restrita a título e resumo [tiab]",
    ),
}

# id: (modo, cadeia, descricao)
OPENALEX_CONSULTAS = {
    "O1": ("title_and_abstract.search", PRINCIPAL, "cadeia principal em título e resumo"),
    "O2": (
        "title_and_abstract.search",
        '("record linkage" OR "data linkage") AND ("machine learning" OR supervised OR classifier) '
        "AND (Brazil OR Brazilian OR SINAN OR DATASUS OR SUS)",
        "variante com classifier e SUS, em título e resumo",
    ),
    "O3": ("search", PRINCIPAL, "cadeia principal, parâmetro search (inclui texto integral quando indexado)"),
    "O4": ("search", NOVOS_TERMOS, "termos novos (A1), parâmetro search"),
    "O5": ("title_and_abstract.search", AMPLIADA, "cadeia ampliada (A1) em título e resumo"),
}


def _get(url: str, params: dict, tentativas: int = 5, **kw) -> requests.Response:
    for i in range(tentativas):
        r = requests.get(url, params=params, timeout=120, **kw)
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 * (i + 1))
            continue
        return r
    return r


# --- PubMed ----------------------------------------------------------------------------------
def pubmed_esearch(termo: str) -> tuple[dict, str]:
    params = {
        "db": "pubmed",
        "term": termo,
        "retmax": 1000,
        "retmode": "json",
        "datetype": "pdat",
        "mindate": ANO_INI,
        "maxdate": ANO_FIM,
    }
    r = _get(EUTILS + "esearch.fcgi", params)
    r.raise_for_status()
    return r.json()["esearchresult"], r.url


def pubmed_parse(xml_bytes: bytes) -> list[dict]:
    regs = []
    for art in ET.fromstring(xml_bytes).findall(".//PubmedArticle"):
        t = art.find(".//ArticleTitle")
        doi = ""
        for aid in art.findall("./PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = (aid.text or "").strip()
        afil = " ".join("".join(a.itertext()) for a in art.findall(".//AffiliationInfo/Affiliation"))
        regs.append(
            {
                "pmid": art.findtext(".//MedlineCitation/PMID"),
                "ano": art.findtext(".//JournalIssue/PubDate/Year")
                or (art.findtext(".//JournalIssue/PubDate/MedlineDate") or "")[:4],
                "revista": art.findtext(".//Journal/Title") or "",
                "primeiro_autor": art.findtext(".//AuthorList/Author/LastName") or "",
                "titulo": "".join(t.itertext()) if t is not None else "",
                "doi": doi,
                "afiliacao_brasil": bool(re.search(r"brazil|brasil", afil, re.I)),
            }
        )
    return regs


def pubmed_efetch(pmids: list[str]) -> list[dict]:
    regs: list[dict] = []
    for i in range(0, len(pmids), 100):
        r = _get(EUTILS + "efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids[i : i + 100]), "retmode": "xml"})
        r.raise_for_status()
        regs.extend(pubmed_parse(r.content))
        time.sleep(0.4)
    return regs


def coletar_pubmed(data_busca: str) -> dict:
    saida = {"fonte": "PubMed (E-utilities)", "data_busca": data_busca, "periodo": f"{ANO_INI}-{ANO_FIM}", "consultas": {}, "registros": {}}
    uniao: set[str] = set()
    for cid, (termo, desc) in PUBMED.items():
        res, url = pubmed_esearch(termo)
        ids = res["idlist"]
        saida["consultas"][cid] = {
            "string_exata": termo,
            "descricao": desc,
            "url": url,
            "filtros": f"datetype=pdat; mindate={ANO_INI}; maxdate={ANO_FIM}; retmax=1000",
            "n": int(res["count"]),
            "pmids": ids,
            "traducao_automatica": res.get("querytranslation", ""),
        }
        uniao.update(ids)
        time.sleep(0.4)
    for reg in pubmed_efetch(sorted(uniao)):
        saida["registros"][reg["pmid"]] = reg
    return saida


# --- OpenAlex --------------------------------------------------------------------------------
SELECT = "id,doi,title,publication_year,authorships,primary_location,type"


def openalex_registro(w: dict) -> dict:
    loc = (w.get("primary_location") or {}).get("source") or {}
    aut = w["authorships"][0]["author"]["display_name"] if w.get("authorships") else ""
    return {
        "id": w["id"].rsplit("/", 1)[-1],
        "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
        "titulo": w.get("title") or "",
        "ano": w.get("publication_year"),
        "primeiro_autor": aut,
        "periodico": loc.get("display_name") or "",
        "tipo": w.get("type") or "",
    }


def openalex_params(modo: str, cadeia: str) -> dict:
    filtro_ano = f"publication_year:{ANO_INI}-{ANO_FIM}"
    p = {"per-page": PER_PAGE, "select": SELECT, "mailto": "marquinho.jardim@gmail.com"}
    if modo == "search":
        p["search"] = cadeia
        p["filter"] = filtro_ano
    else:
        p["filter"] = f"{modo}:{cadeia},{filtro_ano}"
    return p


def coletar_openalex(data_busca: str) -> dict:
    saida = {"fonte": "OpenAlex (API)", "data_busca": data_busca, "periodo": f"{ANO_INI}-{ANO_FIM}", "consultas": {}, "registros": {}}
    for cid, (modo, cadeia, desc) in OPENALEX_CONSULTAS.items():
        params = openalex_params(modo, cadeia)
        paginas: list[str] = []
        ids: list[str] = []
        n = None
        pagina = 1
        while True:
            r = _get(OPENALEX, {**params, "page": pagina}, headers={"User-Agent": CONTATO})
            r.raise_for_status()
            paginas.append(r.url)
            js = r.json()
            n = js["meta"]["count"]
            for w in js["results"]:
                reg = openalex_registro(w)
                saida["registros"][reg["id"]] = reg
                ids.append(reg["id"])
            if not js["results"] or len(ids) >= n or pagina * PER_PAGE >= 10000:
                break
            pagina += 1
            time.sleep(0.2)
        saida["consultas"][cid] = {
            "string_exata": cadeia,
            "descricao": desc,
            "modo": modo,
            "filtros": f"publication_year:{ANO_INI}-{ANO_FIM}; per-page={PER_PAGE}; paginação por page=1..N",
            "url_primeira_pagina": paginas[0],
            "n_paginas": len(paginas),
            "n": n,
            "ids": ids,
        }
    sinalizar_resumos(saida["registros"])
    return saida

BRASIL_RE = re.compile(r"brazil|brasil|brasileir|sinan|datasus|sinasc|cidacs|sih-sus|\bSUS\b", re.I)
ML_RE = re.compile(
    r"machine learning|aprendizado de m|supervised|supervisionad|classifier|classificador|random forest|"
    r"xgboost|lightgbm|gradient boosting|neural|deep learning|logistic regression|decision tree|svm",
    re.I,
)


def resumo_de(inv: dict | None) -> str | None:
    """Reconstroi o resumo a partir do indice invertido do OpenAlex (usado so em memoria)."""
    if not inv:
        return None
    pos = sorted((i, p) for p, idx in inv.items() for i in idx)
    return " ".join(p for _, p in pos)


def sinalizar_resumos(registros: dict) -> None:
    """Para titulos com termo de linkage, registra apenas sinalizadores (Brasil, ML) do resumo.

    Os resumos nao sao gravados: servem so ao filtro de palavras-chave que precede a leitura.
    """
    alvo = [i for i, r in registros.items() if LINK_RE.search(r["titulo"])]
    for k in range(0, len(alvo), 50):
        lote = alvo[k : k + 50]
        r = _get(
            OPENALEX,
            {"filter": "openalex:" + "|".join(lote), "select": "id,abstract_inverted_index", "per-page": 50},
            headers={"User-Agent": CONTATO},
        )
        r.raise_for_status()
        vistos = set()
        for w in r.json()["results"]:
            oid = w["id"].rsplit("/", 1)[-1]
            vistos.add(oid)
            txt = resumo_de(w.get("abstract_inverted_index"))
            reg = registros[oid]
            reg["resumo_disponivel"] = txt is not None
            reg["resumo_brasil"] = bool(txt and BRASIL_RE.search(txt))
            reg["resumo_ml"] = bool(txt and ML_RE.search(txt))
        for oid in set(lote) - vistos:
            registros[oid]["resumo_disponivel"] = False
        time.sleep(0.2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="não coleta; regera o CSV a partir dos JSONs")
    ap.add_argument("--data-busca", default=date.today().isoformat())
    ap.add_argument("--out-csv", default=str(DATA / "protocolo_busca_T6.csv"))
    args = ap.parse_args()
    f_pm = DATA / "protocolo_busca_T6_pubmed.json"
    f_oa = DATA / "protocolo_busca_T6_openalex.json"
    f_man = DATA / "protocolo_busca_T6_manual.json"
    if not args.offline:
        for caminho, coletor in ((f_pm, coletar_pubmed), (f_oa, coletar_openalex)):
            dados = coletor(args.data_busca)
            caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")
            print(caminho.name, {k: v["n"] for k, v in dados["consultas"].items()})
    n = gerar_csv(f_pm, f_oa, f_man, Path(args.out_csv))
    print("CSV:", n["linhas"], "linhas;", n["sem_decisao"], "sem decisão")


if __name__ == "__main__":
    main()
