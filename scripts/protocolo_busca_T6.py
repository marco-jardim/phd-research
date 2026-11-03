#!/usr/bin/env python3
"""Protocolo de busca bibliografica (T6): consultas ao PubMed via E-utilities.

Executa as strings do protocolo, registra data, string exata, contagem e
metadados bibliograficos (PMID, ano, periodico, primeiro autor, titulo, DOI,
paises de afiliacao). Nao grava resumos nem dados pessoais.

Uso:
    python scripts/protocolo_busca_T6.py --out $env:TEMP/t6_pubmed.json

Saida: JSON com consultas (string, contagem, traducao automatica do PubMed) e
registros recuperados. SciELO e BVS/LILACS exigem navegador (desafio de
JavaScript) e foram consultadas manualmente; ver data/protocolo_busca_T6.md.
"""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from datetime import date

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PERIODO = {"datetype": "pdat", "mindate": "2010", "maxdate": "2026"}

CONSULTAS = {
    "P1": (
        '("record linkage" OR "relacionamento de registros" OR "pareamento probabilístico") '
        'AND ("machine learning" OR "aprendizado de máquina" OR supervised) '
        "AND (Brasil OR Brazil OR SIM OR SINAN OR DATASUS)"
    ),
    "P2": (
        '("record linkage"[tiab] OR "relacionamento de registros"[tiab] OR "pareamento probabilístico"[tiab]) '
        'AND ("machine learning"[tiab] OR "aprendizado de máquina"[tiab] OR supervised[tiab]) '
        "AND (Brasil[tiab] OR Brazil[tiab] OR Brazilian[tiab] OR SIM[tiab] OR SINAN[tiab] OR DATASUS[tiab])"
    ),
    "P3": (
        '("record linkage"[tiab] OR "data linkage"[tiab]) '
        'AND ("machine learning"[tiab] OR supervised[tiab]) '
        "AND (Brazil[ad] OR Brasil[ad])"
    ),
    "P4": (
        '("record linkage"[tiab] OR "data linkage"[tiab]) '
        'AND ("machine learning"[tiab] OR supervised[tiab])'
    ),
}


def esearch(term: str) -> dict:
    params = {"db": "pubmed", "term": term, "retmax": 500, "retmode": "json", **PERIODO}
    r = requests.get(EUTILS + "esearch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    return r.json()["esearchresult"]


def efetch(pmids: list[str]) -> list[dict]:
    regs: list[dict] = []
    for i in range(0, len(pmids), 100):
        lote = pmids[i : i + 100]
        r = requests.get(
            EUTILS + "efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(lote), "retmode": "xml"},
            timeout=120,
        )
        r.raise_for_status()
        raiz = ET.fromstring(r.content)
        for art in raiz.findall(".//PubmedArticle"):
            pmid = art.findtext(".//MedlineCitation/PMID")
            titulo = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
            ano = art.findtext(".//JournalIssue/PubDate/Year") or (art.findtext(".//JournalIssue/PubDate/MedlineDate") or "")[:4]
            revista = art.findtext(".//Journal/Title") or ""
            autor = art.findtext(".//AuthorList/Author/LastName") or ""
            doi = ""
            for aid in art.findall("./PubmedData/ArticleIdList/ArticleId"):
                if aid.get("IdType") == "doi":
                    doi = (aid.text or "").strip()
            afil = " ".join("".join(a.itertext()) for a in art.findall(".//AffiliationInfo/Affiliation"))
            pais_br = "Brazil" in afil or "Brasil" in afil
            regs.append(
                {
                    "pmid": pmid,
                    "ano": ano,
                    "revista": revista,
                    "primeiro_autor": autor,
                    "titulo": titulo,
                    "doi": doi,
                    "afiliacao_brasil": pais_br,
                }
            )
        time.sleep(0.4)
    return regs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    saida = {"data_busca": date.today().isoformat(), "periodo": "2010-2026", "consultas": {}, "registros": {}}
    uniao: set[str] = set()
    for chave, termo in CONSULTAS.items():
        res = esearch(termo)
        ids = res["idlist"]
        saida["consultas"][chave] = {
            "string": termo,
            "n": int(res["count"]),
            "pmids": ids,
            "traducao_automatica": res.get("querytranslation", ""),
        }
        uniao.update(ids)
        time.sleep(0.4)
    for reg in efetch(sorted(uniao)):
        saida["registros"][reg["pmid"]] = reg
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=1)
    for k, v in saida["consultas"].items():
        print(k, v["n"])
    print("uniao", len(uniao))


if __name__ == "__main__":
    main()
