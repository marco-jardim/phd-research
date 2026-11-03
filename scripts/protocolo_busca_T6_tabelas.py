#!/usr/bin/env python3
"""Gera data/protocolo_busca_T6.csv a partir dos JSONs do protocolo T6.

Entradas (todas versionadas em data/):
  protocolo_busca_T6_pubmed.json    brutos do PubMed (IDs e metadados)
  protocolo_busca_T6_openalex.json  brutos do OpenAlex (IDs e metadados)
  protocolo_busca_T6_manual.json    consultas por navegador (SciELO, BVS/LILACS), Consensus,
                                    buscas exploratorias e itens conhecidos, com decisao de triagem

O CSV traz apenas metadados bibliograficos (sem resumos e sem dados pessoais).
As decisoes de triagem (titulo; resumo lido nos registros mais proximos do criterio)
foram tomadas por leitura e estao codificadas abaixo; nada alem da regra de titulo para
registros sem qualquer termo de linkage e inferido automaticamente.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

COLUNAS = [
    "base", "consulta", "etapa", "data_busca", "string_exata", "filtros", "n_recuperado",
    "registro_id", "primeiro_autor", "ano", "titulo", "periodico", "doi", "decisao", "motivo_ou_nota",
]

SEM_DECISAO = "SEM DECISAO REGISTRADA"

# --- PubMed: motivos de exclusao por PMID ----------------------------------------------------
EHR = {
    "21992670", "23920910", "23920911", "23920682", "25900270", "25889690", "26262308",
    "26262421", "27480749", "29871778", "30706199", "40270399",
}
REVISAO = {"26248090", "22403083", "36287481", "42659440", "36333542", "42628293"}
METODO = {
    "20694369", "21808821", "21344122", "21877013", "23920618", "25210903", "26886364",
    "27556963", "29947655", "31482984", "33167998", "35846888", "32711532",
    "34130670", "29673602", "42144281", "42531684", "28716074", "30921353", "32887639",
    "41337538", "22341207", "38824562", "20392352",
}
SEGMENTACAO = {"25388307"}
APLICACAO = {
    "21359465", "21503555", "22124739", "22892975", "23383591", "25210906", "26152148", "26425565",
    "26884501", "27192405", "28146165", "28444001", "28492777", "28545053", "28728061", "28971480",
    "29211201", "29471902", "29489943", "29928025", "30328938", "31166417", "31166527", "31596402",
    "31625225", "31800783", "32074197", "33439942", "34637451",
}
DUARTE = "31721900"
MOTIVOS = {
    "ehr": "interoperabilidade ou registro eletrônico em saúde; sem classificação de pares candidatos",
    "revisao": "revisão, editorial ou estrutura conceitual; sem classificador supervisionado de pares",
    "metodo": "método ou avaliação de linkage (determinístico, probabilístico, por similaridade ou escore) sem classificador supervisionado de pares",
    "segmentacao": "segmentação de nomes com modelo oculto de Markov; não classifica pares candidatos",
    "aplicacao": "aplicação epidemiológica que usa o linkage como etapa; sem classificador supervisionado de pares",
    "duarte": "método de escore para identificar pares verdadeiros sem revisão manual (resumo lido); não descreve classificador treinado em pares rotulados; precedente brasileiro mais próximo",
    "internacional": "sem afiliação no Brasil (referência internacional); critério exige bases brasileiras de saúde",
    "sem_brasil": "título trata de linkage, mas o resumo não faz referência ao Brasil nem a base brasileira (filtro de palavras-chave sobre o resumo)",
    "sem_ml": "título trata de linkage e o resumo cita o Brasil, mas não cita aprendizado supervisionado (filtro de palavras-chave sobre o resumo)",
    "titulo": "título sem relação com linkage ou com classificação de pares (triagem por título)",
}
# Registros recuperados apenas pelas cadeias ampliadas (A1): codigo de motivo por PMID.
PUBMED_EXTRA: dict[str, str] = {"33886759": "aplicacao", "39999528": "aplicacao"}
PUBMED_RELACIONADOS = {
    "29505402": "ferramenta híbrida de linkage probabilístico (AtyImo) na coorte de 114 milhões de brasileiros; o resumo não descreve classificador supervisionado; relacionado a Pita et al. (2017)",
}

# --- OpenAlex ------------------------------------------------------------------------------
OA_INCLUIDOS = {
    "10.1038/s41598-026-73928-3": ("incluido", "classificador supervisionado aplicado a pares candidatos em bases brasileiras de saúde (SIM, SIH-SUS, Sinan, e-SUS VS)"),
    "10.23889/ijpds.v9i5.2902": ("incluido", "classificador supervisionado aplicado a pares candidatos em dados brasileiros do SUS"),
    "10.1007/978-3-319-64283-3_16": ("incluido_com_ressalva", "classificadores supervisionados na avaliação da acurácia do linkage probabilístico; o resumo não detalha a base brasileira"),
}
OA_DUPLICADO = {"10.21203/rs.3.rs-10229230/v1": "preprint do mesmo estudo de Lages et al. (2026), contado uma vez"}
OA_EXCLUIDOS = {
    "10.1016/j.compag.2019.104857": "linkage de dados agropecuários; fora de saúde e do Brasil",
    "10.17605/osf.io/pbf8t": "protocolo epidemiológico (ISTs e vulnerabilidade social); sem classificador de pares",
    "10.5753/sbc.14090.5.4": "minicurso sobre índices aprendidos na gestão de dados de saúde; indexação, não classificação de pares (triagem por título)",
    "10.1093/eurpub/ckaa165.1208": "resumo de oficina (EHESP); sem método de classificação de pares",
    "10.1097/01.ccm.0001182192.43989.30": "resumo de congresso de medicina intensiva; fora do escopo",
    "10.1093/biosci/biz140": "coleções biológicas; fora do escopo",
    "10.5281/zenodo.7957491": "conjunto de dados de políticos brasileiros; fora de saúde",
    "10.3389/fmed.2025.1557589": "comentário sobre mortalidade pós-COVID; linkage como etapa, sem classificador de pares",
    "10.3310/nihropenres.1115247.1": "panorama de infraestrutura de dados (Reino Unido); fora do escopo",
}
# Decisoes por ID OpenAlex para registros sem DOI ou recuperados nas cadeias ampliadas: id -> (decisao, motivo)
OA_POR_ID: dict[str, tuple[str, str]] = {
    "W2107799451": ("excluido", "linkage genérico, sem indicação de saúde ou de base brasileira (triagem por título; resumo indisponível)"),
    "W2112263062": ("excluido", "linkage genérico, sem indicação de saúde ou de base brasileira (triagem por título; resumo indisponível)"),
    "W2145908616": ("excluido", "linkage genérico, sem indicação de saúde ou de base brasileira (triagem por título; resumo indisponível)"),
    "W2606267278": ("excluido", "linkage genérico, sem indicação de saúde ou de base brasileira (triagem por título; resumo indisponível)"),
    "W2726467847": ("excluido", "linkage genérico, sem indicação de saúde ou de base brasileira (triagem por título; resumo indisponível)"),
    "W2733471169": ("excluido", "linkage genérico, sem indicação de saúde ou de base brasileira (triagem por título; resumo indisponível)"),
    "W2791169425": ("excluido", "linkage probabilístico em sistemas paralelos híbridos; sem classificador supervisionado (triagem por título; resumo indisponível)"),
    "W2923968079": ("excluido", "linkage genérico, sem indicação de saúde ou de base brasileira (triagem por título; resumo indisponível)"),
    "W2929962453": ("excluido", "linkage genérico, sem indicação de saúde ou de base brasileira (triagem por título; resumo indisponível)"),
    "W2946978350": ("excluido", "aprendizado supervisionado em linkage de registros médicos, sem indicação de base brasileira no título (resumo indisponível)"),
    "W2986757196": ("excluido", "desfecho clínico (overdose) com linkage como etapa; fora do escopo (triagem por título)"),
    "W3014074327": ("excluido", "linkage genérico, sem indicação de saúde ou de base brasileira (triagem por título; resumo indisponível)"),
    "W4241084441": ("excluido", "documento acessório de revisão por pares; fora do escopo (triagem por título)"),
    "W4285598592": ("excluido", "documento acessório de revisão por pares; fora do escopo (triagem por título)"),
    "W4311428403": ("excluido", "linkage genérico, sem indicação de saúde ou de base brasileira (triagem por título; resumo indisponível)"),
    "W4413323467": ("excluido", "linkage genérico, sem indicação de saúde ou de base brasileira (triagem por título; resumo indisponível)"),
    "W4416469396": ("excluido", "documento acessório de revisão por pares; fora do escopo (triagem por título)"),
}
LINK_RE = re.compile(
    r"linkage|record match|entity (resolution|matching)|deduplic|data matching|record link|"
    r"relacionamento|pareamento|vincula|patient matching|identity resolution|name matching|"
    r"probabilistic (matching|linking)|record-linkage",
    re.I,
)


def _carregar(caminho: Path) -> dict:
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def _linha(**kw: object) -> dict:
    return {c: ("" if kw.get(c) is None else kw.get(c)) for c in COLUNAS}


def decidir_pubmed(pmid: str, reg: dict, consultas: list[str]) -> tuple[str, str]:
    if pmid == DUARTE:
        return "excluido", MOTIVOS["duarte"]
    for conj, chave in ((EHR, "ehr"), (REVISAO, "revisao"), (METODO, "metodo"), (SEGMENTACAO, "segmentacao"), (APLICACAO, "aplicacao")):
        if pmid in conj:
            return "excluido", MOTIVOS[chave]
    if pmid in PUBMED_RELACIONADOS:
        return "relacionado", PUBMED_RELACIONADOS[pmid]
    if pmid in PUBMED_EXTRA:
        return "excluido", MOTIVOS.get(PUBMED_EXTRA[pmid], PUBMED_EXTRA[pmid])
    if not reg.get("afiliacao_brasil") and set(consultas) <= {"P4"}:
        return "excluido", MOTIVOS["internacional"]
    if not LINK_RE.search(reg.get("titulo", "")):
        return "excluido", MOTIVOS["titulo"]
    return "excluido", SEM_DECISAO


def decidir_openalex(reg: dict) -> tuple[str, str]:
    doi = reg.get("doi", "")
    if doi in OA_INCLUIDOS:
        return OA_INCLUIDOS[doi]
    if doi in OA_DUPLICADO:
        return "duplicado", OA_DUPLICADO[doi]
    if doi in OA_EXCLUIDOS:
        return "excluido", OA_EXCLUIDOS[doi]
    if reg["id"] in OA_POR_ID:
        return OA_POR_ID[reg["id"]]
    if not LINK_RE.search(reg.get("titulo", "")):
        return "excluido", MOTIVOS["titulo"]
    if reg.get("resumo_disponivel"):
        if not reg.get("resumo_brasil"):
            return "excluido", MOTIVOS["sem_brasil"]
        if not reg.get("resumo_ml"):
            return "excluido", MOTIVOS["sem_ml"]
    return "excluido", SEM_DECISAO


def gerar_csv(f_pubmed: Path, f_openalex: Path, f_manual: Path, saida: Path) -> dict:
    pm, oa, man = _carregar(f_pubmed), _carregar(f_openalex), _carregar(f_manual)
    L: list[dict] = []

    # PubMed
    for cid, c in pm["consultas"].items():
        L.append(_linha(base="PubMed", consulta=cid, etapa="consulta", data_busca=pm["data_busca"],
                        string_exata=c["string_exata"], filtros=c["filtros"], n_recuperado=c["n"],
                        motivo_ou_nota=c.get("descricao", "")))
    por_pmid: dict[str, list[str]] = {}
    for cid, c in pm["consultas"].items():
        for p in c["pmids"]:
            por_pmid.setdefault(p, []).append(cid)
    for pmid in sorted(por_pmid, key=lambda x: (pm["registros"][x]["ano"], x)):
        r = pm["registros"][pmid]
        dec, mot = decidir_pubmed(pmid, r, por_pmid[pmid])
        L.append(_linha(base="PubMed", consulta=";".join(por_pmid[pmid]), etapa="triagem", data_busca=pm["data_busca"],
                        registro_id="PMID " + pmid, primeiro_autor=r["primeiro_autor"], ano=r["ano"],
                        titulo=r["titulo"], periodico=r["revista"], doi=r["doi"], decisao=dec, motivo_ou_nota=mot))

    # OpenAlex
    for cid, c in oa["consultas"].items():
        L.append(_linha(base="OpenAlex", consulta=cid, etapa="consulta", data_busca=oa["data_busca"],
                        string_exata=c["string_exata"],
                        filtros=f"{c['modo']}; {c['filtros']}; url: {c['url_primeira_pagina']}; páginas: {c['n_paginas']}",
                        n_recuperado=c["n"], motivo_ou_nota=c.get("descricao", "")))
    por_id: dict[str, list[str]] = {}
    for cid, c in oa["consultas"].items():
        for i in c["ids"]:
            por_id.setdefault(i, []).append(cid)
    for oid in sorted(por_id, key=lambda x: (-(oa["registros"][x]["ano"] or 0), x)):
        r = oa["registros"][oid]
        dec, mot = decidir_openalex(r)
        L.append(_linha(base="OpenAlex", consulta=";".join(por_id[oid]), etapa="triagem", data_busca=oa["data_busca"],
                        registro_id=oid, primeiro_autor=r["primeiro_autor"], ano=r["ano"], titulo=r["titulo"],
                        periodico=r["periodico"], doi=r["doi"], decisao=dec, motivo_ou_nota=mot))

    # Registro manual (navegador, Consensus, exploratorias, itens conhecidos)
    for c in man["consultas"]:
        L.append(_linha(base=c["base"], consulta=c["consulta"], etapa="consulta", data_busca=c.get("data_busca", man["data_busca"]),
                        string_exata=c.get("string_exata", ""), filtros=c.get("filtros", ""),
                        n_recuperado=c.get("n"), motivo_ou_nota=c.get("nota", "")))
    for r in man["registros"]:
        L.append(_linha(base=r["base"], consulta=r.get("consulta", ""), etapa=r.get("etapa", "triagem"),
                        data_busca=r.get("data_busca", man["data_busca"]), registro_id=r.get("registro_id", ""),
                        primeiro_autor=r.get("primeiro_autor", ""), ano=r.get("ano", ""), titulo=r.get("titulo", ""),
                        periodico=r.get("periodico", ""), doi=r.get("doi", ""), decisao=r["decisao"],
                        motivo_ou_nota=r.get("motivo", "")))

    saida = Path(saida)
    with saida.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        w.writeheader()
        w.writerows(L)
    sem = sum(1 for r in L if r["motivo_ou_nota"] == SEM_DECISAO)
    return {"linhas": len(L), "sem_decisao": sem}
