#!/usr/bin/env python3
"""Gera data/protocolo_busca_T6.csv a partir dos resultados do protocolo T6.

Entradas:
  --pubmed   JSON produzido por scripts/protocolo_busca_T6.py
  --openalex JSON com os resultados das consultas O1 e O2 (OpenAlex), salvo
             pelo mesmo procedimento descrito em data/protocolo_busca_T6.md

O CSV contem apenas metadados bibliograficos (sem resumos, sem dados pessoais).
As decisoes de triagem (titulo/resumo) foram tomadas por leitura manual e estao
codificadas abaixo; nada e inferido automaticamente.
"""

from __future__ import annotations

import argparse
import csv
import json

DATA_BUSCA = "2026-11-03"
PERIODO = "2010-2026"

# --- PubMed P1: motivos de exclusao por PMID (leitura de titulo; resumo lido para os mais proximos)
EHR = {
    "21992670", "23920910", "23920911", "23920682", "25900270", "25889690", "26262308",
    "26262421", "27480749", "29871778", "30706199", "40270399",
}
REVISAO = {"26248090", "22403083", "36287481", "42659440", "36333542", "42628293"}
METODO = {
    "20694369", "21808821", "21344122", "21877013", "23920618", "25210903", "26886364",
    "27556963", "29947655", "31721900", "31482984", "33167998", "35846888", "32711532",
    "34130670", "29673602", "42144281", "42531684", "28716074", "30921353", "32887639",
    "41337538", "22341207", "38824562", "20392352",
}
SEGMENTACAO = {"25388307"}
MOTIVOS = {
    "ehr": "interoperabilidade ou registro eletronico em saude; sem classificacao de pares candidatos",
    "revisao": "revisao, editorial ou estrutura conceitual; sem classificador supervisionado de pares",
    "metodo": "metodo ou avaliacao de linkage (deterministico, probabilistico, por similaridade ou escore) sem classificador supervisionado de pares",
    "segmentacao": "segmentacao de nomes com modelo oculto de Markov; nao classifica pares candidatos",
    "aplicacao": "aplicacao epidemiologica que usa o linkage como etapa; sem classificador supervisionado de pares",
}

# --- OpenAlex: decisao por DOI (ou titulo, quando nao ha DOI)
OA_INCLUIDOS = {
    "10.1038/s41598-026-73928-3",
    "10.1007/978-3-319-64283-3_16",
    "10.23889/ijpds.v9i5.2902",
}
OA_DUPLICADO = {"10.21203/rs.3.rs-10229230/v1": "preprint do mesmo estudo de Lages et al. (2026), contado uma vez"}
OA_EXCLUIDOS = {
    "10.1016/j.compag.2019.104857": "linkage de dados agropecuarios; fora de saude e do Brasil",
    "10.17605/osf.io/pbf8t": "protocolo epidemiologico (ISTs e vulnerabilidade social); sem classificador de pares",
    "10.5753/sbc.14090.5.4": "minicurso sobre indices aprendidos na gestao de dados de saude; indexacao, nao classificacao de pares (triagem por titulo)",
    "10.1093/eurpub/ckaa165.1208": "resumo de oficina (EHESP); sem metodo de classificacao de pares",
    "10.1097/01.ccm.0001182192.43989.30": "resumo de congresso de medicina intensiva; fora do escopo",
    "10.1093/biosci/biz140": "colecoes biologicas; fora do escopo",
    "10.5281/zenodo.7957491": "conjunto de dados de politicos brasileiros; fora de saude",
    "10.3389/fmed.2025.1557589": "comentario sobre mortalidade pos-COVID; linkage como etapa, sem classificador de pares",
    "10.3310/nihropenres.1115247.1": "panorama de infraestrutura de dados (Reino Unido); fora do escopo",
}

# --- SciELO e BVS: amostra de registros proximos, lidos por titulo
SCIELO_AMOSTRA = [
    ("Brustulin", "2018", "Inclusao de etapa de pos-processamento deterministico para o aumento de performance do relacionamento (linkage) probabilistico", "Cadernos de Saude Publica", "10.1590/0102-311X00088117", "pos-processamento deterministico; sem classificador supervisionado"),
    ("Girianelli", "2019", "Confiabilidade interobservadores na classificacao de pares formados no relacionamento probabilistico entre bases de dados do SISMAMA", "Revista Brasileira de Epidemiologia", "10.1590/1980-549720190045", "concordancia entre revisores humanos na classificacao de pares; sem classificador supervisionado"),
    ("Goncalves", "2014", "Name segmentation using hidden Markov models and its application in record linkage", "Cadernos de Saude Publica", "10.1590/0102-311X00191313", "segmentacao de nomes; nao classifica pares candidatos"),
    ("Oliveira", "2016", "Accuracy of probabilistic and deterministic record linkage: the case of tuberculosis", "Revista de Saude Publica", "10.1590/S1518-8787.2016050006327", "acuracia de linkage probabilistico e deterministico; sem classificador supervisionado"),
    ("Rocha", "2026", "Data mining technique for identifying mortality patterns in a pulmonary tuberculosis cohort", "Cadernos de Saude Publica", "", "mineracao de dados sobre desfecho em coorte ja vinculada; nao classifica pares (DOI nao conferido)"),
    ("Garcia", "2026", "Record linkage in tuberculosis epidemiological surveillance in Brazil: systematic methodological literature review", "Revista Brasileira de Epidemiologia", "10.1590/1980-549720260045", "revisao de literatura; sem classificador supervisionado como objeto"),
    ("Tassinari", "2025", "Record linkage in public health datasets: a practical experience in a fast in-process analytical database", "Revista Brasileira de Epidemiologia", "10.1590/1980-549720250053", "experiencia pratica de linkage em banco analitico; sem classificador supervisionado"),
]

COLUNAS = [
    "base", "consulta", "etapa", "registro_id", "primeiro_autor", "ano", "titulo",
    "periodico", "doi", "decisao", "motivo_ou_nota",
]


def linha(**kw: str) -> dict:
    return {c: kw.get(c, "") for c in COLUNAS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pubmed", required=True)
    ap.add_argument("--openalex", required=True)
    ap.add_argument("--out", default="data/protocolo_busca_T6.csv")
    args = ap.parse_args()
    pm = json.load(open(args.pubmed, encoding="utf-8"))
    oa = json.load(open(args.openalex, encoding="utf-8"))
    L: list[dict] = []

    # contagens por consulta (etapa = consulta)
    cont = [
        ("PubMed", "P1", pm["consultas"]["P1"]["n"], "string do protocolo, sem rotulos de campo (mapeamento automatico de termos do PubMed); filtro 2010-2026 (data de publicacao)"),
        ("PubMed", "P2", pm["consultas"]["P2"]["n"], "mesma string restrita a titulo/resumo [tiab]"),
        ("PubMed", "P3", pm["consultas"]["P3"]["n"], "linkage AND aprendizado supervisionado em titulo/resumo AND Brazil ou Brasil em afiliacao [ad]"),
        ("PubMed", "P4", pm["consultas"]["P4"]["n"], "linkage AND aprendizado supervisionado em titulo/resumo, sem restricao de pais (referencia internacional); nenhum registro com Brazil ou Brasil nas afiliacoes"),
        ("SciELO", "S1", 0, "string do protocolo; sem filtro de ano (nenhum documento)"),
        ("SciELO", "S2", 131, "\"record linkage\"; sem filtro de ano; 131 registros, 78 titulos unicos, 56 de 2010 em diante"),
        ("SciELO", "S3", 2, "\"relacionamento de registros\"; sem filtro de ano"),
        ("SciELO", "S4", 6, "\"pareamento probabilistico\"; sem filtro de ano"),
        ("SciELO", "S5", 0, "\"record linkage\" AND (\"machine learning\" OR \"aprendizado de maquina\" OR supervised)"),
        ("SciELO", "S6", 0, "termos em portugues de linkage probabilistico AND aprendizado de maquina ou supervisionado"),
        ("BVS/LILACS", "B1", 3, "string do protocolo; sem filtro de ano; com filtro 2010-2026: 1"),
        ("BVS/LILACS", "B2", 71, "\"record linkage\" AND (\"machine learning\" OR \"aprendizado de maquina\"), filtro 2010-2026; colecao completa da BVS (71), subconjunto LILACS Plus (1); nao triada individualmente"),
        ("BVS/LILACS", "B3", 1, "linkage em portugues AND aprendizado de maquina, restrita a base LILACS; sem filtro de ano"),
        ("OpenAlex", "O1", len(oa["O1"]), "string do protocolo em titulo/resumo; 2010-2026"),
        ("OpenAlex", "O2", len(oa["O2"]), "variante com classifier e SUS; 2010-2026"),
        ("Scopus", "-", "", "nao executada: exige acesso institucional"),
        ("Google Scholar", "-", "", "nao usado como base (nao reprodutivel)"),
    ]
    for base, cid, n, nota in cont:
        L.append(linha(base=base, consulta=cid, etapa="consulta", registro_id=str(n), motivo_ou_nota=nota))

    # PubMed P1: todos os 73 registros
    for pmid in sorted(pm["consultas"]["P1"]["pmids"], key=lambda x: (pm["registros"][x]["ano"], x)):
        r = pm["registros"][pmid]
        if pmid == "31721900":
            m = "metodo de escore para identificar pares verdadeiros sem revisao manual (resumo lido); nao descreve classificador treinado em pares rotulados; precedente brasileiro mais proximo"
        elif pmid in EHR:
            m = MOTIVOS["ehr"]
        elif pmid in REVISAO:
            m = MOTIVOS["revisao"]
        elif pmid in METODO:
            m = MOTIVOS["metodo"]
        elif pmid in SEGMENTACAO:
            m = MOTIVOS["segmentacao"]
        else:
            m = MOTIVOS["aplicacao"]
        L.append(linha(base="PubMed", consulta="P1", etapa="triagem", registro_id="PMID " + pmid,
                       primeiro_autor=r["primeiro_autor"], ano=r["ano"], titulo=r["titulo"],
                       periodico=r["revista"], doi=r["doi"], decisao="excluido", motivo_ou_nota=m))

    # SciELO: amostra
    for aut, ano, tit, per, doi, mot in SCIELO_AMOSTRA:
        L.append(linha(base="SciELO", consulta="S2", etapa="triagem", primeiro_autor=aut, ano=ano,
                       titulo=tit, periodico=per, doi=doi, decisao="excluido", motivo_ou_nota=mot))

    # BVS
    L.append(linha(base="BVS/LILACS", consulta="B1", etapa="triagem", registro_id="MEDLINE mdl-40270399",
                   primeiro_autor="Lorenzer", ano="2025",
                   titulo="Harmonization of Electronic Medical Records for Federated Learning: Addressing Challenges in International Healthcare Collaborations",
                   periodico="Studies in Health Technology and Informatics", doi="10.3233/SHTI250175",
                   decisao="excluido", motivo_ou_nota=MOTIVOS["ehr"]))
    L.append(linha(base="BVS/LILACS", consulta="B1", etapa="triagem", registro_id="MEDLINE mdl-18038088; LILACS lil-467823",
                   primeiro_autor="Bierrenbach", ano="2007",
                   titulo="Tuberculosis incidence and cure rates, Brazil, 2000-2004", periodico="Revista de Saude Publica",
                   decisao="excluido", motivo_ou_nota="fora do periodo (2007); registro duplicado em MEDLINE e LILACS"))
    L.append(linha(base="BVS/LILACS", consulta="B3", etapa="triagem", registro_id="LILACS biblio-1058589",
                   primeiro_autor="Villena", ano="2019",
                   titulo="Automatic keyword retrieval from clinical texts: an application of natural language processing to massive data of Chilean suspected diagnosis",
                   periodico="Revista medica de Chile", decisao="excluido",
                   motivo_ou_nota="processamento de linguagem natural em textos clinicos do Chile; nao classifica pares de linkage"))

    # OpenAlex
    for chave in ("O1", "O2"):
        for w in oa[chave]:
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            aut = w["authorships"][0]["author"]["display_name"] if w["authorships"] else ""
            loc = (w.get("primary_location") or {}).get("source") or {}
            if doi in OA_INCLUIDOS:
                dec, mot = "incluido", "classificador supervisionado aplicado a pares candidatos em bases brasileiras de saude"
            elif doi in OA_DUPLICADO:
                dec, mot = "duplicado", OA_DUPLICADO[doi]
            else:
                dec, mot = "excluido", OA_EXCLUIDOS.get(doi, "SEM DECISAO REGISTRADA")
            L.append(linha(base="OpenAlex", consulta=chave, etapa="triagem", registro_id=w["id"].rsplit("/", 1)[-1],
                           primeiro_autor=aut, ano=str(w["publication_year"]), titulo=w["title"] or "",
                           periodico=loc.get("display_name") or "", doi=doi, decisao=dec, motivo_ou_nota=mot))

    # Incluidos finais (nao recuperados por string: identificados por item conhecido)
    L.append(linha(base="item conhecido", consulta="-", etapa="incluido", primeiro_autor="Paiva", ano="2025",
                   titulo="Utilizando aprendizado de maquina para automatizar processos de revisao manual de links de dados provenientes do linkage probabilistico de dados dos sistemas de informacao do Sistema Unico de Saude",
                   periodico="Dissertacao, Instituto de Computacao, UFRJ", doi="",
                   decisao="incluido", motivo_ou_nota="nao recuperada por nenhuma string; identificada por conhecimento previo (http://hdl.handle.net/11422/25950); arvores de decisao em pares do SIM e do Sinan-TB"))
    L.append(linha(base="item conhecido", consulta="-", etapa="relacionado", primeiro_autor="Pita", ano="2018",
                   titulo="On the Accuracy and Scalability of Probabilistic Data Linkage Over the Brazilian 114 Million Cohort",
                   periodico="IEEE Journal of Biomedical and Health Informatics", doi="10.1109/JBHI.2018.2796941",
                   decisao="relacionado", motivo_ou_nota="nao recuperado por nenhuma string; o resumo descreve a ferramenta hibrida AtyImo (linkage probabilistico) sem classificador supervisionado explicito; mantido como estudo relacionado a Pita et al. (2017)"))
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        w.writeheader()
        w.writerows(L)
    pend = [r for r in L if r["motivo_ou_nota"] == "SEM DECISAO REGISTRADA"]
    print(len(L), "linhas;", len(pend), "sem decisao")


if __name__ == "__main__":
    main()
