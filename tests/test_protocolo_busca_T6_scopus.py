"""Consistência do registro manual da busca Scopus executada em 2026-11-04."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
COLUNAS = [
    "base", "consulta", "etapa", "data_busca", "string_exata", "filtros",
    "n_recuperado", "registro_id", "primeiro_autor", "ano", "titulo",
    "periodico", "doi", "decisao", "motivo_ou_nota",
]


def _ler(nome: str) -> list[dict[str, str]]:
    with (ROOT / "data" / nome).open(encoding="utf-8", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        assert leitor.fieldnames == COLUNAS
        linhas = list(leitor)
    assert all(set(linha) == set(COLUNAS) for linha in linhas)
    assert all(valor is not None for linha in linhas for valor in linha.values())
    return linhas


def test_scopus_consultas_executadas_preservam_template_e_cadeias() -> None:
    template = _ler("protocolo_busca_T6_scopus_template.csv")
    executado = _ler("protocolo_busca_T6_scopus_executado.csv")
    consultas = [linha for linha in executado if linha["etapa"] == "consulta"]
    assert len(template) == len(consultas) == 6
    assert [linha["consulta"] for linha in consultas] == [f"Sc{i}" for i in range(1, 7)]
    assert [linha["n_recuperado"] for linha in consultas] == ["0", "3", "2", "6", "222", "2"]
    campos_bibliograficos = ["registro_id", "primeiro_autor", "ano", "titulo", "periodico", "doi", "decisao"]
    for planejada, realizada in zip(template, consultas):
        assert planejada["data_busca"] == planejada["n_recuperado"] == ""
        assert planejada["string_exata"] == realizada["string_exata"]
        assert realizada["base"] == "Scopus"
        assert realizada["data_busca"] == "2026-11-04"
        assert realizada["filtros"] == "PUBYEAR > 2009 AND PUBYEAR < 2027; sem filtro de idioma"
        assert all(realizada[campo] == "" for campo in campos_bibliograficos)
        assert "não houve exportação CSV em lote" in realizada["motivo_ou_nota"]
        assert "sem alteração de sintaxe" in realizada["motivo_ou_nota"]
    assert "referência internacional sem triagem" in consultas[4]["motivo_ou_nota"]


def test_scopus_triagem_deduplicada_rastreavel_por_eid_e_consulta() -> None:
    executado = _ler("protocolo_busca_T6_scopus_executado.csv")
    triagem = [linha for linha in executado if linha["etapa"] == "triagem"]
    assert len(executado) == 13
    assert len(triagem) == len({linha["doi"].lower() for linha in triagem}) == 7
    eids = {
        "105042821544": "Sc4",
        "105024222035": "Sc4",
        "85184522448": "Sc4",
        "85081933081": "Sc4",
        "85067679029": "Sc2;Sc3;Sc4;Sc6",
        "85028460352": "Sc2;Sc3;Sc4;Sc6",
        "85188631736": "Sc2",
    }
    for eid, consulta in eids.items():
        correspondentes = [linha for linha in triagem if f"EID Scopus: {eid};" in linha["motivo_ou_nota"]]
        assert len(correspondentes) == 1
        linha = correspondentes[0]
        assert linha["consulta"] == consulta
        assert linha["registro_id"] == ("W2739969158" if eid == "85028460352" else eid)
        assert linha["base"] == "Scopus" and linha["data_busca"] == "2026-11-04"
        assert all(linha[campo] for campo in ["primeiro_autor", "ano", "titulo", "periodico", "doi"])
        assert all(linha[campo] == "" for campo in ["string_exata", "filtros", "n_recuperado"])
        assert "consultas que recuperaram: " + consulta.replace(";", ", ") in linha["motivo_ou_nota"]
    assert all("Sc5" not in linha["consulta"] for linha in triagem)


def test_scopus_preserva_pita_e_nao_acrescenta_incluidos() -> None:
    original = _ler("protocolo_busca_T6.csv")
    triagem = [linha for linha in _ler("protocolo_busca_T6_scopus_executado.csv") if linha["etapa"] == "triagem"]
    incluidos = [linha for linha in triagem if linha["decisao"].startswith("incluido")]
    assert len(incluidos) == 1
    pita = incluidos[0]
    anterior = next(linha for linha in original if linha["registro_id"] == "W2739969158")
    assert pita["registro_id"] == anterior["registro_id"]
    assert pita["doi"] == anterior["doi"]
    assert pita["decisao"] == anterior["decisao"] == "incluido_com_ressalva"
    assert anterior["motivo_ou_nota"] in pita["motivo_ou_nota"]
    assert "recuperado também na Scopus" in pita["motivo_ou_nota"]
    assert sum(linha["decisao"] == "excluido" for linha in triagem) == 6
    anteriores = {linha["doi"].strip().lower() for linha in original if linha["etapa"] == "triagem" and linha["doi"]}
    assert sum(linha["doi"].lower() in anteriores for linha in triagem) == 4
    assert sum(linha["etapa"] == "triagem" for linha in original) + len(triagem) == 3431


def test_scopus_registra_afiliacoes_e_doi_completado() -> None:
    triagem = {linha["registro_id"]: linha for linha in _ler("protocolo_busca_T6_scopus_executado.csv") if linha["etapa"] == "triagem"}
    peng = triagem["85081933081"]
    assert peng["primeiro_autor"] == "Yaohao Peng"
    assert peng["periodico"] == "Journal of Business Cycle Research"
    assert "Brazilian Secretariat for Economic Policy" in peng["motivo_ou_nota"]
    assert "Ipea" in peng["motivo_ou_nota"]
    assert "pares rotulados de bases brasileiras de saúde" in peng["motivo_ou_nota"]
    piers = triagem["85188631736"]
    assert piers["doi"] == "10.1016/s2589-7500(23)00267-4"
    assert "predição clínica" in piers["motivo_ou_nota"]


def test_scopus_apendice_reflete_execucao_sem_alterar_numero_de_incluidos() -> None:
    texto = (ROOT / "text" / "latex" / "apendice_busca.tex").read_text(encoding="utf-8")
    assert "A Scopus não foi consultada" not in texto
    assert "4 de novembro de 2026" in texto
    assert "3.431 decisões registradas, e não estudos únicos" in texto
    assert "222 documentos, não triados individualmente" in texto
    assert "quatro estudos incluídos" in texto
    assert "exportação CSV em lote não foi realizada" in texto
    assert "OpenAlex e Scopus" in texto
    assert "—" not in texto and "--" not in texto
