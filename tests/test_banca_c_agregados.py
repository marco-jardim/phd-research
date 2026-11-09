"""Contas e camadas editoriais do Estudo 4, sem dependência de dados privados."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts.banca_c_agregados import (
    MODES,
    aggregate_qw2,
    equivalent_costs,
    read_csv,
    render_qw2,
    write_csv,
)
from terminologia import contem

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LATEX = ROOT / "text/latex"


def config():
    return yaml.safe_load((ROOT / "gzcmd/gzcmd_v3_config.yaml").read_text(encoding="utf-8"))


def test_cost_aggregate_reproduces_and_reconciles_all_components(tmp_path):
    rows = equivalent_costs(DATA, config())
    output = tmp_path / "costs.csv"
    write_csv(output, rows)
    assert output.read_bytes() == (DATA / "custos_equivalentes_exploratorios.csv").read_bytes()
    assert len(rows) == 12
    for row in rows:
        assert row["exploratorio_post_hoc"] and row["revisor_perfeito"]
        assert row["custo_humano"] == row["revisoes_humanas"] * row["C_H"]
        assert row["custo_llm"] == row["pares_llm"] * row["C_LLM"]
        assert row["custo_fp"] == row["fp"] * row["C_FP"]
        assert row["perda_total"] == pytest.approx(
            row["subtotal_revisao_fp"] + row["custo_fn_comum"])
    vigil = {r["braco"]: r for r in rows if r["modo"] == "vigilancia"}
    assert [vigil[k]["subtotal_revisao_fp"] for k in
            ("P", "O3_recuperacao_equivalente", "P_MATCH_ao_humano")] == [307.72, 125, 122.72]
    equivalent = [r for r in vigil.values() if "mesma_carga" not in r["braco"]]
    assert {r["par2_recuperados"] for r in equivalent} == {23}
    assert {r["custo_fn_comum"] for r in equivalent} == {650}


def test_o2_costs_same_load_and_equal_recovery():
    rows = equivalent_costs(DATA, config())
    confirmation = {r["braco"]: r for r in rows if r["modo"] == "confirmacao"}
    assert confirmation["O2_mesma_carga_humana_P"]["revisoes_humanas"] == 17
    assert confirmation["O2_recuperacao_equivalente"]["revisoes_humanas"] == 11
    for key in ("O2_mesma_carga_humana_P", "O2_recuperacao_equivalente"):
        assert confirmation[key]["par2_recuperados"] == 9
        assert confirmation[key]["fp"] == 0
    vigilance = {r["braco"]: r for r in rows if r["modo"] == "vigilancia"}
    assert vigilance["O2_recuperacao_equivalente"]["subtotal_revisao_fp"] == 4163
    assert vigilance["O2_mesma_carga_humana_P"]["par2_recuperados"] == 12
    assert vigilance["O2_mesma_carga_humana_P"]["par2_perdidos"] == 24


def test_costs_are_read_from_policy_not_hardcoded():
    changed = copy.deepcopy(config())
    changed["decision_policy"]["modes"]["vigilancia"]["costs"]["llm_review"] = 0.01
    p = next(r for r in equivalent_costs(DATA, changed)
             if r["modo"] == "vigilancia" and r["braco"] == "P")
    assert p["custo_llm"] == 3.44
    assert p["subtotal_revisao_fp"] == 309.44


def test_qw2_aggregate_has_only_whitelisted_fields_and_reconciled_counts():
    rows = read_csv(DATA / "qw2_kimi_agregado.csv")
    assert [r["modo"] for r in rows] == list(MODES)
    assert [int(rows[0][k]) for k in ("vp", "fp", "fn", "vn")] == [72, 6, 32, 1300]
    for row in rows:
        assert len(row["fonte_sha256"]) == 64
        assert int(row["n"]) == sum(int(row[k]) for k in ("vp", "fp", "fn", "vn"))
        assert int(row["n"]) == int(row["consenso"]) + int(row["arbitro"])
        assert set(row) == {"modo", "fonte", "fonte_sha256", "timestamp", "modelo",
                            "seed", "n", "vp", "fp", "fn", "vn", "consenso", "arbitro"}
    for name, text in render_qw2(rows).items():
        assert text == (LATEX / "tables/editorial" / name).read_text(encoding="utf-8")
        assert "fontes não versionadas" not in text.lower()
        assert r"\path{data/qw2_kimi_agregado.csv}" in text


def test_qw2_extraction_never_exports_free_text_or_pair_ids(tmp_path):
    for mode in MODES:
        doc = dict(mode=mode, n_reviewed=4, n_errors=0, timestamp="2026-02-15",
                   model="test", seed=43, results=[{"pair_id": "PRIVATE", "text": "PRIVATE"}],
                   metrics=dict(gt_tp=1, gt_fp=1, gt_fn=1, gt_tn=1,
                                decisions={"MATCH": 2, "NONMATCH": 2},
                                protocols={"consensus": 3, "arbiter": 1}))
        (tmp_path / f"qw2_{mode}_kimi.json").write_text(json.dumps(doc), encoding="utf-8")
    rows = aggregate_qw2(tmp_path)
    assert len(rows) == 2 and "PRIVATE" not in json.dumps(rows)
    path = tmp_path / "qw2_vigilancia_kimi.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["metrics"]["gt_fn"] = 2
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="matriz QW2"):
        aggregate_qw2(tmp_path)


def test_editorial_order_and_limitations_remain_explicit():
    text = (LATEX / "estudo4_gzcmd.tex").read_text(encoding="utf-8")
    assert text.index(r"\label{sec:e4-exploratorias-metodo}") < text.index(r"\section{Resultados}")
    assert text.index(r"\label{sec:e4-arranjo}") < text.index(r"\label{sec:e4-robustez}") < text.index(r"\label{sec:e4-mesma-carga}")
    appendix = (LATEX / "apendice_gzcmd.tex").read_text(encoding="utf-8")
    section = text.split(r"\section{Discussão}", 1)[1]
    discussion = section.split("\n\n", 2)[1]
    # Revisão 2 (D2/C01r): a Discussão abre pela ordenação; P é o braço do registro interno,
    # nunca o "resultado principal".
    assert discussion.startswith("Nesta coorte, abaixo do corte do OpenRecLink, a redução da revisão manual veio da ordenação")
    assert "braço ASR" in discussion
    assert "resultado principal" not in section.split(r"\subsection{Limitações}")[0]
    assert "ASR é o braço do registro interno e não o resultado principal do capítulo" in text
    acceptor = next(p for p in section.split("\n\n") if p.startswith("Abaixo do corte, o ganho automático de ASR veio do LLM"))
    assert "25 e 12 potenciais pares" in acceptor and "dos quais 21 e 4" in acceptor
    assert "fig_custo_computacional.pdf" not in text
    assert "fig:cap8-cascata-reducao" not in text
    cascade = (LATEX / "tables/editorial/tab_cap8_cascata_v31.tex").read_text(encoding="utf-8")
    # Decisão do autor (fatia T): a tabela editorial é a saída do gerador, com o rótulo histórico único.
    assert cascade.count(r"\label{tab:cap8-cascata-reducao}") == 1
    assert r"\label{tab:cap8-cascata-reducao}" not in text
    for term in ("viés de verificação", "circularidade", r"\textit{post mortem}",
                 "checkout original do Windows", "depósito Zenodo com acesso restrito",
                 "reavaliação periódica dos LLM", "307{,}72", "122{,}72",
                 "A análise de estabilidade descreve a estabilidade interna"):
        assert term in text
    assert not contem(text, "área cinza convencional", arquivo="text/latex/estudo4_gzcmd.tex")  # migrado: terminologia 3.6
    assert "taxas" not in text.replace("taxas_crossfit", "").replace("taxas_erro", "")
    assert "Ele ajuda" not in text
    # A matriz do QW2 foi para o Apêndice (revisão 2).
    assert r"\input{tables/editorial/tab_qw2_confusion.tex}" in appendix


def test_arrangement_headings_identify_scope_without_changing_numeric_cells():
    text = (LATEX / "tables/editorial/tab_cap8_arranjo_corte.tex").read_text(encoding="utf-8")
    # Decisão do autor (fatia T): cabeçalhos e prefixos (i)/(ii) voltam ao original.
    assert "Vigilância: base inteira" in text
    assert "Confirmação: base inteira" in text
    assert "escopo: 61.696 pares" not in text
    assert "(i) Corte & 211 &" in text and "(ii) Humano integral &" in text
    from scripts.tabelas_arranjo_editorial import render_arrangement
    original = (LATEX / "tables/tab_cap8_arranjo_corte.tex").read_text(encoding="utf-8")
    assert text == render_arrangement(original)
