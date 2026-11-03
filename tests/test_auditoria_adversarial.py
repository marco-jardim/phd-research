import numpy as np
import pandas as pd
import pytest

from gzcmd.intervals import wilson95
from scripts.auditar_funil_adversarial import endpoint_projection, estimate_rates, stage_metrics
from scripts.fig_varredura_custos import decimal_pt


def test_wilson_known_cascade_interval():
    assert wilson95(203, 247) == pytest.approx((.769327, .864540), abs=1e-6)
    assert wilson95(0, 10)[0] == 0
    assert wilson95(10, 10)[1] == 1
    with pytest.raises(ValueError):
        wilson95(1, 0)


def test_new_stage_excludes_unsure_from_resolved_denominators():
    frame = pd.DataFrame({"y": [1, 1, 1, 0, 0, 0],
                          "llm_decision": ["MATCH", "NONMATCH", "UNSURE", "MATCH", "NONMATCH", "UNSURE"]})
    metrics = stage_metrics(frame)
    assert metrics["VP"] == metrics["FP"] == metrics["FN"] == metrics["VN"] == 1
    assert metrics["sens"] == metrics["espec"] == .5
    assert metrics["sens_todos_vinculos"] == pytest.approx(1 / 3)
    assert metrics["UNSURE"] == 2


def test_leaveout_reestimation_includes_unsure_in_m3_denominators():
    ids = [str(i) for i in range(12)]
    frame = pd.DataFrame({"band": ["a"] * 12, "y": [1] * 6 + [0] * 6}, index=ids)
    decisions = {pid: "UNSURE" for pid in ids}
    decisions["0"], decisions["6"] = "NONMATCH", "MATCH"
    fixed = {"a": {"e_fp": .2, "e_fn": .3}}
    rates, details = estimate_rates(frame, decisions, fixed, {"0"})
    assert rates["a"].e_fn == 0
    assert rates["a"].e_fp == pytest.approx(1 / 6)
    assert rates["a"].e_unsure == pytest.approx(10 / 11)
    assert details[0]["n"] == 11
    fallback, _ = estimate_rates(frame, decisions, fixed, {"0", "1", "2"})
    assert fallback["a"].e_fp == .2 and fallback["a"].e_fn == .3
    assert fallback["a"].e_unsure == 0


def test_decimal_formatter():
    assert decimal_pt(97.5) == "97,5"
    assert decimal_pt(1250) == "1.250"
    assert decimal_pt(.001) == "0,001"


def test_pending_acceptance_specificity_has_bounds_not_imputed_classes():
    from gzcmd.policy_engine import Costs, HumanError, LLMError
    frame = pd.DataFrame({"y": [0, 0, 1], "p_cal": [.1, .2, .8], "band": ["a"] * 3,
                          "action": ["LLM_REVIEW"] * 3,
                          "final": ["MATCH", "SEM_RESPOSTA_LLM", "UNSURE"]})
    row = endpoint_projection(frame, Costs(10, 50, .005), {"a": LLMError(.2, .1, .1)}, HumanError())
    assert row["FP_aceitacao_esperados"] == pytest.approx(1.2)
    assert row["espec_aceitacao_esperada"] == pytest.approx(.4)
    assert row["espec_aceitacao_inferior"] == 0
    assert row["espec_aceitacao_superior"] == .5
    assert frame.final.iloc[1] == "SEM_RESPOSTA_LLM"


def test_reconciliation_conserves_components():
    from scripts.auditar_funil_adversarial import ROOT
    path = ROOT / "data/varredura_vs_cascata.csv"
    table = pd.read_csv(path)
    assert len(table) == 4
    np.testing.assert_allclose(table.diferenca_pre_lote_pp,
                               table.efeito_reutilizacao_pp + table.efeito_orcamento_pp, atol=1e-12)
    np.testing.assert_allclose(table.cascata_pos_lote - table.cascata_pre_lote,
                               table.efeito_respostas_novas_pp / 100, atol=1e-12)
