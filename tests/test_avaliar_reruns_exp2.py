"""Contratos agregados da avaliação dos reruns, sem dados identificáveis."""

import pandas as pd
import pytest
import yaml

from scripts import avaliar_reruns_exp2 as evaluation


def test_checkpoint_is_partial_and_missing_is_not_unsure():
    run = {"results": [{"pair_id": "a", "decision": "MATCH"}],
           "n_reviewed": 1, "n_review_pairs": 2, "complete": False}
    frame, info = evaluation.parse_run(run, {"a", "b"})
    assert info == {"n_respondido": 1, "n_esperado": 2, "n_sem_resposta": 1, "completo": False}
    assert frame.decision.tolist() == ["MATCH"]
    run["results"].append({"pair_id": "b", "decision": "UNSURE"})
    run["n_reviewed"] = 2
    assert evaluation.parse_run(run, {"a", "b"})[1]["completo"] is False
    run["complete"] = True
    assert evaluation.parse_run(run, {"a", "b"})[1]["completo"] is True


def test_run_rejects_duplicate_or_foreign_pairs():
    with pytest.raises(ValueError, match="duplicado"):
        evaluation.parse_run({"results": [{"pair_id": "a", "decision": "MATCH"}] * 2}, {"a"})
    with pytest.raises(ValueError, match="fora da coorte"):
        evaluation.parse_run({"results": [{"pair_id": "x", "decision": "MATCH"}]}, {"a"})


def test_mcnemar_exact_and_strict_accuracy():
    assert evaluation.mcnemar_exact(0, 6) == pytest.approx(0.03125)
    assert evaluation.mcnemar_exact(0, 0) == 1
    a = pd.DataFrame({"pair_id": list("abcdefgh"), "y": [1] * 7 + [0],
                      "decision": ["NONMATCH"] * 7 + ["NONMATCH"]})
    b = a.copy()
    b["decision"] = ["MATCH"] * 6 + ["UNSURE", "MATCH"]
    result = evaluation.paired_metrics(a, b)
    assert result["FN_revertidos"] == 6
    assert result["FN_para_UNSURE"] == 1
    assert result["FP_novos"] == 1
    assert result["UNSURE_novos"] == 1
    assert result["acerto_a_erro_b"] == 1
    assert result["erro_a_acerto_b"] == 6
    assert result["mcnemar_p_exato"] == pytest.approx(0.125)
    assert result["kappa"] == pytest.approx(0)
    assert evaluation.paired_metrics(a, b.iloc[:6])["n_so_a"] == 2


def test_unsure_policies_and_zero_denominator():
    part = pd.DataFrame({"y": [1, 1, 1, 0, 0, 0],
                         "decision": ["MATCH", "NONMATCH", "UNSURE"] * 2})
    result = evaluation.metrics(part)
    assert result["sens"] == 0.5
    assert result["espec"] == 0.5
    assert result["sens_unsure_nonmatch"] == pytest.approx(1 / 3)
    assert result["espec_unsure_nonmatch"] == pytest.approx(2 / 3)
    assert result["sens_incl_unsure"] == pytest.approx(2 / 3)
    assert result["fracao_unsure"] == pytest.approx(1 / 3)
    assert result["sens_ci95_low"] < result["sens"] < result["sens_ci95_high"]
    assert evaluation.metrics(part.iloc[:0])["sens"] is None


def test_v4_yaml_format_pools_only_v4_and_preserves_fallback():
    definitions = {"test": {"min": 0, "max": 10}, "empty": {"min": 10, "max": 20}}
    fixed = {"test": {"e_fp": 0.1, "e_fn": 0.2}, "empty": {"e_fp": 0.3, "e_fn": 0.4}}
    data = pd.DataFrame([
        {"mode": mode, "model": model, "dossie": "sem_mo", "prompt": prompt,
         "band": "test", "y": y, "decision": decision}
        for mode in ("vigilancia", "confirmacao")
        for model in ("claude-haiku-5-5", "deepseek-v4.1-flash")
        for prompt, y, decision in (("v4", 1, "NONMATCH"), ("v4", 0, "NONMATCH"), ("v3", 0, "MATCH"))
    ])
    sources = [{"mode": mode, "model": model, "dossie": "sem_mo", "prompt": "v4", "completo": False}
               for mode in ("vigilancia", "confirmacao")
               for model in ("claude-haiku-5-5", "deepseek-v4.1-flash")]
    result = yaml.safe_load(yaml.safe_dump(evaluation.make_v4_yaml(data, sources, definitions, fixed, b"config")))
    assert result["schema_version"] == 1
    assert result["provisional"] is True
    assert result["pooled_excludes_partial"] is False
    assert result["llm_review"]["reliability"]["error_rates_by_band"]["test"] == {"e_fp": 0, "e_fn": 1}
    assert result["pooled"]["fallback_fixed"]["empty"] == ["e_fp", "e_fn"]
    assert result["pooled"]["error_rates_by_band"]["empty"] == fixed["empty"]
    assert result["by_mode"]["vigilancia"]["pooled"]["partial"] is True
    assert set(result["models"]) == {"claude-haiku-5-5", "deepseek-v4.1-flash"}


def test_latex_only_complete_filters_partial_rows():
    rows = [{"mode": "vigilancia", "model": "claude-haiku-5-5", "dossie": "sem_mo", "prompt": "v4",
             "completo": False, "n_respondido": 1, "n_esperado": 2,
             "VP": 1, "FP": 0, "FN": 0, "UNSURE": 0, "sens": 1, "espec": None, "fracao_unsure": 0}]
    text = evaluation.make_latex(rows)
    assert r"\textsuperscript{*}" in text
    assert "1/2" in text
    assert "100,0" in text
    assert "Claude Haiku 5.5" not in evaluation.make_latex(rows, only_complete=True)
