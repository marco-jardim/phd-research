from __future__ import annotations

import json

import pandas as pd
import pytest

from gzcmd.final_funnel import acceptance_metrics, cascade_cells
from gzcmd.llm_usage_cost import price_usage, selected_cost
from gzcmd.policy_engine import Costs, HumanError, LLMError
from scripts import funil_gzcmd_v2 as funnel
from scripts.fig_cascata_v31 import terminal_counts


def run_record(mode, pid, decision, cohort=None):
    return {"mode": mode, "model": "claude-haiku-5-5", "prompt_version": "v4", "complete": True,
            "prompt_sha256": "same-prompt", "cohort": cohort, "selected_pair_ids": [pid],
            "removed_fields": ["model_outputs." + key for key in ("p_cal", "band", "base_choice", "evr", "guardrail", "guardrail_reason")],
            "results": [{"pair_id": pid, "decision": decision}]}


def test_completed_cohort_is_merged_and_not_filtered_by_old_batch(tmp_path, monkeypatch):
    old, extra = tmp_path / "old.json", tmp_path / "extra.json"
    old.write_text(json.dumps(run_record("vigilancia", "a", "NONMATCH")), encoding="utf-8")
    extra.write_text(json.dumps(run_record("vigilancia", "b", "MATCH", "v31_lot")), encoding="utf-8")
    monkeypatch.setattr(funnel, "artifact_paths", lambda mode, model, include_v31=False: [old, extra] if include_v31 else [old])
    historic, _ = funnel.load_responses("vigilancia", funnel.HAIKU_V4)
    merged, _ = funnel.load_responses("vigilancia", funnel.HAIKU_V4, include_v31=True)
    assert historic == {"a": "NONMATCH"} and merged["b"] == "MATCH"
    frame = pd.DataFrame({"y": [1], "p_cal": [.8], "band": ["a"], "action": ["LLM_REVIEW"]}, index=["b"])
    observed, metrics = funnel.metrics_with_projection(frame, merged, historic, {"a"}, Costs(10, 50, .005),
                                                       {"a": LLMError(.1, .2, .1)}, HumanError())
    assert observed.final.iloc[0] == "MATCH"
    assert metrics["sem_resposta_LLM"] == 0 and metrics["cobertura_llm_completa"]
    assert metrics["sens_observada"] == 1
    assert metrics["sens_esperada_pre_lote31"] == pytest.approx(.7)


@pytest.mark.parametrize("problem", ["cohort", "duplicate", "prompt"])
def test_invalid_supplement_cannot_silently_replace_results(tmp_path, monkeypatch, problem):
    old, extra = tmp_path / "old.json", tmp_path / "extra.json"
    first = run_record("vigilancia", "a", "NONMATCH")
    second = run_record("vigilancia", "b", "MATCH", "v31_lot")
    if problem == "cohort":
        second["cohort"] = "another-cohort"
    elif problem == "duplicate":
        second["results"][0]["pair_id"] = "a"
        second["selected_pair_ids"] = ["a"]
    else:
        second["prompt_sha256"] = "different-prompt"
    old.write_text(json.dumps(first), encoding="utf-8")
    extra.write_text(json.dumps(second), encoding="utf-8")
    monkeypatch.setattr(funnel, "artifact_paths", lambda *args, **kwargs: [old, extra])
    with pytest.raises(ValueError):
        funnel.load_responses("vigilancia", funnel.HAIKU_V4, include_v31=True)


def test_observed_acceptance_matrix_keeps_human_outcomes_separate():
    frame = pd.DataFrame({"y": [1, 0, 1, 0, 1, 0],
                          "final": ["MATCH", "MATCH", "NONMATCH", "NONMATCH", "UNSURE", "DEFERRED"],
                          "action": ["LLM_REVIEW"] * 5 + ["DEFERRED"]})
    m = acceptance_metrics(frame)
    assert [m[k] for k in ["VP_cadeia", "FP_cadeia", "FN_cadeia", "VN_cadeia"]] == [1, 1, 2, 2]
    assert m["FN_nonmatch_observado"] == m["VN_nonmatch_observado"] == 1
    assert m["pares_ao_humano"] == 2 and m["vinculos_ao_humano"] == 1
    assert m["sens_cadeia_observada"] == pytest.approx(1 / 3)
    assert m["espec_cadeia_observada"] == pytest.approx(2 / 3)
    frame.loc[0, "final"] = "SEM_RESPOSTA_LLM"
    with pytest.raises(ValueError, match="complete LLM"):
        acceptance_metrics(frame)


def test_cascade_partitions_and_human_rollup_do_not_double_count():
    frame = pd.DataFrame({"faixa": ["<5", "[5,9)", "[5,9)", ">=9", "[5,9)"],
                          "y": [0, 1, 1, 0, 0],
                          "decisao_motor": ["ALWAYS_NONMATCH", "AUTO_MATCH", "LLM", "LLM", "DEFERRED"],
                          "action": ["NONMATCH", "MATCH", "LLM_REVIEW", "LLM_REVIEW", "DEFERRED"],
                          "final": ["NONMATCH", "MATCH", "UNSURE", "NONMATCH", "DEFERRED"],
                          "llm_decision": [None, None, "UNSURE", "NONMATCH", None]})
    cells = pd.DataFrame(cascade_cells(frame, "vigilancia", "v3.1"))
    assert cells.loc[cells.nivel.eq(2), "n"].sum() == 5
    assert cells.loc[cells.nivel.eq(3), "n"].sum() == 2
    assert cells.loc[cells.nivel.eq(4), "n"].sum() == 2
    terminals = terminal_counts(cells, "vigilancia", "v3.1")
    assert terminals.n.sum() == 5 and terminals.vinculos.sum() == 2


def test_per_pair_recorded_token_cost_and_no_double_charging(tmp_path):
    usage = {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 50,
             "cache_creation_input_tokens": 0, "max_request_input_tokens": 2000}
    assert price_usage(usage) == pytest.approx(.0002005)
    pid_a, pid_b = "a" * 16, "b" * 16
    paths = []
    for name, pid in [("old", pid_a), ("extra", pid_b)]:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"results": [{"pair_id": pid}], "cost_estimate_usd": .0002005}), encoding="utf-8")
        journal = tmp_path / f"{name}_journal"
        journal.mkdir()
        (journal / f"{pid}.transport.json").write_text(json.dumps(usage), encoding="utf-8")
        paths.append(path)
    costs, sources = selected_cost(paths, {pid_a, pid_b})
    assert sum(costs.values()) == pytest.approx(.000401)
    assert len(sources) == 2
    with pytest.raises(ValueError, match="charged twice"):
        selected_cost([paths[0], paths[0]], {pid_a})
