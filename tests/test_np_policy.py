from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
import yaml

from gzcmd.config import load_config
from gzcmd.guardrails import high_guard_birth_variant, strong_birth_conflict
from gzcmd.np_policy import NPConstraints, NPInfeasibleError, select_match_prefix
from gzcmd.policy_engine import Budget, Costs, HumanBudget, LLMError, PolicyEngineV3
from gzcmd.runner import build_engine_from_config


def test_np_greedy_prefix_is_optimal_against_all_subsets() -> None:
    p = pd.Series([.9, 1., .5, .995, .97], index=["a", "b", "c", "d", "e"])
    fixed = pd.Series([.99, 1.])
    constraint = NPConstraints(max_fp_rate_auto=.02)
    chosen, info = select_match_prefix(p, fixed, constraint)
    assert chosen.tolist() == ["b", "d", "e"]
    assert info["np_fp_esperado_auto"] <= info["np_fp_limite"] + 1e-12
    feasible_rewards = []
    for size in range(len(p) + 1):
        for subset in combinations(p.index, size):
            values = p.loc[list(subset)]
            fp = float((1 - fixed).sum() + (1 - values).sum())
            if fp <= .02 * (len(fixed) + len(values)) + 1e-12:
                feasible_rewards.append(float(values.sum()))
    assert p.loc[chosen].sum() == pytest.approx(max(feasible_rewards))


def test_np_absolute_fp_budget_overrides_rate_and_includes_guards() -> None:
    p = pd.Series([.99, .98, .96, .6])
    selected, info = select_match_prefix(p, pd.Series([.99]), NPConstraints(.00001, .045))
    assert selected.tolist() == [0, 1]
    assert info["np_fp_esperado_auto"] == pytest.approx(.04)
    assert info["np_fp_limite"] == .045


def test_np_initial_guard_rate_can_be_repaired_by_zero_risk_prefix() -> None:
    selected, info = select_match_prefix(pd.Series([1., 1., .97]), pd.Series([.99]), NPConstraints(.005))
    assert selected.tolist() == [0, 1]
    assert info["np_taxa_fp_auto"] <= .005


def test_np_impossible_guards_raise_without_consuming_budgets() -> None:
    frame = pd.DataFrame({"nota final": [9., 6.], "p_cal": [.9, 1.], "band": ["a"] * 2,
                          "guardrail": ["ALWAYS_MATCH", pd.NA]})
    budget, human = Budget(10), HumanBudget()
    engine = PolicyEngineV3(Costs(100, 20, .005), {"a": LLMError(0, 0)}, budget,
                           policy_version="v3.1", policy_kind="np", human_budget=human,
                           np_constraints=NPConstraints(.001))
    with pytest.raises(NPInfeasibleError) as error:
        engine.triage(frame)
    assert error.value.diagnostics["motivo"] == "guardas_fp"
    assert error.value.diagnostics["taxa_minima_atingivel"] == pytest.approx(.05)
    assert budget.llm_used == human.used == 0


def test_np_review_gain_and_nonmatch_threshold() -> None:
    frame = pd.DataFrame({"nota final": [6.] * 4, "p_cal": [1., .99, .1, .001], "band": ["a"] * 4})
    engine = PolicyEngineV3(Costs(100, 20, .005), {"a": LLMError(0, .05)}, Budget(2),
                           policy_version="v3.1", policy_kind="np", human_budget=HumanBudget(0))
    out = engine.triage(frame)
    assert out.action.tolist() == ["MATCH", "LLM_REVIEW", "LLM_REVIEW", "NONMATCH"]
    assert out.np_fp_esperado_auto.iloc[0] <= out.np_fp_limite.iloc[0] + 1e-12
    assert engine.budget.llm_used == 2


def test_np_human_capacity_does_not_enable_unsafe_auto_fallback() -> None:
    frame = pd.DataFrame({"nota final": [6.], "p_cal": [.5], "band": ["a"]})
    engine = PolicyEngineV3(Costs(100, 20, .005), {"a": LLMError(1, 1)}, Budget(2),
                           policy_version="v3.1", policy_kind="np", human_budget=HumanBudget(0))
    with pytest.raises(NPInfeasibleError) as error:
        engine.triage(frame)
    assert error.value.diagnostics["motivo"] == "orcamento_humano"
    assert engine.budget.llm_used == engine.human_budget.used == 0


def test_birth_variant_is_stricter_subset_and_never_recovers_rejected_p() -> None:
    frame = pd.DataFrame({"nota final": [9.] * 5, "p_cal": [.99, .96, .98, .94, .89],
                          "diff_ano": [0., 2., np.nan, 0., 0.]})
    assert strong_birth_conflict(frame).tolist() == [False, True, False, False, False]
    variant = high_guard_birth_variant(frame)
    assert variant.tolist() == [True, False, True, False, False]
    assert not (variant & frame.p_cal.lt(.9)).any()


def test_np_config_default_and_selectable_and_rates_override(tmp_path) -> None:
    cfg = load_config("gzcmd/gzcmd_v3_config.yaml")
    c = cfg.decision_policy.modes["confirmacao"]
    assert c.policy_kind == "cost" and c.np_policy.max_fp_rate_auto == .002
    modes = dict(cfg.decision_policy.modes)
    modes["confirmacao"] = replace(c, policy_kind="np")
    cfg = replace(cfg, decision_policy=replace(cfg.decision_policy, modes=modes))
    fixed_block = {"estimates_by_band": {"high": {"n": 10, "u": .2}},
                   "error_rates_by_band": {"high": {"e_fp": .3, "e_fn": .4}}}
    source = tmp_path / "rates.yaml"
    source.write_text(yaml.safe_dump({"by_mode": {"confirmacao": {"pooled": fixed_block}}}), encoding="utf-8")
    engine = build_engine_from_config(cfg, mode="confirmacao", rates_yaml=source)
    assert engine.policy_kind == "np" and engine.llm_error_by_band["high"].e_fp == .3
    assert build_engine_from_config(cfg, mode="confirmacao", policy_kind="cost").policy_kind == "cost"
    assert build_engine_from_config(cfg, mode="confirmacao", policy_version="v3").policy_kind == "cost"


def test_infeasible_comparison_has_missing_metrics_not_fabricated_zeroes() -> None:
    from gzcmd.np_evaluation import compare_confirmation
    cfg = load_config("gzcmd/gzcmd_v3_config.yaml")
    frame = pd.DataFrame({"nota final": [9., 6.], "p_cal": [.9, 1.], "band": ["high", "grey_mid"],
                          "TARGET": [1, 1], "guardrail": ["ALWAYS_MATCH", pd.NA]})
    table = compare_confirmation(frame, cfg, cfg.llm_review.estimated_rates_path)
    constrained = table[table.policy_kind.eq("np")]
    assert not constrained.viavel.any()
    assert constrained.AUTO_MATCH.isna().all()
    assert constrained.fp_esperado_total.isna().all()
    assert constrained.motivo.eq("guardas_fp").all()


@pytest.mark.parametrize("constraints", [(-.1, None, .02), (.1, -1, .02), (.1, None, 2)])
def test_invalid_np_constraints(constraints) -> None:
    with pytest.raises(ValueError):
        NPConstraints(*constraints)
