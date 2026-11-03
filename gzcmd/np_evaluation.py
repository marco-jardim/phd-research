"""Safe pooled comparison tables, including explicit infeasible NP scenarios."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .cost_evaluation import expected_funnel_metrics
from .guardrails import high_guard_birth_variant
from .np_policy import NPConstraints, NPInfeasibleError
from .policy_engine import Budget, Costs, HumanBudget, HumanError, PolicyEngineV3
from .reliability import load_band_rates

NP_RATES = (.0005, .001, .002, .005, .01)


def evaluate_np_point(frame, cfg, rates, *, alpha, error=None, cap="config", kind="np"):
    c = cfg.decision_policy.modes["confirmacao"]
    human = HumanError(c.human_e_fp, c.human_e_fn) if error is None else HumanError(error, error)
    budget = c.human_max_pairs_per_window if cap == "config" else cap
    constraints = replace(c.np_policy, max_fp_rate_auto=alpha)
    row = {"modo": "confirmacao", "policy_kind": kind, "max_fp_rate_auto": alpha if kind == "np" else np.nan,
           "max_fp_expected_total": constraints.max_fp_expected_total,
           "max_fn_rate_auto": constraints.max_fn_rate_auto, "C_llm": c.llm_review_cost,
           "e_fp_h": human.e_fp, "e_fn_h": human.e_fn,
           "orcamento_humano": np.inf if budget is None else budget}
    engine = PolicyEngineV3(Costs(c.false_positive_cost, c.false_negative_cost, c.llm_review_cost, c.human_review_cost),
                           rates, Budget(c.llm_max_calls_per_window), policy_version="v3.1", policy_kind=kind,
                           human_error=human, human_budget=HumanBudget(budget), np_constraints=constraints)
    try:
        out = engine.triage(frame)
    except NPInfeasibleError as exc:
        row.update({"viavel": False, "status": "inviavel", **exc.diagnostics})
        return row, None
    row.update({"viavel": True, "status": "viavel" if kind == "np" else "custo_sem_restricao",
                **expected_funnel_metrics(out, rates, human), **engine.np_diagnostics})
    return row, out


def compare_confirmation(frame, cfg, rates_path):
    rates = load_band_rates(cfg.llm_review.error_rates_by_band, mode="confirmacao", source=rates_path)
    rows = []
    for kind, alpha in (("cost", cfg.decision_policy.modes["confirmacao"].np_policy.max_fp_rate_auto), ("np", .001), ("np", .002), ("np", .01)):
        row, _ = evaluate_np_point(frame, cfg, rates, alpha=alpha, kind=kind)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_birth_variant(frame):
    score = frame["nota final"]
    gold = frame.TARGET.eq(1)
    current = score.ge(9) & frame.p_cal.ge(.9)
    variant = high_guard_birth_variant(frame)
    rejected = score.ge(9) & ~current
    return pd.DataFrame([{
        "pares_ge9": int(score.ge(9).sum()), "fora_guarda_atual": int(rejected.sum()),
        "guarda_atual_n": int(current.sum()), "guarda_atual_FP": int((current & ~gold).sum()),
        "variante_n": int(variant.sum()), "variante_vinculos": int((variant & gold).sum()),
        "variante_FP": int((variant & ~gold).sum()), "recuperados_dos_17": int((variant & rejected).sum()),
        "FP_recuperados": int((variant & rejected & ~gold).sum()),
        "variante_fp_esperado": float((1 - frame.p_cal[variant]).sum()),
        "criterio_zero_fp_atendido": not (variant & ~gold).any(),
        "adotada": False,
    }])
