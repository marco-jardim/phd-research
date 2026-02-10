from __future__ import annotations

import pandas as pd

from gzcmd.policy_engine import Budget, Costs, LLMError, PolicyEngineV3


def test_policy_engine_respects_budget_and_guardrails() -> None:
    engine = PolicyEngineV3(
        costs=Costs(false_positive=10.0, false_negative=50.0, llm_review=0.25),
        llm_error_by_band={
            "low": LLMError(e_fp=0.1, e_fn=0.1),
            "grey_mid": LLMError(e_fp=0.1, e_fn=0.1),
            "high": LLMError(e_fp=0.1, e_fn=0.1),
        },
        budget=Budget(llm_max=1, llm_used=0),
        min_auto_match=0.85,
        max_auto_nonmatch=0.15,
    )

    df = pd.DataFrame(
        {
            "p_cal": [0.99, 0.01, 0.50, 0.50],
            "band": ["high", "low", "grey_mid", "grey_mid"],
            "guardrail": [pd.NA, pd.NA, "FORCE_REVIEW", "ALWAYS_NONMATCH"],
        }
    )

    out = engine.triage(df)
    # FORCE_REVIEW should consume the only budgeted LLM call.
    assert out.loc[2, "action"] == "LLM_REVIEW"
    assert int(out["action"].eq("LLM_REVIEW").sum()) == 1
    # ALWAYS_NONMATCH should be deterministic and never selected.
    assert out.loc[3, "action"] == "NONMATCH"
    assert engine.budget.llm_used == 1


def test_policy_engine_review_priority_force_then_caps_then_evr() -> None:
    engine = PolicyEngineV3(
        costs=Costs(false_positive=10.0, false_negative=50.0, llm_review=0.25),
        llm_error_by_band={
            "grey_mid": LLMError(e_fp=0.01, e_fn=0.01),
            "high": LLMError(e_fp=0.01, e_fn=0.01),
        },
        budget=Budget(llm_max=2, llm_used=0),
        min_auto_match=0.85,
        max_auto_nonmatch=0.15,
    )

    df = pd.DataFrame(
        {
            # Row 0: FORCE_REVIEW (should always be selected first)
            "p_cal": [0.99, 0.30, 0.50],
            "band": ["high", "grey_mid", "grey_mid"],
            "guardrail": ["FORCE_REVIEW", pd.NA, pd.NA],
        }
    )

    out = engine.triage(df)
    # Budget is 2: expect FORCE_REVIEW and the cap-triggered row (p<0.85) to be selected.
    assert out.loc[0, "action"] == "LLM_REVIEW"
    assert out.loc[1, "action"] == "LLM_REVIEW"
    assert out.loc[2, "action"] != "LLM_REVIEW"
