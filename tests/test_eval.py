from __future__ import annotations

import pandas as pd

from gzcmd.config import (
    BandsConfig,
    BandDefinition,
    CalibrationConfig,
    DecisionPolicyConfig,
    EvaluationDefaults,
    GuardrailsParams,
    GZCMDConfig,
    LLMReviewConfig,
    ModePolicyConfig,
)
from gzcmd.eval import evaluate_v3_dataframe


def test_evaluate_v3_dataframe_runs_and_returns_metrics() -> None:
    cfg = GZCMDConfig(
        version="3",
        bands=BandsConfig(
            definitions=(
                BandDefinition(name="low", min=0.0, max=5.0, inclusive_max=False),
                BandDefinition(name="high", min=5.0, max=999.0, inclusive_max=True),
            )
        ),
        decision_policy=DecisionPolicyConfig(
            modes={
                "vigilancia": ModePolicyConfig(
                    false_positive_cost=10.0,
                    false_negative_cost=50.0,
                    llm_review_cost=0.25,
                    min_auto_match_threshold=0.85,
                    max_auto_nonmatch_threshold=0.15,
                    llm_max_calls_per_window=2,
                )
            }
        ),
        llm_review=LLMReviewConfig(
            enabled=True,
            error_rates_by_band={
                "low": {"e_fp": 0.1, "e_fn": 0.1},
                "high": {"e_fp": 0.1, "e_fn": 0.1},
            },
        ),
        guardrails=GuardrailsParams(),
        calibration=CalibrationConfig(),
        evaluation=EvaluationDefaults(),
    )

    # Build a small dataset with both classes and reasonable spread of scores.
    df = pd.DataFrame(
        {
            "nota final": [
                9.0,
                8.5,
                8.0,
                7.5,
                7.0,
                6.5,
                6.0,
                5.5,
                4.5,
                4.0,
                3.5,
                3.0,
                2.5,
                2.0,
                1.5,
                1.0,
                0.5,
                5.2,
                5.8,
                6.2,
            ],
            "TARGET": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            "COMPREC": [f"C{i // 2}" for i in range(20)],
            "REFREC": [f"R{i // 2}" for i in range(20)],
        }
    )

    out = evaluate_v3_dataframe(
        df,
        cfg=cfg,
        modes=["vigilancia"],
        split_by="row",
        seeds=[42],
        test_size=0.3,
        group_stratify=True,
        calibration="platt",
        macd_enabled=False,
    )

    assert len(out) == 1
    row = out.iloc[0]
    assert "exp_fbeta" in out.columns
    assert row["n_test"] > 0
    assert 0.0 <= float(row["auto_coverage"]) <= 1.0
