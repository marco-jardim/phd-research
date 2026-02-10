from __future__ import annotations

import pandas as pd

from gzcmd.reporting import summarize_runs, summary_to_latex_table


def test_reporting_summary_and_latex_table() -> None:
    runs = pd.DataFrame(
        {
            "Config": ["a", "a", "b"],
            "exp_fbeta": [0.9, 1.0, 0.5],
            "auto_coverage": [0.8, 0.9, 1.0],
        }
    )

    summary = summarize_runs(
        runs, group_cols=["Config"], metric_cols=["exp_fbeta", "auto_coverage"]
    )
    assert "exp_fbeta_mean" in summary.columns
    assert "exp_fbeta_std" in summary.columns

    latex = summary_to_latex_table(
        summary,
        group_cols=["Config"],
        metrics=[("exp_fbeta", "Fbeta"), ("auto_coverage", "Coverage")],
        caption="Caption",
        label="tab:test",
        decimals=3,
    )
    assert "\\toprule" in latex
    assert "\\pm" in latex
    assert "\\label{tab:test}" in latex
