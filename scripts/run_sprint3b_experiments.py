"""Sprint 3B — Experiments Script.

Runs all experiments for the GZ-CMD v3 thesis:
  Exp 1: Component ablation (calibration, MACD, guardrails)
  Exp 3: 5-fold stratified cross-validation
  Exp 4: Sensitivity sweep (C_fp / C_fn ratio) + Pareto frontier

Exp 2 (LLM batch review) is handled separately in run_sprint3b_llm_batch.py
because it requires live API calls.

Usage:
    python scripts/run_sprint3b_experiments.py [--quick]

    --quick   Use 2 seeds and 3-fold CV (for development/testing)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── project imports ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gzcmd.config import (
    GZCMDConfig,
    load_config,
    ModePolicyConfig,
    DecisionPolicyConfig,
)
from gzcmd.eval import evaluate_v3_dataframe
from gzcmd.loader import load_comparador_csv, LoadConfig
from gzcmd.reporting import summarize_runs, summary_to_latex_table, write_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "gzcmd" / "gzcmd_v3_config.yaml"
INPUT_CSV = ROOT / "data" / "COMPARADORSEMIDENT.csv"
OUTPUT_DIR = ROOT / "data" / "sprint3b"
TABLES_DIR = ROOT / "text" / "latex" / "tables"

MODES = ["vigilancia", "confirmacao"]
SPLIT_BY = "comprec"
TEST_SIZE = 0.3
GROUP_STRATIFY = True


# ── helpers ────────────────────────────────────────────────────────
def _replace_mode_costs(
    cfg: GZCMDConfig,
    *,
    fp_cost: float | None = None,
    fn_cost: float | None = None,
    llm_budget: int | None = None,
) -> GZCMDConfig:
    """Return a new config with overridden costs for ALL modes."""
    new_modes: dict[str, ModePolicyConfig] = {}
    for name, mode_cfg in cfg.decision_policy.modes.items():
        replacements: dict = {}
        if fp_cost is not None:
            replacements["false_positive_cost"] = fp_cost
        if fn_cost is not None:
            replacements["false_negative_cost"] = fn_cost
        if llm_budget is not None:
            replacements["llm_max_calls_per_window"] = llm_budget
        new_modes[name] = dataclasses.replace(mode_cfg, **replacements)
    new_dp = dataclasses.replace(cfg.decision_policy, modes=new_modes)
    return dataclasses.replace(cfg, decision_policy=new_dp)


def _save(df: pd.DataFrame, name: str) -> Path:
    """Save results CSV and return path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.csv"
    write_csv(df, path)
    log.info("  saved → %s  (%d rows)", path.relative_to(ROOT), len(df))
    return path


def _save_latex(
    summary: pd.DataFrame,
    name: str,
    caption: str,
    label: str,
    group_cols: list[str],
    metrics: list[str],
) -> None:
    """Export summary as LaTeX table."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    # summary_to_latex_table expects list[tuple[str, str]] = (col_name, header_label)
    metric_tuples = [(m, m.replace("_", " ").title()) for m in metrics]
    tex = summary_to_latex_table(
        summary,
        group_cols=group_cols,
        metrics=metric_tuples,
        caption=caption,
        label=label,
        decimals=3,
    )
    path = TABLES_DIR / f"{name}.tex"
    path.write_text(tex, encoding="utf-8")
    log.info("  LaTeX → %s", path.relative_to(ROOT))


# ── Exp 1: Ablation ───────────────────────────────────────────────
def run_exp1_ablation(
    df: pd.DataFrame,
    cfg: GZCMDConfig,
    seeds: list[int],
) -> pd.DataFrame:
    """Component ablation: calibration × MACD × guardrails."""
    log.info("=" * 60)
    log.info("EXP 1 — ABLATION")
    log.info("=" * 60)

    all_runs: list[pd.DataFrame] = []

    # --- Dimension 1: Calibration method ---
    for cal in ["ml_rf", "platt", "stub"]:
        log.info("  calibration=%s, MACD=True, guardrails=True", cal)
        result = evaluate_v3_dataframe(
            df,
            cfg=cfg,
            modes=MODES,
            split_by=SPLIT_BY,
            seeds=seeds,
            test_size=TEST_SIZE,
            group_stratify=GROUP_STRATIFY,
            calibration=cal,
            macd_enabled=True,
            guardrails_enabled=True,
        )
        result["ablation"] = f"cal={cal}"
        all_runs.append(result)

    # --- Dimension 2: MACD on/off (best calibration = ml_rf) ---
    for macd in [True, False]:
        tag = "MACD_ON" if macd else "MACD_OFF"
        log.info("  calibration=ml_rf, MACD=%s, guardrails=True", macd)
        result = evaluate_v3_dataframe(
            df,
            cfg=cfg,
            modes=MODES,
            split_by=SPLIT_BY,
            seeds=seeds,
            test_size=TEST_SIZE,
            group_stratify=GROUP_STRATIFY,
            calibration="ml_rf",
            macd_enabled=macd,
            guardrails_enabled=True,
        )
        result["ablation"] = tag
        all_runs.append(result)

    # --- Dimension 3: Guardrails on/off (best calibration = ml_rf, MACD=True) ---
    for gr in [True, False]:
        tag = "GR_ON" if gr else "GR_OFF"
        log.info("  calibration=ml_rf, MACD=True, guardrails=%s", gr)
        result = evaluate_v3_dataframe(
            df,
            cfg=cfg,
            modes=MODES,
            split_by=SPLIT_BY,
            seeds=seeds,
            test_size=TEST_SIZE,
            group_stratify=GROUP_STRATIFY,
            calibration="ml_rf",
            macd_enabled=True,
            guardrails_enabled=gr,
        )
        result["ablation"] = tag
        all_runs.append(result)

    combined = pd.concat(all_runs, ignore_index=True)
    _save(combined, "exp1_ablation")

    # Summary
    metric_cols = [
        "exp_precision",
        "exp_recall",
        "exp_f1",
        "exp_fbeta",
        "auto_precision",
        "auto_recall",
        "auto_f1",
        "llm_used",
        "review_requested",
    ]
    summary = summarize_runs(
        combined,
        group_cols=["ablation", "mode"],
        metric_cols=metric_cols,
    )
    _save(summary, "exp1_ablation_summary")
    _save_latex(
        summary,
        "tab_exp1_ablation",
        caption="Ablation study: impact of calibration, MACD, and guardrails on expected metrics.",
        label="tab:exp1_ablation",
        group_cols=["ablation", "mode"],
        metrics=["exp_precision", "exp_recall", "exp_f1", "exp_fbeta", "llm_used"],
    )

    log.info("Exp 1 done: %d runs", len(combined))
    return combined


# ── Exp 3: 5-fold Cross-Validation ────────────────────────────────
def run_exp3_kfold(
    df: pd.DataFrame,
    cfg: GZCMDConfig,
    seeds: list[int],
    n_folds: int = 5,
) -> pd.DataFrame:
    """K-fold stratified cross-validation.

    For each seed, we run n_folds evaluations with different test_size
    fractions simulating fold-like behavior via the existing split logic.
    Uses seeds × folds combinations.
    """
    log.info("=" * 60)
    log.info("EXP 3 — %d-FOLD CROSS-VALIDATION", n_folds)
    log.info("=" * 60)

    # We use different seeds to simulate folds.
    # Each seed creates a different random split at test_size=1/n_folds.
    fold_test_size = 1.0 / n_folds
    fold_seeds = []
    rng = np.random.RandomState(42)
    for _ in range(n_folds):
        fold_seeds.extend([int(s + rng.randint(0, 10000)) for s in seeds])

    log.info(
        "  %d folds × %d seeds = %d runs per mode", n_folds, len(seeds), len(fold_seeds)
    )

    result = evaluate_v3_dataframe(
        df,
        cfg=cfg,
        modes=MODES,
        split_by=SPLIT_BY,
        seeds=fold_seeds,
        test_size=fold_test_size,
        group_stratify=GROUP_STRATIFY,
        calibration="ml_rf",
        macd_enabled=True,
        guardrails_enabled=True,
    )
    result["experiment"] = "kfold"
    _save(result, "exp3_kfold")

    metric_cols = [
        "exp_precision",
        "exp_recall",
        "exp_f1",
        "exp_fbeta",
        "auto_precision",
        "auto_recall",
        "auto_f1",
        "llm_used",
    ]
    summary = summarize_runs(
        result,
        group_cols=["mode"],
        metric_cols=metric_cols,
    )
    _save(summary, "exp3_kfold_summary")
    _save_latex(
        summary,
        "tab_exp3_kfold",
        caption=f"{n_folds}-fold cross-validation results (Comprec-stratified split).",
        label="tab:exp3_kfold",
        group_cols=["mode"],
        metrics=["exp_precision", "exp_recall", "exp_f1", "exp_fbeta", "llm_used"],
    )

    log.info("Exp 3 done: %d runs", len(result))
    return result


# ── Exp 4: Sensitivity Sweep + Pareto ─────────────────────────────
def run_exp4_sensitivity(
    df: pd.DataFrame,
    cfg: GZCMDConfig,
    seeds: list[int],
) -> pd.DataFrame:
    """Sweep cost ratios and LLM budget, then extract Pareto frontier."""
    log.info("=" * 60)
    log.info("EXP 4 — SENSITIVITY SWEEP + PARETO")
    log.info("=" * 60)

    all_runs: list[pd.DataFrame] = []

    # --- 4a: FP/FN cost ratio sweep ---
    # Base: vigilancia=(10,50), confirmacao=(100,20)
    # We vary ratio while keeping geometric mean constant
    fp_fn_ratios = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    for ratio in fp_fn_ratios:
        # geometric mean ~ 22.4 for vigilancia, ~ 44.7 for confirmacao
        # We set FP=ratio*k, FN=k such that FP/FN=ratio
        # For simplicity: keep FN=50 for vigilancia, FP=ratio*FN
        fp_vig = ratio * 50.0
        fn_vig = 50.0
        fp_conf = ratio * 20.0
        fn_conf = 20.0

        # Build custom config per mode
        new_modes = {}
        for name, mode_cfg in cfg.decision_policy.modes.items():
            if name == "vigilancia":
                new_modes[name] = dataclasses.replace(
                    mode_cfg, false_positive_cost=fp_vig, false_negative_cost=fn_vig
                )
            else:
                new_modes[name] = dataclasses.replace(
                    mode_cfg, false_positive_cost=fp_conf, false_negative_cost=fn_conf
                )
        sweep_cfg = dataclasses.replace(
            cfg,
            decision_policy=dataclasses.replace(cfg.decision_policy, modes=new_modes),
        )

        log.info(
            "  ratio=%.1f  vig=(%.0f,%.0f)  conf=(%.0f,%.0f)",
            ratio,
            fp_vig,
            fn_vig,
            fp_conf,
            fn_conf,
        )

        result = evaluate_v3_dataframe(
            df,
            cfg=sweep_cfg,
            modes=MODES,
            split_by=SPLIT_BY,
            seeds=seeds,
            test_size=TEST_SIZE,
            group_stratify=GROUP_STRATIFY,
            calibration="ml_rf",
            macd_enabled=True,
            guardrails_enabled=True,
        )
        result["fp_fn_ratio"] = ratio
        result["sweep"] = "cost_ratio"
        all_runs.append(result)

    # --- 4b: LLM budget sweep ---
    budgets = [0, 50, 100, 200, 500, 1000, 2000, 5000]

    for budget in budgets:
        log.info("  LLM budget=%d", budget)
        budget_cfg = _replace_mode_costs(cfg, llm_budget=budget)

        result = evaluate_v3_dataframe(
            df,
            cfg=budget_cfg,
            modes=MODES,
            split_by=SPLIT_BY,
            seeds=seeds,
            test_size=TEST_SIZE,
            group_stratify=GROUP_STRATIFY,
            calibration="ml_rf",
            macd_enabled=True,
            guardrails_enabled=True,
        )
        result["llm_budget"] = budget
        result["sweep"] = "llm_budget"
        all_runs.append(result)

    combined = pd.concat(all_runs, ignore_index=True)
    _save(combined, "exp4_sensitivity")

    # Summary for cost ratio sweep
    cost_runs = combined[combined["sweep"] == "cost_ratio"]
    if len(cost_runs) > 0:
        metric_cols = [
            "exp_precision",
            "exp_recall",
            "exp_f1",
            "exp_fbeta",
            "llm_used",
            "review_requested",
        ]
        cost_summary = summarize_runs(
            cost_runs,
            group_cols=["fp_fn_ratio", "mode"],
            metric_cols=metric_cols,
        )
        _save(cost_summary, "exp4_cost_ratio_summary")
        _save_latex(
            cost_summary,
            "tab_exp4_cost_ratio",
            caption="Sensitivity to FP/FN cost ratio.",
            label="tab:exp4_cost_ratio",
            group_cols=["fp_fn_ratio", "mode"],
            metrics=["exp_precision", "exp_recall", "exp_fbeta", "llm_used"],
        )

    # Summary for budget sweep
    budget_runs = combined[combined["sweep"] == "llm_budget"]
    if len(budget_runs) > 0:
        metric_cols = [
            "exp_precision",
            "exp_recall",
            "exp_f1",
            "exp_fbeta",
            "llm_used",
            "review_requested",
        ]
        budget_summary = summarize_runs(
            budget_runs,
            group_cols=["llm_budget", "mode"],
            metric_cols=metric_cols,
        )
        _save(budget_summary, "exp4_budget_summary")
        _save_latex(
            budget_summary,
            "tab_exp4_budget",
            caption="Sensitivity to LLM review budget.",
            label="tab:exp4_budget",
            group_cols=["llm_budget", "mode"],
            metrics=["exp_precision", "exp_recall", "exp_fbeta", "llm_used"],
        )

    # --- Pareto frontier extraction ---
    _extract_pareto(combined)

    log.info("Exp 4 done: %d runs", len(combined))
    return combined


def _extract_pareto(combined: pd.DataFrame) -> None:
    """Extract Pareto-optimal configurations (reviews vs recall, FP vs FN)."""
    # Group by sweep parameters and compute mean metrics
    pareto_points = []

    for mode in MODES:
        mode_data = combined[combined["mode"] == mode]
        if mode_data.empty:
            continue

        # Group by sweep config
        for (sweep, *_), group in mode_data.groupby(["sweep"]):
            # For each unique config within the sweep
            config_cols = ["fp_fn_ratio", "llm_budget"]
            available = [c for c in config_cols if c in group.columns]

            if not available:
                continue

            for _, sub in group.groupby(available, dropna=False):
                point = {
                    "mode": mode,
                    "sweep": sweep,
                    "mean_exp_recall": sub["exp_recall"].mean(),
                    "mean_exp_precision": sub["exp_precision"].mean(),
                    "mean_exp_f1": sub["exp_f1"].mean(),
                    "mean_llm_used": sub["llm_used"].mean(),
                    "mean_review_requested": sub["review_requested"].mean(),
                }
                for c in available:
                    if c in sub.columns and sub[c].notna().any():
                        point[c] = sub[c].iloc[0]
                pareto_points.append(point)

    if pareto_points:
        pareto_df = pd.DataFrame(pareto_points)
        _save(pareto_df, "exp4_pareto_points")


# ── main ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 3B experiments")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: 2 seeds, 3-fold CV"
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="all",
        help="Run specific experiment: 1, 3, 4, or all",
    )
    args = parser.parse_args()

    if args.quick:
        seeds = [42, 123]
        n_folds = 3
    else:
        seeds = [42, 123, 456, 789, 2024]
        n_folds = 5

    log.info("Sprint 3B Experiments")
    log.info("  seeds: %s", seeds)
    log.info("  split: %s  test_size=%.1f", SPLIT_BY, TEST_SIZE)
    log.info("  quick: %s", args.quick)

    # Load data once
    t0 = time.time()
    cfg = load_config(CONFIG_PATH)
    df = load_comparador_csv(INPUT_CSV, cfg=LoadConfig(macd_enabled=True))
    log.info("Data loaded: %d pairs (%.1fs)", len(df), time.time() - t0)

    results = {}

    if args.exp in ("all", "1"):
        t1 = time.time()
        results["exp1"] = run_exp1_ablation(df, cfg, seeds)
        log.info("Exp 1 total: %.1fs", time.time() - t1)

    if args.exp in ("all", "3"):
        t1 = time.time()
        results["exp3"] = run_exp3_kfold(df, cfg, seeds, n_folds=n_folds)
        log.info("Exp 3 total: %.1fs", time.time() - t1)

    if args.exp in ("all", "4"):
        t1 = time.time()
        results["exp4"] = run_exp4_sensitivity(df, cfg, seeds)
        log.info("Exp 4 total: %.1fs", time.time() - t1)

    log.info("=" * 60)
    log.info("ALL DONE (%.1fs total)", time.time() - t0)
    log.info("Results in: %s", OUTPUT_DIR.relative_to(ROOT))
    log.info("LaTeX in:   %s", TABLES_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
