#!/usr/bin/env python
"""
C2 Pilot: 517-pair stratified multi-model evaluation
=====================================================
Runs 4 LLM models on 517 stratified grey-zone pairs.
Models: Kimi K2.5, Qwen3-235B, DeepSeek R1, GPT-4o

Usage:
    python scripts/pilot_c2_517.py                  # run all models
    python scripts/pilot_c2_517.py --model kimi_k2.5 # run single model
    python scripts/pilot_c2_517.py --sample-only      # just create sample, no LLM calls
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gzcmd.llm_dossier import build_dossier, make_pair_id
from gzcmd.llm_review import LLMCallError, LLMReviewer
from gzcmd.runner import run_v3

load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pilot_c2")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_SAMPLE = 517
SEED = 42
N_WORKERS = 10
MODE = "vigilancia"
CONFIG_PATH = ROOT / "gzcmd" / "gzcmd_v3_config.yaml"
INPUT_CSV = ROOT / "data" / "COMPARADORSEMIDENT.csv"
OUTPUT_DIR = ROOT / "data" / "pilot_c2"

MODELS: dict[str, dict] = {
    "kimi_k2.5": {
        "api_key_env": "FIREWORKS_API_KEY",
        "base_url": "https://api.fireworks.ai/inference/v1/chat/completions",
        "model": "accounts/fireworks/models/kimi-k2p5",
    },
    "qwen3_vl_30b": {
        "api_key_env": "FIREWORKS_API_KEY",
        "base_url": "https://api.fireworks.ai/inference/v1/chat/completions",
        "model": "accounts/fireworks/models/qwen3-vl-30b-a3b-instruct",
    },
    "deepseek_v3.2": {
        "api_key_env": "FIREWORKS_API_KEY",
        "base_url": "https://api.fireworks.ai/inference/v1/chat/completions",
        "model": "accounts/fireworks/models/deepseek-v3p2",
    },
    "gpt4o": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o",
    },
    "glm5": {
        "api_key_env": "FIREWORKS_API_KEY",
        "base_url": "https://api.fireworks.ai/inference/v1/chat/completions",
        "model": "accounts/fireworks/models/glm-5",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z**2 / n
    centre = p_hat + z**2 / (2 * n)
    spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    lo = (centre - spread) / denom
    hi = (centre + spread) / denom
    return (max(0.0, lo), min(1.0, hi))


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Proportional stratified sample by band from LLM_REVIEW rows."""
    review_df = df[df["action"] == "LLM_REVIEW"].copy()
    band_counts = review_df["band"].value_counts()
    proportions = band_counts / band_counts.sum()
    allocations = (proportions * n).round().astype(int)

    # Adjust rounding residual
    diff = n - allocations.sum()
    if diff != 0:
        allocations[allocations.idxmax()] += diff

    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for band, alloc in allocations.items():
        band_df = review_df[review_df["band"] == band]
        if len(band_df) <= alloc:
            parts.append(band_df)
        else:
            idx = rng.choice(len(band_df), size=alloc, replace=False)
            parts.append(band_df.iloc[idx])

    return pd.concat(parts, ignore_index=True)


def result_to_dict(r) -> dict:
    """Normalize ReviewResult or error into a serializable dict."""
    if isinstance(r, dict):
        return r
    if isinstance(r, LLMCallError):
        return {"pair_id": getattr(r, "pair_id", "?"), "error": str(r)}
    # ReviewResult
    return {
        "pair_id": r.pair_id,
        "decision": r.decision if isinstance(r.decision, str) else str(r.decision),
        "confidence": r.confidence,
        "reason_codes": list(r.reason_codes) if r.reason_codes else [],
        "protocol": r.protocol,
        "total_latency_s": r.total_latency_s,
    }


# ---------------------------------------------------------------------------
# Model runner (concurrent)
# ---------------------------------------------------------------------------


def run_model(
    model_name: str,
    model_cfg: dict,
    dossiers: list,
    output_dir: Path,
    n_workers: int = N_WORKERS,
) -> list[dict] | None:
    """Run one model on all dossiers with ThreadPool concurrency + checkpointing."""
    api_key = os.environ.get(model_cfg["api_key_env"], "")
    if not api_key:
        log.warning(
            "[SKIP] %s: missing env var %s", model_name, model_cfg["api_key_env"]
        )
        return None

    reviewer = LLMReviewer(
        api_key=api_key,
        base_url=model_cfg["base_url"],
        model=model_cfg["model"],
        fallback_model=model_cfg.get("fallback_model", ""),
        temperature=0.0,
        max_tokens=4096,
        top_p=1.0,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )

    # Resume from checkpoint
    ckpt_path = output_dir / f"{model_name}_checkpoint.json"
    results: list[dict] = []
    done_ids: set[str] = set()
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            results = json.load(f)
        done_ids = {r["pair_id"] for r in results if "pair_id" in r}
        log.info("  Resuming %s: %d already done", model_name, len(done_ids))

    remaining = [d for d in dossiers if d.pair_id not in done_ids]
    if not remaining:
        log.info("  %s: all %d already done", model_name, len(dossiers))
        return results

    log.info(
        "  Running %s: %d dossiers × %d workers...",
        model_name,
        len(remaining),
        n_workers,
    )
    errors = 0
    t0 = time.time()

    def _review(dossier):
        try:
            return reviewer.review(dossier)
        except Exception as exc:
            return {"pair_id": dossier.pair_id, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_review, d): d for d in remaining}
        for i, future in enumerate(as_completed(futures), 1):
            raw = future.result()
            d = result_to_dict(raw)
            results.append(d)
            if "error" in d:
                errors += 1
            # Checkpoint every 50
            if i % 50 == 0 or i == len(remaining):
                with open(ckpt_path, "w") as f:
                    json.dump(results, f)
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                log.info(
                    "    [%s] %d/%d (%.1f/s, %d err, %.0fs)",
                    model_name,
                    i,
                    len(remaining),
                    rate,
                    errors,
                    elapsed,
                )

    elapsed = time.time() - t0
    log.info(
        "  %s done: %d in %.1fs (%d errors)",
        model_name,
        len(remaining),
        elapsed,
        errors,
    )

    # Save final, remove checkpoint
    final_path = output_dir / f"{model_name}_results.json"
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    if ckpt_path.exists():
        ckpt_path.unlink()

    return results


# ---------------------------------------------------------------------------
# Concordance metrics
# ---------------------------------------------------------------------------


def compute_concordance(
    results: list[dict],
    gold_map: dict[str, int],
) -> dict:
    """Concordance of LLM decisions vs gold standard (TARGET)."""
    valid = [r for r in results if r.get("decision") in ("MATCH", "NONMATCH")]
    unsure = [r for r in results if r.get("decision") == "UNSURE"]
    errored = [r for r in results if "error" in r]

    y_true, y_pred = [], []
    for r in valid:
        pid = r["pair_id"]
        if pid in gold_map:
            y_true.append(gold_map[pid])
            y_pred.append(1 if r["decision"] == "MATCH" else 0)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)

    if n == 0:
        return {"error": "no valid results matched gold standard"}

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    acc_lo, acc_hi = wilson_ci(acc, n)

    n_total = len(valid) + len(unsure)
    unsure_rate = len(unsure) / n_total if n_total > 0 else 0.0

    confs = [r["confidence"] for r in valid if "confidence" in r]
    lats = [r["total_latency_s"] for r in valid + unsure if "total_latency_s" in r]

    protocols: dict[str, int] = {}
    for r in valid + unsure:
        p = r.get("protocol", "unknown")
        protocols[p] = protocols.get(p, 0) + 1

    return {
        "n_evaluated": n_total,
        "n_valid": len(valid),
        "n_unsure": len(unsure),
        "n_errors": len(errored),
        "concordance": round(acc, 4),
        "concordance_ci95": [round(acc_lo, 4), round(acc_hi, 4)],
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "unsure_rate": round(unsure_rate, 4),
        "mean_confidence": round(float(np.mean(confs)), 4) if confs else None,
        "mean_latency_s": round(float(np.mean(lats)), 2) if lats else None,
        "median_latency_s": round(float(np.median(lats)), 2) if lats else None,
        "protocols": protocols,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> dict:
    parser = argparse.ArgumentParser(
        description="C2 Pilot: 517-pair multi-model evaluation"
    )
    parser.add_argument(
        "--model", choices=list(MODELS.keys()), help="Run single model only"
    )
    parser.add_argument(
        "--sample-only", action="store_true", help="Just create sample, no LLM"
    )
    parser.add_argument(
        "--workers", type=int, default=N_WORKERS, help="Concurrent workers"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("C2 PILOT: 517-pair stratified multi-model evaluation")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Run GZ-CMD v3 pipeline
    # ------------------------------------------------------------------
    log.info("[1] Running GZ-CMD v3 pipeline (vigilância)...")
    df, summary = run_v3(
        input_csv=str(INPUT_CSV),
        config_path=str(CONFIG_PATH),
        mode=MODE,
        p_cal="fit_platt",
    )
    log.info("    Total pairs: %d", summary.rows)
    log.info("    Actions: %s", summary.actions)
    n_review = summary.actions.get("LLM_REVIEW", 0)

    # ------------------------------------------------------------------
    # Step 2: Stratified sample
    # ------------------------------------------------------------------
    if n_review < N_SAMPLE:
        log.warning(
            "    Only %d LLM_REVIEW pairs (need %d). Using all.", n_review, N_SAMPLE
        )
        sample_df = df[df["action"] == "LLM_REVIEW"].copy()
    else:
        log.info("[2] Stratified sampling %d pairs (seed=%d)...", N_SAMPLE, SEED)
        sample_df = stratified_sample(df, N_SAMPLE, SEED)

    log.info("    Sampled: %d pairs", len(sample_df))
    log.info("    Band distribution:")
    for band, count in sample_df["band"].value_counts().sort_index().items():
        log.info("      %s: %d", band, count)

    # Build pair_id -> TARGET gold mapping
    gold_map: dict[str, int] = {}
    pair_ids: list[str] = []
    for _, row in sample_df.iterrows():
        pid = make_pair_id(row["COMPREC"], row["REFREC"], row.get("PASSO", "1"))
        gold_map[pid] = int(row["TARGET"])
        pair_ids.append(pid)

    # Save sample metadata
    sample_meta = {
        "n_sample": len(sample_df),
        "n_total_review": n_review,
        "seed": SEED,
        "mode": MODE,
        "band_distribution": {
            k: int(v) for k, v in sample_df["band"].value_counts().items()
        },
        "target_distribution": {
            str(k): int(v) for k, v in sample_df["TARGET"].value_counts().items()
        },
        "pair_ids": pair_ids,
    }
    with open(OUTPUT_DIR / "sample_meta.json", "w") as f:
        json.dump(sample_meta, f, indent=2, ensure_ascii=False)
    log.info("    Sample metadata saved to %s", OUTPUT_DIR / "sample_meta.json")

    if args.sample_only:
        log.info("--sample-only flag set. Exiting.")
        return {}

    # ------------------------------------------------------------------
    # Step 3: Build dossiers
    # ------------------------------------------------------------------
    log.info("[3] Building dossiers...")
    dossiers = [
        build_dossier(row, include_macd=True) for _, row in sample_df.iterrows()
    ]
    log.info("    Built %d dossiers", len(dossiers))

    # ------------------------------------------------------------------
    # Step 4: Run models
    # ------------------------------------------------------------------
    models_to_run = {args.model: MODELS[args.model]} if args.model else MODELS
    all_metrics: dict[str, dict] = {}

    for model_name, model_cfg in models_to_run.items():
        log.info("[4] Model: %s", model_name)
        results = run_model(model_name, model_cfg, dossiers, OUTPUT_DIR, args.workers)
        if results is None:
            continue

        metrics = compute_concordance(results, gold_map)
        all_metrics[model_name] = metrics
        log.info(
            "    Concordance: %.4f  CI95: %s",
            metrics["concordance"],
            metrics["concordance_ci95"],
        )
        log.info(
            "    Prec: %.4f  Rec: %.4f  F1: %.4f",
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )
        log.info(
            "    Unsure: %.2f%%  Errors: %d",
            metrics["unsure_rate"] * 100,
            metrics["n_errors"],
        )
        if metrics["mean_latency_s"]:
            log.info(
                "    Latency: mean=%.2fs  median=%.2fs",
                metrics["mean_latency_s"],
                metrics["median_latency_s"],
            )

    # ------------------------------------------------------------------
    # Step 5: Summary
    # ------------------------------------------------------------------
    summary_path = OUTPUT_DIR / "pilot_c2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    log.info("")
    log.info("=" * 80)
    hdr = f"{'Model':<16} {'Conc':>7} {'CI95':>15} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Uns%':>6} {'Lat':>6}"
    log.info(hdr)
    log.info("-" * 80)
    for name, m in all_metrics.items():
        ci = m.get("concordance_ci95", [0, 0])
        lat = m.get("mean_latency_s", "-")
        log.info(
            "%-16s %7.4f [%5.3f,%5.3f] %7.4f %7.4f %7.4f %5.1f%% %6s",
            name,
            m["concordance"],
            ci[0],
            ci[1],
            m["precision"],
            m["recall"],
            m["f1"],
            m["unsure_rate"] * 100,
            lat,
        )
    log.info("=" * 80)
    log.info("Results: %s", OUTPUT_DIR)

    return all_metrics


if __name__ == "__main__":
    main()
