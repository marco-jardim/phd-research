#!/usr/bin/env python
"""Exp 2 — LLM clerical review via synchronous parallel calls.

Uses ThreadPoolExecutor to run dual-agent reviews in parallel.
Each worker calls LLMReviewer.review() which does Agent-A → Agent-B → Arbiter
sequentially per pair, but multiple pairs are reviewed concurrently.

Usage:
    python scripts/run_exp2_llm_sync.py --mode vigilancia
    python scripts/run_exp2_llm_sync.py --mode vigilancia --limit 50 --workers 5
    python scripts/run_exp2_llm_sync.py --mode vigilancia --resume data/sprint3b/exp2_llm_sync_vigilancia.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gzcmd.config import load_config  # noqa: E402
from gzcmd.llm_dossier import Dossier, build_dossiers  # noqa: E402
from gzcmd.llm_review import LLMReviewer, ReviewResult  # noqa: E402
from gzcmd.runner import run_v3  # noqa: E402

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy int/float types in JSON serialization."""

    def default(self, o: object) -> object:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
DATA_CSV = Path("data/COMPARADORSEMIDENT.csv")
CONFIG_YAML = Path("gzcmd/gzcmd_v3_config.yaml")
OUT_DIR = Path("data/sprint3b")


# ── Review worker ──────────────────────────────────────────────────────
def _review_one(
    reviewer: LLMReviewer, dossier: Dossier, idx: int, total: int
) -> tuple[str, ReviewResult | None, str | None]:
    """Review a single dossier. Returns (pair_id, result_or_None, error_or_None)."""
    pair_id = dossier.pair_id
    try:
        result = reviewer.review(dossier)
        log.info(
            "[%d/%d] %s → %s (%.1f) via %s  %.1fs",
            idx,
            total,
            pair_id[:12],
            result.decision,
            result.confidence,
            result.protocol,
            result.total_latency_s,
        )
        return pair_id, result, None
    except Exception as e:
        log.warning("[%d/%d] %s FAILED: %s", idx, total, pair_id[:12], e)
        return pair_id, None, str(e)


# ── Metrics ────────────────────────────────────────────────────────────
def _compute_metrics(
    results: list[dict],
    errors: list[dict],
    elapsed_s: float,
    mode: str,
) -> dict:
    """Compute summary metrics from review results."""
    n = len(results)
    decisions = Counter(r["decision"] for r in results)
    protocols = Counter(r["protocol"] for r in results)

    metrics: dict = {
        "mode": mode,
        "n_reviewed": n,
        "n_errors": len(errors),
        "total_elapsed_s": round(elapsed_s, 1),
        "avg_latency_s": round(sum(r["total_latency_s"] for r in results) / n, 2)
        if n
        else 0,
        "decisions": dict(decisions),
        "protocols": dict(protocols),
        "consensus_rate": round(protocols.get("consensus", 0) / n, 4) if n else 0,
        "arbiter_rate": round(protocols.get("arbiter", 0) / n, 4) if n else 0,
    }

    # Confidence distribution
    confs = [r["confidence"] for r in results]
    if confs:
        metrics["confidence_mean"] = round(sum(confs) / len(confs), 3)
        metrics["confidence_min"] = round(min(confs), 3)
        metrics["confidence_max"] = round(max(confs), 3)

    return metrics


# ── Main ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 2 — Sync parallel LLM review")
    parser.add_argument(
        "--mode",
        choices=["vigilancia", "confirmacao"],
        default="vigilancia",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit pairs (0=all)")
    parser.add_argument("--workers", type=int, default=10, help="Thread pool workers")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show counts, don't call LLM"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to partial results JSON to resume from",
    )
    args = parser.parse_args()
    mode = args.mode

    # ── Step 1: Run pipeline ──
    log.info("Step 1: Running v3 pipeline (mode=%s)...", mode)
    df, _summary = run_v3(
        input_csv=str(DATA_CSV),
        config_path=str(CONFIG_YAML),
        mode=mode,
        p_cal="fit_platt",
    )
    n_review = (df["action"] == "LLM_REVIEW").sum()
    log.info("Pipeline done: %d total pairs, %d LLM_REVIEW", len(df), n_review)

    # ── Step 2: Build dossiers ──
    log.info("Step 2: Building dossiers...")
    dossiers = build_dossiers(df, only_llm_review=True)
    log.info("Built %d dossiers", len(dossiers))

    # ── Step 3: Resume check ──
    done_ids: set[str] = set()
    prev_results: list[dict] = []
    prev_errors: list[dict] = []
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path) as f:
                prev = json.load(f)
            prev_results = prev.get("results", [])
            prev_errors = prev.get("errors", [])
            done_ids = {r["pair_id"] for r in prev_results}
            done_ids |= {e["pair_id"] for e in prev_errors}
            log.info("Resuming: %d already done", len(done_ids))

    pending = [d for d in dossiers if d.pair_id not in done_ids]
    if args.limit > 0:
        pending = pending[: args.limit]

    log.info("To review: %d pairs (%d workers)", len(pending), args.workers)

    if args.dry_run:
        log.info("DRY RUN — exiting")
        return

    # ── Step 4: Parallel review ──
    reviewer = LLMReviewer.from_env()
    results: list[dict] = list(prev_results)
    errors: list[dict] = list(prev_errors)
    total = len(pending)

    out_path = OUT_DIR / f"exp2_llm_sync_{mode}.json"
    checkpoint_interval = 50  # save every N completions

    t0 = time.perf_counter()
    completed_since_checkpoint = 0

    log.info("Step 4: Starting parallel review (%d workers)...", args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_review_one, reviewer, d, i + 1, total): d
            for i, d in enumerate(pending)
        }

        for future in as_completed(futures):
            pair_id, result, error = future.result()
            if result is not None:
                results.append(result.to_dict())
            else:
                errors.append({"pair_id": pair_id, "error": error or "unknown"})
            completed_since_checkpoint += 1

            # Periodic checkpoint
            if completed_since_checkpoint >= checkpoint_interval:
                _save_checkpoint(out_path, results, errors, mode, t0)
                completed_since_checkpoint = 0

    elapsed = time.perf_counter() - t0

    # ── Step 5: Compute metrics & save ──
    log.info("Step 5: Computing metrics...")
    metrics = _compute_metrics(results, errors, elapsed, mode)

    output = {
        "experiment": "exp2_llm_sync",
        "mode": mode,
        "model": reviewer.model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_review_pairs": n_review,
        "n_reviewed": len(results),
        "n_errors": len(errors),
        "workers": args.workers,
        "metrics": metrics,
        "results": results,
        "errors": errors,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    log.info("Saved: %s", out_path)

    # Also save CSV
    csv_path = out_path.with_suffix(".csv")
    _save_csv(results, csv_path)
    log.info("Saved: %s", csv_path)

    # ── Summary ──
    _print_summary(metrics)


def _save_checkpoint(
    path: Path,
    results: list[dict],
    errors: list[dict],
    mode: str,
    t0: float,
) -> None:
    """Save intermediate results for resume capability."""
    elapsed = time.perf_counter() - t0
    checkpoint = {
        "experiment": "exp2_llm_sync",
        "mode": mode,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_reviewed": len(results),
        "n_errors": len(errors),
        "elapsed_s": round(elapsed, 1),
        "results": results,
        "errors": errors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    log.info(
        "Checkpoint: %d results, %d errors (%.0fs)", len(results), len(errors), elapsed
    )


def _save_csv(results: list[dict], path: Path) -> None:
    """Save results as CSV."""
    if not results:
        return
    import csv

    keys = [
        "pair_id",
        "decision",
        "confidence",
        "protocol",
        "reason_codes",
        "total_latency_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            if isinstance(row.get("reason_codes"), list):
                row["reason_codes"] = ";".join(row["reason_codes"])
            writer.writerow(row)


def _print_summary(metrics: dict) -> None:
    """Print summary to log."""
    log.info("=" * 60)
    log.info("EXP 2 — LLM SYNC REVIEW SUMMARY")
    log.info("=" * 60)
    log.info("Mode:            %s", metrics["mode"])
    log.info("Reviewed:        %d", metrics["n_reviewed"])
    log.info("Errors:          %d", metrics["n_errors"])
    log.info("Total time:      %.1fs", metrics["total_elapsed_s"])
    log.info("Avg latency:     %.2fs", metrics["avg_latency_s"])
    log.info("Decisions:       %s", metrics["decisions"])
    log.info("Consensus rate:  %.1f%%", metrics["consensus_rate"] * 100)
    log.info("Arbiter rate:    %.1f%%", metrics["arbiter_rate"] * 100)
    if "confidence_mean" in metrics:
        log.info(
            "Confidence:      mean=%.3f  [%.3f, %.3f]",
            metrics["confidence_mean"],
            metrics["confidence_min"],
            metrics["confidence_max"],
        )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
