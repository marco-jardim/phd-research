"""End-to-end integration test for Sprint 3A: dossier build + LLM review.

Runs the full pipeline on a sample of grey-zone pairs:
  1. Load data + feature engineering
  2. Calibrate (Platt) + triage via policy engine
  3. Build dossiers for LLM_REVIEW pairs
  4. Send to LLM via dual-agent + arbiter protocol
  5. Print comparison table

Usage:
    python scripts/test_sprint3a_e2e.py [--n-pairs 4] [--mode vigilancia]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gzcmd.loader import load_comparador_csv, LoadConfig
from gzcmd.calibration import (
    get_nota_series,
    get_target_series,
    fit_platt,
    predict_platt,
)
from gzcmd.bands import BandAssigner, BandDefinition
from gzcmd.guardrails import apply_guardrails
from gzcmd.config import load_config
from gzcmd.runner import build_engine_from_config
from gzcmd.llm_dossier import build_dossier, build_dossiers, make_pair_id
from gzcmd.llm_review import LLMReviewer, ReviewResult, LLMCallError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def sample_llm_review_pairs(df, *, n_pairs: int = 4):
    """Get a sample of LLM_REVIEW pairs, stratified across bands."""
    llm_rows = df[df["action"] == "LLM_REVIEW"].copy()
    if llm_rows.empty:
        logger.warning("No LLM_REVIEW pairs found — sampling from grey zone instead")
        nota = df["nota_final"] if "nota_final" in df.columns else df["nota final"]
        llm_rows = df[(nota >= 4.0) & (nota <= 8.0)].copy()

    if len(llm_rows) <= n_pairs:
        return llm_rows

    # Stratify by band
    bands_present = llm_rows["band"].unique()
    per_band = max(1, n_pairs // len(bands_present))
    sampled = []
    for band in bands_present:
        band_rows = llm_rows[llm_rows["band"] == band]
        sampled.append(band_rows.head(per_band))

    result = __import__("pandas").concat(sampled).head(n_pairs)
    return result


def main():
    parser = argparse.ArgumentParser(description="Sprint 3A E2E test")
    parser.add_argument("--n-pairs", type=int, default=4)
    parser.add_argument("--mode", default="vigilancia")
    parser.add_argument("--config", default="gzcmd/gzcmd_v3_config.yaml")
    parser.add_argument("--input", default="data/COMPARADORSEMIDENT.csv")
    args = parser.parse_args()

    # --- Step 1: Load & feature engineer ---
    print("=" * 70)
    print("SPRINT 3A — END-TO-END INTEGRATION TEST")
    print("=" * 70)

    print("\n[1/5] Loading data and engineering features...")
    df = load_comparador_csv(args.input, cfg=LoadConfig(macd_enabled=True))
    print(f"  Loaded {len(df)} pairs, {df.columns.size} columns")

    # --- Step 2: Calibrate + band + guardrails + triage ---
    print("\n[2/5] Running pipeline (calibrate → band → guardrails → triage)...")
    cfg = load_config(args.config)

    # Band assignment
    nota = get_nota_series(df)
    assigner = BandAssigner.from_config(cfg)
    df["band"] = assigner.assign(nota)

    # Calibration (Platt)
    target = get_target_series(df)
    platt_model = fit_platt(nota, target)
    df["p_cal"] = predict_platt(nota, model=platt_model)

    # Guardrails
    gout = apply_guardrails(df)
    df["guardrail"] = gout.guardrail
    df["guardrail_reason"] = gout.reason

    # Triage
    engine = build_engine_from_config(cfg, mode=args.mode)
    df = engine.triage(df)

    # Stats
    action_counts = df["action"].value_counts()
    print(f"  Actions: {dict(action_counts)}")

    llm_count = action_counts.get("LLM_REVIEW", 0)
    print(f"  LLM_REVIEW pairs: {llm_count}")

    # --- Step 3: Build dossiers ---
    print(f"\n[3/5] Building dossiers (sampling {args.n_pairs} pairs)...")
    sample = sample_llm_review_pairs(df, n_pairs=args.n_pairs)
    dossiers = build_dossiers(sample, only_llm_review=False)
    print(f"  Built {len(dossiers)} dossiers")

    # Print one sample dossier
    if dossiers:
        print(f"\n  Sample dossier (pair_id={dossiers[0].pair_id}):")
        d = dossiers[0].to_dict()
        print(f"    nota_final: {d['features']['nota_final']}")
        print(f"    band: {d['model_outputs']['band']}")
        print(f"    p_cal: {d['model_outputs']['p_cal']}")
        print(f"    name scores: {d['features']['name']}")
        print(f"    dob scores: {d['features']['dob']}")
        print(f"    mother scores: {d['features']['mother']}")
        print(f"    flags: {d['features']['flags']}")

    # --- Step 4: LLM review ---
    print(f"\n[4/5] Running LLM dual-agent review ({len(dossiers)} dossiers)...")
    reviewer = LLMReviewer.from_env()
    results = reviewer.review_batch(dossiers)

    # --- Step 5: Results ---
    print("\n[5/5] RESULTS")
    print("=" * 70)

    # Table header
    header = f"{'Pair ID':<18} {'Band':<12} {'Nota':>6} {'p_cal':>6} {'Decision':<10} {'Conf':>5} {'Protocol':<10} {'Time':>5}"
    print(header)
    print("-" * len(header))

    successes = []
    errors = []
    for r in results:
        if isinstance(r, LLMCallError):
            errors.append(r)
            print(f"  ERROR: {r}")
            continue
        successes.append(r)
        # Find matching dossier for nota/band
        dos = next((d for d in dossiers if d.pair_id == r.pair_id), None)
        nota_val = dos.features.nota_final if dos else 0.0
        band_val = dos.model_outputs.band if dos else "?"
        print(
            f"{r.pair_id:<18} {band_val:<12} {nota_val:>6.2f} "
            f"{dos.model_outputs.p_cal if dos else 0:>6.3f} "
            f"{r.decision:<10} {r.confidence:>5.2f} {r.protocol:<10} "
            f"{r.total_latency_s:>5.1f}s"
        )

    # Agreement stats
    print(f"\n--- Summary ---")
    print(f"Total: {len(results)}, Success: {len(successes)}, Errors: {len(errors)}")
    if successes:
        consensus_count = sum(1 for r in successes if r.protocol == "consensus")
        arbiter_count = sum(1 for r in successes if r.protocol == "arbiter")
        print(f"Consensus: {consensus_count}, Arbiter needed: {arbiter_count}")

        decisions = {}
        for r in successes:
            decisions[r.decision] = decisions.get(r.decision, 0) + 1
        print(f"Decisions: {decisions}")

        avg_latency = sum(r.total_latency_s for r in successes) / len(successes)
        print(f"Avg latency: {avg_latency:.1f}s")

        # Agent agreement detail
        print(f"\n--- Agent Detail ---")
        for r in successes:
            a_d = r.agent_a.decision
            b_d = r.agent_b.decision
            arb = f" → Arbiter: {r.arbiter.decision}" if r.arbiter else ""
            print(
                f"  {r.pair_id}: A={a_d}({r.agent_a.confidence:.2f}) B={b_d}({r.agent_b.confidence:.2f}){arb}"
            )

    # Save to JSON
    out_path = Path("data") / "sprint3a_e2e_results.json"
    out_data = [
        r.to_dict() if isinstance(r, ReviewResult) else {"error": str(r)}
        for r in results
    ]
    out_path.write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
