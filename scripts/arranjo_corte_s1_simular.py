"""Preregister, then simulate the fixed ORL arrangement without any API calls.

--preregister only fingerprints inputs/code; it does not train or simulate.
S1 requires --prereg-sha256 and the preregistration already committed/pushed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gzcmd.arranjo_corte import (
    CUTOFFS, MODEL, RATE_COLUMNS, REMOVED_FIELDS, below_cutoff, cache_matches,
    crossfit_rates, nested_cutoff_probabilities, numeric_features, pending_cost,
    sha256, summarize_actions,
)
from gzcmd.bands import BandAssigner
from gzcmd.config import load_config
from gzcmd.guardrails import apply_guardrails
from gzcmd.ml_oof import pair_ids
from gzcmd.runner import build_engine_from_config, run_v3

PREREG = ROOT / "gzcmd/prereg_arranjo_corte.yaml"
CONFIG = ROOT / "gzcmd/gzcmd_v3_config.yaml"
RATES = ROOT / "gzcmd/error_rates_by_band_v4.yaml"
PROMPT = ROOT / "gzcmd/gzcmd_v4_llm_prompt.md"
LOCAL = ROOT / ".local/arranjo_corte"
MODES = ("vigilancia", "confirmacao")
CODE = ("gzcmd/arranjo_corte.py", "scripts/arranjo_corte_s1_simular.py",
        "scripts/arranjo_corte_s2_executar.py",
        "tests/test_arranjo_corte.py", "gzcmd/M_arranjo_corte.md",
        "scripts/run_exp2_sem_model_outputs.py", "gzcmd/ml_oof.py",
        "gzcmd/runner.py", "gzcmd/reliability.py", "gzcmd/guardrails.py",
        "gzcmd/calibration.py", "gzcmd/gzcmd_v3_policy_engine.py",
        "gzcmd/llm_dossier.py", "scripts/run_exp2_ablacao_dossie.py",
        "scripts/funil_gzcmd_v2.py")


def cohort_sources(data_root):
    specification = yaml.safe_load(RATES.read_text(encoding="utf-8"))
    sources = [s for s in specification["sources"] if s["model"] == MODEL]
    if len(sources) != 6:
        raise ValueError("Exactly three frozen Haiku v4 cohorts per mode required")
    for source in sources:
        path = data_root / source["path"]
        run = json.loads(path.read_text(encoding="utf-8"))
        if (run.get("mode") != source["mode"] or run.get("model") != MODEL
                or run.get("prompt_version") != "v4" or run.get("complete") is not True
                or run.get("n_reviewed") != source["n_respondido"]
                or run.get("prompt_sha256") != sha256(PROMPT)
                or set(run.get("removed_fields", [])) != set(REMOVED_FIELDS)):
            raise ValueError(f"Invalid source cohort provenance: {source['path']}")
        # The published rates manifest predates final checkpoint rewrites. Pin
        # actual bytes in this preregistration and retain its declared hash too.
        source["rates_yaml_sha256"] = source["sha256"]
        source["sha256"] = sha256(path)
        source["sha256_matches_rates_manifest"] = source["sha256"] == source["rates_yaml_sha256"]
    return sources


def fingerprints(data_root):
    sources = cohort_sources(data_root)
    paths = {str(p): sha256(ROOT / p) for p in CODE}
    paths["gzcmd/gzcmd_v3_config.yaml"] = sha256(CONFIG)
    paths["gzcmd/error_rates_by_band_v4.yaml"] = sha256(RATES)
    paths["gzcmd/gzcmd_v4_llm_prompt.md"] = sha256(PROMPT)
    for p in ("data/COMPARADORSEMIDENT.csv", "data/pcal_ml_oof.csv", "data/pcal_ml_oof.json"):
        paths[p] = sha256(data_root / p)
    for source in sources:
        paths[source["path"]] = sha256(data_root / source["path"])
        path = Path(source["path"])
        identity = path.parent / (path.stem + "_journal") / "identity.json"
        if (data_root / identity).exists():
            paths[identity.as_posix()] = sha256(data_root / identity)
    return paths


def preregister(data_root):
    if PREREG.exists():
        raise ValueError("Preregistration exists; amendments require author approval")
    cfg = load_config(CONFIG)
    sources = cohort_sources(data_root)
    prompt = sha256(PROMPT)
    reference = json.loads((data_root / "data/sprint3b/anthropic_sem_mo_v4_lote31_refrec/exp2_llm_sync_vigilancia.json").read_text(encoding="utf-8"))
    if reference["prompt_sha256"] != prompt or not prompt.startswith("0693f4a8"):
        raise ValueError("Approved prompt fingerprint mismatch")
    import sklearn
    import xgboost
    spec = {
        "schema_version": 1, "protocol": "ORL aceita SCORE >= corte; GZ-CMD v3.1 abaixo",
        "base_commit": "b09e979", "created_at": datetime.now(timezone.utc).isoformat(),
        "sha256": fingerprints(data_root),
        "classifier": {"source": "gzcmd/ml_oof.py:_classifier,numeric_features",
                       "features": 30, "outer_folds": 5, "inner_folds": 5,
                       "split_by": "refrec", "seed": 42, "n_jobs": 2,
                       "n_estimators": 300, "max_depth": 6, "learning_rate": .05,
                       "subsample": .8, "colsample_bytree": .8,
                       "scale_pos_weight": "n_negative/n_positive in each fit",
                       "eval_metric": "logloss", "sklearn": sklearn.__version__,
                       "xgboost": xgboost.__version__},
        "orl_cutoffs": CUTOFFS, "partition_check": {"above": 211, "below": 61485, "below_PAR2": 36,
                                                    "above_exactly_PAR1": True},
        "costs_and_budgets": {mode: asdict(cfg.decision_policy.modes[mode]) for mode in MODES},
        "bands": [asdict(b) for b in cfg.bands.definitions],
        "guardrails": {"version": "v3.1", "nonmatch": "nota < 5", "match": "nota >= 9 and recalibrated_p >= 0.9"},
        "arms": {"P": "Same full-data classifiers and 5x5 splits; nested isotonic only on below-cutoff training inner OOF",
                 "P-Platt": "Same inner OOF, Platt on clipped raw XGBoost logit; calibration.py defaults l2=.001 max_iter=100 tol=1e-10",
                 "B1": "Entire base, original REFREC p, cross-fitted reviewer rates"},
        "sensitivities": {"A3": "Restrict XGBoost fitting to below cutoff, retaining original outer and inner REFREC splits",
                          "P-PAR2-grey": "P with e_fn estimated only on PAR=2, pooling [5,9) bands; same M3 fallback"},
        "calibration_abort_rule": "If fewer than 10 PAR=2 in a calibration set, use Platt as principal in that fold and record it",
        "reviewer_rates": {"estimator": "M3 FP/negative, FN/positive, UNSURE/n; fixed if n<10 or denominator=0",
                           "sources": sources, "crossfit": "Same external REFREC folds; exclude evaluated fold; freeze before S2",
                           "s2_reestimation": False},
        "llm": {"provider": "anthropic", "model": MODEL, "prompt_version": "v4", "prompt_sha256": prompt,
                "removed_fields": list(REMOVED_FIELDS), "workers": 4, "modes_sequential": True,
                "dossier_reconstruction": "run_v3 fit_platt/v3; build_dossiers; DossierView(remove_model_outputs=True)",
                "cache_key": ["mode", "pair_id", "model", "prompt_sha256", "dossier_sha256", "removed_fields"],
                "legacy_identity": "Compare existing journal identity.json; legacy complete checkpoints without journal identity retain their embedded, SHA256-pinned identity; never treat absence of dossier hash as a hit",
                "checkpoint": "Existing incremental runner with Windows replacement backoff and transport journal"},
        "primary_outcomes_S3_only": ["Among 36 PAR=2: automatically accepted, referred to human, lost; Wilson 95% intervals",
                                    "Full base and nota [5,9) by mode: acceptance sensitivity/specificity, FP/FN, human referrals, LLM calls, costs; human 5% is scenario, not observed truth"],
        "S1_outputs": {"summary": "mode x arm x band x action counts ONLY", "rates": "mode x fold x band reviewer probabilities",
                       "pending": "Ignored local pair_id-only CSV, union of P and B1 per mode; no sensitivity API calls"},
        "stop_rule": {"pending_union_P_B1_more_than": 500, "estimated_usd_more_than": 2,
                      "input_tokens_per_pair": 6847, "output_tokens_per_pair": 3904,
                      "input_usd_per_million": .10, "output_usd_per_million": .50,
                      "retry_multiplier": 1.20, "action": "Stop and consult author; do not call API"},
        "amendments": [],
    }
    PREREG.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("PREREG_SHA256", sha256(PREREG))


def verify_prereg(data_root, expected_hash):
    if not expected_hash or sha256(PREREG) != expected_hash:
        raise ValueError("Preregistration SHA256 mismatch")
    spec = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    if spec["sha256"] != fingerprints(data_root):
        raise ValueError("Preregistered input/code fingerprint mismatch")
    commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", "gzcmd/prereg_arranjo_corte.yaml"], cwd=ROOT, text=True).strip()
    if not commit:
        raise ValueError("Preregistration must be committed before S1")
    recorded = subprocess.check_output(["git", "show", commit + ":gzcmd/prereg_arranjo_corte.yaml"], cwd=ROOT)
    if yaml.safe_load(recorded) != spec:
        raise ValueError("Working preregistration differs from committed version")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/pq/arranjo-corte"], cwd=ROOT, check=True)
    return spec, commit


def load_raw(path):
    raw = pd.read_csv(path, sep=";", dtype=str, low_memory=False)
    raw.columns = [c.split(",")[0] for c in raw.columns]
    if not raw.columns.is_unique:
        raise ValueError("Ambiguous normalized headers")
    return raw


def validated_responses(data_root, mode, views, sources):
    from scripts import funil_gzcmd_v2 as funnel
    original_root = funnel.ROOT
    funnel.ROOT = data_root
    try:
        responses, available = funnel.load_responses(mode, funnel.HAIKU_V4, include_v31=True)
    finally:
        funnel.ROOT = original_root
    if not available:
        raise ValueError("Required reviewer sources missing")
    valid = {}
    embedded_identity_cohorts = 0
    for source in sources:
        if source["mode"] != mode:
            continue
        path = data_root / source["path"]
        run = json.loads(path.read_text(encoding="utf-8"))
        identity_path = path.parent / (path.stem + "_journal") / "identity.json"
        if identity_path.exists():
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
        else:
            # Old runner checkpoints already store the exact per-pair identities.
            # Pin the whole checkpoint, never synthesize a missing dossier hash.
            identity = run
            embedded_identity_cohorts += 1
        for row in run["results"]:
            pid = row["pair_id"]
            digest = hashlib.sha256(views[pid].to_json(sort_keys=True).encode()).hexdigest()
            if not cache_matches(identity, run, mode=mode, pair_id=pid,
                                 prompt_sha256=sha256(PROMPT), dossier_sha256=digest):
                raise ValueError(f"Reference cache identity mismatch: {mode}/{pid}")
            valid[pid] = row["decision"].strip().upper()
    if valid != responses:
        raise ValueError("load_responses includes cohorts outside the frozen three-cohort source set")
    return valid, embedded_identity_cohorts


def simulate(data_root, expected_hash):
    spec, commit = verify_prereg(data_root, expected_hash)
    LOCAL.mkdir(parents=True, exist_ok=True)
    raw = load_raw(data_root / "data/COMPARADORSEMIDENT.csv")
    below = below_cutoff(raw)
    print("PARTITION above=211 below=61485 below_PAR2=36", flush=True)
    x = numeric_features(raw)
    if len(x.columns) != 30:
        raise ValueError("Expected exactly 30 classifier attributes")
    par = pd.to_numeric(raw.PAR, errors="coerce").fillna(0).astype(int).to_numpy()
    groups = raw.REFREC.astype("string").str.strip()
    if groups.isna().any() or groups.eq("").any():
        raise ValueError("Missing REFREC")
    import psutil
    memory = psutil.virtual_memory()
    print("RAM_AVAILABLE_GIB", round(memory.available / 2**30, 3), "N_JOBS", 2, flush=True)
    events = []

    def audit(fold, stage, fit, predict, external):
        if not set(groups.iloc[fit]).isdisjoint(groups.iloc[external]):
            raise ValueError("External REFREC leaked into fit")
        if stage.endswith("calibrator") and stage != "outer_and_calibrator" and not below[fit].all():
            raise ValueError("Above-cutoff pair leaked into calibrator")
        events.append({"fold": fold, "stage": stage, "n_fit": len(fit), "n_predict": len(predict)})

    probabilities, folds, calibration = nested_cutoff_probabilities(
        x, np.isin(par, [1, 2]).astype(int), below,
        groups.to_numpy(dtype=str), par2=par == 2, audit=audit)
    ids = pair_ids(raw)
    original = pd.read_csv(data_root / "data/pcal_ml_oof.csv").set_index("pair_id").loc[ids]
    np.testing.assert_array_equal(folds, original.fold.to_numpy())
    np.testing.assert_allclose(probabilities["B1"], original.p_cal_ml.to_numpy(), rtol=0, atol=1e-14)
    # Persist pair-level scores only in ignored local storage for the S3 replay.
    np.savez_compressed(LOCAL / "oof.npz", pair_id=ids.to_numpy(dtype=str), fold=folds,
                        **probabilities)
    cfg = load_config(CONFIG)
    band = BandAssigner.from_config(cfg).assign(x["nota final"])
    if band.isna().any():
        raise ValueError("Unassigned band")
    sources = cohort_sources(data_root)
    summaries, rate_rows = [], []
    pending_counts, cache_metadata, rate_sensitivity = {}, {}, {}
    for mode in MODES:
        # Reconstruct exactly the policy-independent historical dossier payload.
        import run_exp2_sem_model_outputs as runner
        historical, _ = run_v3(input_csv=data_root / "data/COMPARADORSEMIDENT.csv",
                               config_path=CONFIG, mode=mode, p_cal="fit_platt", policy_version="v3")
        wanted = set().union(*(set(json.loads((data_root / s["path"]).read_text(encoding="utf-8"))["selected_pair_ids"])
                              for s in sources if s["mode"] == mode))
        # Build only required rows; build_dossiers is row-wise and payloads do not
        # depend on other rows. Never write dossier contents to disk.
        selected = historical.loc[pair_ids(historical).isin(wanted)]
        views = {d.pair_id: runner.abl.DossierView(d, True)
                 for d in runner.ant.exp2.build_dossiers(selected, only_llm_review=False)}
        for view in views.values():
            runner.audit_view(view)
        responses, embedded = validated_responses(data_root, mode, views, sources)
        cache_metadata[mode] = {"valid_cached_pairs": len(responses), "legacy_embedded_identity_cohorts": embedded}
        records = pd.DataFrame({"pair_id": ids, "fold": folds, "band": band,
                                "y": np.isin(par, [1, 2]).astype(int), "par": par})
        records["decision"] = records.pair_id.map(responses)
        records = records.loc[records.decision.notna()]
        rates = crossfit_rates(records, cfg.llm_review.error_rates_by_band, list(range(5)))
        sensitivity = crossfit_rates(records, cfg.llm_review.error_rates_by_band, list(range(5)), par2_grey=True)
        rate_sensitivity[mode] = {str(f): {b: asdict(r) for b, r in rs.items()} for f, rs in sensitivity.items()}
        for fold, rs in rates.items():
            for b, rate in rs.items():
                rate_rows.append({"modo": mode, "fold": fold, "banda": b, **asdict(rate)})
        needed = set()
        for arm in ("P", "P-Platt", "B1", "A3", "P-PAR2-grey"):
            p = probabilities["P" if arm == "P-PAR2-grey" else arm]
            arm_rates = sensitivity if arm == "P-PAR2-grey" else rates
            parts = []
            for fold in range(5):
                mask = (folds == fold) & (np.ones(len(raw), dtype=bool) if arm == "B1" else below)
                frame = pd.DataFrame({"pair_id": ids[mask], "nota final": x.loc[mask, "nota final"],
                                      "band": band[mask], "p_cal": p[mask]})
                guards = apply_guardrails(frame, policy_version="v3.1")
                frame["guardrail"], frame["guardrail_reason"] = guards.guardrail, guards.reason
                engine = build_engine_from_config(cfg, mode=mode, policy_version="v3.1", rates_yaml=RATES)
                engine.llm_error_by_band = arm_rates[fold]
                parts.append(engine.triage(frame))
            out = pd.concat(parts).sort_index()
            if int(out.action.eq("LLM_REVIEW").sum()) > cfg.decision_policy.modes[mode].llm_max_calls_per_window:
                raise ValueError("Fold-wise allocation exceeds global LLM budget; author decision required")
            summaries.append(summarize_actions(out, mode, arm))
            # Strictly private per-pair policy cache, no labels.
            out[["pair_id", "band", "p_cal", "action"]].to_csv(LOCAL / f"policy_{mode}_{arm}.csv", index=False)
            if arm in {"P", "B1"}:
                needed.update(out.loc[out.action.eq("LLM_REVIEW"), "pair_id"])
        pending = sorted(needed - responses.keys())
        pd.DataFrame({"pair_id": pending}).to_csv(LOCAL / f"pendentes_{mode}.csv", index=False)
        pending_counts[mode] = len(pending)
        del historical, views
    summary = pd.concat(summaries, ignore_index=True)
    rates_frame = pd.DataFrame(rate_rows)[RATE_COLUMNS]
    summary.to_csv(ROOT / "data/arranjo_corte_s1_resumo.csv", index=False)
    rates_frame.to_csv(ROOT / "data/arranjo_corte_taxas_crossfit.csv", index=False)
    n_pending = sum(pending_counts.values())
    metadata = {"stage": "S1_no_API_no_outcome_metrics", "prereg_commit": commit,
                "prereg_sha256": expected_hash, "input_fingerprints": spec["sha256"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "partition": {"above": 211, "below": 61485, "below_PAR2": 36},
                "calibration": calibration, "audit_fit_boundaries": events,
                "original_REFREC_oof_reproduced": True, "cache": cache_metadata,
                "pending_P_union_B1_by_mode": pending_counts, "pending_total": n_pending,
                "pending_sha256": {mode: sha256(LOCAL / f"pendentes_{mode}.csv") for mode in MODES},
                "estimated_cost_usd_including_retries": pending_cost(n_pending),
                "stop_rule_triggered": n_pending > 500 or pending_cost(n_pending) > 2,
                "frozen_rates_sha256": sha256(ROOT / "data/arranjo_corte_taxas_crossfit.csv"),
                "summary_sha256": sha256(ROOT / "data/arranjo_corte_s1_resumo.csv"),
                "sensitivity_PAR2_grey_rates": rate_sensitivity,
                "sensitivity_API_calls": 0, "api_calls": 0,
                "ram_available_before_training_bytes": memory.available}
    (ROOT / "data/arranjo_corte_s1_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print("S1_PENDING", json.dumps(pending_counts), "ESTIMATED_USD", pending_cost(n_pending),
          "STOP", metadata["stop_rule_triggered"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--preregister", action="store_true")
    parser.add_argument("--prereg-sha256")
    args = parser.parse_args()
    if args.preregister:
        preregister(args.data_root.resolve())
    else:
        simulate(args.data_root.resolve(), args.prereg_sha256)


if __name__ == "__main__":
    main()
