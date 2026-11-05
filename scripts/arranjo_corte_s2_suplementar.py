"""Authorized sensitivity-only S2 supplement; freeze its queue before paid calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gzcmd.arranjo_corte import MODEL, REMOVED_FIELDS, cache_matches, pending_cost, sha256
from scripts.arranjo_corte_s1_simular import MODES, PREREG, PROMPT, verify_prereg
from scripts import arranjo_corte_s2_executar as executor
from scripts import funil_gzcmd_v2 as funnel

LOCAL = ROOT / ".local/arranjo_corte/suplemento"
OUT = ROOT / "data/sprint3b/anthropic_sem_mo_v4_arranjo_corte_supl"
BASE_OUT = ROOT / "data/sprint3b/anthropic_sem_mo_v4_arranjo_corte"


def read_run(path, mode):
    run = json.loads(path.read_text(encoding="utf-8"))
    if (run.get("mode") != mode or run.get("model") != MODEL
            or not run.get("complete") or run.get("n_errors") != 0
            or run.get("prompt_sha256") != sha256(PROMPT)
            or set(run.get("removed_fields", [])) != set(REMOVED_FIELDS)):
        raise ValueError(f"Invalid reference run: {path}")
    return run


def validate_dossiers(path, run, mode, views):
    """Never reuse a response with a different policy-independent dossier."""
    import hashlib
    identity_path = path.with_name(path.stem + "_journal") / "identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.exists() else run
    for result in run["results"]:
        pid = result["pair_id"]
        digest = hashlib.sha256(views[pid].to_json(sort_keys=True).encode()).hexdigest()
        if not cache_matches(identity, run, mode=mode, pair_id=pid,
                             prompt_sha256=sha256(PROMPT), dossier_sha256=digest):
            raise ValueError(f"Dossier cache identity mismatch: {mode}/{pid}")


def responses(data_root, mode, *, include_supplement=False, views=None):
    old = funnel.ROOT
    funnel.ROOT = data_root
    try:
        decisions, available = funnel.load_responses(mode, funnel.HAIKU_V4, include_v31=True)
        paths = list(funnel.artifact_paths(mode, funnel.HAIKU_V4, include_v31=True))
    finally:
        funnel.ROOT = old
    if not available:
        raise ValueError("Historical responses unavailable")
    additional = [BASE_OUT / f"exp2_llm_sync_{mode}.json"]
    if include_supplement:
        additional.append(OUT / f"exp2_llm_sync_{mode}.json")
    for path in additional:
        run = read_run(path, mode)
        for result in run["results"]:
            pid = result["pair_id"]
            if pid in decisions:
                raise ValueError("A pair was reviewed twice across cohorts")
            decision = result["decision"].strip().upper()
            if decision not in {"MATCH", "NONMATCH", "UNSURE"}:
                raise ValueError("Invalid structured reviewer decision")
            decisions[pid] = decision
        paths.append(path)
    if views is None:
        import run_exp2_sem_model_outputs as runner
        from gzcmd.ml_oof import pair_ids
        from gzcmd.runner import run_v3
        from scripts.arranjo_corte_s1_simular import CONFIG
        historical, _ = run_v3(input_csv=data_root / "data/COMPARADORSEMIDENT.csv",
                               config_path=CONFIG, mode=mode, p_cal="fit_platt", policy_version="v3")
        selected = historical.loc[pair_ids(historical).isin(decisions)]
        views = {d.pair_id: runner.abl.DossierView(d, True)
                 for d in runner.ant.exp2.build_dossiers(selected, only_llm_review=False)}
    for path in paths:
        validate_dossiers(path, read_run(path, mode), mode, views)
    return decisions, paths


def prepare(data_root):
    LOCAL.mkdir(parents=True, exist_ok=True)
    queues, counts, hashes = {}, {}, {}
    for mode in MODES:
        known, _ = responses(data_root, mode)
        selected = set()
        for arm in ("P-Platt", "A3"):
            policy = pd.read_csv(ROOT / f".local/arranjo_corte/policy_{mode}_{arm}.csv")
            selected.update(policy.loc[policy.action.eq("LLM_REVIEW"), "pair_id"])
        queues[mode] = sorted(selected - known.keys())
        counts[mode] = len(queues[mode])
    if counts != {"vigilancia": 66, "confirmacao": 3}:
        raise ValueError("Authorized sensitivity supplement must be exactly 66/3 pairs")
    for mode in MODES:
        path = LOCAL / f"pendentes_{mode}.csv"
        pd.DataFrame({"pair_id": queues[mode]}).to_csv(path, index=False)
        hashes[mode] = sha256(path)
    note = {
        "number": 3, "kind": "authorized_sensitivity_execution_supplement",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "Author/orchestrator explicitly authorized exactly 69 unanswered sensitivity pairs before S3",
        "reason": "S2 covered P union B1 only; P-Platt and A3 require additional observed decisions for S3",
        "pending_by_mode": counts, "pending_total": 69,
        "estimated_additional_usd_with_retry_margin": pending_cost(69),
        "estimated_initial_and_supplement_usd": pending_cost(93),
        "original_S1_prereg_commit": "7b40cfb3821b46e0eb402feb44f8e529dba801a3",
        "original_S1_prereg_sha256": "b55899f0e5ab114bf89a00c3e529bd9df14f48b49cc1b10ae7ec7e9e01644601",
        "frozen_rates_sha256": sha256(ROOT / "data/arranjo_corte_taxas_crossfit.csv"),
        "pending_sha256": hashes, "execution_script_sha256": sha256(Path(__file__)),
        "parameter_change": False, "reestimate_reviewer_rates": False,
        "unchanged_protocol": "Same registered arms, features, classifier, folds, seed, calibration, costs, prompt and model; no new arm or parameter",
        "output_directory": "data/sprint3b/anthropic_sem_mo_v4_arranjo_corte_supl",
    }
    (LOCAL / "execution_note.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(json.dumps(note, indent=2))


def execute(data_root, prereg_hash, env_file):
    specification, commit = verify_prereg(data_root, prereg_hash)
    note = specification["amendments"][-1]
    if note.get("kind") != "authorized_sensitivity_execution_supplement":
        raise ValueError("Supplement must be registered and pushed before paid calls")
    if note["execution_script_sha256"] != sha256(Path(__file__)):
        raise ValueError("Registered execution script changed")
    if note["pending_by_mode"] != {"vigilancia": 66, "confirmacao": 3}:
        raise ValueError("Supplement differs from authorized 66/3")
    if note["pending_total"] > 500 or note["estimated_initial_and_supplement_usd"] > 2:
        raise ValueError("STOP: registered spending rule exceeded")
    rates = ROOT / "data/arranjo_corte_taxas_crossfit.csv"
    if sha256(rates) != note["frozen_rates_sha256"]:
        raise ValueError("Frozen reviewer rates changed")
    for mode in MODES:
        path = LOCAL / f"pendentes_{mode}.csv"
        if sha256(path) != note["pending_sha256"][mode]:
            raise ValueError("Registered pending queue changed")
        known, _ = responses(data_root, mode)
        ids = set(pd.read_csv(path).pair_id)
        if len(ids) != note["pending_by_mode"][mode] or ids & known.keys():
            raise ValueError("Supplement would repeat a reviewed pair")
    from dotenv import load_dotenv
    load_dotenv(env_file)
    OUT.mkdir(parents=True, exist_ok=True)
    executor.OUT, executor.LOCAL = OUT, LOCAL
    runs = [executor.run_mode(mode) for mode in MODES]
    if sha256(rates) != note["frozen_rates_sha256"]:
        raise ValueError("Reviewer rates changed during supplement")
    report = {"stage": "S2_authorized_sensitivity_supplement", "prereg_commit": commit,
              "prereg_sha256": prereg_hash, "frozen_rates_sha256": note["frozen_rates_sha256"],
              "runs": runs, "total_pairs": sum(r["n_reviewed"] for r in runs),
              "total_api_calls": sum(r["transport"]["calls"] for r in runs),
              "total_cost_estimate_usd": sum(r["cost_estimate_usd"] for r in runs)}
    (OUT / "execucao.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["secret_audit"] = executor.audit_secrets()
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--prereg-sha256")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.data_root)
    else:
        if not args.env_file or not args.prereg_sha256:
            parser.error("--env-file and --prereg-sha256 required for execution")
        execute(args.data_root, args.prereg_sha256, args.env_file)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("SUPPLEMENT_FATAL", type(exc).__name__, flush=True)
        sys.exit(1)
