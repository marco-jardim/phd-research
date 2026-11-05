"""Offline exploratory workers: each process fits exactly one replica."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.arranjo_corte import below_cutoff, sha256
from gzcmd.ml_oof import numeric_features, pair_ids
from gzcmd.subamostragem import evaluate, summarize, undersampled_oof
from scripts.comparador_mesma_carga import align_original, load_raw
from scripts.qw3_fairness_analysis import _recode_sex

LOCAL = ROOT / ".local/subamostragem"
SOURCE_URL = "https://dalpozz.github.io/static/pdf/SSCI_calib_final_noCC.pdf"


def atomic_json(path, payload):
    """Atomic local checkpoint; bounded retry of Windows sharing violations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.1 * 2 ** attempt)


def versions():
    import sklearn
    import xgboost
    actual = {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__}
    if actual != {"sklearn": "1.8.0", "xgboost": "3.2.0"}:
        raise RuntimeError(f"Package version gate failed: {actual}")
    return actual


def identity(source):
    return {"input_sha256": {"COMPARADORSEMIDENT.csv": sha256(source),
                             "pcal_ml_oof.csv": sha256(ROOT / "data/pcal_ml_oof.csv"),
                             "pcal_ml_oof.json": sha256(ROOT / "data/pcal_ml_oof.json")},
            "code_sha256": {p: sha256(ROOT / p) for p in (
                "gzcmd/subamostragem.py", "scripts/subamostragem_treino.py",
                "scripts/subamostragem_treino.ps1", "gzcmd/ml_oof.py",
                "gzcmd/arranjo_corte.py", "scripts/qw3_fairness_analysis.py",
                "scripts/equidade_v31.py", "scripts/equidade_arranjo.py")},
            "versions": versions()}


def preflight(source):
    import psutil
    # Never terminate a live parent's pool; only genuinely orphaned loky workers.
    terminated = []
    for process in psutil.process_iter(["pid", "ppid", "cmdline"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
            if "loky" in command and "popen_loky" in command and not psutil.pid_exists(process.info["ppid"]):
                process.terminate()
                process.wait(timeout=10)
                terminated.append(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            print(f"Process inspection: {type(exc).__name__}", flush=True)
    memory = psutil.virtual_memory()
    # Conservative headroom for the ~15 MB feature matrix and one 8-thread model.
    if memory.available < 2 * 2**30:
        raise RuntimeError("Less than 2 GiB free RAM; no model started")
    payload = {"identity": identity(source), "free_ram_bytes": memory.available,
               "terminated_orphan_loky": terminated,
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    atomic_json(LOCAL / "preflight.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def load_inputs(source):
    raw = load_raw(source)
    x = numeric_features(raw)
    expected = json.loads((ROOT / "data/pcal_ml_oof.json").read_text(encoding="utf-8"))
    if list(x.columns) != expected["features"] or x.shape != (61696, 30):
        raise ValueError("Original 61696 x 30 design required")
    y = pd.to_numeric(raw.PAR, errors="raise").isin([1, 2]).astype(int).to_numpy()
    if int(y.sum()) != 247:
        raise ValueError("Full universe must have 247 links")
    below = below_cutoff(raw)
    original = align_original(raw, ROOT / "data/pcal_ml_oof.csv", source)
    return raw, x, y, below, original


def serializable_records(frame):
    return json.loads(frame.to_json(orient="records", double_precision=15))


def worker(source, ratio, seed):
    fingerprint = identity(source)
    pre = json.loads((LOCAL / "preflight.json").read_text(encoding="utf-8"))
    if pre["identity"] != fingerprint:
        raise ValueError("Preflight fingerprint changed")
    name = "original" if ratio is None else f"1:{ratio}"
    checkpoint = LOCAL / f"replica_{ratio or 'original'}_{seed}.json"
    if checkpoint.exists():
        cached = json.loads(checkpoint.read_text(encoding="utf-8"))
        if cached["identity"] == fingerprint:
            print(f"CHECKPOINT {name} {seed}", flush=True)
            return
    if ratio is not None:
        gate = json.loads((LOCAL / "replica_original_1000.json").read_text(encoding="utf-8"))
        if gate["identity"] != fingerprint or not gate["reference_reproduced"]:
            raise ValueError("Reference gate required before subsampling")
    start = time.monotonic()
    raw, x, y, below, original = load_inputs(source)
    probabilities, fold, effective = undersampled_oof(
        x, y, raw.REFREC.str.strip().to_numpy(), ratio=ratio, removal_seed=seed)
    np.testing.assert_array_equal(fold, original.fold.to_numpy())
    max_error = None
    if ratio is None:
        max_error = float(np.max(np.abs(probabilities["isotonic"] - original.p_cal_ml.to_numpy())))
        np.testing.assert_allclose(probabilities["isotonic"], original.p_cal_ml.to_numpy(), rtol=0, atol=1e-14)
    metrics, curves = evaluate(
        y, probabilities, pair_ids(raw).to_numpy(), below,
        raw.PASSO.str.strip().to_numpy(), _recode_sex(raw).fillna("MISSING").to_numpy(),
        ratio=name, removal_seed=seed)
    atomic_json(checkpoint, {"identity": fingerprint, "ratio": name, "seed": seed,
                            "seconds": time.monotonic() - start,
                            "reference_reproduced": ratio is None,
                            "reference_max_abs_error": max_error,
                            "effective_training": effective,
                            "metrics": serializable_records(metrics),
                            "curve": serializable_records(curves)})
    print(f"COMPLETE {name} {seed} seconds={time.monotonic()-start:.2f} max_error={max_error}", flush=True)


def aggregate(source, analysis_date):
    fingerprint = identity(source)
    checkpoints = [LOCAL / "replica_original_1000.json"] + [
        LOCAL / f"replica_{ratio}_{seed}.json" for ratio in (1, 5, 20, 50) for seed in range(1000, 1010)]
    replicas = [json.loads(p.read_text(encoding="utf-8")) for p in checkpoints]
    if any(replica["identity"] != fingerprint for replica in replicas):
        raise ValueError("Replica fingerprint mismatch")
    if not replicas[0]["reference_reproduced"]:
        raise ValueError("Original reference not reproduced")
    metrics = pd.DataFrame([r for replica in replicas for r in replica["metrics"]])
    curves = pd.DataFrame([r for replica in replicas for r in replica["curve"]])
    summary, curve = summarize(metrics, curves)
    output = ROOT / "data"
    for name, table in (("treino", metrics), ("treino_resumo", summary), ("treino_curva", curve)):
        if "pair_id" in table.columns:
            raise ValueError("Pair identifiers forbidden in outputs")
        table.to_csv(output / f"subamostragem_{name}.csv", index=False, lineterminator="\n", float_format="%.17g")
    metadata = {"analysis": "exploratory_post_hoc", "registered": False,
                "date": analysis_date, **fingerprint,
                "warning": "subamostragem só no treino; avaliação na prevalência real",
                "api_calls": 0, "llm_calls": 0, "fold_seed": 42, "folds": "REFREC 5x5",
                "removal_seeds": list(range(1000, 1010)), "ratios": ["original", "1:1", "1:5", "1:20", "1:50"],
                "reference_max_abs_error": replicas[0]["reference_max_abs_error"],
                "evaluation": {"below": 61485, "below_PAR2": 36, "full": 61696, "full_links": 247},
                "prior_source": {"doi": "10.1109/SSCI.2015.33", "url": SOURCE_URL,
                                 "pdf_sha256": "263540d93a6854c0bcc59d5d86396b4e030d435683b7ff83f57327ea1a9a7055",
                                 "physical_page": 3, "equation": 9, "beta": "retained negatives / original training negatives"},
                "variants": {"isotonic": "Original nested calibration on unsampled inner OOF labels",
                             "raw": "Sampled-training XGBoost probabilities before isotonic",
                             "raw_prior": "Equation (9) on raw probabilities, using actual outer-fit beta",
                             "isotonic_prior": "Diagnostic only: equation (9) after real-prevalence isotonic; may double-correct"},
                "scale_pos_weight": "Original model_factory: actual training negatives / positives (also after sampling)",
                "prior_limitation": "Class-weighted XGBoost is not guaranteed to estimate the sampled posterior; no calibration improvement assumed",
                "auc_pr": "Average precision (non-interpolated PR area), not trapezoidal PR AUC",
                "ece": "10 probability quantile bins, duplicate edges removed; constant probabilities one bin; ties unsplit",
                "ranking": "Descending probability, ascending lexical pair hash on ties; exact k; hashes never output",
                "subgroups": "Global below-cutoff top-96, not top-96 per group; same sex recode as equidade_arranjo/equidade_v31; MISSING or n<30 sensitivity undefined, coverage retained",
                "limitations": "Post hoc; 36 positives below cutoff; labels share verification/incorporation bias; no inferential or causal claim",
                "total_worker_seconds": sum(r["seconds"] for r in replicas),
                "training_counts": [{"ratio": r["ratio"], "seed": r["seed"], "fits": r["effective_training"]} for r in replicas]}
    metadata["output_sha256"] = {p.name: sha256(p) for p in sorted(output.glob("subamostragem_treino*.csv"))}
    atomic_json(output / "subamostragem_treino_metadata.json", metadata)
    print(summary.loc[summary.population.eq("below") & summary.variant.eq("isotonic")].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "worker", "aggregate"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ratio", choices=["original", "1", "5", "20", "50"], default="original")
    parser.add_argument("--seed", type=int, choices=range(1000, 1010), default=1000)
    parser.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    args = parser.parse_args()
    versions()
    if args.action == "prepare":
        preflight(args.source)
    elif args.action == "worker":
        worker(args.source, None if args.ratio == "original" else int(args.ratio), args.seed)
    else:
        aggregate(args.source, args.date)


if __name__ == "__main__":
    main()
