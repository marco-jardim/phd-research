"""Gate S2 on the frozen S1, execute sequential Windows processes, audit usage.

No outcomes or cascade correctness metrics are computed. Per-pair results stay
ignored; only aggregate execution metadata and a secret audit are published.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gzcmd.arranjo_corte import MODEL, pending_cost, sha256
from scripts.arranjo_corte_s1_simular import LOCAL, MODES, verify_prereg

OUT = ROOT / "data/sprint3b/anthropic_sem_mo_v4_arranjo_corte"


def gate(metadata):
    counts = {}
    for mode in MODES:
        path = LOCAL / f"pendentes_{mode}.csv"
        if sha256(path) != metadata["pending_sha256"][mode]:
            raise ValueError("Pending IDs differ from frozen S1")
        import csv
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["pair_id"]:
                raise ValueError("Pending CSV must contain pair_id only")
            ids = [row["pair_id"] for row in reader]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate pending IDs")
        counts[mode] = len(ids)
    n = sum(counts.values())
    if counts != metadata["pending_P_union_B1_by_mode"] or n != metadata["pending_total"]:
        raise ValueError("Pending counts differ from frozen S1")
    if metadata["stop_rule_triggered"] or n > 500 or pending_cost(n) > 2:
        raise ValueError("STOP: author approval required; no API calls allowed")
    if sha256(ROOT / "data/arranjo_corte_taxas_crossfit.csv") != metadata["frozen_rates_sha256"]:
        raise ValueError("Frozen reviewer rates changed")
    if sha256(ROOT / "data/arranjo_corte_s1_resumo.csv") != metadata["summary_sha256"]:
        raise ValueError("Frozen S1 summary changed")
    return counts


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def run_mode(mode):
    args = [str(ROOT / "scripts/run_exp2_sem_model_outputs.py"), "--provider", "anthropic",
            "--modelo", MODEL, "--prompt-version", "v4", "--allow-new-cohort",
            "--saida-dir", str(OUT), "--modo", mode, "--pares",
            str(LOCAL / f"pendentes_{mode}.csv"), "--workers", "4"]
    # Start-Process joins ArgumentList; double-quote paths for the child parser.
    argument_line = " ".join('"' + a.replace('"', '\\"') + '"' for a in args)
    script = "\n".join([
        "$ErrorActionPreference = 'Stop'",
        "$process = Start-Process -FilePath " + ps_quote(sys.executable)
        + " -ArgumentList " + ps_quote(argument_line)
        + " -WorkingDirectory " + ps_quote(ROOT)
        + " -RedirectStandardOutput " + ps_quote(LOCAL / f"s2_{mode}.stdout.log")
        + " -RedirectStandardError " + ps_quote(LOCAL / f"s2_{mode}.stderr.log")
        + " -PassThru",
        "if (-not $process.HasExited) { Wait-Process -InputObject $process }",
        "$process.WaitForExit()",
        "exit $process.ExitCode",
    ])
    path = LOCAL / f"s2_{mode}.ps1"
    path.write_text(script, encoding="utf-8")
    started = time.monotonic()
    completed = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-File", str(path)], cwd=ROOT)
    if completed.returncode:
        raise RuntimeError(f"S2 process failed: {mode}, exit={completed.returncode}; sanitized logs in .local")
    run_path = OUT / f"exp2_llm_sync_{mode}.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("complete") is not True or run.get("n_errors") != 0:
        raise ValueError(f"S2 incomplete or pair errors: {mode}")
    transport = run["transport"]
    if transport["http_errors"] or transport["response_parse_errors"]:
        raise ValueError(f"S2 transport or parsing errors: {mode}")
    allowed = ("mode", "model", "prompt_sha256", "prompt_version", "removed_fields",
               "timestamp", "updated_at", "n_review_pairs", "n_reviewed", "n_errors",
               "complete", "workers", "elapsed_s", "metadata", "transport",
               "cost_estimate_usd", "cost_note")
    result = {k: run[k] for k in allowed}
    result["run_sha256"] = sha256(run_path)
    result["wall_seconds"] = round(time.monotonic() - started, 3)
    return result


def audit_secrets():
    secrets = [value for name in ("ANTHROPIC_API_KEY", "OPENCODE_GO_API_KEY", "SAGE_API_KEY")
               if (value := os.environ.get(name))]
    if not secrets:
        raise ValueError("Secret audit needs loaded credentials (values suppressed)")
    files = [p for p in OUT.rglob("*") if p.is_file()]
    files += list(LOCAL.glob("s2_*.log"))
    matches = sum(any(secret in p.read_text(encoding="utf-8") for secret in secrets) for p in files)
    if matches:
        raise ValueError("Secret audit failed (values suppressed)")
    result = {"timestamp": datetime.now(timezone.utc).isoformat(), "files_scanned": len(files),
              "secret_matches": 0, "free_response_fields_saved": False, "dossiers_saved": False}
    (OUT / "auditoria.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--prereg-sha256", required=True)
    args = parser.parse_args()
    _, commit = verify_prereg(args.data_root, args.prereg_sha256)
    metadata = json.loads((ROOT / "data/arranjo_corte_s1_metadata.json").read_text(encoding="utf-8"))
    if metadata["prereg_sha256"] != args.prereg_sha256 or metadata["prereg_commit"] != commit:
        raise ValueError("S1 preregistration identity mismatch")
    counts = gate(metadata)
    if not sum(counts.values()):
        print("S2_SKIPPED no pending pairs")
        return
    from dotenv import load_dotenv
    load_dotenv(args.env_file)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("Missing ANTHROPIC_API_KEY (value suppressed)")
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for mode in MODES:
        if counts[mode]:
            results.append(run_mode(mode))
    gate(metadata)  # Responses never feed back into the frozen reviewer rates.
    report = {"stage": "S2_pending_only", "prereg_commit": commit, "prereg_sha256": args.prereg_sha256,
              "frozen_rates_sha256": metadata["frozen_rates_sha256"], "runs": results,
              "total_pairs": sum(r["n_reviewed"] for r in results),
              "total_api_calls": sum(r["transport"]["calls"] for r in results),
              "total_cost_estimate_usd": sum(r["cost_estimate_usd"] for r in results)}
    (OUT / "execucao.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["secret_audit"] = audit_secrets()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # No credential/provider text in this launcher's exception boundary either.
        print("S2_FATAL", type(exc).__name__, flush=True)
        sys.exit(1)
