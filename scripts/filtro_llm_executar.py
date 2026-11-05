"""Execução autorizada e limitada das pendências do filtro, sem recalcular taxas."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.arranjo_corte import MODEL, sha256
from gzcmd.llm_usage_cost import price_usage
from scripts.arranjo_corte_s2_executar import ps_quote
from scripts import filtro_llm as filtro

OUT_NAME = "data/sprint3b/anthropic_sem_mo_v4_filtro_llm"
OUT = ROOT / OUT_NAME
LOCAL = filtro.LOCAL
PROTOCOL_NAME = "gzcmd/protocolo_filtro_llm.yaml"


def budget_gate(counts, cost):
    if sum(counts.values()) > 100 or cost > .50:
        raise ValueError("STOP: limite de 100 pares ou US$ 0,50 excedido")


def pending_gate(note, *, snapshot=False):
    counts = {}
    for mode in filtro.MODES:
        path = LOCAL / f"{'aprovados' if snapshot else 'pendentes'}_{mode}.csv"
        if sha256(path) != note["pending_sha256"][mode]:
            raise ValueError("Pending list differs from author-approved SHA256")
        frame = pd.read_csv(path, dtype=str)
        if list(frame) != ["pair_id"] or not frame.pair_id.is_unique:
            raise ValueError("Invalid pending CSV schema/uniqueness")
        if not frame.pair_id.str.fullmatch(r"[0-9a-f]{16}").all():
            raise ValueError("Invalid pending identifier format")
        counts[mode] = len(frame)
    if counts != note["pending_counts"]:
        raise ValueError("Pending counts differ from authorization")
    budget_gate(counts, sum(counts.values()) * .00316)
    return counts


def amend():
    spec = yaml.safe_load(filtro.PROTOCOL.read_text(encoding="utf-8"))
    if spec.get("amendments"):
        raise ValueError("Amendment already recorded; refusing overwrite")
    note = {
        "number": 1, "date": datetime.now(timezone.utc).isoformat(),
        "authorization": "Autor aprovou explicitamente as 83 chamadas de pares pendentes, cerca de US$ 0,26, para completar cobertura até k=125 nos dois modos.",
        "overrides": "api_calls_allowed=false aplica-se à fase inicial offline; esta emenda autoriza somente as listas fixadas abaixo.",
        "pending_counts": {"vigilancia": 12, "confirmacao": 71},
        "pending_sha256": {mode: sha256(LOCAL / f"pendentes_{mode}.csv") for mode in filtro.MODES},
        "pending_paths": {mode: f".local/filtro_llm/pendentes_{mode}.csv" for mode in filtro.MODES},
        "runner": "scripts/run_exp2_sem_model_outputs.py",
        "runner_sha256": sha256(ROOT / "scripts/run_exp2_sem_model_outputs.py"),
        "arguments": "--provider anthropic --modelo claude-haiku-5-5 --prompt-version v4 --allow-new-cohort --pares <CSV aprovado do modo> --modo <modo> --saida-dir data/sprint3b/anthropic_sem_mo_v4_filtro_llm --workers 4",
        "dossier": "Mesmo runner histórico, dossiê sem model_outputs; validar prompt e dossier_sha256 contra reconstrução.",
        "output_directory": OUT_NAME,
        "response_precedence": "Coortes congeladas (a resposta usada por P), S2, suplemento, robustez e por último as respostas novas desta emenda.",
        "stop_rule": {"max_pairs": 100, "max_estimated_usd": .50,
                      "action": "Interromper e relatar se exceder qualquer limite; monitorar custo dos journals durante o processo; nunca iniciar o modo seguinte após falha."},
        "acceptance": "complete=true, zero erros por par/HTTP/parsing, secret_matches=0 e identidade validada",
        "estimated_usd_no_cache": round(83 * .00316, 8),
        "new_responses_reestimate_rates": False,
        "new_responses_feed_other_arms": False,
        "temporal_stochastic_limitation": "As respostas novas têm outra data de execução e o modelo é estocástico; o filtro mistura respostas antigas e novas. Não é uma avaliação prospectiva independente.",
        "execution_code_sha256": sha256(Path(__file__)),
    }
    pending_gate(note)
    for mode in filtro.MODES:
        shutil.copyfile(LOCAL / f"pendentes_{mode}.csv", LOCAL / f"aprovados_{mode}.csv")
    spec["amendments"] = [note]
    filtro.PROTOCOL.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
    print(json.dumps({"protocol_sha256": sha256(filtro.PROTOCOL), "pending_counts": note["pending_counts"],
                      "pending_sha256": note["pending_sha256"], "estimated_usd": note["estimated_usd_no_cache"]}, indent=2))


def verify_authorization():
    spec, commit = filtro.verify_protocol(Path("D:/git/phd-research"))
    note = spec["amendments"][-1]
    if note["runner_sha256"] != sha256(ROOT / note["runner"]):
        raise ValueError("Historical runner changed")
    if note["execution_code_sha256"] != sha256(Path(__file__)):
        raise ValueError("Authorized execution code changed")
    counts = pending_gate(note, snapshot=True)
    return note, commit, counts


def run_mode(mode):
    args = [str(ROOT / "scripts/run_exp2_sem_model_outputs.py"), "--provider", "anthropic",
            "--modelo", MODEL, "--prompt-version", "v4", "--allow-new-cohort",
            "--saida-dir", str(OUT), "--modo", mode, "--pares",
            str(LOCAL / f"aprovados_{mode}.csv"), "--workers", "4"]
    argument_line = " ".join('"' + a.replace('"', '\\"') + '"' for a in args)
    # The historical runner is unmodified. Monitor its durable usage journals
    # while Wait-Process waits, and kill only this child if the cost cap is crossed.
    ps = "\n".join([
        "$ErrorActionPreference = 'Stop'",
        "$p = Start-Process -FilePath " + ps_quote(sys.executable)
        + " -ArgumentList " + ps_quote(argument_line) + " -WorkingDirectory " + ps_quote(ROOT)
        + " -RedirectStandardOutput " + ps_quote(LOCAL / f"pendentes_{mode}.stdout.log")
        + " -RedirectStandardError " + ps_quote(LOCAL / f"pendentes_{mode}.stderr.log") + " -PassThru",
        "try {",
        "while (-not $p.HasExited) {",
        "  Wait-Process -InputObject $p -Timeout 2 -ErrorAction SilentlyContinue",
        "  $cost = 0.0",
        "  Get-ChildItem -LiteralPath " + ps_quote(OUT) + " -Filter '*.transport.json' -Recurse | ForEach-Object {",
        "    try { $u = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json; $cost += ($u.input_tokens * 0.10 + $u.output_tokens * 0.50 + $u.cache_read_input_tokens * 0.01 + $u.cache_creation_input_tokens * 0.125) / 1000000 } catch { throw 'Usage journal unreadable; stop for safety' }",
        "  }",
        "  if ($cost -gt 0.50) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; throw 'STOP: cost exceeds USD 0.50' }",
        "  $p.Refresh()",
        "}",
        "$p.WaitForExit()", "exit $p.ExitCode",
        "} finally { if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }",
    ])
    ps_path = LOCAL / f"chamadas_{mode}.ps1"
    ps_path.write_text(ps, encoding="utf-8")
    completed = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-File", str(ps_path)], cwd=ROOT)
    if completed.returncode:
        raise RuntimeError(f"Historical runner failed for {mode}; inspect private logs")
    run_path = OUT / f"exp2_llm_sync_{mode}.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("complete") is not True or run.get("n_errors") != 0:
        raise ValueError("Run incomplete or pair errors")
    if run["transport"]["http_errors"] or run["transport"]["response_parse_errors"]:
        raise ValueError("Transport/parsing errors; do not start next mode")
    keys = ("mode", "model", "prompt_sha256", "prompt_version", "removed_fields", "timestamp",
            "updated_at", "n_review_pairs", "n_reviewed", "n_errors", "complete", "workers",
            "elapsed_s", "metadata", "transport", "cost_estimate_usd", "cost_note")
    result = {key: run[key] for key in keys}
    result["run_sha256"] = sha256(run_path)
    return result


def audit_secrets():
    secrets = [value for name in ("ANTHROPIC_API_KEY", "OPENCODE_GO_API_KEY", "SAGE_API_KEY")
               if (value := os.environ.get(name))]
    if not secrets:
        raise ValueError("Credentials required for secret audit; values suppressed")
    files = [p for p in OUT.rglob("*") if p.is_file()]
    files.extend(LOCAL.glob("pendentes_*.log"))
    matches = sum(any(secret in p.read_text(encoding="utf-8") for secret in secrets) for p in files)
    if matches:
        raise ValueError("Secret audit failed; values suppressed")
    from scripts.run_exp2_sem_model_outputs import ALLOWED_RESULT, VALID_REASON_CODES
    for mode in filtro.MODES:
        run = json.loads((OUT / f"exp2_llm_sync_{mode}.json").read_text(encoding="utf-8"))
        for row in run["results"]:
            if not set(row) <= ALLOWED_RESULT or not set(row["reason_codes"]) <= VALID_REASON_CODES:
                raise ValueError("Unexpected free-response field/reason code")
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "files_scanned": len(files),
              "secret_matches": 0, "free_response_fields_saved": False,
              "dossiers_saved": False, "result_schema_allowlist_valid": True}
    (OUT / "auditoria.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def execute():
    note, commit, counts = verify_authorization()
    from dotenv import load_dotenv
    env = ROOT / ".env"
    if not env.exists():
        env = Path("D:/git/phd-research/.env")
    load_dotenv(env)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("Missing API credential; value suppressed")
    source = ROOT / "data/COMPARADORSEMIDENT.csv"
    if not source.exists():
        shutil.copyfile(filtro.resolve(Path("D:/git/phd-research"), "data/COMPARADORSEMIDENT.csv"), source)
    frozen_rates = ROOT / "data/arranjo_corte_taxas_crossfit.csv"
    rates_sha = sha256(frozen_rates)
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for mode in filtro.MODES:
        budget_gate(counts, sum(r["cost_estimate_usd"] for r in results))
        result = run_mode(mode)
        if result["n_reviewed"] != counts[mode]:
            raise ValueError("Executed count differs from approval")
        results.append(result)
        budget_gate(counts, sum(r["cost_estimate_usd"] for r in results))
    pending_gate(note, snapshot=True)
    if sha256(frozen_rates) != rates_sha:
        raise ValueError("Reviewer rates changed during execution")
    report = {"stage": "filtro_llm_authorized_pending_only", "amendment_commit": commit,
              "protocol_sha256": sha256(filtro.PROTOCOL), "runs": results,
              "total_pairs": sum(r["n_reviewed"] for r in results),
              "total_api_calls": sum(r["transport"]["calls"] for r in results),
              "total_cost_estimate_usd": sum(price_usage(r["transport"]) for r in results),
              "frozen_rates_sha256": rates_sha, "new_responses_reestimate_rates": False,
              "new_responses_feed_other_arms": False,
              "temporal_stochastic_limitation": note["temporal_stochastic_limitation"]}
    (OUT / "execucao.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["secret_audit"] = audit_secrets()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amend", action="store_true")
    args = parser.parse_args()
    try:
        amend() if args.amend else execute()
    except Exception as exc:
        print("FILTRO_EXECUTION_STOP", type(exc).__name__, flush=True)
        sys.exit(1)
