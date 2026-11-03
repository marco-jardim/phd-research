"""Quota-gated sequential launcher for the historical DeepSeek ablation cohorts.

No credentials, dossiers or raw API responses cross the persistence boundary.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_exp2_sem_model_outputs as sem


def quota_probe(out, *, sleep=time.sleep):
    import requests
    import run_exp2_opencode_go as go
    key = os.environ.get("OPENCODE_GO_API_KEY")
    if not key:
        raise ValueError("Missing OPENCODE_GO_API_KEY (value suppressed)")
    transport = go._RequestsWithSession(requests, go._session_id("deepseek-sem-mo-quota"),
                                        go.EXTRA_BY_MODEL[go.MODEL])
    attempts = []
    for attempt in range(2):
        started = time.monotonic()
        response = transport.post(go.BASE_URL, headers={"Authorization": f"Bearer {key}"},
                                  json={"model": go.MODEL, "messages": [{"role": "user", "content": 'Return {"ok":true}'}],
                                        **{k: v for k, v in go.GEN_PARAMS.items() if k not in {"timeout_s", "max_retries", "max_tokens"}},
                                        "max_tokens": 16, "response_format": {"type": "json_object"}}, timeout=120)
        status = response.status_code
        # Go can disguise exhausted quota as HTTP 400; never save its body.
        quota = status in (400, 402, 429)
        attempts.append({"status": status, "elapsed_s": round(time.monotonic() - started, 2)})
        sem.save(out / "quota_probe.json", {"model": go.MODEL, "timestamp": sem.utcnow(), "attempts": attempts,
                                           "released": status == 200, "retry_wait_s": 900 if attempt else 0})
        print(f"QUOTA_PROBE attempt={attempt + 1} status={status}", flush=True)
        if status == 200:
            return True
        if not quota or attempt:
            return False
        # One retry only, after the maximum requested backoff (15 minutes).
        sleep(900)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    sem.configure_provider("opencode-go")
    out = sem.OUT
    out.mkdir(parents=True, exist_ok=True)
    if not quota_probe(out):
        print("STOP quota unavailable after probe; no cohort inference started", flush=True)
        return 2
    if args.probe_only:
        return 0
    started = time.monotonic()
    exit_codes = {}
    for mode, expected in (("vigilancia", 1410), ("confirmacao", 435)):
        source = sem.original_path(mode)
        if len(sem.load_ids(source)) != expected:
            raise ValueError("Historical cohort size differs from dispatch")
        command = [sys.executable, str(ROOT / "scripts/run_exp2_sem_model_outputs.py"),
                   "--provider", "opencode-go", "--modo", mode, "--pares", str(source),
                   "--saida", str(out / f"exp2_llm_sync_{mode}.json"), "--workers", "4"]
        with (out / f"run_{mode}.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log, check=False)
        exit_codes[mode] = result.returncode
        print(f"MODE_END {mode} exit={result.returncode}", flush=True)
        if result.returncode:
            break
    sem.save(out / "execucao.json", {"timestamp": sem.utcnow(), "wall_seconds": round(time.monotonic() - started, 2),
                                      "exit_codes": exit_codes, "workers": 4, "modes_sequential": True})
    if all((out / f"exp2_llm_sync_{m}.json").exists() for m in ("vigilancia", "confirmacao")):
        sem.summarize()
    return 1 if any(exit_codes.values()) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("FATAL " + type(exc).__name__, flush=True)
        sys.exit(1)
