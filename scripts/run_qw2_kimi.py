"""Quick Win 2 — Kimi-only LLM review expansion.

Runs the LLM review protocol only with Fireworks Kimi, with deterministic
sampling, strict binary decisions, checkpointing, and raw response capture.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gzcmd.llm_dossier import Dossier, build_dossiers, make_pair_id
from gzcmd.llm_review import (
    build_arbiter_messages,
    build_review_messages,
    extract_and_validate,
)
from gzcmd.runner import run_v3


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


DATA_CSV = Path("data/COMPARADORSEMIDENT.csv")
CONFIG_YAML = Path("gzcmd/gzcmd_v3_config.yaml")
OUT_DIR = Path("data/qw2")


MODEL = "accounts/fireworks/models/kimi-k2p5"
BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = 40
MAX_TOKENS = 4096
SEED = 43
MAX_RETRIES = 3
RETRY_DELAY_S = 2.0


@dataclass
class AttemptRecord:
    pair_id: str
    agent: str
    attempt: int
    raw: str
    parsed: dict[str, Any] | None
    error: str | None
    latency_s: float
    changed_from_first: bool
    succeeded: bool


def _require_api_key() -> str:
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY not set")
    return api_key


def _binary_decision(value: Any) -> str:
    if str(value).strip().upper() == "MATCH":
        return "MATCH"
    return "NONMATCH"


def _coerce_confidence(value: Any) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if value_f != value_f:
        return 0.0
    return round(max(0.0, min(1.0, value_f)), 4)


def _normalize_binary_payload(payload: dict[str, Any], pair_id: str) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "decision": _binary_decision(payload.get("decision")),
        "confidence": _coerce_confidence(payload.get("confidence", 0.0)),
        "reason_codes": payload.get("reason_codes", []),
        "evidence_summary": payload.get("evidence_summary", {}),
        "quality_flags": payload.get("quality_flags", {}),
    }


def _call_api_with_retries(
    *,
    api_key: str,
    messages: list[dict[str, str]],
    pair_id: str,
    agent: str,
) -> tuple[dict[str, Any] | None, list[AttemptRecord], bool]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "seed": SEED,
        "response_format": {"type": "json_object"},
    }

    attempts: list[AttemptRecord] = []
    first_signature: tuple[str, float] | None = None
    retry_changed = False

    for attempt_no in range(1, MAX_RETRIES + 1):
        t0 = time.perf_counter()
        raw_text = ""
        parsed: dict[str, Any] | None = None
        error_msg: str | None = None
        changed_from_first = False

        try:
            response = requests.post(
                BASE_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            latency = round(time.perf_counter() - t0, 3)
            if response.status_code == 429:
                raise RuntimeError("rate_limit")
            response.raise_for_status()

            body = response.json()
            raw_text = body["choices"][0]["message"]["content"]
            parsed = _normalize_binary_payload(
                extract_and_validate(raw_text, pair_id),
                pair_id,
            )

            signature = (parsed["decision"], parsed["confidence"])
            if first_signature is None:
                first_signature = signature
            elif signature != first_signature:
                retry_changed = True
                changed_from_first = True

            attempts.append(
                AttemptRecord(
                    pair_id=pair_id,
                    agent=agent,
                    attempt=attempt_no,
                    raw=raw_text,
                    parsed=parsed,
                    error=None,
                    latency_s=latency,
                    changed_from_first=changed_from_first,
                    succeeded=True,
                )
            )
            return parsed, attempts, retry_changed

        except (requests.RequestException, ValueError, KeyError, RuntimeError) as exc:
            latency = round(time.perf_counter() - t0, 3)
            error_msg = str(exc)
            attempts.append(
                AttemptRecord(
                    pair_id=pair_id,
                    agent=agent,
                    attempt=attempt_no,
                    raw=raw_text,
                    parsed=None,
                    error=error_msg,
                    latency_s=latency,
                    changed_from_first=changed_from_first,
                    succeeded=False,
                )
            )
            if attempt_no < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S * (2 ** (attempt_no - 1)))

    return None, attempts, retry_changed


def _call_agent(
    *, api_key: str, dossier: Dossier, role: str
) -> tuple[dict[str, Any] | None, list[AttemptRecord], bool]:
    messages = build_review_messages(dossier, agent_role=role, batch_mode=True)
    return _call_api_with_retries(
        api_key=api_key, messages=messages, pair_id=dossier.pair_id, agent=role
    )


def _call_arbiter(
    *,
    api_key: str,
    dossier: Dossier,
    parsed_a: dict[str, Any],
    parsed_b: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[AttemptRecord], bool]:
    messages = build_arbiter_messages(dossier, parsed_a, parsed_b)
    return _call_api_with_retries(
        api_key=api_key,
        messages=messages,
        pair_id=dossier.pair_id,
        agent="Arbiter",
    )


def _review_one(
    *,
    dossier: Dossier,
    api_key: str,
) -> tuple[str, dict[str, Any] | None, list[AttemptRecord], dict[str, Any] | None]:
    pair_id = dossier.pair_id
    all_attempts: list[AttemptRecord] = []

    parsed_a: dict[str, Any] | None
    parsed_b: dict[str, Any] | None
    parsed_arb: dict[str, Any] | None = None

    parsed_a, attempts_a, retry_a = _call_agent(
        api_key=api_key, dossier=dossier, role="Agent-A"
    )
    all_attempts.extend(attempts_a)

    parsed_b, attempts_b, retry_b = _call_agent(
        api_key=api_key, dossier=dossier, role="Agent-B"
    )
    all_attempts.extend(attempts_b)

    if parsed_a is None or parsed_b is None:
        error_messages: list[str] = []
        if parsed_a is None:
            error_messages.append("Agent-A failed")
        if parsed_b is None:
            error_messages.append("Agent-B failed")

        return (
            pair_id,
            None,
            all_attempts,
            {
                "pair_id": pair_id,
                "error": "; ".join(error_messages),
                "model": MODEL,
                "seed": SEED,
                "retry_changed": retry_a or retry_b,
            },
        )

    if parsed_a["decision"] == parsed_b["decision"]:
        final_decision = parsed_a["decision"]
        final_confidence = round(
            (parsed_a["confidence"] + parsed_b["confidence"]) / 2, 4
        )
        final_protocol = "consensus"
        retry_arb = False
    else:
        parsed_arb, attempts_arb, retry_arb = _call_arbiter(
            api_key=api_key,
            dossier=dossier,
            parsed_a=parsed_a,
            parsed_b=parsed_b,
        )
        all_attempts.extend(attempts_arb)

        if parsed_arb is None:
            return (
                pair_id,
                None,
                all_attempts,
                {
                    "pair_id": pair_id,
                    "error": "Arbiter failed",
                    "model": MODEL,
                    "seed": SEED,
                    "retry_changed": retry_a or retry_b,
                },
            )

        final_decision = parsed_arb["decision"]
        final_confidence = parsed_arb["confidence"]
        final_protocol = "arbiter"

    reason_codes = list(
        dict.fromkeys(
            list(parsed_a.get("reason_codes", []))
            + list(parsed_b.get("reason_codes", []))
            + list(parsed_arb.get("reason_codes", []) if parsed_arb else [])
        )
    )

    retry_changed = retry_a or retry_b or retry_arb

    result = {
        "pair_id": pair_id,
        "decision": final_decision,
        "confidence": final_confidence,
        "protocol": final_protocol,
        "agent_a_decision": parsed_a["decision"],
        "agent_b_decision": parsed_b["decision"],
        "arbiter_decision": parsed_arb["decision"] if parsed_arb else None,
        "reason_codes": reason_codes,
        "evidence_summary": {
            "agent_a": parsed_a.get("evidence_summary", {}),
            "agent_b": parsed_b.get("evidence_summary", {}),
            "arbiter": parsed_arb.get("evidence_summary", {}) if parsed_arb else {},
        },
        "quality_flags": {
            "agent_a": parsed_a.get("quality_flags", {}),
            "agent_b": parsed_b.get("quality_flags", {}),
            "arbiter": parsed_arb.get("quality_flags", {}) if parsed_arb else {},
        },
        "retry_changed": retry_changed,
        "agent_a_retry_changed": retry_a,
        "agent_b_retry_changed": retry_b,
        "arbiter_retry_changed": False,
        "attempt_count": len(all_attempts),
        "model": MODEL,
        "seed": SEED,
    }

    if parsed_arb is not None:
        result["arbiter_retry_changed"] = False
        for attempt in [x for x in attempts_arb if x.changed_from_first]:
            if attempt.succeeded:
                result["arbiter_retry_changed"] = True

    return pair_id, result, all_attempts, None


def _serialize_attempt(attempt: AttemptRecord) -> dict[str, Any]:
    return {
        "pair_id": attempt.pair_id,
        "agent": attempt.agent,
        "attempt": attempt.attempt,
        "succeeded": attempt.succeeded,
        "latency_s": attempt.latency_s,
        "changed_from_first": attempt.changed_from_first,
        "raw": attempt.raw,
        "parsed": attempt.parsed,
        "error": attempt.error,
    }


def _is_valid_target(value: Any) -> int | None:
    if value in (0, 1, 0.0, 1.0):
        return int(value)
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text in {"0", "1"}:
            return int(text)
        if text in {"0.0", "1.0"}:
            return int(float(text))
    return None


def _build_target_map(df: pd.DataFrame) -> dict[str, int | None]:
    target_map: dict[str, int | None] = {}
    llm_rows = df[df["action"] == "LLM_REVIEW"]
    for _, row in llm_rows.iterrows():
        pid = make_pair_id(
            str(row.get("COMPREC", "")),
            str(row.get("REFREC", "")),
            str(row.get("PASSO", "")),
        )
        target_map[pid] = _is_valid_target(row.get("TARGET"))
    return target_map


def _compute_metrics(
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    target_map: dict[str, int | None],
) -> dict[str, Any]:
    n_results = len(results)
    n_errors = len(errors)
    metrics: dict[str, Any] = {
        "n": n_results,
        "n_errors": n_errors,
        "decisions": {},
        "protocols": {},
        "consensus_rate": 0.0,
        "arbiter_rate": 0.0,
        "unsure_rate": 0.0,
        "retry_changed_rate": 0.0,
        "avg_confidence": 0.0,
        "gt_n": 0,
        "gt_decided_n": 0,
        "gt_accuracy": None,
        "gt_tp": 0,
        "gt_fp": 0,
        "gt_tn": 0,
        "gt_fn": 0,
    }

    if not results:
        return metrics

    decisions = Counter(r["decision"] for r in results)
    protocols = Counter(r["protocol"] for r in results)
    retry_changed_count = sum(1 for r in results if r.get("retry_changed"))

    metrics.update(
        {
            "decisions": dict(decisions),
            "protocols": dict(protocols),
            "consensus_rate": round(protocols.get("consensus", 0) / n_results, 4),
            "arbiter_rate": round(protocols.get("arbiter", 0) / n_results, 4),
            "unsure_rate": round(0.0, 4),
            "retry_changed_rate": round(retry_changed_count / n_results, 4),
            "avg_confidence": round(
                sum(r["confidence"] for r in results) / n_results, 4
            ),
        }
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    for row in results:
        target = target_map.get(row["pair_id"])
        if target is None:
            continue
        y_true.append(int(target))
        y_pred.append(1 if row["decision"] == "MATCH" else 0)

    if y_true:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        decided = tp + fp + tn + fn
        metrics.update(
            {
                "gt_n": len(y_true),
                "gt_decided_n": decided,
                "gt_tp": tp,
                "gt_fp": fp,
                "gt_tn": tn,
                "gt_fn": fn,
                "gt_accuracy": round((tp + tn) / decided, 4) if decided else None,
            }
        )

    return metrics


def _dump_raw_calls(raw_calls: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in raw_calls),
        encoding="utf-8",
    )


def _save_checkpoint(
    path: Path,
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    raw_calls: list[dict[str, Any]],
    mode: str,
    *,
    n_review_pairs: int,
    elapsed_s: float,
) -> None:
    payload = {
        "experiment": "qw2_kimi_batch_expand",
        "mode": mode,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_review_pairs": n_review_pairs,
        "n_completed": len(results) + len(errors),
        "n_results": len(results),
        "n_errors": len(errors),
        "elapsed_s": round(elapsed_s, 1),
        "results": results,
        "errors": errors,
        "raw_calls": raw_calls,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_mode(mode: str, args: argparse.Namespace, api_key: str) -> None:
    log.info("[MODE=%s] running v3 pipeline", mode)
    df, _summary = run_v3(
        input_csv=str(DATA_CSV),
        config_path=str(CONFIG_YAML),
        mode=mode,
        p_cal="fit_platt",
    )

    dossiers = build_dossiers(df, only_llm_review=True)
    if args.limit > 0 and len(dossiers) > args.limit:
        dossiers = dossiers[: args.limit]

    target_map = _build_target_map(df)

    out_mode_dir = OUT_DIR
    out_mode_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_mode_dir / f"qw2_{mode}_kimi.json"
    raw_path = out_mode_dir / f"qw2_{mode}_kimi_raw.jsonl"
    csv_path = out_mode_dir / f"qw2_{mode}_kimi.csv"
    checkpoint_path = out_mode_dir / f"qw2_{mode}_kimi_checkpoint.json"

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    raw_calls: list[dict[str, Any]] = []
    done_ids: set[str] = set()

    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            ckpt = json.load(f)
        results = ckpt.get("results", [])
        errors = ckpt.get("errors", [])
        raw_calls = ckpt.get("raw_calls", [])
        done_ids = {r["pair_id"] for r in results} | {
            e["pair_id"] for e in errors if "pair_id" in e
        }
        log.info(
            "[MODE=%s] resumed from checkpoint: %d results, %d errors",
            mode,
            len(results),
            len(errors),
        )

    pending = [d for d in dossiers if d.pair_id not in done_ids]
    n_review_pairs = len(dossiers)
    if not pending:
        completion = (len(results) + len(errors)) / max(len(dossiers), 1)
        metrics = _compute_metrics(results, errors, target_map)
        metrics["completion_rate"] = round(completion, 4)
        if completion < 0.95:
            metrics["completion_note"] = (
                "completed below 95%; results are partial and should be resumed"
            )
        final_payload = {
            "experiment": "qw2_kimi_batch_expand",
            "mode": mode,
            "model": MODEL,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seed": SEED,
            "n_review_pairs": n_review_pairs,
            "n_reviewed": len(results),
            "n_errors": len(errors),
            "metrics": metrics,
            "results": results,
            "errors": errors,
            "raw_calls_path": str(raw_path),
            "completion_note": metrics.get("completion_note"),
        }
        final_path.write_text(
            json.dumps(final_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _dump_raw_calls(raw_calls, raw_path)
        _write_csv_results(csv_path, results)
        log.info("[MODE=%s] already complete (or resumed): %s", mode, final_path)
        return

    log.info(
        "[MODE=%s] pipeline done: %d total candidates, %d pending",
        mode,
        len(dossiers),
        len(pending),
    )

    start_ts = time.perf_counter()
    completed_since_ckpt = 0
    total_pending = len(pending)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_review_one, dossier=d, api_key=api_key): d.pair_id
            for d in pending
        }

        for i, future in enumerate(as_completed(futures), start=1):
            pair_id, result, attempts, error_record = future.result()
            for attempt in attempts:
                raw_calls.append(_serialize_attempt(attempt))

            if result is not None:
                result["mode"] = mode
                results.append(result)
            else:
                if error_record is not None:
                    errors.append(error_record)
                else:
                    errors.append({"pair_id": pair_id, "error": "unknown"})

            completed_since_ckpt += 1

            if completed_since_ckpt >= max(1, args.checkpoint_interval):
                elapsed = time.perf_counter() - start_ts
                _dump_raw_calls(raw_calls, raw_path)
                _save_checkpoint(
                    checkpoint_path,
                    results=results,
                    errors=errors,
                    raw_calls=raw_calls,
                    mode=mode,
                    n_review_pairs=n_review_pairs,
                    elapsed_s=elapsed,
                )
                completed_since_ckpt = 0

            if i % 10 == 0 or i == total_pending:
                elapsed = time.perf_counter() - start_ts
                log.info(
                    "[MODE=%s] %d/%d done (%.1fs)",
                    mode,
                    i,
                    total_pending,
                    elapsed,
                )

    elapsed = time.perf_counter() - start_ts
    completion = (len(results) + len(errors)) / max(n_review_pairs, 1)
    metrics = _compute_metrics(results, errors, target_map)
    metrics["completion_rate"] = round(completion, 4)
    if completion < 0.95:
        metrics["completion_note"] = (
            "completed below 95%; partial fallback dataset. "
            "Resume from checkpoint for full coverage."
        )

    _dump_raw_calls(raw_calls, raw_path)

    final_payload = {
        "experiment": "qw2_kimi_batch_expand",
        "mode": mode,
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "n_review_pairs": n_review_pairs,
        "n_reviewed": len(results),
        "n_errors": len(errors),
        "elapsed_s": round(elapsed, 1),
        "metrics": metrics,
        "results": results,
        "errors": errors,
        "raw_calls_path": str(raw_path),
    }
    final_payload["completion_note"] = metrics.get("completion_note")

    final_path.write_text(
        json.dumps(final_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv_results(csv_path, results)

    if completion >= 0.95:
        checkpoint_path.unlink(missing_ok=True)

    for k, v in metrics.items():
        if k in {"decisions", "protocols", "gt_accuracy"}:
            continue
        log.info("[MODE=%s] %s: %s", mode, k, v)

    log.info("[MODE=%s] saved: %s", mode, final_path)
    log.info("[MODE=%s] raw calls: %s", mode, raw_path)


def _write_csv_results(path: Path, results: list[dict[str, Any]]) -> None:
    if not results:
        path.unlink(missing_ok=True)
        return

    fieldnames = [
        "pair_id",
        "mode",
        "decision",
        "confidence",
        "protocol",
        "agent_a_decision",
        "agent_b_decision",
        "arbiter_decision",
        "reason_codes",
        "retry_changed",
        "agent_a_retry_changed",
        "agent_b_retry_changed",
        "arbiter_retry_changed",
        "attempt_count",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            row_out = dict(row)
            row_out["reason_codes"] = ";".join(row_out.get("reason_codes", []))
            writer.writerow({k: row_out.get(k) for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick Win 2: run Kimi review expansion"
    )
    parser.add_argument(
        "--mode",
        choices=["vigilancia", "confirmacao", "both"],
        default="both",
        help="Run one mode or both modes",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit pairs per mode")
    parser.add_argument("--workers", type=int, default=6, help="Thread pool workers")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=25,
        help="Write checkpoint every N completions",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing mode checkpoints",
    )
    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.checkpoint_interval < 1:
        raise ValueError("--checkpoint-interval must be >= 1")

    api_key = _require_api_key()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.resume:
        for mode in ("vigilancia", "confirmacao"):
            checkpoint = OUT_DIR / f"qw2_{mode}_kimi_checkpoint.json"
            if checkpoint.exists():
                checkpoint.unlink()

    modes = ["vigilancia", "confirmacao"] if args.mode == "both" else [args.mode]
    for mode in modes:
        _run_mode(mode, args, api_key)


if __name__ == "__main__":
    main()
