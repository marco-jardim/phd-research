#!/usr/bin/env python
"""Exp 2 — LLM Batch Review of Grey-Zone Pairs via Fireworks Batch API.

Runs the full pipeline → identifies LLM_REVIEW pairs → builds dossiers →
submits dual-agent review as Fireworks batch jobs → measures outcomes.

Protocol:
  Phase 1: Submit Agent-A + Agent-B batch jobs in parallel (2 × N requests)
  Phase 2: Parse results, identify disagreements
  Phase 3: Submit Arbiter batch for disagreements only
  Phase 4: Assemble final decisions + compute metrics

Metrics:
  - Agent consensus rate (A+B agree without arbiter)
  - Decision distribution (MATCH / NONMATCH / UNSURE) per band
  - FP reduction: how many auto-MATCH would flip to NONMATCH/UNSURE
  - Comparison with ground truth (TARGET column) where available

Usage:
    python scripts/run_exp2_llm_batch.py [--mode vigilancia] [--dry-run] [--limit N]
    python scripts/run_exp2_llm_batch.py --resume data/exp2_checkpoint_vigilancia.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests as http_requests

# ── project imports ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gzcmd.config import load_config  # noqa: E402
from gzcmd.llm_dossier import Dossier, build_dossiers, make_pair_id  # noqa: E402
from gzcmd.llm_review import (  # noqa: E402
    build_arbiter_messages,
    build_review_messages,
    extract_and_validate,
)
from gzcmd.runner import run_v3  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_CSV = Path("data/COMPARADORSEMIDENT.csv")
CONFIG_YAML = Path("gzcmd/gzcmd_v3_config.yaml")
OUT_DIR = Path("data")

# ── Fireworks batch config ────────────────────────────────────────────
MODEL = "accounts/fireworks/models/kimi-k2p5"
MAX_TOKENS = 4096
TEMPERATURE = 0.0
TOP_P = 1.0
POLL_INTERVAL_S = 15
POLL_TIMEOUT_S = 7200  # 2 hours max

TERMINAL_STATES = frozenset(
    {
        "JOB_STATE_COMPLETED",
        "JOB_STATE_FAILED",
        "JOB_STATE_EXPIRED",
        "JOB_STATE_CANCELLED",
    }
)


# ── Fireworks batch helpers ───────────────────────────────────────────
@dataclass
class BatchRequest:
    """One request in a Fireworks batch job."""

    custom_id: str
    messages: list[dict[str, str]]

    def to_jsonl_row(self) -> dict:
        return {
            "custom_id": self.custom_id,
            "body": {
                "messages": self.messages,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            },
        }


def _get_fireworks_creds() -> tuple[str, str]:
    """Return (api_key, account_id)."""
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    account_id = os.environ.get("FIREWORKS_ACCOUNT_ID", "")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY not set")
    if not account_id:
        raise RuntimeError("FIREWORKS_ACCOUNT_ID not set")
    return api_key, account_id


def _write_jsonl(requests: list[BatchRequest], path: Path) -> None:
    """Write batch requests as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req.to_jsonl_row(), ensure_ascii=False) + "\n")


def _create_dataset(client: Any, dataset_id: str, example_count: int) -> str:
    """Create a Fireworks dataset and return its full name."""
    payload: dict[str, Any] = {
        "userUploaded": {},
        "exampleCount": str(max(example_count, 1)),
    }
    ds = client.datasets.create(
        dataset_id=dataset_id,
        dataset=payload,
    )
    return ds.name


def _upload_jsonl(client: Any, dataset_id: str, jsonl_path: Path) -> None:
    """Upload JSONL to dataset."""
    client.datasets.upload(dataset_id=dataset_id, file=jsonl_path)


def _create_batch_job(
    api_key: str,
    account_id: str,
    job_id: str,
    model: str,
    input_ds_name: str,
    output_ds_name: str,
) -> dict:
    """Create a batch inference job via Fireworks REST API."""
    url = (
        f"https://api.fireworks.ai/v1/accounts/{account_id}"
        f"/batchInferenceJobs?batchInferenceJobId={job_id}"
    )
    body = {
        "model": model,
        "inputDatasetId": input_ds_name,
        "outputDatasetId": output_ds_name,
        "inferenceParameters": {
            "maxTokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "topP": TOP_P,
        },
    }
    resp = http_requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def _poll_batch_job(api_key: str, account_id: str, job_id: str) -> str:
    """Poll until batch job reaches terminal state. Returns final state."""
    url = (
        f"https://api.fireworks.ai/v1/accounts/{account_id}/batchInferenceJobs/{job_id}"
    )
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.time() + POLL_TIMEOUT_S

    while time.time() < deadline:
        resp = http_requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        state = data.get("state", "UNKNOWN")
        log.info("  Batch %s: %s", job_id[:12], state)
        if state in TERMINAL_STATES:
            return state
        time.sleep(POLL_INTERVAL_S)

    raise TimeoutError(f"Batch job {job_id} timed out after {POLL_TIMEOUT_S}s")


def _download_results(client: Any, dataset_id: str) -> list[dict]:
    """Download results from output dataset."""
    dl = client.datasets.get_download_endpoint(dataset_id=dataset_id)
    rows = []
    for _filename, signed_url in dl.filename_to_signed_urls.items():
        resp = http_requests.get(signed_url, timeout=120)
        resp.raise_for_status()
        for line in resp.text.strip().split("\n"):
            if line.strip():
                rows.append(json.loads(line))
    return rows


@dataclass
class _BatchHandle:
    """Opaque handle returned by start_batch for later polling."""

    label: str
    job_id: str
    output_ds_id: str
    input_ds_id: str
    api_key: str
    account_id: str
    n_requests: int


def start_batch(
    requests: list[BatchRequest],
    label: str,
) -> _BatchHandle:
    """Submit a batch job and return a handle (non-blocking)."""
    from fireworks import Fireworks

    api_key, account_id = _get_fireworks_creds()
    client = Fireworks(api_key=api_key, account_id=account_id)

    run_id = uuid.uuid4().hex[:8]
    input_ds_id = f"gzcmd-exp2-{label}-in-{run_id}"
    output_ds_id = f"gzcmd-exp2-{label}-out-{run_id}"
    job_id = f"gzcmd-exp2-{label}-{run_id}"

    # Write JSONL
    tmp_dir = Path(tempfile.mkdtemp(prefix="gzcmd_batch_"))
    jsonl_path = tmp_dir / f"{label}.jsonl"
    _write_jsonl(requests, jsonl_path)
    log.info("  Wrote %d requests to %s", len(requests), jsonl_path)

    # Create datasets + upload
    input_ds_name = _create_dataset(client, input_ds_id, len(requests))
    output_ds_name = f"accounts/{account_id}/datasets/{output_ds_id}"
    _upload_jsonl(client, input_ds_id, jsonl_path)
    log.info("  Uploaded to dataset %s", input_ds_id)

    # Submit job
    _create_batch_job(
        api_key,
        account_id,
        job_id,
        MODEL,
        input_ds_name,
        output_ds_name,
    )
    log.info("  Batch job %s submitted (non-blocking)", job_id)

    return _BatchHandle(
        label=label,
        job_id=job_id,
        output_ds_id=output_ds_id,
        input_ds_id=input_ds_id,
        api_key=api_key,
        account_id=account_id,
        n_requests=len(requests),
    )


def await_batch(handle: _BatchHandle) -> dict[str, str]:
    """Poll a previously submitted batch job, download + parse results."""
    from fireworks import Fireworks

    client = Fireworks(
        api_key=handle.api_key,
        account_id=handle.account_id,
    )

    state = _poll_batch_job(handle.api_key, handle.account_id, handle.job_id)
    if state != "JOB_STATE_COMPLETED":
        raise RuntimeError(f"Batch job {handle.job_id} ended with state: {state}")

    # Download results
    rows = _download_results(client, handle.output_ds_id)
    log.info("  Downloaded %d results for %s", len(rows), handle.label)

    # Parse
    results: dict[str, str] = {}
    for row in rows:
        cid = row.get("custom_id", "")
        try:
            content = row["response"]["choices"][0]["message"]["content"]
            results[cid] = content
        except (KeyError, IndexError):
            log.warning("  Missing content for %s", cid)

    # Save raw responses for debug before cleanup
    raw_dump = Path(f"data/sprint3b/exp2_raw_{handle.label}.json")
    raw_dump.parent.mkdir(parents=True, exist_ok=True)
    raw_dump.write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    log.info("  Saved raw responses to %s", raw_dump)

    # Cleanup datasets
    try:
        client.datasets.delete(dataset_id=handle.input_ds_id)
        client.datasets.delete(dataset_id=handle.output_ds_id)
    except Exception as e:
        log.warning("  Cleanup failed: %s", e)

    return results


def submit_batch(
    requests: list[BatchRequest],
    label: str,
) -> dict[str, str]:
    """Submit a batch job, wait, and return {custom_id: response_content}."""
    handle = start_batch(requests, label)
    return await_batch(handle)


# ── result assembly ───────────────────────────────────────────────────
def _assemble_result(
    pair_id: str,
    dossier: Dossier,
    parsed_a: dict[str, Any],
    parsed_b: dict[str, Any],
    parsed_arb: dict[str, Any] | None,
    target: int | None,
) -> dict[str, Any]:
    """Assemble final result dict from agent/arbiter parsed responses."""
    if parsed_arb is not None:
        # Arbiter decided
        final_decision = parsed_arb.get("decision", "UNSURE")
        final_confidence = float(parsed_arb.get("confidence", 0.5))
        final_reasons = parsed_arb.get("reason_codes", [])
        final_evidence = parsed_arb.get("evidence_summary", "")
        final_quality = parsed_arb.get("quality_flags", [])
        protocol = "arbiter"
    elif parsed_a.get("decision") == parsed_b.get("decision"):
        # Consensus
        final_decision = parsed_a["decision"]
        final_confidence = (
            float(parsed_a.get("confidence", 0.5))
            + float(parsed_b.get("confidence", 0.5))
        ) / 2
        final_reasons = list(
            set(parsed_a.get("reason_codes", []))
            | set(parsed_b.get("reason_codes", []))
        )
        final_evidence = parsed_a.get("evidence_summary", "")
        final_quality = list(
            set(parsed_a.get("quality_flags", []))
            | set(parsed_b.get("quality_flags", []))
        )
        protocol = "consensus"
    else:
        # Disagreement without arbiter (shouldn't happen, but safe fallback)
        final_decision = "UNSURE"
        final_confidence = 0.5
        final_reasons = ["AGENT_DISAGREEMENT"]
        final_evidence = "Agents disagreed but no arbiter available"
        final_quality = ["disagreement_unresolved"]
        protocol = "disagreement_unresolved"

    return {
        "pair_id": pair_id,
        "decision": final_decision,
        "confidence": round(final_confidence, 3),
        "reason_codes": final_reasons,
        "evidence_summary": final_evidence,
        "quality_flags": final_quality,
        "protocol": protocol,
        "agent_a_decision": parsed_a.get("decision"),
        "agent_b_decision": parsed_b.get("decision"),
        "arbiter_decision": parsed_arb.get("decision") if parsed_arb else None,
        "target": target,
        "nota_final": dossier.features.nota_final,
        "band": dossier.model_outputs.band,
        "p_cal": dossier.model_outputs.p_cal,
        "base_choice": dossier.model_outputs.base_choice,
        "evr": dossier.model_outputs.evr,
        "guardrail": dossier.model_outputs.guardrail,
    }


def _compute_metrics(results_df: pd.DataFrame) -> dict[str, Any]:
    """Compute aggregate metrics from the results DataFrame."""
    n = len(results_df)
    if n == 0:
        return {"n": 0}

    decisions = results_df["decision"].value_counts().to_dict()
    protocols = results_df["protocol"].value_counts().to_dict()

    consensus_rate = protocols.get("consensus", 0) / n
    arbiter_rate = protocols.get("arbiter", 0) / n

    # Per-band breakdown
    band_stats: dict[str, Any] = {}
    for band, grp in results_df.groupby("band"):
        band_stats[str(band)] = grp["decision"].value_counts().to_dict()

    # FP analysis: pairs where base_choice was MATCH but LLM said NONMATCH
    flipped_to_non = results_df[
        (results_df["base_choice"] == "MATCH") & (results_df["decision"] == "NONMATCH")
    ]
    flipped_to_unsure = results_df[
        (results_df["base_choice"] == "MATCH") & (results_df["decision"] == "UNSURE")
    ]

    # Ground truth comparison (where available)
    has_gt = results_df[results_df["target"].notna()].copy()
    gt_metrics: dict[str, Any] = {}
    if len(has_gt) > 0:
        has_gt.loc[:, "target"] = has_gt["target"].astype(int)
        has_gt.loc[:, "llm_correct"] = (
            (has_gt["decision"] == "MATCH") & (has_gt["target"] == 1)
        ) | ((has_gt["decision"] == "NONMATCH") & (has_gt["target"] == 0))
        decided = has_gt[has_gt["decision"] != "UNSURE"]
        gt_metrics = {
            "gt_n": int(len(has_gt)),
            "gt_decided_n": int(len(decided)),
            "gt_accuracy": (
                round(float(decided["llm_correct"].mean()), 4)
                if len(decided) > 0
                else None
            ),
            "gt_unsure_rate": round(float((has_gt["decision"] == "UNSURE").mean()), 4),
            "gt_tp": int(
                ((decided["decision"] == "MATCH") & (decided["target"] == 1)).sum()
            ),
            "gt_fp": int(
                ((decided["decision"] == "MATCH") & (decided["target"] == 0)).sum()
            ),
            "gt_tn": int(
                ((decided["decision"] == "NONMATCH") & (decided["target"] == 0)).sum()
            ),
            "gt_fn": int(
                ((decided["decision"] == "NONMATCH") & (decided["target"] == 1)).sum()
            ),
        }

    return {
        "n": n,
        "decisions": decisions,
        "consensus_rate": round(consensus_rate, 4),
        "arbiter_rate": round(arbiter_rate, 4),
        "unsure_rate": round(decisions.get("UNSURE", 0) / n, 4),
        "flipped_match_to_nonmatch": len(flipped_to_non),
        "flipped_match_to_unsure": len(flipped_to_unsure),
        "band_breakdown": band_stats,
        **gt_metrics,
    }


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 2 — LLM batch review")
    parser.add_argument(
        "--mode",
        default="vigilancia",
        choices=["vigilancia", "confirmacao"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of pairs to review",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build dossiers but don't call LLM",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to partial results JSON to resume from",
    )
    args = parser.parse_args()

    mode = args.mode
    log.info("═══ EXP 2 — LLM BATCH REVIEW (%s mode) ═══", mode.upper())

    # ── Step 1: Run pipeline to get LLM_REVIEW pairs ──────────────────
    log.info("Step 1: Running pipeline to identify LLM_REVIEW pairs...")
    df, summary = run_v3(
        input_csv=str(DATA_CSV),
        config_path=str(CONFIG_YAML),
        mode=mode,
        p_cal="fit_platt",
    )
    llm_mask = df["action"] == "LLM_REVIEW"
    n_review = int(llm_mask.sum())
    log.info("  Pipeline: %d total pairs, %d LLM_REVIEW", len(df), n_review)

    # ── Step 2: Build dossiers ────────────────────────────────────────
    log.info("Step 2: Building dossiers for %d LLM_REVIEW pairs...", n_review)
    dossiers = build_dossiers(df, only_llm_review=True)
    log.info("  Built %d dossiers", len(dossiers))

    if args.limit and len(dossiers) > args.limit:
        log.info("  Limiting to first %d dossiers (--limit)", args.limit)
        dossiers = dossiers[: args.limit]

    if args.dry_run:
        log.info("  --dry-run: skipping LLM calls. Saving dossier metadata only.")
        meta = [
            {
                "pair_id": d.pair_id,
                "nota_final": d.features.nota_final,
                "band": d.model_outputs.band,
                "p_cal": d.model_outputs.p_cal,
                "base_choice": d.model_outputs.base_choice,
            }
            for d in dossiers
        ]
        out_path = OUT_DIR / f"exp2_dossiers_dryrun_{mode}.json"
        out_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        log.info("  Saved %d dossiers to %s", len(meta), out_path)
        return

    # ── Step 3: Check for resume data ─────────────────────────────────
    done_ids: set[str] = set()
    completed_results: list[dict[str, Any]] = []
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path) as f:
                partial_data = json.load(f)
            completed_results = partial_data.get("results", [])
            done_ids = {r["pair_id"] for r in completed_results}
            log.info("  Resuming: %d pairs already completed", len(done_ids))

    remaining = [d for d in dossiers if d.pair_id not in done_ids]
    log.info("  Remaining to review: %d pairs", len(remaining))

    if not remaining:
        log.info("  All pairs already reviewed. Recomputing metrics...")
        results_df = pd.DataFrame(completed_results)
        metrics = _compute_metrics(results_df)
        _print_summary(metrics, mode, len(completed_results), n_review, 0, 0.0)
        return

    # Build target map
    target_map: dict[str, int | None] = {}
    llm_df = df[llm_mask].copy()
    for _, row in llm_df.iterrows():
        pid = make_pair_id(
            str(row.get("COMPREC", "")),
            str(row.get("REFREC", "")),
            str(row.get("PASSO", "")),
        )
        target_map[pid] = int(row["TARGET"]) if pd.notna(row.get("TARGET")) else None

    # Index dossiers by pair_id
    dossier_map = {d.pair_id: d for d in remaining}

    # ── Step 4: Phase 1 — Submit Agent-A + Agent-B batch jobs ─────────
    log.info("Step 4: Building batch requests for %d pairs...", len(remaining))

    agent_a_requests: list[BatchRequest] = []
    agent_b_requests: list[BatchRequest] = []

    for d in remaining:
        msgs_a = build_review_messages(d, agent_role="Agent-A", batch_mode=True)
        msgs_b = build_review_messages(d, agent_role="Agent-B", batch_mode=True)
        agent_a_requests.append(
            BatchRequest(custom_id=f"A_{d.pair_id}", messages=msgs_a)
        )
        agent_b_requests.append(
            BatchRequest(custom_id=f"B_{d.pair_id}", messages=msgs_b)
        )

    log.info(
        "  Agent-A: %d requests, Agent-B: %d requests",
        len(agent_a_requests),
        len(agent_b_requests),
    )

    t0 = time.time()

    # Submit Agent-A + Agent-B in parallel (non-blocking)
    log.info("Step 5a: Submitting Agent-A batch job (non-blocking)...")
    handle_a = start_batch(agent_a_requests, label=f"agent-a-{mode}")
    log.info("Step 5b: Submitting Agent-B batch job (non-blocking)...")
    handle_b = start_batch(agent_b_requests, label=f"agent-b-{mode}")

    log.info("Both batch jobs submitted — polling for completion...")

    # Await both (Agent-A first, then B — both are already running)
    log.info("  Awaiting Agent-A batch...")
    raw_a = await_batch(handle_a)
    log.info("  Agent-A batch complete: %d responses", len(raw_a))

    log.info("  Awaiting Agent-B batch...")
    raw_b = await_batch(handle_b)
    log.info("  Agent-B batch complete: %d responses", len(raw_b))

    # ── Step 5: Parse responses ───────────────────────────────────────
    log.info("Step 6: Parsing agent responses...")
    parsed_a: dict[str, dict[str, Any]] = {}
    parsed_b: dict[str, dict[str, Any]] = {}
    parse_errors: list[dict[str, Any]] = []

    for pair_id in dossier_map:
        cid_a = f"A_{pair_id}"
        cid_b = f"B_{pair_id}"

        try:
            if cid_a in raw_a:
                parsed_a[pair_id] = extract_and_validate(raw_a[cid_a], pair_id)
            else:
                parse_errors.append(
                    {"pair_id": pair_id, "agent": "A", "error": "missing"}
                )
        except Exception as e:
            parse_errors.append(
                {
                    "pair_id": pair_id,
                    "agent": "A",
                    "error": str(e),
                    "raw_content": (raw_a.get(cid_a, ""))[:500],
                }
            )

        try:
            if cid_b in raw_b:
                parsed_b[pair_id] = extract_and_validate(raw_b[cid_b], pair_id)
            else:
                parse_errors.append(
                    {"pair_id": pair_id, "agent": "B", "error": "missing"}
                )
        except Exception as e:
            parse_errors.append(
                {
                    "pair_id": pair_id,
                    "agent": "B",
                    "error": str(e),
                    "raw_content": (raw_b.get(cid_b, ""))[:500],
                }
            )

    log.info(
        "  Parsed: A=%d, B=%d, errors=%d",
        len(parsed_a),
        len(parsed_b),
        len(parse_errors),
    )

    # ── Step 6: Identify disagreements → Arbiter batch ────────────────
    both_ok = set(parsed_a.keys()) & set(parsed_b.keys())
    agreements: list[str] = []
    disagreements: list[str] = []

    for pid in both_ok:
        if parsed_a[pid].get("decision") == parsed_b[pid].get("decision"):
            agreements.append(pid)
        else:
            disagreements.append(pid)

    log.info("  Agreements: %d, Disagreements: %d", len(agreements), len(disagreements))

    parsed_arb: dict[str, dict[str, Any]] = {}
    if disagreements:
        log.info("Step 7: Submitting Arbiter batch for %d pairs...", len(disagreements))
        arb_requests: list[BatchRequest] = []
        for pid in disagreements:
            d = dossier_map[pid]
            msgs = build_arbiter_messages(d, parsed_a[pid], parsed_b[pid])
            arb_requests.append(BatchRequest(custom_id=f"ARB_{pid}", messages=msgs))

        raw_arb = submit_batch(arb_requests, label=f"arbiter-{mode}")
        log.info("  Arbiter batch complete: %d responses", len(raw_arb))

        for pid in disagreements:
            cid_arb = f"ARB_{pid}"
            try:
                if cid_arb in raw_arb:
                    parsed_arb[pid] = extract_and_validate(raw_arb[cid_arb], pid)
            except Exception as e:
                parse_errors.append(
                    {"pair_id": pid, "agent": "Arbiter", "error": str(e)}
                )
    else:
        log.info("  No disagreements — skipping arbiter phase.")

    elapsed = time.time() - t0

    # ── Step 7: Assemble final results ────────────────────────────────
    log.info("Step 8: Assembling %d results...", len(both_ok))
    results: list[dict[str, Any]] = list(completed_results)

    for pid in both_ok:
        d = dossier_map[pid]
        arb = parsed_arb.get(pid)
        result = _assemble_result(
            pid,
            d,
            parsed_a[pid],
            parsed_b[pid],
            arb,
            target_map.get(pid),
        )
        results.append(result)

    # ── Step 8: Compute metrics + save ────────────────────────────────
    log.info("Step 9: Computing metrics...")
    results_df = pd.DataFrame(results)
    metrics = _compute_metrics(results_df)
    metrics["mode"] = mode
    metrics["total_elapsed_s"] = round(elapsed, 1)
    metrics["n_parse_errors"] = len(parse_errors)

    output = {
        "experiment": "exp2_llm_batch_review",
        "mode": mode,
        "model": MODEL,
        "timestamp": pd.Timestamp.now().isoformat(),
        "n_review_pairs": n_review,
        "n_reviewed": len(results),
        "n_parse_errors": len(parse_errors),
        "metrics": metrics,
        "results": results,
        "parse_errors": parse_errors,
    }

    out_path = OUT_DIR / f"exp2_llm_batch_{mode}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    log.info("Results saved to %s", out_path)

    if len(results) > 0:
        csv_path = OUT_DIR / f"exp2_llm_batch_{mode}.csv"
        results_df.to_csv(csv_path, index=False)
        log.info("CSV saved to %s", csv_path)

    _print_summary(metrics, mode, len(results), n_review, len(parse_errors), elapsed)


def _print_summary(
    metrics: dict[str, Any],
    mode: str,
    n_reviewed: int,
    n_review: int,
    n_errors: int,
    elapsed: float,
) -> None:
    log.info("═══ EXP 2 SUMMARY ═══")
    log.info("  Mode: %s", mode)
    log.info("  Pairs reviewed: %d / %d", n_reviewed, n_review)
    log.info("  Parse errors: %d", n_errors)
    log.info("  Decisions: %s", metrics.get("decisions", {}))
    log.info("  Consensus rate: %.1f%%", metrics.get("consensus_rate", 0) * 100)
    log.info("  Arbiter rate: %.1f%%", metrics.get("arbiter_rate", 0) * 100)
    log.info("  UNSURE rate: %.1f%%", metrics.get("unsure_rate", 0) * 100)
    log.info(
        "  Flipped MATCH→NONMATCH: %d",
        metrics.get("flipped_match_to_nonmatch", 0),
    )
    log.info("  Flipped MATCH→UNSURE: %d", metrics.get("flipped_match_to_unsure", 0))

    if "gt_accuracy" in metrics and metrics["gt_accuracy"] is not None:
        log.info("  GT accuracy (decided): %.1f%%", metrics["gt_accuracy"] * 100)
        log.info(
            "  GT confusion: TP=%d FP=%d TN=%d FN=%d",
            metrics.get("gt_tp", 0),
            metrics.get("gt_fp", 0),
            metrics.get("gt_tn", 0),
            metrics.get("gt_fn", 0),
        )

    log.info("  Total elapsed: %.0fs", elapsed)
    log.info("═══ DONE ═══")


if __name__ == "__main__":
    main()
