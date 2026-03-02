"""Fireworks AI Batch Inference for L3 relevance evaluation.

Collects all LLM evaluation requests, submits them as a single batch job
to Fireworks AI (50% cost savings), polls for completion, and returns
parsed results compatible with the real-time e3_relevance pipeline.

Uses the official ``fireworks-ai`` SDK for dataset management and raw
HTTP for batch-job creation/polling (not yet in SDK).

Usage (via CLI):
    python -m scripts.ref_audit --batch
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import requests as http_requests  # raw HTTP only for batch-job endpoints

from . import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FIREWORKS_API = "https://api.fireworks.ai/v1"
_POLL_INTERVAL_S = 10
_POLL_TIMEOUT_S = 60 * 60  # 1 hour max wait
_TERMINAL_STATES = {
    "JOB_STATE_COMPLETED",
    "JOB_STATE_FAILED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_CANCELLED",
}


def _get_client():
    """Lazy-init a sync Fireworks client."""
    from fireworks import Fireworks

    return Fireworks(
        api_key=config.LLM_API_KEY,
        account_id=config.FIREWORKS_ACCOUNT_ID,
    )


# ---------------------------------------------------------------------------
# Dataset helpers (SDK)
# ---------------------------------------------------------------------------


def _create_dataset(client, dataset_id: str, example_count: int = 0) -> str:
    """Create an empty user-uploaded dataset. Returns full dataset name."""
    result = client.datasets.create(
        dataset_id=dataset_id,
        dataset={
            "userUploaded": {},
            "exampleCount": str(example_count),
        },
    )
    name = getattr(result, "name", None) or (
        f"accounts/{config.FIREWORKS_ACCOUNT_ID}/datasets/{dataset_id}"
    )
    log.info("Created dataset: %s", name)
    return name


def _upload_jsonl(client, dataset_id: str, jsonl_path: Path) -> None:
    """Upload a JSONL file to an existing dataset."""
    client.datasets.upload(
        dataset_id=dataset_id,
        file=jsonl_path,
    )
    log.info("Uploaded %s to dataset %s", jsonl_path.name, dataset_id)


def _download_results(client, output_dataset_id: str) -> list[dict[str, Any]]:
    """Download results JSONL from the output dataset."""
    endpoint = client.datasets.get_download_endpoint(
        dataset_id=output_dataset_id,
    )
    signed_urls: dict[str, str] = getattr(endpoint, "filename_to_signed_urls", {}) or {}
    if not signed_urls:
        # Fallback: try dict access
        if hasattr(endpoint, "to_dict"):
            signed_urls = endpoint.to_dict().get("filenameToSignedUrls", {})

    results: list[dict[str, Any]] = []
    for filename, signed_url in signed_urls.items():
        if "error" in filename.lower():
            err_resp = http_requests.get(signed_url, timeout=60)
            if err_resp.ok and err_resp.text.strip():
                log.warning("Batch errors in %s:\n%s", filename, err_resp.text[:2000])
            continue

        dl_resp = http_requests.get(signed_url, timeout=120)
        dl_resp.raise_for_status()
        for line in dl_resp.text.strip().splitlines():
            if line.strip():
                results.append(json.loads(line))

    log.info("Downloaded %d results from batch output", len(results))
    return results


def _cleanup_dataset(client, dataset_id: str) -> None:
    """Best-effort delete a dataset."""
    try:
        client.datasets.delete(dataset_id=dataset_id)
        log.debug("Deleted dataset %s", dataset_id)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Batch-job helpers (raw HTTP — not yet in SDK)
# ---------------------------------------------------------------------------


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _account_url() -> str:
    acct = config.FIREWORKS_ACCOUNT_ID
    if not acct:
        raise RuntimeError(
            "FIREWORKS_ACCOUNT_ID not set. Add it to .env or environment."
        )
    return f"{_FIREWORKS_API}/accounts/{acct}"


def _create_batch_job(
    job_id: str,
    model: str,
    input_dataset_name: str,
    output_dataset_name: str,
) -> dict[str, Any]:
    """Create a batch inference job."""
    url = f"{_account_url()}/batchInferenceJobs?batchInferenceJobId={job_id}"
    body = {
        "model": model,
        "inputDatasetId": input_dataset_name,
        "outputDatasetId": output_dataset_name,
        "inferenceParameters": {
            "maxTokens": 2048,
            "temperature": 0.1,
            "topP": 1.0,
        },
    }
    resp = http_requests.post(url, headers=_auth_headers(), json=body, timeout=30)
    if not resp.ok:
        log.error("Batch job creation failed: %s %s", resp.status_code, resp.text[:500])
    resp.raise_for_status()
    data = resp.json()
    log.info("Created batch job: %s (state=%s)", job_id, data.get("state"))
    return data


def _poll_job(job_id: str) -> dict[str, Any]:
    """Poll until terminal state. Returns final job object."""
    url = f"{_account_url()}/batchInferenceJobs/{job_id}"
    elapsed = 0
    while elapsed < _POLL_TIMEOUT_S:
        resp = http_requests.get(url, headers=_auth_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        state = data.get("state", "UNKNOWN")
        progress = data.get("progress", {})
        done = progress.get("completed", 0)
        total = progress.get("total", "?")
        log.info(
            "  Batch job %s: state=%s (%s/%s requests)",
            job_id,
            state,
            done,
            total,
        )
        if state in _TERMINAL_STATES:
            return data
        time.sleep(_POLL_INTERVAL_S)
        elapsed += _POLL_INTERVAL_S

    raise TimeoutError(f"Batch job {job_id} did not complete within {_POLL_TIMEOUT_S}s")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class BatchRequest:
    """A single LLM evaluation request to be batched."""

    __slots__ = ("custom_id", "messages")

    def __init__(self, custom_id: str, messages: list[dict[str, str]]) -> None:
        self.custom_id = custom_id
        self.messages = messages

    def to_jsonl_row(self) -> dict[str, Any]:
        return {
            "custom_id": self.custom_id,
            "body": {
                "messages": self.messages,
                "max_tokens": 2048,
                "temperature": 0.1,
                "top_p": 1,
                "top_k": 40,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            },
        }


def submit_batch(
    batch_requests: list[BatchRequest],
    model: str | None = None,
) -> dict[str, str]:
    """Submit a batch of LLM requests and wait for results.

    Args:
        batch_requests: List of BatchRequest objects.
        model: Fireworks model slug. Defaults to config.LLM_MODEL.

    Returns:
        Dict mapping custom_id -> raw LLM response text (JSON string).
    """
    if not batch_requests:
        log.warning("No requests to batch.")
        return {}

    model = model or config.LLM_MODEL
    client = _get_client()
    run_id = uuid.uuid4().hex[:8]
    input_ds_id = f"ref-audit-input-{run_id}"
    output_ds_id = f"ref-audit-output-{run_id}"
    job_id = f"ref-audit-job-{run_id}"

    # Write JSONL locally
    jsonl_path = config.CACHE_DIR / f"batch_input_{run_id}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for req in batch_requests:
            f.write(json.dumps(req.to_jsonl_row(), ensure_ascii=False) + "\n")

    log.info(
        "Batch: %d requests, model=%s, job=%s",
        len(batch_requests),
        model,
        job_id,
    )

    acct = config.FIREWORKS_ACCOUNT_ID
    output_ds_name = f"accounts/{acct}/datasets/{output_ds_id}"
    try:
        # 1. Create input dataset (output created by batch job)
        input_ds_name = _create_dataset(
            client, input_ds_id, example_count=len(batch_requests)
        )

        # 2. Upload JSONL
        _upload_jsonl(client, input_ds_id, jsonl_path)

        # 3. Create batch job
        _create_batch_job(
            job_id=job_id,
            model=model,
            input_dataset_name=input_ds_name,
            output_dataset_name=output_ds_name,
        )

        # 4. Poll until done
        final = _poll_job(job_id)
        state = final.get("state", "UNKNOWN")
        if state != "JOB_STATE_COMPLETED":
            raise RuntimeError(f"Batch job ended with state={state}: {final}")

        # 5. Download results
        raw_results = _download_results(client, output_ds_id)

        # 6. Map custom_id -> response content
        result_map: dict[str, str] = {}
        for row in raw_results:
            cid = row.get("custom_id", "")
            try:
                content = (
                    row.get("response", {})
                    .get("body", {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                result_map[cid] = content
            except (IndexError, KeyError, TypeError) as exc:
                log.warning("Failed to parse result for %s: %s", cid, exc)

        log.info(
            "Batch complete: %d/%d results parsed",
            len(result_map),
            len(batch_requests),
        )
        return result_map

    finally:
        _cleanup_dataset(client, input_ds_id)
        _cleanup_dataset(client, output_ds_id)
        log.debug("Local JSONL kept at %s", jsonl_path)
