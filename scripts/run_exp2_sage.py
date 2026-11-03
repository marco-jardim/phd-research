"""Sage choice reviewer for Exp2; only pseudonymous structured outputs persist.

The dossier is reconstructed locally and never written to disk. See --help for
preflight, smoke, resumable execution, and offline summary commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import redirect_stdout
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_exp2_ablacao_dossie import REMOVED_FIELDS, ablate
from comparar_revisores_exp2 import gold
from gzcmd.llm_dossier import build_dossier, make_pair_id
from gzcmd.llm_review import _load_system_prompt
from gzcmd.runner import run_v3

BASE = "https://sage.levanto.ai"
MODES = ("vigilancia", "confirmacao")
PARAMS = {"reasoning": "auto", "grounding": {"trigger": "never"}}
OPTIONS = [
    {"option": "match", "description": "Os dois registros representam a mesma pessoa, segundo os critérios e o modo descritos."},
    {"option": "nonmatch", "description": "Os dois registros não devem ser vinculados, segundo os critérios e o modo descritos."},
]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    replace_checkpoint(tmp, path)


def replace_checkpoint(tmp, path):
    # Windows readers/antivirus can briefly deny rename of an open checkpoint.
    # The durable temp file is retained if all bounded attempts fail.
    for attempt in range(8):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.05 * 2 ** attempt, 2))


def recover_pending_checkpoint(path):
    tmp = path.with_suffix(".tmp")
    if not path.exists() or not tmp.exists():
        return
    old = json.loads(path.read_text(encoding="utf-8"))
    pending = json.loads(tmp.read_text(encoding="utf-8"))
    keys = ("mode", "arm", "instructions", "selected_pair_ids", "request_sha256")
    if any(old.get(k) != pending.get(k) for k in keys):
        raise ValueError("Pending checkpoint input mismatch; refusing recovery")
    previous, candidate = old["results"], pending["results"]
    if len(candidate) < len(previous) or candidate[:len(previous)] != previous:
        raise ValueError("Pending checkpoint would replace existing responses")
    ids = [r["pair_id"] for r in candidate]
    if len(ids) != len(set(ids)) or not set(ids) <= set(old["selected_pair_ids"]):
        raise ValueError("Invalid pending checkpoint pair cohort")
    replace_checkpoint(tmp, path)
    print(f"RECOVERED_CHECKPOINT {path.name} preserved={len(previous)} recovered={len(candidate) - len(previous)}", flush=True)


def instructions(mode, arm):
    # Preserve source criteria verbatim, but explicitly replace the response
    # schema/protocol and the risk preference for this single-choice transport.
    original = _load_system_prompt()
    policy = ("VIGILÂNCIA: priorize sensibilidade; incerteza moderada favorece o vínculo (match), "
              "sem ignorar contradições fortes."
              if mode == "vigilancia" else
              "CONFIRMAÇÃO: priorize especificidade; match exige evidência forte e convergente. "
              "Não aceite vínculo apenas por falta de evidência contrária.")
    boundary = ("Todo o bloco model_outputs foi removido intencionalmente. Não infira seus valores nem trate "
                "sua ausência como dado faltante de pessoa. Use somente as features disponíveis."
                if arm == "sem_mo" else "O dossiê inclui model_outputs, além das features.")
    return (original + "\n\nADAPTAÇÃO SAGE (prevalece sobre formato e protocolo acima):\n"
            "Decida se este par representa a mesma pessoa. Uma única decisão kind=choice, sem agentes A/B ou árbitro. "
            "Não produza o JSON de resposta do revisor original, reason_codes, evidência textual ou PII. "
            "As únicas opções são match e nonmatch, nesta ordem fixa. Em empate/incerteza decisiva, "
            "a abstenção nativa chosen=null encaminha a humano (UNSURE); não force escolha. "
            + policy + " " + boundary)


def prepare(mode, arm):
    source = ROOT / f"data/sprint3b/anthropic/exp2_llm_sync_{mode}.json"
    previous = json.loads(source.read_text(encoding="utf-8"))
    ids = [r["pair_id"] for r in previous["results"]]
    expected = 1410 if mode == "vigilancia" else 435
    if len(ids) != expected or len(set(ids)) != expected or previous["mode"] != mode:
        raise ValueError("Original cohort incomplete, duplicated, or wrong mode")
    # The pipeline can emit verbose information: suppress it rather than writing
    # anything other than the allowlisted aggregate progress messages below.
    with open(os.devnull, "w") as sink, redirect_stdout(sink):
        frame, _ = run_v3(input_csv=str(ROOT / "data/COMPARADORSEMIDENT.csv"),
                          config_path=str(ROOT / "gzcmd/gzcmd_v3_config.yaml"), mode=mode, p_cal="fit_platt")
    # Identifier dtypes can change in the pipeline loader (e.g. 001 -> 1).
    # Recover the original string-hashed ID by a validated, one-to-one join on
    # canonical identifiers; never hash a coerced pipeline identifier directly.
    # Select the fixed historical cohort, not today's possibly changed triage.
    dossiers = aligned_dossiers(frame, set(ids))
    if not set(ids) <= dossiers.keys():
        raise ValueError("Original cohort not found in reconstructed dossiers")
    payloads = {pid: ablate(dossiers[pid]) if arm == "sem_mo" else dossiers[pid] for pid in ids}
    if arm == "sem_mo" and any("model_outputs" in d for d in payloads.values()):
        raise ValueError("Ablation failed")
    text = instructions(mode, arm)
    requests_by_id = {pid: {"content": json.dumps(payloads[pid], ensure_ascii=False, indent=2, allow_nan=False),
                            "question": {"id": "par_match", "kind": "choice", "instructions": text,
                                         "options": OPTIONS}, **PARAMS} for pid in ids}
    return ids, requests_by_id, text


def canonical_identifier(value):
    text = str(value).strip()
    try:
        numeric = Decimal(text)
        if numeric.is_finite() and numeric == numeric.to_integral_value():
            return str(int(numeric))
    except InvalidOperation:
        pass
    return text


def aligned_dossiers(frame, selected):
    raw = pd.read_csv(ROOT / "data/COMPARADORSEMIDENT.csv", sep=";", dtype=str, low_memory=False)
    def columns(df):
        return [next(c for c in df.columns if c == n or c.startswith(n + ","))
                for n in ("COMPREC", "REFREC", "PASSO")]
    mapping = {}
    for values in raw[columns(raw)].itertuples(index=False, name=None):
        original = tuple(str(v).strip() for v in values)
        pid = make_pair_id(*original)
        key = tuple(canonical_identifier(v) for v in original)
        if key in mapping:
            raise ValueError("Ambiguous normalized source identifier triple")
        mapping[key] = pid
    dossiers = {}
    cols = columns(frame)
    for _, row in frame.iterrows():
        key = tuple(canonical_identifier(row[c]) for c in cols)
        pid = mapping.get(key)
        if pid in selected:
            if pid in dossiers:
                raise ValueError("Ambiguous normalized pipeline identifier triple")
            dossiers[pid] = build_dossier(row, pair_id=pid).to_dict()
    if set(dossiers) != selected:
        raise ValueError(f"Aligned fixed cohort incomplete: found={len(dossiers)} expected={len(selected)}")
    return dossiers


class APIError(Exception):
    def __init__(self, status, body_literal=None, final_url=None):
        self.status = status
        self.body_literal = body_literal
        self.final_url = final_url
        super().__init__(f"HTTP_{status}")


class Sage:
    def __init__(self, key):
        self.key = key
        self.stop = threading.Event()
        self.quota = threading.Event()
        self.lock = threading.Lock()
        self.calls = Counter()
        self.http_diagnostics = []
        self.grounding_null = False

    def call(self, endpoint, body=None, auth=True):
        if self.grounding_null and body is not None and body.get("grounding") == {"trigger": "never"}:
            # OpenAPI DecideRequestPublic.grounding is nullable, default null.
            # Confidential deployments reject even {trigger: never}; null keeps
            # grounding disabled without sending a GroundingConfig object.
            body = {**body, "grounding": None}
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = "Bearer " + self.key
        for attempt in range(3):
            if self.stop.is_set():
                raise APIError("stopped")
            with self.lock:
                self.calls[endpoint] += 1
            try:
                response = requests.request("GET" if body is None else "POST", BASE + endpoint,
                                            json=body, headers=headers, timeout=(15, 90), allow_redirects=False)
            except requests.RequestException:
                # Never stringify requests exceptions (may carry credentials/body).
                raise APIError("connection") from None
            status = response.status_code
            error_body, final_url = None, None
            if status >= 400:
                # Error bodies are requested for diagnosing provider-side 404.
                # The submitted dossiers contain only scores; still redact any
                # credential-like string before logging the literal response.
                error_body = response.text.replace(self.key, "[REDACTED]") if self.key else response.text
                error_body = re.sub(r"lv_live_[A-Za-z0-9_.-]+", "[REDACTED]", error_body)
                final_url = response.url
                self.http_diagnostics.append({"endpoint": endpoint, "attempt": attempt + 1,
                                              "status": status, "final_url": final_url,
                                              "body_literal": error_body})
                print("HTTP_ERROR " + json.dumps(self.http_diagnostics[-1], ensure_ascii=False), flush=True)
            if status == 402:
                self.quota.set()
                self.stop.set()
                raise APIError(402, error_body, final_url)
            if status in (404, 429, 500, 502, 503, 504) and attempt < 2:
                delay = 2 ** (attempt + 1)
                retry = response.headers.get("Retry-After")
                if retry:
                    try:
                        delay = max(delay, float(retry))
                    except ValueError:
                        try:
                            delay = max(delay, (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds())
                        except (ValueError, TypeError):
                            pass
                print(f"RETRY endpoint={endpoint} status={status} delay_s={delay:.1f}", flush=True)
                if self.stop.wait(delay):
                    raise APIError("stopped")
                continue
            if not 200 <= status < 300:
                self.stop.set()
                raise APIError(status, error_body, final_url)
            try:
                return response.json()
            except ValueError:
                self.stop.set()
                raise APIError("invalid_json") from None
        raise APIError("retry_exhausted")


def number(value, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Invalid API numeric field")
    return float(value)


def safe_numeric_tree(value):
    """Persist quantitative metadata/booleans only, never free-text reasoning."""
    if isinstance(value, dict):
        return {k: safe_numeric_tree(v) for k, v in value.items()
                if re.fullmatch(r"[a-zA-Z0-9_]+", k) and not isinstance(v, str)}
    if isinstance(value, list):
        return [safe_numeric_tree(v) for v in value if not isinstance(v, str)]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return number(value)
    return None


def parse_response(pid, raw, elapsed):
    if raw.get("id") != "par_match" or raw.get("kind") != "choice":
        raise ValueError("Unexpected decision schema")
    result, meta = raw["result"], raw["meta"]
    grounding_meta = raw.get("grounding_meta")
    if grounding_meta is not None and grounding_meta.get("triggered") is not False:
        raise ValueError("Unexpected grounding activation")
    chosen = result["chosen"]
    if chosen not in ("match", "nonmatch", None):
        raise ValueError("Unknown choice")
    probability = number(result["probability"], nullable=True)
    if (chosen is not None and probability is None) or (probability is not None and not 0 <= probability <= 1):
        raise ValueError("Invalid chosen confidence")
    probabilities = [{"option": item["option"], "probability": number(item["probability"])}
                     for item in result["probabilities"]]
    if [p["option"] for p in probabilities] != ["match", "nonmatch"]:
        raise ValueError("Option order changed")
    if any(not 0 <= p["probability"] <= 1 for p in probabilities) or abs(sum(p["probability"] for p in probabilities) - 1) > .025:
        raise ValueError("Invalid probability distribution")
    model = meta["model"]
    if not isinstance(model, str) or not re.fullmatch(r"levanto-sage-[a-zA-Z0-9._-]+", model):
        raise ValueError("Unexpected model identifier")
    # reasoning may include generated text: preserve quantitative structure,
    # record redaction explicitly rather than leaking an arbitrary rationale.
    reasoning = safe_numeric_tree(meta.get("reasoning"))
    if isinstance(reasoning, dict) and meta["reasoning"].get("limited") in ("cap", "timeout", "budget"):
        reasoning["limited"] = meta["reasoning"]["limited"]
    clean_meta = {"model": model, "latency_ms": number(meta["latency_ms"]), "reasoning": reasoning}
    clean_meta["grounding_triggered"] = False if grounding_meta is None else grounding_meta["triggered"]
    for field in ("usage", "cost", "cost_usd", "input_tokens", "tokens"):
        if field in meta:
            clean_meta[field] = safe_numeric_tree(meta[field])
    return {"pair_id": pid, "decision": {"match": "MATCH", "nonmatch": "NONMATCH", None: "UNSURE"}[chosen],
            "chosen": chosen, "confidence": probability, "probability": probability,
            "probabilities": probabilities, "p_match": probabilities[0]["probability"],
            "meta": clean_meta, "latency_ms": clean_meta["latency_ms"],
            "total_latency_s": round(elapsed, 6), "total_tokens": clean_meta.get("input_tokens"),
            "reason_codes": [], "quality_flags": {}, "protocol": "sage_choice",
            "usage": safe_numeric_tree(raw.get("usage")), "reasoning_text_persisted": False}


def run_path(mode, arm):
    return ROOT / f"data/sprint3b/sage_{arm}/exp2_llm_sync_{mode}.json"


def init_run(mode, arm, ids, payloads, text):
    path = run_path(mode, arm)
    recover_pending_checkpoint(path)
    fingerprints = {pid: hashlib.sha256(json.dumps(payloads[pid], ensure_ascii=False, sort_keys=True).encode()).hexdigest() for pid in ids}
    if path.exists():
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["request_sha256"] != fingerprints or run["instructions"] != text or run["selected_pair_ids"] != ids:
            raise ValueError("Checkpoint inputs changed; refusing mixed run")
        if len({r["pair_id"] for r in run["results"]}) != len(run["results"]):
            raise ValueError("Duplicate checkpoint results")
        return run
    run = {"experiment": "exp2_llm_sync", "mode": mode, "arm": arm, "model": None, "timestamp": now(),
           "parameters": {**PARAMS, "options": OPTIONS, "timeout_s": 90, "max_attempts": 3},
           "instructions": text, "source_prompt": "gzcmd/gzcmd_v3_llm_prompt.md",
           "removed_fields": REMOVED_FIELDS if arm == "sem_mo" else [], "selected_pair_ids": ids,
           "request_sha256": fingerprints, "n_review_pairs": len(ids), "n_reviewed": 0, "n_errors": 0,
           "results": [], "errors": [], "status": "prepared", "elapsed_s": 0, "metrics": {}}
    save(path, run)
    return run


def execute_run(api, mode, arm, ids, payloads, text, limit, workers, resume_after_quota=False):
    run = init_run(mode, arm, ids, payloads, text)
    if run["status"] == "quota_exhausted" and not resume_after_quota:
        api.stop.set()
        api.quota.set()
        return run
    if run["status"] == "quota_exhausted":
        run.setdefault("quota_resumptions", []).append({"timestamp": now(), "n_preserved": len(run["results"]),
            "preserved_results_sha256": hashlib.sha256(json.dumps(run["results"], sort_keys=True).encode()).hexdigest()})
    transport = run.setdefault("transport", {"decide_calls_before_tracking_lower_bound": len(run["results"]) +
        sum(isinstance(e.get("status"), int) for e in run.get("errors", [])), "decide_calls_tracked": 0,
        "historical_call_count_exact": False})
    calls_at_start = api.calls["/decide"]
    prior_tracked = transport["decide_calls_tracked"]
    if api.grounding_null and not transport.get("grounding_wire_null"):
        run.setdefault("wire_encoding_changes", []).append({"timestamp": now(), "n_preserved": len(run["results"]),
            "preserved_results_sha256": hashlib.sha256(json.dumps(run["results"], sort_keys=True).encode()).hexdigest(),
            "canonical_grounding": {"trigger": "never"}, "wire_grounding": None,
            "schema_source": "https://docs.levanto.ai/openapi-schema.md",
            "reason": "HTTP400 confidential deployment rejects GroundingConfig; nullable field disables grounding"})
        transport["grounding_wire_null"] = True
    done = {r["pair_id"] for r in run["results"]}
    selected = ids[:limit] if limit else ids
    pending = iter(pid for pid in selected if pid not in done)
    started, base_elapsed = time.monotonic(), run["elapsed_s"]
    run["workers"], run["status"] = workers, "running"
    run["parameters"]["max_attempts"] = 3

    def one(pid):
        start = time.monotonic()
        raw = api.call("/decide", payloads[pid])
        parsed = parse_response(pid, raw, time.monotonic() - start)
        if limit:
            # Inspect a privacy-safe projection of the *actual* raw response.
            print("SMOKE_RESPONSE " + json.dumps({"pair_id": pid, "id": raw.get("id"), "kind": raw.get("kind"),
                  "result": {k: parsed[k] for k in ("chosen", "probability", "probabilities")},
                  "meta": parsed["meta"], "raw_top_keys": sorted(raw),
                  "raw_meta_keys": sorted(raw["meta"]), "raw_reasoning_keys": sorted(raw["meta"].get("reasoning", {}))
                  if isinstance(raw["meta"].get("reasoning"), dict) else []}, ensure_ascii=False), flush=True)
        return parsed

    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}

        def submit():
            if api.stop.is_set():
                return
            pid = next(pending, None)
            if pid is not None:
                active[pool.submit(one, pid)] = pid

        for _ in range(workers):
            submit()
        failures = 0
        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                pid = active.pop(future)
                try:
                    result = future.result()
                    if run["model"] not in (None, result["meta"]["model"]):
                        raise ValueError("Observed model changed within run")
                    run["model"] = result["meta"]["model"]
                    run["results"].append(result)
                    failures = 0
                    print(f"RESULT {mode} {arm} {len(run['results'])}/{len(ids)} {pid} {result['decision']}", flush=True)
                except Exception as exc:
                    failures += 1
                    status = getattr(exc, "status", None)
                    run["errors"].append({"pair_id": pid, "type": type(exc).__name__, "status": status, "timestamp": now()})
                    if getattr(exc, "body_literal", None) is not None:
                        run["errors"][-1].update({"body_literal": exc.body_literal, "final_url": exc.final_url})
                    print(f"ERROR {mode} {arm} completed={len(run['results'])} status={status} type={type(exc).__name__}", flush=True)
                    if failures >= 2 or isinstance(exc, (ValueError, KeyError, TypeError)):
                        api.stop.set()
                run["n_reviewed"], run["n_errors"] = len(run["results"]), len(run["errors"])
                run["elapsed_s"] = round(base_elapsed + time.monotonic() - started, 3)
                run["status"] = "quota_exhausted" if api.quota.is_set() else "running"
                run["updated_at"] = now()
                transport["decide_calls_tracked"] = prior_tracked + api.calls["/decide"] - calls_at_start
                save(run_path(mode, arm), run)
            for _ in completed:
                submit()
    run["status"] = ("quota_exhausted" if api.quota.is_set() else "stopped" if api.stop.is_set()
                     else "complete" if len(run["results"]) == len(ids) else "smoke_complete")
    run["metrics"] = dict(Counter(r["decision"] for r in run["results"]))
    save(run_path(mode, arm), run)
    print(f"DONE {mode} {arm} completed={run['n_reviewed']}/{len(ids)} status={run['status']} "
          f"model={run['model']}", flush=True)
    return run


def rates(frame):
    dec, y = frame.decision, frame.y
    vp = int(((dec == "MATCH") & (y == 1)).sum())
    fp = int(((dec == "MATCH") & (y == 0)).sum())
    fn = int(((dec == "NONMATCH") & (y == 1)).sum())
    vn = int(((dec == "NONMATCH") & (y == 0)).sum())
    u1 = int(((dec == "UNSURE") & (y == 1)).sum())
    u0 = int(((dec == "UNSURE") & (y == 0)).sum())
    divide = lambda a, b: a / b if b else None
    return {"n": len(frame), "VP": vp, "FP": fp, "FN": fn, "VN": vn, "UNSURE": u1 + u0,
            "UNSURE_pos": u1, "UNSURE_neg": u0, "sens_sem_unsure": divide(vp, vp + fn),
            "espec_sem_unsure": divide(vn, vn + fp), "sens_unsure_como_erro": divide(vp, vp + fn + u1),
            "espec_unsure_como_erro": divide(vn, vn + fp + u0),
            "sens_unsure_como_vinculo": divide(vp + u1, vp + fn + u1),
            "espec_unsure_como_vinculo": divide(vn, vn + fp + u0)}


def numeric_totals(value, prefix=""):
    """Flatten quantitative usage/cost fields, preserving unit-specific names."""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(numeric_totals(child, f"{prefix}_{key}" if prefix else key))
        return result
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {prefix: value}
    return {}


def exact_mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def anchoring_metrics(pairs):
    gained = sum(r["com_mo"] != "MATCH" and r["sem_mo"] == "MATCH" for r in pairs)
    lost = sum(r["com_mo"] == "MATCH" and r["sem_mo"] != "MATCH" for r in pairs)
    positives = [r for r in pairs if r["gabarito"] == 1]
    fn_reverted = sum(r["com_mo"] == "NONMATCH" and r["sem_mo"] != "NONMATCH" for r in positives)
    fn_introduced = sum(r["com_mo"] != "NONMATCH" and r["sem_mo"] == "NONMATCH" for r in positives)
    return {"n_paired": len(pairs), "gained_match_sem_mo": gained, "lost_match_sem_mo": lost,
            "mcnemar_match_exact_p": exact_mcnemar(gained, lost),
            "fn_com_mo": sum(r["com_mo"] == "NONMATCH" for r in positives),
            "fn_sem_mo": sum(r["sem_mo"] == "NONMATCH" for r in positives),
            "fn_com_to_sem_match": sum(r["com_mo"] == "NONMATCH" and r["sem_mo"] == "MATCH" for r in positives),
            "fn_com_to_sem_unsure": sum(r["com_mo"] == "NONMATCH" and r["sem_mo"] == "UNSURE" for r in positives),
            "fn_reverted_to_nonfn": fn_reverted, "fn_introduced": fn_introduced,
            "mcnemar_fn_exact_p": exact_mcnemar(fn_reverted, fn_introduced)}


def summarize():
    g = gold()
    if not g.index.is_unique:
        raise ValueError("Duplicate gold pair IDs")
    rows, reliability, thresholds, comparisons, telemetry = [], [], [], [], []
    pair_comparisons, operational = [], []
    for arm in ("sem_mo", "com_mo"):
        for mode in MODES:
            path = run_path(mode, arm)
            if not path.exists():
                continue
            run = json.loads(path.read_text(encoding="utf-8"))
            frame = pd.DataFrame(run["results"], columns=["pair_id", "decision", "probability", "p_match", "latency_ms", "total_latency_s"])
            frame = frame.set_index("pair_id").join(g, how="left", validate="one_to_one")
            if frame.y.isna().any():
                raise ValueError("Missing gold")
            label = {"modo": mode, "braco": arm, "modelo": run["model"]}
            metric = {**label, "status": run["status"], "n_expected": run["n_review_pairs"],
                      "n_errors": len(run.get("errors", [])),
                      "last_http_status": run["errors"][-1].get("status") if run.get("errors") else None,
                      **rates(frame)}
            for key, value in run.get("transport", {}).items():
                metric["transport_" + key] = value
            usage_totals, usage_reported = Counter(), Counter()
            for result in run["results"]:
                reported = {"response_usage": result.get("usage"),
                            **{k: v for k, v in result["meta"].items()
                               if k in ("usage", "cost", "cost_usd", "input_tokens", "tokens")}}
                flat = numeric_totals(reported)
                usage_totals.update(flat)
                usage_reported.update(flat.keys())
            for field, total in usage_totals.items():
                metric[f"api_{field}_sum"] = total
                metric[f"api_{field}_n_reported"] = usage_reported[field]
            for score, suffix in (("p_match", ""), ("probability", "_raw_chosen_vs_y")):
                valid = frame[frame[score].notna()]
                metric["n_probability" + suffix] = len(valid)
                metric["auc" + suffix] = float(roc_auc_score(valid.y, valid[score])) if valid.y.nunique() == 2 else None
                metric["brier" + suffix] = float(brier_score_loss(valid.y, valid[score])) if len(valid) else None
            for field in ("latency_ms", "total_latency_s"):
                for q, name in ((0, "min"), (.5, "p50"), (.9, "p90"), (.95, "p95"), (.99, "p99"), (1, "max")):
                    metric[f"{field}_{name}"] = float(frame[field].quantile(q)) if len(frame) else None
                metric[f"{field}_mean"] = float(frame[field].mean()) if len(frame) else None
            rows.append(metric)
            operational.append({**label, "status": run["status"], "completed": len(frame),
                                "expected": run["n_review_pairs"], "last_historical_error": run.get("errors", [])[-1:][0]
                                if run.get("errors") else None})
            for decile in range(10):
                part = frame[(frame.p_match >= decile / 10) & ((frame.p_match < (decile + 1) / 10) if decile < 9 else frame.p_match.le(1))]
                reliability.append({**label, "decil": decile + 1, "inferior": decile / 10, "superior": (decile + 1) / 10,
                                    "n": len(part), "probability_media": float(part.p_match.mean()) if len(part) else None,
                                    "fracao_positiva": float(part.y.mean()) if len(part) else None})
            for threshold in (.8, .9, .95, .98):
                part = frame.copy()
                part.loc[part.probability.isna() | part.probability.lt(threshold), "decision"] = "UNSURE"
                thresholds.append({**label, "limiar_confidence_chosen": threshold, **rates(part)})
            csv = frame.rename(columns={"nota_final": "nota", "decision": "decisao", "y": "gabarito"})
            suffix = "" if arm == "sem_mo" else "_com_mo"
            csv[["nota", "decisao", "probability", "gabarito"]].to_csv(ROOT / f"data/exp2_sage_{mode}{suffix}.csv")
            other_path = ROOT / f"data/exp2_tres_revisores_{mode}.csv"
            if other_path.exists():
                other = pd.read_csv(other_path).set_index("pair_id")
                for reviewer in ("Haiku 5.5", "DeepSeek V4.1 Flash", "Kimi K2.5", "Kimi K3"):
                    if reviewer not in other:
                        continue
                    common = frame.join(other[[reviewer]], how="inner")
                    valid = common[common[reviewer].isin(["MATCH", "NONMATCH", "UNSURE"])]
                    original = valid.copy()
                    original["decision"] = original[reviewer]
                    comparisons.append({**label, "revisor": reviewer, "fonte": str(other_path.relative_to(ROOT)),
                                        "n_common": len(valid), "agreement": float((valid.decision == valid[reviewer]).mean()) if len(valid) else None,
                                        **{"sage_" + k: v for k, v in rates(valid).items()},
                                        **{"outro_" + k: v for k, v in rates(original).items()}})
                    for pid, pair in valid.iterrows():
                        pair_comparisons.append({**label, "pair_id": pid, "revisor": reviewer,
                            "sage_decisao": pair["decision"], "outro_decisao": pair[reviewer],
                            "gabarito": int(pair["y"]), "p_match": float(pair["p_match"])})
            # Quantitative fields returned by provider; absent is not zero cost.
            telemetry.append({**label, "n": len(frame), "metadata": [
                {"pair_id": r["pair_id"], "usage": r.get("usage"),
                 **{k: v for k, v in r["meta"].items() if k not in ("model", "latency_ms", "reasoning")}}
                for r in run["results"]]})
            print("SUMMARY " + json.dumps(metric, ensure_ascii=False, allow_nan=False), flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "data/exp2_sage_resumo.csv", index=False)
    pd.DataFrame(reliability).to_csv(ROOT / "data/exp2_sage_confiabilidade.csv", index=False)
    pd.DataFrame(thresholds).to_csv(ROOT / "data/exp2_sage_limiares.csv", index=False)
    pd.DataFrame(comparisons).to_csv(ROOT / "data/exp2_sage_comparacao.csv", index=False)
    pd.DataFrame(pair_comparisons, columns=["modo", "braco", "modelo", "pair_id", "revisor", "sage_decisao",
                                          "outro_decisao", "gabarito", "p_match"]).to_csv(
                                              ROOT / "data/exp2_sage_comparacao_pares.csv", index=False)
    save(ROOT / "data/sprint3b/sage_sem_mo/bloqueio_operacional.json", {"updated_at": now(), "runs": operational,
        "quota_exhausted": any(r["status"] == "quota_exhausted" for r in operational),
        "all_present_runs_complete": all(r["status"] == "complete" for r in operational),
        "note": "Historical errors remain for provenance; status and counts describe the latest checkpoint, not a running process."})
    anchoring = []
    if run_path("vigilancia", "com_mo").exists():
        controls = json.loads(run_path("vigilancia", "com_mo").read_text(encoding="utf-8"))
        ablated = json.loads(run_path("vigilancia", "sem_mo").read_text(encoding="utf-8"))
        before = {r["pair_id"]: r for r in controls["results"]}
        after = {r["pair_id"]: r for r in ablated["results"]}
        for pid in sorted(before.keys() & after.keys()):
            anchoring.append({"pair_id": pid, "gabarito": int(g.loc[pid, "y"]),
                              "com_mo": before[pid]["decision"], "sem_mo": after[pid]["decision"],
                              "p_match_com_mo": before[pid]["p_match"], "p_match_sem_mo": after[pid]["p_match"]})
        pd.DataFrame(anchoring).to_csv(ROOT / "data/exp2_sage_ancoragem.csv", index=False)
    save(ROOT / "data/sprint3b/sage_sem_mo/telemetria.json", {"runs": telemetry})
    report = ["# Exp2 — Sage (Levanto)", "", "## Método", "",
              "Coortes e ordem dos JSONs Anthropic originais: 1.410 vigilância, 435 confirmação. "
              "Gabarito reconstruído de COMPARADORSEMIDENT.csv (dtype=str), PAR∈{1,2}; "
              "SHA256(COMPREC|REFREC|PASSO)[:16], campos aparados. Dossiês originais gzcmd/llm_dossier.py; "
              "sem_mo remove o bloco model_outputs inteiro. A seleção é a coorte histórica fixa, não a triagem "
              "recalculada hoje. Os identificadores do pipeline são alinhados aos campos string originais "
              "por junção normalizada validada um-para-um; somente o hash string original é persistido. "
              "Não se salvam dossiês, PII, chaves ou raciocínio livre.", "",
              "Uma decisão choice por par (não o protocolo A+B+árbitro); opções fixas match/nonmatch. "
              "reasoning=auto, grounding.trigger=never. Instruções originais + adaptação por modo registradas nos JSONs. "
              "Vigilância tolera incerteza moderada a favor do vínculo; confirmação exige evidência forte. "
              "chosen=null é UNSURE, sem retry. Checkpoint a cada par, hash da requisição na retomada; "
              "429/500/502/503/504 têm backoff/Retry-After; 404 recebe até três tentativas com backoff conforme diagnóstico solicitado; "
              "402 interrompe imediatamente a sequência. Concorrência padrão 1 para parada sem outras chamadas em voo.", "",
              "Compatibilidade de transporte: após HTTP400 do deployment confidencial, a retomada pode representar "
              "grounding desligado como null (campo nullable, default null no OpenAPI oficial), em vez de "
              "{trigger: never}. A mudança é registrada em wire_encoding_changes, com hash das respostas preservadas; "
              "os hashes canônicos dos dossiês/instruções não são alterados. Cada nova resposta é verificada para "
              "ausência de ativação de grounding.", "",
              "## Probabilidades e denominadores", "",
              "Segundo https://docs.levanto.ai/decision-model/choice.md, probability=P(chosen correto), "
              "não P(match); pode ser null na abstenção. O valor bruto é preservado. AUC/Brier e confiabilidade "
              "principais usam P(match) em probabilities[match], inclusive UNSURE. Os campos *_raw_chosen_vs_y "
              "atendem ao diagnóstico literal solicitado, mas NÃO são métricas válidas de risco de vínculo, pois "
              "a classe-alvo muda conforme chosen. Limiares 0,8/0,9/0,95/0,98 aplicam-se à confiança da escolha "
              "(probability), com abstenção abaixo; não são limiares P(match). Curva por decis fixos [0,0.1), ..., [0.9,1].", "",
              "sens/espec_sem_unsure excluem abstenções; *_unsure_como_erro mantêm todos os gabaritos "
              "nos denominadores sem dar crédito por abstenção. Também são reportadas taxas com UNSURE como vínculo. "
              "Denominador zero = NA. Execuções parciais não representam a coorte completa. Comparação dos outros "
              "revisores é restrita aos mesmos pares concluídos; diferenças incluem transporte/protocolo e não isolam modelo.", "",
              "## Resultados (NA quando não há observações)", "",
              "| Modo | Braço | Modelo observado | Status | N/esperado | VP | FP | FN | VN | UNSURE | AUC P(match) | Brier |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        report.append(f"| {r['modo']} | {r['braco']} | {r['modelo']} | {r['status']} | {r['n']}/{r['n_expected']} | "
                      f"{r['VP']} | {r['FP']} | {r['FN']} | {r['VN']} | {r['UNSURE']} | {r['auc']} | {r['brier']} |")
    fmt = lambda value: "NA" if value is None else f"{value:.6g}"
    report += ["", "### Sensibilidade e especificidade (proporções)", "",
               "| Modo/braço | Sens. sem U | Espec. sem U | Sens. U=erro | Espec. U=erro |",
               "|---|---:|---:|---:|---:|"]
    for r in rows:
        report.append(f"| {r['modo']}/{r['braco']} | {fmt(r['sens_sem_unsure'])} | {fmt(r['espec_sem_unsure'])} | "
                      f"{fmt(r['sens_unsure_como_erro'])} | {fmt(r['espec_unsure_como_erro'])} |")
    report += ["", "### Limiar próprio na confiança da escolha", "",
               "| Modo/braço | Limiar | VP | FP | FN | VN | U | Sens. sem U | Espec. sem U |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in thresholds:
        report.append(f"| {r['modo']}/{r['braco']} | {r['limiar_confidence_chosen']} | {r['VP']} | {r['FP']} | "
                      f"{r['FN']} | {r['VN']} | {r['UNSURE']} | {fmt(r['sens_sem_unsure'])} | {fmt(r['espec_sem_unsure'])} |")
    report += ["", "### Comparação nos mesmos pares concluídos", "",
               "| Modo/braço | Revisor original | N comum | Concordância | VP/FP/FN/VN/U original |",
               "|---|---|---:|---:|---|"]
    for r in comparisons:
        counts = "/".join(str(r["outro_" + field]) for field in ("VP", "FP", "FN", "VN", "UNSURE"))
        report.append(f"| {r['modo']}/{r['braco']} | {r['revisor']} | {r['n_common']} | {fmt(r['agreement'])} | {counts} |")
    for run in rows:
        present = {r["revisor"] for r in comparisons if r["modo"] == run["modo"] and r["braco"] == run["braco"]}
        kimi = "Kimi K2.5" if run["modo"] == "vigilancia" else "Kimi K3"
        missing = {"Haiku 5.5", "DeepSeek V4.1 Flash", kimi} - present
        if missing:
            report.append(f"Comparação indisponível no CSV fornecido para {run['modo']}/{run['braco']}: "
                          + ", ".join(sorted(missing)) + ". Não se imputam decisões nem se usa outro provedor.")
    report += ["", "O CSV fornecido de confirmação identifica Kimi K3, não K2.5; a comparação mantém esse rótulo "
               "e restringe-se aos pares com decisão disponível. Não se fundem versões Kimi nem se imputam ausências. "
               "As decisões pareadas, gabarito e P(match) estão em data/exp2_sage_comparacao_pares.csv; "
               "data/exp2_sage_comparacao.csv contém os agregados por revisor/modo/braço."]
    failures = [r for r in rows if r["n_errors"] and r["status"] != "complete"]
    if failures:
        report += ["", "## Bloqueio operacional", ""]
        for r in failures:
            report.append(f"- {r['modo']}/{r['braco']}: último status HTTP {r['last_http_status']}, "
                          f"{r['n_errors']} erro(s), {r['n']}/{r['n_expected']} decisões concluídas. "
                          "Corpos literais disponíveis, quando capturados na execução diagnóstica, constam no JSON "
                          "e nos logs (segredos redigidos). A falha original não capturou o corpo. "
                          "Não se retentam 400/401/402; 404/429/500/502/503/504 param após três tentativas.")
        report += ["", "N=0 significa ausência de observações: as contagens zero NÃO demonstram ausência de erros "
                   "do revisor. Sensibilidade, especificidade, AUC, Brier e latência são NA; "
                   "nenhum meta.model foi observado sem resposta de decisão bem-sucedida. "
                   "O braço com_mo não é iniciado sem conclusão dos dois modos sem_mo."]
    if anchoring:
        transitions = Counter(f"{r['com_mo']}->{r['sem_mo']}" for r in anchoring)
        paired_metrics = anchoring_metrics(anchoring)
        save(ROOT / "data/exp2_sage_ancoragem_resumo.json", paired_metrics)
        report += ["", "## Ancoragem — vigilância pareada", "",
                   f"Pares em comum: {len(anchoring)}. Transições com_mo → sem_mo: "
                   + json.dumps(dict(transitions), ensure_ascii=False) + ". "
                   "CSV pareado: data/exp2_sage_ancoragem.csv. A ordem fixa (sem_mo primeiro) "
                   "e uma única realização por braço limitam interpretação causal/variabilidade.",
                   "McNemar exato bilateral: MATCH vs demais; endpoint secundário FN explícito vs demais somente "
                   "nos vínculos ouro. UNSURE não é correção estrita; reversões FN→MATCH e FN→UNSURE separadas. "
                   "Sem ajuste de multiplicidade; efeito não isola p_cal de seus proxies.",
                   "```json", json.dumps(paired_metrics, ensure_ascii=False, indent=2), "```"]
    report += ["", "## Latência, consumo e limites", "",
               "Distribuição min/p50/p90/p95/p99/max/média de latência da API e tempo do cliente no resumo CSV. "
               "Tokens/custo quantitativos são preservados se devolvidos, em telemetria.json e nos metadados por par; "
               "somas por campo e número de decisões com o campo informado constam no resumo CSV (api_*). "
               "campos ausentes não significam consumo zero. A estimativa de um dossiê fica em preflight.json, "
               "não é medição de todo o lote nem fatura. Não extrapolar custo sem tarifa/consumo observados. "
               "Resumo, curva de confiabilidade, limiares e comparação estão nos CSVs data/exp2_sage_*.csv.", ""]
    for r in rows:
        report += [f"- {r['modo']}/{r['braco']}: latência API p50={fmt(r['latency_ms_p50'])} ms, "
                   f"p95={fmt(r['latency_ms_p95'])} ms; "
                   f"input_tokens={fmt(r.get('api_usage_input_tokens_sum'))}, "
                   f"output_tokens={fmt(r.get('api_usage_output_tokens_sum'))}. "
                   "Totais das decisões checkpointadas, não uma fatura incluindo sondagens e eventuais falhas. "
                   "Preço do plano/custo USD não foi informado pela API; nenhum custo em dólares é inferido."]
        report += [f"  Chamadas /decide rastreadas após inclusão do contador: {r.get('transport_decide_calls_tracked', 'NA')}; "
                   f"limite inferior histórico anterior: {r.get('transport_decide_calls_before_tracking_lower_bound', 'NA')}. "
                   "O histórico anterior não tinha journaling por tentativa; não se afirma total exato faturado."]
    preflight = ROOT / "data/sprint3b/sage_sem_mo/preflight.json"
    if preflight.exists():
        report += ["## Preflight / interrupção", "", "```json", preflight.read_text(encoding="utf-8"), "```", ""]
    probe = ROOT / "data/sprint3b/sage_sem_mo/quota_probe.json"
    if probe.exists():
        report += ["## Sondagem mínima antes da retomada", "", "```json", probe.read_text(encoding="utf-8"), "```", "",
                   "Uma sondagem curta HTTP 200 apenas confirma que essa requisição foi aceita; "
                   "não garante cota suficiente para o dossiê real. Respostas existentes são preservadas na retomada."]
    diagnostic = ROOT / "data/sprint3b/sage_sem_mo/diagnostico_http.json"
    if diagnostic.exists():
        report += ["## Diagnóstico HTTP isolado", "",
                   "Quickstart oficial confirma https://sage.levanto.ai/decide e Authorization: Bearer. "
                   "A chave do .env tem prefixo lv_live_ (somente os oito caracteres públicos foram exibidos). "
                   "Sondagens sintéticas yesno e choice com reasoning=auto retornaram 200, modelo levanto-sage-2; "
                   "não foi identificada divergência de host/caminho. O 404 inicial não prova rota inexistente. "
                   "Status, URL final e respostas literais das sondagens estão em diagnostico_http.json. "
                   "Variantes /v1/decide são testadas apenas se a sondagem /decide retornar 404.", ""]
    (ROOT / "data/exp2_sage_relatorio.md").write_text("\n".join(report), encoding="utf-8")


def audit(key):
    paths = [Path(__file__), ROOT / "scripts/diagnosticar_exp2_sage.py", ROOT / "scripts/probe_exp2_sage_quota.py", ROOT / "tests/test_exp2_sage.py",
             *ROOT.glob("data/exp2_sage_*"),
             *ROOT.glob("data/sprint3b/sage_sem_mo/*"), *ROOT.glob("data/sprint3b/sage_com_mo/*")]
    scanned = 0
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if key and key in text:
            raise ValueError("Secret audit failed (value suppressed)")
        if path.name.startswith("exp2_llm_sync_") and path.suffix == ".json":
            run = json.loads(text)
            allowed = {"pair_id", "decision", "chosen", "confidence", "probability", "probabilities", "p_match",
                       "meta", "latency_ms", "total_latency_s", "total_tokens", "reason_codes", "quality_flags",
                       "protocol", "usage", "reasoning_text_persisted"}
            for result in run["results"]:
                if not set(result) <= allowed or not re.fullmatch(r"[0-9a-f]{16}", result["pair_id"]):
                    raise ValueError("Unapproved result field or identifier")
                if result["reasoning_text_persisted"] or result["reason_codes"] or result["quality_flags"]:
                    raise ValueError("Unexpected free-form result metadata")
        if '"content":' in text and path.suffix == ".json":
            # The isolated HTTP probe intentionally persists one public,
            # synthetic sentence. It is not a reconstructed pair dossier.
            if path.name != "diagnostico_http.json":
                raise ValueError("Dossier content persisted")
            def check_synthetic(value):
                if isinstance(value, dict):
                    for field, child in value.items():
                        if field == "content" and child != "The light is green.":
                            raise ValueError("Unexpected diagnostic content")
                        check_synthetic(child)
                elif isinstance(value, list):
                    for child in value:
                        check_synthetic(child)
            check_synthetic(json.loads(text))
        scanned += 1
    record = {"timestamp": now(), "files_scanned": scanned, "secret_matches": 0,
              "dossier_content_saved": False, "synthetic_diagnostic_content_allowed": "The light is green.",
              "free_reasoning_text_saved": False}
    save(ROOT / "data/sprint3b/sage_sem_mo/auditoria.json", record)
    print("AUDIT " + json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modo", "--mode", choices=MODES, default="vigilancia")
    parser.add_argument("--arm", choices=("sem_mo", "com_mo"), default="sem_mo")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workflow", action="store_true", help="Ready, estimate, 3/mode smoke, full runs, anchoring if complete")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--resume-after-quota", action="store_true", help="Explicitly resume after a successful single minimal quota probe")
    parser.add_argument("--grounding-null", action="store_true", help="Encode disabled grounding as OpenAPI null for confidential deployment compatibility")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    load_dotenv(ROOT / ".env")
    logging.disable(logging.CRITICAL)
    key = os.environ.get("SAGE_API_KEY", "")
    if args.summarize_only:
        summarize()
        audit(key)
        return
    if not key:
        raise SystemExit("SAGE_API_KEY missing (value suppressed)")
    if args.resume_after_quota:
        probe_path = ROOT / "data/sprint3b/sage_sem_mo/quota_probe.json"
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(probe["timestamp"])).total_seconds()
        if probe["status"] != 200 or not 0 <= age < 86400:
            raise SystemExit("Resume requires a recent successful quota_probe.json (no new API call made)")
    api = Sage(key)
    api.grounding_null = args.grounding_null
    if not args.workflow:
        prepared = prepare(args.modo, args.arm)
        execute_run(api, args.modo, args.arm, *prepared, args.limit, args.workers, args.resume_after_quota)
        summarize()
        audit(key)
        return
    folder = ROOT / "data/sprint3b/sage_sem_mo"
    preflight_path = folder / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.exists() else {"started_at": now()}
    prepared = {}
    # Prepare both empty checkpoints even if the first charge fails, so the
    # report documents the expected second cohort explicitly as not run.
    for mode in MODES:
        print(f"PREPARE_START {mode} {now()}", flush=True)
        prepared[mode] = prepare(mode, "sem_mo")
        init_run(mode, "sem_mo", *prepared[mode])
        print(f"PREPARE_DONE {mode} n={len(prepared[mode][0])} {now()}", flush=True)
    try:
        ready = api.call("/ready", auth=False)
        preflight["ready"] = {k: v for k, v in ready.items() if k in ("status", "load") and v in ("ready", "low", "medium", "high")}
        print("READY " + json.dumps(preflight["ready"]), flush=True)
        if ready.get("status") != "ready":
            raise APIError("not_ready")
        save(preflight_path, preflight)
        if "estimate" not in preflight:
            first = prepared["vigilancia"][0][0]
            estimate = api.call("/usage/estimate", prepared["vigilancia"][1][first])
            preflight["estimate"] = safe_numeric_tree(estimate)
            preflight["estimate_pair_id"] = first
            preflight["estimate_raw_keys"] = sorted(estimate)
            print("ESTIMATE " + json.dumps(preflight["estimate"]), flush=True)
            save(preflight_path, preflight)
        for mode in (() if args.resume_after_quota else MODES):
            log_path = folder / f"run_{mode}.log"
            with log_path.open("a", encoding="utf-8", buffering=1) as log, redirect_stdout(log):
                print(f"SMOKE {now()} {mode}", flush=True)
                run = execute_run(api, mode, "sem_mo", *prepared[mode], 3, args.workers)
            print(f"SMOKE {mode} n={run['n_reviewed']} status={run['status']}", flush=True)
            if api.stop.is_set():
                break
        if not api.stop.is_set():
            for mode in MODES:
                with (folder / f"run_{mode}.log").open("a", encoding="utf-8", buffering=1) as log, redirect_stdout(log):
                    print(f"FULL {now()} {mode}", flush=True)
                    run = execute_run(api, mode, "sem_mo", *prepared[mode], 0, args.workers, args.resume_after_quota)
                print(f"FULL {mode} n={run['n_reviewed']} status={run['status']}", flush=True)
                if api.stop.is_set() or run["status"] != "complete":
                    break
        complete = all(json.loads(run_path(m, "sem_mo").read_text(encoding="utf-8"))["status"] == "complete" for m in MODES)
        if complete and not api.stop.is_set():
            control = prepare("vigilancia", "com_mo")
            out = run_path("vigilancia", "com_mo").parent
            out.mkdir(parents=True, exist_ok=True)
            with (out / "run_vigilancia.log").open("a", encoding="utf-8", buffering=1) as log, redirect_stdout(log):
                execute_run(api, "vigilancia", "com_mo", *control, 0, args.workers, args.resume_after_quota)
    except APIError as exc:
        preflight["error_status"] = exc.status
        print(f"PREFLIGHT_ERROR status={exc.status}", flush=True)
    finally:
        preflight.setdefault("execution_history", []).append({"finished_at": now(),
            "resume_after_quota": args.resume_after_quota, "quota_exhausted": api.quota.is_set(),
            "calls": dict(api.calls), "completed": {m: len(json.loads(run_path(m, "sem_mo").read_text(encoding="utf-8"))["results"]) for m in MODES}})
        preflight.update({"updated_at": now(), "quota_exhausted": api.quota.is_set(), "calls_this_process": dict(api.calls),
                          "http_diagnostics_this_process": api.http_diagnostics,
                          "completed": {m: len(json.loads(run_path(m, "sem_mo").read_text(encoding="utf-8"))["results"]) for m in MODES}})
        save(preflight_path, preflight)
        summarize()
        audit(key)


if __name__ == "__main__":
    main()
