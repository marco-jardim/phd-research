"""New Anthropic-only Exp2 runner: v4, output-free dossiers and prompt caching.

Historical hash-pinned runners stay untouched. Only allowlisted structured
results, pseudonymous IDs, hashes and usage counters are persisted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_exp2_ablacao_dossie as abl
import run_exp2_llm_sync as exp2
from gzcmd import anthropic_transport as ant
from gzcmd.llm_review import VALID_REASON_CODES
from gzcmd.llm_usage_cost import price_usage

FORBIDDEN = {"model_outputs", "p_cal", "base_choice", "band", "evr", "guardrail", "guardrail_reason"}


class WarmupError(RuntimeError):
    def __init__(self, error):
        super().__init__("Cache warmup failed; workers were not released")
        self.error = error


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def configure_provider(provider):
    if provider != "anthropic":
        raise ValueError("Somente Anthropic é permitido; Fireworks é proibido neste projeto")


def configure_prompt(version):
    if version != "v4":
        raise ValueError("O runner com cache exige o prompt v4")
    path = ROOT / "gzcmd/gzcmd_v4_llm_prompt.md"
    ant.llm_review._PROMPT_PATH = path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"prompt_version": version, "prompt_file_sha256": digest, "prompt_sha256": digest}


def retry_file_operation(operation):
    for attempt in range(12):
        try:
            return operation()
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.02 * (2 ** attempt), .5))


def save(path, value):
    """Same Windows-safe checkpoint boundary as the historical output-free run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    retry_file_operation(lambda: tmp.write_text(content, encoding="utf-8"))
    for delay in (0, .2, .4, .8, 1.6, 3):
        if delay:
            time.sleep(delay)
        try:
            tmp.replace(path)
            return
        except PermissionError:
            pass
    alternate = path.with_name(path.name + f".fallback-{time.time_ns()}.json")
    retry_file_operation(lambda: alternate.write_text(content, encoding="utf-8"))
    print(f"CHECKPOINT_FALLBACK {path.name} {alternate.name}", flush=True)


def recover_candidates(candidates, *, result_checkpoint=False):
    existing, valid = [], []
    for candidate in candidates:
        try:
            existing.append((candidate.stat().st_mtime_ns, candidate))
        except FileNotFoundError:
            continue
    for modified_at, candidate in sorted(existing, key=lambda item: item[0], reverse=True):
        try:
            value = retry_file_operation(lambda: json.loads(candidate.read_text(encoding="utf-8")))
            if not result_checkpoint:
                return value
            valid.append(((len(value.get("results", [])), modified_at), value))
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            continue
    return max(valid, key=lambda item: item[0])[1] if valid else None


def recover_json(path, *, result_checkpoint=False):
    return recover_candidates([path, path.with_suffix(".tmp"),
                               *path.parent.glob(path.name + ".fallback-*.json")],
                              result_checkpoint=result_checkpoint)


def load_ids(path):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            ids = [row["pair_id"] for row in csv.DictReader(stream)]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("results", data.get("pair_ids", data.get("selected_pair_ids", [])))
        ids = [row["pair_id"] if isinstance(row, dict) else row for row in data]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r"[0-9a-f]{16}", p) for p in ids):
        raise ValueError("Expected unique SHA256[:16] pair IDs")
    return ids


def audit_view(view):
    def check(value):
        if isinstance(value, dict):
            assert not FORBIDDEN.intersection(value), "Engine output survived ablation"
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    assert set(view.payload) == {"schema_version", "pair_id", "features"}
    check(view.payload)


def prepare(mode, ids):
    df, _ = exp2.run_v3(input_csv=str(ROOT / "data/COMPARADORSEMIDENT.csv"),
                        config_path=str(ROOT / "gzcmd/gzcmd_v3_config.yaml"), mode=mode,
                        p_cal="fit_platt", policy_version="v3")
    dossiers = {d.pair_id: d for d in exp2.build_dossiers(df, only_llm_review=False)}
    if set(ids) - dossiers.keys():
        raise ValueError("Historical cohort missing rebuilt dossiers")
    views = {pid: abl.DossierView(dossiers[pid], True) for pid in ids}
    for view in views.values():
        audit_view(view)
    return views


class CachedReviewer(ant.AnthropicReviewer):
    def _parse_response(self, text):
        try:
            return super()._parse_response(text)
        except (ValueError, KeyError):
            abl._LOCAL.stats["response_parse_errors"] += 1
            abl.persist_transport(abl._LOCAL.stats)
            raise


def review_pair(pid, view, model, journal):
    stat_path = journal / f"{pid}.transport.json"
    stats = recover_json(stat_path) or abl.meter()
    abl._LOCAL.stats, abl._LOCAL.stats_path = stats, stat_path
    reviewer = CachedReviewer(api_key=os.environ["ANTHROPIC_API_KEY"],
                              base_url="https://api.anthropic.com", model=model,
                              fallback_model=model, max_tokens=2500, timeout_s=120,
                              max_retries=5, retry_delay_s=3)
    client = reviewer._get_client()

    class Facade:
        messages = abl.MeteredMessages(client.messages, stats)

    reviewer._client = Facade()
    try:
        row = abl.safe_result(reviewer.review(view))
        assert row["decision"] in abl.DECISIONS
        assert set(row["reason_codes"]) <= VALID_REASON_CODES
        save(journal / f"{pid}.result.json", row)
        return row, None
    except Exception as exc:
        return None, {"pair_id": pid, "type": type(exc).__name__, "status": getattr(exc, "status", None)}
    finally:
        client.close()


def aggregate(journal):
    total, groups = abl.meter(), {}
    for path in journal.iterdir():
        if ".transport." in path.name:
            groups.setdefault(path.name[:16], []).append(path)
    for candidates in groups.values():
        stats = recover_candidates(candidates)
        if stats is None:
            continue
        for key, value in stats.items():
            if key == "error_statuses":
                for status, count in value.items():
                    total[key][status] = total[key].get(status, 0) + count
            elif key == "max_request_input_tokens":
                total[key] = max(total[key], value)
            else:
                total[key] += value
    return total


class WarmupExecutor:
    """Resolve the first submitted pair before even constructing the worker pool."""

    def __init__(self, max_workers, warmup=True, executor_factory=ThreadPoolExecutor):
        self.max_workers, self.warmup = max_workers, warmup
        self.executor_factory, self.pool = executor_factory, None

    def __enter__(self):
        return self

    def submit(self, function, *args):
        if self.warmup:
            self.warmup = False
            value = function(*args)
            if value[1] is not None:
                raise WarmupError(value[1])
            future = Future()
            future.set_result(value)
            return future
        if self.pool is None:
            self.pool = self.executor_factory(max_workers=self.max_workers)
        return self.pool.submit(function, *args)

    def __exit__(self, *exc_info):
        if self.pool is not None:
            self.pool.shutdown(wait=True)


def run(args):
    configure_provider(args.provider)
    prompt_metadata = configure_prompt(args.prompt_version)
    prompt_caching = ant.caching_metadata(args.cache_warmup)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("Missing ANTHROPIC_API_KEY (value suppressed)")
    if args.modelo is None:
        args.modelo = "claude-haiku-5-5"
    ids = load_ids(args.pares)
    original = ROOT / f"data/sprint3b/anthropic/exp2_llm_sync_{args.modo}.json"
    if not args.allow_new_cohort and set(ids) != set(load_ids(original)):
        raise ValueError("--pares must identify the historical cohort; use --allow-new-cohort for a new lot")
    views = prepare(args.modo, ids)
    fingerprints = {pid: hashlib.sha256(v.to_json(sort_keys=True).encode()).hexdigest()
                    for pid, v in views.items()}
    path = args.saida
    journal = path.parent / (path.stem + "_journal")
    journal.mkdir(parents=True, exist_ok=True)
    previous = recover_json(path, result_checkpoint=True) or {}
    if previous and previous.get("prompt_caching") != prompt_caching:
        raise ValueError("Use uma saída nova para o transporte com cache")
    identity = {"mode": args.modo, "model": args.modelo, "dossier_sha256": fingerprints,
                "prompt_sha256": prompt_metadata["prompt_sha256"], "prompt_version": args.prompt_version,
                "selected_pair_ids": ids, "source_sha256": hashlib.sha256(args.pares.read_bytes()).hexdigest(),
                "cohort": "v31_lot" if args.allow_new_cohort else "historical"}
    if previous and any(previous.get(k) != v for k, v in identity.items()):
        raise ValueError("Checkpoint identity changed; refusing mixed execution")
    stored_identity = recover_json(journal / "identity.json")
    if stored_identity is not None and stored_identity != identity:
        raise ValueError("Journal identity changed; refusing mixed execution")
    stored_caching = recover_json(journal / "prompt_caching.json")
    has_results = any(".result." in item.name for item in journal.iterdir())
    if ((stored_caching is not None and stored_caching != prompt_caching)
            or (has_results and (stored_identity is None or stored_caching is None))):
        raise ValueError("Journal requires the same identity and prompt caching policy")
    save(journal / "identity.json", identity)
    save(journal / "prompt_caching.json", prompt_caching)
    results = {row["pair_id"]: row for row in previous.get("results", [])}
    for pid in {p.name[:16] for p in journal.iterdir() if ".result." in p.name}:
        row = recover_json(journal / f"{pid}.result.json")
        assert row is not None and row["pair_id"] in ids
        results[row["pair_id"]] = row
    errors = {row["pair_id"]: row for row in previous.get("errors", []) if row["pair_id"] not in results}
    output = {**identity, "experiment": "exp2_llm_sync", "timestamp": previous.get("timestamp", utcnow()),
              "n_review_pairs": len(ids), "workers": args.workers, "removed_fields": abl.REMOVED_FIELDS,
              "prompt_caching": prompt_caching,
              "metadata": {"provider": "Anthropic Messages API", "model_exact": args.modelo,
                           "max_tokens": 2500, "temperature_sent": False, "extended_thinking": False,
                           "max_retries": 5, "retry_delay_s": 3, "timeout_s": 120,
                           "protocol": "unchanged Agent-A + Agent-B + arbiter on disagreement",
                           **prompt_metadata, "prompt_caching": prompt_caching,
                           "free_response_fields_saved": False}}
    started, base_elapsed = time.monotonic(), previous.get("elapsed_s", 0)

    def checkpoint():
        output.update(results=[results[p] for p in ids if p in results],
                      errors=[errors[p] for p in ids if p in errors],
                      failed_pair_ids=[p for p in ids if p not in results],
                      n_reviewed=len(results), n_errors=len(errors), complete=len(results) == len(ids),
                      elapsed_s=round(base_elapsed + time.monotonic() - started, 2), updated_at=utcnow())
        output["transport"] = aggregate(journal)
        output["cost_estimate_usd"] = price_usage(output["transport"])
        output["cost_note"] = "Estimate only, not invoice; standard/global cache prices, context <=100k."
        cache_usage = {field: output["transport"][field] for field in
                       ("cache_read_input_tokens", "cache_creation_input_tokens")}
        output["metadata"].update(cache_usage)
        save(path, output)
        save(path.parent / "execucao.json", {"mode": args.modo, "output": path.name,
             "prompt_caching": prompt_caching, "transport": output["transport"], **cache_usage,
             "prompt_sha256": identity["prompt_sha256"], "updated_at": output["updated_at"]})

    # MeteredMessages persists every call through this same Windows-safe boundary.
    abl.save = save
    checkpoint()
    chosen = ids[:args.limit] if args.limit else ids
    pending = iter(pid for pid in chosen if pid not in results)
    consecutive_errors, stop = 0, False
    with WarmupExecutor(args.workers, warmup=args.cache_warmup) as pool:
        active = {}

        def submit():
            pid = next(pending, None)
            if pid is not None:
                active[pool.submit(review_pair, pid, views[pid], args.modelo, journal)] = pid

        try:
            for _ in range(args.workers):
                submit()
        except WarmupError as exc:
            errors[exc.error["pair_id"]] = exc.error
            checkpoint()
            raise
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                pid = active.pop(future)
                row, error = future.result()
                if error:
                    errors[pid] = error
                    consecutive_errors += 1
                    stop = stop or error["type"] == "FatalAPIError" or consecutive_errors >= 2
                else:
                    results[pid] = row
                    errors.pop(pid, None)
                    consecutive_errors = 0
                checkpoint()
                if not stop:
                    submit()
    checkpoint()
    if stop:
        raise RuntimeError("Stopped after fatal/two consecutive errors; sanitized errors checkpointed")
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("anthropic",), default="anthropic")
    parser.add_argument("--modelo", default="claude-haiku-5-5")
    parser.add_argument("--modo", choices=("vigilancia", "confirmacao"), required=True)
    parser.add_argument("--pares", type=Path, required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--saida", type=Path)
    output.add_argument("--saida-dir", type=Path)
    parser.add_argument("--prompt-version", choices=("v4",), default="v4")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-new-cohort", action="store_true")
    parser.add_argument("--cache-warmup", action=argparse.BooleanOptionalAction, default=True,
                        help="Primeiro par sozinho antes de liberar workers (padrão ligado)")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.saida_dir is not None:
        args.saida = args.saida_dir / f"exp2_llm_sync_{args.modo}.json"
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as exc:
        # Provider exception messages can contain text/credentials.
        print("FATAL " + type(exc).__name__, flush=True)
        sys.exit(1)
