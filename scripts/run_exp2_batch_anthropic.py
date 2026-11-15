"""Transporte local retomável para o probe: mensagens históricas, teto com reservas.

Uso: --pilot, --main ou --availability. Um único processo por diretório.
Nenhum texto livre de resposta ou dossiê é persistido.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_exp2_sem_model_outputs as sem
from probe_frontier_preparar import MODELS, OUT, SEED

RATES = {MODELS[0]: (4., 20., .20), MODELS[1]: (10., 50., .25), MODELS[2]: (.10, .50, .01)}
llm = sem.ant.llm_review


def read(path):
    return sem.recover_json(path)


def save(path, value):
    sem.save(path, value)
    # Never proceed to a paid call unless a recoverable checkpoint is durable.
    if read(path) != value:
        raise RuntimeError("Durable checkpoint unavailable")


def cost(model, usage, batch=False):
    inp, out, hit = RATES[model]
    creation = usage.get("cache_creation", {}) or {}
    onehour = creation.get("ephemeral_1h_input_tokens", 0)
    writes = usage.get("cache_creation_input_tokens", 0)
    total = (usage.get("input_tokens", 0) * inp + usage.get("output_tokens", 0) * out
             + usage.get("cache_read_input_tokens", 0) * hit
             + max(0, writes - onehour) * inp * 1.25 + onehour * inp * 2)
    return total / 1e6 * (.5 if batch else 1.)


class Ledger:
    def __init__(self, path=OUT / "custo.json"):
        self.path = path
        recovered = read(path)
        if path.exists() and recovered is None:
            raise ValueError("Unrecoverable cost ledger; never reset spending")
        self.data = recovered if recovered is not None else {"spent_usd": 0., "reservations": {}, "calls": {}}
        if abs(self.data["spent_usd"] - sum(v["cost_usd"] for v in self.data["calls"].values())) > 1e-9:
            raise ValueError("Cost ledger sum mismatch")

    def persist(self):
        self.data["updated_at"] = sem.utcnow()
        save(self.path, self.data)

    def reserve(self, key, amount):
        if key in self.data["reservations"]:
            raise RuntimeError("Ambiguous previous submission: reconciliation required")
        total = self.data["spent_usd"] + sum(self.data["reservations"].values()) + amount
        if total > 90 + 1e-9:
            raise RuntimeError("Hard budget: requested reservation would exceed USD90")
        self.data["reservations"][key] = amount
        self.persist()

    def release(self, key):
        self.data["reservations"].pop(key, None)
        self.persist()

    def record(self, key, model, usage, batch, stop_reason):
        entry = {"model": model, "usage": usage, "batch": batch, "stop_reason": stop_reason,
                 "cost_usd": cost(model, usage, batch)}
        if key in self.data["calls"]:
            if self.data["calls"][key] != entry:
                raise ValueError("Usage identity mismatch")
            return
        self.data["calls"][key] = entry
        self.data["spent_usd"] = sum(v["cost_usd"] for v in self.data["calls"].values())
        self.persist()


@contextmanager
def exclusive():
    path = OUT / "process.lock"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink()


def request_params(model, messages):
    # Exactly the historical AnthropicReviewer flattening; batch_mode is NOT enabled.
    return {"model": model, "system": "\n\n".join(m["content"] for m in messages if m["role"] == "system"),
            "messages": [{"role": "user", "content": "\n\n".join(m["content"] for m in messages if m["role"] == "user")}],
            "max_tokens": 2500}


def retry_control(call):
    for attempt in range(5):
        try:
            return call()
        except anthropic.APIStatusError as exc:
            if exc.status_code not in (429, 529) or attempt == 4:
                raise
            time.sleep(3 * 2 ** attempt)


def maximum(client, params, batch):
    count = retry_control(lambda: client.messages.count_tokens(
        model=params["model"], system=params["system"], messages=params["messages"]))
    if count.input_tokens > 100000:
        raise ValueError("Unexpected long context tariff")
    # Allow for small server-side token-count variance; output has a strict max_tokens.
    return cost(params["model"], {"input_tokens": int(count.input_tokens * 1.02) + 32,
                                  "output_tokens": params["max_tokens"]}, batch)


def parse(message, pid):
    if message.stop_reason == "max_tokens":
        raise ValueError("Truncated response; usage recorded; no automatic paid retry")
    text = "".join(b.text for b in message.content if b.type == "text")
    return llm._validate_response(llm._extract_json(text), pid)


class PilotReviewer(sem.ant.AnthropicReviewer):
    def _call_api(self, messages, *, model=None):
        params = request_params(model or self.model, messages)
        call_id = f"{self.run_id}:{self.pid}:{self.call_index}"
        self.call_index += 1
        # On resume a partly completed direct pair cannot be repeated for free.
        if call_id in self.ledger.data["calls"]:
            raise RuntimeError("Partial direct pair: refusing duplicate paid call")
        reserved = maximum(self._get_client(), params, False)
        self.ledger.reserve(call_id, reserved)
        start = time.monotonic()
        try:
            response = retry_control(lambda: self._get_client().messages.create(**params))
        except anthropic.APIStatusError:
            # Explicit status has no generated usage. Timeouts leave reservation intact.
            self.ledger.release(call_id)
            raise
        usage = response.usage.model_dump(mode="json")
        self.ledger.record(call_id, self.model, usage, False, response.stop_reason)
        self.ledger.release(call_id)
        if cost(self.model, usage) > reserved + 1e-9:
            raise RuntimeError("Reservation bound exceeded; stop for reconciliation")
        raw = parse(response, self.pid)
        return raw, time.monotonic() - start, usage["input_tokens"], usage["output_tokens"]


def checked_views(mode, ids):
    sem.OUT = OUT / "auditoria_dossie"
    views = sem.prepare(mode, ids)
    expected = read(OUT / "dossier_hashes.json")[mode]
    for pid, view in views.items():
        actual = hashlib.sha256(view.to_json(sort_keys=True).encode()).hexdigest()
        if expected[pid] != actual:
            raise ValueError("Historical dossier mismatch")
    return views


def snapshot(path, model, mode, passed, ids, results, ledger, started, complete, errors=None):
    run_id = path.parent.name + ":" + mode
    calls = {k: v for k, v in ledger.data["calls"].items() if k.startswith(run_id + ":")}
    save(path, {"experiment": "exp2_llm_sync", "model": model, "mode": mode, "passada": passed,
                "selected_pair_ids": ids, "n_review_pairs": len(ids), "n_reviewed": len(results),
                "results": [results[p] for p in ids if p in results], "errors": errors or [],
                "complete": complete, "timestamp": sem.utcnow(), "elapsed_s": round(time.time() - started, 3),
                **sem.configure_prompt("v4"),
                "metadata": {"model_exact": model, "max_tokens": 2500, "temperature_sent": False,
                             "extended_thinking": False, "protocol": "unchanged Agent-A + Agent-B + arbiter on disagreement",
                             "transport": "direct" if passed == "piloto" else "batch", "seed": SEED,
                             "dossier_fingerprints_checked": True, "free_response_fields_saved": False},
                "transport": {"calls": len(calls), "input_tokens": sum(v["usage"].get("input_tokens", 0) for v in calls.values()),
                              "output_tokens": sum(v["usage"].get("output_tokens", 0) for v in calls.values()),
                              "truncated": sum(v["stop_reason"] == "max_tokens" for v in calls.values())},
                "cost_estimate_usd": sum(v["cost_usd"] for v in calls.values()),
                "cost_note": "Observed usage at official tariff, not billing invoice; all retries included."})


def pilot(client, ledger):
    ids = read(OUT / "amostragem.json")["pilot_ids"]
    views = checked_views("vigilancia", ids)
    for model in MODELS[:2]:
        path = OUT / f"{model}_piloto/exp2_llm_sync_vigilancia.json"
        previous = read(path) or {}
        results = {r["pair_id"]: r for r in previous.get("results", [])}
        started = time.time() - previous.get("elapsed_s", 0)
        reviewer = PilotReviewer(api_key=os.environ["ANTHROPIC_API_KEY"], model=model, fallback_model=model,
                                 base_url="https://api.anthropic.com", max_tokens=2500, max_retries=5, timeout_s=120)
        reviewer._client, reviewer.ledger = client, ledger
        reviewer.run_id = path.parent.name + ":vigilancia"
        for pid in ids:
            if pid in results:
                continue
            reviewer.pid, reviewer.call_index = pid, 0
            results[pid] = sem.safe_result(reviewer.review(views[pid]))
            snapshot(path, model, "vigilancia", "piloto", ids, results, ledger, started, len(results) == len(ids))
            print(json.dumps({"stage": "pilot", "model": model, "completed": len(results),
                              "spent_usd": ledger.data["spent_usd"]}), flush=True)


def project(ledger):
    path = OUT / "plano.json"
    if path.exists():
        return read(path)
    frame = pd.read_csv(OUT / "estratos.csv")
    means = {}
    for model in MODELS[:2]:
        data = read(OUT / f"{model}_piloto/exp2_llm_sync_vigilancia.json")
        if not data or not data["complete"] or data["n_reviewed"] != 10:
            raise ValueError("Ten successful pilot pairs required for both models")
        means[model] = {k: data["transport"][k] / 10 for k in ("input_tokens", "output_tokens", "calls")}
    means[MODELS[2]] = {"input_tokens": 7000, "output_tokens": 4000, "calls": 2.2}
    fable_replicate, fable_half = True, False

    def build():
        jobs = []
        for model in MODELS:
            for passed in (["principal", "replica"] if model != MODELS[2] else ["replica_historico"]):
                if model == MODELS[1] and passed == "replica" and not fable_replicate:
                    continue
                for mode in ("vigilancia", "confirmacao"):
                    selected = frame.loc[frame.modo.eq(mode)]
                    if passed == "replica":
                        selected = selected.loc[selected.estrato.ne("S4")]
                    if model == MODELS[1] and fable_half and mode == "vigilancia" and passed == "principal":
                        s4 = sorted(selected.loc[selected.estrato.eq("S4"), "pair_id"])
                        keep = set(np.random.default_rng(SEED).choice(s4, size=len(s4)//2, replace=False))
                        selected = selected.loc[selected.estrato.ne("S4") | selected.pair_id.isin(keep)]
                    jobs.append({"model": model, "passada": passed, "mode": mode,
                                 "ids": selected.pair_id.tolist(), "s4_half": model == MODELS[1] and fable_half})
        estimate = ledger.data["spent_usd"] + sum(len(j["ids"]) * cost(j["model"], means[j["model"]], True) * 1.2 for j in jobs)
        return jobs, estimate

    jobs, estimate = build()
    decisions = []
    if estimate > 85:
        fable_replicate = False
        decisions.append("Retirada a réplica Fable por projeção acima de USD85.")
        jobs, estimate = build()
    if estimate > 85:
        fable_half = True
        decisions.append("S4 Fable reduzido à metade, novo sorteio com seed 20261107 e peso dobrado.")
        jobs, estimate = build()
    plan = {"created_at": sem.utcnow(), "pilot_means": means, "projected_total_usd": estimate,
            "margin": 1.2, "decisions": decisions, "jobs": jobs, "approved_by_rule": estimate <= 85}
    with (OUT / "PREREG.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## Adendo de projeção anterior às chamadas principais\n\n" + json.dumps(
            {k: v for k, v in plan.items() if k != "jobs"}, ensure_ascii=False, indent=2) + "\n")
    save(path, plan)
    print("PROJECTION " + json.dumps({k: v for k, v in plan.items() if k != "jobs"}), flush=True)
    return plan


def reviewer_for(model):
    return sem.ant.AnthropicReviewer(api_key=os.environ["ANTHROPIC_API_KEY"], model=model,
                                     fallback_model=model, max_tokens=2500)


def stage(client, ledger, folder, stage_name, requests, model, *, submit_only=False):
    if not requests:
        return {}, []
    path = folder / f"{stage_name}.json"
    state = read(path)
    digest = hashlib.sha256(json.dumps(requests, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    reservation = folder.parent.name + ":" + folder.name + ":" + stage_name
    if state:
        if state["requests_sha256"] != digest:
            raise ValueError("Batch identity mismatch")
        if not state.get("batch_id"):
            raise RuntimeError("Uncertain batch creation; reconcile manually, never resubmit")
    else:
        upper = sum(maximum(client, r["params"], True) for r in requests)
        ledger.reserve(reservation, upper)
        state = {"state": "submitting", "requests_sha256": digest, "maximum_usd": upper,
                 "submitted_at": sem.utcnow(), "request_ids": [r["custom_id"] for r in requests]}
        save(path, state)
        # No automatic retry on connection/server errors: creation may have succeeded.
        batch = retry_control(lambda: client.messages.batches.create(requests=requests))
        state.update(batch_id=batch.id, state=batch.processing_status)
        save(path, state)
        print("BATCH_SUBMITTED " + json.dumps({"model": model, "stage": stage_name, "batch_id": batch.id,
                                               "requests": len(requests), "maximum_usd": upper,
                                               "spent_usd": ledger.data["spent_usd"]}), flush=True)
    if submit_only:
        return {}, []
    while True:
        batch = retry_control(lambda: client.messages.batches.retrieve(state["batch_id"]))
        state["state"] = batch.processing_status
        state["request_counts"] = batch.request_counts.model_dump()
        save(path, state)
        if batch.processing_status == "ended":
            break
        time.sleep(60)
    parsed, errors, safe = {}, [], {}
    expected = set(state["request_ids"])
    seen = set()
    for item in client.messages.batches.results(state["batch_id"]):
        cid = item.custom_id
        if cid not in expected or cid in seen:
            raise ValueError("Unexpected or duplicate batch custom_id")
        seen.add(cid)
        if item.result.type != "succeeded":
            errors.append({"custom_id": cid, "type": item.result.type})
            continue
        response = item.result.message
        key = folder.parent.name + ":" + folder.name + ":" + state["batch_id"] + ":" + cid
        ledger.record(key, model, response.usage.model_dump(mode="json"), True, response.stop_reason)
        try:
            parsed[cid] = parse(response, cid.split("_")[0])
            safe[cid] = {k: v for k, v in parsed[cid].items() if k != "evidence_summary"}
        except (ValueError, KeyError, TypeError):
            errors.append({"custom_id": cid, "type": "parse_or_truncation"})
        save(folder / f"{stage_name}_results.json", {"results": safe, "errors": errors})
    if seen != expected:
        raise ValueError("Incomplete batch download; reservation retained")
    save(folder / f"{stage_name}_results.json", {"results": safe, "errors": errors})
    ledger.release(reservation)
    state["downloaded_at"] = sem.utcnow()
    save(path, state)
    return parsed, errors


def final_rows(ids, a, b, arb):
    rows = {}
    for pid in ids:
        aa, bb = a.get(pid + "_A"), b.get(pid + "_B")
        if not aa or not bb:
            continue
        consensus = aa["decision"] == bb["decision"]
        winner = (aa if aa["confidence"] >= bb["confidence"] else bb) if consensus else arb.get(pid + "_R")
        if winner is None:
            continue
        row = {k: v for k, v in winner.items() if k != "evidence_summary"}
        row.update(protocol="consensus" if consensus else "arbiter", agent_a_decision=aa["decision"],
                   agent_b_decision=bb["decision"], total_latency_s=0, total_tokens=0)
        if not consensus:
            row["arbiter_decision"] = winner["decision"]
        rows[pid] = row
    return rows


def main_runs(client, ledger):
    plan = project(ledger)
    if not plan["approved_by_rule"]:
        raise RuntimeError("Projection still above USD85; no principal calls")
    # Cache selected views once per mode. Historical hash validation occurs before submission.
    frame = pd.read_csv(OUT / "estratos.csv")
    views = {m: checked_views(m, frame.loc[frame.modo.eq(m), "pair_id"].tolist())
             for m in ("vigilancia", "confirmacao")}
    # Submit ALL stage-1 batches first, so the frontier principal passes do not wait
    # for another model's asynchronous completion (credits expire imminently).
    prepared = []
    incomplete_jobs = []
    for job in plan["jobs"]:
        model, mode, passed, ids = job["model"], job["mode"], job["passada"], job["ids"]
        folder = OUT / f"{model}_{passed}" / mode
        path = folder.parent / f"exp2_llm_sync_{mode}.json"
        previous = read(path) or {}
        if previous.get("complete"):
            continue
        started = time.time() - previous.get("elapsed_s", 0)
        reviewer = reviewer_for(model)
        requests = [{"custom_id": pid + "_" + role[-1],
                     "params": request_params(model, reviewer._build_messages(views[mode][pid], agent_role=role))}
                    for pid in ids for role in ("Agent-A", "Agent-B")]
        stage(client, ledger, folder, "AB", requests, model, submit_only=True)
        prepared.append((job, folder, path, started, reviewer, requests))
    for job, folder, path, started, reviewer, requests in prepared:
        model, mode, passed, ids = job["model"], job["mode"], job["passada"], job["ids"]
        ab, errors = stage(client, ledger, folder, "AB", requests, model)
        rr = []
        for pid in ids:
            a, b = ab.get(pid + "_A"), ab.get(pid + "_B")
            if a and b and a["decision"] != b["decision"]:
                rr.append({"custom_id": pid + "_R", "params": request_params(model,
                    reviewer._build_arbiter_messages(views[mode][pid], a, b))})
        arb, err = stage(client, ledger, folder, "R", rr, model)
        results = final_rows(ids, ab, ab, arb)
        for pid, row in results.items():
            row["total_tokens"] = sum(v["usage"].get("input_tokens", 0) + v["usage"].get("output_tokens", 0)
                                      for k, v in ledger.data["calls"].items()
                                      if k.startswith(f"{model}_{passed}:{mode}:") and pid in k)
        snapshot(path, model, mode, passed, ids, results, ledger, started, len(results) == len(ids), errors + err)
        print("RUN_END " + json.dumps({"model": model, "mode": mode, "passada": passed,
                                       "n": len(results), "errors": len(errors + err),
                                       "spent_usd": ledger.data["spent_usd"]}), flush=True)
        if errors or err:
            # Do not abandon other already-paid batches when one run truncates.
            # Keep failed pairs missing; no regeneration or parameter changes.
            incomplete_jobs.append({"model": model, "mode": mode, "passada": passed,
                                    "n_reviewed": len(results), "n_expected": len(ids),
                                    "response_errors": len(errors + err)})
    if incomplete_jobs:
        print("INCOMPLETE_RUNS " + json.dumps(incomplete_jobs), flush=True)
        raise RuntimeError("All submitted batches collected; incomplete runs retained without paid retries")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--main", action="store_true")
    group.add_argument("--availability", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    if not (OUT / "PREREG.md").exists():
        raise RuntimeError("Preregistration required")
    prompt = sem.configure_prompt("v4")
    sampling = read(OUT / "amostragem.json")
    if prompt["prompt_sha256"] != sampling["prompt_sha256"]:
        raise ValueError("Preregistered prompt changed")
    if hashlib.sha256((OUT / "estratos.csv").read_bytes()).hexdigest() != sampling["estratos_sha256"]:
        raise ValueError("Preregistered sample changed")
    if not (OUT / "custo.json").exists():
        raise ValueError("Missing cost ledger; never initialize during inference")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Missing ANTHROPIC_API_KEY; value suppressed")
    with exclusive(), anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=0, timeout=120) as client:
        ledger = Ledger()
        if args.availability:
            available = {m.id for m in client.models.list(limit=100)}
            status = {m: m in available for m in MODELS}
            save(OUT / "disponibilidade.json", {"timestamp": sem.utcnow(), "models": status})
            print("AVAILABILITY " + json.dumps(status), flush=True)
            if not all(status.values()):
                raise RuntimeError("Requested model not listed; no paid call")
        elif args.pilot:
            if not all(read(OUT / "disponibilidade.json")["models"].values()):
                raise RuntimeError("Availability gate failed")
            pilot(client, ledger)
        else:
            main_runs(client, ledger)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Never print API response/body, prompts or exception messages.
        import traceback
        print("FATAL " + json.dumps({"type": type(exc).__name__, "status": getattr(exc, "status_code", None),
                                     "frames": [{"function": f.name, "line": f.lineno}
                                                for f in traceback.extract_tb(exc.__traceback__)]}), flush=True)
        sys.exit(1)
