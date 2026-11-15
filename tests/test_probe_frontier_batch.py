"""Offline guardrails for the local exploratory probe; never calls the API."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_exp2_batch_anthropic as batch


def test_tariffs_and_batch_discount():
    usage = {"input_tokens": 7000, "output_tokens": 4000}
    assert batch.cost(batch.MODELS[0], usage) == pytest.approx(.108)
    assert batch.cost(batch.MODELS[1], usage, True) == pytest.approx(.135)
    assert batch.cost(batch.MODELS[2], usage) == pytest.approx(.0027)


def test_reservations_and_idempotent_usage(tmp_path):
    ledger = batch.Ledger(tmp_path / "cost.json")
    ledger.reserve("a", 89.)
    with pytest.raises(RuntimeError, match="budget"):
        ledger.reserve("b", 2.)
    ledger.record("result", batch.MODELS[0], {"input_tokens": 1000, "output_tokens": 1000}, True, "end_turn")
    ledger.record("result", batch.MODELS[0], {"input_tokens": 1000, "output_tokens": 1000}, True, "end_turn")
    assert ledger.data["spent_usd"] == pytest.approx(.012)
    ledger.release("a")
    assert not ledger.data["reservations"]
    recovered = batch.Ledger(tmp_path / "cost.json")
    assert recovered.data == ledger.data


def test_truncation_never_becomes_unsure():
    with pytest.raises(ValueError, match="Truncated"):
        batch.parse(SimpleNamespace(stop_reason="max_tokens"), "a" * 16)


def test_exact_message_builders(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "chave-ficticia-de-teste")
    batch.sem.configure_prompt("v4")
    class View:
        def to_json(self, **kwargs):
            return '{"pair_id":"aaaaaaaaaaaaaaaa","features":{}}'
    reviewer = batch.reviewer_for(batch.MODELS[0])
    for role in ("Agent-A", "Agent-B"):
        messages = reviewer._build_messages(View(), agent_role=role)
        assert messages == batch.llm.build_review_messages(View(), agent_role=role, batch_mode=False)
        params = batch.request_params(batch.MODELS[0], messages)
        assert params["system"] == messages[0]["content"]
        assert params["messages"] == [messages[1]]
        assert set(params) == {"model", "system", "messages", "max_tokens"}
    a = {"decision": "MATCH", "confidence": .7}
    b = {"decision": "NONMATCH", "confidence": .8}
    assert reviewer._build_arbiter_messages(View(), a, b) == batch.llm.build_arbiter_messages(View(), a, b)


def test_consensus_winner_and_arbiter():
    def response(decision, confidence):
        return {"pair_id": "a" * 16, "decision": decision, "confidence": confidence,
                "reason_codes": ["NAME_STRONG"], "quality_flags": {}, "evidence_summary": {"text": "not saved"}}
    pid = "a" * 16
    a = {pid + "_A": response("MATCH", .5)}
    b = {pid + "_B": response("MATCH", .9)}
    row = batch.final_rows([pid], a, b, {})[pid]
    assert row["confidence"] == .9
    assert "evidence_summary" not in row
    b[pid + "_B"] = response("NONMATCH", .9)
    assert not batch.final_rows([pid], a, b, {})
    row = batch.final_rows([pid], a, b, {pid + "_R": response("UNSURE", .6)})[pid]
    assert row["protocol"] == "arbiter" and row["decision"] == "UNSURE"


def test_ambiguous_batch_never_resubmitted(tmp_path):
    request = [{"custom_id": "a" * 16 + "_A", "params": {}}]
    import hashlib
    import json
    digest = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    batch.save(tmp_path / "AB.json", {"requests_sha256": digest, "state": "submitting"})
    with pytest.raises(RuntimeError, match="Uncertain"):
        batch.stage(None, None, tmp_path, "AB", request, batch.MODELS[0])


def test_weighted_counts_and_wilson():
    import pandas as pd
    from probe_frontier_analise import counters, wilson, agreement
    frame = pd.DataFrame({"decision": ["MATCH", "MATCH", "NONMATCH", "UNSURE"],
                          "PAR": [2, 0, 2, 2], "peso": [1., 3., 1., 1.]})
    raw = counters(frame)
    weighted = counters(frame, True)
    assert raw["VPP"] == .5 and weighted["VPP"] == .25
    assert raw["PAR2_recuperados"] == raw["PAR2_descartados"] == raw["PAR2_unsure"] == 1
    assert weighted["n"] == 6 and weighted["FP"] == 3
    assert weighted["VPP_IC95_inf"] is None
    low, high = wilson(4, 25)
    assert low == pytest.approx(.0640345, abs=1e-6)
    assert high == pytest.approx(.3465362, abs=1e-6)
    assert agreement(["MATCH"], ["MATCH"])["kappa"] is None
    assert agreement(["MATCH"], [None])["n"] == 0


def test_completed_batch_download_records_usage_before_parse(tmp_path, monkeypatch):
    class Counts:
        def model_dump(self):
            return {"succeeded": 1}
    class Usage:
        def model_dump(self, **kwargs):
            return {"input_tokens": 1000, "output_tokens": 2500}
    message = SimpleNamespace(stop_reason="max_tokens", usage=Usage())
    cid = "a" * 16 + "_A"
    individual = SimpleNamespace(custom_id=cid, result=SimpleNamespace(type="succeeded", message=message))
    class Batches:
        def create(self, **kwargs):
            return SimpleNamespace(id="msgbatch_test", processing_status="in_progress")
        def retrieve(self, batch_id):
            assert batch_id == "msgbatch_test"
            return SimpleNamespace(processing_status="ended", request_counts=Counts())
        def results(self, batch_id):
            return iter([individual])
    client = SimpleNamespace(messages=SimpleNamespace(batches=Batches()))
    monkeypatch.setattr(batch, "maximum", lambda *args: .1)
    ledger = batch.Ledger(tmp_path / "cost.json")
    requests = [{"custom_id": cid, "params": {}}]
    result, errors = batch.stage(client, ledger, tmp_path / "run" / "vigilancia", "AB", requests, batch.MODELS[0])
    assert not result and errors[0]["type"] == "parse_or_truncation"
    assert ledger.data["spent_usd"] == pytest.approx(.027)
    assert not ledger.data["reservations"]
    batch.stage(client, ledger, tmp_path / "run" / "vigilancia", "AB", requests, batch.MODELS[0])
    assert ledger.data["spent_usd"] == pytest.approx(.027)


def test_cost_ledger_corruption_fails_closed(tmp_path):
    path = tmp_path / "cost.json"
    path.write_text("invalid", encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecoverable"):
        batch.Ledger(path)
    batch.save(path, {"spent_usd": 12., "reservations": {}, "calls": {}})
    with pytest.raises(ValueError, match="sum mismatch"):
        batch.Ledger(path)


def test_collects_other_submitted_jobs_after_truncation(tmp_path, monkeypatch):
    import pandas as pd
    pid_a, pid_b = "a" * 16, "b" * 16
    monkeypatch.setattr(batch, "OUT", tmp_path)
    pd.DataFrame({"pair_id": [pid_a, pid_b], "modo": ["vigilancia", "confirmacao"]}).to_csv(tmp_path / "estratos.csv", index=False)
    jobs = [{"model": batch.MODELS[2], "mode": mode, "passada": "replica_historico", "ids": [pid]}
            for mode, pid in (("vigilancia", pid_a), ("confirmacao", pid_b))]
    monkeypatch.setattr(batch, "project", lambda ledger: {"approved_by_rule": True, "jobs": jobs})
    monkeypatch.setattr(batch, "checked_views", lambda mode, ids: {p: object() for p in ids})
    reviewer = SimpleNamespace(_build_messages=lambda *args, **kwargs: [])
    monkeypatch.setattr(batch, "reviewer_for", lambda model: reviewer)
    collected = []
    def stage(client, ledger, folder, stage_name, requests, model, *, submit_only=False):
        if submit_only:
            return {}, []
        if stage_name == "AB":
            collected.append(folder.name)
            return {}, [{"type": "parse_or_truncation"}] if folder.name == "vigilancia" else []
        return {}, []
    monkeypatch.setattr(batch, "stage", stage)
    monkeypatch.setattr(batch, "final_rows", lambda ids, *args: {} if ids == [pid_a] else {pid_b: {}})
    snapshots = []
    monkeypatch.setattr(batch, "snapshot", lambda *args: snapshots.append(args[8]))
    ledger = SimpleNamespace(data={"calls": {}, "spent_usd": 0.})
    with pytest.raises(RuntimeError, match="All submitted batches collected"):
        batch.main_runs(None, ledger)
    assert collected == ["vigilancia", "confirmacao"]
    assert snapshots == [False, True]
