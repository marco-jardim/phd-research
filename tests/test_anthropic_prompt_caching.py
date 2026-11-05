"""Offline wire-format, warmup, persistence and historical-identity regressions."""
import hashlib
import importlib.util
import json
import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from gzcmd import anthropic_transport as transport
from gzcmd import llm_review
from gzcmd.llm_usage_cost import price_usage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("cached_runner", ROOT / "scripts/run_exp2_cached.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def offline_v4(monkeypatch):
    monkeypatch.setattr(llm_review, "_PROMPT_PATH", ROOT / "gzcmd/gzcmd_v4_llm_prompt.md")
    monkeypatch.setattr(runner.abl, "save", runner.abl.save)
    with patch("anthropic.Anthropic") as constructor, \
         patch("requests.post", side_effect=AssertionError("Network forbidden")):
        yield constructor


def dossier(pid="a" * 16):
    payload = {"schema_version": "synthetic", "pair_id": pid,
               "features": {"nota_final": 7, "nome": "sintético"}}
    return runner.abl.DossierView(SimpleNamespace(pair_id=pid, to_dict=lambda: payload), True)


def response(decision="MATCH"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps({"decision": decision}))],
        stop_reason="end_turn", usage=SimpleNamespace(input_tokens=20, output_tokens=30,
            cache_read_input_tokens=1200, cache_creation_input_tokens=100))


@pytest.mark.parametrize("role", ["Agent-A", "Agent-B", "Arbiter"])
def test_blocks_preserve_historical_utf8_bytes(role, offline_v4):
    view = dossier()
    before = hashlib.sha256(view.to_json(sort_keys=True).encode()).hexdigest()
    old = llm_review.LLMReviewer(api_key="offline")
    new = transport.AnthropicReviewer(api_key="offline", model="claude-haiku-5-5")
    if role == "Arbiter":
        messages = old._build_arbiter_messages(view, {"decision": "MATCH"}, {"decision": "NONMATCH"})
    else:
        messages = old._build_messages(view, agent_role=role)
    offline_v4.return_value.messages.create.return_value = response()
    new._call_api(messages)
    request = offline_v4.return_value.messages.create.call_args.kwargs
    system, user = request["system"], request["messages"][0]["content"]
    assert len(system) == len(user) == 2
    assert system[0]["text"] == llm_review._load_system_prompt()
    assert user[0]["text"] == llm_review._load_user_template()
    for blocks, message in [(system, messages[0]), (user, messages[1])]:
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in blocks[1]
        assert blocks[1]["text"].startswith("\n\n")
        assert "".join(b["text"] for b in blocks).encode("utf-8") == message["content"].encode("utf-8")
    if role == "Arbiter":
        assert user[1]["text"].index("### Dossiê original") < user[1]["text"].index("### Opinião Agent-A")
        assert user[1]["text"].index("### Opinião Agent-A") < user[1]["text"].index("### Opinião Agent-B")
    assert hashlib.sha256(view.to_json(sort_keys=True).encode()).hexdigest() == before
    assert "temperature" not in request
    assert "thinking" not in request


def test_warmup_finishes_before_executor_is_created():
    events = []

    def review(pid):
        events.extend([("start", pid), ("end", pid)])
        return {"pair_id": pid}, None

    class FakeExecutor:
        def __init__(self, **kwargs):
            events.append("pool")

        def submit(self, fn, *args):
            events.append("submit")
            future = Future()
            future.set_result(fn(*args))
            return future

        def shutdown(self, **kwargs):
            events.append("shutdown")

    with runner.WarmupExecutor(4, executor_factory=FakeExecutor) as pool:
        pool.submit(review, "first").result()
        pool.submit(review, "second").result()
    assert events == [("start", "first"), ("end", "first"), "pool", "submit",
                      ("start", "second"), ("end", "second"), "shutdown"]
    events.clear()
    with runner.WarmupExecutor(4, warmup=False, executor_factory=FakeExecutor) as pool:
        pool.submit(review, "first").result()
    assert events[:2] == ["pool", "submit"]
    factory = Mock()
    with pytest.raises(RuntimeError, match="warmup failed"):
        with runner.WarmupExecutor(4, executor_factory=factory) as pool:
            pool.submit(lambda: (None, {"type": "FatalAPIError"}))
    factory.assert_not_called()


@pytest.mark.parametrize("mode", ["vigilancia", "confirmacao"])
def test_cached_run_records_usage_identity_resume_and_execution(tmp_path, monkeypatch, offline_v4, mode):
    ids = [f"{i:016x}" for i in range(5)]
    views = {pid: dossier(pid) for pid in ids}
    source = tmp_path / "pairs.json"
    source.write_text(json.dumps(ids), encoding="utf-8")
    args = runner.parse_args(["--modo", mode, "--pares", str(source),
                             "--saida", str(tmp_path / "run.json"), "--allow-new-cohort"])
    assert args.cache_warmup is True and args.prompt_version == "v4"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-not-a-key")
    monkeypatch.setattr(runner, "prepare", lambda *args: views)
    # Every network boundary is mocked; this runs the real two-agent protocol.
    offline_v4.return_value.messages.create.side_effect = lambda **kwargs: response()
    with patch.object(transport, "_extract_json", transport._extract_json):
        output = runner.run(args)
        resumed = runner.run(args)
    assert output["complete"] and resumed["complete"]
    assert offline_v4.return_value.messages.create.call_count == 10
    expected = hashlib.sha256((ROOT / "gzcmd/gzcmd_v4_llm_prompt.md").read_bytes()).hexdigest()
    assert output["prompt_sha256"] == expected
    assert output["dossier_sha256"] == {pid: hashlib.sha256(v.to_json(sort_keys=True).encode()).hexdigest()
                                         for pid, v in views.items()}
    identity = json.loads((tmp_path / "run_journal/identity.json").read_text(encoding="utf-8"))
    assert identity == {"mode": mode, "model": "claude-haiku-5-5",
                        "dossier_sha256": output["dossier_sha256"], "prompt_sha256": expected,
                        "prompt_version": "v4", "selected_pair_ids": ids,
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "cohort": "v31_lot"}
    assert output["prompt_caching"] == transport.caching_metadata(True)
    execution = json.loads((tmp_path / "execucao.json").read_text(encoding="utf-8"))
    for field, expected_tokens in [("cache_read_input_tokens", 12000),
                                   ("cache_creation_input_tokens", 1000)]:
        assert output["transport"][field] == output["metadata"][field] == expected_tokens
        assert execution[field] == execution["transport"][field] == expected_tokens
    assert execution["prompt_caching"] == output["prompt_caching"]
    assert price_usage(output["transport"]) == pytest.approx(.000415)
    assert offline_v4.return_value.close.call_count == 5
    args.cache_warmup = False
    with pytest.raises(ValueError, match="saída nova"):
        runner.run(args)


def test_arbiter_protocol_records_three_calls(tmp_path, monkeypatch, offline_v4):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline")
    runner.abl.save = runner.save
    offline_v4.return_value.messages.create.side_effect = [response("MATCH"), response("NONMATCH"), response("UNSURE")]
    row, error = runner.review_pair("a" * 16, dossier(), "claude-haiku-5-5", tmp_path)
    assert error is None and row["protocol"] == "arbiter" and row["decision"] == "UNSURE"
    usage = runner.recover_json(tmp_path / ("a" * 16 + ".transport.json"))
    assert usage["calls"] == 3
    assert usage["cache_read_input_tokens"] == 3600
    assert usage["cache_creation_input_tokens"] == 300
    request = offline_v4.return_value.messages.create.call_args.kwargs
    assert "### Opinião Agent-B" in request["messages"][0]["content"][1]["text"]


def test_failed_warmup_checkpoints_usage_without_releasing_workers(tmp_path, monkeypatch, offline_v4):
    pid = "a" * 16
    source = tmp_path / "pairs.json"
    source.write_text(json.dumps([pid]), encoding="utf-8")
    args = runner.parse_args(["--modo", "vigilancia", "--pares", str(source),
                             "--saida", str(tmp_path / "run.json"), "--allow-new-cohort"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline")
    monkeypatch.setattr(runner, "prepare", lambda *args: {pid: dossier(pid)})
    truncated = response()
    truncated.stop_reason = "max_tokens"
    offline_v4.return_value.messages.create.return_value = truncated
    with patch.object(transport.time, "sleep"), pytest.raises(runner.WarmupError):
        runner.run(args)
    output = runner.recover_json(args.saida)
    assert output["n_errors"] == 1 and not output["complete"]
    assert output["errors"][0]["type"] == "LLMCallError"
    assert output["metadata"]["cache_read_input_tokens"] == 6000
    assert offline_v4.call_count == 1


def test_default_fireworks_is_blocked_before_post():
    with pytest.raises(RuntimeError, match="Fireworks é proibido neste projeto"):
        llm_review.LLMReviewer(api_key="offline")._call_api([])
    with pytest.raises(ValueError, match="Fireworks é proibido neste projeto"):
        runner.configure_provider("fireworks")


def test_cli_can_disable_warmup():
    args = runner.parse_args(["--modo", "vigilancia", "--pares", "pairs.json",
                             "--saida", "run.json", "--no-cache-warmup"])
    assert args.cache_warmup is False
