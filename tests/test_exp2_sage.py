"""Offline contract tests for the Sage reviewer (no provider calls)."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("sage_runner", ROOT / "scripts/run_exp2_sage.py")
sage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sage)


def response(chosen="nonmatch", probability=.9):
    return {"id": "par_match", "kind": "choice", "result": {
        "chosen": chosen, "probability": probability,
        "probabilities": [{"option": "match", "probability": .1},
                          {"option": "nonmatch", "probability": .9}]},
        "meta": {"model": "levanto-sage-2", "latency_ms": 81,
                 "reasoning": {"triggered": True, "tokens": 3, "text": "NOT_FOR_DISK"}}}


def test_choice_confidence_is_not_match_probability():
    result = sage.parse_response("abc", response(), .2)
    assert result["decision"] == "NONMATCH"
    assert result["probability"] == .9
    assert result["p_match"] == .1
    assert result["meta"]["model"] == "levanto-sage-2"
    assert result["meta"]["reasoning"] == {"triggered": True, "tokens": 3.0}
    assert "NOT_FOR_DISK" not in str(result)


def test_native_abstention_preserves_distribution():
    result = sage.parse_response("abc", response(None, None), .2)
    assert result["decision"] == "UNSURE"
    assert result["probability"] is None
    assert result["p_match"] == .1


def test_documented_reasoning_limit_is_preserved():
    raw = response()
    raw["meta"]["reasoning"]["limited"] = "timeout"
    assert sage.parse_response("abc", raw, .2)["meta"]["reasoning"]["limited"] == "timeout"


def test_fixed_option_order_and_no_silent_unknown_choice():
    raw = response()
    raw["result"]["probabilities"].reverse()
    with pytest.raises(ValueError):
        sage.parse_response("abc", raw, .2)
    raw = response("other")
    with pytest.raises(ValueError):
        sage.parse_response("abc", raw, .2)


def test_ablation_removes_whole_group_without_mutation():
    dossier = {"features": {"nota_final": 7}, "model_outputs": {"p_cal": .9, "band": "high"}}
    assert sage.ablate(dossier) == {"features": {"nota_final": 7}}
    assert "model_outputs" in dossier


def test_402_is_terminal_and_not_retried():
    api = sage.Sage("test-not-secret")
    with patch.object(sage.requests, "request", return_value=Mock(status_code=402, text='{"error":"quota"}', url=sage.BASE + "/decide")) as request:
        with pytest.raises(sage.APIError) as exc:
            api.call("/decide", {})
        assert exc.value.status == 402
        assert api.quota.is_set() and api.stop.is_set()
        with pytest.raises(sage.APIError):
            api.call("/decide", {})
        assert request.call_count == 1


@pytest.mark.parametrize("status", [400, 401, 403])
def test_invalid_requests_not_retried(status):
    api = sage.Sage("test-not-secret")
    with patch.object(sage.requests, "request", return_value=Mock(status_code=status, text='{"error":"invalid"}', url=sage.BASE + "/decide")) as request:
        with pytest.raises(sage.APIError):
            api.call("/decide", {})
        assert request.call_count == 1


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_backoff_respects_retry_after(status):
    api = sage.Sage("test-not-secret")
    success = Mock(status_code=200)
    success.json.return_value = {"ok": True}
    with patch.object(sage.requests, "request", side_effect=[Mock(status_code=status, headers={"Retry-After": "7"}, text='{}', url=sage.BASE + "/decide"), success]), \
            patch.object(api.stop, "wait", return_value=False) as delay:
        assert api.call("/decide", {}) == {"ok": True}
        delay.assert_called_once_with(7)


def test_404_preserves_literal_after_three_backoff_attempts():
    api = sage.Sage("lv_live_test-not-secret")
    error = Mock(status_code=404, headers={}, url="https://sage.levanto.ai/decide",
                 text='{"detail":"Model not found"}')
    with patch.object(sage.requests, "request", return_value=error) as request, \
            patch.object(api.stop, "wait", return_value=False) as delay:
        with pytest.raises(sage.APIError) as exc:
            api.call("/decide", {})
        assert request.call_count == 3
        assert [c.args[0] for c in delay.call_args_list] == [2, 4]
        assert exc.value.body_literal == '{"detail":"Model not found"}'
        assert api.stop.is_set()


def test_error_literal_redacts_secret():
    api = sage.Sage("lv_live_test-not-secret")
    error = Mock(status_code=401, url="https://sage.levanto.ai/decide", text="lv_live_test-not-secret")
    with patch.object(sage.requests, "request", return_value=error):
        with pytest.raises(sage.APIError) as exc:
            api.call("/decide", {})
        assert exc.value.body_literal == "[REDACTED]"


def test_unsure_denominators_are_explicit():
    frame = pd.DataFrame({"decision": ["MATCH", "NONMATCH", "UNSURE", "UNSURE"], "y": [1, 0, 1, 0]})
    metrics = sage.rates(frame)
    assert metrics["sens_sem_unsure"] == metrics["espec_sem_unsure"] == 1
    assert metrics["sens_unsure_como_erro"] == metrics["espec_unsure_como_erro"] == .5
    assert metrics["UNSURE"] == 2


def test_checkpoint_refuses_request_changes(tmp_path):
    with patch.object(sage, "run_path", return_value=tmp_path / "run.json"):
        run = sage.init_run("vigilancia", "sem_mo", ["abc"], {"abc": {"content": "test"}}, "criteria")
        assert run["status"] == "prepared"
        assert sage.init_run("vigilancia", "sem_mo", ["abc"], {"abc": {"content": "test"}}, "criteria") == run
        with pytest.raises(ValueError):
            sage.init_run("vigilancia", "sem_mo", ["abc"], {"abc": {"content": "changed"}}, "criteria")


def test_mode_adaptation_uses_current_prompt():
    text = sage.instructions("confirmacao", "sem_mo")
    assert sage._load_system_prompt() in text
    assert "CONFIRMAÇÃO" in text and "evidência forte" in text
    assert "removido intencionalmente" in text


def test_incremental_checkpoint_stops_at_quota_and_resumes_successes(tmp_path):
    api = sage.Sage("test-not-secret")
    payloads = {pid: {"content": "features only"} for pid in ("a", "b", "c")}
    with patch.object(sage, "run_path", return_value=tmp_path / "run.json"), \
            patch.object(api, "call", side_effect=[response(), sage.APIError(402)]) as call:
        # Mock call preserves the real transport's stop semantics.
        def transport(*args):
            if call.call_count == 1:
                return response()
            api.quota.set()
            api.stop.set()
            raise sage.APIError(402)
        call.side_effect = transport
        run = sage.execute_run(api, "vigilancia", "sem_mo", list(payloads), payloads, "criteria", 0, 1)
        assert run["status"] == "quota_exhausted"
        assert run["n_reviewed"] == 1
        assert run["results"][0]["pair_id"] == "a"
        assert run["errors"][0]["status"] == 402
        assert call.call_count == 2
        # A quota-exhausted checkpoint cannot silently make fresh calls.
        sage.execute_run(api, "vigilancia", "sem_mo", list(payloads), payloads, "criteria", 0, 1)
        assert call.call_count == 2


def test_usage_totals_do_not_mix_units_or_booleans():
    assert sage.numeric_totals({"usage": {"input_tokens": 40, "output_tokens": 3}, "cost_usd": .001,
                                "cached": True}) == {"usage_input_tokens": 40, "usage_output_tokens": 3,
                                                     "cost_usd": .001}


@pytest.mark.parametrize("value,expected", [("001", "1"), ("1.0", "1"), (1.0, "1"),
                                           (" 09 ", "9"), ("x01", "x01")])
def test_identifier_canonicalization_is_only_for_join(value, expected):
    assert sage.canonical_identifier(value) == expected


def test_dossier_alignment_preserves_original_string_hash(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/COMPARADORSEMIDENT.csv").write_text("COMPREC;REFREC;PASSO;PAR\n001;002;01;1\n", encoding="utf-8")
    frame = pd.DataFrame({"COMPREC": [1.0], "REFREC": [2], "PASSO": [1], "nota_final": [7]})
    pid = sage.make_pair_id("001", "002", "01")
    with patch.object(sage, "ROOT", tmp_path):
        dossiers = sage.aligned_dossiers(frame, {pid})
    assert set(dossiers) == {pid}
    assert dossiers[pid]["pair_id"] == pid


def test_empty_prepared_cohort_still_has_report_and_csv(tmp_path):
    path = tmp_path / "data/sprint3b/sage_sem_mo/exp2_llm_sync_vigilancia.json"
    run = {"mode": "vigilancia", "arm": "sem_mo", "model": None, "status": "prepared",
           "n_review_pairs": 1410, "results": []}
    sage.save(path, run)
    g = pd.DataFrame({"y": [1], "nota_final": [7]}, index=pd.Index(["abc"], name="pair_id"))
    with patch.object(sage, "ROOT", tmp_path), patch.object(sage, "gold", return_value=g):
        sage.summarize()
    result = pd.read_csv(tmp_path / "data/exp2_sage_resumo.csv")
    assert result.iloc[0]["n"] == 0
    assert result.iloc[0]["n_expected"] == 1410
    assert pd.isna(result.iloc[0]["auc"])
    assert (tmp_path / "data/exp2_sage_relatorio.md").exists()


def test_summary_uses_match_distribution_not_winning_confidence(tmp_path):
    path = tmp_path / "data/sprint3b/sage_sem_mo/exp2_llm_sync_vigilancia.json"
    a = sage.parse_response("a", response(), .2)
    raw = response("match", .8)
    raw["result"]["probabilities"] = [{"option": "match", "probability": .8},
                                        {"option": "nonmatch", "probability": .2}]
    b = sage.parse_response("b", raw, .2)
    sage.save(path, {"mode": "vigilancia", "arm": "sem_mo", "model": "levanto-sage-2",
                     "status": "complete", "n_review_pairs": 2, "results": [a, b]})
    g = pd.DataFrame({"y": [0, 1], "nota_final": [7, 8]}, index=pd.Index(["a", "b"], name="pair_id"))
    with patch.object(sage, "ROOT", tmp_path), patch.object(sage, "gold", return_value=g):
        sage.summarize()
    result = pd.read_csv(tmp_path / "data/exp2_sage_resumo.csv").iloc[0]
    assert result["auc"] == 1
    assert result["brier"] == pytest.approx(.025)
    assert result["auc_raw_chosen_vs_y"] == 0
    thresholds = pd.read_csv(tmp_path / "data/exp2_sage_limiares.csv")
    assert thresholds.iloc[0]["UNSURE"] == 0
    assert thresholds.iloc[1]["UNSURE"] == 1


def test_audit_allows_only_explicit_synthetic_diagnostic_content(tmp_path):
    probe = tmp_path / "data/sprint3b/sage_sem_mo/diagnostico_http.json"
    sage.save(probe, {"request": {"content": "The light is green."}})
    with patch.object(sage, "ROOT", tmp_path):
        sage.audit("test-not-secret")
        sage.save(probe, {"request": {"content": "not an approved sentence"}})
        with pytest.raises(ValueError, match="Unexpected diagnostic content"):
            sage.audit("test-not-secret")


def test_explicit_quota_resume_skips_existing_response(tmp_path):
    import copy
    api = sage.Sage("test-not-secret")
    payloads = {"a": {"content": "features A"}, "b": {"content": "features B"}}
    with patch.object(sage, "run_path", return_value=tmp_path / "run.json"):
        initial = sage.init_run("vigilancia", "sem_mo", list(payloads), payloads, "criteria")
        first = sage.parse_response("a", response(), .2)
        preserved = copy.deepcopy(first)
        initial.update({"results": [first], "status": "quota_exhausted", "n_reviewed": 1, "model": "levanto-sage-2"})
        sage.save(tmp_path / "run.json", initial)
        def transport(endpoint, payload):
            api.calls[endpoint] += 1
            return response()
        with patch.object(api, "call", side_effect=transport) as call:
            run = sage.execute_run(api, "vigilancia", "sem_mo", list(payloads), payloads, "criteria", 0, 1,
                                   resume_after_quota=True)
            call.assert_called_once_with("/decide", payloads["b"])
        assert run["status"] == "complete"
        assert run["results"][0] == preserved
        assert run["quota_resumptions"][0]["n_preserved"] == 1
        assert run["transport"]["decide_calls_tracked"] == 1


def test_anchoring_counts_fn_reversions_and_exact_mcnemar():
    pairs = [{"gabarito": 1, "com_mo": "NONMATCH", "sem_mo": d} for d in ("MATCH", "MATCH", "UNSURE")]
    result = sage.anchoring_metrics(pairs)
    assert result["fn_com_to_sem_match"] == 2
    assert result["fn_com_to_sem_unsure"] == 1
    assert result["mcnemar_match_exact_p"] == .5
    assert result["mcnemar_fn_exact_p"] == .25
    assert sage.exact_mcnemar(0, 0) == 1


def test_confidential_grounding_null_is_a_wire_only_change():
    api = sage.Sage("test-not-secret")
    api.grounding_null = True
    body = {"content": "synthetic", "grounding": {"trigger": "never"}, "reasoning": "auto"}
    success = Mock(status_code=200)
    success.json.return_value = response()
    with patch.object(sage.requests, "request", return_value=success) as request:
        api.call("/decide", body)
        assert request.call_args.kwargs["json"]["grounding"] is None
        assert request.call_args.kwargs["json"]["reasoning"] == "auto"
    assert body["grounding"] == {"trigger": "never"}


def test_grounding_activation_is_rejected():
    raw = response()
    raw["grounding_meta"] = {"triggered": True}
    with pytest.raises(ValueError, match="Unexpected grounding activation"):
        sage.parse_response("abc", raw, .1)


def test_checkpoint_replace_retries_windows_sharing_violation(tmp_path):
    tmp, target = tmp_path / "run.tmp", tmp_path / "run.json"
    with patch.object(Path, "replace", side_effect=[PermissionError("sharing violation"), target]) as replace, \
            patch.object(sage.time, "sleep") as sleep:
        sage.replace_checkpoint(tmp, target)
    assert replace.call_count == 2
    sleep.assert_called_once_with(.05)


def test_pending_checkpoint_recovers_paid_response_without_repeating(tmp_path):
    import copy
    path = tmp_path / "run.json"
    with patch.object(sage, "run_path", return_value=path):
        old = sage.init_run("vigilancia", "sem_mo", ["a", "b"], {"a": {}, "b": {}}, "criteria")
        old["results"] = [sage.parse_response("a", response(), .1)]
        sage.save(path, old)
        pending = copy.deepcopy(old)
        pending["results"].append(sage.parse_response("b", response(), .1))
        path.with_suffix(".tmp").write_text(sage.json.dumps(pending), encoding="utf-8")
        recovered = sage.init_run("vigilancia", "sem_mo", ["a", "b"], {"a": {}, "b": {}}, "criteria")
    assert recovered["results"] == pending["results"]
    assert not path.with_suffix(".tmp").exists()
