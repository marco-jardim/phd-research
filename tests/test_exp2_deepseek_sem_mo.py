"""Provider isolation and quota gating, with no network or private data."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_exp2_sem_model_outputs as sem
import run_exp2_deepseek_sem_mo as launcher


def test_provider_defaults_and_output_isolation():
    sem.configure_provider("opencode-go")
    assert sem.OUT.name == "opencode_go_sem_mo"
    assert sem.original_path("confirmacao").parent.name == "opencode_go"
    assert sem.artifact_prefix() == "exp2_deepseek_sem_mo"
    sem.configure_provider("anthropic")
    assert sem.OUT.name == "anthropic_sem_mo"
    assert sem.original_path("vigilancia").parent.name == "anthropic"
    assert sem.artifact_prefix() == "exp2_haiku_sem_mo"


def test_quota_probe_one_retry_and_no_raw_persistence(monkeypatch, tmp_path):
    import requests
    statuses = iter((429, 200))
    calls, waits = [], []
    def post(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=next(statuses))
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "unit-test-secret-only")
    monkeypatch.setattr(requests, "post", post)
    assert launcher.quota_probe(tmp_path, sleep=waits.append)
    assert waits == [900]
    assert len(calls) == 2
    assert calls[0]["json"]["thinking"] == {"type": "disabled"}
    assert calls[0]["json"]["reasoning_effort"] == "none"
    assert "unit-test-secret-only" not in (tmp_path / "quota_probe.json").read_text()


def test_quota_probe_stops_after_persistent_quota(monkeypatch, tmp_path):
    import requests
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "unit-test-secret-only")
    monkeypatch.setattr(requests, "post", lambda *a, **kw: SimpleNamespace(status_code=400))
    waits = []
    assert not launcher.quota_probe(tmp_path, sleep=waits.append)
    assert waits == [900]


def test_go_transport_keeps_exception_api_and_sanitized_status(monkeypatch, tmp_path):
    import requests
    import run_exp2_opencode_go as go
    original = go.llm_review.requests
    sem.configure_go_transport()
    try:
        assert go.llm_review.requests.RequestException is requests.RequestException
        sem.abl._LOCAL.stats = sem.abl.meter()
        sem.abl._LOCAL.stats_path = tmp_path / "transport.json"
        monkeypatch.setattr(requests, "post", lambda *a, **kw: SimpleNamespace(status_code=400))
        try:
            go.llm_review.requests.post(go.BASE_URL, json={})
        except sem.abl.FatalAPIError as exc:
            assert exc.status == 400
        else:
            raise AssertionError("Fatal status must interrupt the reviewer")
        assert sem.abl._LOCAL.stats["calls"] == 1
        assert sem.abl._LOCAL.stats["error_statuses"] == {"400": 1}
    finally:
        go.llm_review.requests = original
        sem.abl._LOCAL.stats_path = None
