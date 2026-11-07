"""Offline checks: prompt selection, persisted identity and ablated messages."""
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_exp2_sem_model_outputs as runner
from gzcmd.llm_dossier import Dossier, DossierFeatures, DossierModelOutputs


@pytest.fixture
def isolated_prompt(monkeypatch):
    # Both provider implementations use this same module; restore after each test.
    monkeypatch.setattr(runner.ant.llm_review, "_PROMPT_PATH", runner.ant.llm_review._PROMPT_PATH)
    return runner.ant.llm_review


def output_free_view():
    dossier = Dossier(
        schema_version="test", pair_id="a" * 16,
        features=DossierFeatures(nota_final=7.5, name={}, mother={}, dob={}, address={},
                                 municipality=1.0, composites={}, flags={}),
        model_outputs=DossierModelOutputs(p_cal=.8, band="near_high", base_choice="MATCH",
                                          evr=.1, guardrail=None, guardrail_reason=None),
    )
    return runner.abl.DossierView(dossier, True)


def test_v4_has_no_forbidden_references_and_policy_is_actually_loaded(isolated_prompt):
    raw = (runner.PROMPT_DIR / "gzcmd_v4_llm_prompt.md").read_text(encoding="utf-8")
    for forbidden in runner.FORBIDDEN | {"modelo calibrado", "classificação automática do pipeline"}:
        assert forbidden not in raw.lower()
    runner.configure_prompt("v4")
    system = isolated_prompt._load_system_prompt()
    for required in ("Seja decisivo", "Política única", "**MATCH**", "**NONMATCH**", "**UNSURE**",
                     "inversão dia/mês", "**Sexo**", "Município/UF", "Datas de evento",
                     "apenas contexto do comparador e não decide"):
        assert required in system
    template = isolated_prompt._load_user_template()
    for field in ("pair_id", "decision", "confidence", "reason_codes", "evidence_summary",
                  "supports_match", "supports_nonmatch", "tie_breakers", "quality_flags",
                  "pii_leak_detected", "insufficient_evidence", "inconsistent_input"):
        assert f"`{field}`" in template
    example = json.loads(raw.split("```json\n", 1)[1].split("```", 1)[0])
    assert set(example["reason_codes"]) <= isolated_prompt.VALID_REASON_CODES


def test_v3_default_and_historical_hash_are_preserved(isolated_prompt):
    args = runner.parse_args(["--modo", "vigilancia", "--pares", "cohort.json", "--saida", "out.json"])
    assert args.prompt_version == "v3"
    metadata = runner.configure_prompt(args.prompt_version)
    assert isolated_prompt._PROMPT_PATH.name == "gzcmd_v3_llm_prompt.md"
    effective = isolated_prompt._load_system_prompt() + isolated_prompt._load_user_template()
    assert metadata["prompt_sha256"] == hashlib.sha256(effective.encode()).hexdigest()


@pytest.mark.parametrize("provider,model", [("anthropic", "claude-haiku-5-5"),
                                           ("opencode-go", "deepseek-v4.1-flash")])
@pytest.mark.parametrize("mode", ["vigilancia", "confirmacao"])
def test_v4_cli_loads_file_and_persists_hash_offline(tmp_path, monkeypatch, isolated_prompt, provider, model, mode):
    view = output_free_view()
    runner.audit_view(view)
    serialized = view.to_json(sort_keys=True)
    assert json.loads(serialized)["features"]["nota_final"] == 7.5
    assert not any(field in serialized for field in runner.FORBIDDEN)
    cohort = tmp_path / "cohort.json"
    runner.save(cohort, {"pair_ids": [view.pair_id]})
    args = runner.parse_args(["--provider", provider, "--modelo", model, "--modo", mode,
                              "--pares", str(cohort), "--saida-dir", str(tmp_path / "v4"),
                              "--prompt-version", "v4", "--workers", "1"])
    monkeypatch.setattr(runner, "PROVIDER", provider)
    monkeypatch.setattr(runner, "original_path", lambda _: cohort)
    monkeypatch.setattr(runner, "prepare", lambda *_: {view.pair_id: view})
    monkeypatch.setattr(runner, "configure_go_transport", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-not-a-key")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "offline-not-a-key")

    def offline_review(pid, dossier, selected_model, journal):
        reviewer_class = runner.ant.AnthropicReviewer if provider == "anthropic" else isolated_prompt.LLMReviewer
        reviewer = reviewer_class(api_key="offline-not-a-key", model=selected_model)
        assert isolated_prompt._PROMPT_PATH.name == "gzcmd_v4_llm_prompt.md"
        for role in ("Agent-A", "Agent-B"):
            messages = reviewer._build_messages(dossier, agent_role=role)
            assert "Política única" in messages[0]["content"]
            assert not any(field in str(messages) for field in runner.FORBIDDEN)
        arbiter = reviewer._build_arbiter_messages(dossier, {"decision": "MATCH"}, {"decision": "NONMATCH"})
        assert "Política única" in arbiter[0]["content"]
        return {"pair_id": pid, "decision": "MATCH", "confidence": .8,
                "reason_codes": ["NAME_STRONG", "DOB_STRONG"], "quality_flags": {},
                "protocol": "consensus"}, None

    monkeypatch.setattr(runner, "review_pair", offline_review)
    # Guard both transports: an accidental paid API call fails the test.
    with patch.object(isolated_prompt.LLMReviewer, "_call_api", side_effect=AssertionError("No API calls")), \
         patch.object(runner.ant.AnthropicReviewer, "_call_api", side_effect=AssertionError("No API calls")), \
         patch.object(runner.ant, "_extract_json", runner.ant._extract_json):
        output = runner.run(args)
        assert output["complete"]
        stored = runner.read_json(args.saida)
        expected = hashlib.sha256((runner.PROMPT_DIR / "gzcmd_v4_llm_prompt.md").read_bytes()).hexdigest()
        assert stored["prompt_version"] == stored["metadata"]["prompt_version"] == "v4"
        assert stored["prompt_sha256"] == stored["metadata"]["prompt_sha256"] == expected
        assert stored["metadata"]["prompt_file_sha256"] == expected
        assert args.saida.name == f"exp2_llm_sync_{mode}.json"
        # Never mix a v3 run with a v4 checkpoint, even for the same cohort.
        args.prompt_version = "v3"
        with pytest.raises(ValueError, match="Checkpoint identity changed"):
            runner.run(args)


def test_output_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        runner.parse_args(["--saida", "out.json", "--saida-dir", "other"])
