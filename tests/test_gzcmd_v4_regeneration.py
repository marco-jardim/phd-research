from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from gzcmd.artifacts import preserve_m3v3_artifacts
from gzcmd.config import load_config
from gzcmd.provenance import sha256_file_normalized
from gzcmd.reliability import load_band_rates, reference_rates_model
from gzcmd.runner import build_engine_from_config
from scripts.funil_gzcmd_v2 import export_v31_batch, HAIKU_V3, HAIKU_V4, artifact_paths, load_responses


def test_v4_defaults_to_haiku_not_mixed_pool() -> None:
    source = Path("gzcmd/error_rates_by_band_v4.yaml")
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg = load_config("gzcmd/gzcmd_v3_config.yaml")
    assert reference_rates_model(source) == "claude-haiku-5-5"
    assert Path(cfg.llm_review.estimated_rates_path) == source.resolve()
    for mode in ("vigilancia", "confirmacao"):
        rates = load_band_rates(cfg.llm_review.error_rates_by_band, mode=mode, source=source)
        expected = data["by_mode"][mode]["models"]["claude-haiku-5-5"]["error_rates_by_band"]
        assert rates["grey_high"].e_fp == expected["grey_high"]["e_fp"]
        assert rates["grey_high"].e_fp != data["by_mode"][mode]["pooled"]["error_rates_by_band"]["grey_high"]["e_fp"]
        assert set(rates) == set(cfg.llm_review.error_rates_by_band)
        assert build_engine_from_config(cfg, mode=mode).llm_error_by_band == rates


def test_explicit_missing_reviewer_is_not_silently_pooled(tmp_path) -> None:
    source = tmp_path / "rates.yaml"
    source.write_text(yaml.safe_dump({"by_mode": {"vigilancia": {"pooled": {}}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing explicit reviewer"):
        load_band_rates({"a": {"e_fp": .1, "e_fn": .1}}, mode="vigilancia", source=source, model="nonexistent")


def test_m3v3_archive_is_idempotent_and_detects_tampering(tmp_path) -> None:
    (tmp_path / "data").mkdir()
    figure_dir = tmp_path / "text/latex/figures"
    figure_dir.mkdir(parents=True)
    source = tmp_path / "data/perda_total_por_modelo.csv"
    source.write_text("perda\n10\n", encoding="utf-8")
    (figure_dir / "fig_cap8_varredura_custos.pdf").write_bytes(b"old-figure")
    manifest = preserve_m3v3_artifacts(tmp_path)
    assert len(manifest["files"]) == 2
    saved = tmp_path / "data/perda_total_por_modelo_m3v3.csv"
    assert saved.read_bytes() == source.read_bytes()
    source.write_text("perda\n20\n", encoding="utf-8")
    assert preserve_m3v3_artifacts(tmp_path) == manifest
    assert saved.read_text(encoding="utf-8") == "perda\n10\n"
    saved.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="Preserved artifact changed"):
        preserve_m3v3_artifacts(tmp_path)


def test_batch_and_pending_are_exact_safe_subsets(tmp_path) -> None:
    frame = pd.DataFrame({"action": ["LLM_REVIEW", "MATCH", "LLM_REVIEW"],
                          "nota": [6., 9., 7.], "p_cal": [.2, .99, .4],
                          "band": ["grey_mid", "high", "grey_high"],
                          "decisao_motor": ["LLM", "ALWAYS_MATCH", "LLM"]}, index=pd.Index(["a", "b", "c"], name="pair_id"))
    counts = export_v31_batch(frame, {"a": "UNSURE"}, tmp_path, "vigilancia")
    batch = pd.read_csv(tmp_path / "lote_llm_v31_vigilancia.csv")
    pending = pd.read_csv(tmp_path / "lote_llm_v31_vigilancia_pendentes.csv")
    assert batch.columns.tolist() == ["pair_id", "nota", "p_cal_ml", "banda", "decisao_motor"]
    assert batch.pair_id.tolist() == ["a", "c"]
    assert pending.pair_id.tolist() == ["c"]
    assert counts["com_resposta_haiku_v4"] == 1 and counts["pendentes_haiku_v4"] == 1


def test_reference_artifacts_have_v4_without_model_outputs() -> None:
    assert "anthropic_sem_mo_v4" in artifact_paths("vigilancia", HAIKU_V4)[0].parts
    assert "anthropic" in artifact_paths("vigilancia", HAIKU_V3)[0].parts
    rates = yaml.safe_load(Path("gzcmd/error_rates_by_band_v4.yaml").read_text(encoding="utf-8"))
    for mode, n in (("vigilancia", 1410), ("confirmacao", 435)):
        responses, available = load_responses(mode, HAIKU_V4)
        assert available and len(responses) == n
        path = artifact_paths(mode, HAIKU_V4)[0]
        source = next(s for s in rates["sources"] if s["model"] == "claude-haiku-5-5" and s["mode"] == mode)
        assert source["sha256"] == sha256_file_normalized(path)
        assert json.loads(path.read_text(encoding="utf-8"))["prompt_version"] == "v4"
