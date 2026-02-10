from __future__ import annotations

from pathlib import Path

from gzcmd.config import load_config


def test_load_config_parses_guardrails_calibration_evaluation(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
version: 3

bands:
  definitions:
    low: {min: 0, max: 5}
    high: {min: 5, max: 999, inclusive_max: true}

decision_policy:
  modes:
    vigilancia:
      costs: {false_positive: 10, false_negative: 50, llm_review: 0.25}
      auto: {min_auto_match_threshold: 0.9, max_auto_nonmatch_threshold: 0.1}
      llm_budget: {max_calls_per_window: 5}

llm_review:
  enabled: false

guardrails:
  temporal_days: 200
  nota_always_match: 10
  nota_always_nonmatch: 2.5
  homonimia_min_nota: 8
  homonimia_year_gap: 7

calibration:
  clip_min: 1.0e-5
  clip_max: 0.999
  platt:
    l2: 0.01
    max_iter: 50
    tol: 1.0e-9

evaluation:
  test_size: 0.25
  seeds: [1, 2, 3]
  split_by: comprec
  group_stratify: false
""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.guardrails.temporal_days == 200
    assert cfg.guardrails.nota_always_nonmatch == 2.5
    assert cfg.guardrails.homonimia_year_gap == 7.0

    assert cfg.calibration.clip_min == 1.0e-5
    assert cfg.calibration.clip_max == 0.999
    assert cfg.calibration.platt.l2 == 0.01
    assert cfg.calibration.platt.max_iter == 50

    assert cfg.evaluation.test_size == 0.25
    assert cfg.evaluation.seeds == (1, 2, 3)
    assert cfg.evaluation.split_by == "comprec"
    assert cfg.evaluation.group_stratify is False


def test_load_config_defaults_when_sections_missing(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
version: 3
bands:
  definitions:
    low: {min: 0, max: 10, inclusive_max: true}
decision_policy:
  modes:
    vigilancia:
      costs: {false_positive: 10, false_negative: 50, llm_review: 0.25}
      llm_budget: {max_calls_per_window: 1}
llm_review:
  enabled: false
""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.guardrails.temporal_days == 180
    assert cfg.calibration.clip_min == 1e-6
    assert cfg.evaluation.test_size == 0.3
    assert cfg.evaluation.seeds == (42,)
