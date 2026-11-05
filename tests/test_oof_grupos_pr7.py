from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from gzcmd.ml_oof import nested_oof_probabilities
from gzcmd.runner import run_v3
from local_data import DATA_ROOT, requires_files, requires_reviewer_sources


@pytest.mark.parametrize("split_by", ["refrec", "comprec"])
def test_nested_groups_disjoint_outer_inner_and_calibrator(split_by):
    groups = np.repeat(np.arange(50), 4)
    x = pd.DataFrame({"signal": np.arange(200) / 200})
    y = np.tile([0, 1, 0, 1], 50)
    events = []

    class Model:
        def fit(self, frame, labels):
            self.groups = set(groups[frame.index])

        def predict_proba(self, frame):
            assert self.groups.isdisjoint(groups[frame.index])
            p = frame.signal.to_numpy()
            return np.column_stack([1 - p, p])

    class Calibrator:
        def fit(self, p, labels):
            self.groups = set(groups[np.rint(p * 200).astype(int)])

        def predict(self, p):
            assert self.groups.isdisjoint(groups[np.rint(p * 200).astype(int)])
            return p

    def audit(fold, phase, train, test, outer_test):
        assert set(groups[train]).isdisjoint(groups[test])
        assert set(groups[train]).isdisjoint(groups[outer_test])
        if phase == "inner":
            assert set(groups[test]).isdisjoint(groups[outer_test])
        events.append(phase)

    p, fold = nested_oof_probabilities(x, y, groups=groups, split_by=split_by,
                                       model_factory=lambda *_: Model(), calibrator_factory=Calibrator, audit=audit)
    assert len(events) == 30
    assert np.isfinite(p).all()
    assert pd.DataFrame({"group": groups, "fold": fold}).groupby("group").fold.nunique().eq(1).all()
    again, again_fold = nested_oof_probabilities(x, y, groups=groups, split_by=split_by,
                                                model_factory=lambda *_: Model(), calibrator_factory=Calibrator)
    np.testing.assert_array_equal(fold, again_fold)
    np.testing.assert_array_equal(p, again)


def test_invalid_group_inputs_rejected():
    x, y = pd.DataFrame({"x": np.arange(40)}), np.tile([0, 1], 20)
    for groups in (None, ["one"] * 40, [None] * 40):
        with pytest.raises(ValueError):
            nested_oof_probabilities(x, y, split_by="refrec", groups=groups)
    with pytest.raises(ValueError, match="split_by"):
        nested_oof_probabilities(x, y, split_by="invalid")


def test_group_mode_changes_cache_identity_and_normalizes_header_metadata(tmp_path, monkeypatch):
    import gzcmd.ml_oof as module
    source, output = tmp_path / "input.csv", tmp_path / "oof.csv"
    pd.DataFrame({"COMPREC,C,12,0": [str(i) for i in range(20)],
                  "REFREC,C,12,0": [str(i // 2) for i in range(20)], "PASSO": ["1"] * 20,
                  "PAR": np.tile([0, 1], 10), "nota final": np.linspace(1, 10, 20)}).to_csv(source, sep=";", index=False)
    calls = []

    def nested(x, y, **kwargs):
        calls.append(kwargs)
        return np.full(len(y), .5), np.arange(len(y)) % 5

    monkeypatch.setattr(module, "nested_oof_probabilities", nested)
    module.fit_xgb_oof(source, output_path=output, reuse=True, split_by="row")
    module.fit_xgb_oof(source, output_path=output, reuse=True, split_by="row")
    module.fit_xgb_oof(source, output_path=output, reuse=True, split_by="refrec")
    assert len(calls) == 2
    assert calls[-1]["split_by"] == "refrec"
    assert len(set(calls[-1]["groups"])) == 10
    columns = pd.read_csv(output, nrows=0).columns
    assert not {"y", "PAR", "REFREC", "COMPREC"}.intersection(columns)


@pytest.mark.parametrize("mode,expected", [("vigilancia", 1410), ("confirmacao", 435)])
@requires_files(DATA_ROOT / "data/COMPARADORSEMIDENT.csv")
@requires_reviewer_sources()
def test_legacy_v3_cohort_and_guards(mode, expected):
    root = Path(__file__).resolve().parents[1]
    out, summary = run_v3(input_csv=DATA_ROOT / "data/COMPARADORSEMIDENT.csv",
                          config_path=root / "gzcmd/gzcmd_v3_config.yaml", mode=mode,
                          p_cal="fit_platt", policy_version="v3")
    assert summary.llm_used == expected
    assert summary.guardrails["ALWAYS_MATCH"] == 4
    assert out.action.eq("LLM_REVIEW").sum() == expected
    from gzcmd.ml_oof import pair_ids
    from scripts.funil_gzcmd_v2 import HAIKU_V4, load_responses
    historic, _ = load_responses(mode, HAIKU_V4)
    assert set(pair_ids(out.loc[out.action.eq("LLM_REVIEW")])) == set(historic)


@requires_files(DATA_ROOT / "data/COMPARADORSEMIDENT.csv")
def test_runner_honors_configured_split_by(monkeypatch):
    import gzcmd.runner as runner
    from gzcmd.config import load_config
    cfg = load_config("gzcmd/gzcmd_v3_config.yaml")
    cfg = replace(cfg, evaluation=replace(cfg.evaluation, split_by="comprec"))
    sentinel = RuntimeError("stopped before fitting")
    fit = Mock(side_effect=sentinel)
    monkeypatch.setattr(runner, "load_config", lambda _: cfg)
    monkeypatch.setattr(runner, "fit_xgb_oof", fit)
    with pytest.raises(RuntimeError, match="stopped before fitting"):
        runner.run_v3(input_csv=DATA_ROOT / "data/COMPARADORSEMIDENT.csv", config_path="unused", mode="vigilancia")
    assert fit.call_args.kwargs["split_by"] == "comprec"


@pytest.mark.parametrize("field,new", [("prompt_sha256", "another-prompt"), ("model", "another-model"), ("cohort", "v31_lot")])
def test_journal_identity_rejects_cross_run_mix(tmp_path, field, new):
    from scripts.run_exp2_sem_model_outputs import ensure_journal_identity
    identity = {"prompt_sha256": "hash", "model": "claude-haiku-5-5", "cohort": "historical"}
    ensure_journal_identity(tmp_path, identity)
    with pytest.raises(ValueError, match="Journal identity changed"):
        ensure_journal_identity(tmp_path, {**identity, field: new})


def test_orphan_journal_requires_identity(tmp_path):
    from scripts.run_exp2_sem_model_outputs import ensure_journal_identity, save
    identity = {"prompt_sha256": "hash", "model": "claude-haiku-5-5", "cohort": "historical"}
    save(tmp_path / ("a" * 16 + ".result.json"), {"pair_id": "a" * 16, "decision": "MATCH"})
    with pytest.raises(ValueError, match="require identity.json"):
        ensure_journal_identity(tmp_path, identity)
    ensure_journal_identity(tmp_path, identity, checkpoint_validated=True)
    ensure_journal_identity(tmp_path, identity)


def test_privacy_label_detection_does_not_remove_aggregate_fields():
    from scripts.auditar_privacidade_pr7 import label_columns
    assert label_columns(["y", "PAR,C,1,0", "gabarito", "n_vinculos", "FN"]) == ["y", "PAR,C,1,0", "gabarito"]
