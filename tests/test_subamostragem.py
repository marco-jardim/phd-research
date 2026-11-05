from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gzcmd.ml_oof import nested_oof_probabilities
from gzcmd.subamostragem import (
    correct_prior, ece_quantile, evaluate, sample_training_indices, summarize,
    undersampled_oof,
)
from local_data import DATA_ROOT, requires_files

ROOT = Path(__file__).resolve().parents[1]


def synthetic():
    x = pd.DataFrame({"signal": np.arange(1200) / 1200})
    y = (np.arange(1200) % 12 == 0).astype(int)
    groups = np.arange(1200) // 3
    return x, y, groups


class SpyModel:
    def fit(self, x, y):
        self.fit_indices = set(x.index)
        self.offset = float(y.mean()) * .01
        return self

    def predict_proba(self, x):
        assert self.fit_indices.isdisjoint(x.index)
        p = .01 + .9 * x.signal.to_numpy() + self.offset
        return np.column_stack([1 - p, p])


@pytest.mark.parametrize("ratio", [1, 5, 20, 50, None])
def test_only_negatives_removed_and_effective_ratio(ratio):
    y = np.array([1] * 13 + [0] * 1000)
    chosen, beta = sample_training_indices(y, ratio, np.random.default_rng(1000))
    assert set(np.flatnonzero(y)) <= set(chosen)
    assert (y[np.setdiff1d(np.arange(len(y)), chosen)] == 0).all()
    assert np.unique(chosen).size == len(chosen)
    negatives = 1000 if ratio is None else min(1000, 13 * ratio)
    assert (y[chosen] == 0).sum() == negatives
    assert beta == negatives / 1000


def test_original_factory_reproduces_unchanged_nested_oof():
    x, y, groups = synthetic()
    expected, folds = nested_oof_probabilities(
        x, y, groups=groups, split_by="refrec", model_factory=lambda *_: SpyModel())
    actual, actual_folds, _ = undersampled_oof(x, y, groups, model_factory=lambda *_: SpyModel())
    np.testing.assert_array_equal(actual["isotonic"], expected)
    np.testing.assert_array_equal(actual["isotonic_prior"], expected)
    np.testing.assert_array_equal(actual_folds, folds)


def test_fit_predict_spies_prove_unchanged_evaluation_and_training_only_sampling():
    x, y, groups = synthetic()
    full_predictions, sampled_predictions, fit_events = [], [], []

    class RecordingModel(SpyModel):
        def predict_proba(self, frame):
            sampled_predictions.append(tuple(frame.index))
            return super().predict_proba(frame)

    class FullModel(SpyModel):
        def predict_proba(self, frame):
            full_predictions.append(tuple(frame.index))
            return super().predict_proba(frame)

    nested_oof_probabilities(x, y, groups=groups, split_by="refrec", model_factory=lambda *_: FullModel())

    def audit(stage, original, selected):
        if stage == "fit":
            assert set(original[y[original] == 1]) <= set(selected)
            assert (y[np.setdiff1d(original, selected)] == 0).all()
            fit_events.append((original, selected))

    _, fold, effective = undersampled_oof(
        x, y, groups, ratio=1, audit=audit, model_factory=lambda *_: RecordingModel())
    assert sampled_predictions == full_predictions
    assert len(fit_events) == len(effective) == 30
    assert all(e["n_positive"] == e["n_negative"] for e in effective)
    for outer in range(5):
        test = np.flatnonzero(fold == outer)
        for original, selected in fit_events[outer * 6:(outer + 1) * 6]:
            assert set(groups[selected]).isdisjoint(groups[test])
            assert set(selected) <= set(original)
    outer_predictions = [i for j, pred in enumerate(sampled_predictions) if j % 6 == 5 for i in pred]
    assert sorted(outer_predictions) == list(range(len(x)))


@pytest.mark.parametrize("beta", [.001, .01, .2, 1.0])
def test_prior_equation_9_inverts_equation_2(beta):
    p = np.array([0, .0001, .01, .1, .5, .9, 1.])
    sampled = p / (p + beta * (1 - p))
    np.testing.assert_allclose(correct_prior(sampled, beta), p, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(correct_prior(.5, .1), 1 / 11)


@pytest.mark.parametrize("p,beta", [(-.1, .5), (1.1, .5), (.2, 0), (.2, 1.1), (np.nan, .5)])
def test_prior_rejects_invalid_inputs(p, beta):
    with pytest.raises(ValueError):
        correct_prior(p, beta)


def test_ece_quantile_handles_constant_probabilities():
    assert ece_quantile([0, 0, 1, 1], [.5] * 4) == 0
    assert ece_quantile([0, 0, 1, 1], [.2] * 4) == pytest.approx(.3)


def aggregate_synthetic(seed):
    x, y, groups = synthetic()
    probabilities, _, _ = undersampled_oof(x, y, groups, ratio=1, removal_seed=seed,
                                           model_factory=lambda *_: SpyModel())
    return evaluate(y, probabilities, np.array([f"private-{i:04}" for i in range(len(y))]),
                    np.ones(len(y), dtype=bool), np.repeat(["6", "9", "11"], 400),
                    np.tile(["F", "M"], 600), ratio="1:1", removal_seed=seed)


def test_outputs_are_aggregate_and_regeneration_identical():
    metrics, curve = aggregate_synthetic(1000)
    metrics2, curve2 = aggregate_synthetic(1000)
    pd.testing.assert_frame_equal(metrics, metrics2, check_exact=True)
    pd.testing.assert_frame_equal(curve, curve2, check_exact=True)
    reference = metrics.copy()
    reference["ratio"] = "original"
    summary, bands = summarize(pd.concat([reference, metrics]), curve)
    assert summary.delta_median.dropna().eq(0).all()
    assert summary.loc[summary.metric.eq("sens96_sexo_MISSING"), "delta_median"].isna().all()
    for table in (metrics, curve, summary, bands):
        assert "pair_id" not in table.columns
        assert "private-" not in table.to_csv(index=False)
    assert {"mean", "minimum", "maximum"} <= set(bands)
    assert "seed" not in bands


def test_checkpoint_atomic_replace_has_bounded_backoff(tmp_path, monkeypatch):
    import scripts.subamostragem_treino as script
    import json
    replace = script.os.replace
    attempts, delays = [], []

    def sharing_violation(source, destination):
        attempts.append(destination)
        if len(attempts) < 3:
            raise PermissionError("Synthetic Windows sharing violation")
        replace(source, destination)

    monkeypatch.setattr(script.os, "replace", sharing_violation)
    monkeypatch.setattr(script.time, "sleep", delays.append)
    path = tmp_path / "checkpoint.json"
    script.atomic_json(path, {"complete": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"complete": True}
    assert delays == [.1, .2]
    assert not list(tmp_path.glob("*.tmp"))


def test_version_gate_refuses_mismatch(monkeypatch):
    import sklearn
    from scripts.subamostragem_treino import versions
    monkeypatch.setattr(sklearn, "__version__", "unexpected")
    with pytest.raises(RuntimeError, match="version gate"):
        versions()


def test_persisted_outputs_never_contain_pair_ids():
    for path in (ROOT / "data").glob("subamostragem_treino*.csv"):
        assert "pair_id" not in pd.read_csv(path, nrows=0).columns


@requires_files(DATA_ROOT / "data/COMPARADORSEMIDENT.csv", DATA_ROOT / "data/pcal_ml_oof.csv",
                DATA_ROOT / "data/pcal_ml_oof.json")
def test_private_reference_reproduces_published_oof(monkeypatch):
    from scripts import subamostragem_treino as cli
    monkeypatch.setattr(cli, "ROOT", DATA_ROOT)
    from scripts.subamostragem_treino import load_inputs, versions
    versions()
    raw, x, y, _, original = load_inputs(DATA_ROOT / "data/COMPARADORSEMIDENT.csv")
    probabilities, fold, _ = undersampled_oof(x, y, raw.REFREC.str.strip().to_numpy())
    np.testing.assert_array_equal(fold, original.fold.to_numpy())
    np.testing.assert_allclose(probabilities["isotonic"], original.p_cal_ml.to_numpy(), rtol=0, atol=1e-14)


@requires_files(DATA_ROOT / "data/COMPARADORSEMIDENT.csv", DATA_ROOT / "data/pcal_ml_oof.csv",
                DATA_ROOT / "data/pcal_ml_oof.json")
def test_private_regeneration_is_identical(monkeypatch):
    from scripts import subamostragem_treino as cli
    monkeypatch.setattr(cli, "ROOT", DATA_ROOT)
    from scripts.subamostragem_treino import load_inputs, versions
    versions()
    raw, x, y, below, _ = load_inputs(DATA_ROOT / "data/COMPARADORSEMIDENT.csv")
    arguments = {"ratio": 1, "removal_seed": 1000}
    first, fold1, counts1 = undersampled_oof(x, y, raw.REFREC.str.strip().to_numpy(), **arguments)
    second, fold2, counts2 = undersampled_oof(x, y, raw.REFREC.str.strip().to_numpy(), **arguments)
    assert counts1 == counts2
    np.testing.assert_array_equal(fold1, fold2)
    for variant in first:
        np.testing.assert_array_equal(first[variant], second[variant])
    from gzcmd.ml_oof import pair_ids
    from scripts.qw3_fairness_analysis import _recode_sex
    evaluation = {"ids": pair_ids(raw).to_numpy(), "below": below,
                  "steps": raw.PASSO.str.strip().to_numpy(),
                  "sexes": _recode_sex(raw).fillna("MISSING").to_numpy(),
                  "ratio": "1:1", "removal_seed": 1000}
    tables1 = evaluate(y, first, **evaluation)
    tables2 = evaluate(y, second, **evaluation)
    for a, b in zip(tables1, tables2):
        assert a.to_csv(index=False) == b.to_csv(index=False)
