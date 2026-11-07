"""Inspect actual fit arguments, independently of the production audit hook."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from gzcmd.arranjo_corte import IsotonicRegression, LogitPlatt, nested_cutoff_probabilities
from test_arranjo_corte import Model, synthetic


@pytest.mark.parametrize("fallback", [False, True])
def test_calibrator_fit_receives_only_inner_oof_below_outer_training(monkeypatch, fallback):
    x, y, groups = synthetic()
    below = np.arange(len(y)) % 4 != 0
    par2 = (y == 1) & below
    if fallback:
        par2[np.flatnonzero(par2)[8:]] = False
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(x, y, groups))
    schedule = [(fold, arm) for fold in range(5) for arm in ("P", "P-Platt", "B1", "A3")]
    lookup, observed = {}, []
    predict = Model.predict_proba

    def tagged_predict(self, frame):
        result = predict(self, frame)
        for idx, score in zip(frame.index, result[:, 1]):
            assert float(score) not in lookup or lookup[float(score)] == idx
            lookup[float(score)] = idx
        return result

    monkeypatch.setattr(Model, "predict_proba", tagged_predict)

    def spy(original, kind):
        def fit(self, p, labels, *args, **kwargs):
            fold, arm = schedule[len(observed)]
            # Recover the indices from actual numerical predictions received by
            # fit, not from the indices claimed by nested_cutoff_probabilities.
            received = np.asarray([lookup[float(v)] for v in p])
            train, test = splits[fold]
            expected = train if arm == "B1" else train[below[train]]
            np.testing.assert_array_equal(received, expected)
            np.testing.assert_array_equal(labels, y[received])
            assert len(received) == len(expected)
            assert set(received).isdisjoint(test)
            assert set(groups[received]).isdisjoint(groups[test])
            if arm != "B1":
                assert below[received].all()
            assert kind == ("platt" if arm == "P-Platt" or (fallback and arm == "A3") else "isotonic")
            observed.append({"fold": fold, "arm": arm, "n": len(received), "indices": received.copy()})
            return original(self, p, labels, *args, **kwargs)
        return fit

    monkeypatch.setattr(IsotonicRegression, "fit", spy(IsotonicRegression.fit, "isotonic"))
    monkeypatch.setattr(LogitPlatt, "fit", spy(LogitPlatt.fit, "platt"))
    probabilities, _, _ = nested_cutoff_probabilities(
        x, y, below, groups, par2=par2, seed=42, model_factory=lambda *_: Model())
    assert len(observed) == 20
    assert [e["arm"] for e in observed] == [arm for _, arm in schedule]
    if fallback:
        np.testing.assert_array_equal(probabilities["P"], probabilities["P-Platt"])


def test_inverting_above_cutoff_labels_cannot_enter_P_A3_or_Platt_calibration():
    # Above-cutoff quartets in each REFREC preserve sum(y) and sum(signal*y)
    # on inversion, hence this label-sensitive classifier's predictions and
    # stratified splits. Their calibration relationship DOES change (B1 is the
    # positive leakage control). This isolates calibration leakage, not P's
    # legitimate full-data classifier training on above-cutoff labels in general.
    groups = np.repeat(np.arange(100), 6)
    signal = np.tile([.2, .8, .1, .35, .6, .85], 100)
    x = pd.DataFrame({"signal": signal})
    y = np.tile([0, 1, 0, 1, 1, 0], 100)
    below = np.tile([True, True, False, False, False, False], 100)
    flipped = y.copy()
    flipped[~below] = 1 - flipped[~below]
    original, folds, _ = nested_cutoff_probabilities(
        x, y, below, groups, seed=42, model_factory=lambda *_: Model())
    changed, changed_folds, _ = nested_cutoff_probabilities(
        x, flipped, below, groups, seed=42, model_factory=lambda *_: Model())
    np.testing.assert_array_equal(folds, changed_folds)
    for arm in ("P", "P-Platt", "A3"):
        np.testing.assert_allclose(original[arm], changed[arm], rtol=0, atol=1e-12)
    assert np.max(np.abs(original["B1"] - changed["B1"])) > .01
    # Negative control: the classifier genuinely depends on training labels.
    a, b = Model(), Model()
    a.fit(x.iloc[:200], y[:200])
    b.fit(x.iloc[:200], np.zeros(200, dtype=int))
    assert not np.array_equal(a.predict_proba(x.iloc[200:]), b.predict_proba(x.iloc[200:]))
