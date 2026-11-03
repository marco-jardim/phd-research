"""Checks locais M5: máscaras externas, kernel do pacote, ECE e isolamento OOF."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from scripts.validar_ancoras_ml import (
    decision, fit_anchor_platt, fit_masked_anchor_platt, metrics, predict, rates,
    reconstruct_margins, reliability, table,
)


def test_mask_adapter_matches_public_package_on_score_anchors():
    score = np.linspace(0, 10, 300)
    model = fit_anchor_platt(pd.Series(score), r_plus=.97, r_minus=.0001)
    adapted, audit = fit_masked_anchor_platt(score, [score >= 9, score < 5], [.97, .0001])
    np.testing.assert_allclose(predict(model, score), predict(adapted, score), atol=1e-12)
    assert audit["success"]


def test_external_masks_not_replaced_by_margin_thresholds():
    score = np.tile(np.linspace(-5, 5, 100), 3)
    plus = np.arange(300) < 100
    minus = np.arange(300) >= 100
    score[plus] += .01
    # Quase a mesma distribuição nas duas máscaras: alvos incompatíveis.
    model, audit = fit_masked_anchor_platt(score, [plus, minus], [.9, .01])
    assert audit["max_logit_residual"] > 1
    assert np.isfinite(predict(model, score)).all()


def test_rates_frontier_is_explicit_and_small_anchors_rejected():
    y = np.zeros(80, dtype=int)
    y[:40] = 1
    targets, audit = rates(y, [np.arange(80) < 40, np.arange(80) >= 40])
    assert targets == [40.5 / 41, .5 / 41]
    assert all(r["suavizada"] for r in audit)
    with pytest.raises(ValueError, match="insuficiente"):
        rates(y, [np.arange(80) < 10])


def test_ece_bins_and_tied_deciles():
    y = np.array([0, 0, 1, 1])
    p = np.array([0., .1, .9, 1.])
    assert sum(r["n"] for r in reliability(y, p, "uniforme")) == 4
    assert metrics(y, p)["ece_10"] == pytest.approx(.05)
    tied = reliability(y, np.full(4, .5), "decil")
    assert len(tied) == 1 and tied[0]["n"] == 4


def test_decision_uses_historical_no_guard_simulation_and_table():
    nota = np.array([6., 6.])
    y = np.array([0, 1])
    p = np.array([.00001, .99999])
    probabilities = {name: p for name in ["platt_nota", "ancoras_ml", "isotonica_ml"]}
    result = decision(nota, y, probabilities)
    assert len(result) == 6
    assert (result.auto_match == 1).all() and (result.auto_nonmatch == 1).all()
    assert (result.revisao == 0).all() and (result.perda_revisor_config == 0).all()
    validation = pd.DataFrame([{"calibracao": name, "escopo": "cinza", **metrics(y, p)}
                               for name in probabilities])
    latex = table(validation, result)
    assert r"\label{tab:cap8-ancoras-ml}" in latex
    assert latex.count(r"\begin{tabular}") == latex.count(r"\end{tabular}") == 2


@pytest.mark.parametrize("split_by", ["row", "refrec"])
def test_reconstruction_does_not_fit_on_outer_test(monkeypatch, split_by):
    import scripts.validar_ancoras_ml as module
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    n = 500
    x = pd.DataFrame({"score": np.linspace(-4, 4, n)})
    y = np.arange(n) % 2
    nota = np.tile([3., 4., 9., 9.5, 6.], n // 5)
    groups = np.arange(n) // 10
    calls = []

    class Model:
        def fit(self, xx, yy):
            self.train = xx.index.to_numpy()
            return self

        def predict_proba(self, xx):
            assert not np.intersect1d(self.train, xx.index).size
            if split_by != "row":
                assert set(groups[self.train]).isdisjoint(groups[xx.index])
            calls.append((self.train.copy(), xx.index.to_numpy()))
            p = expit(xx.score.to_numpy())
            return np.column_stack([1 - p, p])

        def predict(self, xx, *, output_margin):
            assert output_margin
            return xx.score.to_numpy()

    monkeypatch.setattr(module, "_classifier", lambda y, seed: Model())
    # Ordenação substituta para média da âncora positiva > negativa.
    x["score"] = nota - 6 + np.linspace(0, .1, n)
    result, audit = reconstruct_margins(x, y, nota, split_by=split_by,
                                        groups=groups if split_by != "row" else None)
    assert len(calls) == 30 and len(audit) == 5
    splitter = StratifiedGroupKFold if split_by != "row" else StratifiedKFold
    outer_cv = splitter(5, shuffle=True, random_state=42)
    outer = list(outer_cv.split(x, y, groups) if split_by != "row" else outer_cv.split(x, y))
    for fold, (train, test) in enumerate(outer):
        for fit, pred in calls[fold * 6: (fold + 1) * 6 - 1]:
            assert set(fit) <= set(train) and set(pred) <= set(train)
            assert not set(test) & set(fit)
        np.testing.assert_array_equal(np.flatnonzero(result.fold == fold), test)
