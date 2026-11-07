import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from scripts.validar_ancoras_beta import (
    NAMES, beta_design, binomial_mixture, fit_beta_moments, fs_posterior,
    infer_fs_base, latex_table, success_metrics,
)


def test_beta_identity_clipping_and_monotonicity():
    s = np.r_[0., np.linspace(.001, .999, 100), 1.]
    x = beta_design(s)
    np.testing.assert_allclose(expit(x @ [1, 1, 0]), np.clip(s, 1e-6, 1 - 1e-6))
    assert (np.diff(expit(x @ [.3, 2., -1])) >= 0).all()
    with pytest.raises(ValueError):
        beta_design(np.array([np.nan]))


def test_beta_moments_recover_known_curve_without_labels():
    s = np.linspace(.001, .999, 600)
    masks = [s < .3, (s >= .3) & (s < .7), s >= .7]
    truth = expit(beta_design(s) @ [1.3, .6, -.2])
    p, audit = fit_beta_moments(s, masks, [truth[m].mean() for m in masks], starts=4)
    np.testing.assert_allclose(p, truth, atol=1e-6)
    assert audit["objective"] < 1e-10
    assert audit["a"] >= 0 and audit["b"] >= 0


def test_boundary_targets_and_small_anchor_are_not_smoothed():
    s = np.linspace(.01, .99, 80)
    masks = [s < .5, (s >= .5) & (s < .7), s >= .7]
    p, audit = fit_beta_moments(s, masks, [0., .5, 1.], starts=4)
    assert audit["targets"] == [0., .5, 1.]
    assert np.isfinite(p).all()
    assert min(audit["standard_errors_plugin"]) > 0


def test_fourth_aggregate_moment_is_fitted_without_individual_targets():
    s = np.linspace(.001, .999, 500)
    masks = [s < .2, (s >= .8) & (s < .9), s >= .9, (s >= .2) & (s < .8)]
    truth = expit(beta_design(s) @ [.7, 1.2, -.8])
    targets = [float(truth[mask].mean()) for mask in masks]
    p, audit = fit_beta_moments(s, masks, targets, starts=4)
    np.testing.assert_allclose(p, truth, atol=1e-6)
    np.testing.assert_allclose(audit["means"], targets, atol=1e-7)
    assert len(audit["weighted_residuals"]) == 4
    assert len(audit["jacobian_singular_values"]) == 3


def test_fs_prior_em_uses_likelihood_ratio_and_audited_units():
    m, u = np.array([.99, .75, .95]), np.array([.01, .02, .05])
    floor = np.log2((1 - m) / (1 - u)).sum()
    base, audit = infer_fs_base(np.array([floor, 15.]))
    assert base == 2 and abs(audit["observed_floor"] + 12.84813778786) < 1e-9
    w = np.r_[np.full(900, -8.), np.full(100, 8.)]
    q, audit = fs_posterior(w, starts=4)
    assert q.mean() == pytest.approx(audit["prior"], abs=1e-10)
    np.testing.assert_allclose(q, expit(logit(audit["prior"]) + np.log(2) * w))
    assert q[:900].mean() < q[900:].mean()


def test_success_requires_nonempty_high_probability_bin():
    y = np.array([0, 1])
    result = success_metrics(y, np.array([.1, .8]), .5)
    assert result["high_n"] == 0 and not result["success_joint"]
    result = success_metrics(y, np.array([.001, .99]), .1)
    assert result["success_joint"]


def test_binomial_fallback_is_deterministic_and_unlabeled():
    nota = np.r_[np.linspace(1, 4, 150), np.linspace(8, 10, 25)]
    p, audit = binomial_mixture(nota)
    q, _ = binomial_mixture(nota)
    np.testing.assert_array_equal(p, q)
    assert p[:150].mean() < p[150:].mean()
    assert audit["theta"][0] < audit["theta"][1]


def test_latex_table_has_booktabs_decimal_comma_and_all_five_models():
    valid = pd.DataFrame([{"calibracao": name, "escopo": "cinza", "brier": .001,
                           "ece_10": .002, "high_predicted": np.nan if name == "platt_nota" else .95,
                           "high_observed": np.nan if name == "platt_nota" else .94}
                          for name in NAMES])
    tex = latex_table(valid)
    assert r"\toprule" in tex and r"\bottomrule" in tex
    assert "0,001000" in tex and "n.a." in tex
    assert all(name in tex for name in NAMES.values())


def test_wrong_exact_grey_count_makes_required_ece_impossible():
    from scripts.validar_ancoras_ml import metrics

    y = np.r_[np.ones(151, dtype=int), np.zeros(21620 - 151, dtype=int)]
    expected_count = 99.91677175101105
    p = np.full(len(y), expected_count / len(y))
    lower_bound = abs(p.mean() - y.mean())
    assert metrics(y, p)["ece_10"] == pytest.approx(lower_bound)
    assert lower_bound > 1.2 * .00075774
