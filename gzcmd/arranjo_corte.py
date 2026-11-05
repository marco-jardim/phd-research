"""Nested REFREC calibration below fixed ORL cutoffs; offline S1 primitives.

This module deliberately does not modify ml_oof or any published profile.
Labels are used for training/reliability only, never for S1 outcome reporting.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold

from .calibration import PlattModel, predict_platt
from .ml_oof import N_SPLITS, SEED, _classifier, numeric_features
from .policy_engine import LLMError

CUTOFFS = {6: 7.02408276169, 9: 7.84444860082, 11: 7.90191406713}
MODEL = "claude-haiku-5-5"
REMOVED_FIELDS = tuple("model_outputs." + name for name in (
    "p_cal", "base_choice", "band", "evr", "guardrail", "guardrail_reason"))
SUMMARY_COLUMNS = ["modo", "braco", "banda", "acao", "n"]
RATE_COLUMNS = ["modo", "fold", "banda", "e_fp", "e_fn", "e_unsure"]


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def below_cutoff(raw: pd.DataFrame, *, check_partition: bool = True) -> np.ndarray:
    step = pd.to_numeric(raw["PASSO"], errors="raise")
    score = pd.to_numeric(raw["SCORE"].astype(str).str.replace(",", ".", regex=False), errors="raise")
    cut = step.map(CUTOFFS)
    if cut.isna().any() or score.isna().any():
        raise ValueError("Missing score or preregistered ORL step")
    below = score.lt(cut).to_numpy()
    if check_partition:
        par = pd.to_numeric(raw["PAR"], errors="coerce")
        if (int((~below).sum()) != 211 or int(below.sum()) != 61485
                or not par.loc[~below].eq(1).all()
                or not np.array_equal(par.eq(1).to_numpy(), ~below)
                or int(par.loc[below].eq(2).sum()) != 36):
            raise ValueError("ORL partition differs from 211/61485/36")
    return below


class LogitPlatt:
    """Platt (1999) targets on clipped XGBoost logits; safeguarded optimization.

    Keep the registered slope penalty .001, 100 iterations and tolerance 1e-10.
    The intercept is unpenalized. Line search changes numerical optimization,
    not the logistic loss or its registered penalty.
    """

    def __init__(self):
        self.model = None

    @staticmethod
    def design(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return pd.Series(np.log(p / (1 - p)))

    def fit(self, p, y):
        from scipy.optimize import minimize
        from scipy.special import expit

        x = self.design(p).to_numpy()
        y = np.asarray(y, dtype=int)
        if len(x) != len(y) or not set(np.unique(y)) == {0, 1}:
            raise ValueError("Aligned binary labels required for Platt")
        if not np.isfinite(x).all():
            raise ValueError("Finite logits required for Platt")
        positive, negative = int(y.sum()), int((y == 0).sum())
        target = np.where(y == 1, (positive + 1) / (positive + 2), 1 / (negative + 2))
        design = np.column_stack([np.ones(len(x)), x])
        l2, max_iter, tol = .001, 100, 1e-10
        mean = float(target.mean())
        weights = np.array([np.log(mean / (1 - mean)), 0.0])

        def objective(w):
            z = design @ w
            loss = float(np.sum(np.logaddexp(0, z) - target * z) + .5 * l2 * w[1] ** 2)
            gradient = design.T @ (expit(z) - target)
            gradient[1] += l2 * w[1]
            return loss, gradient

        def hessian(w):
            prediction = expit(design @ w)
            curvature = prediction * (1 - prediction)
            value = design.T @ (curvature[:, None] * design)
            value[1, 1] += l2
            return value

        converged = False
        for _ in range(max_iter):
            loss, gradient = objective(weights)
            if np.max(np.abs(gradient)) <= tol:
                converged = True
                break
            try:
                delta = np.linalg.solve(hessian(weights), gradient)
            except np.linalg.LinAlgError:
                break  # BFGS below does not invert a potentially singular Hessian.
            descent = float(gradient @ delta)
            if not np.isfinite(delta).all() or descent <= 0:
                break
            step = 1.0
            while step >= np.finfo(float).eps:
                candidate = weights - step * delta
                next_loss, _ = objective(candidate)
                if np.isfinite(next_loss) and next_loss <= loss - 1e-4 * step * descent:
                    weights = candidate
                    break
                step *= .5
            else:
                break
            if np.max(np.abs(step * delta)) <= tol:
                converged = True
                break
        self.optimizer = "newton_backtracking"
        if not converged:
            result = minimize(objective, weights, jac=True, method="BFGS",
                              options={"gtol": tol, "maxiter": max_iter})
            weights = result.x
            self.optimizer = "bfgs"
            # BFGS may report precision loss at the finite-precision optimum.
            # Accept that only if the Newton displacement meets the same tol.
            _, gradient = objective(weights)
            try:
                displacement = np.linalg.solve(hessian(weights), gradient)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError("Platt optimizer has singular final curvature") from exc
            if not result.success and np.max(np.abs(displacement)) > tol:
                raise RuntimeError("Platt optimizer did not converge")
        if not np.isfinite(weights).all():
            raise RuntimeError("Platt optimizer returned nonfinite parameters")
        self.model = PlattModel(intercept=float(weights[0]), slope=float(weights[1]))
        return self

    def predict(self, p):
        if self.model is None:
            raise ValueError("Platt calibrator has not been fitted")
        return predict_platt(self.design(p), model=self.model).to_numpy()


def nested_cutoff_probabilities(
    x, y, below, groups, *, par2=None, n_splits=N_SPLITS, seed=SEED,
    model_factory=None, audit=None, include_a3=True, include_platt=True,
):
    """Reuse full-data 5x5 splits; only calibration membership changes for P.

    audit(fold, stage, fit_indices, prediction_indices, outer_test) exposes the
    actual fitting boundary. A3 restricts model fits, but not split construction.
    The all-mask case is exactly the original nested isotonic calculation.
    """
    factory = model_factory or _classifier
    y = np.asarray(y, dtype=int)
    below = np.asarray(below, dtype=bool)
    groups = np.asarray(groups)
    par2 = y.astype(bool) if par2 is None else np.asarray(par2, dtype=bool)
    if not (len(x) == len(y) == len(below) == len(groups) == len(par2)):
        raise ValueError("Aligned arrays required")
    if set(np.unique(y)) != {0, 1} or pd.isna(groups).any():
        raise ValueError("Binary labels and nonmissing groups required")
    arms = ["P", "B1"] + (["P-Platt"] if include_platt else []) + (["A3"] if include_a3 else [])
    probabilities = {arm: np.full(len(y), np.nan) for arm in arms}
    folds = np.full(len(y), -1, dtype=int)
    metadata = []
    outer = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(outer.split(x, y, groups)):
        if not set(groups[train]).isdisjoint(groups[test]):
            raise ValueError("Outer REFREC leakage")
        inner_probs = np.full(len(train), np.nan)
        a3_probs = np.full(len(train), np.nan)
        inner = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed + fold + 1)
        for inner_train, inner_test in inner.split(x.iloc[train], y[train], groups[train]):
            fit_idx, pred_idx = train[inner_train], train[inner_test]
            if not set(groups[fit_idx]).isdisjoint(groups[pred_idx]):
                raise ValueError("Inner REFREC leakage")
            model = factory(y[fit_idx], seed)
            model.fit(x.iloc[fit_idx], y[fit_idx])
            inner_probs[inner_test] = model.predict_proba(x.iloc[pred_idx])[:, 1]
            if audit is not None:
                audit(fold, "inner", fit_idx.copy(), pred_idx.copy(), test.copy())
            if include_a3:
                restricted = fit_idx[below[fit_idx]]
                model3 = factory(y[restricted], seed)
                model3.fit(x.iloc[restricted], y[restricted])
                a3_probs[inner_test] = model3.predict_proba(x.iloc[pred_idx])[:, 1]
                if audit is not None:
                    audit(fold, "A3_inner", restricted.copy(), pred_idx.copy(), test.copy())
        if not np.isfinite(inner_probs).all():
            raise ValueError("Incomplete inner OOF")
        eligible = below[train]
        calibration = train[eligible]
        if len(np.unique(y[calibration])) != 2:
            raise ValueError("Below-cutoff calibration requires both classes")
        fallback = int(par2[calibration].sum()) < 10
        model = factory(y[train], seed)
        model.fit(x.iloc[train], y[train])
        raw_test = model.predict_proba(x.iloc[test])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(inner_probs[eligible], y[calibration])
        platt = LogitPlatt().fit(inner_probs[eligible], y[calibration]) if include_platt or fallback else None
        if fallback:
            if platt is None:
                raise RuntimeError("Principal fallback requires Platt")
            probabilities["P"][test] = platt.predict(raw_test)
        else:
            probabilities["P"][test] = iso.predict(raw_test)
        if include_platt:
            if platt is None:
                raise RuntimeError("P-Platt requires its calibrator")
            probabilities["P-Platt"][test] = platt.predict(raw_test)
        probabilities["B1"][test] = IsotonicRegression(out_of_bounds="clip").fit(
            inner_probs, y[train]).predict(raw_test)
        if audit is not None:
            audit(fold, "calibrator", calibration.copy(), test.copy(), test.copy())
            audit(fold, "outer_and_calibrator", train.copy(), test.copy(), test.copy())
        if include_a3:
            model3 = factory(y[calibration], seed)
            model3.fit(x.iloc[calibration], y[calibration])
            raw3 = model3.predict_proba(x.iloc[test])[:, 1]
            cal3 = LogitPlatt() if fallback else IsotonicRegression(out_of_bounds="clip")
            probabilities["A3"][test] = cal3.fit(a3_probs[eligible], y[calibration]).predict(raw3)
            if audit is not None:
                audit(fold, "A3_calibrator", calibration.copy(), test.copy(), test.copy())
        folds[test] = fold
        metadata.append({"fold": fold, "calibration_n": len(calibration),
                         "calibration_par2": int(par2[calibration].sum()),
                         "principal": "platt" if fallback else "isotonic"})
    if (folds < 0).any() or any(not np.isfinite(p).all() for p in probabilities.values()):
        raise ValueError("Incomplete outer OOF")
    return probabilities, folds, metadata


def estimate_rates(frame: pd.DataFrame, fixed) -> dict[str, LLMError]:
    """M3 estimator: UNSURE is marginal and is never counted as FP or FN."""
    result = {}
    for band, fallback in fixed.items():
        cell = frame.loc[frame.band.eq(band)]
        negative, positive = cell.y.eq(0), cell.y.eq(1)
        enough = len(cell) >= 10
        result[band] = LLMError(
            float((cell.decision.eq("MATCH") & negative).sum() / negative.sum())
            if enough and negative.any() else float(fallback["e_fp"]),
            float((cell.decision.eq("NONMATCH") & positive).sum() / positive.sum())
            if enough and positive.any() else float(fallback["e_fn"]),
            float(cell.decision.eq("UNSURE").mean())
            if enough else float(fallback.get("e_unsure", 0.0)),
        )
    return result


def crossfit_rates(frame, fixed, folds, *, par2_grey=False):
    """A singleton fold means the pooled M3 check; production requires five."""
    result = {}
    for fold in folds:
        train = frame if len(folds) == 1 else frame.loc[frame.fold.ne(fold)]
        rates = estimate_rates(train, fixed)
        if par2_grey:
            grey = train.loc[train.band.isin(["grey_low", "grey_mid", "grey_high", "near_high"])]
            positives = grey.par.eq(2)
            if len(grey) >= 10 and positives.any():
                fn = float((grey.decision.eq("NONMATCH") & positives).sum() / positives.sum())
                rates = {b: replace(r, e_fn=fn) if b in {
                    "grey_low", "grey_mid", "grey_high", "near_high"} else r
                         for b, r in rates.items()}
        result[int(fold)] = rates
    return result


def cache_matches(identity, run, *, mode, pair_id, prompt_sha256, dossier_sha256):
    """No cache hit without the complete identity, including removed fields."""
    return (identity.get("mode") == run.get("mode") == mode
            and identity.get("model") == run.get("model") == MODEL
            and identity.get("prompt_sha256") == run.get("prompt_sha256") == prompt_sha256
            and identity.get("dossier_sha256", {}).get(pair_id) == dossier_sha256
            and run.get("dossier_sha256", {}).get(pair_id) == dossier_sha256
            and set(run.get("removed_fields", [])) == set(REMOVED_FIELDS)
            and run.get("complete") is True
            and run.get("prompt_version") == "v4")


def summarize_actions(out, mode, arm):
    """Allowlist output: deliberately excludes labels and correctness metrics."""
    summary = out.groupby(["band", "action"], observed=True).size().reset_index(name="n")
    summary = summary.rename(columns={"band": "banda", "action": "acao"})
    summary["modo"], summary["braco"] = mode, arm
    return summary[SUMMARY_COLUMNS]


def pending_cost(n):
    return n * (6847 * .10 + 3904 * .50) / 1_000_000 * 1.20
