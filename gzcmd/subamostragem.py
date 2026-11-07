"""Exploratory post-hoc training-only negative undersampling, offline.

Evaluation and inner isotonic calibration retain the real prevalence. This is
not a registered analysis and does not change any published model or thesis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ml_oof import N_SPLITS, SEED, _classifier, nested_oof_probabilities

RATIOS = (1, 5, 20, 50)
REMOVAL_SEEDS = tuple(range(1000, 1010))
TOP_K = (11, 17, 29, 58, 96, 121, 125, 440, 2522)


def sample_training_indices(y, ratio, rng):
    """Keep every positive; sample at most ratio negatives per positive."""
    y = np.asarray(y, dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Binary training labels required")
    if ratio is None:
        return np.arange(len(y)), 1.0
    if ratio not in RATIOS:
        raise ValueError("Unsupported positive:negative ratio")
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    count = min(len(negatives), ratio * len(positives))
    selected = rng.choice(negatives, size=count, replace=False)
    return np.sort(np.concatenate([positives, selected])), count / len(negatives)


def correct_prior(p, beta):
    """Dal Pozzolo et al. (2015), eq. (9), PDF p. 3; beta retains negatives.

    This assumes probabilities under the sampled prior. Applying it after an
    isotonic fit on real-prevalence labels is only a double-correction diagnostic.
    """
    p, beta = np.asarray(p, dtype=float), np.asarray(beta, dtype=float)
    if (not np.isfinite(p).all() or not np.isfinite(beta).all()
            or ((p < 0) | (p > 1)).any() or ((beta <= 0) | (beta > 1)).any()):
        raise ValueError("Probabilities in [0,1] and beta in (0,1] required")
    return beta * p / (beta * p + (1 - p))


def undersampled_oof(x, y, groups, *, ratio=None, removal_seed=1000,
                     model_factory=None, n_splits=N_SPLITS, audit=None):
    """Delegate unchanged 5x5 splitting/calibration to the original OOF code.

    The factory's wrapper changes only fit membership. Inner prediction sets,
    inner calibration labels and outer evaluation sets are never sampled.
    XGBoost seed remains 42; removal alone uses removal_seed. The original
    classifier computes scale_pos_weight on the actual sampled training labels.
    """
    factory = model_factory or _classifier
    rng = np.random.default_rng(removal_seed)
    positions = pd.Series(np.arange(len(x)), index=x.index)
    if not x.index.is_unique:
        raise ValueError("Unique feature index required")
    outer_raw, outer_beta = np.full(len(x), np.nan), np.full(len(x), np.nan)
    fits, effective = 0, []

    class SampledModel:
        def __init__(self, seed, is_outer):
            self.seed, self.is_outer = seed, is_outer

        def fit(self, frame, labels):
            selected, self.beta = sample_training_indices(labels, ratio, rng)
            sampled_y = np.asarray(labels)[selected]
            self.model = factory(sampled_y, self.seed)
            self.model.fit(frame.iloc[selected], sampled_y)
            effective.append({"stage": "outer" if self.is_outer else "inner",
                              "n_positive": int(sampled_y.sum()),
                              "n_negative": int((sampled_y == 0).sum()),
                              "beta": self.beta})
            if audit is not None:
                audit("fit", frame.index.to_numpy(), frame.iloc[selected].index.to_numpy())
            return self

        def predict_proba(self, frame):
            prediction = self.model.predict_proba(frame)
            if audit is not None:
                audit("predict", frame.index.to_numpy(), frame.index.to_numpy())
            if self.is_outer:
                idx = positions.loc[frame.index].to_numpy()
                outer_raw[idx], outer_beta[idx] = prediction[:, 1], self.beta
            return prediction

    def sampled_factory(labels, seed):
        nonlocal fits
        outer = fits % (n_splits + 1) == n_splits
        fits += 1
        return SampledModel(seed, outer)

    isotonic, folds = nested_oof_probabilities(
        x, y, groups=groups, split_by="refrec", seed=SEED,
        n_splits=n_splits, model_factory=sampled_factory)
    if not np.isfinite(outer_raw).all() or not np.isfinite(outer_beta).all():
        raise ValueError("Incomplete outer predictions")
    return {"isotonic": isotonic, "isotonic_prior": correct_prior(isotonic, outer_beta),
            "raw": outer_raw, "raw_prior": correct_prior(outer_raw, outer_beta)}, folds, effective


def ece_quantile(y, p):
    """10 probability-quantile bins; ties stay together, duplicate edges drop.

    Constant probabilities form one bin. Empty intervals contribute zero.
    """
    y, p = np.asarray(y), np.asarray(p)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    bins = np.searchsorted(edges[1:-1], p, side="left")
    return float(sum(np.mean(bins == b) * abs(y[bins == b].mean() - p[bins == b].mean())
                     for b in np.unique(bins)))


def evaluate(y, probabilities, ids, below, steps, sexes, *, ratio, removal_seed):
    """Return only aggregate long metrics and aggregate cumulative counts."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y, below = np.asarray(y), np.asarray(below, dtype=bool)
    records, curves = [], []
    for variant, p in probabilities.items():
        for population, mask in (("below", below), ("full", np.ones(len(y), dtype=bool))):
            labels, scores = y[mask], p[mask]
            values = {"auc": roc_auc_score(labels, scores),
                      "auc_pr": average_precision_score(labels, scores),
                      "brier": brier_score_loss(labels, scores),
                      "ece": ece_quantile(labels, scores),
                      "n": len(labels), "positives": int(labels.sum())}
            if population == "below":
                # Identical exact-k tie rule to comparador_mesma_carga.py.
                order = np.lexsort((np.asarray(ids)[mask], -scores))
                cumulative = labels[order].cumsum()
                for k in TOP_K:
                    if k <= len(order):
                        values[f"par2_top_{k}"] = int(cumulative[k - 1])
                for target in (9, 23, 36):
                    values[f"k_min_{target}"] = (int(np.searchsorted(cumulative, target) + 1)
                                                   if cumulative[-1] >= target else np.nan)
                selected = np.zeros(len(order), dtype=bool)
                selected[order[:96]] = True
                for dimension, groups in (("passo", np.asarray(steps)[mask]),
                                          ("sexo", np.asarray(sexes)[mask])):
                    names = ("6", "9", "11") if dimension == "passo" else ("F", "M", "MISSING")
                    for group in names:
                        cell = groups.astype(str) == group
                        den = int(labels[cell].sum())
                        num = int(labels[cell & selected].sum())
                        eligible = group != "MISSING" and int(cell.sum()) >= 30
                        prefix = f"sens96_{dimension}_{group}"
                        values[prefix] = num / den if den and eligible else np.nan
                        values[prefix + "_den"] = den
                        values[prefix + "_num"] = num
                        values[prefix + "_n"] = int(cell.sum())
                curves.extend({"ratio": ratio, "seed": removal_seed, "variant": variant,
                               "k": k, "par2": int(cumulative[k - 1])}
                              for k in range(1, min(2522, len(order)) + 1))
            records.extend({"ratio": ratio, "seed": removal_seed, "variant": variant,
                            "population": population, "metric": metric, "value": float(value)}
                           for metric, value in values.items())
    return pd.DataFrame(records), pd.DataFrame(curves)


def summarize(metrics, curves):
    keys = ["variant", "population", "metric"]
    reference = metrics.loc[metrics.ratio.eq("original"), keys + ["value"]].rename(
        columns={"value": "reference"})
    summary = metrics.groupby(["ratio", *keys], sort=True, dropna=False).value.agg(
        median="median", minimum="min", maximum="max", seeds="count").reset_index()
    summary = summary.merge(reference, on=keys, validate="many_to_one")
    for column in ("median", "minimum", "maximum"):
        summary[f"delta_{column}"] = summary[column] - summary.reference
    curve = curves.groupby(["ratio", "variant", "k"], sort=True).par2.agg(
        mean="mean", minimum="min", maximum="max").reset_index()
    return summary, curve
