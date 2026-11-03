"""Supervised, nested cross-fitted XGBoost + isotonic calibration (no PII output).

Unlike calibrating a global OOF vector, outer-fold labels cannot indirectly enter
the calibrator through another fold's classifier. Every outer fold gets its own
inner OOF training probabilities. This is an evaluation artifact, not a deployable
model or label-free anchor calibration. Seeds/splits are deterministic row splits.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

from .calibration import compute_p_cal, fit_platt_from_df
from .llm_dossier import make_pair_id
from .loader import resolve_prefixed_column

SEED = 42
N_SPLITS = 5
METHOD = "nested-xgb-isotonic-5x5-v1"
FEATURE_PREFIXES = ("NOME ", "NOMEMAE ", "DTNASC ", "CODMUNRES ", "ENDERECO ")


def pair_ids(df: pd.DataFrame) -> pd.Series:
    cols = [resolve_prefixed_column(df, name) for name in ("COMPREC", "REFREC", "PASSO")]
    if any(c is None for c in cols):
        raise ValueError("Missing pair metadata")
    values = [df[c].astype(str).str.strip() for c in cols]
    ids = pd.Series([make_pair_id(*v) for v in zip(*values)], index=df.index)
    if not ids.is_unique:
        raise ValueError("pair_id must be unique")
    return ids


def numeric_features(raw: pd.DataFrame) -> pd.DataFrame:
    # Allowlist comparator fields, never raw records, identifiers or labels.
    cols = [c for c in raw if c.startswith(FEATURE_PREFIXES) or c in {"nota final", "nota_final"}]
    x = raw[cols].apply(lambda s: pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce"))
    return x.astype("float64").fillna(0.0)


def _classifier(y: np.ndarray, seed: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=float((y == 0).sum() / (y == 1).sum()),
        eval_metric="logloss", n_jobs=8, random_state=seed, verbosity=0,
    )


def nested_oof_probabilities(
    x: pd.DataFrame, y: np.ndarray, *, n_splits: int = N_SPLITS, seed: int = SEED,
    model_factory=None, calibrator_factory=None, audit=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return probabilities/fold; no fit (including inner fits) sees outer test rows.

    Optional factories/audit make the actual fit/predict boundary testable without
    substituting an in-sample classifier or requiring expensive trees in unit tests.
    """
    model_factory = model_factory or _classifier
    calibrator_factory = calibrator_factory or (lambda: IsotonicRegression(out_of_bounds="clip"))
    y = np.asarray(y, dtype=int)
    if len(x) != len(y) or not set(np.unique(y)) == {0, 1}:
        raise ValueError("Aligned binary labels required")
    if np.bincount(y).min() < n_splits + 1:
        raise ValueError("Too few examples per class for nested stratified OOF")
    probabilities = np.full(len(y), np.nan)
    folds = np.full(len(y), -1, dtype=int)
    outer = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(outer.split(x, y)):
        inner_probs = np.full(len(train), np.nan)
        inner = StratifiedKFold(n_splits, shuffle=True, random_state=seed + fold + 1)
        for inner_train, inner_test in inner.split(x.iloc[train], y[train]):
            fit_idx, pred_idx = train[inner_train], train[inner_test]
            model = model_factory(y[fit_idx], seed)
            model.fit(x.iloc[fit_idx], y[fit_idx])
            inner_probs[inner_test] = model.predict_proba(x.iloc[pred_idx])[:, 1]
            if audit is not None:
                audit(fold, "inner", fit_idx.copy(), pred_idx.copy(), test.copy())
        if not np.isfinite(inner_probs).all():
            raise ValueError("Incomplete inner OOF predictions")
        calibrator = calibrator_factory()
        calibrator.fit(inner_probs, y[train])
        model = model_factory(y[train], seed)
        model.fit(x.iloc[train], y[train])
        probabilities[test] = calibrator.predict(model.predict_proba(x.iloc[test])[:, 1])
        folds[test] = fold
        if audit is not None:
            audit(fold, "outer_and_calibrator", train.copy(), test.copy(), test.copy())
    if not np.isfinite(probabilities).all() or (folds < 0).any():
        raise ValueError("Incomplete outer OOF predictions")
    return probabilities, folds


def fit_xgb_oof(
    input_csv: str | Path, *, output_path: str | Path = "data/pcal_ml_oof.csv",
    reuse: bool = False,
) -> pd.DataFrame:
    """Persist only pair hash, score, Platt comparator, calibrated OOF and fold.

    A cache is opt-in and tied to source/code/package fingerprints; it cannot be
    reused on another dataset or after changes to this implementation.
    """
    import sklearn
    import xgboost

    source, output = Path(input_csv), Path(output_path)
    fingerprint = {
        "method": METHOD, "seed": SEED, "folds": N_SPLITS,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
    }
    meta_path = output.with_suffix(".json")
    if reuse and output.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if (metadata.get("fingerprint") == fingerprint
                and metadata.get("csv_sha256") == hashlib.sha256(output.read_bytes()).hexdigest()):
            return pd.read_csv(output)
    # dtype=str avoids Arrow/nullable arrays; decimal-comma is converted explicitly.
    raw = pd.read_csv(source, sep=";", dtype=str, low_memory=False)
    x = numeric_features(raw)
    y = pd.to_numeric(raw["PAR"], errors="coerce").fillna(0).isin([1, 2]).astype(int).to_numpy()
    p, fold = nested_oof_probabilities(x, y)
    score = x["nota final"] if "nota final" in x else x["nota_final"]
    platt_df = pd.DataFrame({"nota final": score, "TARGET": y})
    platt = compute_p_cal(platt_df, method="platt", model=fit_platt_from_df(platt_df))
    result = pd.DataFrame({"pair_id": pair_ids(raw), "nota": score,
                           "p_cal_platt": platt, "p_cal_ml": p, "fold": fold})
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    meta_path.write_text(json.dumps({"fingerprint": fingerprint, "features": list(x.columns),
                                    "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
    return result
