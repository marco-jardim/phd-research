"""
Robustness Analysis (58 uniform features)
==========================================

C1: 5-fold stratified CV on best configs from each category
C2: Imbalance sensitivity (SMOTE, BorderlineSMOTE, ADASYN, SMOTETomek, class_weight)
C3: SHAP analysis on best RF model + feature importance per score band

Outputs:
  - data/cv_5fold_results.csv
  - data/imbalance_sensitivity.csv
  - data/shap_importance.csv
  - data/feature_importance_per_band.csv
"""

from __future__ import annotations

import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Shared feature engineering
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import (
    load_data,
    engineer_features,
    split_data,
    ALL_FEATURE_COLS,
)

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Rules engine (same as ablation_study.py)
# ---------------------------------------------------------------------------
def regras_score(row) -> float:
    s = 0.0
    nome_qtd = row.get("NOME qtd frag iguais", 0) or 0
    if nome_qtd >= 0.95:
        s += 3
    elif nome_qtd >= 0.85:
        s += 2
    dt_iguais = row.get("DTNASC dt iguais", 0) or 0
    dt_ap = row.get("DTNASC dt ap 1digi", 0) or 0
    if dt_iguais == 1:
        s += 3
    elif dt_ap == 1:
        s += 1.5
    mae_qtd = row.get("NOMEMAE qtd frag iguais", 0) or 0
    if mae_qtd >= 0.7:
        s += 2
    elif mae_qtd >= 0.5:
        s += 1
    if (row.get("CODMUNRES local igual", 0) or 0) == 1:
        s += 1.5
    if (row.get("ENDERECO via igual", 0) or 0) == 1:
        s += 1
    nota = row.get("nota final", 0) or 0
    if nota >= 9:
        s += 2
    elif nota >= 8:
        s += 1
    return s


# ══════════════════════════════════════════════════════════════════════
# C1: Stratified 5-fold CV
# ══════════════════════════════════════════════════════════════════════
def run_kfold_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    print(f"\n{'=' * 70}")
    print(f"C1: STRATIFIED {n_splits}-FOLD CROSS-VALIDATION (58 features)")
    print(f"{'=' * 70}")

    from imblearn.over_sampling import SMOTE

    rules = X.apply(regras_score, axis=1).values
    nota = X["nota final"].values
    X_arr = X.values
    y_arr = y.values

    configs = {
        "RF+SMOTE ≥0.5": {
            "model": "rf_smote",
            "ml_th": 0.5,
            "rules_th": None,
            "mode": "ml",
        },
        "GB ≥0.5": {"model": "gb", "ml_th": 0.5, "rules_th": None, "mode": "ml"},
        "Rules ≥7": {"model": None, "ml_th": None, "rules_th": 7, "mode": "rules"},
        "Hybrid-AND RF+SMOTE≥0.5+Rules≥7": {
            "model": "rf_smote",
            "ml_th": 0.5,
            "rules_th": 7,
            "mode": "and",
        },
        "Hybrid-AND GB≥0.6+Rules≥6": {
            "model": "gb",
            "ml_th": 0.6,
            "rules_th": 6,
            "mode": "and",
        },
        "Hybrid-OR RF+SMOTE≥0.7+Rules≥8": {
            "model": "rf_smote",
            "ml_th": 0.7,
            "rules_th": 8,
            "mode": "or",
        },
        "Naive ≥8": {
            "model": None,
            "ml_th": None,
            "rules_th": None,
            "mode": "naive",
            "naive_th": 8,
        },
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, cfg in configs.items():
        fold_metrics = []
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_arr, y_arr)):
            X_train, X_test_fold = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test_fold = y_arr[train_idx], y_arr[test_idx]
            rules_test = rules[test_idx]
            nota_test = nota[test_idx]

            proba = None
            if cfg.get("model") == "rf_smote":
                sm = SMOTE(
                    sampling_strategy=0.3, k_neighbors=3, random_state=RANDOM_STATE
                )
                X_res, y_res = sm.fit_resample(X_train, y_train)
                model = RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
                model.fit(X_res, y_res)
                proba = model.predict_proba(X_test_fold)[:, 1]
            elif cfg.get("model") == "gb":
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=RANDOM_STATE,
                )
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_test_fold)[:, 1]

            mode = cfg["mode"]
            if mode == "naive":
                pred = (nota_test >= cfg["naive_th"]).astype(int)
            elif mode == "ml":
                pred = (proba >= cfg["ml_th"]).astype(int)
            elif mode == "rules":
                pred = (rules_test >= cfg["rules_th"]).astype(int)
            elif mode == "and":
                pred = (
                    (proba >= cfg["ml_th"]) & (rules_test >= cfg["rules_th"])
                ).astype(int)
            elif mode == "or":
                pred = (
                    (proba >= cfg["ml_th"]) | (rules_test >= cfg["rules_th"])
                ).astype(int)
            else:
                pred = np.zeros(len(y_test_fold), dtype=int)

            p = precision_score(y_test_fold, pred, zero_division=0)
            r = recall_score(y_test_fold, pred, zero_division=0)
            f = f1_score(y_test_fold, pred, zero_division=0)
            fold_metrics.append({"precision": p, "recall": r, "f1": f})

        fm = pd.DataFrame(fold_metrics)
        row = {
            "config": name,
            "prec_mean": fm["precision"].mean(),
            "prec_std": fm["precision"].std(),
            "rec_mean": fm["recall"].mean(),
            "rec_std": fm["recall"].std(),
            "f1_mean": fm["f1"].mean(),
            "f1_std": fm["f1"].std(),
        }
        results.append(row)
        print(
            f"  {name}: F1={row['f1_mean']:.4f}+-{row['f1_std']:.4f}  "
            f"P={row['prec_mean']:.4f}+-{row['prec_std']:.4f}  "
            f"R={row['rec_mean']:.4f}+-{row['rec_std']:.4f}"
        )

    rdf = pd.DataFrame(results)
    rdf.to_csv(DATA / "cv_5fold_results.csv", index=False)
    print(f"\n  -> Saved: data/cv_5fold_results.csv")
    return rdf


# ══════════════════════════════════════════════════════════════════════
# C2: Imbalance Sensitivity
# ══════════════════════════════════════════════════════════════════════
def run_imbalance_sensitivity(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    print(f"\n{'=' * 70}")
    print("C2: IMBALANCE STRATEGY SENSITIVITY (58 features)")
    print(f"{'=' * 70}")

    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    from imblearn.combine import SMOTETomek

    X_arr = X.values
    y_arr = y.values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    strategies = {
        "No balancing": None,
        "Class weight=balanced": "class_weight",
        "SMOTE 0.3": ("smote", 0.3),
        "SMOTE 0.5": ("smote", 0.5),
        "SMOTE 1.0": ("smote", 1.0),
        "BorderlineSMOTE 0.3": ("borderline", 0.3),
        "ADASYN 0.3": ("adasyn", 0.3),
        "SMOTETomek 0.3": ("smotetomek", 0.3),
        "Class weight + SMOTE 0.3": ("cw_smote", 0.3),
    }

    results = []
    for strat_name, strat in strategies.items():
        fold_metrics = []
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_arr, y_arr)):
            X_train, X_test_fold = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test_fold = y_arr[train_idx], y_arr[test_idx]

            cw = None
            X_fit, y_fit = X_train, y_train

            if strat is None:
                pass
            elif strat == "class_weight":
                cw = "balanced"
            else:
                kind, ratio = strat
                try:
                    if kind == "smote":
                        sampler = SMOTE(
                            sampling_strategy=ratio,
                            k_neighbors=3,
                            random_state=RANDOM_STATE,
                        )
                    elif kind == "borderline":
                        sampler = BorderlineSMOTE(
                            sampling_strategy=ratio,
                            k_neighbors=3,
                            random_state=RANDOM_STATE,
                        )
                    elif kind == "adasyn":
                        sampler = ADASYN(
                            sampling_strategy=ratio,
                            n_neighbors=3,
                            random_state=RANDOM_STATE,
                        )
                    elif kind == "smotetomek":
                        sampler = SMOTETomek(
                            smote=SMOTE(
                                sampling_strategy=ratio,
                                k_neighbors=3,
                                random_state=RANDOM_STATE,
                            ),
                            random_state=RANDOM_STATE,
                        )
                    elif kind == "cw_smote":
                        cw = "balanced"
                        sampler = SMOTE(
                            sampling_strategy=ratio,
                            k_neighbors=3,
                            random_state=RANDOM_STATE,
                        )
                    else:
                        continue
                    X_fit, y_fit = sampler.fit_resample(X_train, y_train)
                except Exception as e:
                    print(f"    [WARN] {strat_name} fold {fold_i}: {e}")
                    continue

            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                class_weight=cw,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            model.fit(X_fit, y_fit)
            proba = model.predict_proba(X_test_fold)[:, 1]
            pred = (proba >= 0.5).astype(int)

            p = precision_score(y_test_fold, pred, zero_division=0)
            r = recall_score(y_test_fold, pred, zero_division=0)
            f = f1_score(y_test_fold, pred, zero_division=0)
            fold_metrics.append({"precision": p, "recall": r, "f1": f})

        if not fold_metrics:
            continue
        fm = pd.DataFrame(fold_metrics)
        row = {
            "strategy": strat_name,
            "prec_mean": fm["precision"].mean(),
            "prec_std": fm["precision"].std(),
            "rec_mean": fm["recall"].mean(),
            "rec_std": fm["recall"].std(),
            "f1_mean": fm["f1"].mean(),
            "f1_std": fm["f1"].std(),
            "n_folds": len(fold_metrics),
        }
        results.append(row)
        print(
            f"  {strat_name}: F1={row['f1_mean']:.4f}+-{row['f1_std']:.4f}  "
            f"P={row['prec_mean']:.4f}+-{row['prec_std']:.4f}  "
            f"R={row['rec_mean']:.4f}+-{row['rec_std']:.4f}"
        )

    rdf = pd.DataFrame(results)
    rdf.to_csv(DATA / "imbalance_sensitivity.csv", index=False)
    print(f"\n  -> Saved: data/imbalance_sensitivity.csv")
    return rdf


# ══════════════════════════════════════════════════════════════════════
# C3: SHAP + Feature Importance per Band
# ══════════════════════════════════════════════════════════════════════
def run_feature_importance(
    X: pd.DataFrame, y: pd.Series, df: pd.DataFrame
) -> pd.DataFrame:
    print(f"\n{'=' * 70}")
    print("C3: SHAP & FEATURE IMPORTANCE (58 features)")
    print(f"{'=' * 70}")

    from imblearn.over_sampling import SMOTE

    feat_cols = list(X.columns)

    # Train RF+SMOTE on full data for global importance
    sm = SMOTE(sampling_strategy=0.3, k_neighbors=3, random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X.values, y.values)
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_res, y_res)

    # Global feature importance (Gini)
    imp = pd.DataFrame(
        {
            "feature": feat_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    print("\n  Top 15 features (Gini importance):")
    for _, r in imp.head(15).iterrows():
        print(f"    {r['feature']:35s} {r['importance']:.4f}")

    # Per-band importance
    nota = X["nota final"].values
    bands = [
        (0, 5, "baixo (0-5)"),
        (5, 6, "cinza-baixo (5-6)"),
        (6, 7, "cinza-medio (6-7)"),
        (7, 8, "cinza-alto (7-8)"),
        (8, 10, "alto (8-10)"),
        (10, 999, "muito alto (10+)"),
    ]

    band_results = []
    for lo, hi, label in bands:
        mask = (nota >= lo) & (nota < hi)
        n_total = mask.sum()
        n_pairs = int(y.values[mask].sum())
        if n_total < 50 or n_pairs < 2:
            print(f"\n  Band {label}: n={n_total}, pairs={n_pairs} -- skip")
            continue

        X_band = X.values[mask]
        y_band = y.values[mask]

        try:
            if n_pairs >= 5:
                sm_band = SMOTE(
                    sampling_strategy=0.3,
                    k_neighbors=min(3, n_pairs - 1),
                    random_state=RANDOM_STATE,
                )
                X_b, y_b = sm_band.fit_resample(X_band, y_band)
            else:
                X_b, y_b = X_band, y_band
            m = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            m.fit(X_b, y_b)
            band_imp = pd.DataFrame(
                {
                    "feature": feat_cols,
                    "importance": m.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            band_imp["band"] = label
            band_imp["n_total"] = n_total
            band_imp["n_pairs"] = n_pairs
            band_results.append(band_imp)
            print(f"\n  Band {label} (n={n_total}, pairs={n_pairs}):")
            for _, r in band_imp.head(5).iterrows():
                print(f"    {r['feature']:35s} {r['importance']:.4f}")
        except Exception as e:
            print(f"\n  Band {label}: ERROR -- {e}")

    if band_results:
        all_bands = pd.concat(band_results, ignore_index=True)
        all_bands.to_csv(DATA / "feature_importance_per_band.csv", index=False)
        print(f"\n  -> Saved: data/feature_importance_per_band.csv")

    # SHAP analysis
    try:
        import shap

        print("\n  Computing SHAP values (TreeExplainer, sample=2000)...")
        explainer = shap.TreeExplainer(model)
        sample_idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X), size=min(2000, len(X)), replace=False
        )
        X_sample = X.iloc[sample_idx]
        shap_values = explainer.shap_values(X_sample.values)

        if isinstance(shap_values, list):
            shap_vals = np.array(shap_values[1])
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            shap_vals = shap_values[:, :, 1]
        else:
            shap_vals = np.array(shap_values)

        shap_imp = pd.DataFrame(
            {
                "feature": feat_cols,
                "shap_mean_abs": np.abs(shap_vals).mean(axis=0),
            }
        ).sort_values("shap_mean_abs", ascending=False)
        shap_imp.to_csv(DATA / "shap_importance.csv", index=False)
        print("\n  Top 15 features (SHAP |mean|):")
        for _, r in shap_imp.head(15).iterrows():
            print(f"    {r['feature']:35s} {r['shap_mean_abs']:.4f}")
        print(f"\n  -> Saved: data/shap_importance.csv")

    except ImportError:
        print("\n  [INFO] shap not installed -- skipping SHAP analysis")
        print("         Install with: pip install shap")
        # Save Gini importance as shap_importance fallback
        shap_imp = imp.rename(columns={"importance": "shap_mean_abs"})
        shap_imp.to_csv(DATA / "shap_importance.csv", index=False)
        print("  -> Saved Gini importance as shap_importance.csv (fallback)")

    return imp


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading data with shared feature_engineering (58 features)...")
    df = load_data()
    X, y = engineer_features(df)
    print(f"  Dataset: {len(df)} records, {y.sum()} pairs ({y.mean() * 100:.2f}%)")
    print(f"  Features: {X.shape[1]} (58 uniform)")

    run_kfold_cv(X, y)
    run_imbalance_sensitivity(X, y)
    run_feature_importance(X, y, df)

    print(f"\n{'=' * 70}")
    print("ROBUSTNESS ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
