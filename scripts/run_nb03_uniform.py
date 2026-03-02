"""
NB03 - Estratégia Máxima Precisão (Uniform 58 features)
Replicates notebooks/03_estrategia_maxima_precisao.ipynb
using the shared feature engineering module.
All 58 features used (no top-25 selection).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Optional imports for XGBoost/LightGBM
try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import (
    ALL_FEATURE_COLS,
    engineer_features,
    load_data,
    split_data,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data"
RESULTS_DIR.mkdir(exist_ok=True)


def evaluate(name: str, y_true, y_pred, y_proba) -> dict:
    return {
        "Modelo": name,
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "AUC-ROC": round(roc_auc_score(y_true, y_proba), 4),
        "AUC-PR": round(average_precision_score(y_true, y_proba), 4),
        "N_Positivos": int(y_pred.sum()),
    }


def apply_threshold(y_proba, threshold: float):
    return (y_proba >= threshold).astype(int)


def regras_alta_confianca(row) -> float:
    """Business rules score (0-12.5 max)."""
    score = 0.0
    # Name
    if row.get("NOME qtd frag iguais", 0) >= 0.95:
        score += 3.0
    elif row.get("NOME qtd frag iguais", 0) >= 0.85:
        score += 2.0
    # Birth date
    if row.get("DTNASC dt iguais", 0) == 1:
        score += 3.0
    elif row.get("DTNASC dt ap 1digi", 0) == 1:
        score += 1.5
    # Mother name
    if row.get("NOMEMAE qtd frag iguais", 0) >= 0.7:
        score += 2.0
    elif row.get("NOMEMAE qtd frag iguais", 0) >= 0.5:
        score += 1.0
    # Municipality
    if row.get("CODMUNRES local igual", 0) == 1:
        score += 1.5
    # Address
    if row.get("ENDERECO via igual", 0) == 1:
        score += 1.0
    # nota final
    nf = row.get("nota final", 0)
    if nf >= 9:
        score += 2.0
    elif nf >= 8:
        score += 1.0
    return score


def main():
    print("=" * 60)
    print("NB03 - Estratégia Máxima Precisão (58 features uniformes)")
    print("=" * 60)

    df = load_data()
    X, y = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Train: {X_train.shape}, pos={y_train.sum()}")
    print(f"Test:  {X_test.shape}, pos={y_test.sum()}")

    # SMOTE (moderate)
    smote = SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_train_smote.shape[0]} samples, pos={y_train_smote.sum()}")

    results = []

    # --- 1. XGBoost ---
    if HAS_XGB:
        print("\n1. XGBoost (precision-tuned)...")
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            reg_alpha=1.0,
            reg_lambda=2.0,
            scale_pos_weight=10,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        clf.fit(X_train_smote, y_train_smote)
        proba = clf.predict_proba(X_test)[:, 1]
        for th in [0.6, 0.7, 0.8, 0.9]:
            pred = apply_threshold(proba, th)
            results.append(evaluate(f"XGBoost (th={th})", y_test, pred, proba))
    else:
        print("SKIP: XGBoost not installed")

    # --- 2. LightGBM ---
    if HAS_LGB:
        print("2. LightGBM (conservative)...")
        clf = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=50,
            reg_alpha=0.5,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        clf.fit(X_train_smote, y_train_smote)
        proba = clf.predict_proba(X_test)[:, 1]
        for th in [0.6, 0.7, 0.8, 0.9]:
            pred = apply_threshold(proba, th)
            results.append(evaluate(f"LightGBM (th={th})", y_test, pred, proba))
    else:
        print("SKIP: LightGBM not installed")

    # --- 3. RF + Isotonic Calibration ---
    print("3. RF + Isotonic Calibration...")
    rf_base = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf = CalibratedClassifierCV(rf_base, method="isotonic", cv=5)
    clf.fit(X_train_smote, y_train_smote)
    proba = clf.predict_proba(X_test)[:, 1]
    for th in [0.7, 0.8, 0.9]:
        pred = apply_threshold(proba, th)
        results.append(evaluate(f"RF Calibrado (th={th})", y_test, pred, proba))

    # --- 4. Stacking ---
    print("4. Stacking Classifier...")
    estimators = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "gb",
            GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
            ),
        ),
    ]
    if HAS_XGB:
        estimators.insert(
            0,
            (
                "xgb",
                xgb.XGBClassifier(
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric="logloss",
                ),
            ),
        )
    if HAS_LGB:
        estimators.insert(
            1,
            (
                "lgbm",
                lgb.LGBMClassifier(
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                    verbose=-1,
                ),
            ),
        )

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=1000, random_state=42
        ),
        cv=5,
        passthrough=True,
        n_jobs=-1,
    )
    stacking.fit(X_train_smote, y_train_smote)
    proba_stacking = stacking.predict_proba(X_test)[:, 1]
    for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = apply_threshold(proba_stacking, th)
        results.append(evaluate(f"Stacking (th={th})", y_test, pred, proba_stacking))

    # --- 5. Consensus (unanimity) ---
    print("5. Consensus (unanimity)...")
    consensus_models = []
    if HAS_XGB:
        m = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        m.fit(X_train_smote, y_train_smote)
        consensus_models.append(m)
    if HAS_LGB:
        m = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
            verbose=-1,
        )
        m.fit(X_train_smote, y_train_smote)
        consensus_models.append(m)
    rf_cons = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_cons.fit(X_train_smote, y_train_smote)
    consensus_models.append(rf_cons)
    gb_cons = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        random_state=42,
    )
    gb_cons.fit(X_train_smote, y_train_smote)
    consensus_models.append(gb_cons)

    all_probas = np.column_stack(
        [m.predict_proba(X_test)[:, 1] for m in consensus_models]
    )
    all_preds = (all_probas >= 0.7).astype(int)
    unanimity = all_preds.prod(axis=1)
    mean_proba = all_probas.mean(axis=1)
    results.append(evaluate("Consensus (unanimidade)", y_test, unanimity, mean_proba))

    # --- 6. Business Rules ---
    print("6. Business Rules...")
    # Need original score columns for rules
    df_test_scores = df.iloc[X_test.index][
        [
            "NOME qtd frag iguais",
            "DTNASC dt iguais",
            "DTNASC dt ap 1digi",
            "NOMEMAE qtd frag iguais",
            "CODMUNRES local igual",
            "ENDERECO via igual",
            "nota final",
        ]
    ].fillna(0)
    rule_scores = df_test_scores.apply(regras_alta_confianca, axis=1)

    for th in [6, 7, 8, 9, 10]:
        pred = (rule_scores >= th).astype(int)
        # Use rule scores normalized as "proba" for AUC
        norm_scores = rule_scores / 12.5
        r = evaluate(f"Regras (score≥{th})", y_test, pred, norm_scores)
        results.append(r)

    # --- 7. Hybrid ML + Rules ---
    print("7. Hybrid ML + Rules...")
    for ml_th in [0.5, 0.6, 0.7]:
        for rule_th in [5, 6, 7]:
            pred = ((proba_stacking >= ml_th) & (rule_scores >= rule_th)).astype(int)
            if pred.sum() > 0:
                hybrid_proba = (proba_stacking + rule_scores / 12.5) / 2
                r = evaluate(
                    f"Hybrid AND (ML≥{ml_th}, Rules≥{rule_th})",
                    y_test,
                    pred,
                    hybrid_proba,
                )
                results.append(r)

    # Hybrid OR
    for ml_th in [0.5, 0.6, 0.7]:
        for rule_th in [6, 7, 8]:
            pred = ((proba_stacking >= ml_th) | (rule_scores >= rule_th)).astype(int)
            if pred.sum() > 0:
                hybrid_proba = np.maximum(proba_stacking, rule_scores / 12.5)
                r = evaluate(
                    f"Hybrid OR (ML≥{ml_th}, Rules≥{rule_th})",
                    y_test,
                    pred,
                    hybrid_proba,
                )
                results.append(r)

    # --- Save ---
    df_results = pd.DataFrame(results).sort_values("Precision", ascending=False)
    df_results.to_csv(RESULTS_DIR / "nb03_comparacao_modelos.csv", index=False, sep=";")

    print("\n" + "=" * 60)
    print("RESULTS (sorted by Precision, top 20):")
    print(df_results.head(20).to_string(index=False))
    print("=" * 60)

    best_prec = df_results.iloc[0]
    best_f1_row = df_results.sort_values("F1-Score", ascending=False).iloc[0]

    summary = {
        "notebook": "03_estrategia_maxima_precisao",
        "n_features": 58,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "pos_train": int(y_train.sum()),
        "pos_test": int(y_test.sum()),
        "best_by_precision": {
            "model": best_prec["Modelo"],
            "precision": float(best_prec["Precision"]),
            "recall": float(best_prec["Recall"]),
            "f1": float(best_prec["F1-Score"]),
        },
        "best_by_f1": {
            "model": best_f1_row["Modelo"],
            "precision": float(best_f1_row["Precision"]),
            "recall": float(best_f1_row["Recall"]),
            "f1": float(best_f1_row["F1-Score"]),
        },
        "all_results": results,
    }
    with open(RESULTS_DIR / "nb03_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {RESULTS_DIR / 'nb03_comparacao_modelos.csv'}")
    print(f"Saved: {RESULTS_DIR / 'nb03_results.json'}")


if __name__ == "__main__":
    main()
