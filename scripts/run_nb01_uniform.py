"""
NB01 - Análise Comparativa de Técnicas (Uniform 58 features)
Replicates notebooks/01_analise_comparativa_tecnicas.ipynb
using the shared feature engineering module.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# --- Add scripts/ to path ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import engineer_features, load_data, split_data

# --- Paths ---
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
    }


def main():
    print("=" * 60)
    print("NB01 - Análise Comparativa (58 features uniformes)")
    print("=" * 60)

    # Load and engineer
    df = load_data()
    X, y = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Train: {X_train.shape}, pos={y_train.sum()}")
    print(f"Test:  {X_test.shape}, pos={y_test.sum()}")

    # Scaling for SVM and MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- SMOTE for model 6 ---
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # --- Models ---
    models = {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            X_train,
            X_test,
            y_train,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            X_train,
            X_test,
            y_train,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            ),
            X_train,
            X_test,
            y_train,
        ),
        "SVM (RBF)": (
            SVC(
                kernel="rbf", class_weight="balanced", probability=True, random_state=42
            ),
            X_train_scaled,
            X_test_scaled,
            y_train,
        ),
        "MLP Neural Network": (
            MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                max_iter=500,
                random_state=42,
                early_stopping=True,
            ),
            X_train_scaled,
            X_test_scaled,
            y_train,
        ),
        "RF + SMOTE": (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            ),
            X_train_smote,
            X_test,
            y_train_smote,
        ),
    }

    results = []
    trained = {}

    for name, (clf, Xtr, Xte, ytr) in models.items():
        print(f"\nTraining: {name}...")
        clf.fit(Xtr, ytr)
        y_pred = clf.predict(Xte)
        y_proba = clf.predict_proba(Xte)[:, 1]
        r = evaluate(name, y_test, y_pred, y_proba)
        results.append(r)
        trained[name] = clf
        print(f"  P={r['Precision']:.4f}  R={r['Recall']:.4f}  F1={r['F1-Score']:.4f}")

    # Stacking
    print("\nTraining: Stacking Ensemble...")
    stacking = StackingClassifier(
        estimators=[
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=4,
                    random_state=42,
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ],
        final_estimator=LogisticRegression(class_weight="balanced"),
        cv=5,
        n_jobs=-1,
    )
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_test)
    y_proba = stacking.predict_proba(X_test)[:, 1]
    r = evaluate("Stacking Ensemble", y_test, y_pred, y_proba)
    results.append(r)
    trained["Stacking Ensemble"] = stacking
    print(f"  P={r['Precision']:.4f}  R={r['Recall']:.4f}  F1={r['F1-Score']:.4f}")

    # --- Threshold optimization (RF) ---
    print("\n--- Threshold Optimization (RF) ---")
    rf = trained["Random Forest"]
    rf_proba = rf.predict_proba(X_test)[:, 1]
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.01, 1.0, 0.01):
        preds = (rf_proba >= th).astype(int)
        f = f1_score(y_test, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_th = th
    print(f"Best F1={best_f1:.4f} at threshold={best_th:.2f}")

    # --- Save ---
    df_results = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
    df_results.to_csv(RESULTS_DIR / "nb01_comparacao_modelos.csv", index=False, sep=";")

    print("\n" + "=" * 60)
    print("RESULTS:")
    print(df_results.to_string(index=False))
    print("=" * 60)

    # JSON summary
    best = df_results.iloc[0]
    summary = {
        "notebook": "01_analise_comparativa_tecnicas",
        "n_features": 58,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "pos_train": int(y_train.sum()),
        "pos_test": int(y_test.sum()),
        "best_model": best["Modelo"],
        "best_f1": float(best["F1-Score"]),
        "best_precision": float(best["Precision"]),
        "best_recall": float(best["Recall"]),
        "best_threshold_rf": float(best_th),
        "best_f1_at_threshold": float(best_f1),
        "all_results": results,
    }
    with open(RESULTS_DIR / "nb01_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {RESULTS_DIR / 'nb01_comparacao_modelos.csv'}")
    print(f"Saved: {RESULTS_DIR / 'nb01_results.json'}")


if __name__ == "__main__":
    main()
