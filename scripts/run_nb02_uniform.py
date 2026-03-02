"""
NB02 - Estratégia Máximo Recall (Uniform 58 features)
Replicates notebooks/02_estrategia_maximo_recall.ipynb
using the shared feature engineering module.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import engineer_features, load_data, split_data

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


def apply_threshold(y_proba, threshold: float):
    return (y_proba >= threshold).astype(int)


def main():
    print("=" * 60)
    print("NB02 - Estratégia Máximo Recall (58 features uniformes)")
    print("=" * 60)

    df = load_data()
    X, y = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Train: {X_train.shape}, pos={y_train.sum()}")
    print(f"Test:  {X_test.shape}, pos={y_test.sum()}")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Balancing variants ---
    balancers = {
        "SMOTE": SMOTE(random_state=42, k_neighbors=3),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=42, k_neighbors=3),
        "ADASYN": ADASYN(random_state=42, n_neighbors=3),
        "SMOTETomek": SMOTETomek(random_state=42),
    }

    resampled = {}
    for name, bal in balancers.items():
        Xr, yr = bal.fit_resample(X_train, y_train)
        resampled[name] = (Xr, yr)
        print(f"  {name}: {Xr.shape[0]} samples, pos={yr.sum()}")

    # --- Models ---
    results = []

    # 1. RF class_weight extremo
    print("\n1. RF Class Weight Extremo (th=0.3)...")
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_leaf=1,
        class_weight={0: 1, 1: 500},
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = apply_threshold(proba, 0.3)
    results.append(evaluate("RF ClassWeight Extremo (th=0.3)", y_test, pred, proba))

    # 2. RF + SMOTE
    print("2. RF + SMOTE (th=0.3)...")
    Xr, yr = resampled["SMOTE"]
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(Xr, yr)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = apply_threshold(proba, 0.3)
    results.append(evaluate("RF + SMOTE (th=0.3)", y_test, pred, proba))

    # 3. RF + Borderline SMOTE
    print("3. RF + Borderline SMOTE (th=0.3)...")
    Xr, yr = resampled["BorderlineSMOTE"]
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(Xr, yr)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = apply_threshold(proba, 0.3)
    results.append(evaluate("RF + BorderlineSMOTE (th=0.3)", y_test, pred, proba))

    # 4. GB + SMOTE + class weight
    print("4. GB + SMOTE + Class Weight (th=0.3)...")
    Xr, yr = resampled["SMOTE"]
    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    clf.fit(Xr, yr)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = apply_threshold(proba, 0.3)
    results.append(evaluate("GB + SMOTE + Class Weight (th=0.3)", y_test, pred, proba))

    # 5. AdaBoost + SMOTE
    print("5. AdaBoost + SMOTE (th=0.3)...")
    Xr, yr = resampled["SMOTE"]
    clf = AdaBoostClassifier(n_estimators=100, random_state=42)
    clf.fit(Xr, yr)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = apply_threshold(proba, 0.3)
    results.append(evaluate("AdaBoost + SMOTE (th=0.3)", y_test, pred, proba))

    # 6. MLP + SMOTE
    print("6. MLP + SMOTE (th=0.25)...")
    Xr, yr = resampled["SMOTE"]
    scaler2 = StandardScaler()
    Xr_s = scaler2.fit_transform(Xr)
    Xte_s = scaler2.transform(X_test)
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    clf.fit(Xr_s, yr)
    proba = clf.predict_proba(Xte_s)[:, 1]
    pred = apply_threshold(proba, 0.25)
    results.append(evaluate("MLP + SMOTE (th=0.25)", y_test, pred, proba))

    # 7. Ensemble Soft Voting
    print("7. Ensemble Soft Voting (th=0.15)...")
    Xr, yr = resampled["SMOTE"]
    ensemble = VotingClassifier(
        estimators=[
            (
                "rf1",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    class_weight={0: 1, 1: 300},
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42,
                ),
            ),
            ("ada", AdaBoostClassifier(n_estimators=100, random_state=42)),
        ],
        voting="soft",
        weights=[2, 1, 1],
    )
    ensemble.fit(Xr, yr)
    proba = ensemble.predict_proba(X_test)[:, 1]
    pred = apply_threshold(proba, 0.15)
    results.append(evaluate("Ensemble Soft (th=0.15)", y_test, pred, proba))

    # --- Multi-threshold analysis for RF+SMOTE ---
    print("\n--- Análise multi-limiar RF+SMOTE ---")
    Xr, yr = resampled["SMOTE"]
    rf_smote = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        n_jobs=-1,
        random_state=42,
    )
    rf_smote.fit(Xr, yr)
    proba_rf = rf_smote.predict_proba(X_test)[:, 1]

    th_results = []
    for th in [0.2, 0.3, 0.4]:
        pred_th = apply_threshold(proba_rf, th)
        r = evaluate(f"RF+SMOTE (th={th})", y_test, pred_th, proba_rf)
        th_results.append(r)
        print(
            f"  th={th:.2f}: P={r['Precision']:.4f} R={r['Recall']:.4f} F1={r['F1-Score']:.4f}"
        )

    # --- Save ---
    df_results = pd.DataFrame(results).sort_values("Recall", ascending=False)
    df_results.to_csv(RESULTS_DIR / "nb02_comparacao_modelos.csv", index=False, sep=";")

    df_th = pd.DataFrame(th_results)
    df_th.to_csv(RESULTS_DIR / "nb02_threshold_analysis.csv", index=False, sep=";")

    print("\n" + "=" * 60)
    print("RESULTS (sorted by Recall):")
    print(df_results.to_string(index=False))
    print("=" * 60)

    best = df_results.iloc[0]
    summary = {
        "notebook": "02_estrategia_maximo_recall",
        "n_features": 58,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "pos_train": int(y_train.sum()),
        "pos_test": int(y_test.sum()),
        "best_model": best["Modelo"],
        "best_recall": float(best["Recall"]),
        "best_precision": float(best["Precision"]),
        "best_f1": float(best["F1-Score"]),
        "all_results": results,
        "threshold_analysis": th_results,
    }
    with open(RESULTS_DIR / "nb02_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {RESULTS_DIR / 'nb02_comparacao_modelos.csv'}")
    print(f"Saved: {RESULTS_DIR / 'nb02_results.json'}")


if __name__ == "__main__":
    main()
