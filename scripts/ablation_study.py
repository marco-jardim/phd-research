"""
Ablation Study: Two-Stage Hybrid Classification for Probabilistic Linkage (CHDLP)
==================================================================================

Uses the shared feature_engineering module (58 uniform features).

Decision categories:
  1. Naive threshold (nota final ≥ 7, 8, 9)
  2. Rules-only (limiar 6, 7, 8, 9)
  3. ML-only RF+SMOTE (th 0.3, 0.5, 0.7)
  4. ML-only GB (th 0.3, 0.5, 0.7)
  5. Hybrid-AND: ML≥th AND rules≥limiar
  6. Hybrid-OR:  ML≥th OR  rules≥limiar
  7. Cascade ML→Rules
  8. Cascade Rules→ML
  9. Consensus ML + Consensus+Rules

Outputs:
  - data/ablation_results.csv
  - data/pareto_grid.csv
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Shared feature engineering
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import load_data, engineer_features, split_data

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

try:
    from imblearn.over_sampling import SMOTE

    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    log.warning("imblearn not installed — SMOTE disabled")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Rules engine (deterministic domain rules, max = 12.5)
# ---------------------------------------------------------------------------
def regras_alta_confianca(row: pd.Series) -> float:
    score = 0.0
    nome_qtd = row.get("NOME qtd frag iguais", 0) or 0
    if nome_qtd >= 0.95:
        score += 3
    elif nome_qtd >= 0.85:
        score += 2

    dt_iguais = row.get("DTNASC dt iguais", 0) or 0
    dt_ap = row.get("DTNASC dt ap 1digi", 0) or 0
    if dt_iguais == 1:
        score += 3
    elif dt_ap == 1:
        score += 1.5

    mae_qtd = row.get("NOMEMAE qtd frag iguais", 0) or 0
    if mae_qtd >= 0.7:
        score += 2
    elif mae_qtd >= 0.5:
        score += 1

    if (row.get("CODMUNRES local igual", 0) or 0) == 1:
        score += 1.5

    if (row.get("ENDERECO via igual", 0) or 0) == 1:
        score += 1

    nota = row.get("nota final", 0) or 0
    if nota >= 9:
        score += 2
    elif nota >= 8:
        score += 1

    return score


def compute_rules_scores(df: pd.DataFrame) -> np.ndarray:
    return df.apply(regras_alta_confianca, axis=1).values


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def eval_binary(y_true, y_pred, y_proba=None) -> dict:
    result = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
        "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
        "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
        "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        result["auc_roc"] = roc_auc_score(y_true, y_proba)
    return result


# ---------------------------------------------------------------------------
# Main ablation
# ---------------------------------------------------------------------------
def run_ablation():
    log.info("Loading data with shared feature_engineering (58 features)...")
    df = load_data()
    X, y = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    log.info(f"Dataset: {len(df)} records, {y.sum()} pairs ({y.mean() * 100:.2f}%)")
    log.info(f"Features: {X.shape[1]} (58 uniform)")
    log.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # We need the original df columns for rules — build a test df with score cols
    # X_test already has the 58 feature columns including the base score cols
    df_test_features = X_test.copy()

    # Rules scores on test set (uses the base score columns present in X_test)
    rules_test = compute_rules_scores(df_test_features)

    # nota final from test features
    nota_test = X_test["nota final"].values

    # ── Train ML models ──
    models = {
        "RF": (
            RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
        "RF+SMOTE": (
            RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            True,
        ),
        "GB": (
            GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
    }

    smote = (
        SMOTE(sampling_strategy=0.3, k_neighbors=3, random_state=RANDOM_STATE)
        if HAS_SMOTE
        else None
    )

    trained = {}
    probas = {}
    for name, (model, use_smote) in models.items():
        log.info(f"Training {name}...")
        if use_smote and smote is not None:
            X_res, y_res = smote.fit_resample(X_train, y_train)
            model.fit(X_res, y_res)
        else:
            model.fit(X_train, y_train)
        trained[name] = model
        probas[name] = model.predict_proba(X_test)[:, 1]

    # ===================================================================
    # Configurations
    # ===================================================================
    results = []

    # --- 1. Naive threshold ---
    for th in [7, 8, 9]:
        preds = (nota_test >= th).astype(int)
        m = eval_binary(y_test, preds)
        results.append(
            {"config": f"Naive score≥{th}", "category": "naive-threshold", **m}
        )

    # --- 2. Rules-only ---
    for limiar in [6, 7, 8, 9]:
        preds = (rules_test >= limiar).astype(int)
        m = eval_binary(y_test, preds)
        results.append(
            {"config": f"Rules-only (≥{limiar})", "category": "rules-only", **m}
        )

    # --- 3. ML-only (RF, GB) ---
    ml_thresholds = [0.3, 0.5, 0.7]
    for model_name in trained:
        for th in ml_thresholds:
            preds = (probas[model_name] >= th).astype(int)
            m = eval_binary(y_test, preds, probas[model_name])
            results.append(
                {
                    "config": f"ML-only {model_name} (≥{th})",
                    "category": "ml-only",
                    **m,
                }
            )

    # --- 4. Hybrid AND ---
    hybrid_ml_ths = [0.5, 0.6, 0.7]
    hybrid_rule_ths = [5, 6, 7, 8]
    for ml_name in ["RF+SMOTE", "GB"]:
        if ml_name not in probas:
            continue
        for ml_th in hybrid_ml_ths:
            for rule_th in hybrid_rule_ths:
                preds = ((probas[ml_name] >= ml_th) & (rules_test >= rule_th)).astype(
                    int
                )
                m = eval_binary(y_test, preds, probas[ml_name])
                results.append(
                    {
                        "config": f"Hybrid-AND {ml_name} ≥{ml_th} + Rules≥{rule_th}",
                        "category": "hybrid-and",
                        **m,
                    }
                )

    # --- 5. Hybrid OR ---
    for ml_name in ["RF+SMOTE", "GB"]:
        if ml_name not in probas:
            continue
        for ml_th in [0.7, 0.9]:
            for rule_th in [8, 9]:
                preds = ((probas[ml_name] >= ml_th) | (rules_test >= rule_th)).astype(
                    int
                )
                m = eval_binary(y_test, preds, probas[ml_name])
                results.append(
                    {
                        "config": f"Hybrid-OR {ml_name} ≥{ml_th} + Rules≥{rule_th}",
                        "category": "hybrid-or",
                        **m,
                    }
                )

    # --- 6. Cascade ML→Rules ---
    for ml_name in ["RF+SMOTE", "GB"]:
        if ml_name not in probas:
            continue
        for ml_th in [0.3, 0.5]:
            for rule_th in [7, 8]:
                ml_pass = probas[ml_name] >= ml_th
                preds = np.zeros(len(y_test), dtype=int)
                preds[ml_pass & (rules_test >= rule_th)] = 1
                m = eval_binary(y_test, preds, probas[ml_name])
                results.append(
                    {
                        "config": f"Cascade ML→Rules {ml_name} ≥{ml_th} → Rules≥{rule_th}",
                        "category": "cascade-ml-rules",
                        **m,
                    }
                )

    # --- 7. Cascade Rules→ML ---
    for ml_name in ["RF+SMOTE", "GB"]:
        if ml_name not in probas:
            continue
        for rule_th in [5, 6]:
            for ml_th in [0.5, 0.7]:
                rule_pass = rules_test >= rule_th
                preds = np.zeros(len(y_test), dtype=int)
                preds[rule_pass & (probas[ml_name] >= ml_th)] = 1
                m = eval_binary(y_test, preds, probas[ml_name])
                results.append(
                    {
                        "config": f"Cascade Rules→ML Rules≥{rule_th} → {ml_name} ≥{ml_th}",
                        "category": "cascade-rules-ml",
                        **m,
                    }
                )

    # --- 8. Consensus ML ---
    ml_names_consensus = [n for n in ["RF", "RF+SMOTE", "GB"] if n in probas]
    for consensus_th in [0.5, 0.7]:
        votes = np.zeros(len(y_test))
        for mn in ml_names_consensus:
            votes += (probas[mn] >= consensus_th).astype(float)
        majority = (votes >= len(ml_names_consensus) / 2).astype(int)
        m = eval_binary(y_test, majority)
        results.append(
            {
                "config": f"Consensus ML-majority (th={consensus_th}, {len(ml_names_consensus)} models)",
                "category": "consensus-ml",
                **m,
            }
        )

        # --- 9. Consensus + rules ---
        for rule_th in [6, 7]:
            preds = (majority & (rules_test >= rule_th)).astype(int)
            m = eval_binary(y_test, preds)
            results.append(
                {
                    "config": f"Consensus+Rules majority(th={consensus_th}) AND Rules≥{rule_th}",
                    "category": "consensus-hybrid",
                    **m,
                }
            )

    # ===================================================================
    # Save results
    # ===================================================================
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("f1", ascending=False).reset_index(drop=True)
    results_df.to_csv(OUT_DIR / "ablation_results.csv", index=False)
    log.info(f"Saved {len(results_df)} configs to {OUT_DIR / 'ablation_results.csv'}")

    # Print summary
    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS (58 uniform features)")
    print("=" * 100)
    print(
        f"\n{'Config':<65} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}"
    )
    print("-" * 100)
    for _, row in results_df.head(30).iterrows():
        print(
            f"{row['config']:<65} {row['precision']:>6.3f} {row['recall']:>6.3f} "
            f"{row['f1']:>6.3f} {row['tp']:>5.0f} {row['fp']:>5.0f} {row['fn']:>5.0f}"
        )

    # ===================================================================
    # Pareto grid search
    # ===================================================================
    log.info("Running Pareto grid search...")
    pareto_results = []
    for ml_name in ["RF+SMOTE", "GB"]:
        if ml_name not in probas:
            continue
        for ml_th in np.arange(0.1, 1.0, 0.05):
            for rule_th in np.arange(0, 13, 0.5):
                preds = ((probas[ml_name] >= ml_th) & (rules_test >= rule_th)).astype(
                    int
                )
                if preds.sum() == 0:
                    continue
                m = eval_binary(y_test, preds)
                pareto_results.append(
                    {
                        "ml_model": ml_name,
                        "ml_threshold": round(ml_th, 2),
                        "rules_threshold": round(rule_th, 1),
                        **m,
                    }
                )

    pareto_df = pd.DataFrame(pareto_results)
    pareto_df.to_csv(OUT_DIR / "pareto_grid.csv", index=False)
    log.info(f"Saved {len(pareto_df)} grid points to {OUT_DIR / 'pareto_grid.csv'}")

    # ===================================================================
    # Category summary
    # ===================================================================
    print("\n" + "=" * 100)
    print("BEST F1 PER CATEGORY")
    print("=" * 100)
    for cat in [
        "naive-threshold",
        "rules-only",
        "ml-only",
        "hybrid-and",
        "hybrid-or",
        "cascade-ml-rules",
        "cascade-rules-ml",
        "consensus-ml",
        "consensus-hybrid",
    ]:
        cat_df = results_df[results_df["category"] == cat]
        if not cat_df.empty:
            best = cat_df.iloc[0]
            print(f"  {cat:<25} -> {best['config']:<55} F1={best['f1']:.4f}")

    print(f"\nTotal configurations tested: {len(results_df)}")
    log.info("Ablation study complete.")
    return results_df, pareto_df


if __name__ == "__main__":
    run_ablation()
