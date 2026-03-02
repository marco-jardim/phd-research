"""QW-1 Unified Modal Pipeline - Complete XGBoost Integration

Executes the full 6-stage pipeline on Modal with GPU L4:
1. Train 4 models (XGBoost, TabNet, RF+SMOTE, GB) on holdout set
2. Export per-record probabilities (4 CSVs)
3. Calculate SHAP + feature importance (RF and XGBoost)
4. Update 6 intermediate CSVs
5. Generate 9 PGFs + 2 LaTeX tables
6. Package everything in Modal volume qw1-data under pipeline_output/

Usage:
    modal run scripts/qw1_modal_unified_pipeline.py

Estimated runtime: 15-20 minutes on L4 GPU (~$1.20 cost)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import modal

# ═══════════════════════════════════════════════════════════════════════
# Modal Configuration
# ═══════════════════════════════════════════════════════════════════════

app = modal.App("qw1-unified-pipeline")

# Modal volume for persistent storage
volume = modal.Volume.from_name("qw1-data", create_if_missing=True)

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.2",
        "pytorch-tabnet==4.1.0",
        "xgboost>=2.0",
        "scikit-learn>=1.3",
        "imbalanced-learn>=0.11",
        "pandas>=2.0,<3.0",
        "numpy>=1.26,<2.0",
        "shap>=0.43",
        "matplotlib>=3.8",
        "seaborn>=0.12",
        "joblib>=1.3",
    )
    .apt_install(
        "texlive-latex-base",
        "texlive-latex-extra",
        "texlive-fonts-recommended",
        "cm-super",
    )
    .add_local_dir("scripts/", remote_path="/root/scripts/")
    .add_local_dir("gzcmd/", remote_path="/root/gzcmd/")
    .add_local_file(
        "data/COMPARADORSEMIDENT.csv", remote_path="/root/data/COMPARADORSEMIDENT.csv"
    )
    .add_local_file(
        "data/qw1/cv_summary.csv", remote_path="/root/data/qw1/cv_summary.csv"
    )
    .add_local_file(
        "data/cv_5fold_results.csv", remote_path="/root/data/cv_5fold_results.csv"
    )
    .add_local_file(
        "data/ablation_results.csv", remote_path="/root/data/ablation_results.csv"
    )
    .add_local_file("data/pareto_grid.csv", remote_path="/root/data/pareto_grid.csv")
    .add_local_file(
        "data/epidemiological_impact.csv",
        remote_path="/root/data/epidemiological_impact.csv",
    )
    .add_local_file(
        "data/shap_importance.csv", remote_path="/root/data/shap_importance.csv"
    )
    .add_local_file(
        "data/feature_importance_per_band.csv",
        remote_path="/root/data/feature_importance_per_band.csv",
    )
)

# ═══════════════════════════════════════════════════════════════════════
# Helper Functions (executed inside Modal container)
# ═══════════════════════════════════════════════════════════════════════


def setup_logging():
    """Configure logging for pipeline execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_and_prepare_data(logger):
    """Load dataset and engineer 58 features using canonical module."""
    import sys

    sys.path.insert(0, "/root")

    from scripts.feature_engineering import engineer_features, load_data, split_data

    logger.info("Loading COMPARADORSEMIDENT.csv...")
    df = load_data("/root/data/COMPARADORSEMIDENT.csv")

    logger.info("Engineering 58 features...")
    X, y = engineer_features(df)

    logger.info("Splitting 70/30 stratified (seed=42)...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=42)

    logger.info(
        "Split complete: train=%d (pos=%d), test=%d (pos=%d)",
        len(X_train),
        y_train.sum(),
        len(X_test),
        y_test.sum(),
    )

    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, y_train, logger):
    """Train XGBoost with QW-1 hyperparameters."""
    import xgboost as xgb

    logger.info("Training XGBoost (tree_method=hist, GPU accelerated)...")

    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        random_state=42,
    )

    model.fit(X_train, y_train)
    logger.info("XGBoost training complete")
    return model


def train_tabnet(X_train, y_train, logger):
    """Train TabNet model with early stopping via validation split."""
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    from sklearn.model_selection import train_test_split
    import numpy as np

    logger.info("Training TabNet (max_epochs=100, patience=15)...")

    # TabNet requires numpy arrays with 1D target
    X_arr = X_train.values if hasattr(X_train, "values") else np.array(X_train)
    y_arr = (
        y_train.to_numpy().ravel()
        if hasattr(y_train, "to_numpy")
        else np.array(y_train).ravel()
    )

    # Split 80/20 for early stopping eval_set (stratified)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
    )
    logger.info("TabNet train/val split: train=%d, val=%d", len(X_tr), len(X_val))

    model = TabNetClassifier(
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": 2e-2},
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=42,
        verbose=1,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=100,
        patience=15,
        batch_size=1024,
        virtual_batch_size=128,
    )

    logger.info(
        "TabNet training complete (best epoch: %s)", getattr(model, "best_epoch", "N/A")
    )
    return model


def train_rf_smote(X_train, y_train, logger):
    """Train Random Forest with SMOTE."""
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import RandomForestClassifier

    logger.info("Training RF+SMOTE (150 trees, max_depth=12)...")

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    logger.info(
        "SMOTE applied: %d → %d samples (pos=%d)",
        len(X_train),
        len(X_resampled),
        y_resampled.sum(),
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_resampled, y_resampled)
    logger.info("RF+SMOTE training complete")
    return model


def train_gradient_boosting(X_train, y_train, logger):
    """Train Gradient Boosting."""
    from sklearn.ensemble import GradientBoostingClassifier

    logger.info("Training GradientBoosting (100 trees, max_depth=5)...")

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)
    logger.info("GradientBoosting training complete")
    return model


def export_probabilities(models, model_names, X_test, y_test, output_dir, logger):
    """Export per-record probabilities for all models."""
    import numpy as np
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, model in zip(model_names, models):
        logger.info(f"Exporting probabilities for {name}...")

        if name == "tabnet":
            # TabNet requires numpy arrays
            probas = model.predict_proba(X_test.values)[:, 1]
        else:
            probas = model.predict_proba(X_test)[:, 1]

        df = pd.DataFrame(
            {"row_index": range(len(X_test)), "y_true": y_test.values, "proba": probas}
        )

        output_path = output_dir / f"{name}_holdout_probas.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {output_path} ({len(df)} rows)")


def regras_alta_confianca(row):
    """Compute deterministic rule-based confidence score (max 12.5)."""

    def v(key):
        value = row.get(key, 0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    score = 0.0

    # Name agreement
    if v("NOME qtd frag iguais") >= 2:
        score += 3
    if v("NOME compativel bool") > 0:
        score += 1

    # Birth date agreement
    if v("DTNASC dt iguais") == 1:
        score += 3
    elif v("DTNASC dt ap 1digi") == 1:
        score += 2

    # Mother name agreement
    if v("NOMEMAE qtd frag iguais") >= 2:
        score += 2
    elif v("NOMEMAE compativel bool") == 1:
        score += 1

    # Location/address agreement
    if v("CODMUNRES local igual") == 1:
        score += 1
    if v("ENDERECO via igual") == 1:
        score += 0.5

    # Legacy score bucket
    nota_final = v("nota final")
    if nota_final >= 9:
        score += 2
    elif nota_final >= 8:
        score += 1

    return score


def compute_rules_scores(df):
    """Compute rule-based score for each row of feature dataframe."""
    return df.apply(regras_alta_confianca, axis=1).to_numpy(dtype=float)


def evaluate_binary_predictions(y_true, y_pred, y_score=None):
    """Return precision/recall/F1/confusion metrics (+ optional AUC)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    tp = int(((y_pred_arr == 1) & (y_true_arr == 1)).sum())
    fp = int(((y_pred_arr == 1) & (y_true_arr == 0)).sum())
    fn = int(((y_pred_arr == 0) & (y_true_arr == 1)).sum())
    tn = int(((y_pred_arr == 0) & (y_true_arr == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    auc_roc = float("nan")
    if y_score is not None:
        try:
            auc_roc = float(roc_auc_score(y_true_arr, y_score))
        except ValueError:
            auc_roc = float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "auc_roc": auc_roc,
    }


def build_ablation_scored_holdout(volume_path, X_test, y_test, logger):
    """Create per-record artifact with XGBoost proba + rule score."""
    import numpy as np
    import pandas as pd

    probas_dir = volume_path / "probas"
    output_path = probas_dir / "xgboost_holdout_scored.csv"

    y_true = (
        y_test.to_numpy().ravel() if hasattr(y_test, "to_numpy") else np.asarray(y_test)
    )
    rules_scores = compute_rules_scores(X_test)

    df_scored = pd.DataFrame(
        {
            "row_index": np.arange(len(X_test)),
            "y_true": y_true,
            "rules_score": rules_scores,
        }
    )

    if "nota final" in X_test.columns:
        df_scored["nota_final"] = X_test["nota final"].to_numpy()
    else:
        df_scored["nota_final"] = np.nan

    model_files = {
        "xgboost": "xgboost_holdout_probas.csv",
        "rf_smote": "rf_smote_holdout_probas.csv",
        "gb": "gb_holdout_probas.csv",
        "tabnet": "tabnet_holdout_probas.csv",
    }

    for model_key, file_name in model_files.items():
        path = probas_dir / file_name

        if not path.exists():
            if model_key == "xgboost":
                raise FileNotFoundError(f"Required file not found: {path}")
            logger.warning(f"Optional file missing for ablation artifact: {path}")
            continue

        df_model = pd.read_csv(path)
        if len(df_model) != len(df_scored):
            raise ValueError(
                f"Length mismatch in {file_name}: expected {len(df_scored)}, got {len(df_model)}"
            )

        if "proba" not in df_model.columns:
            raise ValueError(f"Column 'proba' not found in {file_name}")

        df_scored[f"proba_{model_key}"] = df_model["proba"].to_numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_csv(output_path, index=False)
    logger.info(f"Saved {output_path} ({len(df_scored)} rows)")
    return output_path


def calculate_shap_values(xgb_model, rf_model, X_train, X_test, output_dir, logger):
    """Calculate SHAP values for XGBoost and RF using TreeExplainer."""
    import numpy as np
    import pandas as pd
    import shap

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- XGBoost SHAP (native pred_contribs) ---
    # Use XGBoost's built-in TreeSHAP via predict(pred_contribs=True).
    # This bypasses shap library's XGBTreeModelLoader which is incompatible
    # with XGBoost 2.0's array-format base_score ('[val]' instead of 'val').
    # Same exact TreeSHAP algorithm, zero compatibility issues.
    import xgboost as xgb

    logger.info("Computing SHAP values for XGBoost (native pred_contribs)...")
    booster = xgb_model.get_booster()
    feature_names = X_test.columns.tolist()
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    # pred_contribs returns (n_samples, n_features + 1); last column is bias
    shap_contribs = booster.predict(dtest, pred_contribs=True)
    shap_values_xgb = shap_contribs[:, :-1]  # drop bias column

    np.save(output_dir / "xgboost_shap_values.npy", shap_values_xgb)

    shap_importance_xgb = pd.DataFrame(
        {
            "feature": X_test.columns,
            "shap_mean_abs": np.abs(shap_values_xgb).mean(axis=0),
        }
    ).sort_values("shap_mean_abs", ascending=False)

    shap_importance_xgb.to_csv(output_dir / "xgboost_shap_importance.csv", index=False)
    logger.info(
        f"XGBoost SHAP complete: {shap_values_xgb.shape} matrix, top feature: {shap_importance_xgb.iloc[0]['feature']}"
    )

    # --- Random Forest SHAP ---
    logger.info("Computing SHAP values for RF...")
    explainer_rf = shap.TreeExplainer(rf_model)
    shap_values_rf = explainer_rf.shap_values(X_test)

    # Handle binary classification output (2D array per class → positive class)
    if isinstance(shap_values_rf, list):
        shap_values_rf = shap_values_rf[1]  # Positive class
    elif shap_values_rf.ndim == 3:
        shap_values_rf = shap_values_rf[:, :, 1]

    np.save(output_dir / "rf_shap_values.npy", shap_values_rf)

    shap_importance_rf = pd.DataFrame(
        {
            "feature": X_test.columns,
            "shap_mean_abs": np.abs(shap_values_rf).mean(axis=0),
        }
    ).sort_values("shap_mean_abs", ascending=False)

    shap_importance_rf.to_csv(output_dir / "rf_shap_importance.csv", index=False)
    logger.info(
        f"RF SHAP complete: {shap_values_rf.shape} matrix, top feature: {shap_importance_rf.iloc[0]['feature']}"
    )


def calculate_feature_importance(xgb_model, X_train, output_dir, logger):
    """Calculate native feature importance for XGBoost (gain-based)."""
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    # Global feature importance (gain-based)
    logger.info("Computing XGBoost feature importance (gain-based)...")
    importance_dict = xgb_model.get_booster().get_score(importance_type="gain")

    # Map feature names (XGBoost uses actual column names when trained with DataFrame)
    feature_names = X_train.columns.tolist()
    importance_data = []

    for feature in feature_names:
        gain = importance_dict.get(feature, 0.0)
        importance_data.append({"feature": feature, "importance_gain": gain})

    df_importance = pd.DataFrame(importance_data).sort_values(
        "importance_gain", ascending=False
    )

    output_path = output_dir / "xgboost_feature_importance_global.csv"
    df_importance.to_csv(output_path, index=False)
    logger.info(
        f"Global feature importance saved: {output_path}, top={df_importance.iloc[0]['feature']}"
    )


def update_cv_results(volume_path, logger):
    """Update cv_5fold_results.csv with XGBoost metrics."""
    import pandas as pd

    logger.info("Updating cv_5fold_results.csv...")

    # Load existing CV results and QW-1 summary
    cv_path = volume_path / "csvs" / "cv_5fold_results.csv"

    df_cv = pd.read_csv("/root/data/cv_5fold_results.csv")
    df_qw1 = pd.read_csv("/root/data/qw1/cv_summary.csv")

    # Filter XGBoost from QW-1 summary (column is "model", not "config")
    xgb_row = df_qw1[df_qw1["model"] == "XGBoost"].copy()

    if not xgb_row.empty:
        # Map column names from cv_summary format to cv_5fold format
        col_map = {
            "model": "config",
            "precision_mean": "prec_mean",
            "precision_std": "prec_std",
            "recall_mean": "rec_mean",
            "recall_std": "rec_std",
        }
        xgb_row = xgb_row.rename(columns=col_map)
        xgb_row["config"] = "XGBoost ≥0.5"

        # Keep only columns that exist in cv_5fold
        xgb_row = xgb_row[[c for c in df_cv.columns if c in xgb_row.columns]]

        # Add to CV results if not already present
        if "XGBoost ≥0.5" not in df_cv["config"].values:
            df_cv = pd.concat([df_cv, xgb_row], ignore_index=True)
            logger.info("Added XGBoost ≥0.5 to CV results")
        else:
            logger.info("XGBoost ≥0.5 already in CV results")

    df_cv.to_csv(cv_path, index=False)
    logger.info(f"Updated {cv_path}")


def update_ablation_results(volume_path, X_test, y_test, logger):
    """Update ablation_results.csv with XGBoost across all ablation categories."""
    import pandas as pd

    logger.info("Updating ablation_results.csv with XGBoost (all categories)...")

    ablation_path = volume_path / "csvs" / "ablation_results.csv"
    scored_path = volume_path / "probas" / "xgboost_holdout_scored.csv"

    if not scored_path.exists():
        logger.info(
            "Scored holdout artifact not found; rebuilding from X_test + probas"
        )
        build_ablation_scored_holdout(volume_path, X_test, y_test, logger)

    df_ablation = pd.read_csv("/root/data/ablation_results.csv")
    df_scored = pd.read_csv(scored_path)

    y_true = df_scored["y_true"].to_numpy()
    xgb_proba = df_scored["proba_xgboost"].to_numpy()
    rules_scores = df_scored["rules_score"].to_numpy()

    target_categories = {
        "ml-only",
        "hybrid-and",
        "hybrid-or",
        "cascade-ml-rules",
        "cascade-rules-ml",
        "consensus-ml",
        "consensus-hybrid",
    }

    xgb_existing_mask = df_ablation["category"].isin(target_categories) & df_ablation[
        "config"
    ].astype(str).str.contains("XGBoost", case=False, na=False)
    removed = int(xgb_existing_mask.sum())
    if removed > 0:
        df_ablation = df_ablation.loc[~xgb_existing_mask].copy()
        logger.info(f"Removed {removed} previous XGBoost ablation rows")

    new_rows = []

    def add_row(config_name, category, y_pred, y_score):
        metrics = evaluate_binary_predictions(y_true, y_pred, y_score=y_score)
        new_rows.append(
            {
                "config": config_name,
                "category": category,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tn": metrics["tn"],
                "auc_roc": metrics["auc_roc"],
            }
        )

    # 1) ML-only
    for ml_th in [0.3, 0.5, 0.7]:
        y_pred = (xgb_proba >= ml_th).astype(int)
        add_row(
            config_name=f"ML-only XGBoost (>={ml_th})",
            category="ml-only",
            y_pred=y_pred,
            y_score=xgb_proba,
        )

    # 2) Hybrid AND
    for ml_th in [0.5, 0.6, 0.7]:
        for rule_th in [5, 6, 7, 8]:
            y_pred = ((xgb_proba >= ml_th) & (rules_scores >= rule_th)).astype(int)
            add_row(
                config_name=f"Hybrid-AND XGBoost >={ml_th} + Rules>={rule_th}",
                category="hybrid-and",
                y_pred=y_pred,
                y_score=xgb_proba,
            )

    # 3) Hybrid OR
    for ml_th in [0.7, 0.9]:
        for rule_th in [8, 9]:
            y_pred = ((xgb_proba >= ml_th) | (rules_scores >= rule_th)).astype(int)
            add_row(
                config_name=f"Hybrid-OR XGBoost >={ml_th} + Rules>={rule_th}",
                category="hybrid-or",
                y_pred=y_pred,
                y_score=xgb_proba,
            )

    # 4) Cascade ML -> Rules
    for ml_th in [0.3, 0.5]:
        ml_pass = xgb_proba >= ml_th
        for rule_th in [7, 8]:
            y_pred = (ml_pass & (rules_scores >= rule_th)).astype(int)
            add_row(
                config_name=f"Cascade XGBoost >={ml_th} -> Rules>={rule_th}",
                category="cascade-ml-rules",
                y_pred=y_pred,
                y_score=xgb_proba,
            )

    # 5) Cascade Rules -> ML
    for rule_th in [5, 6]:
        rules_pass = rules_scores >= rule_th
        for ml_th in [0.5, 0.7]:
            y_pred = (rules_pass & (xgb_proba >= ml_th)).astype(int)
            add_row(
                config_name=f"Cascade Rules>={rule_th} -> XGBoost >={ml_th}",
                category="cascade-rules-ml",
                y_pred=y_pred,
                y_score=xgb_proba,
            )

    # 6) Consensus categories (3-model majority including XGBoost)
    consensus_cols = ["proba_xgboost", "proba_rf_smote", "proba_gb"]
    if all(c in df_scored.columns for c in consensus_cols):
        import numpy as np

        consensus_matrix = df_scored[consensus_cols].to_numpy()
        mean_consensus_score = np.mean(consensus_matrix, axis=1)
        votes_required = 2  # majority of 3 models

        for consensus_th in [0.5, 0.7]:
            votes = (consensus_matrix >= consensus_th).sum(axis=1)
            y_consensus = (votes >= votes_required).astype(int)

            add_row(
                config_name=f"Consensus-ML XGBoost majority (th={consensus_th}, 3 models)",
                category="consensus-ml",
                y_pred=y_consensus,
                y_score=mean_consensus_score,
            )

            for rule_th in [6, 7]:
                y_consensus_rules = (y_consensus & (rules_scores >= rule_th)).astype(
                    int
                )
                add_row(
                    config_name=(
                        "Consensus+Rules XGBoost majority "
                        f"(th={consensus_th}) + Rules>={rule_th}"
                    ),
                    category="consensus-hybrid",
                    y_pred=y_consensus_rules,
                    y_score=mean_consensus_score,
                )
    else:
        missing = [c for c in consensus_cols if c not in df_scored.columns]
        logger.warning(
            "Skipping XGBoost consensus categories (missing proba columns: %s)",
            ", ".join(missing),
        )

    df_new = pd.DataFrame(new_rows)
    if not df_new.empty:
        df_ablation = pd.concat([df_ablation, df_new], ignore_index=True)
        df_ablation.to_csv(ablation_path, index=False)

        logger.info(
            "Updated %s with %d XGBoost rows across categories",
            ablation_path,
            len(df_new),
        )

        best = df_new.sort_values("f1", ascending=False).head(8)
        for _, row in best.iterrows():
            logger.info(
                "Top XGBoost config: %s | %s | F1=%.4f",
                row["category"],
                row["config"],
                row["f1"],
            )
    else:
        logger.warning("No new XGBoost rows computed for ablation_results.csv")


def update_pareto_grid(volume_path, logger):
    """Update pareto_grid.csv with XGBoost grid search."""
    import pandas as pd

    logger.info("Updating pareto_grid.csv with XGBoost grid...")

    # Load existing pareto and XGBoost probas
    pareto_path = volume_path / "csvs" / "pareto_grid.csv"
    probas_path = volume_path / "probas" / "xgboost_holdout_probas.csv"

    df_pareto = pd.read_csv("/root/data/pareto_grid.csv")
    df_probas = pd.read_csv(probas_path)

    y_true = df_probas["y_true"].values
    probas = df_probas["proba"].values

    # Grid: ML thresholds × Rules thresholds (simplified, placeholder)
    ml_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    new_rows = []

    for ml_th in ml_thresholds:
        y_pred = (probas >= ml_th).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Check if XGBoost entry exists
        existing = df_pareto[
            (df_pareto["ml_model"] == "XGBoost") & (df_pareto["ml_threshold"] == ml_th)
        ]

        if existing.empty:
            new_rows.append(
                {
                    "ml_model": "XGBoost",
                    "ml_threshold": ml_th,
                    "rules_threshold": 0.0,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                }
            )

    if new_rows:
        df_pareto = pd.concat([df_pareto, pd.DataFrame(new_rows)], ignore_index=True)
        df_pareto.to_csv(pareto_path, index=False)
        logger.info(f"Updated {pareto_path} with {len(new_rows)} XGBoost grid points")
    else:
        logger.info("XGBoost grid already in pareto_grid.csv")


def update_epidemiological_impact(volume_path, logger):
    """Update epidemiological_impact.csv with XGBoost ≥0.5/≥0.7."""
    import pandas as pd

    logger.info("Updating epidemiological_impact.csv...")

    # Load existing impact and XGBoost probas
    epi_path = volume_path / "csvs" / "epidemiological_impact.csv"
    probas_path = volume_path / "probas" / "xgboost_holdout_probas.csv"

    df_epi = pd.read_csv("/root/data/epidemiological_impact.csv")
    df_probas = pd.read_csv(probas_path)

    y_true = df_probas["y_true"].values
    probas = df_probas["proba"].values

    # Calculate metrics for thresholds 0.5 and 0.7
    thresholds = [0.5, 0.7]
    new_rows = []

    for th in thresholds:
        y_pred = (probas >= th).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        detected = tp
        missed = fn
        to_review = tp + fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Epidemiological metrics (simplified)
        total_deaths = 185  # From epidemiological context (placeholder)
        rate_detected_pct = detected / total_deaths
        pct_of_true_found = recall * 100
        cost_per_death = to_review / detected if detected > 0 else 0.0

        method_name = f"ML XGBoost >={th}"

        # Check if already exists
        if method_name not in df_epi["method"].values:
            new_rows.append(
                {
                    "method": method_name,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "detected": detected,
                    "missed": missed,
                    "to_review": to_review,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "rate_detected_pct": rate_detected_pct,
                    "rate_true_pct": 0.3998,  # From existing data
                    "pct_of_true_found": pct_of_true_found,
                    "cost_per_death": cost_per_death,
                }
            )
            logger.info(
                f"XGBoost th={th}: detected={detected}, missed={missed}, cost={cost_per_death:.2f}"
            )

    if new_rows:
        df_epi = pd.concat([df_epi, pd.DataFrame(new_rows)], ignore_index=True)
        df_epi.to_csv(epi_path, index=False)
        logger.info(f"Updated {epi_path} with {len(new_rows)} XGBoost methods")
    else:
        logger.info("XGBoost methods already in epidemiological_impact.csv")


def update_shap_combined(volume_path, logger):
    """Combine RF + XGBoost SHAP importance into single CSV."""
    import pandas as pd

    logger.info("Creating shap_importance_combined.csv...")

    shap_dir = volume_path / "shap"
    output_path = volume_path / "csvs" / "shap_importance_combined.csv"

    # Load individual SHAP importances
    df_xgb = pd.read_csv(shap_dir / "xgboost_shap_importance.csv")
    df_rf = pd.read_csv(shap_dir / "rf_shap_importance.csv")

    # Add model column
    df_xgb["model"] = "XGBoost"
    df_rf["model"] = "RF"

    # Combine
    df_combined = pd.concat([df_xgb, df_rf], ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    logger.info(f"Saved {output_path} ({len(df_combined)} rows)")


def update_feature_importance_combined(volume_path, logger):
    """Combine RF Gini + XGBoost gain into single CSV."""
    import pandas as pd

    logger.info("Creating feature_importance_combined.csv...")

    shap_dir = volume_path / "shap"
    output_path = volume_path / "csvs" / "feature_importance_per_band_combined.csv"

    # Load XGBoost global importance
    df_xgb = pd.read_csv(shap_dir / "xgboost_feature_importance_global.csv")

    # Load existing RF importance (if available)
    rf_path = Path("/root/data/feature_importance_per_band.csv")

    if rf_path.exists():
        df_rf = pd.read_csv(rf_path)
        # Add model column if not present
        if "model" not in df_rf.columns:
            df_rf["model"] = "RF"

        # Standardize XGBoost format
        df_xgb_std = df_xgb.rename(columns={"importance_gain": "importance"})
        df_xgb_std["model"] = "XGBoost"

        # Combine
        df_combined = pd.concat([df_rf, df_xgb_std], ignore_index=True)
    else:
        logger.warning("RF feature importance not found, using XGBoost only")
        df_combined = df_xgb.copy()
        df_combined["model"] = "XGBoost"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    logger.info(f"Saved {output_path} ({len(df_combined)} rows)")


def generate_pgf_figures(volume_path, logger):
    """Generate all 9 PGF figures for thesis."""
    import matplotlib

    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
            "pgf.preamble": "\n".join(
                [
                    r"\usepackage[utf8]{inputenc}",
                    r"\DeclareUnicodeCharacter{2265}{\ensuremath{\geq}}",
                    r"\DeclareUnicodeCharacter{2264}{\ensuremath{\leq}}",
                    r"\DeclareUnicodeCharacter{00B1}{\ensuremath{\pm}}",
                ]
            ),
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    import matplotlib.pyplot as plt

    # Force PGF backend even if pyplot was already initialized (e.g. by SHAP)
    plt.switch_backend("pgf")
    import numpy as np
    import pandas as pd

    output_dir = volume_path / "pgfs"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating PGF figures...")

    # ───────────────────────────────────────────────────────────────────
    # 1. cv_boxplots_v2.pgf
    # ───────────────────────────────────────────────────────────────────
    logger.info("1/9 Generating cv_boxplots_v2.pgf...")
    df_cv = pd.read_csv(volume_path / "csvs" / "cv_5fold_results.csv")

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    configs = df_cv["config"].tolist()
    f1_means = df_cv["f1_mean"].tolist()
    f1_stds = df_cv["f1_std"].tolist()

    x = np.arange(len(configs))
    ax.bar(x, f1_means, color="#2ca02c", edgecolor="black", linewidth=0.4)
    ax.errorbar(
        x, f1_means, yerr=f1_stds, fmt="none", ecolor="black", capsize=3, linewidth=1
    )
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1-score (média $\\pm$ desvio)")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "cv_boxplots_v2.pgf")
    plt.close()

    # ───────────────────────────────────────────────────────────────────
    # 2. pareto_frontier_v2.pgf
    # ───────────────────────────────────────────────────────────────────
    logger.info("2/9 Generating pareto_frontier_v2.pgf...")
    df_pareto = pd.read_csv(volume_path / "csvs" / "pareto_grid.csv")

    fig, ax = plt.subplots(figsize=(5, 4))

    for model in df_pareto["ml_model"].unique():
        subset = df_pareto[df_pareto["ml_model"] == model]
        ax.scatter(
            subset["recall"],
            subset["precision"],
            label=model,
            alpha=0.7,
            s=30,
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisão")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(output_dir / "pareto_frontier_v2.pgf")
    plt.close()

    # ───────────────────────────────────────────────────────────────────
    # 3-5. Epidemiological impact figures (epi_deteccao, epi_revisoes, epi_custo)
    # ───────────────────────────────────────────────────────────────────
    logger.info("3-5/9 Generating epidemiological impact figures...")
    df_epi = pd.read_csv(volume_path / "csvs" / "epidemiological_impact.csv")

    # Filter relevant methods for visualization
    methods_to_plot = [
        "Naive threshold >=8",
        "ML RF+SMOTE >=0.5",
        "Hybrid-OR RF>=0.7+Rules>=8",
        "ML XGBoost >=0.5",
    ]

    df_plot = df_epi[df_epi["method"].isin(methods_to_plot)].copy()

    # 3. epi_deteccao.pgf
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    x_pos = np.arange(len(df_plot))
    w = 0.32

    ax.bar(
        x_pos - w / 2,
        df_plot["detected"],
        w,
        label="Detectados",
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x_pos + w / 2,
        df_plot["missed"],
        w,
        label="Perdidos",
        color="#d62728",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [
            m.replace("ML ", "").replace("Naive threshold ", "Limiar ")
            for m in df_plot["method"]
        ],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("N. de pares verdadeiros")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "epi_deteccao.pgf")
    plt.close()

    # 4. epi_revisoes.pgf
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    colors = ["#ff7f0e", "#2ca02c", "#17becf", "#9467bd"]
    ax.bar(
        x_pos,
        df_plot["to_review"],
        0.45,
        color=colors[: len(df_plot)],
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [
            m.replace("ML ", "").replace("Naive threshold ", "Limiar ")
            for m in df_plot["method"]
        ],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Total de pares para revisão")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "epi_revisoes.pgf")
    plt.close()

    # 5. epi_custo.pgf
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.bar(
        x_pos,
        df_plot["cost_per_death"],
        0.45,
        color="#e377c2",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [
            m.replace("ML ", "").replace("Naive threshold ", "Limiar ")
            for m in df_plot["method"]
        ],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Custo por óbito detectado")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "epi_custo.pgf")
    plt.close()

    # ───────────────────────────────────────────────────────────────────
    # 6. nb01_model_comparison.pgf
    # ───────────────────────────────────────────────────────────────────
    logger.info("6/9 Generating nb01_model_comparison.pgf...")

    # Read actual holdout metrics from probas CSVs
    from sklearn.metrics import precision_score, recall_score, f1_score

    model_display = []
    precision_vals = []
    recall_vals = []
    f1_vals = []

    probas_dir = volume_path / "probas"
    for fname, label in [
        ("xgboost_holdout_probas.csv", "XGBoost"),
        ("rf_smote_holdout_probas.csv", "RF+SMOTE"),
        ("gb_holdout_probas.csv", "GB"),
    ]:
        fpath = probas_dir / fname
        if fpath.exists():
            df_p = pd.read_csv(fpath)
            y_t = df_p["y_true"].values
            y_p = (df_p["proba"].values >= 0.5).astype(int)
            model_display.append(label)
            precision_vals.append(precision_score(y_t, y_p, zero_division=0))
            recall_vals.append(recall_score(y_t, y_p, zero_division=0))
            f1_vals.append(f1_score(y_t, y_p, zero_division=0))

    # Add TabNet if available
    tabnet_path = probas_dir / "tabnet_holdout_probas.csv"
    if tabnet_path.exists():
        df_p = pd.read_csv(tabnet_path)
        y_t = df_p["y_true"].values
        y_p = (df_p["proba"].values >= 0.5).astype(int)
        model_display.append("TabNet")
        precision_vals.append(precision_score(y_t, y_p, zero_division=0))
        recall_vals.append(recall_score(y_t, y_p, zero_division=0))
        f1_vals.append(f1_score(y_t, y_p, zero_division=0))

    x_pos = np.arange(len(model_display))
    w = 0.25

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars_precision = ax.bar(
        x_pos - w,
        precision_vals,
        w,
        label="Precisão",
        color="#1f77b4",
        edgecolor="black",
    )
    bars_recall = ax.bar(
        x_pos,
        recall_vals,
        w,
        label="Revocação",
        color="#ff7f0e",
        edgecolor="black",
    )
    bars_f1 = ax.bar(
        x_pos + w,
        f1_vals,
        w,
        label="Escore F1",
        color="#2ca02c",
        edgecolor="black",
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_display, fontsize=9)
    ax.set_ylabel("Escore (0--1)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    for bars in (bars_precision, bars_recall, bars_f1):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.012,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "nb01_model_comparison.pgf")
    plt.close()

    # ───────────────────────────────────────────────────────────────────
    # 7. ablation_best_category.pgf
    # ───────────────────────────────────────────────────────────────────
    logger.info("7/9 Generating ablation_best_category.pgf...")
    df_ablation = pd.read_csv(volume_path / "csvs" / "ablation_results.csv")

    # Get best config per category
    best_per_category = df_ablation.loc[df_ablation.groupby("category")["f1"].idxmax()]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x_pos = np.arange(len(best_per_category))
    bars = ax.bar(
        x_pos,
        best_per_category["f1"],
        color="#4daf4a",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        best_per_category["category"], rotation=45, ha="right", fontsize=8
    )
    ax.set_ylabel("Escore F1 (melhor configuração)")
    ax.set_ylim(0, 1.05)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.012,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "ablation_best_category.pgf")
    plt.close()

    # ───────────────────────────────────────────────────────────────────
    # 8. shap_summary.pgf (RF vs XGBoost paired bars, top-20)
    # ───────────────────────────────────────────────────────────────────
    logger.info("8/9 Generating shap_summary.pgf...")
    df_shap = pd.read_csv(volume_path / "csvs" / "shap_importance_combined.csv")

    # Pivot to get RF and XGBoost side-by-side
    df_shap_pivot = df_shap.pivot(
        index="feature", columns="model", values="shap_mean_abs"
    ).fillna(0)

    # Sort by XGBoost importance, take top 20
    df_shap_pivot = df_shap_pivot.sort_values("XGBoost", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(7, 5))
    x_pos = np.arange(len(df_shap_pivot))
    w = 0.35

    ax.barh(
        x_pos - w / 2,
        df_shap_pivot["XGBoost"],
        w,
        label="XGBoost",
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.3,
    )
    ax.barh(
        x_pos + w / 2,
        df_shap_pivot["RF"],
        w,
        label="RF",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.3,
    )

    ax.set_yticks(x_pos)
    ax.set_yticklabels([f.replace("_", r"\_") for f in df_shap_pivot.index], fontsize=8)
    ax.set_xlabel("SHAP importância (média absoluta)")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "shap_summary.pgf")
    plt.close()

    # ───────────────────────────────────────────────────────────────────
    # 9. feature_importance_heatmap.pgf (XGBoost gain-based)
    # ───────────────────────────────────────────────────────────────────
    logger.info("9/9 Generating feature_importance_heatmap.pgf...")
    df_importance = pd.read_csv(
        volume_path / "shap" / "xgboost_feature_importance_global.csv"
    )

    # Take top 15 features
    top15 = df_importance.head(15)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(top15)))

    ax.barh(
        range(len(top15)),
        top15["importance_gain"],
        color=colors,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels([f.replace("_", r"\_") for f in top15["feature"]], fontsize=8)
    ax.set_xlabel("Importância (gain)")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / "feature_importance_heatmap.pgf")
    plt.close()

    logger.info("All 9 PGF figures generated successfully")


def generate_latex_tables(volume_path, logger):
    """Generate 2 LaTeX tables."""
    logger.info("Generating LaTeX tables...")

    tables_dir = volume_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    # ───────────────────────────────────────────────────────────────────
    # 1. tab_impacto_epidemiologico.tex
    # ───────────────────────────────────────────────────────────────────
    logger.info("1/2 Generating tab_impacto_epidemiologico.tex...")

    df_epi = pd.read_csv(volume_path / "csvs" / "epidemiological_impact.csv")

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Comparação de métodos: óbitos detectados, custo operacional e taxa corrigida}"
    )
    lines.append(r"\label{tab:impacto-epidemiologico}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{@{}lrrrrrl@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"Método & Det. & Perd. & Rev. & Prec. & \textit{Recall} & \% Verd. \\"
    )
    lines.append(r"\midrule")

    for _, row in df_epi.iterrows():
        method = (
            str(row["method"]).replace("ML XGBoost", "XGBoost").replace(">=", r"$\geq$")
        )
        det = int(row["detected"])
        miss = int(row["missed"])
        rev = int(row["to_review"])
        prec = row["precision"]
        rec = row["recall"]
        pct = row["pct_of_true_found"]

        lines.append(
            f"  {method} & {det} & {miss} & {rev} & {prec:.2f} & {rec:.2f} & {pct:.1f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    output_path = tables_dir / "tab_impacto_epidemiologico.tex"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved {output_path}")

    # ───────────────────────────────────────────────────────────────────
    # 2. tab_feature_importance.tex
    # ───────────────────────────────────────────────────────────────────
    logger.info("2/2 Generating tab_feature_importance.tex...")

    df_importance = pd.read_csv(
        volume_path / "shap" / "xgboost_feature_importance_global.csv"
    )

    top15 = df_importance.head(15)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Importância dos 15 atributos mais relevantes (XGBoost, gain)}"
    )
    lines.append(r"\label{tab:feature-importance-xgboost}")
    lines.append(r"\begin{tabular}{clc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{\#} & \textbf{Atributo} & \textbf{Importância (gain)} \\")
    lines.append(r"\midrule")

    for i, row in enumerate(top15.itertuples(), 1):
        feature = str(row.feature).replace("_", r"\_")
        importance = row.importance_gain
        lines.append(f"  {i} & {feature} & {importance:.4f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    output_path = tables_dir / "tab_feature_importance.tex"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved {output_path}")

    logger.info("All LaTeX tables generated successfully")


def create_manifest(volume_path, logger):
    """Create manifest.json with file listing and checksums."""
    import hashlib

    logger.info("Creating manifest.json...")

    from datetime import datetime, timezone

    manifest = {"files": [], "generated_at": datetime.now(timezone.utc).isoformat()}

    # Collect all files
    for subdir in ["probas", "shap", "csvs", "pgfs", "tables"]:
        dir_path = volume_path / subdir
        if dir_path.exists():
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    # Calculate checksum
                    md5 = hashlib.md5(file_path.read_bytes()).hexdigest()
                    manifest["files"].append(
                        {
                            "path": str(file_path.relative_to(volume_path)),
                            "size_bytes": file_path.stat().st_size,
                            "md5": md5,
                        }
                    )

    manifest_path = volume_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Manifest created: {len(manifest['files'])} files tracked")


# ═══════════════════════════════════════════════════════════════════════
# Main Pipeline Function
# ═══════════════════════════════════════════════════════════════════════


@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
    volumes={"/vol": volume},
)
def run_pipeline():
    """Execute full 6-stage pipeline."""
    import pandas as pd
    from pathlib import Path

    logger = setup_logging()
    volume_path = Path("/vol/pipeline_output")

    logger.info("=" * 70)
    logger.info("QW-1 UNIFIED PIPELINE - Starting")
    logger.info("=" * 70)

    # ───────────────────────────────────────────────────────────────────
    # Stage 1: Load data and train models
    # ───────────────────────────────────────────────────────────────────
    logger.info("\n[STAGE 1/6] Training models...")

    X_train, X_test, y_train, y_test = load_and_prepare_data(logger)

    # Train models
    logger.info("Training XGBoost (tree_method=hist, GPU accelerated)...")
    xgboost_model = train_xgboost(X_train, y_train, logger)

    tabnet_model = train_tabnet(X_train, y_train, logger)

    logger.info("Training RF+SMOTE...")
    rf_model = train_rf_smote(X_train, y_train, logger)

    logger.info("Training Gradient Boosting...")
    gb_model = train_gradient_boosting(X_train, y_train, logger)

    logger.info("All models trained successfully")

    # Export probabilities
    logger.info("\n[STAGE 2/6] Exporting probabilities...")
    models = [xgboost_model, tabnet_model, rf_model, gb_model]
    model_names = ["xgboost", "tabnet", "rf_smote", "gb"]

    probas_dir = volume_path / "probas"
    export_probabilities(models, model_names, X_test, y_test, probas_dir, logger)
    build_ablation_scored_holdout(volume_path, X_test, y_test, logger)

    # ───────────────────────────────────────────────────────────────────
    # Stage 3: Calculate SHAP + feature importance
    # ───────────────────────────────────────────────────────────────────
    logger.info("\n[STAGE 3/6] Calculating SHAP + feature importance...")

    shap_dir = volume_path / "shap"
    calculate_shap_values(xgboost_model, rf_model, X_train, X_test, shap_dir, logger)
    calculate_feature_importance(xgboost_model, X_train, shap_dir, logger)

    # ───────────────────────────────────────────────────────────────────
    # Stage 4: Update intermediate CSVs
    # ───────────────────────────────────────────────────────────────────
    logger.info("\n[STAGE 4/6] Updating intermediate CSVs...")

    csvs_dir = volume_path / "csvs"
    csvs_dir.mkdir(parents=True, exist_ok=True)

    # Copy intermediate CSVs to volume for updates
    import shutil

    for csv_name in [
        "cv_5fold_results.csv",
        "ablation_results.csv",
        "pareto_grid.csv",
        "epidemiological_impact.csv",
    ]:
        src = Path(f"/root/data/{csv_name}")
        dst = csvs_dir / csv_name
        shutil.copy(src, dst)

    update_cv_results(volume_path, logger)
    update_ablation_results(volume_path, X_test, y_test, logger)
    update_pareto_grid(volume_path, logger)
    update_epidemiological_impact(volume_path, logger)
    update_shap_combined(volume_path, logger)
    update_feature_importance_combined(volume_path, logger)

    # ───────────────────────────────────────────────────────────────────
    # Stage 5: Generate figures and tables
    # ───────────────────────────────────────────────────────────────────
    logger.info("\n[STAGE 5/6] Generating PGF figures and LaTeX tables...")

    generate_pgf_figures(volume_path, logger)
    generate_latex_tables(volume_path, logger)

    # ───────────────────────────────────────────────────────────────────
    # Stage 6: Create manifest and finalize
    # ───────────────────────────────────────────────────────────────────
    logger.info("\n[STAGE 6/6] Creating manifest and finalizing...")

    create_manifest(volume_path, logger)

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"All outputs saved to Modal volume qw1-data:/pipeline_output/")
    logger.info("=" * 70)
    logger.info("\nTo download outputs:")
    logger.info("  modal volume get qw1-data pipeline_output/ data/pipeline_output/")


# ═══════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════


@app.local_entrypoint()
def main():
    """Local entry point for running the pipeline."""
    print("Launching QW-1 unified pipeline on Modal...")
    run_pipeline.remote()
    print("\nPipeline execution complete!")
    print("Download outputs with:")
    print("  modal volume get qw1-data pipeline_output/ data/pipeline_output/")
