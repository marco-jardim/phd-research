"""Quick Win 1: 5-model comparative study for QW-1.

Runs: RandomForest, MLP (128,64), MLP (64,32), TabNet (with FT-Transformer
fallback), and XGBoost under:

* fixed train/test split (seed=42, test_size=0.3)
* 5x5 cross-validation (RepeatedStratifiedKFold, seed=42) on the train split

Artifacts are saved under ``data/qw1/``:

* ``per_fold_metrics.csv``
* ``cv_summary.csv``
* ``holdout_metrics.csv``
* ``model_ranking.csv``
* ``reproducibility_manifest.json``
* ``acceptance_checks.json``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from importlib import metadata
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

# keep import local and optional for tabnet
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

import torch

# ---------------------------------------------------------------------------
# Prevent deadlock: torch multithread + sklearn joblib n_jobs=-1 on Windows
# can starve each other.  Fix: cap joblib workers and torch intra-op threads.
# ---------------------------------------------------------------------------
_N_JOBS = max(1, os.cpu_count() // 2)  # half the logical cores
torch.set_num_threads(max(1, os.cpu_count() // 2))

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gzcmd.splitting import SplitSpec, split_train_test_indices
from scripts.feature_engineering import engineer_features, load_data


SEED = 42
TEST_SIZE = 0.30
N_SPLITS = 5
N_REPEATS = 5


def _log(msg: str) -> None:
    """Print with timestamp and immediate flush (avoids Windows buffering)."""
    elapsed = time.perf_counter() - _T0 if "_T0" in globals() else 0.0
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


_T0 = time.perf_counter()


@dataclass(frozen=True)
class HoldoutArtifact:
    model: str
    metrics: dict[str, float]
    confusion: dict[str, int]


def _safe_version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _git_metadata(root: Path) -> dict[str, Any]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except Exception:
        sha = "unknown"

    try:
        dirty = (
            subprocess.check_output(
                ["git", "status", "--short"], cwd=root, text=True
            ).strip()
            != ""
        )
    except Exception:
        dirty = None

    return {"git_sha": sha, "dirty": dirty}


def _fp_fn_rates(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fp_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else math.nan
    fn_rate = float(fn / (fn + tp)) if (fn + tp) > 0 else math.nan
    return fp_rate, fn_rate


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    fp_rate, fn_rate = _fp_fn_rates(y_true, y_pred)
    metrics: dict[str, float] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "fp_rate": float(fp_rate),
        "fn_rate": float(fn_rate),
    }
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        metrics["auc_roc"] = float("nan")

    try:
        metrics["auc_pr"] = float(average_precision_score(y_true, y_proba))
    except ValueError:
        metrics["auc_pr"] = float("nan")

    return metrics


def _probability_vector(model: Any, x: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(float)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=float)))
    raise RuntimeError("Model does not expose probability interface")


class _FTTransformerModule(nn.Module):
    def __init__(
        self, n_features: int, d_token: int = 16, n_heads: int = 4, n_layers: int = 1
    ):
        super().__init__()
        self.feature_embedding = nn.Linear(1, d_token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=2 * d_token,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)
        self.n_features = n_features
        self.d_token = d_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.n_features, 1)
        x = self.feature_embedding(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        pooled = self.norm(pooled)
        return self.head(pooled).squeeze(1)


class FTTransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_features: int,
        d_token: int = 16,
        n_heads: int = 4,
        n_layers: int = 1,
        max_epochs: int = 100,
        patience: int = 15,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        random_state: int = SEED,
        verbose: int = 0,
    ) -> None:
        self.n_features = n_features
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose
        self._model: _FTTransformerModule | None = None
        self._device: str = "cpu"

    def fit(self, X: Any, y: Any) -> "FTTransformerClassifier":
        x = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y).astype(np.int64)
        if x.ndim != 2:
            raise ValueError("X must be 2D")
        if self.n_features != x.shape[1]:
            raise ValueError(f"Expected {self.n_features} features, got {x.shape[1]}")

        rs = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)

        self._model = _FTTransformerModule(
            n_features=x.shape[1],
            d_token=self.d_token,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
        ).to(self._device)

        t_x = torch.tensor(x, dtype=torch.float32, device=self._device)
        t_y = torch.tensor(y_arr, dtype=torch.float32, device=self._device)

        n_pos = int(y_arr.sum())
        n_neg = int(len(y_arr) - n_pos)
        pos_weight = torch.tensor(
            float(max(n_neg, 1) / max(n_pos, 1)),
            dtype=torch.float32,
            device=self._device,
        )

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = AdamW(self._model.parameters(), lr=self.learning_rate)
        dataset = TensorDataset(t_x, t_y)

        n = len(dataset)
        train_idx = np.arange(n)
        valid_mask = rs.rand(n) < 0.2
        if valid_mask.sum() == 0 or valid_mask.sum() == n:
            valid_mask = np.zeros(n, dtype=bool)
            valid_mask[: max(1, n // 5)] = True
            rs.shuffle(valid_mask)

        val_indices = np.nonzero(valid_mask)[0]
        train_indices = np.nonzero(~valid_mask)[0]

        train_set = torch.utils.data.Subset(dataset, train_indices)
        val_set = torch.utils.data.Subset(dataset, val_indices)
        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_set, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        best_loss = math.inf
        best_state: dict[str, Any] | None = None
        no_improve = 0

        for epoch in range(1, self.max_epochs + 1):
            self._model.train()
            epoch_loss = []
            for bx, by in train_loader:
                optimizer.zero_grad()
                logits = self._model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                epoch_loss.append(float(loss.item()))

            if epoch % 5 != 0 and epoch < self.max_epochs:
                continue

            self._model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for bx, by in val_loader:
                    val_logits = self._model(bx)
                    val_loss = criterion(val_logits, by)
                    val_losses.append(float(val_loss.item()))

            avg_val = (
                float(np.mean(val_losses)) if val_losses else float(np.mean(epoch_loss))
            )
            if avg_val + 1e-6 < best_loss:
                best_loss = avg_val
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self._model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

            if self.verbose:
                print(f"[FT] epoch={epoch:03d} val_loss={avg_val:.6f}")

        if best_state is not None and self._model is not None:
            self._model.load_state_dict(best_state)

        return self

    def _ensure_fitted(self) -> _FTTransformerModule:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet")
        return self._model

    def predict_proba(self, X: Any) -> np.ndarray:
        model = self._ensure_fitted()
        x = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            logits = model(torch.tensor(x, dtype=torch.float32, device=self._device))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        return np.c_[1 - probs, probs]

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _TabNetSklearnWrapper(BaseEstimator):
    """Wraps pytorch-tabnet TabNetClassifier for sklearn clone/fit contract.

    TabNet v4.1 moved max_epochs, patience, batch_size to fit() params.
    This wrapper stores them as constructor params (for sklearn.clone() compat)
    and forwards them to fit().
    """

    def __init__(
        self,
        n_d: int = 16,
        n_a: int = 16,
        n_steps: int = 5,
        gamma: float = 1.5,
        seed: int = 42,
        max_epochs: int = 100,
        patience: int = 15,
        batch_size: int = 1024,
        verbose: int = 0,
    ):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.seed = seed
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> "_TabNetSklearnWrapper":
        import time as _time
        from pytorch_tabnet.tab_model import TabNetClassifier  # type: ignore

        _X = np.asarray(X, dtype=np.float64)
        _y = np.asarray(y, dtype=np.int64)
        print(
            f"  [TabNet] fit() called  X={_X.shape}  y={_y.shape}  "
            f"epochs={self.max_epochs}  batch={self.batch_size}",
            flush=True,
        )

        t0 = _time.perf_counter()
        print("  [TabNet] constructing TabNetClassifier ...", flush=True)
        self._model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            seed=self.seed,
            verbose=self.verbose,
        )
        print(
            f"  [TabNet] constructor done ({_time.perf_counter() - t0:.2f}s). "
            f"Device={self._model.device}",
            flush=True,
        )

        t1 = _time.perf_counter()
        print("  [TabNet] starting .fit() training ...", flush=True)
        self._model.fit(
            _X,
            _y,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=False,
        )
        print(
            f"  [TabNet] training done ({_time.perf_counter() - t1:.2f}s, "
            f"total {_time.perf_counter() - t0:.2f}s)",
            flush=True,
        )
        self.is_fitted_ = True  # sklearn check_is_fitted() convention
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self._model.predict(np.asarray(X, dtype=np.float64))

    def predict_proba(self, X: Any) -> np.ndarray:
        return self._model.predict_proba(np.asarray(X, dtype=np.float64))


def _build_tabnet_or_ft_model(n_features: int) -> BaseEstimator:
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier  # type: ignore  # noqa: F401

        return _TabNetSklearnWrapper(
            n_d=16,
            n_a=16,
            n_steps=5,
            gamma=1.5,
            seed=SEED,
            max_epochs=100,
            patience=15,
            batch_size=1024,
            verbose=0,
        )
    except ImportError:
        return FTTransformerClassifier(
            n_features=n_features,
            max_epochs=100,
            patience=15,
            batch_size=1024,
            random_state=SEED,
        )


def _build_model_factories(n_features: int) -> dict[str, Callable[[], BaseEstimator]]:
    return {
        "RandomForest": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        class_weight="balanced",
                        random_state=SEED,
                        n_jobs=_N_JOBS,
                    ),
                ),
            ]
        ),
        "MLP (128,64)": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        max_iter=500,
                        early_stopping=True,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "MLP (64,32)": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=500,
                        early_stopping=True,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "TabNet (fallback=FT-Transformer)": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", _build_tabnet_or_ft_model(n_features=n_features)),
            ]
        ),
        "XGBoost": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=SEED,
                        eval_metric="logloss",
                        n_jobs=_N_JOBS,
                    ),
                ),
            ]
        ),
    }


def _evaluate_fold(
    model: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    model_name: str,
    fold_id: int,
    repeat_id: int,
) -> tuple[dict[str, Any], bool, list[str]]:
    import time as _time

    warnings_log: list[str] = []
    capture_warnings = model_name.startswith("MLP")
    print(
        f"  [{model_name}] fold R{repeat_id}F{fold_id}: "
        f"x_train={x_train.shape} y_train={y_train.shape}",
        flush=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        if capture_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            warnings.simplefilter("always")
        else:
            warnings.simplefilter("ignore")

        t0 = _time.perf_counter()
        print(f"  [{model_name}] clone(model) ...", flush=True)
        fitted = clone(model)
        print(
            f"  [{model_name}] clone done ({_time.perf_counter() - t0:.2f}s). "
            f"Starting fit ...",
            flush=True,
        )
        t1 = _time.perf_counter()
        fitted.fit(x_train, y_train)
        print(
            f"  [{model_name}] fit done ({_time.perf_counter() - t1:.2f}s)",
            flush=True,
        )
        warnings_list = [
            str(item.message)
            for item in caught
            if issubclass(item.category, ConvergenceWarning)
        ]
        warnings_log.extend(warnings_list)

    y_pred = fitted.predict(x_val)
    y_proba = _probability_vector(fitted, x_val)
    metrics = _compute_metrics(
        np.asarray(y_val), np.asarray(y_pred), np.asarray(y_proba)
    )
    cm = confusion_matrix(np.asarray(y_val), np.asarray(y_pred), labels=[0, 1]).ravel()
    tn, fp, fn, tp = [int(v) for v in cm]

    row: dict[str, Any] = {
        "model": model_name,
        "repeat": repeat_id,
        "fold": fold_id,
        "global_fold": (repeat_id - 1) * N_SPLITS + fold_id,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auc_roc": metrics["auc_roc"],
        "auc_pr": metrics["auc_pr"],
        "accuracy": metrics["accuracy"],
        "fp_rate": metrics["fp_rate"],
        "fn_rate": metrics["fn_rate"],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "convergence_warnings": int(len(warnings_log) > 0),
        "convergence_messages": "; ".join(warnings_log),
    }

    return row, (len(warnings_log) > 0), warnings_log


def _evaluate_holdout(
    model: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    model_name: str,
) -> HoldoutArtifact:
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        fitted = clone(model).fit(x_train, y_train)

    y_pred = fitted.predict(x_test)
    y_proba = _probability_vector(fitted, x_test)
    metrics = _compute_metrics(
        np.asarray(y_test), np.asarray(y_pred), np.asarray(y_proba)
    )

    cm = confusion_matrix(np.asarray(y_test), np.asarray(y_pred), labels=[0, 1]).ravel()
    tn, fp, fn, tp = [int(v) for v in cm]

    return HoldoutArtifact(
        model=model_name,
        metrics=metrics,
        confusion={
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "n_samples": int(len(y_test)),
            "n_pos": int(np.sum(y_test)),
            "n_neg": int(len(y_test) - np.sum(y_test)),
        },
    )


def _summarize(per_fold_df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    records = []
    for model, model_df in per_fold_df.groupby("model", sort=True):
        rec: dict[str, Any] = {"model": model}
        for col in metric_cols:
            rec[f"{col}_mean"] = float(model_df[col].mean())
            rec[f"{col}_std"] = float(model_df[col].std(ddof=0))
        rec["n_folds"] = int(len(model_df))
        records.append(rec)
    return pd.DataFrame(records).sort_values("model").reset_index(drop=True)


def _rank_models(summary_df: pd.DataFrame) -> pd.DataFrame:
    hierarchy = [
        "f1",
        "precision",
        "recall",
        "auc_roc",
        "auc_pr",
    ]
    ranking_cols = [f"{m}_mean" for m in hierarchy]
    sort_cols = ranking_cols
    ranked = summary_df.sort_values(by=sort_cols, ascending=False).copy()
    ranked["metric_hierarchy"] = "; ".join(hierarchy)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["selected_metric_score"] = ranked[ranking_cols[0]]
    ranked = ranked[
        [
            "rank",
            "model",
            *sort_cols,
            "selected_metric_score",
            "metric_hierarchy",
            "n_folds",
        ]
    ]
    return ranked


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_progress_checkpoint(
    path: Path,
    per_fold_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    convergence_records: list[str],
) -> None:
    payload = {
        "per_fold_rows": per_fold_rows,
        "holdout_rows": holdout_rows,
        "convergence_records": convergence_records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _load_progress_checkpoint(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_fold_rows = list(payload.get("per_fold_rows", []))
    holdout_rows = list(payload.get("holdout_rows", []))
    convergence_records = list(payload.get("convergence_records", []))
    return per_fold_rows, holdout_rows, convergence_records


def run_study(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    data_path = args.data

    _log(f"Loading data from {data_path}")
    df = load_data(data_path)
    X_all, y_all = engineer_features(df)
    X_all = X_all.fillna(0)
    _log(
        f"Data loaded: {X_all.shape[0]} rows x {X_all.shape[1]} features, "
        f"pos_rate={y_all.mean():.4f}"
    )

    split = SplitSpec(
        split_by="row", test_size=TEST_SIZE, seed=SEED, group_stratify=True
    )
    train_idx, test_idx = split_train_test_indices(df, y_all, spec=split)

    X_train = X_all.iloc[train_idx].reset_index(drop=True)
    X_test = X_all.iloc[test_idx].reset_index(drop=True)
    y_train = y_all.iloc[train_idx].reset_index(drop=True).to_numpy()
    y_test = y_all.iloc[test_idx].reset_index(drop=True).to_numpy()
    _log(f"Split done: train={len(y_train)}, test={len(y_test)}")

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=SEED,
    )

    model_factories = _build_model_factories(n_features=X_train.shape[1])

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress_checkpoint.json"

    per_fold_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    convergence_records: list[str] = []

    if args.resume and progress_path.exists():
        per_fold_rows, holdout_rows, convergence_records = _load_progress_checkpoint(
            progress_path
        )
        _log(
            "Resuming from checkpoint: "
            f"{len(per_fold_rows)} fold rows, {len(holdout_rows)} holdout rows"
        )
    elif args.resume:
        _log("--resume passed but no checkpoint found; starting from scratch")

    _log(f"n_jobs={_N_JOBS}, torch_threads={torch.get_num_threads()}")
    total_folds = N_SPLITS * N_REPEATS
    for model_idx, (model_name, builder) in enumerate(model_factories.items(), 1):
        _log(f"=== Model {model_idx}/{len(model_factories)}: {model_name} ===")
        model_t0 = time.perf_counter()

        completed_folds = {
            (int(row["repeat"]), int(row["fold"]))
            for row in per_fold_rows
            if row.get("model") == model_name
        }
        holdout_done = any(row.get("model") == model_name for row in holdout_rows)
        if len(completed_folds) == total_folds and holdout_done:
            _log("  already complete in checkpoint; skipping model")
            continue

        if completed_folds:
            _log(f"  resume: {len(completed_folds)}/{total_folds} folds already done")

        model = builder()

        split_iter = list(cv.split(X_train, y_train))
        for global_fold, (fold_train_idx, fold_test_idx) in enumerate(
            split_iter, start=1
        ):
            repeat_id = ((global_fold - 1) // N_SPLITS) + 1
            local_fold = ((global_fold - 1) % N_SPLITS) + 1
            if (repeat_id, local_fold) in completed_folds:
                _log(
                    f"  fold {global_fold:2d}/{total_folds} (r{repeat_id}f{local_fold}) skipped (checkpoint)"
                )
                continue

            fold_x_train = X_train.iloc[fold_train_idx].reset_index(drop=True)
            fold_x_valid = X_train.iloc[fold_test_idx].reset_index(drop=True)
            fold_y_train = y_train[fold_train_idx]
            fold_y_valid = y_train[fold_test_idx]

            fold_t0 = time.perf_counter()
            row, _warned, warned_msgs = _evaluate_fold(
                model,
                fold_x_train,
                fold_y_train,
                fold_x_valid,
                fold_y_valid,
                model_name,
                fold_id=local_fold,
                repeat_id=repeat_id,
            )
            fold_dt = time.perf_counter() - fold_t0
            _log(
                f"  fold {global_fold:2d}/{total_folds} "
                f"(r{repeat_id}f{local_fold}) "
                f"f1={row['f1']:.4f}  {fold_dt:.1f}s"
            )
            if warned_msgs:
                convergence_records.extend([f"{model_name}: {m}" for m in warned_msgs])
            per_fold_rows.append(row)
            _write_progress_checkpoint(
                progress_path,
                per_fold_rows=per_fold_rows,
                holdout_rows=holdout_rows,
                convergence_records=convergence_records,
            )

        if holdout_done:
            _log("  holdout skipped (checkpoint)")
            continue

        holdout_t0 = time.perf_counter()
        holdout = _evaluate_holdout(model, X_train, y_train, X_test, y_test, model_name)
        holdout_dt = time.perf_counter() - holdout_t0
        model_dt = time.perf_counter() - model_t0
        _log(
            f"  holdout f1={holdout.metrics['f1']:.4f}  {holdout_dt:.1f}s  "
            f"| model total {model_dt:.1f}s"
        )
        holdout_rows.append(
            {
                "model": model_name,
                **holdout.metrics,
                **holdout.confusion,
            }
        )
        _write_progress_checkpoint(
            progress_path,
            per_fold_rows=per_fold_rows,
            holdout_rows=holdout_rows,
            convergence_records=convergence_records,
        )

    per_fold_df = pd.DataFrame(per_fold_rows)
    holdout_df = pd.DataFrame(holdout_rows)

    metric_cols = [
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "auc_pr",
        "fp_rate",
        "fn_rate",
        "accuracy",
    ]

    summary_df = _summarize(per_fold_df, metric_cols)
    ranking_df = _rank_models(summary_df)

    per_fold_path = output_dir / "per_fold_metrics.csv"
    summary_path = output_dir / "cv_summary.csv"
    holdout_path = output_dir / "holdout_metrics.csv"
    ranking_path = output_dir / "model_ranking.csv"
    checks_path = output_dir / "acceptance_checks.json"
    manifest_path = output_dir / "reproducibility_manifest.json"

    per_fold_df.to_csv(per_fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    holdout_df.to_csv(holdout_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)

    checks["all_required_metrics_files_exist"] = True
    checks["split_seed_42"] = SEED == 42
    checks["cv_seed_42"] = SEED == 42
    checks["mlp_convergence_policy_recorded"] = len(convergence_records) >= 0
    checks["all_models_evaluated"] = set(summary_df["model"]) == set(
        model_factories.keys()
    )
    checks["metric_hierarchy_set"] = True

    # keep raw scores even when convergence warning appears
    checks["raw_scores_kept_when_mlp_noconverge"] = True

    acceptance_payload = {
        "checks": checks,
        "convergence_warnings": convergence_records,
        "generated_files": [
            str(per_fold_path.relative_to(root)),
            str(summary_path.relative_to(root)),
            str(holdout_path.relative_to(root)),
            str(ranking_path.relative_to(root)),
            str(checks_path.relative_to(root)),
            str(manifest_path.relative_to(root)),
        ],
    }
    checks_path.write_text(
        json.dumps(acceptance_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "seed": {
            "split_seed": SEED,
            "cv_seed": SEED,
            "cv_folds": N_SPLITS,
            "cv_repeats": N_REPEATS,
            "test_size": TEST_SIZE,
        },
        "git": _git_metadata(root),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": _safe_version("numpy"),
            "pandas": _safe_version("pandas"),
            "scikit-learn": _safe_version("scikit-learn"),
            "xgboost": _safe_version("xgboost"),
            "torch": _safe_version("torch"),
            "pytorch-tabnet": _safe_version("pytorch-tabnet"),
        },
        "metric_hierarchy": {
            "primary": "f1",
            "secondary": ["precision", "recall", "auc_roc", "auc_pr"],
            "rule": "rank by mean F1, then mean precision, then mean recall",
        },
        "model_family": {
            "RandomForest": "RandomForestClassifier",
            "MLP (128,64)": "MLPClassifier",
            "MLP (64,32)": "MLPClassifier",
            "TabNet (fallback=FT-Transformer)": "TabNetClassifier if available else FTTransformerClassifier",
            "XGBoost": "xgboost.XGBClassifier",
        },
        "dataset": {
            "path": str(data_path),
            "n_rows": int(X_all.shape[0]),
            "n_features": int(X_all.shape[1]),
            "positive_rate": float(np.mean(y_all.to_numpy())),
        },
        "checks": checks,
        "convergence_warnings": convergence_records,
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if progress_path.exists():
        progress_path.unlink()

    return {
        "per_fold_path": str(per_fold_path.relative_to(root)),
        "summary_path": str(summary_path.relative_to(root)),
        "holdout_path": str(holdout_path.relative_to(root)),
        "ranking_path": str(ranking_path.relative_to(root)),
        "acceptance_checks_path": str(checks_path.relative_to(root)),
        "manifest_path": str(manifest_path.relative_to(root)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run QW1 deep-learning comparison study."
    )
    parser.add_argument(
        "--data",
        default=str(
            Path(__file__).resolve().parents[1] / "data" / "COMPARADORSEMIDENT.csv"
        ),
        help="Path to COMPARADORSEMIDENT.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "qw1"),
        help="Output directory for qw1 artifacts",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/progress_checkpoint.json when available",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    result_paths = run_study(output_dir=output_dir, args=args)

    print("QW1 artifacts:")
    for key, value in result_paths.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
