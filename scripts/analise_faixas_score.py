"""
Análise de desempenho de ML por faixa de score OpenRecLink.
Reproduz feature engineering do NB01, treina RF+SMOTE global,
e avalia predições segmentadas por faixa de score.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# ── 1. Load data ──────────────────────────────────────────────
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "COMPARADORSEMIDENT.csv"
df = pd.read_csv(DATA_PATH, sep=";", low_memory=False)


# Clean column names (strip suffix like ',C,12,0')
def clean_col(c: str) -> str:
    return c.split(",")[0].strip()


df.columns = [clean_col(c) for c in df.columns]

# Convert numeric columns (comma decimal)
for col in df.columns:
    if df[col].dtype == object:
        try:
            df[col] = df[col].str.replace(",", ".").astype(float)
        except (ValueError, AttributeError):
            pass

# ── 2. Target ─────────────────────────────────────────────────
df["target"] = (df["PAR"].isin([1, 2])).astype(int)

# ── 3. Feature engineering (same as NB01) ─────────────────────
nome_cols = [
    c for c in df.columns if c.startswith("NOME") and "MAE" not in c and c != "NOME"
]
if not nome_cols:
    nome_cols = [
        "NOME prim frag igual",
        "NOME ult frag igual",
        "NOME qtd frag iguais",
        "NOME qtd frag raros",
        "NOME qtd frag comuns",
        "NOME qtd frag muito parec",
        "NOME qtd frag abrev",
    ]

mae_cols = [c for c in df.columns if c.startswith("NOMEMAE")]
if not mae_cols:
    mae_cols = [
        "NOMEMAE prim frag igual",
        "NOMEMAE ult frag igual",
        "NOMEMAE qtd frag iguais",
        "NOMEMAE qtd frag raros",
        "NOMEMAE qtd frag comuns",
        "NOMEMAE qtd frag muito parec",
        "NOMEMAE qtd frag abrev",
    ]

dtnasc_cols = [
    c
    for c in df.columns
    if c.startswith("DTNASC")
    and not c.startswith("DTNASC2")
    and c not in ("R_DTNASC", "C_DTNASC", "R_DTNASC2", "C_DTNASC2")
]
if not dtnasc_cols:
    dtnasc_cols = [
        "DTNASC dt iguais",
        "DTNASC ap 1 digi",
        "DTNASC inv dia",
        "DTNASC inv mes",
        "DTNASC inv ano",
    ]

mun_cols = [c for c in df.columns if c.startswith("CODMUNRES")]
if not mun_cols:
    mun_cols = ["CODMUNRES uf igual", "CODMUNRES local igual", "CODMUNRES local prox"]

end_cols = [c for c in df.columns if c.startswith("ENDERECO")]
if not end_cols:
    end_cols = [
        "ENDERECO via igual",
        "ENDERECO via prox",
        "ENDERECO numero igual",
        "ENDERECO compl prox",
        "ENDERECO texto prox",
    ]

# Identify actual matching-score columns
base_feature_cols = []
for col_list in [nome_cols, mae_cols, dtnasc_cols, mun_cols, end_cols]:
    for c in col_list:
        if c in df.columns:
            base_feature_cols.append(c)

# Add nota final if present
if "nota final" in df.columns:
    base_feature_cols.append("nota final")

# Ensure numeric
for c in base_feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# Engineered features
actual_nome = [c for c in nome_cols if c in df.columns]
actual_mae = [c for c in mae_cols if c in df.columns]
actual_dt = [c for c in dtnasc_cols if c in df.columns]
actual_end = [c for c in end_cols if c in df.columns]

df["nome_score_total"] = df[actual_nome].sum(axis=1) if actual_nome else 0
df["mae_score_total"] = df[actual_mae].sum(axis=1) if actual_mae else 0
df["dtnasc_score_total"] = df[actual_dt].sum(axis=1) if actual_dt else 0
df["endereco_score_total"] = df[actual_end].sum(axis=1) if actual_end else 0
df["nome_x_dtnasc"] = df["nome_score_total"] * df["dtnasc_score_total"]
df["nome_x_mae"] = df["nome_score_total"] * df["mae_score_total"]
df["nome_x_endereco"] = df["nome_score_total"] * df["endereco_score_total"]

max_nome = df["nome_score_total"].max() if df["nome_score_total"].max() > 0 else 1
df["nome_perfeito"] = (df["nome_score_total"] / max_nome >= 0.95).astype(int)

max_dt = df["dtnasc_score_total"].max() if df["dtnasc_score_total"].max() > 0 else 1
df["dtnasc_perfeito"] = (df["dtnasc_score_total"] / max_dt >= 1.0).astype(int)

df["mae_presente"] = (df["mae_score_total"] > 0).astype(int)
df["endereco_match"] = (df["endereco_score_total"] > 0).astype(int)

# C_SITUENCE for óbito flag
if "C_SITUENCE" in df.columns:
    df["C_SITUENCE"] = pd.to_numeric(df["C_SITUENCE"], errors="coerce").fillna(0)
    df["obito_sinan"] = df["C_SITUENCE"].isin([3, 4]).astype(int)
else:
    df["obito_sinan"] = 0

eng_cols = [
    "nome_score_total",
    "mae_score_total",
    "dtnasc_score_total",
    "endereco_score_total",
    "nome_x_dtnasc",
    "nome_x_mae",
    "nome_x_endereco",
    "nome_perfeito",
    "dtnasc_perfeito",
    "mae_presente",
    "endereco_match",
    "obito_sinan",
]

feature_cols = base_feature_cols + eng_cols
X = df[feature_cols].fillna(0)
y = df["target"]

print(f"Dataset: {len(df)} registros, {y.sum()} pares ({y.mean() * 100:.2f}%)")
print(
    f"Features: {len(feature_cols)} ({len(base_feature_cols)} base + {len(eng_cols)} eng)"
)

# ── 4. Train/test split (same as NB01) ───────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Keep nota final for band analysis
nota_train = X_train["nota final"].values if "nota final" in X_train.columns else None
nota_test = X_test["nota final"].values if "nota final" in X_test.columns else None

print(
    f"Train: {len(X_train)} ({y_train.sum()} pares) | Test: {len(X_test)} ({y_test.sum()} pares)"
)

# ── 5. Train models ──────────────────────────────────────────
# RF+SMOTE (best from NB01)
smote = SMOTE(k_neighbors=3, random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
rf_smote = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_smote.fit(X_res, y_res)

# Gradient Boosting (second best)
gb = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
)
gb.fit(X_train, y_train)

# ── 6. Predict on test set ───────────────────────────────────
y_pred_rf = rf_smote.predict(X_test)
y_prob_rf = rf_smote.predict_proba(X_test)[:, 1]

y_pred_gb = gb.predict(X_test)
y_prob_gb = gb.predict_proba(X_test)[:, 1]

# ── 7. Global results ────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTADOS GLOBAIS (teste)")
print("=" * 70)
for name, y_p in [("RF+SMOTE", y_pred_rf), ("Gradient Boosting", y_pred_gb)]:
    p = precision_score(y_test, y_p, zero_division=0)
    r = recall_score(y_test, y_p, zero_division=0)
    f = f1_score(y_test, y_p, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_p).ravel()
    print(
        f"{name:25s}  Prec={p:.3f}  Rec={r:.3f}  F1={f:.3f}  TP={tp} FP={fp} FN={fn} TN={tn}"
    )

# ── 8. Analysis by score band ────────────────────────────────
BANDS = [
    (0, 3, "0-3"),
    (3, 5, "3-5"),
    (5, 6, "5-6"),
    (6, 7, "6-7"),
    (7, 8, "7-8"),
    (8, 9, "8-9"),
    (9, 10, "9-10"),
    (10, float("inf"), "10+"),
]

print("\n" + "=" * 70)
print("ANÁLISE POR FAIXA DE SCORE OpenRecLink (conjunto de teste)")
print("=" * 70)

header = f"{'Faixa':>6s} | {'N':>6s} | {'Pares':>5s} | {'%Par':>6s} | {'Modelo':>20s} | {'TP':>3s} {'FP':>3s} {'FN':>3s} {'TN':>5s} | {'Prec':>5s} {'Rec':>5s} {'F1':>5s}"
print(header)
print("-" * len(header))

results = []

for lo, hi, label in BANDS:
    mask = (nota_test >= lo) & (nota_test < hi)
    n = mask.sum()
    if n == 0:
        continue

    y_band = y_test.values[mask]
    n_pairs = y_band.sum()
    pct_pairs = n_pairs / n * 100 if n > 0 else 0

    for model_name, y_p in [("RF+SMOTE", y_pred_rf), ("GradientBoosting", y_pred_gb)]:
        y_p_band = y_p[mask]

        if len(np.unique(y_band)) < 2 and n_pairs == 0:
            # All negatives
            fp = (y_p_band == 1).sum()
            tn = (y_p_band == 0).sum()
            tp = fn = 0
            p = 0.0 if fp == 0 else 0.0
            r = 0.0
            f = 0.0
        elif len(np.unique(y_band)) < 2 and n_pairs == n:
            # All positives
            tp = (y_p_band == 1).sum()
            fn = (y_p_band == 0).sum()
            fp = tn = 0
            p = 1.0 if tp > 0 else 0.0
            r = tp / n if n > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        else:
            tn, fp, fn, tp = confusion_matrix(y_band, y_p_band).ravel()
            p = precision_score(y_band, y_p_band, zero_division=0)
            r = recall_score(y_band, y_p_band, zero_division=0)
            f = f1_score(y_band, y_p_band, zero_division=0)

        print(
            f"{label:>6s} | {n:>6d} | {n_pairs:>5d} | {pct_pairs:>5.1f}% | {model_name:>20s} | {tp:>3d} {fp:>3d} {fn:>3d} {tn:>5d} | {p:>5.3f} {r:>5.3f} {f:>5.3f}"
        )
        results.append(
            {
                "faixa": label,
                "n": n,
                "pares": n_pairs,
                "pct_pares": pct_pairs,
                "modelo": model_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": p,
                "recall": r,
                "f1": f,
            }
        )

# ── 9. Naive threshold comparison ────────────────────────────
print("\n" + "=" * 70)
print("COMPARAÇÃO: Limiar fixo vs ML (conjunto de teste)")
print("=" * 70)

for threshold in [7.0, 8.0, 9.0]:
    y_naive = (nota_test >= threshold).astype(int)
    p = precision_score(y_test, y_naive, zero_division=0)
    r = recall_score(y_test, y_naive, zero_division=0)
    f = f1_score(y_test, y_naive, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_naive).ravel()
    print(
        f"Score >= {threshold:.0f}           Prec={p:.3f}  Rec={r:.3f}  F1={f:.3f}  TP={tp} FP={fp} FN={fn} TN={tn}"
    )

print(
    f"{'RF+SMOTE (ML)':25s}  Prec={precision_score(y_test, y_pred_rf, zero_division=0):.3f}  Rec={recall_score(y_test, y_pred_rf, zero_division=0):.3f}  F1={f1_score(y_test, y_pred_rf, zero_division=0):.3f}"
)
print(
    f"{'GradBoosting (ML)':25s}  Prec={precision_score(y_test, y_pred_gb, zero_division=0):.3f}  Rec={recall_score(y_test, y_pred_gb, zero_division=0):.3f}  F1={f1_score(y_test, y_pred_gb, zero_division=0):.3f}"
)

# ── 10. Grey zone deep dive ──────────────────────────────────
print("\n" + "=" * 70)
print("ZONA CINZENTA (score 5-8): onde ML mais agrega valor")
print("=" * 70)

mask_grey = (nota_test >= 5) & (nota_test < 8)
n_grey = mask_grey.sum()
y_grey = y_test.values[mask_grey]
n_pairs_grey = y_grey.sum()

print(
    f"Registros na zona cinzenta: {n_grey} ({n_pairs_grey} pares, {n_pairs_grey / n_grey * 100:.1f}%)"
)

for model_name, y_p in [("RF+SMOTE", y_pred_rf), ("GradientBoosting", y_pred_gb)]:
    y_p_grey = y_p[mask_grey]
    if len(np.unique(y_grey)) >= 2:
        tn, fp, fn, tp = confusion_matrix(y_grey, y_p_grey).ravel()
        p = precision_score(y_grey, y_p_grey, zero_division=0)
        r = recall_score(y_grey, y_p_grey, zero_division=0)
        f = f1_score(y_grey, y_p_grey, zero_division=0)
        print(
            f"{model_name:25s}  Prec={p:.3f}  Rec={r:.3f}  F1={f:.3f}  TP={tp} FP={fp} FN={fn} TN={tn}"
        )

# Naive in grey zone
for threshold in [6.0, 7.0]:
    y_naive_grey = (nota_test[mask_grey] >= threshold).astype(int)
    if len(np.unique(y_grey)) >= 2:
        p = precision_score(y_grey, y_naive_grey, zero_division=0)
        r = recall_score(y_grey, y_naive_grey, zero_division=0)
        f = f1_score(y_grey, y_naive_grey, zero_division=0)
        print(
            f"Score >= {threshold:.0f} (zona cinz)    Prec={p:.3f}  Rec={r:.3f}  F1={f:.3f}"
        )

# ── 11. Save results ─────────────────────────────────────────
results_df = pd.DataFrame(results)
out_path = (
    Path(__file__).resolve().parent.parent / "data" / "resultados_faixas_score.csv"
)
results_df.to_csv(out_path, index=False, sep=";")
print(f"\nResultados salvos em: {out_path}")
