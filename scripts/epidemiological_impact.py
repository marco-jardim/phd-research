"""
Epidemiological Impact Analysis (58 uniform features)
======================================================

Compares 3 main methods:
  1. Naive threshold (nota final >= 8)
  2. ML-only best (RF+SMOTE >= 0.5)
  3. Hybrid-OR best (RF+SMOTE >= 0.7 OR Rules >= 8)

Metrics: detected/missed true pairs, manual reviews needed, cost per death

Outputs:
  - data/epidemiological_impact.csv
  - data/epidemiological_summary.json
"""

from __future__ import annotations

import json
import warnings
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Shared feature engineering
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import load_data, engineer_features, split_data

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Rules engine (same as ablation)
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


# ---------------------------------------------------------------------------
# Clinical field extraction
# ---------------------------------------------------------------------------
def parse_date_yyyymmdd(val: object):
    if pd.isna(val):
        return None
    s = str(int(val)).strip()
    if len(s) == 8:
        try:
            return pd.Timestamp(datetime.strptime(s, "%Y%m%d"))
        except ValueError:
            return None
    return None


def parse_date_ddmmyyyy(val: object):
    if pd.isna(val):
        return None
    s = str(val).strip().split(".")[0]
    s = s.zfill(8)
    try:
        return pd.Timestamp(datetime.strptime(s, "%d%m%Y"))
    except ValueError:
        return None


def extract_clinical(df: pd.DataFrame) -> pd.DataFrame:
    clin = pd.DataFrame(index=df.index)

    if "R_DTNASC" in df.columns:
        clin["dt_nasc"] = df["R_DTNASC"].apply(parse_date_yyyymmdd)
    else:
        clin["dt_nasc"] = pd.NaT

    if "R_DTOBITO" in df.columns:
        clin["dt_obito"] = df["R_DTOBITO"].apply(parse_date_ddmmyyyy)
    else:
        clin["dt_obito"] = pd.NaT

    dt_nasc = pd.to_datetime(clin["dt_nasc"], errors="coerce")
    dt_obito = pd.to_datetime(clin["dt_obito"], errors="coerce")
    clin["idade_obito"] = (dt_obito - dt_nasc).dt.days / 365.25

    clin["sexo"] = (
        df["R_SEXO"].astype(str).str.strip().str.upper()
        if "R_SEXO" in df.columns
        else "?"
    )
    clin["bairro"] = (
        df["R_BAIRES"].astype(str).str.strip().str.upper()
        if "R_BAIRES" in df.columns
        else "?"
    )

    if "C_DTNOTI" in df.columns:
        clin["dt_notificacao"] = df["C_DTNOTI"].apply(parse_date_ddmmyyyy)
        dt_noti = pd.to_datetime(clin["dt_notificacao"], errors="coerce")
        clin["dias_noti_obito"] = (dt_obito - dt_noti).dt.days
    else:
        clin["dias_noti_obito"] = np.nan

    clin["nota_final"] = df["nota final"] if "nota final" in df.columns else 0.0
    clin["passo"] = df["PASSO"] if "PASSO" in df.columns else 0
    clin["par"] = df["PAR"] if "PAR" in df.columns else 0

    return clin


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("EPIDEMIOLOGICAL IMPACT ANALYSIS (58 uniform features)")
    print("=" * 70)

    # Load data with shared module
    df = load_data()
    X, y = engineer_features(df)
    clin = extract_clinical(df)

    # Same split as ablation/notebooks
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Align clinical data with test set
    test_idx = X_test.index
    clin_test = clin.loc[test_idx].copy()

    print(f"\nDataset: {len(df)} records, {y.sum()} true pairs")
    print(f"Features: {X.shape[1]} (58 uniform)")
    print(f"Test set: {len(X_test)} records, {y_test.sum()} true pairs")

    # ── Train models ──
    print("\n-- Training models --")

    # RF+SMOTE
    smote = SMOTE(sampling_strategy=0.3, k_neighbors=3, random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_res, y_res)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    # GB
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )
    gb.fit(X_train, y_train)
    gb_proba = gb.predict_proba(X_test)[:, 1]

    # Rules score
    rules_score = X_test.apply(regras_alta_confianca, axis=1)

    # nota final from X_test
    nota_test = X_test["nota final"]

    # ── Define methods ──
    methods = {
        "Naive threshold >=7": (nota_test >= 7).astype(int),
        "Naive threshold >=8": (nota_test >= 8).astype(int),
        "Naive threshold >=9": (nota_test >= 9).astype(int),
        "Rules >=7": (rules_score >= 7).astype(int),
        "Rules >=8": (rules_score >= 8).astype(int),
        "ML RF+SMOTE >=0.5": (rf_proba >= 0.5).astype(int),
        "ML RF+SMOTE >=0.7": (rf_proba >= 0.7).astype(int),
        "ML GB >=0.5": (gb_proba >= 0.5).astype(int),
        "Hybrid-OR RF>=0.7+Rules>=8": ((rf_proba >= 0.7) | (rules_score >= 8)).astype(
            int
        ),
        "Hybrid-AND RF>=0.5+Rules>=7": ((rf_proba >= 0.5) & (rules_score >= 7)).astype(
            int
        ),
    }

    total_test = len(X_test)
    true_deaths = int(y_test.sum())

    print(f"\n{'=' * 70}")
    print("CORRECTED MORTALITY RATE")
    print(f"{'=' * 70}")

    results = []
    for name, preds in methods.items():
        preds_arr = np.array(preds)
        y_arr = np.array(y_test)
        tp = int(((preds_arr == 1) & (y_arr == 1)).sum())
        fp = int(((preds_arr == 1) & (y_arr == 0)).sum())
        fn = int(((preds_arr == 0) & (y_arr == 1)).sum())
        tn = int(((preds_arr == 0) & (y_arr == 0)).sum())
        detected = tp
        missed = fn
        to_review = tp + fp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        rate_detected = detected / total_test * 100
        rate_true = true_deaths / total_test * 100
        pct_found = round(detected / true_deaths * 100, 1) if true_deaths > 0 else 0

        results.append(
            {
                "method": name,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "detected": detected,
                "missed": missed,
                "to_review": to_review,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1_val, 4),
                "rate_detected_pct": round(rate_detected, 4),
                "rate_true_pct": round(rate_true, 4),
                "pct_of_true_found": pct_found,
                "cost_per_death": round(to_review / detected, 2)
                if detected > 0
                else float("inf"),
            }
        )

    results_df = pd.DataFrame(results)

    print(
        f"\n{'Method':<35} {'Det':>5} {'Missed':>7} {'Review':>7} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Cost/D':>7} {'%Found':>7}"
    )
    print("-" * 95)
    for r in results:
        print(
            f"{r['method']:<35} {r['detected']:>5} {r['missed']:>7} {r['to_review']:>7} "
            f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} "
            f"{r['cost_per_death']:>7.1f} {r['pct_of_true_found']:>6.1f}%"
        )

    # ── Profile of recovered deaths ──
    print(f"\n{'=' * 70}")
    print("PROFILE OF DEATHS RECOVERED BY ML")
    print(f"{'=' * 70}")

    ml_preds = (rf_proba >= 0.5).astype(int)
    naive_preds = (nota_test >= 8).astype(int).values

    ml_finds = (ml_preds == 1) & (np.array(y_test) == 1)
    naive_finds = (naive_preds == 1) & (np.array(y_test) == 1)
    recovered = ml_finds & ~naive_finds

    n_recovered = int(recovered.sum())
    print(f"\n  True deaths found by ML but missed by naive>=8: {n_recovered}")
    print(f"  (out of {true_deaths} true deaths in test set)")

    if n_recovered > 0:
        clin_recovered = clin_test[recovered].copy()
        clin_found = clin_test[ml_finds & naive_finds].copy()

        age_rec = clin_recovered["idade_obito"].dropna()
        if len(age_rec) > 0:
            print(
                f"\n  Age at death (recovered): mean={age_rec.mean():.1f}, "
                f"median={age_rec.median():.1f}"
            )
        age_found = clin_found["idade_obito"].dropna()
        if len(age_found) > 0:
            print(
                f"  Age at death (already found): mean={age_found.mean():.1f}, "
                f"median={age_found.median():.1f}"
            )

        sex_rec = clin_recovered["sexo"].value_counts()
        print(f"\n  Sex (recovered): {dict(sex_rec)}")

        nota_rec = clin_recovered["nota_final"]
        print(
            f"\n  Score (recovered): mean={nota_rec.mean():.2f}, "
            f"median={nota_rec.median():.2f}"
        )
        for low, high in [(5, 6), (6, 7), (7, 8)]:
            n_band = int(((nota_rec >= low) & (nota_rec < high)).sum())
            pct = n_band / n_recovered * 100 if n_recovered > 0 else 0
            print(f"    Band {low}-{high}: {n_band} ({pct:.1f}%)")

    # ── Save outputs ──
    results_df.to_csv(OUT_DIR / "epidemiological_impact.csv", index=False)
    print(f"\n-> Saved: data/epidemiological_impact.csv")

    # Summary JSON
    summary = {
        "total_test": total_test,
        "true_pairs_test": true_deaths,
        "methods": {},
    }
    for r in results:
        summary["methods"][r["method"]] = {
            "detected": r["detected"],
            "missed": r["missed"],
            "to_review": r["to_review"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "cost_per_death": r["cost_per_death"],
            "pct_of_true_found": r["pct_of_true_found"],
        }
    summary["recovered_by_ml_vs_naive8"] = n_recovered
    summary["pct_recovered"] = (
        round(n_recovered / true_deaths * 100, 1) if true_deaths > 0 else 0
    )

    with open(OUT_DIR / "epidemiological_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"-> Saved: data/epidemiological_summary.json")

    print(f"\n{'=' * 70}")
    print("EPIDEMIOLOGICAL IMPACT ANALYSIS COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
