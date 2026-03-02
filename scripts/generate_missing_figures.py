"""Generate all missing figures and tables for Cap5-7.

Outputs:
  figures/score_band_distribution.pgf  - Histogram pairs vs non-pairs per score band
  figures/nb01_model_comparison.pgf     - Bar chart P/R/F1 for 7 NB01 models
  figures/imbalance_sensitivity.pgf     - Bar chart F1 per SMOTE strategy
  figures/cv_boxplots.pgf               - Box plot F1 per config across 5 folds
  figures/epidemiological_impact.pgf    - Bar chart deaths detected per method
  figures/feature_importance_heatmap.pgf - Heatmap features x score bands
  text/latex/tables/tab_score_bands.tex - LaTeX table score band distribution
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
TAB = ROOT / "text" / "latex" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# ── style (PGF-compatible) ─────────────────────────────────────────────
plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": "\n".join(
            [
                r"\usepackage[T1]{fontenc}",
                r"\usepackage[utf8]{inputenc}",
                r"\usepackage{lmodern}",
            ]
        ),
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)
COLORS = {
    "pair": "#2166ac",
    "nonpair": "#b2182b",
    "ml": "#2166ac",
    "hybrid": "#4daf4a",
    "rules": "#ff7f00",
    "naive": "#999999",
}


def _tex(s: str) -> str:
    """Escape special LaTeX characters for PGF backend."""
    return (
        s.replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. Score Band Distribution
# ═══════════════════════════════════════════════════════════════════════
def fig_score_bands() -> None:
    print("1/7  Score band distribution...", end=" ", flush=True)
    df = pd.read_csv(DATA / "COMPARADORSEMIDENT.csv", sep=";")
    # clean nota final
    if df["nota final"].dtype == object:
        df["nota final"] = (
            df["nota final"]
            .astype(str)
            .str.split(",")
            .str[0]
            .str.replace(",", ".", regex=False)
            .astype(float)
        )
    df["target"] = df["PAR"].isin([1, 2]).astype(int)

    edges = [0, 3, 5, 6, 7, 8, 9, 10, 15]
    labels = ["0-3", "3-5", "5-6", "6-7", "7-8", "8-9", "9-10", "10+"]
    df["band"] = pd.cut(df["nota final"], bins=edges, labels=labels, right=False)

    pairs = df[df["target"] == 1].groupby("band", observed=False).size()
    nonpairs = df[df["target"] == 0].groupby("band", observed=False).size()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(labels))
    w = 0.38
    ax1.bar(
        x - w / 2,
        nonpairs.values,
        w,
        color=COLORS["nonpair"],
        alpha=0.8,
        label="Não-pares",
    )
    ax1.bar(x + w / 2, pairs.values, w, color=COLORS["pair"], alpha=0.8, label="Pares")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_xlabel("Faixa de escore")
    ax1.set_ylabel("Frequência")
    ax1.set_title("(a) Contagem absoluta")
    ax1.set_yscale("log")
    ax1.legend(fontsize=8)

    total = pairs + nonpairs
    pct = (pairs / total * 100).fillna(0)
    ax2.bar(x, pct.values, color=COLORS["pair"], alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_xlabel("Faixa de escore")
    ax2.set_ylabel("% pares verdadeiros")
    ax2.set_title("(b) Proporção de pares por faixa")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())

    fig.suptitle(
        "Distribuição dos registros por faixa de escore do OpenRecLink",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG / "score_band_distribution.pgf")
    plt.close(fig)

    # ── LaTeX table ──
    rows = []
    for lab in labels:
        t = int(total.get(lab, 0))
        p = int(pairs.get(lab, 0))
        pctv = f"{pct.get(lab, 0):.2f}"
        rows.append(f"    {lab} & {t:,} & {p} & {pctv}\\% \\\\")

    tex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Distribuição dos pares candidatos por faixa de escore do OpenRecLink}\n"
        "\\label{tab:score-bands}\n"
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Faixa de escore & Total & Pares & \\% Pares \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    (TAB / "tab_score_bands.tex").write_text(tex, encoding="utf-8")
    print("OK")


# ═══════════════════════════════════════════════════════════════════════
# 2. NB01 Model Comparison
# ═══════════════════════════════════════════════════════════════════════
def fig_nb01_comparison() -> None:
    print("2/7  NB01 model comparison...", end=" ", flush=True)
    # Results from NB01 execution (threshold 0.5)
    models = [
        "RF+SMOTE",
        "Gradient\nBoosting",
        "MLP",
        "Random\nForest",
        "SVM (RBF)",
        "Regressão\nLogística",
        "Stacking",
    ]
    prec = [0.890, 0.944, 0.875, 0.779, 0.474, 0.333, 0.311]
    rec = [0.987, 0.919, 0.851, 0.905, 0.743, 0.987, 1.000]
    f1 = [0.936, 0.932, 0.863, 0.838, 0.579, 0.498, 0.474]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    w = 0.25

    bars_p = ax.bar(x - w, prec, w, label="Precisão", color="#2166ac", alpha=0.85)
    bars_r = ax.bar(x, rec, w, label="Recall", color="#b2182b", alpha=0.85)
    bars_f = ax.bar(x + w, f1, w, label="F1", color="#4daf4a", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Valor da métrica")
    ax.set_ylim(0, 1.12)
    ax.set_title("Comparação de técnicas de ML no limiar padrão (0,5)")
    ax.legend(loc="upper right", fontsize=9)

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(FIG / "nb01_model_comparison.pgf")
    plt.close(fig)
    print("OK")


# ═══════════════════════════════════════════════════════════════════════
# 3. Imbalance Sensitivity
# ═══════════════════════════════════════════════════════════════════════
def fig_imbalance() -> None:
    print("3/7  Imbalance sensitivity...", end=" ", flush=True)
    df = pd.read_csv(DATA / "imbalance_sensitivity.csv")
    df = df.sort_values("f1_mean", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(df))
    bars = ax.barh(
        y,
        df["f1_mean"],
        xerr=df["f1_std"],
        capsize=3,
        color="#4daf4a",
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([_tex(s) for s in df["strategy"]], fontsize=9)
    ax.set_xlabel("F1-score (média ± desvio, 5-fold CV)")
    ax.set_title("Sensibilidade ao método de balanceamento (RF, 200 árvores)")
    ax.set_xlim(0.75, 0.95)

    for i, (val, std) in enumerate(zip(df["f1_mean"], df["f1_std"])):
        ax.text(val + std + 0.003, i, f"{val:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG / "imbalance_sensitivity.pgf")
    plt.close(fig)
    print("OK")


# ═══════════════════════════════════════════════════════════════════════
# 4. CV Box Plots (regenerate fold-level data)
# ═══════════════════════════════════════════════════════════════════════
def fig_cv_boxplots() -> None:
    print("4/7  CV box plots (re-running 5-fold)...", end=" ", flush=True)

    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import SMOTE

    # Load data (same pipeline as ablation)
    csv = pd.read_csv(DATA / "COMPARADORSEMIDENT.csv", sep=";")
    for c in csv.columns:
        if csv[c].dtype == object:
            csv[c] = (
                csv[c]
                .astype(str)
                .str.split(",")
                .str[0]
                .str.replace(",", ".", regex=False)
            )
            try:
                csv[c] = pd.to_numeric(csv[c])
            except (ValueError, TypeError):
                pass
    csv["target"] = csv["PAR"].isin([1, 2]).astype(int)

    score_cols = [
        c
        for c in csv.columns
        if any(
            k in c
            for k in [
                "NOME",
                "NOMEMAE",
                "DTNASC",
                "CODMUNRES",
                "ENDERECO",
                "nota final",
            ]
        )
        and c not in ["PAR", "PASSO", "target"]
    ]

    feat = csv[score_cols].fillna(0).copy()
    # Engineered features
    feat["nome_score_total"] = (
        feat.get("NOME prim frag igual", 0)
        + feat.get("NOME ult frag igual", 0)
        + feat.get("NOME qtd frag iguais", 0)
    )
    feat["mae_score_total"] = (
        feat.get("NOMEMAE prim frag igual", 0)
        + feat.get("NOMEMAE ult frag igual", 0)
        + feat.get("NOMEMAE qtd frag iguais", 0)
    )
    feat["nome_perfeito"] = (feat.get("NOME qtd frag iguais", 0) >= 0.95).astype(int)
    feat["dtnasc_perfeito"] = (feat.get("DTNASC dt iguais", 0) == 1).astype(int)
    y = csv["target"]

    # Rules function
    def rules_score(row: pd.Series) -> float:
        s = 0.0
        nome_qtd = row.get("NOME qtd frag iguais", 0)
        if nome_qtd >= 0.95:
            s += 3
        elif nome_qtd >= 0.85:
            s += 2
        dt = row.get("DTNASC dt iguais", 0)
        if dt == 1:
            s += 3
        elif row.get("DTNASC dt ap 1digi", 0) == 1:
            s += 1.5
        mae = row.get("NOMEMAE qtd frag iguais", 0)
        if mae >= 0.7:
            s += 2
        elif mae >= 0.5:
            s += 1
        if row.get("CODMUNRES local igual", 0) == 1:
            s += 1.5
        if row.get("ENDERECO via igual", 0) == 1:
            s += 1
        nota = row.get("nota final", 0)
        if nota >= 9:
            s += 2
        elif nota >= 8:
            s += 1
        return s

    rules = csv[score_cols].fillna(0).apply(rules_score, axis=1)

    configs = {
        r"RF+SMOTE $\geq$0.5": {"model": "rf_smote", "ml_th": 0.5, "rules_th": None},
        r"GB $\geq$0.5": {"model": "gb", "ml_th": 0.5, "rules_th": None},
        "Hybrid-OR\n" + r"RF$\geq$0.7+R$\geq$8": {
            "model": "rf_smote",
            "ml_th": 0.7,
            "rules_th": 8,
            "mode": "or",
        },
        "Hybrid-AND\n" + r"RF$\geq$0.5+R$\geq$6": {
            "model": "rf_smote",
            "ml_th": 0.5,
            "rules_th": 6,
            "mode": "and",
        },
        "Hybrid-AND\n" + r"GB$\geq$0.6+R$\geq$5": {
            "model": "gb",
            "ml_th": 0.6,
            "rules_th": 5,
            "mode": "and",
        },
        r"Rules $\geq$6": {"model": None, "ml_th": None, "rules_th": 6},
        r"Naive $\geq$8": {"model": None, "ml_th": None, "rules_th": None, "naive": 8},
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results: dict[str, list[float]] = {k: [] for k in configs}

    X_arr = feat.values
    y_arr = y.values
    r_arr = rules.values
    nota_arr = csv["nota final"].fillna(0).values

    for train_idx, test_idx in skf.split(X_arr, y_arr):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        r_te = r_arr[test_idx]
        n_te = nota_arr[test_idx]

        # Train models
        smote = SMOTE(sampling_strategy=0.3, k_neighbors=3, random_state=42)
        X_sm, y_sm = smote.fit_resample(X_tr, y_tr)

        rf_sm = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf_sm.fit(X_sm, y_sm)
        rf_proba = rf_sm.predict_proba(X_te)[:, 1]

        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        gb.fit(X_tr, y_tr)
        gb_proba = gb.predict_proba(X_te)[:, 1]

        for name, cfg in configs.items():
            if "naive" in cfg:
                pred = (n_te >= cfg["naive"]).astype(int)
            elif cfg["model"] is None:
                pred = (r_te >= cfg["rules_th"]).astype(int)
            else:
                proba = rf_proba if "rf" in cfg["model"] else gb_proba
                ml_pred = proba >= cfg["ml_th"]
                if cfg["rules_th"] is None:
                    pred = ml_pred.astype(int)
                elif cfg.get("mode") == "or":
                    pred = (ml_pred | (r_te >= cfg["rules_th"])).astype(int)
                else:
                    pred = (ml_pred & (r_te >= cfg["rules_th"])).astype(int)

            fold_results[name].append(f1_score(y_te, pred, zero_division=0))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [fold_results[k] for k in configs]
    labels_list = list(configs.keys())

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color="black", linewidth=1.5),
    )

    cat_colors = [
        "#2166ac",
        "#2166ac",
        "#4daf4a",
        "#4daf4a",
        "#4daf4a",
        "#ff7f00",
        "#999999",
    ]
    for patch, color in zip(bp["boxes"], cat_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels_list, fontsize=8)
    ax.set_ylabel("F1-score")
    ax.set_title("Estabilidade das configurações (5-fold CV estratificado)")

    # Add mean markers
    means = [np.mean(d) for d in data]
    ax.scatter(
        range(1, len(means) + 1),
        means,
        marker="D",
        color="red",
        s=30,
        zorder=5,
        label="Média",
    )
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG / "cv_boxplots.pgf")
    plt.close(fig)
    print("OK")


# ═══════════════════════════════════════════════════════════════════════
# 5. Epidemiological Impact
# ═══════════════════════════════════════════════════════════════════════
def fig_epidemiological() -> None:
    print("5/7  Epidemiological impact...", end=" ", flush=True)
    df = pd.read_csv(DATA / "epidemiological_impact.csv")

    # Select key methods for clarity
    key_methods = [
        "Limiar >=7",
        "Limiar >=8",
        "Limiar >=9",
        "Rules >=6",
        "ML RF+SMOTE >=0.5",
        "ML GB >=0.5",
        "Hybrid-AND GB>=0.6+R>=5",
    ]
    df = df[df["method"].isin(key_methods)].copy()
    df["short"] = df["method"].replace(
        {
            "Limiar >=7": "Limiar\n" + r"$\geq$7",
            "Limiar >=8": "Limiar\n" + r"$\geq$8",
            "Limiar >=9": "Limiar\n" + r"$\geq$9",
            "Rules >=6": "Regras\n" + r"$\geq$6",
            "ML RF+SMOTE >=0.5": "RF+SMOTE\n" + r"$\geq$0.5",
            "ML GB >=0.5": "GB\n" + r"$\geq$0.5",
            "Hybrid-AND GB>=0.6+R>=5": r"H\'{i}brido" + "\n" + r"GB+R$\geq$5",
        }
    )

    cat_colors = [
        "#999999",
        "#999999",
        "#999999",
        "#ff7f00",
        "#2166ac",
        "#2166ac",
        "#4daf4a",
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Deaths detected + missed
    x = np.arange(len(df))
    detected = df["detected"].values
    missed = df["missed"].values

    ax1.bar(x, detected, color=cat_colors, alpha=0.85, label="Detectados")
    ax1.bar(
        x, missed, bottom=detected, color="#d9d9d9", alpha=0.7, label="Não detectados"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["short"], fontsize=8)
    ax1.set_ylabel("Número de óbitos")
    ax1.set_title("(a) Óbitos detectados vs. não detectados")
    ax1.legend(fontsize=8)
    ax1.axhline(y=74, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax1.text(len(x) - 0.5, 75, "Total: 74", fontsize=7, color="red", ha="right")

    for i, (d, m) in enumerate(zip(detected, missed)):
        ax1.text(
            i,
            d / 2,
            str(int(d)),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

    # (b) Cost per death (reviews/death)
    cost = df["to_review"].values / np.maximum(detected, 1)
    ax2.bar(x, cost, color=cat_colors, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["short"], fontsize=8)
    ax2.set_ylabel("Revisões por óbito detectado")
    ax2.set_title("(b) Custo operacional por óbito")

    for i, c in enumerate(cost):
        ax2.text(i, c + 0.1, f"{c:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Impacto epidemiológico: detecção de óbitos e custo operacional",
        fontsize=12,
        y=1.02,
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.25, top=0.90, wspace=0.35)
    fig.savefig(FIG / "epidemiological_impact.pgf")
    plt.close(fig)
    print("OK")


# ═══════════════════════════════════════════════════════════════════════
# 6. Feature Importance Heatmap per Score Band
# ═══════════════════════════════════════════════════════════════════════
def fig_feature_heatmap() -> None:
    print("6/7  Feature importance heatmap...", end=" ", flush=True)
    df = pd.read_csv(DATA / "feature_importance_per_band.csv")

    # Get top 12 features globally
    global_imp = df.groupby("feature")["importance"].mean().nlargest(12)
    top_features = global_imp.index.tolist()

    # Pivot
    sub = df[df["feature"].isin(top_features)]
    pivot = sub.pivot_table(
        index="feature", columns="band", values="importance", aggfunc="mean"
    )
    # Reorder — detect actual band names from CSV
    all_bands = sorted(pivot.columns.tolist())
    band_order_candidates = [
        ["baixo (0-5)", "cinza-baixo (5-6)", "cinza-médio (6-7)", "cinza-alto (7-8)"],
        ["baixo(0-5)", "cinza-inferior(5-7)", "cinza-superior(7-8)", "alto(8+)"],
    ]
    band_order = all_bands  # fallback
    for candidate in band_order_candidates:
        matched = [b for b in candidate if b in pivot.columns]
        if len(matched) >= 2:
            band_order = matched
            break
    pivot = pivot[band_order]
    available = [f for f in top_features[::-1] if f in pivot.index]
    pivot = pivot.loc[available]

    # Short labels
    short_bands = {
        "baixo (0-5)": "Baixo\n(0-5)",
        "cinza-baixo (5-6)": "Cinza baixo\n(5-6)",
        "cinza-médio (6-7)": "Cinza médio\n(6-7)",
        "cinza-alto (7-8)": "Cinza alto\n(7-8)",
        "baixo(0-5)": "Baixo\n(0-5)",
        "cinza-inferior(5-7)": "Cinza inf.\n(5-7)",
        "cinza-superior(7-8)": "Cinza sup.\n(7-8)",
        "alto(8+)": "Alto\n(8+)",
    }
    pivot.columns = [short_bands.get(c, c) for c in pivot.columns]

    # Short feature names
    short_feat = {
        "nota final": "nota final",
        "NOME qtd frag iguais": "nome qtd iguais",
        "nome_perfeito": "nome perfeito",
        "nome_score_total": "nome total",
        "NOMEMAE qtd frag iguais": "mae qtd iguais",
        "mae_score_total": "mae total",
        "DTNASC dt iguais": "dtnasc iguais",
        "dtnasc_perfeito": "dtnasc perfeito",
        "NOME prim frag igual": "nome prim igual",
        "NOME ult frag igual": "nome ult igual",
        "CODMUNRES local igual": "mun local igual",
        "ENDERECO via igual": "end via igual",
    }
    pivot.index = [_tex(short_feat.get(f, f)) for f in pivot.index]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val > pivot.values.max() * 0.6 else "black"
            ax.text(
                j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color
            )

    ax.set_title("Importância dos atributos por faixa de escore")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Importância (Gini)")
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance_heatmap.pgf")
    plt.close(fig)
    print("OK")


# ═══════════════════════════════════════════════════════════════════════
# 7. tab_score_bands.tex is generated inside fig_score_bands()
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    print("=" * 60)
    print("Generating missing figures and tables")
    print("=" * 60)

    fig_score_bands()  # 1
    fig_nb01_comparison()  # 2
    fig_imbalance()  # 3
    fig_cv_boxplots()  # 4
    fig_epidemiological()  # 5
    fig_feature_heatmap()  # 6

    print("=" * 60)
    print("All done!")
    print(f"Figures: {list(FIG.glob('*.pgf'))}")
    print(f"Tables:  {list(TAB.glob('tab_score_bands*'))}")


if __name__ == "__main__":
    main()
