"""Regenerate orphan thesis artifacts (tables/figures) from data CSVs.

Ensures all latex artifacts are reproducible from data, replacing manual/orphan files.
"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Configure Matplotlib for PGF (LaTeX)
matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (6.0, 4.0),
    }
)

DATA_DIR = Path("data")
TABLES_DIR = Path("text/latex/tables")
FIG_DOC_DIR = Path("text/latex/fig_doc")


def generate_tab_feature_importance():
    """Regenerate tab_feature_importance.tex from feature_importance_global.csv"""
    src = DATA_DIR / "feature_importance_global.csv"
    dst = TABLES_DIR / "tab_feature_importance.tex"

    if not src.exists():
        print(f"[SKIP] {src} not found")
        return

    df = pd.read_csv(src)
    # Take top 15
    top15 = df.head(15).copy()

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Importância dos 15 atributos mais relevantes (RF+SMOTE, Gini)}"
    )
    lines.append(r"\label{tab:feature-importance}")
    lines.append(r"\begin{tabular}{clc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{\#} & \textbf{Atributo} & \textbf{Importância} \\")
    lines.append(r"\midrule")

    for i, row in top15.iterrows():
        name = str(row["feature"]).replace("_", r"\_")
        val = row["importance"]
        lines.append(f"  {i + 1} & {name} & {val:.4f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    dst.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Generated {dst}")


def generate_tab_impacto_epidemiologico():
    """Regenerate tab_impacto_epidemiologico.tex from epidemiological_impact.csv"""
    src = DATA_DIR / "epidemiological_impact.csv"
    dst = TABLES_DIR / "tab_impacto_epidemiologico.tex"

    if not src.exists():
        print(f"[SKIP] {src} not found")
        return

    df = pd.read_csv(src)

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

    for _, row in df.iterrows():
        # Translate method names to Portuguese/LaTeX
        m = str(row["method"])
        m = m.replace("Naive threshold >=", r"Limiar ingênuo $\geq$")
        m = m.replace("Rules >=", r"Regras $\geq$")
        m = m.replace("ML RF+SMOTE >=", r"ML RF+SMOTE $\geq$")
        m = m.replace("ML GB >=", r"ML GB $\geq$")
        m = m.replace("Hybrid-OR RF>=", r"Híb.-OR RF$\geq$")
        m = m.replace("+Rules>=", r"+R$\geq$")
        m = m.replace("Hybrid-AND RF>=", r"Híb.-AND RF$\geq$")

        det = int(row["TP"]) if "TP" in row else int(row["detected"])
        perd = int(row["FN"]) if "FN" in row else int(row["missed"])
        rev = int(row["to_review"])
        prec = row["precision"]
        rec = row["recall"]
        pct = (det / (det + perd)) * 100 if (det + perd) > 0 else 0

        lines.append(
            f"{m} & {det} & {perd} & {rev} & {prec:.3f} & {rec:.3f} & {pct:.1f}\\% \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(
        r"\multicolumn{7}{@{}l}{\footnotesize Det.=Detectados; Perd.=Perdidos; Rev.=Revisões; Prec.=Precisão; Verd.=Verdadeiros.}"
    )
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    dst.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Generated {dst}")


def generate_fig_shap_summary():
    """Regenerate shap_summary.pgf from shap_importance.csv"""
    src = DATA_DIR / "shap_importance.csv"
    dst = FIG_DOC_DIR / "shap_summary.pgf"

    if not src.exists():
        print(f"[SKIP] {src} not found")
        return

    df = pd.read_csv(src)
    top20 = df.head(20).copy()
    top20 = top20.sort_values(
        "shap_mean_abs", ascending=True
    )  # Sort for horiz bar plot

    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    # Clean feature names
    names = top20["feature"].apply(lambda x: str(x).replace("_", " "))

    ax.barh(
        names, top20["shap_mean_abs"], color="#1f77b4", edgecolor="black", linewidth=0.5
    )
    ax.set_xlabel(r"Impacto médio no modelo ($|$SHAP value$|$)")
    ax.set_title("Importância global dos atributos (SHAP)")

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(dst)
    plt.close()
    print(f"[OK] Generated {dst}")


def generate_fig_ablation():
    """Regenerate ablation figures from ablation_results.csv"""
    src = DATA_DIR / "ablation_results.csv"
    dst_bar = FIG_DOC_DIR / "ablation_barplot.pgf"
    dst_best = FIG_DOC_DIR / "ablation_best_category.pgf"

    if not src.exists():
        print(f"[SKIP] {src} not found")
        return

    df = pd.read_csv(src)

    # 1. Ablation Best Category (Grouped Bar)
    # Group by category, take max F1
    cats = df.groupby("category")["f1"].max().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # Map category names to display names
    cat_map = {
        "naive-threshold": "Limiar Ingênuo",
        "rules-only": "Regras Determinísticas",
        "consensus-ml": "Consenso (ML)",
        "consensus-hybrid": "Consenso (Híbrido)",
        "cascade-rules-ml": "Cascata Regras -> ML",
        "cascade-ml-rules": "Cascata ML -> Regras",
        "hybrid-and": "Híbrido AND (Interseção)",
        "hybrid-or": "Híbrido OR (União)",
        "ml-only": "ML Supervisionado (RF+SMOTE)",
    }

    display_names = [cat_map.get(c, c) for c in cats.index]

    bars = ax.barh(
        display_names, cats.values, color="#ff7f0e", edgecolor="black", linewidth=0.5
    )

    # Highlight the best one (ML-only)
    bars[-1].set_color("#2ca02c")
    bars[-1].set_edgecolor("black")

    ax.set_xlabel(r"F1-Score máximo")
    ax.set_xlim(0, 1.05)

    # Add values
    for rect in bars:
        w = rect.get_width()
        ax.text(
            w + 0.01,
            rect.get_y() + rect.get_height() / 2,
            f"{w:.3f}",
            ha="left",
            va="center",
            fontsize=8,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(dst_best)
    plt.close()
    print(f"[OK] Generated {dst_best}")


def main():
    print("Regenerating orphan artifacts...")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DOC_DIR.mkdir(parents=True, exist_ok=True)

    generate_tab_feature_importance()
    generate_tab_impacto_epidemiologico()
    generate_fig_shap_summary()
    generate_fig_ablation()

    print("Done.")


if __name__ == "__main__":
    main()
