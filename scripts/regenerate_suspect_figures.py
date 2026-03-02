from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "text" / "latex" / "tables"
FIG_DOC = ROOT / "text" / "latex" / "fig_doc"
DATA = ROOT / "data"

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)


def parse_score_bands() -> tuple[list[str], list[int], list[int], list[float]]:
    path = TABLES / "tab_score_bands.tex"
    lines = path.read_text(encoding="utf-8").splitlines()
    bands: list[str] = []
    totals: list[int] = []
    pairs: list[int] = []
    pcts: list[float] = []

    pat = re.compile(
        r"\s*([0-9+\-]+)\s*&\s*([0-9,]+)\s*&\s*([0-9]+)\s*&\s*([0-9.]+)\\%"
    )

    for line in lines:
        m = pat.search(line)
        if not m:
            continue
        bands.append(m.group(1).replace("-", "--"))
        totals.append(int(m.group(2).replace(",", "")))
        pairs.append(int(m.group(3)))
        pcts.append(float(m.group(4)))

    if not bands:
        raise RuntimeError("Could not parse tab_score_bands.tex")
    return bands, totals, pairs, pcts


def regenerate_score_band_figures() -> None:
    bands, totals, pairs, pcts = parse_score_bands()
    x = np.arange(len(bands))

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(
        x,
        totals,
        0.6,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.3,
        label="Total de pares candidatos",
    )
    ax.bar(
        x,
        pairs,
        0.6,
        color="#d62728",
        edgecolor="black",
        linewidth=0.3,
        label="Pares verdadeiros",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=9)
    ax.set_xlabel("Faixa do escore agregado")
    ax.set_ylabel(r"Quantidade (escala $\log_{10}$)")
    ax.set_yscale("log")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, p in enumerate(pairs):
        if p > 0:
            ax.text(
                i,
                p * 1.25,
                str(p),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#d62728",
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(FIG_DOC / "score_band_volume.pgf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    colors = [
        "#aec7e8" if p == 0 else ("#ff9896" if pc < 5 else "#d62728")
        for p, pc in zip(pairs, pcts)
    ]
    ax.bar(x, pcts, 0.6, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=9)
    ax.set_xlabel("Faixa do escore agregado")
    ax.set_ylabel(r"Pares verdadeiros / total (\%)")

    for i, pc in enumerate(pcts):
        if pc > 0:
            ax.text(i, pc + 1.3, f"{pc:.1f}\\%", ha="center", va="bottom", fontsize=8)

    ax.axvspan(1.5, 4.5, alpha=0.08, color="gray")
    ax.text(
        3,
        max(pcts) * 0.86,
        r"\textit{Zona cinzenta}",
        ha="center",
        fontsize=8,
        color="gray",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DOC / "score_band_pct.pgf")
    plt.close(fig)


def regenerate_nb01_comparison() -> None:
    df = pd.read_csv(DATA / "nb01_comparacao_modelos.csv", sep=";")
    order = [
        "RF + SMOTE",
        "Gradient Boosting",
        "MLP Neural Network",
        "Random Forest",
        "SVM (RBF)",
        "Logistic Regression",
        "Stacking Ensemble",
    ]
    label_map = {
        "RF + SMOTE": "RF+SMOTE",
        "Gradient Boosting": "Gradient\nBoosting",
        "MLP Neural Network": "MLP",
        "Random Forest": "Random\nForest",
        "SVM (RBF)": "SVM (RBF)",
        "Logistic Regression": "Regressao\nLogistica",
        "Stacking Ensemble": "Stacking",
    }

    df = df.set_index("Modelo").loc[order].reset_index()
    models = [label_map[m] for m in df["Modelo"]]
    precision = df["Precision"].tolist()
    recall = df["Recall"].tolist()
    f1 = df["F1-Score"].tolist()

    x = np.arange(len(models))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_p = ax.bar(x - w, precision, w, label="Precisao", color="#2166ac", alpha=0.85)
    bars_r = ax.bar(x, recall, w, label="Sensibilidade", color="#b2182b", alpha=0.85)
    bars_f = ax.bar(x + w, f1, w, label="F1", color="#4daf4a", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Valor da metrica")
    ax.set_ylim(0, 1.12)
    ax.set_title("Comparacao de tecnicas de ML no limiar padrao (0,5)")
    ax.legend(loc="upper right", fontsize=9)

    for bars in (bars_p, bars_r, bars_f):
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
    fig.savefig(FIG_DOC / "nb01_model_comparison.pgf")
    plt.close(fig)


def regenerate_ablation_best_category() -> None:
    lines = (
        (TABLES / "tab_ablation_best_category.tex")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    categories: list[str] = []
    f1_rf: list[float] = []

    pat = re.compile(r"^([^&]+?)\s*&\s*[^&]*&\s*([0-9.]+)\s*&\s*[^&]*&\s*(?:\\textbf\{)?([0-9.]+)(?:\})?")
    for line in lines:
        m = pat.search(line)
        if not m:
            continue
        categories.append(m.group(1).strip())
        f1_rf.append(float(m.group(2)))

    if len(categories) < 5:
        raise RuntimeError(
            "Could not parse RF values from tab_ablation_best_category.tex"
        )

    label_map = {
        r"Limiar ing\^enuo": "naive-threshold",
        "Somente regras": "rules-only",
        "Somente ML": "ml-only",
        r"H\'ibrido AND": "hybrid-and",
        r"H\'ibrido OR": "hybrid-or",
        r"Cascata ML$\rightarrow$Rules": "cascade-ml-rules",
        r"Cascata Rules$\rightarrow$ML": "cascade-rules-ml",
        "Consenso ML": "consensus-ml",
        "Consenso+Rules": "consensus-hybrid",
    }
    labels = [label_map.get(c, c) for c in categories]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, f1_rf, color="#4daf4a", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Escore F1 (melhor configuracao)")
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
    fig.tight_layout()
    fig.savefig(FIG_DOC / "ablation_best_category.pgf")
    plt.close(fig)


def main() -> None:
    FIG_DOC.mkdir(parents=True, exist_ok=True)
    regenerate_score_band_figures()
    regenerate_nb01_comparison()
    regenerate_ablation_best_category()
    print("OK regenerated 4 suspicious figures")


if __name__ == "__main__":
    main()
