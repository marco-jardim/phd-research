from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
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


def pct_exato(num: int, den: int, casas: int = 1) -> str:
    """Percentual a partir das contagens, meia unidade para cima, vírgula decimal.

    Evita o arredondamento duplo (valor de 4 casas ou de 2 casas arredondado de novo).
    """
    q = (Decimal(num) * 100 / Decimal(den)).quantize(
        Decimal(1).scaleb(-casas), rounding=ROUND_HALF_UP
    )
    return f"{q:.{casas}f}".replace(".", ",")


def rotulo_pct(num: int, den: int) -> str:
    """Rótulo sobre a barra: '<0,1' quando há pares verdadeiros mas o arredondamento daria 0,0."""
    texto = pct_exato(num, den)
    if num > 0 and texto == "0,0":
        return "$<$0,1"
    return texto


def parse_score_bands() -> tuple[list[str], list[int], list[int], list[float]]:
    """Lê as contagens por faixa de data/tab_score_bands_proveniencia.csv (band, n, vinculos)."""
    frame = pd.read_csv(DATA / "tab_score_bands_proveniencia.csv")
    if frame.empty:
        raise RuntimeError("tab_score_bands_proveniencia.csv vazio")
    bands = [str(b).replace("-", "--") for b in frame["band"]]
    totals = [int(n) for n in frame["n"]]
    pairs = [int(v) for v in frame["vinculos"]]
    pcts = [100.0 * p / t for p, t in zip(pairs, totals)]
    return bands, totals, pairs, pcts


POS_TESTE_NB01 = 74


def fracao_3casas(num: int, den: int) -> str:
    """Fração com 3 casas, meia unidade para cima, vírgula decimal."""
    q = (Decimal(num) / Decimal(den)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return f"{q}".replace(".", ",")


def contagens_nb01(precisao: float, sensibilidade: float) -> tuple[int, int]:
    """Reconstrói (VP, positivos preditos) do NB01 a partir das métricas de 4 casas.

    Falha se as contagens não reproduzirem os valores do artefato.
    """
    vp = round(sensibilidade * POS_TESTE_NB01)
    n_pred = round(vp / precisao)
    if round(vp / POS_TESTE_NB01, 4) != round(sensibilidade, 4) or round(vp / n_pred, 4) != round(precisao, 4):
        raise ValueError(f"Contagens inconsistentes com o artefato: VP={vp}, N={n_pred}")
    return vp, n_pred


def rotulo_faixa(banda: str) -> str:
    """'10+' é escrito como ≥10, como no texto e em tab_desempenho_faixas."""
    return r"$\geq$10" if banda == "10+" else banda


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
        label="Total de potenciais pares",
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
    ax.set_xticklabels([rotulo_faixa(b) for b in bands], fontsize=9)
    ax.set_xlabel("Nota final do comparador")
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
    # A faixa intermediária da nota final é [5,9): barras de índice 2 a 5 (5--6, 6--7, 7--8, 8--9).
    # A cor segue a faixa, e não um limiar de proporção.
    inicio, fim = bands.index("5--6"), bands.index("8--9")
    colors = [
        "#aec7e8"
        if p == 0
        else ("#d62728" if inicio <= i <= fim else "#ff9896")
        for i, p in enumerate(pairs)
    ]
    ax.bar(x, pcts, 0.6, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([rotulo_faixa(b) for b in bands], fontsize=9)
    ax.set_xlabel("Nota final do comparador")
    ax.set_ylabel(r"Pares verdadeiros / total (\%)")

    for i, (p, t) in enumerate(zip(pairs, totals)):
        if p > 0:
            ax.text(
                i,
                pcts[i] + 1.3,
                rotulo_pct(p, t) + "\\%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Sombreado de [5,9): de meia unidade antes da barra 5--6 a meia unidade depois da barra 8--9.
    ax.axvspan(inicio - 0.5, fim + 0.5, alpha=0.08, color="gray")
    ax.text(
        (inicio + fim) / 2,
        max(pcts) * 0.86,
        "Faixa intermediária",
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
        "Logistic Regression": "Regressão\nlogística",
        "Stacking Ensemble": "Empilhamento",
    }

    df = df.set_index("Modelo").loc[order].reset_index()
    models = [label_map[m] for m in df["Modelo"]]
    # O artefato guarda as métricas com 4 casas. VP, positivos preditos e F1 são reconstruídos
    # das contagens (74 pares verdadeiros no teste) para não arredondar duas vezes.
    precision, recall, f1 = [], [], []
    for _, row in df.iterrows():
        vp, n_pred = contagens_nb01(row["Precision"], row["Recall"])
        precision.append((vp, n_pred))
        recall.append((vp, POS_TESTE_NB01))
        f1.append((2 * vp, n_pred + POS_TESTE_NB01))

    x = np.arange(len(models))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_p = ax.bar(
        x - w, [a / b for a, b in precision], w, label="VPP", color="#2166ac", alpha=0.85
    )
    bars_r = ax.bar(
        x, [a / b for a, b in recall], w, label="Sensibilidade", color="#b2182b", alpha=0.85
    )
    bars_f = ax.bar(x + w, [a / b for a, b in f1], w, label=r"$F_1$", color="#4daf4a", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Valor da métrica")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right", fontsize=9)

    for bars, fracoes in ((bars_p, precision), (bars_r, recall), (bars_f, f1)):
        for bar, (num, den) in zip(bars, fracoes):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                fracao_3casas(num, den),
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

    pat = re.compile(r"^([^&]+?)\s*&\s*[^&]*&\s*([0-9.,]+)\s*&\s*[^&]*&\s*(?:\\textbf\{)?([0-9.,]+)(?:\})?")
    for line in lines:
        m = pat.search(line)
        if not m:
            continue
        categories.append(m.group(1).strip())
        f1_rf.append(float(m.group(2).replace(",", ".")))

    if len(categories) < 5:
        raise RuntimeError(
            "Could not parse RF values from tab_ablation_best_category.tex"
        )

    label_map = {
        r"Decis\~ao de limiar": "decisão de limiar",
        "Somente regras": "somente regras",
        "Somente ML": "somente ML",
        r"H\'ibrido AND": "híbrido AND",
        r"H\'ibrido OR": "híbrido OR",
        r"Cascata ML$\rightarrow$Regras": "cascata ML$\\rightarrow$regras",
        r"Cascata Regras$\rightarrow$ML": "cascata regras$\\rightarrow$ML",
        "Consenso ML": "consenso ML",
        "Consenso+Regras": "consenso + regras",
    }
    labels = [label_map.get(c, c) for c in categories]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, f1_rf, color="#4daf4a", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"$F_1$ (melhor configuração)")
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
