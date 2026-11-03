"""Observed terminal branches, v3 versus v3.1, without an embedded title.

python scripts/funil_gzcmd_v2.py --reuse-oof
python scripts/fig_cascata_v31.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STAGES = ("MATCH automático", "NONMATCH automático", "LLM_MATCH", "LLM_NONMATCH", "HUMANO_TOTAL")
LABELS = ("MATCH\nautomático", "NONMATCH\nautomático", "LLM:\nMATCH", "LLM:\nNONMATCH", "Revisor\nhumano")


def terminal_counts(table, mode, version):
    part = table[table.modo.eq(mode) & table.versao.eq(version)]
    cells = part[part.tipo_linha.eq("cascata")]
    branches = []
    for stage in STAGES:
        if stage == "MATCH automático":
            subset = cells[cells.nivel.eq(2) & cells.etapa.isin(["ALWAYS_MATCH", "AUTO_MATCH"])]
        elif stage == "NONMATCH automático":
            subset = cells[cells.nivel.eq(2) & cells.etapa.isin(["ALWAYS_NONMATCH", "AUTO_NONMATCH"])]
        elif stage == "HUMANO_TOTAL":
            subset = cells[cells.nivel.eq(4) & cells.etapa.eq(stage)]
        else:
            subset = cells[cells.nivel.eq(3) & cells.etapa.eq(stage)]
        branches.append({"etapa": stage, "n": int(subset.n.sum()), "vinculos": int(subset.vinculos.sum())})
    result = pd.DataFrame(branches)
    total = cells[cells.nivel.eq(0)].iloc[0]
    if result.n.sum() != total.n or result.vinculos.sum() != total.vinculos:
        raise ValueError("Terminal branches must partition the complete cohort")
    return result


def render(table, output):
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
             "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(10., 7.7), sharex=True)
        fig.subplots_adjust(left=.09, right=.98, top=.91, bottom=.14, hspace=.30, wspace=.25)
        for col, mode in enumerate(("vigilancia", "confirmacao")):
            name = "Vigilância" if mode == "vigilancia" else "Confirmação"
            for version, color, offset in (("v3", "#3b82a0", -.18), ("v3.1", "#166534", .18)):
                data = terminal_counts(table, mode, version)
                for row, field in ((0, "n"), (1, "vinculos")):
                    ax = axes[row, col]
                    x = np.arange(len(data)) + offset
                    bars = ax.bar(x, data[field], width=.34, color=color, edgecolor="white", lw=.5, label=version, zorder=3)
                    for bar, value in zip(bars, data[field]):
                        # Stagger long count labels so adjacent 60k/61k bars do not overlap.
                        height = max(value, .8) * (1.12 if version == "v3" else 1.65) if row == 0 else value + 3
                        ax.text(bar.get_x() + bar.get_width() / 2, height,
                                f"{value:,}".replace(",", "."), ha="center", va="bottom", fontsize=7)
            top, bottom = axes[:, col]
            top.set_yscale("log")
            top.set_ylim(.65, 300_000)
            top.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
            top.set_ylabel(f"{name}\nPotenciais pares (escala log)")
            bottom.set_ylim(0, 275)
            bottom.set_ylabel(f"{name}\nVínculos verdadeiros (n)")
            for ax in (top, bottom):
                ax.set_xticks(np.arange(len(LABELS)), LABELS)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", which="major", lw=.35, alpha=.4, zorder=0)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.text(.53, .025, "Ramos terminais disjuntos; revisor humano = encaminhamento direto + inconclusivos.\nClaude Haiku 5.5, prompt v4, dossiê sem saídas do motor.",
                 ha="center", fontsize=8)
        output.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "png"):
            path = output.with_suffix("." + extension)
            fig.savefig(path, bbox_inches="tight", dpi=300)
            print(f"FIGURA CASCATA {path}: {path.stat().st_size} bytes")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data/funil_gzcmd_final.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "text/latex/figures/fig_cap8_cascata_v31")
    args = parser.parse_args()
    table = pd.read_csv(args.csv)
    render(table, args.output)


if __name__ == "__main__":
    main()
