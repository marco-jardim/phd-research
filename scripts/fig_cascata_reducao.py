#!/usr/bin/env python3
"""Generate fig_cascata_reducao.pdf — cascade reduction + prevalence enrichment.

Changes vs. original:
  (a) Arrows removed; log scale stated in y-axis label; percentage labels
      placed directly above each bar.
  (b) Annotation "18× enriquecimento" repositioned above bars to avoid
      overlap; legend moved to upper-left corner away from data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------- data -----------------------------------------------------------
stages = [
    "Sem modelo\n(baseline)",
    "Limiares\nFellegi–Sunter",
    "+ Regras\nde guarda",
    "+ Calibração +\nMotor de perda",
    "+ Revisão LLM\ndual-agent",
]
pairs = [61_696, 21_620, 18_799, 1_410, 41]
tp = [247, 151, 151, 104, 16]
prev = [tp_i / p_i * 100 for tp_i, p_i in zip(tp, pairs)]

pct_reductions = [None]  # first bar has no predecessor
for i in range(1, len(pairs)):
    pct_reductions.append((pairs[i - 1] - pairs[i]) / pairs[i - 1] * 100)

# ---------- colours --------------------------------------------------------
bar_colors_a = ["#d45500", "#c44e00", "#3b82f6", "#3b82f6", "#166534"]
bar_colors_b_prev = "#3b82f6"
bar_colors_b_tp = "#e8742a"

# ---------- figure ---------------------------------------------------------
fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(9.0, 8.5), gridspec_kw={"hspace": 0.42})

# ===== subplot (a): cascade reduction ======================================
x = np.arange(len(stages))
bars_a = ax_a.bar(
    x, pairs, color=bar_colors_a, width=0.60, edgecolor="white", linewidth=0.5, zorder=3
)

ax_a.set_yscale("log")
ax_a.set_ylim(8, 200_000)
ax_a.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "."))
)
ax_a.set_ylabel("Pares encaminhados a revisão\n(escala logarítmica)", fontsize=10)
ax_a.set_xticks(x)
ax_a.set_xticklabels(stages, fontsize=8.5)
ax_a.set_title(
    "(a) Redução cumulativa da carga de revisão clerical",
    fontsize=11.5,
    fontweight="bold",
    loc="left",
    pad=10,
)
ax_a.grid(axis="y", which="major", linewidth=0.4, alpha=0.5, zorder=0)
ax_a.grid(axis="y", which="minor", linewidth=0.2, alpha=0.3, zorder=0)

# value labels + percentage labels above bars
for i, (bar, val, pct) in enumerate(zip(bars_a, pairs, pct_reductions)):
    # value on top
    ax_a.text(
        bar.get_x() + bar.get_width() / 2,
        val * 1.25,
        f"{val:,}".replace(",", "."),
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
    # percentage reduction BETWEEN bars (skip first bar)
    if pct is not None:
        # position midway between current bar and previous bar
        x_mid = (x[i - 1] + x[i]) / 2
        y_mid = max(pairs[i - 1], val) * 0.55
        ax_a.text(
            x_mid,
            y_mid,
            f"−{pct:.1f}%".replace(".", ","),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#333333",
            bbox=dict(
                boxstyle="round,pad=0.2", fc="white", ec="#aaaaaa", alpha=0.85, lw=0.5
            ),
        )

# "Redução total" annotation
ax_a.annotate(
    "Redução total: 99,8%\n(3 ordens de grandeza)",
    xy=(4, 41),
    xytext=(2.6, 25),
    fontsize=8.5,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.4", fc="#e0f7e9", ec="#166534", alpha=0.9),
    arrowprops=dict(arrowstyle="-|>", color="#166534", lw=1.2),
    zorder=5,
)

ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# ===== subplot (b): prevalence enrichment ==================================
width = 0.30
x_b = np.arange(len(stages))

ax_b2 = ax_b.twinx()

# blue bars — prevalence (left axis)
bars_prev = ax_b.bar(
    x_b - width / 2,
    prev,
    width,
    color=bar_colors_b_prev,
    label="Prevalência TP (%)",
    zorder=3,
    alpha=0.85,
)
# orange bars — TP count (right axis)
bars_tp = ax_b2.bar(
    x_b + width / 2,
    tp,
    width,
    color=bar_colors_b_tp,
    label="TP contidos (n)",
    zorder=3,
    alpha=0.85,
)

# left axis
ax_b.set_ylabel(
    "Prevalência de pares\nverdadeiros (%)", fontsize=10, color=bar_colors_b_prev
)
ax_b.set_ylim(0, 55)
ax_b.tick_params(axis="y", labelcolor=bar_colors_b_prev)

# right axis
ax_b2.set_ylabel("Pares verdadeiros\ncontidos (n)", fontsize=10, color=bar_colors_b_tp)
ax_b2.set_ylim(0, 300)
ax_b2.tick_params(axis="y", labelcolor=bar_colors_b_tp)

ax_b.set_xticks(x_b)
ax_b.set_xticklabels(stages, fontsize=8.5)
ax_b.set_title(
    "(b) Enriquecimento progressivo da prevalência",
    fontsize=11.5,
    fontweight="bold",
    loc="left",
    pad=10,
)

# prevalence labels above blue bars
for bar, v in zip(bars_prev, prev):
    ax_b.text(
        bar.get_x() + bar.get_width() / 2,
        v + 1.2,
        f"{v:.1f}%".replace(".", ","),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=bar_colors_b_prev,
        fontweight="bold",
    )

# TP count labels above orange bars
for bar, v in zip(bars_tp, tp):
    ax_b2.text(
        bar.get_x() + bar.get_width() / 2,
        v + 5,
        str(v),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#c44e00",
        fontweight="bold",
    )

# annotation — repositioned to upper-left to avoid overlap
ax_b.annotate(
    "18× enriquecimento\n(0,4% → 7,4%)",
    xy=(3 - width / 2, prev[3]),
    xytext=(0.8, 42),
    fontsize=8.5,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.4", fc="#dbeafe", ec="#3b82f6", alpha=0.9),
    arrowprops=dict(arrowstyle="-|>", color="#3b82f6", lw=1.2),
    zorder=5,
)

# combined legend — upper-right, inside, away from data
lines_a, labels_a = ax_b.get_legend_handles_labels()
lines_b, labels_b = ax_b2.get_legend_handles_labels()
ax_b2.legend(
    lines_a + lines_b,
    labels_a + labels_b,
    loc="upper right",
    framealpha=0.9,
    fontsize=8.5,
    edgecolor="#cccccc",
)

ax_b.spines["top"].set_visible(False)
ax_b2.spines["top"].set_visible(False)
ax_b.grid(axis="y", linewidth=0.3, alpha=0.4, zorder=0)

# ---------- save -----------------------------------------------------------
out = (
    Path(__file__).resolve().parent.parent
    / "text"
    / "latex"
    / "figures"
    / "fig_cascata_reducao.pdf"
)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight", dpi=300)
print(f"Saved → {out}  ({out.stat().st_size / 1024:.0f} KB)")
plt.close(fig)
