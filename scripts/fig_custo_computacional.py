#!/usr/bin/env python3
"""Generate computational cost figure for GZ-CMD cascade (Figure 7.x).

Two panels:
  (a) Unit cost per pair (log scale) — 7 orders of magnitude
  (b) Total cost per stage (log scale) — volume reduction compensates unit cost
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────
stages = [
    "Limiares\nFellegi–Sunter",
    "Regras\nde guarda",
    "Platt +\nMotor de perda",
    "Revisão LLM\ndual-agent",
]
n_pairs = np.array([61_696, 21_620, 18_799, 41])
unit_s = np.array([80e-9, 700e-9, 3e-6, 5.0])  # seconds per pair
total_s = n_pairs * unit_s  # seconds per stage

# Colors matching cascade figure palette
colors = ["#d35400", "#e67e22", "#2980b9", "#27ae60"]

# ── Figure ────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel (a): Unit cost per pair
bars1 = ax1.bar(stages, unit_s, color=colors, edgecolor="white", linewidth=0.8)
ax1.set_yscale("log")
ax1.set_ylabel("Custo unitário por par (escala logarítmica)", fontsize=11)
ax1.set_title("(a) Custo unitário por par", fontsize=13, fontweight="bold", pad=12)

# Annotate with human-readable labels
unit_labels = ["80 ns", "700 ns", "3 μs", "~5 s"]
for bar, label in zip(bars1, unit_labels):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 2.5,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#333333",
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", edgecolor="#999999", alpha=0.9
        ),
    )

# Format y-axis
ax1.set_ylim(1e-9, 50)
ax1.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
ax1.grid(axis="y", alpha=0.3, linestyle="--")

# Brace / annotation for 7 orders of magnitude
ax1.annotate(
    "7 ordens\nde grandeza",
    xy=(0.08, 80e-9),
    xycoords=("axes fraction", "data"),
    xytext=(0.08, 0.05),
    textcoords=("axes fraction", "data"),
    fontsize=9,
    ha="center",
    va="top",
    color="#666666",
    arrowprops=dict(arrowstyle="<->", color="#666666", lw=1.5),
)

# Panel (b): Total cost per stage
bars2 = ax2.bar(stages, total_s, color=colors, edgecolor="white", linewidth=0.8)
ax2.set_yscale("log")
ax2.set_ylabel("Custo total da etapa (escala logarítmica)", fontsize=11)
ax2.set_title("(b) Custo total por etapa", fontsize=13, fontweight="bold", pad=12)

# Annotate with human-readable labels
total_labels = ["~5 ms", "~15 ms", "~50 ms", "~200 s"]
for bar, label in zip(bars2, total_labels):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 2.5,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#333333",
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", edgecolor="#999999", alpha=0.9
        ),
    )

# Add counterfactual annotation
ax2.axhline(y=61_696 * 5.0, color="#c0392b", linestyle="--", linewidth=1.5, alpha=0.7)
ax2.text(
    3.0,
    61_696 * 5.0 * 1.8,
    "Contrafactual: 61.696 × 5 s\n≈ 85 horas (~$100 USD)",
    ha="center",
    va="bottom",
    fontsize=9,
    color="#c0392b",
    bbox=dict(
        boxstyle="round,pad=0.3", facecolor="#fdecea", edgecolor="#c0392b", alpha=0.9
    ),
)

# Format y-axis
ax2.set_ylim(1e-3, 1e7)
ax2.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
ax2.grid(axis="y", alpha=0.3, linestyle="--")

# ── Layout ────────────────────────────────────────────────────────────
fig.tight_layout(w_pad=3)

out = (
    Path(__file__).resolve().parent.parent
    / "text"
    / "latex"
    / "figures"
    / "fig_custo_computacional.pdf"
)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight", dpi=300)
print(f"Saved: {out}  ({out.stat().st_size:,} bytes)")
