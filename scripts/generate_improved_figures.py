"""Generate improved figures for thesis: 5.1 split, 5.4 Pareto, 5.5 CV, 6.1 split."""

import matplotlib

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
    }
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "text/latex/fig_doc"

# ============================================================
# 1) FIG 6.1 split into 3 separate figures
# ============================================================
methods_short = [
    "Limiar\n($\\geq$8)",
    "ML-only\nRF+SMOTE",
    "Hybrid-OR\nRF+Rules",
]
detected = [44, 73, 69]
missed = [29, 0, 4]
reviews = [308, 80, 95]
cost = [7.0, 1.1, 1.4]
x3 = np.arange(3)

# 6.1a - Deaths detected vs missed
fig, ax = plt.subplots(figsize=(4.8, 3.0))
w = 0.32
b1 = ax.bar(
    x3 - w / 2,
    detected,
    w,
    label="Detectados",
    color="#2ca02c",
    edgecolor="black",
    linewidth=0.4,
)
b2 = ax.bar(
    x3 + w / 2,
    missed,
    w,
    label="Perdidos",
    color="#d62728",
    edgecolor="black",
    linewidth=0.4,
)
ax.set_xticks(x3)
ax.set_xticklabels(methods_short, fontsize=9)
ax.set_ylabel(r"N. de pares verdadeiros")
ax.legend(fontsize=9, framealpha=0.9)
for bar in b1:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        str(int(bar.get_height())),
        ha="center",
        fontsize=9,
    )
for bar in b2:
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        max(h, 0) + 1,
        str(int(h)),
        ha="center",
        fontsize=9,
    )
ax.set_ylim(0, 85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUT}/epi_deteccao.pgf")
plt.close()
print("OK: epi_deteccao.pgf")

# 6.1b - Manual reviews
fig, ax = plt.subplots(figsize=(4.8, 3.0))
colors3 = ["#ff7f0e", "#2ca02c", "#17becf"]
b3 = ax.bar(x3, reviews, 0.45, color=colors3, edgecolor="black", linewidth=0.4)
ax.set_xticks(x3)
ax.set_xticklabels(methods_short, fontsize=9)
ax.set_ylabel(r"Total de pares para revis\~ao")
for bar in b3:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        str(int(bar.get_height())),
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
ax.set_ylim(0, 360)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUT}/epi_revisoes.pgf")
plt.close()
print("OK: epi_revisoes.pgf")

# 6.1c - Cost per recovered death
fig, ax = plt.subplots(figsize=(4.8, 3.0))
b4 = ax.bar(x3, cost, 0.45, color=colors3, edgecolor="black", linewidth=0.4)
ax.set_xticks(x3)
ax.set_xticklabels(methods_short, fontsize=9)
ax.set_ylabel(r"Revis\~oes por par verdadeiro")
for bar in b4:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.12,
        f"{bar.get_height():.1f}",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )
ax.set_ylim(0, 8.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUT}/epi_custo.pgf")
plt.close()
print("OK: epi_custo.pgf")

# ============================================================
# 2) FIG 5.1 split into 2 figures
# ============================================================
bands = ["0--3", "3--5", "5--6", "6--7", "7--8", "8--9", "9--10", "10+"]
total = [1607, 10316, 4336, 1846, 337, 36, 15, 16]
pairs = [0, 0, 3, 13, 15, 12, 15, 16]
pct_pairs = [100.0 * p / t if t > 0 else 0 for p, t in zip(pairs, total)]

xb = np.arange(len(bands))

# 5.1a - Volume (log scale)
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.bar(
    xb,
    total,
    0.6,
    color="#1f77b4",
    edgecolor="black",
    linewidth=0.3,
    label="Total de pares candidatos",
)
ax.bar(
    xb,
    pairs,
    0.6,
    color="#d62728",
    edgecolor="black",
    linewidth=0.3,
    label="Pares verdadeiros",
)
ax.set_xticks(xb)
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
            p * 1.5,
            str(p),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#d62728",
            fontweight="bold",
        )
plt.tight_layout()
fig.savefig(f"{OUT}/score_band_volume.pgf")
plt.close()
print("OK: score_band_volume.pgf")

# 5.1b - Percentage of true pairs per band + grey zone
fig, ax = plt.subplots(figsize=(5.5, 3.2))
colors_pct = []
for p, pct in zip(pairs, pct_pairs):
    if p == 0:
        colors_pct.append("#aec7e8")
    elif pct < 5:
        colors_pct.append("#ff9896")
    else:
        colors_pct.append("#d62728")
bars = ax.bar(xb, pct_pairs, 0.6, color=colors_pct, edgecolor="black", linewidth=0.3)
ax.set_xticks(xb)
ax.set_xticklabels(bands, fontsize=9)
ax.set_xlabel("Faixa do escore agregado")
ax.set_ylabel(r"Pares verdadeiros / total (\%)")
for i, (pct, p) in enumerate(zip(pct_pairs, pairs)):
    if pct > 0:
        ax.text(i, pct + 1.5, f"{pct:.1f}\\%", ha="center", va="bottom", fontsize=8)
# grey zone shading
ax.axvspan(1.5, 4.5, alpha=0.08, color="gray")
ax.text(
    3,
    max(pct_pairs) * 0.85,
    r"\textit{Zona cinzenta}",
    ha="center",
    fontsize=8,
    color="gray",
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUT}/score_band_pct.pgf")
plt.close()
print("OK: score_band_pct.pgf")

# ============================================================
# 3) FIG 5.5 improved cv_boxplots (horizontal bar + error bars)
# ============================================================
cv = pd.read_csv("data/cv_5fold_results.csv")
configs = cv["config"].tolist()
f1_mean = cv["f1_mean"].tolist()
f1_std = cv["f1_std"].tolist()

order = np.argsort(f1_mean)[::-1]
configs_s = [configs[i] for i in order]
f1_m = [f1_mean[i] for i in order]
f1_s = [f1_std[i] for i in order]

label_map = {}
for c in configs:
    if "Naive" in c:
        label_map[c] = r"Limiar ($\geq$8)"
    elif c.startswith("Rules"):
        label_map[c] = r"Regras ($\geq$6)"
    elif "Hybrid-AND" in c and "RF" in c:
        label_map[c] = r"Hybrid-AND RF+Rules"
    elif "Hybrid-AND" in c and "GB" in c:
        label_map[c] = r"Hybrid-AND GB+Rules"
    elif "Hybrid-OR" in c:
        label_map[c] = r"Hybrid-OR RF+Rules"
    elif "RF" in c:
        label_map[c] = r"RF+SMOTE ($\geq$0.5)"
    elif "GB" in c:
        label_map[c] = r"GB ($\geq$0.5)"
    else:
        label_map[c] = c
labels = [label_map.get(c, c) for c in configs_s]

fig, ax = plt.subplots(figsize=(5.8, 3.5))
y = np.arange(len(labels))
colors_cv = []
for c in configs_s:
    if "Naive" in c:
        colors_cv.append("#d62728")
    elif "Rules" in c and "Hybrid" not in c:
        colors_cv.append("#ff7f0e")
    elif "Hybrid-OR" in c:
        colors_cv.append("#17becf")
    elif "Hybrid-AND" in c:
        colors_cv.append("#1f77b4")
    elif "RF" in c:
        colors_cv.append("#2ca02c")
    elif "GB" in c:
        colors_cv.append("#9467bd")
    else:
        colors_cv.append("#7f7f7f")

ax.barh(
    y,
    f1_m,
    xerr=f1_s,
    height=0.55,
    color=colors_cv,
    edgecolor="black",
    linewidth=0.3,
    capsize=3,
    error_kw={"linewidth": 1.0},
)
for i, (m, s) in enumerate(zip(f1_m, f1_s)):
    ax.text(m + s + 0.008, i, f"${m:.3f} \\pm {s:.3f}$", va="center", fontsize=7.5)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel(r"$F_1$ (5-\textit{fold} CV)")
ax.set_xlim(0.4, 1.08)
ax.invert_yaxis()
ax.axvline(x=0.571, color="gray", linestyle=":", linewidth=0.6)
ax.text(0.48, len(labels) - 0.3, r"\textit{baseline}", fontsize=7, color="gray")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUT}/cv_boxplots_v2.pgf")
plt.close()
print("OK: cv_boxplots_v2.pgf")

# ============================================================
# 4) FIG 5.4 improved Pareto frontier
# ============================================================
pg = pd.read_csv("data/pareto_grid.csv")
rf = pg[pg["ml_model"] == "RF+SMOTE"].copy()
gb = pg[pg["ml_model"] == "GB"].copy()


def pareto_front(df):
    """Return Pareto-optimal rows (non-dominated in precision & recall)."""
    pts = df[["precision", "recall"]].values
    front = []
    for i in range(len(pts)):
        dominated = False
        for j in range(len(pts)):
            if i != j:
                if (
                    pts[j][0] >= pts[i][0]
                    and pts[j][1] >= pts[i][1]
                    and (pts[j][0] > pts[i][0] or pts[j][1] > pts[i][1])
                ):
                    dominated = True
                    break
        if not dominated:
            front.append(i)
    return df.iloc[front].sort_values("recall", ascending=False)


pf_rf = pareto_front(rf)
pf_gb = pareto_front(gb)

fig, ax = plt.subplots(figsize=(5.5, 4.0))
# scatter all points (low alpha)
ax.scatter(
    rf["precision"], rf["recall"], alpha=0.10, s=10, c="#1f77b4", label="_nolegend_"
)
ax.scatter(
    gb["precision"], gb["recall"], alpha=0.10, s=10, c="#ff7f0e", label="_nolegend_"
)
# Pareto front lines
ax.plot(
    pf_rf["precision"],
    pf_rf["recall"],
    "o-",
    color="#1f77b4",
    markersize=5,
    linewidth=1.5,
    label="RF+SMOTE (Pareto)",
    zorder=5,
)
ax.plot(
    pf_gb["precision"],
    pf_gb["recall"],
    "s-",
    color="#ff7f0e",
    markersize=5,
    linewidth=1.5,
    label="GB (Pareto)",
    zorder=5,
)

# Annotate key points
key_pts = [
    (0.923, 0.973, r"RF th=0.5" + "\n" + r"$F_1$=0.947", (-45, 5)),
    (0.957, 0.892, r"RF th=0.7" + "\n" + r"$F_1$=0.923", (-50, -18)),
]
for px, py, txt, ofs in key_pts:
    ax.annotate(
        txt,
        (px, py),
        textcoords="offset points",
        xytext=ofs,
        fontsize=7,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.9
        ),
    )

# Naive baseline
ax.scatter(
    [0.642], [0.581], marker="X", s=80, c="#d62728", zorder=6, label=r"Limiar $\geq$8"
)
ax.annotate(
    r"Limiar $\geq$8" + "\n" + r"$F_1$=0.610",
    (0.642, 0.581),
    textcoords="offset points",
    xytext=(-55, -20),
    fontsize=7,
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.7),
    bbox=dict(
        boxstyle="round,pad=0.2", facecolor="#fff0f0", edgecolor="#d62728", alpha=0.9
    ),
)

ax.set_xlabel(r"Precis\~ao")
ax.set_ylabel(r"Revoca\c{c}\~ao")
ax.set_xlim(0.4, 1.05)
ax.set_ylim(0.0, 1.05)
ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(f"{OUT}/pareto_frontier_v2.pgf")
plt.close()
print("OK: pareto_frontier_v2.pgf")

print("\nAll figures generated successfully!")
