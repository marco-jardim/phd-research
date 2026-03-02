"""Generate LaTeX tables from ablation study results for Cap5."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path(__file__).resolve().parent.parent / "text" / "latex" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────
abl = pd.read_csv(DATA / "ablation_results.csv")
pareto = pd.read_csv(DATA / "pareto_grid.csv")

# ── Table 1: Best per category ───────────────────────────────────────
CATEGORY_ORDER = [
    "naive-threshold",
    "rules-only",
    "ml-only",
    "hybrid-and",
    "hybrid-or",
    "cascade-ml-rules",
    "cascade-rules-ml",
    "consensus-ml",
    "consensus-hybrid",
]

CATEGORY_LABELS = {
    "naive-threshold": "Limiar ingênuo (escore $\\geq t$)",
    "rules-only": "Somente regras determinísticas",
    "ml-only": "Somente ML",
    "hybrid-and": "Híbrido ML $\\cap$ Regras (AND)",
    "hybrid-or": "Híbrido ML $\\cup$ Regras (OR)",
    "cascade-ml-rules": "Cascata ML $\\rightarrow$ Regras",
    "cascade-rules-ml": "Cascata Regras $\\rightarrow$ ML",
    "consensus-ml": "Consenso entre modelos ML",
    "consensus-hybrid": "Consenso + Regras",
}

rows = []
for cat in CATEGORY_ORDER:
    subset = abl[abl["category"] == cat]
    if subset.empty:
        continue
    best = subset.sort_values("f1", ascending=False).iloc[0]
    rows.append(
        {
            "Categoria": CATEGORY_LABELS.get(cat, cat),
            "Configuração": best["config"],
            "Precisão": f"{best['precision']:.3f}",
            "Revocação": f"{best['recall']:.3f}",
            "F1": f"\\textbf{{{best['f1']:.3f}}}"
            if best["f1"] >= 0.93
            else f"{best['f1']:.3f}",
        }
    )

tex = []
tex.append(r"\begin{table}[htbp]")
tex.append(r"\centering")
tex.append(
    r"\caption{Melhor configuração por categoria de classificação — estudo de ablação.}"
)
tex.append(r"\label{tab:ablation-best-category}")
tex.append(r"\small")
tex.append(r"\begin{tabular}{p{4.5cm}p{4.5cm}ccc}")
tex.append(r"\toprule")
tex.append(r"Categoria & Configuração & Precisão & Revocação & F1 \\")
tex.append(r"\midrule")
for r in rows:
    tex.append(
        f"{r['Categoria']} & {r['Configuração']} & {r['Precisão']} & {r['Revocação']} & {r['F1']} \\\\"
    )
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table}")

(OUT / "tab_ablation_best_category.tex").write_text("\n".join(tex), encoding="utf-8")
print(f"[OK] {OUT / 'tab_ablation_best_category.tex'}")

# ── Table 2: Top 10 overall ─────────────────────────────────────────
top10 = abl.sort_values("f1", ascending=False).head(10)

tex2 = []
tex2.append(r"\begin{table}[htbp]")
tex2.append(r"\centering")
tex2.append(r"\caption{Dez melhores configurações por F1 — estudo de ablação.}")
tex2.append(r"\label{tab:ablation-top10}")
tex2.append(r"\small")
tex2.append(r"\begin{tabular}{clccc}")
tex2.append(r"\toprule")
tex2.append(r"\# & Configuração & Precisão & Revocação & F1 \\")
tex2.append(r"\midrule")
for i, (_, r) in enumerate(top10.iterrows(), 1):
    f1_str = f"\\textbf{{{r['f1']:.3f}}}" if i == 1 else f"{r['f1']:.3f}"
    tex2.append(
        f"{i} & {r['config']} & {r['precision']:.3f} & {r['recall']:.3f} & {f1_str} \\\\"
    )
tex2.append(r"\bottomrule")
tex2.append(r"\end{tabular}")
tex2.append(r"\end{table}")

(OUT / "tab_ablation_top10.tex").write_text("\n".join(tex2), encoding="utf-8")
print(f"[OK] {OUT / 'tab_ablation_top10.tex'}")

# ── Table 3: Pareto frontier (distinct operating points) ────────────
# Extract actual Pareto-dominant points from RF+SMOTE grid
rf = pareto[pareto["ml_model"] == "RF+SMOTE"].copy()
frontier = []
for _, row in rf.iterrows():
    dominated = False
    for _, other in rf.iterrows():
        if (
            other["precision"] >= row["precision"]
            and other["recall"] >= row["recall"]
            and (
                other["precision"] > row["precision"] or other["recall"] > row["recall"]
            )
        ):
            dominated = True
            break
    if not dominated:
        frontier.append(row)

if frontier:
    fdf = pd.DataFrame(frontier).sort_values("recall", ascending=False)
    # Deduplicate by rounding to 3 decimals
    fdf["p3"] = fdf["precision"].round(3)
    fdf["r3"] = fdf["recall"].round(3)
    fdf = fdf.drop_duplicates(subset=["p3", "r3"]).head(8)

    tex3 = []
    tex3.append(r"\begin{table}[htbp]")
    tex3.append(r"\centering")
    tex3.append(
        r"\caption{Fronteira de Pareto: pontos operacionais do classificador híbrido (RF+SMOTE).}"
    )
    tex3.append(r"\label{tab:pareto-frontier}")
    tex3.append(r"\small")
    tex3.append(r"\begin{tabular}{cccccc}")
    tex3.append(r"\toprule")
    tex3.append(
        r"$\theta_{\text{ML}}$ & $\theta_{\text{Regras}}$ & Precisão & Revocação & F1 & Perfil \\"
    )
    tex3.append(r"\midrule")
    for _, r in fdf.iterrows():
        # Assign profile label
        if r["recall"] >= 0.98:
            profile = "Máx. revocação"
        elif r["precision"] >= 0.97:
            profile = "Máx. precisão"
        elif r["f1"] >= 0.94:
            profile = "\\textbf{Equilíbrio ótimo}"
        else:
            profile = "Intermediário"
        tex3.append(
            f"{r['ml_threshold']:.2f} & {r['rules_threshold']:.1f} & "
            f"{r['precision']:.3f} & {r['recall']:.3f} & {r['f1']:.3f} & {profile} \\\\"
        )
    tex3.append(r"\bottomrule")
    tex3.append(r"\end{tabular}")
    tex3.append(r"\end{table}")

    (OUT / "tab_pareto_frontier.tex").write_text("\n".join(tex3), encoding="utf-8")
    print(f"[OK] {OUT / 'tab_pareto_frontier.tex'}")

# ── Table 4: CV results ─────────────────────────────────────────────
cv_file = DATA / "kfold_cv_results.csv"
if cv_file.exists():
    cv = pd.read_csv(cv_file)
    tex4 = []
    tex4.append(r"\begin{table}[htbp]")
    tex4.append(r"\centering")
    tex4.append(
        r"\caption{Validação cruzada estratificada (3-\textit{fold}) das configurações selecionadas.}"
    )
    tex4.append(r"\label{tab:cv-results}")
    tex4.append(r"\small")
    tex4.append(r"\begin{tabular}{lccc}")
    tex4.append(r"\toprule")
    tex4.append(r"Configuração & Precisão & Revocação & F1 \\")
    tex4.append(r"\midrule")
    for _, r in cv.iterrows():
        tex4.append(
            f"{r['config']} & "
            f"${r['precision_mean']:.3f} \\pm {r['precision_std']:.3f}$ & "
            f"${r['recall_mean']:.3f} \\pm {r['recall_std']:.3f}$ & "
            f"${r['f1_mean']:.3f} \\pm {r['f1_std']:.3f}$ \\\\"
        )
    tex4.append(r"\bottomrule")
    tex4.append(r"\end{tabular}")
    tex4.append(r"\end{table}")

    (OUT / "tab_cv_results.tex").write_text("\n".join(tex4), encoding="utf-8")
    print(f"[OK] {OUT / 'tab_cv_results.tex'}")
else:
    print("[SKIP] cv_results.csv not found")

print("\n✓ All LaTeX tables generated.")
