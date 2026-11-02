"""Validação dos conjuntos âncora contra o padrão-ouro (Tabela tab:cap8-ancoras).

Calcula, sobre o extrato sem identificadores (data/COMPARADORSEMIDENT.csv),
o tamanho e a pureza dos conjuntos âncora definidos por escore extremo da
nota final do comparador, e compara a calibração por âncoras (restrições de
média, sem rótulos) com a calibração supervisionada (Platt sobre a nota final)
na área cinza 5 <= s < 9.

Uso:
    python scripts/validar_ancoras.py [--csv data/COMPARADORSEMIDENT.csv]
Saídas:
    data/ancoras_validacao.csv          tabela A+/A- (N, acertos, erros, pureza)
    data/ancoras_calibracao_faixas.csv  taxa observada vs p_cal por faixa
    text/latex/tables/tab_cap8_ancoras.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
T_PLUS, T_MINUS = 9.0, 5.0
BANDS = [(5, 6), (6, 7), (7, 8), (8, 9)]


def fit_anchor_platt(s: np.ndarray, a_plus: np.ndarray, a_minus: np.ndarray, r_plus: float, r_minus: float):
    """Ajusta sigmoid(alpha*s+beta) igualando as médias em A+ e A- aos alvos (resíduos em logit)."""

    def loss(w):
        mp = expit(w[0] * s[a_plus] + w[1]).mean()
        mn = expit(w[0] * s[a_minus] + w[1]).mean()
        return (logit(mp) - logit(r_plus)) ** 2 + (logit(mn) - logit(r_minus)) ** 2

    res = minimize(loss, x0=[1.0, -7.0], method="L-BFGS-B", bounds=[(0.0, 10.0), (-50.0, 50.0)])
    return res.x, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "COMPARADORSEMIDENT.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv, sep=";", low_memory=False, decimal=",")
    s = pd.to_numeric(df["nota final"], errors="coerce").to_numpy()
    y = df["PAR"].isin([1, 2]).astype(int).to_numpy()
    assert not np.isnan(s).any()

    a_plus, a_minus = s >= T_PLUS, s < T_MINUS
    rows = []
    for nome, crit, m, alvo_pos in [
        ("A+", f"s >= {T_PLUS:.1f}", a_plus, True),
        ("A-", f"s < {T_MINUS:.1f}", a_minus, False),
    ]:
        n, pos = int(m.sum()), int(y[m].sum())
        acertos = pos if alvo_pos else n - pos
        rows.append({"conjunto": nome, "criterio": crit, "N": n, "acertos": acertos, "erros": n - acertos,
                     "pureza_pct": round(100 * acertos / n, 2), "taxa_pos": round(pos / n, 5)})
    tab = pd.DataFrame(rows)
    r_plus, r_minus = tab.loc[0, "taxa_pos"], tab.loc[1, "taxa_pos"]

    w_anc, res = fit_anchor_platt(s, a_plus, a_minus, r_plus, r_minus)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(s.reshape(-1, 1), y)
    w_sup = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))

    faixas = []
    for a, b in BANDS:
        m = (s >= a) & (s < b)
        faixas.append({"faixa": f"[{a},{b})", "n": int(m.sum()), "pos": int(y[m].sum()),
                       "obs_pct": round(100 * y[m].mean(), 2),
                       "p_ancoras_pct": round(100 * expit(w_anc[0] * s[m] + w_anc[1]).mean(), 2),
                       "p_superv_pct": round(100 * expit(w_sup[0] * s[m] + w_sup[1]).mean(), 2)})
    g = (s >= T_MINUS) & (s < T_PLUS)
    brier = {"ancoras": brier_score_loss(y[g], expit(w_anc[0] * s[g] + w_anc[1])),
             "supervisionado": brier_score_loss(y[g], expit(w_sup[0] * s[g] + w_sup[1]))}

    out = ROOT / "data"
    tab.to_csv(out / "ancoras_validacao.csv", index=False)
    pd.DataFrame(faixas).to_csv(out / "ancoras_calibracao_faixas.csv", index=False)

    def fmt(n: int) -> str:
        return f"{n:,}".replace(",", ".")

    tex = [
        "% Gerado por scripts/validar_ancoras.py sobre data/COMPARADORSEMIDENT.csv",
        r"\begin{table}[!ht]", r"\centering",
        r"\caption{Validação dos conjuntos âncora contra o padrão-ouro (extrato sem identificadores, $N = 61.696$ pares, 247 vínculos verdadeiros).}",
        r"\label{tab:cap8-ancoras}", r"\begin{tabular}{|l|c|r|r|r|c|}", r"\hline",
        r"Conjunto & Critério & $N$ & Acertos & Erros & Pureza (\%) \\", r"\hline",
    ]
    for r in rows:
        crit = r"$s\geq 9{,}0$" if r["conjunto"] == "A+" else r"$s<5{,}0$"
        nome = "$A^+$ (âncoras positivas)" if r["conjunto"] == "A+" else "$A^-$ (âncoras negativas)"
        pureza = f"{r['pureza_pct']:.2f}".replace(".", ",")
        tex.append(f"{nome} & {crit} & {fmt(r['N'])} & {fmt(r['acertos'])} & {r['erros']} & {pureza} \\\\")
        tex.append(r"\hline")
    tex += [r"\end{tabular}", r"\end{table}"]
    (ROOT / "text" / "latex" / "tables" / "tab_cap8_ancoras.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    print(tab.to_string(index=False))
    print(f"\nalpha,beta ancoras = {w_anc.round(3)} (converged={res.success}); supervisionado = {np.round(w_sup, 3)}")
    print(pd.DataFrame(faixas).to_string(index=False))
    print("Brier area cinza:", {k: round(v, 4) for k, v in brier.items()})


if __name__ == "__main__":
    main()
