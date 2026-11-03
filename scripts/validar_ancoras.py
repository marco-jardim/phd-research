"""Validação dos conjuntos âncora contra o padrão-ouro (Tabela tab:cap8-ancoras).

Calcula, sobre o extrato sem identificadores (data/COMPARADORSEMIDENT.csv),
o tamanho e a pureza dos conjuntos âncora definidos por escore extremo da
nota final do comparador, e compara a calibração por âncoras (restrições de
média, sem rótulos) com a calibração supervisionada (Platt sobre a nota final)
na área cinza 5 <= s < 9.

Uso:
    python scripts/validar_ancoras.py [--csv data/COMPARADORSEMIDENT.csv]
Dependência: gzcmd-record-linkage==0.2.0, declarada no pyproject.toml.
Se o índice não disponibilizar essa versão, instalar explicitamente um checkout
da versão 0.2.0 com ``python -m pip install /caminho/do/checkout``; não se altera
sys.path para importar um repositório vizinho implicitamente.
Saídas:
    data/ancoras_validacao.csv          tabela A+/A- (N, acertos, erros, pureza)
    data/ancoras_calibracao_faixas.csv  taxa observada vs p_cal por faixa
    text/latex/tables/tab_cap8_ancoras.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
# Motor de decisão da tese (cópia local usada nos experimentos).
sys.path.insert(0, str(ROOT))
from gzcmd.gzcmd_v3_policy_engine import Budget, Costs, LLMError, PolicyEngineV3  # noqa: E402
from gzcmd_record_linkage.calibration import fit_anchor_platt as pkg_fit_anchor_platt  # noqa: E402
T_PLUS, T_MINUS = 9.0, 5.0
BANDS = [(5, 6), (6, 7), (7, 8), (8, 9)]


def fmt(n: int) -> str:
    """Milhar com ponto, conforme a tese."""
    return f"{n:,}".replace(",", ".")


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

    # Três cenários de calibração da nota final:
    #  (1) supervisionado: Platt com todos os rótulos;
    #  (2) âncoras com prevalências medidas: os rótulos fixam apenas r+ e r- (duas constantes);
    #  (3) âncoras com alvos a priori (sem rótulos): r+ = 0,99 e r- = 0,001, mais a variante r- = 0,0001.
    # O ajuste por âncoras usa gzcmd_record_linkage.calibration.fit_anchor_platt (pacote da tese);
    # a implementação local fit_anchor_platt acima é mantida apenas como conferência.
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(s.reshape(-1, 1), y)
    nota = pd.Series(s)

    def pkg_fit(rp: float, rn: float) -> np.ndarray:
        m = pkg_fit_anchor_platt(nota, s_plus=T_PLUS, s_minus=T_MINUS, r_plus=rp, r_minus=rn)
        return np.array([m.slope, m.intercept])

    cenarios = {
        "supervisionado": np.array([float(lr.coef_[0, 0]), float(lr.intercept_[0])]),
        "ancoras_medidas": pkg_fit(r_plus, r_minus),
        "ancoras_apriori_0.99_0.001": pkg_fit(0.99, 0.001),
        "ancoras_apriori_0.99_0.0001": pkg_fit(0.99, 0.0001),
    }
    w_anc, w_sup = cenarios["ancoras_medidas"], cenarios["supervisionado"]
    w_local, res = fit_anchor_platt(s, a_plus, a_minus, r_plus, r_minus)
    print(f"conferência: pacote alpha,beta={w_anc.round(4)} | local={w_local.round(4)}")

    faixas = []
    for a, b in BANDS:
        m = (s >= a) & (s < b)
        row = {"faixa": f"[{a},{b})", "n": int(m.sum()), "pos": int(y[m].sum()), "obs_pct": round(100 * y[m].mean(), 2)}
        for k, w in cenarios.items():
            row[f"p_{k}_pct"] = round(100 * expit(w[0] * s[m] + w[1]).mean(), 2)
        faixas.append(row)
    g = (s >= T_MINUS) & (s < T_PLUS)
    brier = {k: brier_score_loss(y[g], expit(w[0] * s[g] + w[1])) for k, w in cenarios.items()}

    # Política de decisão por custo esperado na área cinza, executada pelo próprio motor do
    # framework (gzcmd/gzcmd_v3_policy_engine.py: PolicyEngineV3.triage), sem regras de guarda
    # nem restrições de consistência, para isolar o efeito da calibração. Custos, tetos
    # automáticos, orçamento e taxas de erro do revisor por faixa vêm de gzcmd_v3_config.yaml
    # (decision_policy.modes, llm_review.reliability.error_rates_by_band).
    modes = {"vigilancia": (10.0, 50.0, 0.25, 0.85, 0.15, 2000), "confirmacao": (100.0, 20.0, 0.25, 0.95, 0.10, 1000)}
    err_by_band = {"grey_low": LLMError(0.08, 0.12), "grey_mid": LLMError(0.06, 0.10),
                   "grey_high": LLMError(0.04, 0.08), "near_high": LLMError(0.02, 0.06)}
    sg, yg = s[g], y[g]
    band = np.where(sg < 6, "grey_low", np.where(sg < 7, "grey_mid", np.where(sg < 8, "grey_high", "near_high")))
    e_fp = np.array([err_by_band[b].e_fp for b in band]); e_fn = np.array([err_by_band[b].e_fn for b in band])
    decisao = []
    for mode, (cfp, cfn, cllm, t_match, t_non, budget) in modes.items():
        for k, w in cenarios.items():
            p = expit(w[0] * sg + w[1])
            engine = PolicyEngineV3(Costs(cfp, cfn, cllm), err_by_band, Budget(llm_max=budget),
                                    min_auto_match=t_match, max_auto_nonmatch=t_non)
            out = engine.triage(pd.DataFrame({"p_cal": p, "band": band}))
            act = out["action"].map({"MATCH": "M", "NONMATCH": "N", "LLM_REVIEW": "R"}).to_numpy()
            is_r = act == "R"
            fn_auto = int(((act == "N") & (yg == 1)).sum()); fp_auto = int(((act == "M") & (yg == 0)).sum())
            perda = fn_auto * cfn + fp_auto * cfp + is_r.sum() * cllm
            perda_rev = perda + ((is_r & (yg == 1)) * e_fn * cfn).sum() + ((is_r & (yg == 0)) * e_fp * cfp).sum()
            decisao.append({"modo": mode, "calibracao": k, "revisao": int(is_r.sum()), "auto_match": int((act == "M").sum()),
                            "auto_nonmatch": int((act == "N").sum()), "fn_auto": fn_auto, "fp_auto": fp_auto,
                            "perda_revisor_perfeito": round(perda, 1), "perda_revisor_config": round(perda_rev, 1)})
    dec = pd.DataFrame(decisao)

    out = ROOT / "data"
    tab.to_csv(out / "ancoras_validacao.csv", index=False)
    pd.DataFrame(faixas).to_csv(out / "ancoras_calibracao_faixas.csv", index=False)
    dec.to_csv(out / "ancoras_decisao_simulada.csv", index=False)

    nomes = {"supervisionado": "Supervisionada (Platt, 247 rótulos)",
             "ancoras_medidas": "Âncoras, prevalências medidas",
             "ancoras_apriori_0.99_0.001": "Âncoras a priori ($0{,}99$ / $0{,}001$)",
             "ancoras_apriori_0.99_0.0001": "Âncoras a priori ($0{,}99$ / $0{,}0001$)"}
    tex2 = [
        "% Gerado por scripts/validar_ancoras.py (simulação simplificada da política de decisão na área cinza)",
        r"\begin{table}[!ht]", r"\centering",
        r"\caption{Efeito da calibração sobre a decisão por custo esperado na área cinza ($5\leq s<9$; 21.620 pares, 151 vínculos), por modo operacional. Executado pelo motor de decisão do \textit{framework}, sem regras de guarda nem restrições de consistência; custos, tetos, orçamento e taxas de erro do revisor conforme a configuração. Perda realizada com revisor perfeito e com as taxas de erro configuradas.}",
        r"\label{tab:cap8-ancoras-decisao}", r"\footnotesize", r"\begin{tabular}{|l|l|r|r|r|r|r|}", r"\hline",
        r"Modo & Calibração & Revisão & FN auto & FP auto & Perda (rev.\ perfeito) & Perda (rev.\ config.) \\", r"\hline",
    ]
    for mode in modes:
        for k in cenarios:
            r = dec[(dec.modo == mode) & (dec.calibracao == k)].iloc[0]
            tex2.append(f"{'Vigilância' if mode == 'vigilancia' else 'Confirmação'} & {nomes[k]} & {fmt(int(r.revisao))} & {int(r.fn_auto)} & {int(r.fp_auto)} & {fmt(int(round(r.perda_revisor_perfeito)))} & {fmt(int(round(r.perda_revisor_config)))} \\\\")
        tex2.append(r"\hline")
    tex2 += [r"\end{tabular}", r"\end{table}"]
    (ROOT / "text" / "latex" / "tables" / "tab_cap8_ancoras_decisao.tex").write_text("\n".join(tex2) + "\n", encoding="utf-8")
    print(dec.to_string(index=False))

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
