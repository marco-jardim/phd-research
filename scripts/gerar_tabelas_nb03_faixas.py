"""Gera tab_nb03_precision.tex (a partir de data/nb03_results.json),
tab_desempenho_faixas.tex (a partir de data/desempenho_faixas_ablacao.csv, gerado por
scripts/desempenho_faixas_ablacao.py) e tab_e4_desempenho_faixas_oof.tex (a partir de
data/desempenho_faixas_oof_refrec.csv, gerado por scripts/desempenho_faixas_oof_refrec.py).

As tabelas são regeneradas integralmente a partir dos artefatos; nenhum valor
é digitado à mão. Uso: python scripts/gerar_tabelas_nb03_faixas.py
"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "text" / "latex" / "tables"


def d3(x: float) -> str:
    q = Decimal(str(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return f"{q}".replace(".", ",")


def d1pct(x: float) -> str:
    q = Decimal(str(float(x))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{q}".replace(".", ",")


def d4(x: float) -> str:
    q = Decimal(str(float(x))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    return f"{q}".replace(".", ",")


def fmt(x: float, f=d3) -> str:
    """Formata razao; denominador zero (NaN) vira n.d."""
    return "n.d." if pd.isna(x) else f(x)


def milhar(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def tabela_nb03() -> None:
    d = json.loads((DATA / "nb03_results.json").read_text(encoding="utf-8"))
    r = {x["Modelo"]: x for x in d["all_results"]}

    def linha(rotulo: str, limiar: str, chave: str, bold_prec: bool = False) -> str:
        x = r[chave]
        # Reconstrói VP a partir da sensibilidade registrada (4 casas) para evitar
        # arredondamento duplo: sensibilidade = VP/pos_test e precisão = VP/N.
        n_pos = x["N_Positivos"]
        vp = round(x["Recall"] * d["pos_test"])
        prec, rec = vp / n_pos, vp / d["pos_test"]
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        p = d3(prec)
        if bold_prec:
            p = f"\\textbf{{{p}}}"
        return f"{rotulo} & {limiar} & {p} & {d3(rec)} & {d3(f1)} & {n_pos} \\\\"

    ind = ["XGBoost", "LightGBM", "RF Calibrado", "Stacking"]
    ind_rows = sorted(
        ((m, r[f"{m} (th=0.9)"]) for m in ind), key=lambda t: -t[1]["Precision"]
    )
    linhas = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Desempenho dos modelos na estratégia de máxima precisão (NB03). "
        f"Conjunto de teste com {d['pos_test']} pares verdadeiros dentre {milhar(d['n_test'])} pares candidatos "
        f"(treino com {milhar(d['n_train'])} pares e {d['pos_train']} verdadeiros; {d['n_features']} atributos). "
        r"Resultados ordenados por precisão decrescente em cada bloco.}",
        r"\label{tab:nb03-precision}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\small",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"\textbf{Modelo} & \textbf{Limiar} & \textbf{Precisão} & \textbf{Sensibilidade} & \boldmath$F_1$ & \textbf{N} \\",
        r"\midrule",
        r"\multicolumn{6}{@{}l}{\textit{Abordagens compostas}} \\[2pt]",
        linha("Regras de negócio", r"escore $\geq 8$", "Regras (score≥8)", True),
        linha("Consenso (unanimidade)", r"unanimidade\textsuperscript{a}", "Consensus (unanimidade)"),
        linha("Híbrido ML + Regras (AND)", r"ML${\geq}$0,6; R${\geq}$6", "Hybrid AND (ML≥0.6, Rules≥6)"),
        r"\midrule",
        r"\multicolumn{6}{@{}l}{\textit{Modelos individuais com limiar 0,9}} \\[2pt]",
    ]
    for m, _ in ind_rows:
        linhas.append(linha(m, "0,9", f"{m} (th=0.9)"))
    linhas += [
        r"\midrule",
        r"\multicolumn{6}{@{}l}{\textit{Efeito da variação de limiar (LightGBM)}} \\[2pt]",
    ]
    for th in ("0.9", "0.8", "0.7", "0.6"):
        linhas.append(linha("LightGBM", th.replace(".", ","), f"LightGBM (th={th})"))
    auc = {
        "XGBoost": "XGBoost (th=0.9)",
        "LightGBM": "LightGBM (th=0.9)",
        "Consenso": "Consensus (unanimidade)",
        "Stacking": "Stacking (th=0.9)",
        "RF Calibrado": "RF Calibrado (th=0.9)",
        "Regras": "Regras (score≥8)",
    }
    auc_txt = "; ".join(f"{k}~{d3(r[v]['AUC-PR'])}" for k, v in auc.items())
    linhas += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{4pt}",
        r"\footnotesize",
        r"\textsuperscript{a}~Consenso por unanimidade entre os modelos base do caderno 03; o artefato "
        r"\texttt{data/nb03\_results.json} registra o desempenho do consenso, mas não o limiar de cada modelo base.",
        r"N~=~número de pares classificados como positivos no conjunto de teste. "
        r"Os limiares foram avaliados no próprio conjunto de teste, sem conjunto de validação, e a semente aleatória é 42 "
        r"(\texttt{scripts/run\_nb03\_uniform.py}).",
        r"AUC-PR (área sob a curva de precisão e sensibilidade, independente de limiar): " + auc_txt + ".",
        r"\end{table}",
        "",
    ]
    (OUT / "tab_nb03_precision.tex").write_text("\n".join(linhas), encoding="utf-8")
    print("[OK] tab_nb03_precision.tex")


def tabela_faixas() -> None:
    f = pd.read_csv(DATA / "desempenho_faixas_ablacao.csv")
    rot = {"0-3": "0--3", "3-5": "3--5", "5-6": "5--6", "6-7": "6--7", "7-8": "7--8", "8-9": "8--9", "9-10": "9--10", "10+": r"$\geq$10", "total": "Total"}
    ordem = list(rot)
    tot = f[f.faixa == "total"].iloc[0]
    n_teste, n_vinc = milhar(tot.n), int(tot.vinculos)
    linhas = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Desempenho do RF+SMOTE e do \textit{Gradient Boosting} por faixa da nota final do comparador, no conjunto de teste "
        f"({n_teste} potenciais pares; {n_vinc} vínculos verdadeiros), com a configuração da ablação: 58 atributos, SMOTE com razão 0,3 apenas no treino, "
        r"limiar de decisão 0,5 fixado antes da análise e semente 42.}",
        r"\label{tab:desempenho-faixas}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrcccccc@{}}",
        r"\toprule",
        r"& & & \multicolumn{3}{c}{RF+SMOTE} & \multicolumn{3}{c}{\textit{Gradient Boosting}} \\",
        r"\cmidrule(lr){4-6} \cmidrule(l){7-9}",
        r"Faixa & Total & Vínculos & VP/FP/FN & Sensib. & Prec. & VP/FP/FN & Sensib. & Prec. \\",
        r"\midrule",
    ]
    for faixa in ordem:
        if faixa == "total":
            linhas.append(r"\midrule")
        g = f[f.faixa == faixa].set_index("modelo")
        a, b = g.loc["RF+SMOTE"], g.loc["GB"]

        def bloco(x) -> str:
            return f"{int(x.vp)}/{int(x.fp)}/{int(x.fn)} & {fmt(x.sensibilidade)} & {fmt(x.precisao)}"

        linhas.append(f"{rot[faixa]} & {milhar(a.n)} & {int(a.vinculos)} & {bloco(a)} & {bloco(b)} \\\\")
    linhas += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{2pt}",
        r"\par\footnotesize Nota: VP, FP e FN são verdadeiros positivos, falsos positivos e falsos negativos. "
        r"Faixas fechadas à esquerda e abertas à direita; a última inclui notas a partir de 10. "
        r"n.d.: razão não definida (sem vínculos ou sem predições positivas na faixa). "
        r"A área cinza corresponde às faixas de 5 a 9. Fonte: \texttt{data/desempenho\_faixas\_ablacao.csv}.",
        r"\end{table}",
        "",
    ]
    (OUT / "tab_desempenho_faixas.tex").write_text("\n".join(linhas), encoding="utf-8")
    print("[OK] tab_desempenho_faixas.tex")


def tabela_e4_oof() -> None:
    f = pd.read_csv(DATA / "desempenho_faixas_oof_refrec.csv").set_index("faixa")
    rot = {
        "<5": r"$<5$",
        "5-6": "5--6",
        "6-7": "6--7",
        "7-8": "7--8",
        "8-9": "8--9",
        ">=9": r"$\geq9$",
        "area_cinza": r"Área cinza ($5\leq s<9$)",
        "total": "Total",
    }
    tot = f.loc["total"]
    linhas = [
        r"% Gerado por scripts/gerar_tabelas_nb03_faixas.py a partir de data/desempenho_faixas_oof_refrec.csv.",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Desempenho do estimador \texttt{p\_cal\_ml} (XGBoost com calibração isotônica) por faixa da nota final $s$ do comparador, "
        f"nos {milhar(tot.n)} potenciais pares e {int(tot.vinculos)} vínculos do extrato. "
        r"As probabilidades são estimativas fora da amostra agrupadas por REFREC; contagens e razões usam o limiar 0,5, fixado antes da análise e não otimizado. "
        r"AUC e AP: área sob a curva ROC e precisão média; n.d.: não definida.}",
        r"\label{tab:e4-desempenho-faixas-oof}",
        r"\begin{tabular}{@{}lrrcccccccc@{}}",
        r"\toprule",
        r"Faixa & Pares & Vínculos & AUC & AP & Brier & VP & FP & FN & Sensib. & Prec. \\",
        r"\midrule",
    ]
    for chave, nome in rot.items():
        if chave == "area_cinza":
            linhas.append(r"\midrule")
        x = f.loc[chave]
        linhas.append(
            f"{nome} & {milhar(x.n)} & {int(x.vinculos)} & {fmt(x.auc, d4)} & {fmt(x.ap)} & {fmt(x.brier, d4)} & "
            f"{int(x.vp)} & {int(x.fp)} & {int(x.fn)} & {fmt(x.sensibilidade)} & {fmt(x.precisao)} \\\\"
        )
    linhas += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{2pt}",
        r"\par\footnotesize Nota: faixas fechadas à esquerda e abertas à direita. VP, FP e FN são verdadeiros positivos, falsos positivos e falsos negativos "
        r"com \texttt{p\_cal\_ml} $\geq 0{,}5$. Fonte: \texttt{data/desempenho\_faixas\_oof\_refrec.csv}.",
        r"\end{table}",
        "",
    ]
    (OUT / "tab_e4_desempenho_faixas_oof.tex").write_text("\n".join(linhas), encoding="utf-8")
    print("[OK] tab_e4_desempenho_faixas_oof.tex")


if __name__ == "__main__":
    tabela_nb03()
    tabela_faixas()
    tabela_e4_oof()