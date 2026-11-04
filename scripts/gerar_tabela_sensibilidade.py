"""Gera a tabela do apendice exclusivamente dos agregados do CSV."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data/sensibilidade_sem_obito_refrec.csv"
TEX = ROOT / "text/latex/tables/tab_sensibilidade_2x2.tex"


def numero(n, casas=3):
    return f"{n:,.{casas}f}".replace(",", "@SPEC@").replace(".", ",").replace("@SPEC@", ".")


def gerar(res=None, destino=TEX):
    if res is None:
        res = pd.read_csv(CSV)
    linhas = [r"% Gerado por scripts/gerar_tabela_sensibilidade.py de data/sensibilidade_sem_obito_refrec.csv",
              r"\begin{table}[htbp]", r"\centering\footnotesize",
              r"\setlength{\tabcolsep}{3pt}",
              r"\caption{Sensibilidade da ablação e do impacto epidemiológico à retirada dos indicadores de óbito e à partição por REFREC.}",
              r"\label{tab:sensibilidade-2x2}", r"\begin{tabular}{lrrrrrrr}",
              r"\toprule",
              r"Cenário & $n$ teste & Vínculos & \shortstack{$F_1$ RF\\+SMOTE} & $F_1$ GB & \shortstack{$F_1$ XGB\\local} & \shortstack{VP/FP\\RF+SMOTE} & \shortstack{Recuperados/\\líquido} \\",
              r"\midrule"]
    for r in res.itertuples():
        nome = f"{r.cenario}: {r.particao}/{r.n_atributos}"
        valores = [nome, numero(r.n_teste, 0), numero(r.vinculos_teste, 0),
                   numero(r.rf_smote_f1), numero(r.gb_f1), numero(r.xgb_local_f1),
                   f"{r.rf_smote_tp}/{r.rf_smote_fp}", f"{r.recuperados}/{r.liquido}"]
        linhas.append(" & ".join(valores) + r" \\")
    linhas += [r"\bottomrule", r"\end{tabular}",
               r"\par\smallskip\begin{minipage}{\textwidth}\footnotesize",
               r"Nota: RF: Random Forest; GB: Gradient Boosting; XGB: XGBoost local (CPU), não o resultado do fluxo Modal. O limiar de probabilidade de 0,5 e a nota $\geq 8$ são os da ablação. Recuperados são os vínculos encontrados pelo RF+SMOTE e não pela nota; líquido é a diferença entre as detecções. Com agrupamento por REFREC, o denominador e os vínculos avaliados mudam. 56 atributos excluem \texttt{obito\_tb} e \texttt{obito\_sinan}. Fonte: \texttt{data/sensibilidade\_sem\_obito\_refrec.csv}.",
               r"\end{minipage}", r"\end{table}"]
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    gerar()
