"""Cálculo de poder do piloto de seleção de modelos (Estudo 4).

Tamanho de amostra por grupo para comparar duas proporções
independentes (teste bilateral, aproximação normal, sem correção de
continuidade). Proporção de referência de 97,8% (concordância do
Kimi K2.5 no piloto de 517 pares). Saída: data/poder_amostral_piloto.csv.
"""
from __future__ import annotations

import argparse
from math import ceil, sqrt
from pathlib import Path

import pandas as pd
from scipy.stats import norm

RAIZ = Path(__file__).resolve().parents[1]


def n_por_grupo(p1: float, p2: float, alpha: float = 0.05, poder: float = 0.80) -> float:
    za = norm.ppf(1 - alpha / 2)
    zb = norm.ppf(poder)
    pbar = (p1 + p2) / 2
    num = (za * sqrt(2 * pbar * (1 - pbar)) + zb * sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return num / (p1 - p2) ** 2


def tabela(p1: float = 0.978, deltas=(0.035, 0.015), alpha: float = 0.05, poder: float = 0.80) -> pd.DataFrame:
    linhas = []
    for d in deltas:
        p2 = p1 - d
        linhas.append(
            {
                "p1": p1,
                "p2": round(p2, 4),
                "diferenca_pp": round(d * 100, 1),
                "alpha": alpha,
                "poder": poder,
                "n_por_grupo": ceil(n_por_grupo(p1, p2, alpha, poder)),
            }
        )
    return pd.DataFrame(linhas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RAIZ / "data" / "poder_amostral_piloto.csv"))
    args = ap.parse_args()
    df = tabela()
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
