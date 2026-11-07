"""Testes t do Experimento 1 (ablação MACD ON vs OFF), corte comprec.

Lê data/ablation_comprec_macd.csv e data/ablation_comprec_nomacd.csv,
emparelha as cinco rodadas por semente e calcula o teste t pareado
(e o de Welch, como comparação) para a diferença de F1 esperado
(exp_f1), precisão esperada (exp_precision) e número de revisões
(review_selected), por modo operacional. Saída: data/exp1_testes_t.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
METRICAS = ["exp_f1", "exp_precision", "exp_recall", "review_selected"]


def comparar(
    raiz: Path = RAIZ, *, on_path: Path | None = None, off_path: Path | None = None,
) -> pd.DataFrame:
    on = pd.read_csv(on_path or raiz / "data" / "ablation_comprec_macd.csv", sep=";")
    off = pd.read_csv(off_path or raiz / "data" / "ablation_comprec_nomacd.csv", sep=";")
    linhas = []
    for modo in sorted(on["mode"].unique()):
        a = on[on["mode"] == modo].sort_values("seed").reset_index(drop=True)
        b = off[off["mode"] == modo].sort_values("seed").reset_index(drop=True)
        if not (a["seed"].to_numpy() == b["seed"].to_numpy()).all():
            raise ValueError(f"sementes não coincidem no modo {modo}")
        for m in METRICAS:
            x, y = a[m].to_numpy(float), b[m].to_numpy(float)
            d = x - y
            tp = stats.ttest_rel(x, y)
            tw = stats.ttest_ind(x, y, equal_var=False)
            linhas.append(
                {
                    "modo": modo,
                    "metrica": m,
                    "n_rodadas": len(d),
                    "media_macd_on": x.mean(),
                    "media_macd_off": y.mean(),
                    "dp_macd_on": x.std(ddof=1),
                    "dp_macd_off": y.std(ddof=1),
                    "diferenca_media": d.mean(),
                    "t_pareado": float(tp.statistic),
                    "gl_pareado": len(d) - 1,
                    "p_pareado": float(tp.pvalue),
                    "t_welch": float(tw.statistic),
                    "p_welch": float(tw.pvalue),
                }
            )
    return pd.DataFrame(linhas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RAIZ / "data" / "exp1_testes_t.csv"))
    ap.add_argument("--on", type=Path, default=None)
    ap.add_argument("--off", type=Path, default=None)
    args = ap.parse_args()
    df = comparar(on_path=args.on, off_path=args.off)
    df.to_csv(args.out, index=False, float_format="%.6g")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
