"""Comparador de referência pós-ORL: contagens por faixa de nota e hipótese de erro humano.

Lê o extrato anonimizado (data/COMPARADORSEMIDENT.csv, não versionado), conta pares e
vínculos (PAR em {1, 2}) por faixa da nota final do Comparador de Registros e grava
data/referencia_comparador_pos_orl.csv, sem linhas por par.

Estratégias:
  * revisão humana integral da área cinza [5, 9), com aceitação automática dos pares com nota >= 9;
  * corte da nota em 9 (nada vai ao humano);
  * limiar >= 8 (opcional, referência do corte convencional da tese).

A sensibilidade com humano usa a hipótese de erro humano simétrico de 5%: um vínculo encaminhado
ao humano é recuperado com probabilidade 0,95, e um não-vínculo encaminhado é aceito por engano
com probabilidade 0,05. Não há decisão humana observada.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ERRO_HUMANO = 0.05


def carregar(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, sep=";", dtype=str)
    nota = pd.to_numeric(df["nota final"].str.replace(",", ".", regex=False), errors="raise")
    par = pd.to_numeric(df["PAR"], errors="raise")
    return pd.DataFrame({"nota": nota, "vinculo": par.isin([1, 2])})


def linha(estrategia: str, humano: pd.DataFrame, auto: pd.DataFrame, total_vinculos: int) -> dict:
    vp = int(auto["vinculo"].sum())
    fp = int((~auto["vinculo"]).sum())
    v_h = int(humano["vinculo"].sum())
    n_h = len(humano)
    sens_auto = vp / total_vinculos
    sens_humano = (vp + (1 - ERRO_HUMANO) * v_h) / total_vinculos if n_h else float("nan")
    return {
        "estrategia": estrategia,
        "pares_ao_humano": n_h,
        "vinculos_ao_humano": v_h,
        "pares_aceitos_automaticamente": len(auto),
        "vp_automaticos": vp,
        "fp_automaticos": fp,
        "total_vinculos": total_vinculos,
        "sens_automatica": sens_auto,
        "erro_humano_hipotese": ERRO_HUMANO if n_h else float("nan"),
        "sens_com_humano": sens_humano,
        "fp_humanos_esperados_hipotese": ERRO_HUMANO * (n_h - v_h) if n_h else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=ROOT / "data/COMPARADORSEMIDENT.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data/referencia_comparador_pos_orl.csv")
    args = ap.parse_args()

    d = carregar(args.csv)
    total = int(d["vinculo"].sum())
    cinza = d[(d.nota >= 5) & (d.nota < 9)]
    alto = d[d.nota >= 9]
    ge8 = d[d.nota >= 8]
    vazio = d.iloc[0:0]
    linhas = [
        linha("pos_orl_revisao_humana_area_cinza", cinza, alto, total),
        linha("pos_orl_corte_nota_9", vazio, alto, total),
        linha("pos_orl_limiar_nota_8", vazio, ge8, total),
    ]
    faixas = {
        "n_nota_lt5": int((d.nota < 5).sum()),
        "vinculos_nota_lt5": int(d[d.nota < 5]["vinculo"].sum()),
        "n_area_cinza": len(cinza),
        "vinculos_area_cinza": int(cinza["vinculo"].sum()),
        "n_nota_ge9": len(alto),
        "vinculos_nota_ge9": int(alto["vinculo"].sum()),
        "n_total": len(d),
    }
    out = pd.DataFrame(linhas)
    for k, v in faixas.items():
        out[k] = v
    out.to_csv(args.out, index=False)
    print(out.T.to_string())


if __name__ == "__main__":
    main()
