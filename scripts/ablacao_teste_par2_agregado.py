"""Composição do teste canônico da ablação dos classificadores, sem retreino.

Reutiliza engineer_features e split_data, como ablation_study.py e
banca_b_recuperados_por_par.py. Não confundir com a ablação GZ-CMD por COMPREC.
"""
import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.feature_engineering import engineer_features, split_data


def load_raw(path):
    pd.options.future.infer_string = False
    raw = pd.read_csv(path, sep=";", dtype=str)
    raw.columns = [c.split(",")[0] for c in raw.columns]
    return raw


def numeric_frame(raw):
    return raw.apply(lambda c: pd.to_numeric(c.str.replace(",", ".", regex=False), errors="coerce"))


def summarize(raw):
    numeric = numeric_frame(raw)
    X, y = engineer_features(numeric)
    _, test, _, labels = split_data(X, y, random_state=42)
    par = numeric.loc[test.index, "PAR"]
    result = dict(particao="classificadores_linha", seed=42, test_size=0.3,
                  n_teste=len(test), par1=int(par.eq(1).sum()), par2=int(par.eq(2).sum()),
                  vinculos=int(labels.sum()))
    if result["vinculos"] != result["par1"] + result["par2"]:
        raise ValueError("Decomposição PAR incompatível com os labels")
    if (result["n_teste"], result["vinculos"]) != (18509, 74):
        raise ValueError("Partição não reproduz o controle histórico de 18.509/74")
    return pd.DataFrame([result])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/COMPARADORSEMIDENT.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/ablacao_teste_composicao.csv")
    args = parser.parse_args()
    result = summarize(load_raw(args.input))
    result.to_csv(args.output, index=False, lineterminator="\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
