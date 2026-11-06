"""Contagens sem treino; predições históricas opcionais, nunca reconstruídas.

--predictions: CSV privado com indice (índice original do extrato) e probabilidade
para TODAS as linhas do teste RF+SMOTE original. Sem ele, reporta a lacuna.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.feature_engineering import engineer_features, split_data

ROOT = Path(__file__).resolve().parents[1]
CUTS = {6: 7.0240, 9: 7.8444, 11: 7.9019}


def summarize(raw, predictions=None):
    numeric = raw.apply(lambda col: pd.to_numeric(col.str.replace(",", ".", regex=False), errors="coerce"))
    links = numeric.PAR.isin([1, 2])
    if int(links.sum()) != 247:
        raise ValueError("Universo diferente de 247 vínculos")
    result = {"vinculos": 247, "obitos_distintos_247": int(raw.loc[links, "REFREC"].nunique()),
              "vinculos_por_par": {str(p): int(numeric.PAR.eq(p).sum()) for p in (1, 2)},
              "treino_executado": False}
    if predictions is None:
        result.update(status="pendente_predicoes_historicas_RF_SMOTE",
                      motivo="Agregado 67 VP/3 FP/27 recuperados não identifica os integrantes; não treinar novamente.")
        return result
    X, y = engineer_features(numeric)
    _, test, _, y_test = split_data(X, y, random_state=42)
    if not predictions.index.is_unique or set(predictions.index) != set(test.index):
        raise ValueError("Predições devem cobrir exatamente o teste original, sem duplicatas")
    probability = pd.to_numeric(predictions.loc[test.index, "probabilidade"], errors="raise")
    if not probability.between(0, 1).all():
        raise ValueError("Probabilidades inválidas")
    positive = probability.ge(.5)
    tp = positive & y_test.eq(1)
    recovered = tp & test["nota final"].lt(8)
    controls = dict(test_n=len(test), test_positivos=int(y_test.sum()), tp=int(tp.sum()),
                    fp=int((positive & y_test.eq(0)).sum()), recuperados=int(recovered.sum()))
    if controls != dict(test_n=18509, test_positivos=74, tp=67, fp=3, recuperados=27):
        raise ValueError(f"Controle histórico não reproduzido: {controls}")
    result.update(status="completo", controle=controls)
    for label, mask in (("vp", tp), ("recuperados", recovered)):
        idx = mask.index[mask]
        subset = numeric.loc[idx]
        thresholds = subset.PASSO.map(CUTS)
        if subset.SCORE.isna().any():
            raise ValueError("Escore ORL ausente entre os vínculos selecionados")
        above = subset.SCORE.ge(thresholds)
        result[label] = {"n": len(idx), "por_par": {
            str(p): int(subset.PAR.eq(p).sum()) for p in (1, 2)},
            "acima_ou_no_corte_orl": int((above & thresholds.notna()).sum()),
            "abaixo_corte_orl": int((~above & thresholds.notna()).sum()),
            "corte_desconhecido": int(thresholds.isna().sum()),
            "obitos_distintos": int(raw.loc[idx, "REFREC"].nunique())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/COMPARADORSEMIDENT.csv")
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    raw = pd.read_csv(args.input, sep=";", dtype=str)
    raw.columns = [c.split(",")[0] for c in raw.columns]
    predictions = pd.read_csv(args.predictions, index_col="indice") if args.predictions else None
    result = summarize(raw, predictions)
    result["entrada_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if args.predictions:
        result["predicoes_sha256"] = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    (ROOT / "data/banca_b_recuperados_por_par.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
