"""Conta encerramentos por obito nos recuperados, sem exportar linhas ou IDs.

Uso: python -m scripts.banca_b_encerramento_agregado --input CAMINHO
O extrato e lido como texto. A execucao falha se nao reproduzir as contagens
da ablacao original; nesse caso nao se atribuem contagens ao grupo historico.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from scripts.feature_engineering import engineer_features, split_data

ROOT = Path(__file__).resolve().parents[1]


def summarize_situence(values: pd.Series) -> dict[str, int]:
    codes = pd.to_numeric(values.str.replace(",", "."), errors="coerce")
    return {
        "n": len(codes),
        "encerrados_obito_tb": int(codes.eq(3).sum()),
        "encerrados_obito_outras_causas": int(codes.eq(4).sum()),
        "encerrados_obito_total": int(codes.isin([3, 4]).sum()),
        "outros_encerramentos": int((codes.notna() & ~codes.isin([3, 4])).sum()),
        "ausentes_ou_nao_numericos": int(codes.isna().sum()),
    }


def historical_predictions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Reproduz o RF+SMOTE publicado; predições ficam apenas em memória."""
    numeric = raw.apply(
        lambda column: pd.to_numeric(column.str.replace(",", "."), errors="coerce")
    )
    X, y = engineer_features(numeric)
    train, test, y_train, y_test = split_data(X, y)
    X_res, y_res = SMOTE(
        sampling_strategy=0.3, k_neighbors=3, random_state=42
    ).fit_resample(train, y_train)
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
    ).fit(X_res, y_res)
    probability = pd.Series(rf.predict_proba(test)[:, 1], index=test.index)
    positive = probability.ge(0.5)
    recovered = positive & y_test.eq(1) & test["nota final"].lt(8)
    control = {
        "test_n": len(test),
        "test_positivos": int(y_test.sum()),
        "tp": int((positive & y_test.eq(1)).sum()),
        "fp": int((positive & y_test.eq(0)).sum()),
        "recuperados": int(recovered.sum()),
    }
    expected = {"test_n": 18509, "test_positivos": 74, "tp": 67, "fp": 3, "recuperados": 27}
    if control != expected:
        raise ValueError(f"Controle nao reproduzido: {control}; esperado: {expected}")
    return pd.DataFrame({"probabilidade": probability}), control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    pd.options.future.infer_string = False
    raw = pd.read_csv(args.input, sep=";", dtype=str)
    raw.columns = [c.split(",")[0] for c in raw.columns]
    if "C_SITUENCE" not in raw:
        raise ValueError("Extrato sem C_SITUENCE: contagem nao quantificada.")
    predictions, control = historical_predictions(raw)
    selected = raw.loc[predictions.index]
    par = pd.to_numeric(selected.PAR, errors="raise")
    nota = pd.to_numeric(selected["nota final"].str.replace(",", ".", regex=False), errors="raise")
    recovered = predictions.probabilidade.ge(0.5) & par.isin([1, 2]) & nota.lt(8)
    result = {
        "definicao": "RF+SMOTE >=0.5, PAR=1 e nota final <8 no teste original",
        "entrada_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "controle": control,
        "C_SITUENCE": summarize_situence(raw.loc[predictions.index[recovered], "C_SITUENCE"]),
    }
    output = ROOT / "data/banca_b_encerramento_agregado.json"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
