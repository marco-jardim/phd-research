"""Aggregate the published mean absolute SHAP by attribute family, without fitting.

Interactions between families and the mixed score_recall belong to ``outros``;
each attribute is counted once. Summing absolute SHAP is a descriptive allocation,
not a joint/conditional SHAP recomputation or an identifiable individual ranking.
Exact duplicate detection is exhaustive on the canonical seed-42 held-out X.
Distinct columns are not statistical rank or the number of independent causes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.feature_engineering import engineer_features, split_data
from scripts.sensibilidade_sem_obito_refrec import carregar

FAMILIES = ("nome", "nome da mãe", "nascimento", "endereço/município",
            "nota e derivados", "indicadores de óbito", "outros")
ENGINEERED = {
    "nome": {"nome_total", "nome_bom", "nome_prim_ok", "min_score_nome", "nome_squared"},
    "nome da mãe": {"mae_total", "mae_presente"},
    "nascimento": {"dtnasc_total", "dtnasc_ok", "dtnasc_parcial", "max_score_dtnasc",
                    "diff_ano_nasc", "diff_ano_ok", "dtnasc_squared"},
    "endereço/município": {"endereco_total", "local_total", "local_ok"},
    "nota e derivados": {"nota final", "nota_squared"},
    "indicadores de óbito": {"obito_sinan", "obito_tb"},
    "outros": {"score_recall", "nome_x_dtnasc", "nome_x_mae", "nome_x_local",
               "dtnasc_x_local", "nome_x_endereco", "evidencias_positivas",
               "ratio_nome_mae", "ratio_nome_nota"},
}


def family(feature: str) -> str:
    for prefix, label in (("NOME ", "nome"), ("NOMEMAE ", "nome da mãe"),
                          ("DTNASC ", "nascimento"), ("ENDERECO ", "endereço/município"),
                          ("CODMUNRES ", "endereço/município")):
        if feature.startswith(prefix):
            return label
    for label, names in ENGINEERED.items():
        if feature in names:
            return label
    raise ValueError(f"Unclassified feature: {feature}")


def aggregate(source: pd.DataFrame) -> pd.DataFrame:
    if not source.feature.is_unique or source.shap_mean_abs.isna().any() or source.shap_mean_abs.lt(0).any():
        raise ValueError("SHAP attributes must be unique and have nonnegative finite importances")
    if not np.isfinite(source.shap_mean_abs).all():
        raise ValueError("Nonfinite SHAP importance")
    source = source.assign(familia=source.feature.map(family))
    total = float(source.shap_mean_abs.sum())
    if total <= 0:
        raise ValueError("Zero total SHAP importance")
    rows = []
    for label in FAMILIES:
        subset = source.loc[source.familia.eq(label)]
        value = float(subset.shap_mean_abs.sum())
        rows.append({"familia": label, "n_atributos": len(subset),
                     "shap_mean_abs_soma": value, "fracao_total": value / total,
                     "atributos": " | ".join(sorted(subset.feature))})
    return pd.DataFrame(rows)


def duplicate_audit(X: pd.DataFrame) -> dict:
    if not X.columns.is_unique or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Finite X with unique attribute names required")
    groups: list[list[str]] = []
    remaining = list(X.columns)
    while remaining:
        first, *rest = remaining
        same = [name for name in rest if np.array_equal(X[first].to_numpy(), X[name].to_numpy())]
        if same:
            groups.append([first, *same])
        remaining = [name for name in rest if name not in same]
    expected = (("dtnasc_squared", "DTNASC dt iguais"), ("max_score_dtnasc", "dtnasc_ok"))
    verified = []
    for left, right in expected:
        if not np.array_equal(X[left].to_numpy(), X[right].to_numpy()):
            raise ValueError(f"Claimed duplicate is not exact in held-out X: {left}/{right}")
        verified.append({"atributos": [left, right], "igualdade_numerica": True})
    transforms = []
    for derived, original in (("nome_squared", "NOME qtd frag iguais"),
                              ("nota_squared", "nota final"),
                              ("dtnasc_squared", "DTNASC dt iguais")):
        x, y = X[original].to_numpy(), X[derived].to_numpy()
        if not np.array_equal(y, x ** 2):
            raise ValueError(f"Squared transform formula differs: {derived}")
        order = np.argsort(x, kind="stable")
        # Squaring is increasing only on a nonnegative domain. Do not silently
        # assume this for the final score: verify the actual held-out support.
        monotone = bool(np.all(np.diff(y[order]) >= 0))
        transforms.append({"origem": original, "derivado": derived, "formula": "x**2",
                           "minimo_origem_teste": float(x.min()),
                           "maximo_origem_teste": float(x.max()),
                           "monotona_nao_decrescente_no_teste": monotone})
    removed = sum(len(group) - 1 for group in groups)
    return {"semente": 42, "test_size": 0.3, "n_teste": len(X),
            "n_atributos_nominais": X.shape[1], "grupos_duplicatas_exatas_teste": groups,
            "duplicatas_exatas_redundantes": removed,
            "dimensao_apos_remover_duplicatas_exatas_teste": X.shape[1] - removed,
            "pares_exigidos_verificados": verified, "transformacoes": transforms,
            "limite": "Colunas distintas no teste; não é posto matricial, independência estatística ou dimensão intrínseca. Igualdade amostral não prova identidade para entradas futuras.",
            "codigo": {"dtnasc_squared": "DTNASC dt iguais ** 2; igualdade requer escore binário",
                       "max_score_dtnasc": "max(dt iguais, dt ap 1digi); coincide com dtnasc_ok no suporte binário"}}


def generate(extract: Path, data: Path) -> tuple[pd.DataFrame, dict]:
    source = data / "shap_importance.csv"
    importance = pd.read_csv(source)
    frame = carregar(extract)  # dtype=str, then explicit to_numeric
    X, y = engineer_features(frame)
    _, test, _, test_y = split_data(X, y, random_state=42)
    if set(importance.feature) != set(X.columns):
        raise ValueError("SHAP attributes differ from the canonical X")
    audit = duplicate_audit(test)
    audit["vinculos_teste"] = int(test_y.sum())
    audit["fontes_sha256"] = {
        "data/COMPARADORSEMIDENT.csv": hashlib.sha256(extract.read_bytes()).hexdigest(),
        "data/shap_importance.csv": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scripts/feature_engineering.py": hashlib.sha256((ROOT / "scripts/feature_engineering.py").read_bytes()).hexdigest(),
    }
    result = aggregate(importance)
    result.to_csv(data / "shap_por_familia.csv", index=False, lineterminator="\n")
    (data / "atributos_duplicados.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8", newline="\n")
    print(result.to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return result, audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path, default=ROOT / "data/COMPARADORSEMIDENT.csv")
    args = parser.parse_args()
    generate(args.extract, ROOT / "data")
