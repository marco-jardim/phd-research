"""Reproduz Cohen kappa entre revisores do Exp2, com e sem UNSURE.

Uso: python scripts/kappas_revisores_exp2.py
Somente decisões presentes nos dois revisores entram em cada comparação.
O cálculo é determinístico, sem amostragem ou ajuste de classificadores.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = {"MATCH", "NONMATCH", "UNSURE"}
METADATA = {"pair_id", "y", "nota_final"}


def calcular_kappas(df: pd.DataFrame, modo: str) -> pd.DataFrame:
    """Compara todos os revisores, excluindo ausências antes de filtrar UNSURE."""
    if df["pair_id"].isna().any() or df["pair_id"].duplicated().any():
        raise ValueError("pair_id deve ser preenchido e unívoco")
    reviewers = [column for column in df.columns if column not in METADATA]
    rows = []
    for reviewer in reviewers:
        invalid = set(df[reviewer].dropna().unique()) - DECISIONS
        if invalid:
            raise ValueError(f"Decisões inválidas em {reviewer}: {sorted(invalid)}")
    for a, b in combinations(reviewers, 2):
        shared = df.loc[df[a].notna() & df[b].notna(), [a, b]]
        for include_unsure in (True, False):
            paired = shared if include_unsure else shared.loc[
                shared[a].ne("UNSURE") & shared[b].ne("UNSURE")
            ]
            if paired.empty:
                raise ValueError(f"Nenhuma decisão comparável: {modo}, {a}, {b}")
            # Kappa é indefinido se ambos só observam a mesma categoria.
            if pd.unique(paired.to_numpy().ravel()).size < 2:
                raise ValueError(f"Kappa indefinido: {modo}, {a}, {b}")
            opposite = ((paired[a].eq("MATCH") & paired[b].eq("NONMATCH")) |
                        (paired[a].eq("NONMATCH") & paired[b].eq("MATCH")))
            rows.append({
                "modo": modo,
                "revisor_a": a,
                "revisor_b": b,
                "incluir_unsure": include_unsure,
                "n": len(paired),
                "n_ausentes": len(df) - len(shared),
                "n_unsure_excluidos": len(shared) - len(paired),
                "concordantes": int(paired[a].eq(paired[b]).sum()),
                "discordancias_match_nonmatch": int(opposite.sum()),
                "kappa": cohen_kappa_score(paired[a], paired[b]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    tables = []
    for mode in ("vigilancia", "confirmacao"):
        source = args.data_dir / f"exp2_tres_revisores_{mode}.csv"
        if not source.exists():
            raise FileNotFoundError(
                f"{source} não está versionado no git (rótulos por par). Baixe-o do depósito Zenodo, "
                "versão 3.0.0, DOI 10.5281/zenodo.23928226 (https://zenodo.org/records/23928226), "
                "ou regenere-o com os scripts de reconstrução a partir de COMPARADORSEMIDENT.csv.")
        decisions = pd.read_csv(source, dtype={"pair_id": str})
        tables.append(calcular_kappas(decisions, mode))
    result = pd.concat(tables, ignore_index=True)
    output = args.data_dir / "exp2_kappas_revisores.csv"
    result.to_csv(output, index=False, float_format="%.9f")
    print(result.to_string(index=False))
    print(f"Artefato: {output}")


if __name__ == "__main__":
    main()
