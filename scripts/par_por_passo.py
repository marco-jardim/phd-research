"""Contagens agregadas de PAR por passo e decisão observada do arranjo P.

Reutiliza políticas locais e respostas persistidas; não ajusta modelos, não chama
APIs e não exporta identificadores ou atributos individuais.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.arranjo_corte import below_cutoff, sha256
from gzcmd.ml_oof import pair_ids
from scripts.arranjo_corte_s1_simular import LOCAL, MODES, load_raw
from scripts.arranjo_corte_s2_suplementar import responses
from scripts.arranjo_corte_s3_avaliar import replay, require

DECISIONS = ("MATCH", "HUMANO", "NONMATCH")


def aggregate(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Inclui células vazias e reúne UNSURE/DEFERRED como revisão pendente."""
    require(frame.final.isin(["MATCH", "NONMATCH", "UNSURE", "DEFERRED"]).all(),
            "Unknown final decision")
    require(frame.par.isin([0, 1, 2]).all(), "Unknown PAR")
    work = frame[["passo", "par"]].copy()
    work["decisao"] = frame.final.replace({"UNSURE": "HUMANO", "DEFERRED": "HUMANO"})
    index = pd.MultiIndex.from_tuples(
        list(product(sorted(work.passo.unique()), [0, 1, 2], DECISIONS)),
        names=["passo", "par", "decisao"])
    out = work.groupby(["passo", "par", "decisao"]).size().reindex(index, fill_value=0)
    out = out.rename("n").reset_index()
    out.insert(0, "modo", mode)
    require(int(out.n.sum()) == len(frame), "Aggregate accounting differs")
    return out


def generate(data_root: Path) -> pd.DataFrame:
    raw = load_raw(data_root / "data/COMPARADORSEMIDENT.csv")
    base = pd.DataFrame({"pair_id": pair_ids(raw),
                         "passo": pd.to_numeric(raw.PASSO, errors="raise").astype(int),
                         "par": pd.to_numeric(raw.PAR, errors="raise").astype(int),
                         "abaixo": below_cutoff(raw)}).set_index("pair_id")
    require(base.index.is_unique, "Duplicate pair identifier")
    meta = json.loads((ROOT / "data/arranjo_corte_metadata.json").read_text(encoding="utf-8"))
    tables = []
    for mode in MODES:
        path = LOCAL / f"policy_{mode}_P.csv"
        require(sha256(path) == meta["policy_sha256"][path.name], "P policy fingerprint differs")
        policy = pd.read_csv(path).set_index("pair_id")
        require(policy.index.is_unique and set(policy.index) == set(base.index[base.abaixo]),
                "P policy universe differs")
        frame = base.copy()
        frame["action"] = np.where(frame.abaixo, "NONMATCH", "MATCH")
        frame.loc[policy.index, "action"] = policy.action
        known, _ = responses(data_root, mode, include_supplement=True)
        tables.append(aggregate(replay(frame, known), mode))
    return pd.concat(tables, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    table = generate(args.data_root.resolve())
    path = ROOT / "data/par_por_passo.csv"
    table.to_csv(path, index=False)
    print(path)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
