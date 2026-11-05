"""Label-derived lower bound, conditional on descending OpenRecLink review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.arranjo_corte import below_cutoff, sha256
from scripts.arranjo_corte_s1_simular import load_raw


def aggregate(raw):
    below = below_cutoff(raw)
    work = pd.DataFrame({"passo": pd.to_numeric(raw.PASSO),
                         "par": pd.to_numeric(raw.PAR),
                         "score": pd.to_numeric(raw.SCORE.str.replace(",", ".", regex=False)),
                         "nota": pd.to_numeric(raw["nota final"].str.replace(",", ".", regex=False))})
    work = work.loc[below]
    minima = work.loc[work.par.eq(2)].groupby("passo").score.min()
    global_min = float(minima.min())
    rows = []
    for version in ("global", "por_passo"):
        threshold = pd.Series(global_min, index=work.index) if version == "global" else work.passo.map(minima)
        selected = work.score.ge(threshold)
        for step in [*sorted(work.passo.unique()), "TOTAL"]:
            region = pd.Series(True, index=work.index) if step == "TOTAL" else work.passo.eq(step)
            chosen = region & selected
            rows.append({"limiar": version, "passo": step, "escala": "SCORE_OpenRecLink",
                         "score_minimo": global_min if version == "global" else (
                             float(minima.loc[step]) if step != "TOTAL" else None),
                         "total_abaixo_corte": int(region.sum()), "pares_acima_limiar": int(chosen.sum()),
                         "par2_acima_limiar": int((chosen & work.par.eq(2)).sum()),
                         "nao_vinculos_acima_limiar": int((chosen & work.par.eq(0)).sum())})
    return pd.DataFrame(rows), {"SCORE_min_PAR2": global_min,
                                "nota_CdR_min_PAR2": float(work.loc[work.par.eq(2), "nota"].min()),
                                "SCORE_min_PAR2_por_passo": {str(k): float(v) for k, v in minima.items()}}


def generate(data_root, out=ROOT / "data"):
    source = data_root / "data/COMPARADORSEMIDENT.csv"
    table, minima = aggregate(load_raw(source))
    table.to_csv(out / "carga_abaixo_corte.csv", index=False)
    metadata = {"source": "data/COMPARADORSEMIDENT.csv", "source_sha256": sha256(source),
                "script_sha256": sha256(Path(__file__)),
                "ordering_scale": "SCORE (OpenRecLink), not nota final (CdR)", **minima,
                "interpretation": "Label-derived lower bound conditional on uninterrupted descending-score review (globally or within each step, respectively). Includes ties. Not an observed review log or an independent workload estimate; reviewers may have examined further pairs.",
                "undefined_step_threshold": "Steps without PAR=2 contribute zero to the per-step lower bound."}
    (out / "carga_abaixo_corte_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(minima, ensure_ascii=False))
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    generate(parser.parse_args().data_root)
