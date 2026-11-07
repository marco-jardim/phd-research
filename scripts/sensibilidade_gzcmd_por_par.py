"""Aceitação automática v3.1 por PAR, com conferência das métricas finais REFREC."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.area_cinza_por_par import ROOT, carregar, fonte, gravar


def identificar(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    fields = frame[["COMPREC", "REFREC", "PASSO"]]
    if fields.isna().any().any():
        raise ValueError("Identificadores ausentes no extrato")
    frame["pair_id"] = [
        hashlib.sha256("|".join(value.strip() for value in values).encode()).hexdigest()[:16]
        for values in fields.itertuples(index=False, name=None)
    ]
    if frame.pair_id.duplicated().any():
        raise ValueError("pair_id duplicado no extrato")
    return frame[["pair_id", "PAR", "nota", "faixa"]]


def resumir(extrato: pd.DataFrame, funil: pd.DataFrame, modo: str,
            metricas: pd.DataFrame) -> pd.DataFrame:
    if funil.pair_id.duplicated().any():
        raise ValueError("pair_id duplicado no funil")
    if set(extrato.pair_id) != set(funil.pair_id):
        raise ValueError("Universos de pares do extrato e funil não coincidem")
    frame = extrato.merge(funil, on="pair_id", validate="one_to_one", suffixes=("", "_funil"))
    if not np.allclose(frame.nota, frame.nota_funil, rtol=0, atol=1e-10):
        raise ValueError("Notas do extrato e funil não coincidem")
    positive = frame.PAR.isin([1, 2])
    if not frame.y.eq(positive.astype(int)).all():
        raise ValueError("Gabaritos do extrato e funil não coincidem")
    if not frame.final.isin(["MATCH", "NONMATCH", "UNSURE", "DEFERRED"]).all():
        raise ValueError("Funil incompleto ou com decisão desconhecida")
    accepted = frame.final.eq("MATCH")
    reference = metricas.loc[metricas.modo.eq(modo) & metricas.versao.eq("v3.1")]
    if len(reference) != 1:
        raise ValueError("Métricas finais v3.1 ausentes ou duplicadas")
    observed = {
        "VP_cadeia": int((positive & accepted).sum()),
        "FP_cadeia": int((~positive & accepted).sum()),
        "FN_cadeia": int((positive & ~accepted).sum()),
        "VN_cadeia": int((~positive & ~accepted).sum()),
    }
    for name, value in observed.items():
        if value != int(reference.iloc[0][name]):
            raise ValueError(f"{name} diverge das métricas finais em {modo}")
    if not np.isclose(observed["VP_cadeia"] / positive.sum(),
                      reference.iloc[0].sens_cadeia_observada, rtol=0, atol=1e-12):
        raise ValueError("Sensibilidade diverge das métricas finais")
    rows = []
    for scope in ("TODAS", "[5,9)"):
        region = pd.Series(True, index=frame.index) if scope == "TODAS" else frame.faixa.eq(scope)
        for par in (1, 2):
            subset = region & frame.PAR.eq(par)
            total = int(subset.sum())
            automatic = int((subset & accepted).sum())
            rows.append({
                "modo": modo, "versao": "v3.1", "faixa": scope, "PAR": par,
                "vinculos": total, "aceitos_automaticamente": automatic,
                "sensibilidade_automatica": automatic / total if total else float("nan"),
                "aceitos_corte_orl": total if par == 1 else 0,
            })
    return pd.DataFrame(rows)


def executar(extrato: Path, data_dir: Path, out: Path) -> pd.DataFrame:
    frame = identificar(carregar(extrato, identificadores=True))
    metric_path = data_dir / "funil_gzcmd_final_metricas.csv"
    metricas = pd.read_csv(metric_path)
    sources = [fonte(extrato), fonte(metric_path)]
    tables = []
    for mode in ("vigilancia", "confirmacao"):
        path = data_dir / f"funil_gzcmd_{mode}.csv"
        funil = pd.read_csv(path, dtype={"pair_id": str}, usecols=["pair_id", "nota", "y", "final"])
        tables.append(resumir(frame, funil, mode, metricas))
        sources.append(fonte(path))
    table = pd.concat(tables, ignore_index=True)
    grey = table.loc[table.faixa.eq("[5,9)")]
    baseline = int(grey.loc[grey.modo.eq("vigilancia"), "aceitos_corte_orl"].sum())
    auto = grey.groupby("modo").aceitos_automaticamente.sum().astype(int).to_dict()
    total = int(grey.loc[grey.modo.eq("vigilancia"), "vinculos"].sum())
    # Menor recuperação que o corte, em qualquer modo, impede alteração das sínteses e merge.
    stop = any(count < baseline for count in auto.values()) or auto["vigilancia"] <= total / 2
    gravar(table, out, {
        "script": "scripts/sensibilidade_gzcmd_por_par.py", "fontes": sources,
        "endpoint": "final == MATCH, incluindo LLM MATCH; UNSURE e DEFERRED não são aceitos",
        "particionamento": "REFREC; artefatos sem sufixo _row",
        "matrizes_e_sensibilidades_conferidas": True,
        "area_cinza": {"vinculos": total, "corte_orl": baseline, "cascata": auto},
        "regra_de_parada": stop,
        "saida": "Somente agregados, sem pair_id",
    })
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extrato", type=Path, default=ROOT / "data/COMPARADORSEMIDENT.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "data/sensibilidade_gzcmd_por_par.csv")
    args = parser.parse_args()
    print(executar(args.extrato, args.data_dir, args.out).to_string(index=False))


if __name__ == "__main__":
    # Executável tanto por caminho quanto por python -m.
    main()
