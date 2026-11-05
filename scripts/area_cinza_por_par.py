"""Conta vínculos por faixa da nota do CdR e PAR, sem exportar linhas por par."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FAIXAS = ("<5", "[5,9)", ">=9")


def carregar(extrato: Path, *, identificadores: bool = False) -> pd.DataFrame:
    campos = {"PAR", "nota final"}
    if identificadores:
        campos.update({"COMPREC", "REFREC", "PASSO"})
    frame = pd.read_csv(
        extrato, sep=";", dtype=str,
        usecols=lambda column: column.split(",")[0] in campos,
    )
    frame.columns = [column.split(",")[0] for column in frame.columns]
    frame["PAR"] = pd.to_numeric(frame["PAR"], errors="raise")
    frame["nota"] = pd.to_numeric(
        frame.pop("nota final").str.replace(",", ".", regex=False), errors="raise"
    )
    if frame[["PAR", "nota"]].isna().any().any():
        raise ValueError("PAR e nota não podem conter valores ausentes")
    if not np.isfinite(frame["nota"]).all():
        raise ValueError("Nota deve ser finita")
    if not frame["PAR"].isin([0, 1, 2]).all():
        raise ValueError("PAR deve pertencer a {0, 1, 2}")
    frame["faixa"] = pd.cut(
        frame["nota"], [-float("inf"), 5, 9, float("inf")],
        right=False, labels=FAIXAS,
    )
    return frame


def resumir(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"faixa": faixa, "PAR": par,
         "vinculos": int((frame.faixa.eq(faixa) & frame.PAR.eq(par)).sum())}
        for faixa in FAIXAS for par in (1, 2)
    ])


def fonte(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"arquivo": path.name, "sha256": digest.hexdigest()}


def gravar(table: pd.DataFrame, out: Path, metadata: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    out.with_name(out.stem + "_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extrato", type=Path, default=ROOT / "data/COMPARADORSEMIDENT.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data/area_cinza_por_par.csv")
    args = parser.parse_args()
    frame = carregar(args.extrato)
    table = resumir(frame)
    gravar(table, args.out, {
        "script": "scripts/area_cinza_por_par.py", "fontes": [fonte(args.extrato)],
        "faixas": list(FAIXAS), "total_potenciais_pares": len(frame),
        "total_vinculos": int(frame.PAR.isin([1, 2]).sum()),
        "definicao": "PAR em {1, 2}; nota final do CdR; intervalos fechados à esquerda",
        "saida": "Somente agregados, sem identificadores de par",
    })
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
