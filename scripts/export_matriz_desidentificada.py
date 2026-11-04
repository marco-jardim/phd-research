#!/usr/bin/env python3
"""Exportação restrita de potenciais pares, sem alterar o pair_id interno.

As 29 colunas-base são 28 subescores (sem CODMUNRES uf prox) e nota final.
PAR preserva 0 (não-par), 1 (vínculo pelo corte) e 2 (vínculo por revisão).
pair_id_hmac = HMAC-SHA256(key, "COMPREC|REFREC|PASSO")[:16], em hexadecimal.
A chave PAIR_ID_HMAC_KEY contém 32 bytes em 64 caracteres hexadecimais e fica
no ambiente ou no .env local, nunca na matriz ou nos metadados. A matriz é
pseudonimizada, não uma garantia de anonimização. Não foi depositada no Zenodo.

Uso: python scripts/export_matriz_desidentificada.py --extrato <csv> --saida <csv>
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.feature_engineering import SCORE_COLS

SCRIPT_VERSION = "1.0.0"
OUTPUT_COLS = ["pair_id_hmac", "passo", *SCORE_COLS, "PAR"]
EXPECTED_ROWS = 61_696
EXPECTED_LINKS = 247
DATE_PATTERN = r"(?:\d{8}(?:\.0)?|\d{2}/\d{2}/\d{4})"


def load_hmac_key(env_path: Path | None = None) -> bytes:
    """Lê só a chave necessária, sem registrar valores ou modificar o ambiente."""
    value = os.environ.get("PAIR_ID_HMAC_KEY")
    if value is None:
        path = env_path if env_path is not None else Path.cwd() / ".env"
        if path.is_file():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                name, separator, candidate = line.strip().partition("=")
                if separator and name.strip() == "PAIR_ID_HMAC_KEY":
                    value = candidate.strip().strip("\"'")
    if not value:
        raise ValueError(
            "PAIR_ID_HMAC_KEY ausente: configure o ambiente ou o .env local."
        )
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("PAIR_ID_HMAC_KEY deve conter 64 caracteres hexadecimais.")
    return bytes.fromhex(value)


def make_pair_id_hmac(comprec: str, refrec: str, passo: str, key: bytes) -> str:
    """Preserva a composição textual e o truncamento, mas exige segredo externo."""
    raw = f"{comprec}|{refrec}|{passo}".encode()
    return hmac.new(key, raw, hashlib.sha256).hexdigest()[:16]


def validate_matrix(matrix: pd.DataFrame, source: pd.DataFrame) -> None:
    """Recusa esquema, datas e cardinalidades incompatíveis com a exportação.

    Igualdade de cardinalidade é uma heurística conservadora, não prova de PII.
    Em amostras pequenas pode recusar escores legítimos; não se relaxa no CLI.
    O pseudônimo tem formato próprio de 16 caracteres hexadecimais.
    """
    if list(matrix.columns) != OUTPUT_COLS:
        raise ValueError("Esquema de saída fora da lista permitida.")
    id_cardinalities = {source[col].nunique() for col in ("REFREC", "COMPREC")}
    for col in matrix.columns:
        values = matrix[col].astype(str)
        if values.str.fullmatch(DATE_PATTERN).any():
            raise ValueError("Possível data detectada na matriz de saída.")
        if matrix[col].nunique() in id_cardinalities:
            raise ValueError(
                "Cardinalidade de saída coincide com identificador interno."
            )
    if not matrix["pair_id_hmac"].str.fullmatch(r"[0-9a-f]{16}").all():
        raise ValueError("Formato inválido de pair_id_hmac.")
    if not matrix["pair_id_hmac"].is_unique:
        raise ValueError("pair_id_hmac não é unívoco por linha.")


def build_matrix(
    source: pd.DataFrame,
    key: bytes,
    *,
    expected_rows: int = EXPECTED_ROWS,
    expected_links: int = EXPECTED_LINKS,
) -> pd.DataFrame:
    """Seleciona somente a lista permitida; campos extras nunca são serializados."""
    source = source.copy()
    source.columns = [col.split(",")[0] for col in source.columns]
    required = ["COMPREC", "REFREC", "PASSO", *SCORE_COLS, "PAR"]
    if not source.columns.is_unique or not set(required).issubset(source.columns):
        raise ValueError("Extrato com colunas obrigatórias ausentes ou duplicadas.")
    if len(source) != expected_rows:
        raise ValueError(f"Esperadas {expected_rows} linhas; recebidas {len(source)}.")
    for col in ("COMPREC", "REFREC", "PASSO"):
        if source[col].isna().any() or source[col].astype(str).str.strip().eq("").any():
            raise ValueError("Identificadores ou passo ausentes no extrato.")

    matrix = pd.DataFrame(index=source.index)
    matrix["pair_id_hmac"] = [
        make_pair_id_hmac(str(comp), str(ref), str(step), key)
        for comp, ref, step in source[["COMPREC", "REFREC", "PASSO"]].itertuples(
            index=False, name=None
        )
    ]
    numeric_columns = [
        ("passo", "PASSO"), *[(c, c) for c in SCORE_COLS], ("PAR", "PAR")
    ]
    for output, original in numeric_columns:
        values = source[original].astype(str).str.strip()
        if values.str.fullmatch(DATE_PATTERN).any():
            raise ValueError("Possível data detectada em campo numérico permitido.")
        numbers = pd.to_numeric(
            values.str.replace(",", ".", regex=False), errors="coerce"
        )
        if not np.isfinite(numbers.to_numpy(dtype=float)).all():
            raise ValueError("Campo permitido contém valor não numérico ou ausente.")
        matrix[output] = numbers
    if not matrix["PAR"].isin([0, 1, 2]).all():
        raise ValueError("PAR deve preservar a codificação 0/1/2 do extrato.")
    if not matrix["passo"].mod(1).eq(0).all():
        raise ValueError("Passo de blocagem deve ser inteiro.")
    matrix[["passo", "PAR"]] = matrix[["passo", "PAR"]].astype(int)
    if int(matrix["PAR"].isin([1, 2]).sum()) != expected_links:
        raise ValueError(f"Esperados {expected_links} vínculos (PAR igual a 1 ou 2).")
    validate_matrix(matrix, source)
    return matrix


def export_matrix(extrato: Path, saida: Path, *, env_path: Path | None = None) -> dict:
    """Valida antes de gravar CSV e metadados agregados, sem valores de origem."""
    key = load_hmac_key(env_path)
    source = pd.read_csv(extrato, sep=";", dtype=str, keep_default_na=False)
    matrix = build_matrix(source, key)
    saida.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(saida, sep=";", index=False, encoding="utf-8", lineterminator="\n")
    metadata = {
        "n": len(matrix),
        "vinculos": int(matrix["PAR"].isin([1, 2]).sum()),
        "colunas": list(matrix.columns),
        "sha256": hashlib.sha256(saida.read_bytes()).hexdigest(),
        "data": datetime.now(UTC).isoformat(),
        "versao_script": SCRIPT_VERSION,
    }
    saida.with_name(f"{saida.stem}_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extrato", type=Path, required=True)
    parser.add_argument(
        "--saida", type=Path, default=Path("data/export/matriz_desidentificada.csv")
    )
    args = parser.parse_args()
    try:
        load_hmac_key()
    except ValueError as exc:
        parser.error(str(exc))
    try:
        metadata = export_matrix(args.extrato, args.saida)
    except (ValueError, OSError):
        # Erros de leitura/conversão externos podem conter valores do extrato.
        parser.exit(
            1, "Exportação recusada: confira esquema, valores e contagens do extrato.\n"
        )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
