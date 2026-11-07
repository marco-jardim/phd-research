"""Registra a especificação candidata antes da coorte nova, sem ajustar modelos."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "gzcmd/v4_candidata.yaml", "gzcmd/M_v4_candidata.md",
    "scripts/fusao_filas.py", "scripts/contexto_obito.py", "gzcmd/ml_oof.py",
)


def build_registry(root, date=None):
    return {
        "status": "candidata; não validada; sem coorte nova",
        "date_utc": date or datetime.now(timezone.utc).isoformat(),
        "hash_convention": "SHA-256 dos bytes, arquivos versionados com LF",
        "sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    output = args.root / "data/v4_candidata_registro.json"
    output.write_text(json.dumps(build_registry(args.root), ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8", newline="\n")
    print("Registro da v4 candidata gravado; nenhuma coorte nova avaliada.")


if __name__ == "__main__":
    main()
