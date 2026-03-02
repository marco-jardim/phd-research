"""Banco de verificações manuais de referências.

Armazena decisões de revisão humana para que o ref_audit não re-sinalize
entradas já verificadas. O arquivo de dados fica em
``scripts/ref_audit/data/verified_refs.json``.

Esquema (cada entrada):
{
  "<bib_key>": {
    "status": "VERIFIED_OK" | "VERIFIED_ACCEPT" | "VERIFIED_REJECT",
    "original_scores": {"composite": "REF_REVIEW", "l1": "L1_WARN_META", ...},
    "reason": "Motivo da decisão",
    "verified_by": "human | agent",
    "verified_date": "2026-02-08"
  }
}

Semântica dos status:
- VERIFIED_OK     : entrada auditada e confirmada correta / sem ação necessária.
- VERIFIED_ACCEPT : entrada tem limitações conhecidas (e.g., DOI inexistente,
                    autor institucional), mas foi aceita deliberadamente.
- VERIFIED_REJECT : entrada rejeitada — deve ser corrigida ou removida.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parent / "data" / "verified_refs.json"


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def load_verified(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Carrega o banco de verificações do disco."""
    p = path or _DB_PATH
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_verified(db: dict[str, dict[str, Any]], path: Path | None = None) -> Path:
    """Salva o banco de verificações no disco."""
    p = path or _DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False, sort_keys=True)
    logger.info("Banco de verificações salvo: %s (%d entradas)", p, len(db))
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def add_entry(
    db: dict[str, dict[str, Any]],
    bib_key: str,
    status: str,
    reason: str,
    *,
    original_scores: dict[str, str] | None = None,
    verified_by: str = "human",
) -> None:
    """Adiciona ou atualiza uma entrada no banco."""
    if status not in ("VERIFIED_OK", "VERIFIED_ACCEPT", "VERIFIED_REJECT"):
        raise ValueError(f"Status inválido: {status}")
    db[bib_key] = {
        "status": status,
        "original_scores": original_scores or {},
        "reason": reason,
        "verified_by": verified_by,
        "verified_date": date.today().isoformat(),
    }


def is_verified_ok(db: dict[str, dict[str, Any]], bib_key: str) -> bool:
    """Retorna True se a entrada foi verificada e aceita."""
    entry = db.get(bib_key)
    if not entry:
        return False
    return entry["status"] in ("VERIFIED_OK", "VERIFIED_ACCEPT")
