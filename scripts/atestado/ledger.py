"""Ledger versionado da atestação bibliográfica (sem citações nem justificativas).

``exportar_ledger`` lê ``avaliacoes.jsonl`` (e ``artefatos.json`` das obras próprias) de
``.local/biblio/piloto/tese/`` e grava ``docs/atestado/ledger.csv`` (UTF-8, ``;``) e
``docs/atestado/pendencias.yaml``. As funções de checagem (``contextos_sem_atestamento`` e
seguintes) são o núcleo do teste de portão ``tests/test_atestado_ledger.py``.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import yaml

from scripts.atestado.common import read_json, read_jsonl, write_text_atomic

LEDGER_COLUMNS = [
    "id", "chave", "arquivo", "capitulo", "veredito_final", "nivel_evidencia", "adjudicado", "modelo", "data",
]
PROBLEMAS = ("NAO_SUSTENTA", "CONTRADIZ")
PROPRIO = "PROPRIO"
LEDGER_NAME = "ledger.csv"
PENDENCIAS_NAME = "pendencias.yaml"
DECISAO_INICIAL = "pendente"


def _day(stamp: str | None) -> str:
    return (stamp or "")[:10]


def _row_from_evaluation(row: dict, meta: dict) -> dict:
    adj = row.get("adjudicacao") or {}
    return {
        "id": row["id"],
        "chave": row["key"],
        "arquivo": row.get("file", ""),
        "capitulo": row.get("chapter", ""),
        "veredito_final": row["veredito_final"],
        "nivel_evidencia": row.get("nivel_evidencia", ""),
        "adjudicado": "sim" if adj else "nao",
        "modelo": adj.get("modelo") or meta.get("modelo", ""),
        "data": _day(adj.get("em")) or _day(meta.get("data_utc")),
    }


def _row_from_artifact(row: dict, chapters: dict[str, str], meta: dict) -> dict:
    return {
        "id": row["id"],
        "chave": row["key"],
        "arquivo": row.get("file", ""),
        "capitulo": chapters.get(row["id"], ""),
        "veredito_final": PROPRIO,
        "nivel_evidencia": str(row.get("veredito", "artefato")).casefold(),
        "adjudicado": "nao",
        "modelo": "",
        "data": _day(meta.get("data_utc")),
    }


def build_ledger_rows(pilot_dir: Path, out_dir: Path) -> tuple[list[dict], list[str]]:
    """Linhas do ledger e ids ainda sem veredito final (PENDENTE), que não entram."""
    meta = read_json(pilot_dir / "metadados.json", default={}) or {}
    chapters = {c["id"]: c.get("chapter", "") for c in read_jsonl(out_dir / "citacoes.jsonl")}
    rows: dict[str, dict] = {}
    pendentes: list[str] = []
    for art in read_json(pilot_dir / "artefatos.json", default=[]) or []:
        rows[art["id"]] = _row_from_artifact(art, chapters, meta)
    for ev in read_jsonl(pilot_dir / "avaliacoes.jsonl"):
        flags = set(ev.get("flags") or [])
        if ev.get("veredito_final") in (None, "PENDENTE") or {"pendente", "segunda_passada_pendente"} & flags:
            pendentes.append(ev["id"])
            continue
        rows[ev["id"]] = _row_from_evaluation(ev, meta)
    return sorted(rows.values(), key=lambda r: (r["arquivo"], r["chave"], r["id"])), sorted(pendentes)


def render_ledger(rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=LEDGER_COLUMNS, delimiter=";", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def load_ledger(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def load_pendencias(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("pendencias") or [])


def build_pendencias(rows: list[dict], existentes: list[dict]) -> tuple[list[dict], list[str], list[str]]:
    """Pendências atualizadas (decisões existentes preservadas), ids novos e ids resolvidos."""
    previous = {p["id"]: p for p in existentes}
    current: list[dict] = []
    for row in rows:
        if row["veredito_final"] not in PROBLEMAS:
            continue
        old = previous.get(row["id"], {})
        current.append(
            {
                "id": row["id"],
                "chave": row["chave"],
                "arquivo": row["arquivo"],
                "decisao": old.get("decisao", DECISAO_INICIAL),
            }
        )
    ids = {p["id"] for p in current}
    novos = sorted(ids - set(previous))
    resolvidos = sorted(set(previous) - ids)
    return current, novos, resolvidos


def render_pendencias(pendencias: list[dict]) -> str:
    header = (
        "# Problemas abertos aceitos (NAO_SUSTENTA ou CONTRADIZ) da atestação bibliográfica.\n"
        "# Gerado por `scripts/atestar_citacoes.py exportar-ledger`; decisões existentes são preservadas.\n"
        "# Uma pendência nova só entra por aqui, visível no diff; `decisao` começa em \"pendente\".\n"
    )
    body = yaml.safe_dump({"pendencias": pendencias}, sort_keys=False, allow_unicode=True)
    return header + body


def exportar_ledger(pilot_dir: Path, out_dir: Path, dest_dir: Path) -> dict:
    """Grava ``ledger.csv`` e ``pendencias.yaml`` em ``dest_dir``."""
    rows, pendentes = build_ledger_rows(pilot_dir, out_dir)
    if not rows:
        raise ValueError(f"{pilot_dir / 'avaliacoes.jsonl'} ausente ou vazio; rode `avaliar` antes")
    pend_path = dest_dir / PENDENCIAS_NAME
    pendencias, novos, resolvidos = build_pendencias(rows, load_pendencias(pend_path))
    write_text_atomic(dest_dir / LEDGER_NAME, render_ledger(rows))
    write_text_atomic(pend_path, render_pendencias(pendencias))
    return {
        "linhas": len(rows),
        "pendentes_sem_veredito": pendentes,
        "pendencias": len(pendencias),
        "pendencias_novas": novos,
        "pendencias_resolvidas": resolvidos,
    }


# ---------------------------------------------------------------------------
# Portão: o que o teste exige do ledger
# ---------------------------------------------------------------------------
def contextos_sem_atestamento(context_ids: list[str], ledger: list[dict]) -> list[str]:
    """Ids de contexto atuais ausentes do ledger (frases novas ou reescritas)."""
    attested = {row["id"] for row in ledger}
    return sorted({cid for cid in context_ids if cid not in attested})


def problemas_fora_das_pendencias(ledger: list[dict], pendencias: list[dict]) -> list[dict]:
    """Linhas NAO_SUSTENTA/CONTRADIZ do ledger que não constam em ``pendencias.yaml``."""
    listed = {p["id"] for p in pendencias}
    return [row for row in ledger if row["veredito_final"] in PROBLEMAS and row["id"] not in listed]


def proprios_indevidos(ledger: list[dict], chaves_proprias: set[str]) -> list[dict]:
    """Linhas PROPRIO cuja chave não é de obra própria (autoria 'Jardim')."""
    return [row for row in ledger if row["veredito_final"] == PROPRIO and row["chave"] not in chaves_proprias]


def linhas_sem_veredito(ledger: list[dict]) -> list[dict]:
    return [row for row in ledger if not row.get("veredito_final") or row["veredito_final"] == "PENDENTE"]
