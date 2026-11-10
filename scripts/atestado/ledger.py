"""Ledger versionado da atestação bibliográfica (sem citações nem justificativas).

``exportar_ledger`` lê ``avaliacoes.jsonl`` (e ``artefatos.json`` das obras próprias) de
``.local/biblio/piloto/tese/`` e grava ``docs/atestado/ledger.csv`` (UTF-8, ``;``) e
``docs/atestado/pendencias.yaml``. As funções de checagem (``contextos_sem_atestamento`` e
seguintes) são o núcleo do teste de portão ``tests/test_atestado_ledger.py``.

Ids: as avaliações antigas trazem o ``id`` da fórmula legada (chave + sentença). O exportador as
mapeia ao ``id`` novo de ``citacoes.jsonl`` pelo campo ``id_legado``; o ledger grava o ``id`` novo
e, em ``id_legado``, o antigo. Pendências com id legado também são migradas para o id novo.
O mapeamento de uma avaliação só vale quando é 1→1: se o id antigo corresponde a vários ids novos
(a mesma chave citada mais de uma vez na sentença), a avaliação só vale para o contexto cuja
``sentence_text`` é idêntica à da avaliação (como em ``avaliar.find_reusable``); sem esse texto,
nenhum deles entra no ledger e todos precisam ser atestados de novo.

Pendências: o exportador nunca acrescenta ids novos a ``pendencias.yaml`` sem ``aceitar_novas``
(``--aceitar-novas`` na linha de comando); sem a opção, os ids novos só são listados.
``marcar_baseline`` só marca ids do conjunto congelado em ``tests/data/pendencias_baseline_ids.txt``
e recusa (``BaselineRecusada``) qualquer outro.
"""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

import yaml

from scripts.atestado.avaliar import _same_sentence
from scripts.atestado.common import read_json, read_jsonl, write_text_atomic

LEDGER_COLUMNS = [
    "id", "id_legado", "chave", "arquivo", "capitulo", "veredito_final", "nivel_evidencia", "adjudicado",
    "modelo", "data", "hash_avaliacao",
]
PROBLEMAS = ("NAO_SUSTENTA", "CONTRADIZ")
PROPRIO = "PROPRIO"
NAO_ATESTADOS = ("", "PENDENTE", "DIVERGENTE")
LEDGER_NAME = "ledger.csv"
PENDENCIAS_NAME = "pendencias.yaml"
DECISAO_INICIAL = "pendente"
ORIGEM_BASELINE = "baseline-2026-11-08"
BASELINE_IDS_FILE = Path(__file__).resolve().parents[2] / "tests" / "data" / "pendencias_baseline_ids.txt"
HASH_LEN = 16


class BaselineRecusada(ValueError):
    """``marcar_baseline`` com id fora do conjunto congelado da linha de base."""


def _day(stamp: str | None) -> str:
    return (stamp or "")[:10]


def hash_avaliacao(modelo: str, veredito: str, trecho: str) -> str:
    """Impressão digital da avaliação (modelo, veredito final e trecho da fonte); o trecho não é gravado."""
    payload = "\x1f".join([modelo or "", veredito or "", trecho or ""])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:HASH_LEN]


def _row_from_evaluation(row: dict, meta: dict) -> dict:
    adj = row.get("adjudicacao") or {}
    modelo = adj.get("modelo") or meta.get("modelo", "")
    trecho = adj.get("trecho_fonte") if adj else row.get("trecho_fonte")
    return {
        "id": row["id"],
        "id_legado": "",
        "chave": row["key"],
        "arquivo": row.get("file", ""),
        "capitulo": row.get("chapter", ""),
        "veredito_final": row["veredito_final"],
        "nivel_evidencia": row.get("nivel_evidencia", ""),
        "adjudicado": "sim" if adj else "nao",
        "modelo": modelo,
        "data": _day(adj.get("em")) or _day(meta.get("data_utc")),
        "hash_avaliacao": hash_avaliacao(modelo, row["veredito_final"], trecho or ""),
    }


def _row_from_artifact(row: dict, chapters: dict[str, str], meta: dict) -> dict:
    return {
        "id": row["id"],
        "id_legado": "",
        "chave": row["key"],
        "arquivo": row.get("file", ""),
        "capitulo": chapters.get(row["id"], ""),
        "veredito_final": PROPRIO,
        "nivel_evidencia": str(row.get("veredito", "artefato")).casefold(),
        "adjudicado": "nao",
        "modelo": "",
        "data": _day(meta.get("data_utc")),
        "hash_avaliacao": "",
    }


def _destinos(old_id: str, contexts: dict[str, dict], legado: dict[str, list[str]]) -> list[str]:
    """Ids atuais de um id de avaliação: ele mesmo se for atual; senão os ids novos de mesmo ``id_legado``."""
    if old_id in contexts:
        return [old_id]
    return legado.get(old_id) or [old_id]


def _destinos_avaliacao(ev: dict, contexts: dict[str, dict], legado: dict[str, list[str]]) -> list[str]:
    """Como ``_destinos``, mas um id antigo que corresponde a vários ids novos não divide o veredito.

    Vale só para o contexto cuja ``sentence_text`` é idêntica à da avaliação; sem esse texto, nenhum.
    """
    destinos = _destinos(ev["id"], contexts, legado)
    if len(destinos) <= 1:
        return destinos
    sentence = ev.get("sentence_text")
    return [
        cid for cid in destinos if sentence and _same_sentence(sentence, contexts[cid].get("sentence_text", ""))
    ]


def _place(row: dict, ctx: dict | None) -> dict:
    """Reaponta uma linha ao contexto atual (id novo, id_legado, chave, arquivo e capítulo)."""
    if ctx is None:
        return row
    return {
        **row,
        "id": ctx["id"],
        "id_legado": ctx.get("id_legado", ""),
        "chave": ctx["key"],
        "arquivo": ctx.get("file", row["arquivo"]),
        "capitulo": ctx.get("chapter", row["capitulo"]),
    }


def build_ledger_rows(pilot_dir: Path, out_dir: Path) -> tuple[list[dict], list[str]]:
    """Linhas do ledger e ids ainda sem veredito final (PENDENTE), que não entram."""
    meta = read_json(pilot_dir / "metadados.json", default={}) or {}
    contexts = {c["id"]: c for c in read_jsonl(out_dir / "citacoes.jsonl")}
    legado: dict[str, list[str]] = {}
    for cid, ctx in contexts.items():
        if ctx.get("id_legado"):
            legado.setdefault(ctx["id_legado"], []).append(cid)
    chapters = {cid: c.get("chapter", "") for cid, c in contexts.items()}
    rows: dict[str, dict] = {}
    pendentes: list[str] = []
    for art in read_json(pilot_dir / "artefatos.json", default=[]) or []:
        for new_id in _destinos(art["id"], contexts, legado):
            rows[new_id] = _place(_row_from_artifact({**art, "id": new_id}, chapters, meta), contexts.get(new_id))
    # Avaliação com id atual vale mais que a mapeada por id_legado: o mapeamento só preenche o que falta.
    direct: dict[str, dict] = {}
    mapped: dict[str, dict] = {}
    for ev in read_jsonl(pilot_dir / "avaliacoes.jsonl"):
        flags = set(ev.get("flags") or [])
        if ev.get("veredito_final") in (None, "PENDENTE") or {"pendente", "segunda_passada_pendente"} & flags:
            pendentes.append(ev["id"])
            continue
        for new_id in _destinos_avaliacao(ev, contexts, legado):
            row = _place(_row_from_evaluation({**ev, "id": new_id}, meta), contexts.get(new_id))
            if ev["id"] in contexts:
                direct[new_id] = row
            else:
                mapped.setdefault(new_id, row)
    rows.update({new_id: row for new_id, row in mapped.items() if new_id not in direct})
    rows.update(direct)
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


def build_pendencias(
    rows: list[dict],
    existentes: list[dict],
    aceitar_novas: bool = False,
    marcar_baseline: bool = False,
    baseline_ids: set[str] | frozenset[str] = frozenset(),
) -> tuple[list[dict], list[str], list[str]]:
    """Pendências atualizadas, ids novos e ids resolvidos.

    Decisões e ``origem`` existentes são preservadas (inclusive de entradas com id legado, que
    passam ao id novo). Ids novos só entram em ``current`` com ``aceitar_novas``; são sempre
    devolvidos em ``novos``. ``marcar_baseline`` marca com ``origem`` as entradas que já existiam e
    levanta ``BaselineRecusada`` se alguma entrada com essa origem estiver fora de ``baseline_ids``.
    """
    previous = {p["id"]: p for p in existentes}
    matched: set[str] = set()
    current: list[dict] = []
    novos: list[str] = []
    for row in rows:
        if row["veredito_final"] not in PROBLEMAS:
            continue
        old_id = row["id"] if row["id"] in previous else row.get("id_legado", "")
        old = previous.get(old_id)
        if old is None:
            novos.append(row["id"])
            if not aceitar_novas:
                continue
            old = {}
        else:
            matched.add(old_id)
        entry = {
            "id": row["id"],
            "chave": row["chave"],
            "arquivo": row["arquivo"],
            "decisao": old.get("decisao", DECISAO_INICIAL),
        }
        origem = old.get("origem") or (ORIGEM_BASELINE if marcar_baseline and old else None)
        if origem:
            entry["origem"] = origem
        current.append(entry)
    if marcar_baseline:
        recusados = [p["id"] for p in baseline_fora_do_conjunto(current, baseline_ids)]
        if recusados:
            raise BaselineRecusada(
                f"--marcar-baseline recusado: {len(recusados)} id(s) fora de {BASELINE_IDS_FILE.name} "
                "(problemas novos exigem decisão da autora, não origem de linha de base): " + ", ".join(recusados)
            )
    resolvidos = sorted(set(previous) - matched)
    return current, sorted(novos), resolvidos


def render_pendencias(pendencias: list[dict]) -> str:
    header = (
        "# Problemas abertos aceitos (NAO_SUSTENTA ou CONTRADIZ) da atestação bibliográfica.\n"
        "# Gerado por `scripts/atestar_citacoes.py exportar-ledger`; decisões existentes são preservadas.\n"
        f"# `origem: {ORIGEM_BASELINE}` marca o conjunto aceito na linha de base. Uma pendência nova só entra\n"
        "# com `--aceitar-novas`, visível no diff, e o teste de portão exige que o autor troque `decisao: pendente`.\n"
    )
    body = yaml.safe_dump({"pendencias": pendencias}, sort_keys=False, allow_unicode=True)
    return header + body


def exportar_ledger(
    pilot_dir: Path,
    out_dir: Path,
    dest_dir: Path,
    aceitar_novas: bool = False,
    marcar_baseline: bool = False,
    baseline_ids: set[str] | None = None,
) -> dict:
    """Grava ``ledger.csv`` e ``pendencias.yaml`` em ``dest_dir``.

    Sem ``aceitar_novas``, ids NAO_SUSTENTA/CONTRADIZ novos não entram em ``pendencias.yaml``;
    o chamador deve tratá-los como falha (``pendencias_novas`` não vazio e ``novas_aceitas`` falso).
    Com ``marcar_baseline``, ``baseline_ids`` (padrão: o conteúdo de ``BASELINE_IDS_FILE``) limita os
    ids que podem ter ``origem`` de linha de base; fora dele, ``BaselineRecusada`` e nada é gravado.
    """
    rows, pendentes = build_ledger_rows(pilot_dir, out_dir)
    if not rows:
        raise ValueError(f"{pilot_dir / 'avaliacoes.jsonl'} ausente ou vazio; rode `avaliar` antes")
    if marcar_baseline and baseline_ids is None:
        baseline_ids = load_baseline_ids(BASELINE_IDS_FILE)
    pend_path = dest_dir / PENDENCIAS_NAME
    pendencias, novos, resolvidos = build_pendencias(
        rows, load_pendencias(pend_path), aceitar_novas=aceitar_novas, marcar_baseline=marcar_baseline,
        baseline_ids=baseline_ids or frozenset(),
    )
    write_text_atomic(dest_dir / LEDGER_NAME, render_ledger(rows))
    write_text_atomic(pend_path, render_pendencias(pendencias))
    return {
        "linhas": len(rows),
        "pendentes_sem_veredito": pendentes,
        "pendencias": len(pendencias),
        "pendencias_novas": novos,
        "novas_aceitas": aceitar_novas,
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


def pendencias_sem_decisao(pendencias: list[dict]) -> list[dict]:
    """Entradas de ``pendencias.yaml`` sem ``origem`` de linha de base e com ``decisao`` ainda pendente."""
    return [
        p
        for p in pendencias
        if p.get("origem") != ORIGEM_BASELINE and str(p.get("decisao") or DECISAO_INICIAL).strip() == DECISAO_INICIAL
    ]


def load_baseline_ids(path: Path) -> set[str]:
    """Ids congelados da linha de base (um por linha; ``#`` inicia comentário); vazio se faltar o arquivo."""
    if not path.is_file():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            ids.add(value)
    return ids


def baseline_fora_do_conjunto(pendencias: list[dict], congelado: set[str] | frozenset[str]) -> list[dict]:
    """Entradas com ``origem`` de linha de base cujo id não está no conjunto congelado."""
    return [p for p in pendencias if p.get("origem") == ORIGEM_BASELINE and p["id"] not in congelado]


def proprios_indevidos(ledger: list[dict], chaves_proprias: set[str]) -> list[dict]:
    """Linhas PROPRIO cuja chave não é de obra própria (autoria 'Jardim')."""
    return [row for row in ledger if row["veredito_final"] == PROPRIO and row["chave"] not in chaves_proprias]


def linhas_sem_veredito(ledger: list[dict]) -> list[dict]:
    """Linhas sem veredito atestado: vazio, PENDENTE ou DIVERGENTE (ainda sem adjudicação)."""
    return [row for row in ledger if (row.get("veredito_final") or "") in NAO_ATESTADOS]


def chaves_divergentes(ledger: list[dict], chaves_por_id: dict[str, str]) -> list[dict]:
    """Linhas cuja ``chave`` difere da chave do contexto extraído com o mesmo id."""
    return [row for row in ledger if row["id"] in chaves_por_id and chaves_por_id[row["id"]] != row["chave"]]


def linhas_sem_hash(ledger: list[dict]) -> list[dict]:
    """Linhas de avaliação (não PROPRIO) sem ``hash_avaliacao`` de 16 hexadecimais."""
    def valido(value: str) -> bool:
        return len(value) == HASH_LEN and all(ch in "0123456789abcdef" for ch in value)

    return [
        row
        for row in ledger
        if row["veredito_final"] != PROPRIO and not valido((row.get("hash_avaliacao") or "").strip())
    ]
