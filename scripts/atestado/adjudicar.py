"""Etapa 3b: adjudicação dos contextos DIVERGENTE por um modelo mais forte (Claude Opus).

Só os contextos de ``avaliacoes.jsonl`` com ``veredito_final == "DIVERGENTE"`` são enviados, uma
requisição por chave (todos os contextos divergentes da chave juntos). Layout (prefixo em cache):

* ``system[0]``: rubrica de adjudicação, fixa, ``cache_control`` efêmero;
* ``messages[0].content[0]`` (bloco A): o mesmo texto de evidência enviado nas passadas de avaliação
  (mesmo construtor, mesmo ``--max-chars``, mesmo corte);
* ``messages[0].content[1]`` (bloco B): por contexto, id, sentença, parágrafo e as duas avaliações
  anteriores ("Avaliação 1" e "Avaliação 2").

``temperature`` nunca é enviada. O trecho citado passa pela mesma verificação mecânica da avaliação;
trecho não encontrado derruba o veredito para NAO_VERIFICAVEL com a marca ``trecho_nao_encontrado``.
O resultado atualiza ``avaliacoes.jsonl`` no lugar (``passada1``/``passada2`` e o veredito anterior
ficam; entra ``adjudicacao``). O estado fica na seção ``adjudicacao`` de ``checkpoint.json`` (lotes
e respostas), de modo que uma nova execução consulta o lote registrado em vez de reenviá-lo.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

from . import avaliar as av
from .common import read_json, read_jsonl, utc_now_iso, write_json_atomic, write_jsonl_atomic
from .evidencias import cache_path
from .prompts import ADJ_RUBRIC, build_adjudication_block, build_evidence_block
from .trechos import load_english_queries
from .verificar import DIVERGENTE, make_assessment

ADJ_PASSADA = 3  # rótulo da "passada" no checkpoint e no arquivo de lote
CONFIANCAS = ("alta", "media", "baixa")


class AdjCheckpoint(av.Checkpoint):
    """Checkpoint de ``checkpoint.json`` que guarda requisições, lotes e falhas da adjudicação na
    seção ``adjudicacao`` e preserva o restante do arquivo (passadas de avaliação)."""

    def __init__(self, path: Path, meta: dict, sleep: Callable[[float], None] = time.sleep) -> None:
        super().__init__(path, meta, sleep)
        self.section = self.state.setdefault("adjudicacao", {"requisicoes": {}, "lotes": [], "falhas": []})

    @property
    def done(self) -> dict[str, dict]:
        return self.section["requisicoes"]

    @property
    def batches(self) -> list[dict]:
        return self.section["lotes"]

    def fail(self, custom_id: str, motivo: str) -> None:
        self.section["falhas"].append({"custom_id": custom_id, "motivo": motivo, "em": utc_now_iso()})
        self.save()


# ---------------------------------------------------------------------------
# Planejamento
# ---------------------------------------------------------------------------
def build_adj_params(settings: av.Settings, evidence_text: str, contexts_text: str) -> dict:
    """Parâmetros da requisição; nunca inclui ``temperature``."""
    return {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "system": [{"type": "text", "text": ADJ_RUBRIC, "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": evidence_text},
                {"type": "text", "text": contexts_text},
            ]}
        ],
    }


def make_adj_custom_id(key: str, items: list[tuple[dict, dict]], model: str, evidence_text: str) -> str:
    """Identificador estável (``^[A-Za-z0-9_-]{1,64}$``): muda se contextos, avaliações anteriores,
    modelo ou evidência mudarem."""
    priors = json.dumps(
        [[ctx["id"], row.get("passada1"), row.get("passada2")] for ctx, row in items],
        sort_keys=True, ensure_ascii=False,
    )
    digest = hashlib.sha256(
        "|".join([
            key, model, hashlib.sha256(evidence_text.encode()).hexdigest(),
            hashlib.sha256(priors.encode()).hexdigest(),
        ]).encode()
    ).hexdigest()
    return f"adj-{digest[:24]}"


def plan_adjudication(
    groups: dict[str, list[tuple[dict, dict]]],
    evidence: dict[str, str],
    capitulo: str,
    settings: av.Settings,
    global_scope: bool = False,
) -> list[av.PlannedRequest]:
    """Uma requisição por chave, com todos os contextos divergentes da chave."""
    planned: list[av.PlannedRequest] = []
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda it: (it[0].get("file", ""), it[0].get("line", 0), it[0]["id"]))
        params = build_adj_params(
            settings, evidence[key], build_adjudication_block(capitulo, key, items, global_scope)
        )
        cid = make_adj_custom_id(key, items, settings.model, evidence[key])
        planned.append(av.PlannedRequest(cid, key, ADJ_PASSADA, [ctx["id"] for ctx, _row in items], params))
    return planned


def select_todo(
    planned: list[av.PlannedRequest], cp: AdjCheckpoint, refazer_falhas: bool
) -> tuple[list[av.PlannedRequest], list[av.PlannedRequest], list[av.PlannedRequest]]:
    """Separa o que enviar, o que está em lote ainda não baixado e o que falhou e não será reenviado.

    Uma requisição registrada em lote já baixado sem resposta (erro do item) só é reenviada com
    ``refazer_falhas``; uma resposta com erro de conteúdo (JSON inválido, truncada) idem.
    """
    recorded: dict[str, dict] = {}
    for batch in cp.batches:
        for cid in batch.get("custom_ids", []):
            recorded[cid] = batch
    send: list[av.PlannedRequest] = []
    waiting: list[av.PlannedRequest] = []
    failed: list[av.PlannedRequest] = []
    for req in planned:
        entry = cp.done.get(req.custom_id)
        if entry is not None:
            if entry.get("erro"):
                (send if refazer_falhas else failed).append(req)
            continue
        batch = recorded.get(req.custom_id)
        if batch is not None and not batch.get("baixado"):
            waiting.append(req)
        elif batch is not None:
            (send if refazer_falhas else failed).append(req)
        else:
            send.append(req)
    return send, waiting, failed


# ---------------------------------------------------------------------------
# Aplicação do resultado
# ---------------------------------------------------------------------------
def batch_id_of(cp: AdjCheckpoint, custom_id: str) -> str | None:
    found = None
    for batch in cp.batches:
        if custom_id in batch.get("custom_ids", []) and batch.get("id"):
            found = batch["id"]
    return found


def apply_adjudication(row: dict, assessment: dict, raw: dict, model: str, batch_id: str | None, em: str) -> None:
    """Grava o veredito adjudicado na linha, preservando as avaliações anteriores."""
    confianca = str(raw.get("confianca") or "").strip().lower()
    adjudicacao = {
        "modelo": model,
        "veredito": assessment["veredito"],
        "veredito_modelo": assessment["veredito_modelo"],
        "trecho_fonte": assessment["trecho_fonte"],
        "trecho_verificado": assessment["trecho_verificado"],
        "justificativa": assessment["justificativa"],
        "problema": assessment["problema"],
        "confianca": confianca if confianca in CONFIANCAS else None,
        "flags": list(assessment["flags"]),
        "batch_id": batch_id,
        "em": em,
    }
    kept = [f for f in (row.get("flags") or []) if f.startswith("reaproveitado_de=")]
    row["veredito_pre_adjudicacao"] = row["veredito_final"]
    row["flags_pre_adjudicacao"] = list(row.get("flags") or [])
    row["adjudicacao"] = adjudicacao
    row["veredito_final"] = assessment["veredito"]
    row["trecho_fonte"] = assessment["trecho_fonte"]
    row["trecho_verificado"] = assessment["trecho_verificado"]
    row["justificativa"] = assessment["justificativa"]
    row["problema"] = assessment["problema"]
    row["flags"] = [*kept, "adjudicado", *adjudicacao["flags"]]


def run_adjudicacao(
    *,
    capitulo: str,
    contexts: list[dict],
    out_dir: Path,
    pilot_dir: Path,
    settings: av.Settings,
    client: object,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda text: None,
    escopo: str = "capitulo",
) -> dict:
    """Envia os contextos DIVERGENTE ao modelo de adjudicação e atualiza ``avaliacoes.jsonl``."""
    rows = read_jsonl(pilot_dir / "avaliacoes.jsonl")
    divergent = [r for r in rows if r.get("veredito_final") == DIVERGENTE]
    meta_path = pilot_dir / "metadados.json"
    if not divergent:
        log("nenhum contexto DIVERGENTE para adjudicar")
        return {"rows": rows, "metadados": read_json(meta_path, default={}), "incompleto": False,
                "adjudicados": 0}

    cp = AdjCheckpoint(
        pilot_dir / "checkpoint.json",
        {"capitulo": capitulo, "modelo": settings.model, "criado_em": utc_now_iso()},
        sleep,
    )
    rt = av.Runtime(client, settings, cp, pilot_dir / "lotes", sleep, log)
    by_ctx = {c["id"]: c for c in contexts}

    evidence: dict[str, str] = {}
    groups: dict[str, list[tuple[dict, dict]]] = {}
    queries_en = load_english_queries(out_dir)
    for key in sorted({r["key"] for r in divergent}):
        mine = [r for r in divergent if r["key"] == key]
        record = read_json(cache_path(out_dir, key)) or {"key": key}
        # os mesmos contextos da chave que a avaliação usou, para reproduzir a mesma seleção de trechos
        key_contexts = [c for c in contexts if c["key"] == key]
        text, level = build_evidence_block(record, settings.max_chars, key_contexts, out_dir, queries_en)
        if level == "nenhum":
            log(f"{key}: sem evidência; contextos ficam DIVERGENTE")
            continue
        fingerprints = {r["evidencia_fingerprint"] for r in mine if r.get("evidencia_fingerprint")}
        if fingerprints and fingerprints != {av.evidence_fingerprint(text)}:
            log(f"{key}: a evidência atual difere da enviada na avaliação (confira --max-chars e as "
                "fontes); contextos ficam DIVERGENTE")
            continue
        items = []
        for row in mine:
            ctx = by_ctx.get(row["id"])
            if ctx is None:
                log(f"{key}: contexto {row['id']} ausente de citacoes.jsonl; fica DIVERGENTE")
                continue
            items.append((ctx, row))
        if items:
            evidence[key] = text
            groups[key] = items

    planned = plan_adjudication(groups, evidence, capitulo, settings, escopo == "tese")
    send, waiting, failed = select_todo(planned, cp, settings.refazer_falhas)
    if failed:
        log(f"{len(failed)} requisição(ões) com falha registrada não reenviada(s); use --refazer-falhas")
    if settings.modo == "sync":
        if send:
            av.run_sync(rt, send)
    elif send or waiting:
        av.run_batch(rt, [*send, *waiting], ADJ_PASSADA)

    now = utc_now_iso()
    row_by_id = {r["id"]: r for r in rows}
    adjudicados = 0
    for req in planned:
        entry = cp.done.get(req.custom_id)
        if not entry or entry.get("erro"):
            continue
        items_by_id = {str(item.get("id")): item for item in entry["itens"]}
        batch_id = batch_id_of(cp, req.custom_id) if entry.get("modo") == "batch" else None
        for ctx_id in req.ids:
            raw = items_by_id.get(ctx_id)
            row = row_by_id[ctx_id]
            if raw is None or row.get("veredito_final") != DIVERGENTE:
                continue
            apply_adjudication(row, make_assessment(raw, evidence[req.key]), raw, settings.model, batch_id, now)
            adjudicados += 1
    write_jsonl_atomic(pilot_dir / "avaliacoes.jsonl", rows)

    restantes = sum(1 for r in rows if r.get("veredito_final") == DIVERGENTE)
    incomplete = restantes > 0
    meta = read_json(meta_path, default={}) or {}
    meta.setdefault("uso", {})["adjudicacao"] = av.usage_totals(cp)
    meta["adjudicacao"] = {
        "modelo": settings.model,
        "data_utc": utc_now_iso(),
        "parametros": {
            "max_tokens": settings.max_tokens,
            "temperature_enviada": False,
            "modo": settings.modo,
            "max_chars": settings.max_chars,
        },
        "lotes": [b["id"] for b in cp.batches if b.get("id")],
        "requisicoes": len(planned),
        "contextos_divergentes": len(divergent),
        "contextos_adjudicados": sum(1 for r in rows if r.get("adjudicacao")),
        "contextos_ainda_divergentes": restantes,
        "falhas": len(cp.section["falhas"]),
        "status": "incompleto" if incomplete else "completo",
    }
    write_json_atomic(meta_path, meta)
    return {"rows": rows, "metadados": meta, "incompleto": incomplete, "adjudicados": adjudicados}
