"""Etapa 3: avaliação por LLM (Anthropic) com cache de prompt e Message Batches.

Layout de cada requisição (prefixo em cache idêntico byte a byte entre requisições):

* ``system[0]``: rubrica fixa, ``cache_control`` efêmero;
* ``messages[0].content[0]`` (bloco A): evidência da chave, com ``cache_control`` quando a mesma
  chave precisa de mais de uma requisição (vários blocos de contextos ou segunda passada);
* ``messages[0].content[1]`` (bloco B): lista de contextos.

Uma requisição por (chave, capítulo, bloco de até ``max_contextos`` contextos). A segunda passada
(cega) reutiliza o mesmo ``system`` e o mesmo bloco A, para compartilhar o cache, e difere na
redação das instruções do bloco B e na ordem dos contextos; ela não vê o veredito da primeira.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .common import (
    DEFAULT_MODEL,
    ESCOPO_TESE,
    read_json,
    read_jsonl,
    utc_now_iso,
    write_json_atomic,
    write_jsonl_atomic,
)
from .evidencias import BibRef, cache_path, classify_key
from .numeros import numeros_nao_encontrados, prepare_evidence
from .prompts import SYSTEM_RUBRIC, build_contexts_block, build_evidence_block_info
from .trechos import load_english_queries
from .verificar import combine_passes, make_assessment, needs_second_pass

USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
CHECKPOINT_VERSION = 1


# ---------------------------------------------------------------------------
# Configuração e planejamento
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    max_tokens: int = 4096
    temperature: float = 0.0
    send_temperature: bool = False  # o Sonnet atual rejeita temperature; só enviar se pedido
    modo: str = "batch"
    max_chars: int = 120000
    max_contextos: int = 12
    concorrencia: int = 2
    poll_inicial_s: float = 30.0
    poll_max_s: float = 300.0
    espera_max_s: float = 25 * 3600.0
    refazer_falhas: bool = False


@dataclass
class PlannedRequest:
    custom_id: str
    key: str
    passada: int
    ids: list[str]
    params: dict = field(repr=False)


def build_params(settings: Settings, evidence_text: str, contexts_text: str, cache_evidence: bool) -> dict:
    block_a: dict = {"type": "text", "text": evidence_text}
    if cache_evidence:
        block_a["cache_control"] = {"type": "ephemeral"}
    params: dict = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "system": [{"type": "text", "text": SYSTEM_RUBRIC, "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": [block_a, {"type": "text", "text": contexts_text}]}
        ],
    }
    if settings.send_temperature:
        params["temperature"] = settings.temperature
    return params


def chunked(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def make_custom_id(passada: int, key: str, ids: list[str], model: str, evidence_text: str) -> str:
    """Identificador estável (``^[A-Za-z0-9_-]{1,64}$``): muda se o conteúdo mudar."""
    digest = hashlib.sha256(
        "|".join([key, ",".join(ids), model, hashlib.sha256(evidence_text.encode()).hexdigest()]).encode()
    ).hexdigest()
    return f"p{passada}-{digest[:24]}"


def plan_requests(
    groups: dict[str, list[dict]],
    evidence: dict[str, str],
    capitulo: str,
    settings: Settings,
    passada: int,
    cached_before: frozenset[str] | set[str] = frozenset(),
    global_scope: bool = False,
) -> list[PlannedRequest]:
    """Requisições de uma passada. ``groups`` mapeia chave para contextos (ordem estável).

    No escopo global os contextos da chave vêm de vários capítulos e viajam juntos: uma requisição
    por chave (várias só se passarem de ``max_contextos``, e então o bloco de evidência é cacheado).
    """
    planned: list[PlannedRequest] = []
    for key in sorted(groups):
        contexts = sorted(groups[key], key=lambda c: (c.get("file", ""), c.get("line", 0), c["id"]))
        blocks = chunked(contexts, settings.max_contextos)
        cache_evidence = len(blocks) > 1 or key in cached_before
        for block in blocks:
            ids = [c["id"] for c in block]
            params = build_params(
                settings,
                evidence[key],
                build_contexts_block(capitulo, key, block, passada, global_scope),
                cache_evidence,
            )
            cid = make_custom_id(passada, key, ids, settings.model, evidence[key])
            planned.append(PlannedRequest(cid, key, passada, ids, params))
    return planned


# ---------------------------------------------------------------------------
# Resposta do modelo
# ---------------------------------------------------------------------------
def message_text(message) -> str:
    return "".join(
        getattr(block, "text", "") for block in message.content if getattr(block, "type", "") == "text"
    )


def parse_items(text: str) -> list[dict]:
    """Array JSON da resposta (tolera cercas de código e texto ao redor)."""
    start = text.find("[")
    if start < 0:
        raise ValueError("sem array JSON")
    value, _end = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("array JSON inválido")
    return value


def usage_of(message) -> dict[str, int]:
    usage = getattr(message, "usage", None)
    return {name: int(getattr(usage, name, 0) or 0) for name in USAGE_FIELDS}


def done_entry(message, planned: PlannedRequest, modo: str) -> dict:
    """Registro de checkpoint de uma resposta (guarda só os objetos por contexto, não o texto bruto)."""
    entry = {
        "key": planned.key,
        "passada": planned.passada,
        "ids": planned.ids,
        "modo": modo,
        "uso": usage_of(message),
        "stop_reason": getattr(message, "stop_reason", None),
        "itens": [],
        "erro": None,
    }
    if entry["stop_reason"] == "max_tokens":
        entry["erro"] = "resposta_truncada"
        return entry
    try:
        entry["itens"] = parse_items(message_text(message))
    except ValueError as exc:
        entry["erro"] = f"json_invalido: {exc}"
    return entry


# ---------------------------------------------------------------------------
# Chamadas à API (retry, sync e lote)
# ---------------------------------------------------------------------------
class ApiCallError(Exception):
    """Falha da API após as retentativas (sem texto do provedor)."""


def is_api_exception(exc: Exception) -> bool:
    return hasattr(exc, "status_code") or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}


def is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        return True
    return status in (408, 409, 429, 529) or status >= 500


def with_retry(call: Callable[[], object], sleep: Callable[[float], None], attempts: int = 5):
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            if not is_api_exception(exc):
                raise
            if not is_retryable(exc) or attempt == attempts - 1:
                raise ApiCallError(
                    f"{type(exc).__name__} status={getattr(exc, 'status_code', None)}"
                ) from exc
            sleep(3.0 * 2**attempt)
    raise ApiCallError("tentativas esgotadas")


def make_client(api_key: str):
    import anthropic

    return anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=300)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
class Checkpoint:
    """Estado retomável: respostas concluídas por custom_id e lotes enviados."""

    def __init__(self, path: Path, meta: dict, sleep: Callable[[float], None] = time.sleep) -> None:
        self.path = path
        self.sleep = sleep
        stored = read_json(path)
        if stored is not None and stored.get("versao") != CHECKPOINT_VERSION:
            raise ValueError("versão de checkpoint incompatível")
        self.state = stored or {"versao": CHECKPOINT_VERSION, **meta, "requisicoes": {}, "lotes": [], "falhas": []}

    @property
    def done(self) -> dict[str, dict]:
        return self.state["requisicoes"]

    @property
    def batches(self) -> list[dict]:
        return self.state["lotes"]

    def save(self) -> None:
        write_json_atomic(self.path, self.state, sleep=self.sleep)

    def record(self, custom_id: str, entry: dict) -> None:
        self.done[custom_id] = entry
        self.save()

    def fail(self, custom_id: str, motivo: str) -> None:
        self.state["falhas"].append({"custom_id": custom_id, "motivo": motivo, "em": utc_now_iso()})
        self.save()

    def in_flight(self) -> set[str]:
        return {cid for batch in self.batches if not batch.get("baixado") for cid in batch["custom_ids"]}


def pending(planned: list[PlannedRequest], cp: Checkpoint, refazer_falhas: bool) -> list[PlannedRequest]:
    out = []
    for req in planned:
        entry = cp.done.get(req.custom_id)
        if entry is None or (refazer_falhas and entry.get("erro")):
            out.append(req)
    return out


@dataclass
class Runtime:
    client: object
    settings: Settings
    cp: Checkpoint
    lotes_dir: Path
    sleep: Callable[[float], None] = time.sleep
    log: Callable[[str], None] = lambda text: None


def run_sync(rt: Runtime, todo: list[PlannedRequest]) -> None:
    def call(req: PlannedRequest):
        return with_retry(lambda: rt.client.messages.create(**req.params), rt.sleep)

    with ThreadPoolExecutor(max_workers=max(1, rt.settings.concorrencia)) as pool:
        futures = {pool.submit(call, req): req for req in todo}
        for future in as_completed(futures):
            req = futures[future]
            try:
                message = future.result()
            except ApiCallError as exc:
                rt.cp.fail(req.custom_id, str(exc))
                rt.log(f"falha em {req.custom_id}: {exc}")
                continue
            rt.cp.record(req.custom_id, done_entry(message, req, "sync"))


def write_batch_file(rt: Runtime, todo: list[PlannedRequest], passada: int) -> Path:
    digest = hashlib.sha256("|".join(r.custom_id for r in todo).encode()).hexdigest()[:10]
    path = rt.lotes_dir / f"lote_p{passada}_{digest}.jsonl"
    write_jsonl_atomic(path, ({"custom_id": r.custom_id, "params": r.params} for r in todo))
    return path


def submit_batch(rt: Runtime, todo: list[PlannedRequest], passada: int) -> None:
    uncertain = [b for b in rt.cp.batches if b.get("id") is None]
    if uncertain:
        raise RuntimeError(
            "envio de lote anterior sem id registrado: confira os lotes na conta Anthropic "
            "e reconcilie o checkpoint manualmente; não reenviar"
        )
    request_file = write_batch_file(rt, todo, passada)
    entry = {
        "id": None, "passada": passada, "status": "enviando", "baixado": False,
        "custom_ids": [r.custom_id for r in todo], "arquivo": request_file.name,
        "enviado_em": utc_now_iso(),
    }
    rt.cp.batches.append(entry)
    rt.cp.save()
    try:
        batch = with_retry(
            lambda: rt.client.messages.batches.create(
                requests=[{"custom_id": r.custom_id, "params": r.params} for r in todo]
            ),
            rt.sleep,
            attempts=1,
        )
    except ApiCallError as exc:
        status = getattr(exc.__cause__, "status_code", None)
        if status is not None and 400 <= status < 500 and status not in (408, 409, 429):
            # Recusa explícita: o lote não foi criado, então o envio pode ser repetido.
            rt.cp.batches.remove(entry)
            rt.cp.save()
        raise
    entry["id"] = batch.id
    entry["status"] = batch.processing_status
    rt.cp.save()
    rt.log(f"lote enviado: {batch.id} ({len(todo)} requisições)")


def wait_for_batch(rt: Runtime, entry: dict) -> bool:
    """Consulta com backoff até o lote terminar; False se estourar a espera máxima."""
    delay = rt.settings.poll_inicial_s
    waited = 0.0
    while True:
        batch = with_retry(lambda: rt.client.messages.batches.retrieve(entry["id"]), rt.sleep)
        entry["status"] = batch.processing_status
        rt.cp.save()
        if batch.processing_status == "ended":
            return True
        if waited >= rt.settings.espera_max_s:
            return False
        rt.sleep(delay)
        waited += delay
        delay = min(delay * 1.5, rt.settings.poll_max_s)


def download_batch(rt: Runtime, entry: dict, by_id: dict[str, PlannedRequest]) -> None:
    for item in rt.client.messages.batches.results(entry["id"]):
        cid = item.custom_id
        req = by_id.get(cid)
        if req is None:
            rt.cp.fail(cid, "custom_id desconhecido no lote")
            continue
        if item.result.type != "succeeded":
            error_type = getattr(getattr(item.result, "error", None), "type", None)
            rt.cp.fail(cid, f"{item.result.type}:{error_type}")
            continue
        rt.cp.record(cid, done_entry(item.result.message, req, "batch"))
    entry["baixado"] = True
    rt.cp.save()


def run_batch(rt: Runtime, todo: list[PlannedRequest], passada: int) -> None:
    waiting = rt.cp.in_flight()
    new = [r for r in todo if r.custom_id not in waiting]
    if new:
        submit_batch(rt, new, passada)
    by_id = {r.custom_id: r for r in todo}
    for entry in rt.cp.batches:
        if entry.get("baixado") or entry.get("passada") != passada:
            continue
        if wait_for_batch(rt, entry):
            download_batch(rt, entry, by_id)
        else:
            rt.log(f"lote {entry['id']} ainda em processamento; rode novamente para retomar")


def run_requests(rt: Runtime, planned: list[PlannedRequest], passada: int) -> list[PlannedRequest]:
    """Executa o que falta; devolve as requisições ainda sem resposta (lote em andamento ou falha)."""
    todo = pending(planned, rt.cp, rt.settings.refazer_falhas)
    if todo:
        if rt.settings.modo == "sync":
            run_sync(rt, todo)
        else:
            run_batch(rt, todo, passada)
    return pending(planned, rt.cp, False)


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------
def group_by_key(contexts: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for ctx in contexts:
        groups.setdefault(ctx["key"], []).append(ctx)
    return groups


def collect_assessments(
    planned: list[PlannedRequest], cp: Checkpoint, evidence: dict[str, str]
) -> tuple[dict[str, dict], set[str]]:
    """Avaliações por id de contexto nas requisições concluídas; e ids ainda sem resposta."""
    assessments: dict[str, dict] = {}
    unanswered: set[str] = set()
    for req in planned:
        entry = cp.done.get(req.custom_id)
        if entry is None:
            unanswered.update(req.ids)
            continue
        by_id = {str(item.get("id")): item for item in entry["itens"]}
        for ctx_id in req.ids:
            assessments[ctx_id] = make_assessment(by_id.get(ctx_id), evidence[req.key])
            if entry.get("erro"):
                assessments[ctx_id]["flags"].append(entry["erro"].split(":")[0])
    return assessments, unanswered


def usage_totals(cp: Checkpoint) -> dict[str, dict[str, int]]:
    totals = {modo: dict.fromkeys(USAGE_FIELDS, 0) for modo in ("batch", "sync")}
    for entry in cp.done.values():
        for name in USAGE_FIELDS:
            totals[entry["modo"]][name] += int(entry["uso"].get(name, 0))
    return totals


def assemble_final(
    ctx: dict,
    classe: str,
    nivel: str,
    first: dict | None,
    second: dict | None,
    evidence_text: str = "",
    fingerprint: str | None = None,
    trechos: list[dict] | None = None,
) -> dict:
    row = {
        "id": ctx["id"], "key": ctx["key"], "file": ctx.get("file", ""), "line": ctx.get("line", 0),
        "chapter": ctx.get("chapter", ""), "section_path": ctx.get("section_path", []),
        "classe": classe, "nivel_evidencia": nivel,
        "evidencia_fingerprint": fingerprint,
        "trechos_enviados": trechos or [],
        "numeros_nao_encontrados": numeros_nao_encontrados(
            ctx.get("sentence_text", ""), prepare_evidence(evidence_text)
        ),
    }
    if first is None:
        flags = ["sem_evidencia"] if nivel == "nenhum" else ["pendente"]
        return {**row, "veredito_final": "NAO_VERIFICAVEL" if nivel == "nenhum" else "PENDENTE",
                "trecho_fonte": "", "trecho_verificado": None, "justificativa": "",
                "problema": None, "flags": flags, "passada1": None, "passada2": None}
    verdict, flags = combine_passes(first, second)
    return {**row, "veredito_final": verdict, "trecho_fonte": first["trecho_fonte"],
            "trecho_verificado": first["trecho_verificado"], "justificativa": first["justificativa"],
            "problema": first["problema"], "flags": flags, "passada1": first, "passada2": second}


def evidence_fingerprint(evidence_text: str) -> str:
    """Impressão digital do texto de evidência efetivamente enviado."""
    return hashlib.sha256(evidence_text.encode()).hexdigest()[:16]


def is_final_row(row: dict) -> bool:
    flags = set(row.get("flags") or [])
    return row.get("veredito_final") not in (None, "PENDENTE") and not (
        {"pendente", "segunda_passada_pendente"} & flags
    )


def _legacy_evidence_unchanged(state: dict, key: str, ctx_id: str, evidence_text: str) -> bool:
    """Linhas sem fingerprint: confere se o custom_id da passada 1 do checkpoint antigo se
    reproduz com a evidência atual (o custom_id embute o hash da evidência e o modelo)."""
    for cid, entry in (state.get("requisicoes") or {}).items():
        if entry.get("passada") == 1 and entry.get("key") == key and ctx_id in entry.get("ids", []):
            return cid == make_custom_id(1, key, entry["ids"], state.get("modelo", ""), evidence_text)
    return False


def _same_sentence(a: str, b: str) -> bool:
    """Igualdade de sentenças ignorando só diferenças de espaço em branco."""
    return " ".join(a.split()) == " ".join(b.split())


def find_reusable(
    contexts: list[dict],
    evidence_text: dict[str, str],
    reuse_root: Path,
    current_dir: Path,
) -> dict[str, tuple[str, dict]]:
    """Vereditos finais de execuções anteriores (``reuse_root/*/avaliacoes.jsonl``) reaproveitáveis.

    Vale quando o id do contexto tem veredito final e a evidência enviada é a mesma (fingerprint).
    Também casa linhas cujo ``id`` é o ``id_legado`` do contexto (fórmula antiga: chave + sentença
    normalizada). Se um id antigo corresponde a vários contextos novos (a mesma chave citada duas
    vezes na sentença), só vale para o contexto cuja ``sentence_text`` é idêntica à da linha antiga;
    sem esse texto na linha antiga, os contextos são reavaliados. Devolve
    ``{id_atual: (execução_de_origem, linha_anterior)}``.

    A própria pasta de trabalho é lida primeiro (é a avaliação mais recente, com a adjudicação). Nela,
    linhas casadas por ``id_legado`` (id da fórmula antiga) entram como as das outras pastas; linhas com o
    mesmo id atual só entram se já eram reaproveitadas (preserva a origem), porque as demais vêm do
    checkpoint desta execução.
    """
    wanted = {c["id"]: c for c in contexts}
    by_legacy: dict[str, list[dict]] = {}
    for c in contexts:
        if c.get("id_legado"):
            by_legacy.setdefault(c["id_legado"], []).append(c)
    found: dict[str, tuple[str, dict]] = {}
    if not wanted or not reuse_root.is_dir():
        return found
    current = current_dir.resolve()
    run_dirs = sorted(
        (p for p in reuse_root.iterdir() if p.is_dir()), key=lambda p: (p.resolve() != current, p.name)
    )
    for run_dir in run_dirs:
        own = run_dir.resolve() == current
        state: dict | None = None
        for old in read_jsonl(run_dir / "avaliacoes.jsonl"):
            old_id = old.get("id")
            if old_id in wanted:
                targets = [wanted[old_id]]
            else:
                targets = by_legacy.get(old_id, [])
                if len(targets) > 1:
                    old_sentence = old.get("sentence_text")
                    targets = [
                        c for c in targets
                        if old_sentence and _same_sentence(old_sentence, c.get("sentence_text", ""))
                    ]
            same_id_own = own and old_id in wanted and not old.get("reaproveitado_de")
            if not targets or not is_final_row(old) or same_id_own:
                continue
            for ctx in targets:
                if ctx["id"] in found or old.get("key") != ctx["key"]:
                    continue
                text = evidence_text[ctx["key"]]
                if old.get("evidencia_fingerprint"):
                    unchanged = old["evidencia_fingerprint"] == evidence_fingerprint(text)
                else:
                    if state is None:
                        state = read_json(run_dir / "checkpoint.json", default={}) or {}
                    unchanged = _legacy_evidence_unchanged(state, ctx["key"], old_id, text)
                if unchanged:
                    found[ctx["id"]] = (old.get("reaproveitado_de") or run_dir.name, old)
    return found


def reused_row(
    ctx: dict, old: dict, origin: str, evidence_text: str, fingerprint: str, trechos: list[dict] | None = None
) -> dict:
    """Cópia do veredito anterior com id, localização atuais e a marca ``reaproveitado_de``.

    A linha leva o ``id`` novo do contexto; ``id_legado`` é o do contexto (ou, na falta, o da linha anterior).
    """
    row = dict(old)
    legacy_id = ctx.get("id_legado") or old.get("id_legado")
    if legacy_id:
        row["id_legado"] = legacy_id
    row.update(
        id=ctx["id"],
        file=ctx.get("file", ""), line=ctx.get("line", 0), chapter=ctx.get("chapter", ""),
        section_path=ctx.get("section_path", []), evidencia_fingerprint=fingerprint,
        trechos_enviados=trechos or [], reaproveitado_de=origin,
        numeros_nao_encontrados=numeros_nao_encontrados(
            ctx.get("sentence_text", ""), prepare_evidence(evidence_text)
        ),
    )
    flags = [f for f in (old.get("flags") or []) if not f.startswith("reaproveitado_de=")]
    row["flags"] = [*flags, f"reaproveitado_de={origin}"]
    return row


def run_avaliacao(
    *,
    capitulo: str,
    contexts: list[dict],
    refs: dict[str, BibRef],
    proprios: set[str],
    out_dir: Path,
    pilot_dir: Path,
    settings: Settings,
    client: object,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda text: None,
    escopo: str = "capitulo",
    reuse_root: Path | None = None,
) -> dict:
    """Passada 1, verificação mecânica, passada 2 (cega) e veredito final.

    ``escopo="tese"`` planeja uma requisição por chave atravessando os capítulos; com ``reuse_root``
    copia vereditos finais de execuções anteriores cuja evidência não mudou.
    """
    global_scope = escopo == ESCOPO_TESE
    cp = Checkpoint(
        pilot_dir / "checkpoint.json",
        {"capitulo": capitulo, "modelo": settings.model, "criado_em": utc_now_iso()},
        sleep,
    )
    rt = Runtime(client, settings, cp, pilot_dir / "lotes", sleep, log)

    classes = {key: classify_key(refs.get(key), proprios) for key in {c["key"] for c in contexts}}
    graded = [c for c in contexts if classes[c["key"]] != "proprio"]
    evidence_text: dict[str, str] = {}
    levels: dict[str, str] = {}
    sent_passages: dict[str, list[dict]] = {}
    queries_en = load_english_queries(out_dir)
    by_key = group_by_key(graded)
    for key in sorted(by_key):
        record = read_json(cache_path(out_dir, key)) or {"key": key}
        evidence_text[key], levels[key], info = build_evidence_block_info(
            record, settings.max_chars, by_key[key], out_dir, queries_en
        )
        sent_passages[key] = info["trechos"]
        if info["selecao"]:
            log(f"{key}: seleção de {len(info['trechos'])} trechos "
                f"({info['consultas_en']}/{info['consultas']} contextos com consulta em inglês)")

    fingerprints = {key: evidence_fingerprint(text) for key, text in evidence_text.items()}
    candidates = [c for c in graded if levels[c["key"]] != "nenhum"]
    reused: dict[str, tuple[str, dict]] = {}
    if reuse_root is not None:
        reused = find_reusable(candidates, evidence_text, reuse_root, pilot_dir)
        if reused:
            log(f"vereditos reaproveitados: {len(reused)}")

    with_evidence = group_by_key([c for c in candidates if c["id"] not in reused])
    plan1 = plan_requests(with_evidence, evidence_text, capitulo, settings, 1, global_scope=global_scope)
    left1 = run_requests(rt, plan1, 1)
    assess1, unanswered1 = collect_assessments(plan1, cp, evidence_text)

    assess2: dict[str, dict] = {}
    plan2: list[PlannedRequest] = []
    unanswered2: set[str] = set()
    if not left1:
        again = [c for c in graded if c["id"] in assess1 and needs_second_pass(assess1[c["id"]])]
        cached_before = {r.key for r in plan1 if "cache_control" in r.params["messages"][0]["content"][0]}
        plan2 = plan_requests(
            group_by_key(again), evidence_text, capitulo, settings, 2, cached_before, global_scope
        )
        run_requests(rt, plan2, 2)
        assess2, unanswered2 = collect_assessments(plan2, cp, evidence_text)

    rows = []
    for ctx in graded:
        cid, key = ctx["id"], ctx["key"]
        if cid in reused:
            origin, old = reused[cid]
            rows.append(
                reused_row(ctx, old, origin, evidence_text[key], fingerprints[key], sent_passages[key])
            )
            continue
        rows.append(
            assemble_final(
                ctx, classes[key], levels[key], assess1.get(cid), assess2.get(cid),
                evidence_text.get(key, ""), fingerprints.get(key), sent_passages.get(key),
            )
        )
    write_jsonl_atomic(pilot_dir / "avaliacoes.jsonl", rows)

    incomplete = bool(left1 or unanswered1 or unanswered2 or any(r["veredito_final"] == "PENDENTE" for r in rows))
    origins: dict[str, int] = {}
    for origin, _old in reused.values():
        origins[origin] = origins.get(origin, 0) + 1
    meta = build_metadata(
        capitulo, settings, cp, plan1, plan2, len(graded), incomplete,
        escopo=escopo, reaproveitados=origins,
    )
    write_json_atomic(pilot_dir / "metadados.json", meta)
    return {"rows": rows, "metadados": meta, "incompleto": incomplete}


def build_metadata(
    capitulo: str,
    settings: Settings,
    cp: Checkpoint,
    plan1: list[PlannedRequest],
    plan2: list[PlannedRequest],
    n_contexts: int,
    incomplete: bool,
    escopo: str = "capitulo",
    reaproveitados: dict[str, int] | None = None,
) -> dict:
    return {
        "capitulo": capitulo,
        "escopo": escopo,
        "modelo": settings.model,
        "data_utc": utc_now_iso(),
        "parametros": {
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature if settings.send_temperature else None,
            "temperature_enviada": settings.send_temperature,
            "modo": settings.modo,
            "max_chars": settings.max_chars,
            "max_contextos": settings.max_contextos,
            "concorrencia_sync": settings.concorrencia,
        },
        "lotes": [b["id"] for b in cp.batches if b.get("id")],
        "uso": usage_totals(cp),
        "requisicoes": {"passada1": len(plan1), "passada2": len(plan2)},
        "contextos_avaliados": n_contexts,
        "contextos_reaproveitados": sum((reaproveitados or {}).values()),
        "reaproveitados_de": reaproveitados or {},
        "falhas": len(cp.state["falhas"]),
        "status": "incompleto" if incomplete else "completo",
    }

