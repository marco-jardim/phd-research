"""Consultas em inglês para a seleção de trechos (Claude Haiku, Message Batches, uma vez por contexto).

A tese está em português e as fontes longas quase sempre em inglês; a consulta em português acerta
pouco além de números, siglas e nomes próprios. Para cada contexto cuja fonte tem texto completo
maior que ``--max-chars``, uma requisição curta pede ao Haiku a tradução da sentença para o inglês e de
5 a 12 palavras-chave em inglês. O resultado fica em ``<out>/evidencias/_consultas_en.json`` por id de
contexto e nunca é pedido de novo (contextos presentes no arquivo são pulados).

Layout da requisição (sem ``temperature``): ``system[0]`` é o bloco de instruções fixo, com
``cache_control`` efêmero (o cache só vale se o prefixo atingir o mínimo do modelo); a mensagem traz o id,
a sentença e o parágrafo do contexto. O estado fica em ``consultas_checkpoint.json`` na pasta do piloto
(mesmo formato e mesmas regras de retomada de ``avaliar``: lote registrado não é reenviado).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path

from . import avaliar as av
from .common import read_json, utc_now_iso
from .evidencias import cache_path
from .prompts import trim_paragraph
from .trechos import english_query_for, load_english_queries, save_english_queries

N_KEYWORDS_MIN = 5
N_KEYWORDS_MAX = 12
PARAGRAPH_LIMIT = 700

CONSULTAS_SYSTEM = """Você prepara consultas de busca em inglês para localizar, em um texto longo escrito em inglês, a passagem que sustenta uma citação de uma tese de doutorado em português (Saúde Coletiva, relacionamento de registros e aprendizado de máquina).

Cada requisição traz o id de um contexto da tese, a sentença (em que a citação avaliada aparece como [cit] e as demais como [ref]) e o parágrafo em volta, só como contexto.

Faça duas coisas:

1. Traduza para o inglês técnico a afirmação que a sentença atribui à fonte [cit]. Não acrescente nada que a sentença não diga, não resuma e mantenha como estão os números, as siglas, os nomes próprios e os termos técnicos já escritos em inglês.
2. Liste de 5 a 12 palavras-chave em inglês (termos ou expressões curtas) que o texto original provavelmente usa ao tratar desse conteúdo, incluindo sinônimos técnicos usuais.

Responda somente com um array JSON válido de um objeto (sem texto antes ou depois, sem cercas de código):

[{"id": "<id do contexto, copiado exatamente>", "traducao": "<sentença traduzida>", "palavras_chave": ["<termo 1>", "<termo 2>"]}]
"""


def build_query_params(settings: av.Settings, ctx: dict) -> dict:
    """Requisição de um contexto; nunca inclui ``temperature``."""
    sentence = ctx.get("sentence_text", "")
    paragraph = trim_paragraph(ctx.get("paragraph_text", ""), sentence, PARAGRAPH_LIMIT)
    user = f"id: {ctx['id']}\nSentença: {sentence}\nParágrafo: {paragraph}\n"
    return {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "system": [{"type": "text", "text": CONSULTAS_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
    }


def make_query_custom_id(ctx_id: str, model: str) -> str:
    """Identificador estável (``^[A-Za-z0-9_-]{1,64}$``): muda se o contexto, o modelo ou a instrução mudar."""
    digest = hashlib.sha256(
        "|".join([ctx_id, model, hashlib.sha256(CONSULTAS_SYSTEM.encode()).hexdigest()]).encode()
    ).hexdigest()
    return f"q-{digest[:24]}"


def plan_consultas(contexts: list[dict], settings: av.Settings) -> list[av.PlannedRequest]:
    ordered = sorted(contexts, key=lambda c: (c.get("file", ""), c.get("line", 0), c["id"]))
    return [
        av.PlannedRequest(
            make_query_custom_id(c["id"], settings.model), c["key"], 1, [c["id"]],
            build_query_params(settings, c),
        )
        for c in ordered
    ]


def long_text_keys(contexts: list[dict], out_dir: Path, max_chars: int) -> set[str]:
    """Chaves cujo texto completo em cache passa de ``max_chars`` (as únicas que usam seleção)."""
    keys = set()
    for key in {c["key"] for c in contexts}:
        record = read_json(cache_path(out_dir, key)) or {}
        if len(record.get("fulltext") or "") > max_chars:
            keys.add(key)
    return keys


def clean_query(item: dict) -> dict | None:
    """Consulta validada (``traducao`` e ``palavras_chave`` de texto) ou None se a resposta é inútil."""
    translation = item.get("traducao")
    keywords = item.get("palavras_chave")
    if not isinstance(translation, str) or not translation.strip() or not isinstance(keywords, list):
        return None
    words = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
    if not words:
        return None
    return {"traducao": translation.strip(), "palavras_chave": words[: N_KEYWORDS_MAX + 3]}


def collect_queries(planned: list[av.PlannedRequest], cp: av.Checkpoint, model: str) -> tuple[dict[str, dict], int]:
    """Consultas válidas por id de contexto nas respostas concluídas, e quantas vieram inválidas."""
    found: dict[str, dict] = {}
    invalid = 0
    for req in planned:
        entry = cp.done.get(req.custom_id)
        if entry is None:
            continue
        ctx_id = req.ids[0]
        items = entry.get("itens") or []
        item = next((i for i in items if str(i.get("id")) == ctx_id), items[0] if len(items) == 1 else None)
        query = None if entry.get("erro") or item is None else clean_query(item)
        if query is None:
            invalid += 1
            continue
        found[ctx_id] = {**query, "modelo": model, "em": utc_now_iso()}
    return found, invalid


def run_consultas(
    *,
    contexts: list[dict],
    out_dir: Path,
    pilot_dir: Path,
    settings: av.Settings,
    client: object,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda text: None,
) -> dict:
    """Pede (lote ou sync) as consultas em inglês que faltam e grava ``_consultas_en.json``.

    Só entram contextos de chaves com texto completo maior que ``settings.max_chars`` e ainda sem consulta
    no arquivo, pelo id atual ou pelo ``id_legado`` (como na seleção, ``english_query_for``).
    Retorna ``{"pedidas", "novas", "invalidas", "em_cache", "incompleto"}``."""
    cached = load_english_queries(out_dir)
    long_keys = long_text_keys(contexts, out_dir, settings.max_chars)
    targets = [c for c in contexts if c["key"] in long_keys and english_query_for(cached, c) is None]
    summary = {"pedidas": len(targets), "novas": 0, "invalidas": 0, "em_cache": len(cached), "incompleto": False}
    if not targets:
        log("nenhuma consulta nova a pedir")
        return summary
    cp = av.Checkpoint(
        pilot_dir / "consultas_checkpoint.json",
        {"modelo": settings.model, "criado_em": utc_now_iso()},
        sleep,
    )
    rt = av.Runtime(client, settings, cp, pilot_dir / "lotes_consultas", sleep, log)
    planned = plan_consultas(targets, settings)
    left = av.run_requests(rt, planned, 1)
    found, invalid = collect_queries(planned, cp, settings.model)
    if found:
        save_english_queries(out_dir, {**cached, **found})
    summary.update(novas=len(found), invalidas=invalid, em_cache=len(cached) + len(found),
                   incompleto=bool(left))
    return summary
