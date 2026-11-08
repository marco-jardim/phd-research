"""Execução das requisições do cruzamento: modo sync (pool de threads) e Message Batches.

Reaproveita a infraestrutura de ``scripts/atestado/avaliar.py`` (``Settings``, ``Runtime``, ``Checkpoint``
com retry no ``os.replace``, ``with_retry`` com backoff exponencial em 408/409/429/5xx e erros de rede,
envio e espera de lotes). Só a leitura da resposta muda: aqui ela é um objeto JSON (não um array), e o
checkpoint guarda o objeto em ``resultado``.

* ``temperature`` nunca é enviada.
* O bloco de sistema leva ``cache_control`` efêmero (sync e lote).
* Sync: ``concorrencia`` threads chamam a API; cada resposta é gravada no checkpoint, sob trava, assim
  que chega. Uma nova execução pula tudo o que já tem resposta.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from scripts.atestado import avaliar as av
from scripts.atestado.common import utc_now_iso

from .comum import extrair_objeto

CONCORRENCIA_PADRAO = 8
CARACTERES_POR_TOKEN = 3.5  # aproximação para português com LaTeX


def criar_settings(
    modelo: str, modo: str, max_tokens: int, concorrencia: int = CONCORRENCIA_PADRAO,
    refazer_falhas: bool = False,
) -> av.Settings:
    return av.Settings(
        model=modelo, max_tokens=max_tokens, send_temperature=False, modo=modo,
        concorrencia=max(1, concorrencia), refazer_falhas=refazer_falhas,
    )


def montar_params(settings: av.Settings, sistema: str, blocos: list[dict]) -> dict:
    """Parâmetros da requisição: sistema em cache e blocos de usuário; nunca inclui ``temperature``."""
    return {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "system": [{"type": "text", "text": sistema, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": blocos}],
    }


def bloco_texto(texto: str, cache: bool = False) -> dict:
    bloco: dict = {"type": "text", "text": texto}
    if cache:
        bloco["cache_control"] = {"type": "ephemeral"}
    return bloco


def custom_id(prefixo: str, *partes: str) -> str:
    """``^[A-Za-z0-9_-]{1,64}$``; muda se o conteúdo, o modelo ou o sistema mudarem."""
    from .comum import id_curto

    return f"{prefixo}-{id_curto(*partes, n=24)}"


# ---------------------------------------------------------------------------
# Resposta
# ---------------------------------------------------------------------------
def nova_entrada(message, req: av.PlannedRequest, modo: str) -> dict:
    """Registro de checkpoint de uma resposta; JSON inválido ou truncado fica registrado em ``erro``."""
    entrada = {
        "key": req.key,
        "passada": req.passada,
        "ids": req.ids,
        "modo": modo,
        "uso": av.usage_of(message),
        "stop_reason": getattr(message, "stop_reason", None),
        "resultado": None,
        "erro": None,
    }
    if entrada["stop_reason"] == "max_tokens":
        entrada["erro"] = "resposta_truncada"
        return entrada
    texto = av.message_text(message)
    try:
        entrada["resultado"] = extrair_objeto(texto)
    except ValueError as exc:
        entrada["erro"] = f"json_invalido: {exc}"
        entrada["amostra"] = texto[:300]
    return entrada


# ---------------------------------------------------------------------------
# Sync e lote
# ---------------------------------------------------------------------------
def executar_sync(rt: av.Runtime, todo: list[av.PlannedRequest]) -> None:
    trava = Lock()

    def trabalhar(req: av.PlannedRequest) -> None:
        try:
            message = av.with_retry(lambda: rt.client.messages.create(**req.params), rt.sleep)
        except av.ApiCallError as exc:
            with trava:
                rt.cp.fail(req.custom_id, str(exc))
            rt.log(f"falha em {req.custom_id}: {exc}")
            return
        entrada = nova_entrada(message, req, "sync")
        with trava:
            rt.cp.record(req.custom_id, entrada)

    with ThreadPoolExecutor(max_workers=max(1, rt.settings.concorrencia)) as pool:
        list(pool.map(trabalhar, todo))


def baixar_lote(rt: av.Runtime, entrada: dict, por_id: dict[str, av.PlannedRequest]) -> None:
    for item in rt.client.messages.batches.results(entrada["id"]):
        cid = item.custom_id
        req = por_id.get(cid)
        if req is None:
            rt.cp.fail(cid, "custom_id desconhecido no lote")
            continue
        if item.result.type != "succeeded":
            tipo_erro = getattr(getattr(item.result, "error", None), "type", None)
            rt.cp.fail(cid, f"{item.result.type}:{tipo_erro}")
            continue
        rt.cp.record(cid, nova_entrada(item.result.message, req, "batch"))
    entrada["baixado"] = True
    rt.cp.save()


def executar_lote(rt: av.Runtime, todo: list[av.PlannedRequest], passada: int) -> None:
    esperando = rt.cp.in_flight()
    novos = [r for r in todo if r.custom_id not in esperando]
    if novos:
        av.submit_batch(rt, novos, passada)
    por_id = {r.custom_id: r for r in todo}
    for entrada in rt.cp.batches:
        if entrada.get("baixado") or entrada.get("passada") != passada:
            continue
        if av.wait_for_batch(rt, entrada):
            baixar_lote(rt, entrada, por_id)
        else:
            rt.log(f"lote {entrada['id']} ainda em processamento; rode novamente para retomar")


def executar(rt: av.Runtime, planejadas: list[av.PlannedRequest], passada: int) -> list[av.PlannedRequest]:
    """Executa o que falta; devolve as requisições ainda sem resposta (lote em andamento ou falha)."""
    todo = av.pending(planejadas, rt.cp, rt.settings.refazer_falhas)
    if todo:
        if rt.settings.modo == "sync":
            executar_sync(rt, todo)
        else:
            executar_lote(rt, todo, passada)
    return av.pending(planejadas, rt.cp, False)


def abrir_runtime(
    out: Path, etapa: str, settings: av.Settings, client: object,
    sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = lambda texto: None,
) -> av.Runtime:
    cp = av.Checkpoint(
        out / f"{etapa}.checkpoint.json",
        {"etapa": etapa, "modelo": settings.model, "criado_em": utc_now_iso()},
        sleep,
    )
    return av.Runtime(client, settings, cp, out / "lotes" / etapa, sleep, log)


# ---------------------------------------------------------------------------
# Estimativa (sem API)
# ---------------------------------------------------------------------------
def texto_dos_params(params: dict) -> str:
    partes = [bloco["text"] for bloco in params["system"]]
    for mensagem in params["messages"]:
        partes.extend(bloco["text"] for bloco in mensagem["content"])
    return "\n".join(partes)


def tokens_aprox(texto: str) -> int:
    return int(len(texto) / CARACTERES_POR_TOKEN) + 1


def estimar(
    planejadas: list[av.PlannedRequest], saida_por_requisicao: Callable[[av.PlannedRequest, int], int]
) -> dict[str, int]:
    """Requisições e tokens aproximados de entrada e saída (a entrada conta o prefixo em cache cheio)."""
    entrada = 0
    saida = 0
    for req in planejadas:
        n = tokens_aprox(texto_dos_params(req.params))
        entrada += n
        saida += min(req.params["max_tokens"], saida_por_requisicao(req, n))
    return {"requisicoes": len(planejadas), "entrada": entrada, "saida": saida}


def uso_do_checkpoint(cp: av.Checkpoint) -> dict[str, dict[str, int]]:
    return av.usage_totals(cp)
