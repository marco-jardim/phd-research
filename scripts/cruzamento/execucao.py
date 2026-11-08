"""Execução das requisições do cruzamento: modo sync (pool de threads) e Message Batches.

Reaproveita a infraestrutura de ``scripts/atestado/avaliar.py`` (``Settings``, ``Runtime``, ``Checkpoint``
com retry no ``os.replace``, ``with_retry`` com backoff exponencial em 408/409/429/5xx e erros de rede,
envio e espera de lotes). Só a leitura da resposta muda: aqui ela é um objeto JSON (não um array), e o
checkpoint guarda o objeto em ``resultado``.

* ``temperature`` nunca é enviada.
* O bloco de sistema leva ``cache_control`` efêmero (sync e lote).
* Sync: ``concorrencia`` threads por chave chamam a API; cada resposta é gravada no checkpoint, sob
  trava, assim que chega. Uma nova execução pula tudo o que já tem resposta.
* Várias chaves: o cliente pode ser um único objeto (vale como ``ANTHROPIC_API_KEY``) ou um dicionário
  ``{nome da variável: cliente}``. Cada requisição vai à chave ``nomes[i % n]``, com ``i`` o índice estável
  da requisição no plano, de modo que a retomada repete a distribuição. Retry de 429, 529 e erro de rede
  fica na mesma chave. 401/403, ou 400/402 com mensagem de crédito ou saldo, tira a chave desta execução e
  passa as pendentes dela às restantes. No lote, cada lote vai a uma chave e o checkpoint guarda o NOME
  da variável, usado depois na consulta e no download. Só nomes de variáveis são gravados, nunca valores.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Condition, Lock

from scripts.atestado import avaliar as av
from scripts.atestado.common import utc_now_iso

from .comum import extrair_objeto

CONCORRENCIA_PADRAO = 8
CARACTERES_POR_TOKEN = 3.5  # aproximação para português com LaTeX
CHAVE_ENV_PADRAO = "ANTHROPIC_API_KEY"
_MARCAS_SEM_CREDITO = ("credit", "balance", "billing", "saldo", "crédito", "credito")


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
def nova_entrada(message, req: av.PlannedRequest, modo: str, chave: str | None = None) -> dict:
    """Registro de checkpoint de uma resposta; JSON inválido ou truncado fica registrado em ``erro``.

    ``chave`` é o NOME da variável de ambiente da chave usada, nunca o valor.
    """
    entrada = {
        "key": req.key,
        "passada": req.passada,
        "ids": req.ids,
        "modo": modo,
        "chave": chave,
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
def normalizar_clientes(client: object) -> dict[str, object]:
    """``{nome da variável: cliente}`` na ordem dada; um cliente avulso vale como ``ANTHROPIC_API_KEY``."""
    if isinstance(client, dict):
        if not client:
            raise ValueError("nenhum cliente da API")
        return dict(client)
    return {CHAVE_ENV_PADRAO: client}


def status_da_falha(exc: BaseException) -> int | None:
    causa = exc.__cause__ if isinstance(exc, av.ApiCallError) else exc
    return getattr(causa, "status_code", None)


def chave_indisponivel(exc: BaseException) -> bool:
    """401/403, ou 400/402 com mensagem de crédito ou saldo: a chave não serve mais nesta execução."""
    causa = exc.__cause__ if isinstance(exc, av.ApiCallError) else exc
    status = getattr(causa, "status_code", None)
    if status in (401, 403):
        return True
    if status in (400, 402):
        texto = f"{causa} {getattr(causa, 'body', '')}".casefold()
        return any(marca in texto for marca in _MARCAS_SEM_CREDITO)
    return False


class Rodizio:
    """Chave de cada requisição pelo índice estável no plano, desviando das chaves indisponíveis."""

    def __init__(self, nomes) -> None:
        self.nomes = list(nomes)
        self.fora: set[str] = set()

    def disponiveis(self) -> list[str]:
        return [nome for nome in self.nomes if nome not in self.fora]

    def chave(self, indice: int) -> str | None:
        preferida = self.nomes[indice % len(self.nomes)]
        if preferida not in self.fora:
            return preferida
        restantes = self.disponiveis()
        return restantes[indice % len(restantes)] if restantes else None


def _indices(todo: list[av.PlannedRequest], indices: dict[str, int] | None) -> dict[str, int]:
    return indices if indices is not None else {r.custom_id: i for i, r in enumerate(todo)}


def executar_sync(
    rt: av.Runtime, todo: list[av.PlannedRequest], indices: dict[str, int] | None = None
) -> None:
    """Uma fila por chave, ``concorrencia`` threads por chave; failover quando a chave cai."""
    clientes = normalizar_clientes(rt.client)
    indices = _indices(todo, indices)
    rodizio = Rodizio(clientes)
    filas: dict[str, deque[av.PlannedRequest]] = {nome: deque() for nome in clientes}
    for req in todo:
        filas[rodizio.chave(indices[req.custom_id])].append(req)
    cond = Condition()  # filas, chaves fora e contagem em voo
    trava = Lock()  # checkpoint
    em_voo = 0

    def pendentes() -> int:
        return em_voo + sum(len(fila) for fila in filas.values())

    def desativar(nome: str, req: av.PlannedRequest, exc: av.ApiCallError) -> None:
        status = status_da_falha(exc)
        with cond:
            nova = nome not in rodizio.fora
            rodizio.fora.add(nome)
            orfas = sorted([req, *filas[nome]], key=lambda r: indices[r.custom_id])
            filas[nome].clear()
            for orfa in orfas:
                destino = rodizio.chave(indices[orfa.custom_id])
                if destino is not None:
                    filas[destino].append(orfa)
            restantes = rodizio.disponiveis()
            cond.notify_all()
        with trava:
            rt.cp.fail(req.custom_id, f"chave_indisponivel:{nome} status={status}")
        if nova:
            rt.log(f"chave {nome} indisponível nesta execução (status={status}); "
                   f"pendentes dela vão para: {', '.join(restantes) or 'nenhuma'}")
            if not restantes:
                rt.log("todas as chaves indisponíveis; o que falta fica pendente no checkpoint")

    def processar(nome: str, req: av.PlannedRequest) -> None:
        cliente = clientes[nome]
        try:
            message = av.with_retry(lambda: cliente.messages.create(**req.params), rt.sleep)
        except av.ApiCallError as exc:
            if chave_indisponivel(exc):
                desativar(nome, req, exc)
                return
            with trava:
                rt.cp.fail(req.custom_id, str(exc))
            rt.log(f"falha em {req.custom_id}: {exc}")
            return
        entrada = nova_entrada(message, req, "sync", nome)
        with trava:
            rt.cp.record(req.custom_id, entrada)

    def trabalhar(nome: str) -> None:
        nonlocal em_voo
        while True:
            with cond:
                while nome not in rodizio.fora and not filas[nome] and pendentes() > 0:
                    cond.wait()
                if nome in rodizio.fora or not filas[nome]:
                    return
                req = filas[nome].popleft()
                em_voo += 1
            try:
                processar(nome, req)
            finally:
                with cond:
                    em_voo -= 1
                    cond.notify_all()

    por_chave = max(1, rt.settings.concorrencia)
    with ThreadPoolExecutor(max_workers=por_chave * len(clientes)) as pool:
        futuros = [pool.submit(trabalhar, nome) for nome in clientes for _ in range(por_chave)]
        for futuro in futuros:
            futuro.result()


def baixar_lote(
    rt: av.Runtime, entrada: dict, por_id: dict[str, av.PlannedRequest], chave: str | None = None
) -> None:
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
        rt.cp.record(cid, nova_entrada(item.result.message, req, "batch", chave))
    entrada["baixado"] = True
    rt.cp.save()


def enviar_lote(rt: av.Runtime, nome: str, cliente: object, todo: list[av.PlannedRequest], passada: int) -> None:
    """Como ``av.submit_batch``, mas o lote nasce no checkpoint com o NOME da variável da chave."""
    if any(b.get("id") is None for b in rt.cp.batches):
        raise RuntimeError(
            "envio de lote anterior sem id registrado: confira os lotes na conta Anthropic "
            "e reconcilie o checkpoint manualmente; não reenviar"
        )
    arquivo = av.write_batch_file(rt, todo, passada)
    entrada = {
        "id": None, "passada": passada, "chave": nome, "status": "enviando", "baixado": False,
        "custom_ids": [r.custom_id for r in todo], "arquivo": arquivo.name, "enviado_em": utc_now_iso(),
    }
    rt.cp.batches.append(entrada)
    rt.cp.save()
    try:
        lote = av.with_retry(
            lambda: cliente.messages.batches.create(
                requests=[{"custom_id": r.custom_id, "params": r.params} for r in todo]
            ),
            rt.sleep,
            attempts=1,
        )
    except av.ApiCallError as exc:
        status = status_da_falha(exc)
        if status is not None and 400 <= status < 500 and status not in (408, 409, 429):
            # Recusa explícita: o lote não foi criado, então o envio pode ser repetido.
            rt.cp.batches.remove(entrada)
            rt.cp.save()
        raise
    entrada["id"] = lote.id
    entrada["status"] = lote.processing_status
    rt.cp.save()
    rt.log(f"lote enviado: {lote.id} ({len(todo)} requisições; chave {nome})")


def executar_lote(
    rt: av.Runtime, todo: list[av.PlannedRequest], passada: int, indices: dict[str, int] | None = None
) -> None:
    clientes = normalizar_clientes(rt.client)
    indices = _indices(todo, indices)
    rodizio = Rodizio(clientes)
    esperando = rt.cp.in_flight()
    novos = [r for r in todo if r.custom_id not in esperando]
    while novos:
        grupos: dict[str, list[av.PlannedRequest]] = {}
        for req in novos:
            destino = rodizio.chave(indices[req.custom_id])
            if destino is not None:
                grupos.setdefault(destino, []).append(req)
        if not grupos:
            rt.log("todas as chaves indisponíveis; o que falta fica pendente no checkpoint")
            break
        novos = []
        for nome in rodizio.nomes:
            if nome not in grupos:
                continue
            try:
                enviar_lote(rt, nome, clientes[nome], grupos[nome], passada)
            except av.ApiCallError as exc:
                if not chave_indisponivel(exc):
                    raise
                rodizio.fora.add(nome)
                novos.extend(grupos[nome])
                rt.log(f"chave {nome} indisponível nesta execução (status={status_da_falha(exc)}); "
                       f"o lote dela vai para: {', '.join(rodizio.disponiveis()) or 'nenhuma'}")
        novos.sort(key=lambda r: indices[r.custom_id])
    por_id = {r.custom_id: r for r in todo}
    for entrada in rt.cp.batches:
        if entrada.get("baixado") or entrada.get("passada") != passada:
            continue
        nome = entrada.get("chave") or CHAVE_ENV_PADRAO  # lotes anteriores ao campo vinham da chave padrão
        if nome not in clientes or nome in rodizio.fora:
            rt.log(f"lote {entrada['id']} pertence à chave {nome}, indisponível nesta execução; "
                   "inclua-a em --chave-env e rode novamente")
            continue
        rt_chave = replace(rt, client=clientes[nome])
        try:
            if av.wait_for_batch(rt_chave, entrada):
                baixar_lote(rt_chave, entrada, por_id, nome)
            else:
                rt.log(f"lote {entrada['id']} ainda em processamento; rode novamente para retomar")
        except av.ApiCallError as exc:
            if not chave_indisponivel(exc):
                raise
            rodizio.fora.add(nome)
            rt.log(f"chave {nome} indisponível nesta execução (status={status_da_falha(exc)}); "
                   f"lote {entrada['id']} fica para a próxima execução")


def executar(rt: av.Runtime, planejadas: list[av.PlannedRequest], passada: int) -> list[av.PlannedRequest]:
    """Executa o que falta; devolve as requisições ainda sem resposta (lote em andamento ou falha)."""
    todo = av.pending(planejadas, rt.cp, rt.settings.refazer_falhas)
    if todo:
        indices = {r.custom_id: i for i, r in enumerate(planejadas)}
        if rt.settings.modo == "sync":
            executar_sync(rt, todo, indices)
        else:
            executar_lote(rt, todo, passada, indices)
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
