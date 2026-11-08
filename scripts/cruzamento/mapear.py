"""Etapa 2 (map): uma requisição por fatia; relações com as âncoras e afirmações substantivas.

A resposta é um objeto JSON ``{"relacoes": [...], "afirmacoes": [...]}``. Todo ``trecho`` é conferido
mecanicamente contra a fatia (espaços normalizados): o que não existir literalmente fica no resultado com
``trecho_valido = false`` e a marca ``trecho_invalido`` em ``flags`` (nunca é descartado em silêncio, e o
relatório os lista). Quando o trecho existe, a ``linha`` passa a ser a linha real em que ele começa.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path

from scripts.atestado import avaliar as av
from scripts.atestado.common import write_jsonl_atomic

from . import execucao as ex
from .comum import (
    MAPA_NOME,
    POLARIDADES,
    RELACOES,
    TIPOS_AFIRMACAO,
    TOPICO_OUTRO,
    TOPICOS,
    colapsar_espacos,
    id_curto,
    indexar_fatia,
    inteiro_ou_none,
    lista_de_textos,
    localizar_trecho,
)
from .prompts import sistema_mapear, usuario_mapear

PASSADA = 1


def planejar_mapear(
    fatias: list[dict], ancoras: list[dict], settings: av.Settings, ficha: str = ""
) -> list[av.PlannedRequest]:
    sistema = sistema_mapear(ancoras, ficha)
    digest = hashlib.sha256(sistema.encode("utf-8")).hexdigest()[:12]
    planejadas = []
    for fatia in fatias:
        params = ex.montar_params(settings, sistema, [ex.bloco_texto(usuario_mapear(fatia))])
        cid = ex.custom_id("map", fatia["id"], settings.model, digest)
        planejadas.append(av.PlannedRequest(cid, fatia["id"], PASSADA, [fatia["id"]], params))
    return planejadas


def estimar_mapear(
    fatias: list[dict], ancoras: list[dict], settings: av.Settings, ficha: str = ""
) -> dict[str, int]:
    """Saída aproximada: 150 tokens mais 30% dos tokens do texto da fatia."""
    sistema_tokens = ex.tokens_aprox(sistema_mapear(ancoras, ficha))
    return ex.estimar(
        planejar_mapear(fatias, ancoras, settings, ficha),
        lambda req, n: 150 + int(0.3 * max(0, n - sistema_tokens)),
    )


# ---------------------------------------------------------------------------
# Processamento da resposta
# ---------------------------------------------------------------------------
def _verificar_trecho(fatia: dict, indice, item: dict) -> tuple[str, int | None, bool, list[str]]:
    trecho = str(item.get("trecho") or "")
    linha_modelo = inteiro_ou_none(item.get("linha"))
    linha_real = localizar_trecho(indice, trecho)
    flags: list[str] = []
    if linha_real is None:
        flags.append("trecho_invalido")
        return trecho, linha_modelo, False, flags
    if linha_modelo != linha_real:
        flags.append("linha_corrigida")
    return trecho, linha_real, True, flags


def _relacao(item: dict, fatia: dict, indice, ids_ancoras: set[str]) -> dict:
    trecho, linha, valido, flags = _verificar_trecho(fatia, indice, item)
    ancora = str(item.get("ancora_id") or "").strip()
    relacao = str(item.get("relacao") or "").strip()
    if relacao not in RELACOES:
        flags.append("relacao_invalida")
    if ancora not in ids_ancoras:
        flags.append("ancora_desconhecida")
    return {
        "id": id_curto(fatia["id"], ancora, relacao, colapsar_espacos(trecho)),
        "fatia_id": fatia["id"],
        "arquivo": fatia["arquivo"],
        "ancora_id": ancora,
        "relacao": relacao,
        "trecho": trecho,
        "linha": linha,
        "justificativa": str(item.get("justificativa") or ""),
        "trecho_valido": valido,
        "flags": flags,
    }


def _afirmacao(item: dict, fatia: dict, indice) -> dict:
    trecho, linha, valido, flags = _verificar_trecho(fatia, indice, item)
    tipo = str(item.get("tipo") or "").strip()
    polaridade = str(item.get("polaridade") or "").strip()
    if tipo not in TIPOS_AFIRMACAO:
        flags.append("tipo_invalido")
    if polaridade not in POLARIDADES:
        flags.append("polaridade_invalida")
    pedidos = lista_de_textos(item.get("topicos"))
    topicos = [t for t in pedidos if t in TOPICOS]
    if len(topicos) != len(pedidos) or not topicos:
        flags.append("topico_fora_da_taxonomia")
    topicos = topicos or [TOPICO_OUTRO]
    numeros = lista_de_textos(item.get("numeros"))
    normalizado = colapsar_espacos(trecho)
    if any(colapsar_espacos(n) not in normalizado for n in numeros):
        flags.append("numero_fora_do_trecho")
    return {
        "id": id_curto(fatia["id"], normalizado),
        "fatia_id": fatia["id"],
        "arquivo": fatia["arquivo"],
        "trecho": trecho,
        "linha": linha,
        "tipo": tipo,
        "topicos": topicos,
        "numeros": numeros,
        "polaridade": polaridade,
        "escopo": str(item.get("escopo") or ""),
        "trecho_valido": valido,
        "flags": flags,
    }


def processar_resultado(fatia: dict, resultado: dict, ids_ancoras: set[str]) -> dict:
    """Relações e afirmações da fatia, com verificação de trecho; itens malformados são contados."""
    indice = indexar_fatia(fatia)
    relacoes: list[dict] = []
    afirmacoes: list[dict] = []
    malformados = 0
    for chave, lista, construir in (
        ("relacoes", relacoes, lambda it: _relacao(it, fatia, indice, ids_ancoras)),
        ("afirmacoes", afirmacoes, lambda it: _afirmacao(it, fatia, indice)),
    ):
        bruto = resultado.get(chave)
        if bruto is None:
            continue
        if not isinstance(bruto, list):
            malformados += 1
            continue
        for item in bruto:
            if isinstance(item, dict):
                lista.append(construir(item))
            else:
                malformados += 1
    return {"relacoes": relacoes, "afirmacoes": afirmacoes, "malformados": malformados}


def consolidar_mapa(
    fatias: list[dict], planejadas: list[av.PlannedRequest], cp: av.Checkpoint, ancoras: list[dict]
) -> list[dict]:
    """Uma linha por fatia, em ordem de fatia: erro (``pendente``, JSON inválido...) ou itens verificados."""
    ids_ancoras = {a["id"] for a in ancoras}
    por_fatia = {req.key: req for req in planejadas}
    linhas = []
    for fatia in fatias:
        base = {
            "fatia_id": fatia["id"], "arquivo": fatia["arquivo"], "linha_inicio": fatia["linha_inicio"],
            "secao": fatia["secao"], "fonte": fatia["fonte"],
        }
        entrada = cp.done.get(por_fatia[fatia["id"]].custom_id)
        if entrada is None:
            linhas.append({**base, "erro": "pendente", "relacoes": [], "afirmacoes": [], "malformados": 0})
        elif entrada.get("erro"):
            linhas.append({**base, "erro": entrada["erro"], "relacoes": [], "afirmacoes": [], "malformados": 0})
        else:
            linhas.append({**base, "erro": None, **processar_resultado(fatia, entrada["resultado"], ids_ancoras)})
    return linhas


def run_mapear(
    *,
    fatias: list[dict],
    ancoras: list[dict],
    out: Path,
    settings: av.Settings,
    client: object,
    ficha: str = "",
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda texto: None,
) -> dict:
    rt = ex.abrir_runtime(out, "mapear", settings, client, sleep, log)
    planejadas = planejar_mapear(fatias, ancoras, settings, ficha)
    restantes = ex.executar(rt, planejadas, PASSADA)
    linhas = consolidar_mapa(fatias, planejadas, rt.cp, ancoras)
    write_jsonl_atomic(out / MAPA_NOME, linhas)
    invalidos = sum(
        1 for linha in linhas for item in (*linha["relacoes"], *linha["afirmacoes"]) if not item["trecho_valido"]
    )
    return {
        "linhas": linhas,
        "requisicoes": len(planejadas),
        "pendentes": len(restantes),
        "erros": sum(1 for linha in linhas if linha["erro"] and linha["erro"] != "pendente"),
        "trechos_invalidos": invalidos,
        "incompleto": bool(restantes),
    }
