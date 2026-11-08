"""Etapa 3 (reduce por tópico): pares de afirmações inconsistentes entre arquivos ou seções.

As afirmações com trecho verificado (``mapa.jsonl``) são agrupadas por tópico (uma afirmação com vários
tópicos entra em cada um; ``outro`` fica de fora, salvo ``incluir_outro``). Cada tópico vai numa requisição.
Se passar de ``max_afirmacoes`` (padrão 60), o tópico é dividido em lotes ordenados por arquivo e linha e
cada lote é conferido com ele mesmo (grupo A) e com cada lote seguinte (grupos A e B; o modelo só aponta
pares com um item em cada grupo), para que nenhum par fique de fora por causa da divisão.

O modelo devolve apenas rótulos (``A3``, ``B17``), o tipo e uma justificativa; os trechos literais, o
arquivo e a linha de cada lado vêm do mapa (portanto são sempre literais). Rótulos inexistentes e pares de
uma afirmação consigo mesma vão para ``pares_problemas.jsonl``, sem descarte silencioso.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from scripts.atestado import avaliar as av
from scripts.atestado.common import write_jsonl_atomic

from . import execucao as ex
from .comum import PARES_NOME, TIPOS_INCONSISTENCIA, TOPICO_OUTRO, TOPICOS, id_curto
from .prompts import bloco_afirmacoes, sistema_cruzar, usuario_cruzar_topico

PASSADA = 2
MAX_AFIRMACOES_PADRAO = 60
PARES_PROBLEMAS_NOME = "pares_problemas.jsonl"
RE_ROTULO = re.compile(r"^\s*([AB])\s*(\d+)\s*$")


@dataclass
class PlanoCruzar:
    req: av.PlannedRequest
    topico: str
    a: list[dict]
    b: list[dict] | None


def afirmacoes_validas(mapa: list[dict]) -> list[dict]:
    """Afirmações com trecho literal na fatia, com arquivo e linha, em ordem estável."""
    saida = []
    for linha in mapa:
        for af in linha.get("afirmacoes", []):
            if af.get("trecho_valido") and af.get("linha"):
                saida.append(af)
    return sorted(saida, key=lambda af: (af["arquivo"], af["linha"], af["id"]))


def agrupar_por_topico(afirmacoes: list[dict], incluir_outro: bool = False) -> dict[str, list[dict]]:
    grupos: dict[str, list[dict]] = {}
    for af in afirmacoes:
        for topico in af.get("topicos") or [TOPICO_OUTRO]:
            if topico == TOPICO_OUTRO and not incluir_outro:
                continue
            lista = grupos.setdefault(topico, [])
            if all(existente["id"] != af["id"] for existente in lista):
                lista.append(af)
    ordem = {t: i for i, t in enumerate(TOPICOS)}
    return {t: grupos[t] for t in sorted(grupos, key=lambda t: (ordem.get(t, len(ordem)), t))}


def dividir_lotes(afirmacoes: list[dict], tamanho: int) -> list[list[dict]]:
    tamanho = max(1, tamanho)
    return [afirmacoes[i : i + tamanho] for i in range(0, len(afirmacoes), tamanho)]


def planejar_cruzar(
    mapa: list[dict],
    settings: av.Settings,
    max_afirmacoes: int = MAX_AFIRMACOES_PADRAO,
    incluir_outro: bool = False,
) -> list[PlanoCruzar]:
    sistema = sistema_cruzar()
    digest = hashlib.sha256(sistema.encode("utf-8")).hexdigest()[:12]
    planos: list[PlanoCruzar] = []
    grupos = agrupar_por_topico(afirmacoes_validas(mapa), incluir_outro)
    for topico, afirmacoes in grupos.items():
        if len(afirmacoes) < 2:
            continue
        lotes = dividir_lotes(afirmacoes, max_afirmacoes)
        dividido = len(lotes) > 1
        for i, lote_a in enumerate(lotes):
            for j in range(i, len(lotes)):
                lote_b = None if j == i else lotes[j]
                if lote_b is None and len(lote_a) < 2:
                    continue  # lote de uma afirmação só: não há par dentro dele
                bloco_a = ex.bloco_texto(
                    usuario_cruzar_topico(topico) + bloco_afirmacoes("A", lote_a), cache=dividido
                )
                blocos = [bloco_a]
                if lote_b is not None:
                    blocos.append(ex.bloco_texto(bloco_afirmacoes("B", lote_b)))
                ids = [af["id"] for af in lote_a] + [af["id"] for af in (lote_b or [])]
                cid = ex.custom_id("crz", topico, ",".join(ids), settings.model, digest)
                params = ex.montar_params(settings, sistema, blocos)
                planos.append(PlanoCruzar(av.PlannedRequest(cid, topico, PASSADA, ids, params), topico, lote_a, lote_b))
    return planos


def estimar_cruzar(
    mapa: list[dict], settings: av.Settings, max_afirmacoes: int = MAX_AFIRMACOES_PADRAO,
    incluir_outro: bool = False,
) -> dict[str, int]:
    """Saída aproximada: 200 tokens mais 5 por afirmação enviada."""
    planos = planejar_cruzar(mapa, settings, max_afirmacoes, incluir_outro)
    return ex.estimar([p.req for p in planos], lambda req, n: 200 + 5 * len(req.ids))


# ---------------------------------------------------------------------------
# Processamento
# ---------------------------------------------------------------------------
def _lado(af: dict) -> dict:
    return {
        "afirmacao_id": af["id"], "fatia_id": af["fatia_id"], "arquivo": af["arquivo"],
        "linha": af["linha"], "trecho": af["trecho"],
    }


def _resolver(rotulo, plano: PlanoCruzar) -> dict | None:
    m = RE_ROTULO.match(str(rotulo))
    if not m:
        return None
    lista = plano.a if m.group(1) == "A" else plano.b
    indice = int(m.group(2))
    if lista is None or not 1 <= indice <= len(lista):
        return None
    return lista[indice - 1]


def consolidar_pares(planos: list[PlanoCruzar], cp: av.Checkpoint) -> tuple[list[dict], list[dict]]:
    """Pares únicos (por afirmações e tipo) e problemas de resposta, em ordem estável."""
    pares: dict[str, dict] = {}
    problemas: list[dict] = []
    for plano in planos:
        entrada = cp.done.get(plano.req.custom_id)
        if entrada is None:
            continue
        if entrada.get("erro"):
            problemas.append({"custom_id": plano.req.custom_id, "topico": plano.topico, "motivo": entrada["erro"]})
            continue
        bruto = entrada["resultado"].get("pares")
        if not isinstance(bruto, list):
            problemas.append({"custom_id": plano.req.custom_id, "topico": plano.topico, "motivo": "pares_ausente"})
            continue
        for item in bruto:
            if not isinstance(item, dict):
                problemas.append({"custom_id": plano.req.custom_id, "topico": plano.topico, "motivo": "item_malformado"})
                continue
            af_a, af_b = _resolver(item.get("a"), plano), _resolver(item.get("b"), plano)
            if af_a is None or af_b is None:
                problemas.append({
                    "custom_id": plano.req.custom_id, "topico": plano.topico, "motivo": "rotulo_invalido",
                    "a": str(item.get("a")), "b": str(item.get("b")),
                })
                continue
            if af_a["id"] == af_b["id"]:
                problemas.append({
                    "custom_id": plano.req.custom_id, "topico": plano.topico, "motivo": "par_consigo_mesmo",
                    "a": str(item.get("a")), "b": str(item.get("b")),
                })
                continue
            tipo = str(item.get("tipo") or "").strip()
            flags = [] if tipo in TIPOS_INCONSISTENCIA else ["tipo_invalido"]
            lados = sorted([af_a, af_b], key=lambda af: (af["arquivo"], af["linha"], af["id"]))
            pid = id_curto(lados[0]["id"], lados[1]["id"], tipo)
            existente = pares.get(pid)
            if existente is not None:
                if plano.topico not in existente["topicos"]:
                    existente["topicos"].append(plano.topico)
                continue
            pares[pid] = {
                "id": pid, "tipo": tipo, "topico": plano.topico, "topicos": [plano.topico],
                "justificativa": str(item.get("justificativa") or ""),
                "a": _lado(lados[0]), "b": _lado(lados[1]), "flags": flags,
            }
    return list(pares.values()), problemas


def run_cruzar(
    *,
    mapa: list[dict],
    out: Path,
    settings: av.Settings,
    client: object,
    max_afirmacoes: int = MAX_AFIRMACOES_PADRAO,
    incluir_outro: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda texto: None,
) -> dict:
    rt = ex.abrir_runtime(out, "cruzar", settings, client, sleep, log)
    planos = planejar_cruzar(mapa, settings, max_afirmacoes, incluir_outro)
    restantes = ex.executar(rt, [p.req for p in planos], PASSADA)
    pares, problemas = consolidar_pares(planos, rt.cp)
    write_jsonl_atomic(out / PARES_NOME, pares)
    write_jsonl_atomic(out / PARES_PROBLEMAS_NOME, problemas)
    return {
        "pares": pares, "problemas": problemas, "requisicoes": len(planos),
        "pendentes": len(restantes), "incompleto": bool(restantes),
    }
