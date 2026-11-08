"""Etapa 4: adjudicação (Claude Opus) de cada par apontado em ``cruzar`` e de cada relação
``contradiz``/``exagera`` de ``mapear``, com as fatias completas dos dois lados.

Veredito: CONFIRMADO, FALSO_ALARME ou DEPENDE_DE_ESCOPO, com a correção mínima sugerida. O bloco de
sistema (rubrica) vai em cache; o primeiro bloco de fatia também, quando a mesma fatia aparece em mais
de um apontamento. ``temperature`` nunca é enviada.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from scripts.atestado import avaliar as av
from scripts.atestado.common import write_jsonl_atomic

from . import execucao as ex
from .comum import ADJUDICACOES_NOME, RELACOES_ADJUDICAVEIS, VEREDITOS
from .prompts import bloco_apontamento_par, bloco_apontamento_relacao, bloco_fatia, sistema_adjudicar

PASSADA = 3
SAIDA_ESTIMADA_TOKENS = 300


@dataclass
class PlanoAdjudicar:
    req: av.PlannedRequest
    alvo: dict


def relacoes_adjudicaveis(mapa: list[dict], ids_ancoras: set[str]) -> list[dict]:
    """Relações contradiz/exagera com trecho verificado e âncora conhecida, em ordem estável."""
    saida = []
    for linha in mapa:
        for rel in linha.get("relacoes", []):
            if (
                rel.get("relacao") in RELACOES_ADJUDICAVEIS
                and rel.get("trecho_valido")
                and rel.get("linha")
                and rel.get("ancora_id") in ids_ancoras
            ):
                saida.append(rel)
    return sorted(saida, key=lambda r: (r["arquivo"], r["linha"], r["id"]))


def planejar_adjudicar(
    pares: list[dict], mapa: list[dict], fatias: list[dict], ancoras: list[dict], settings: av.Settings,
    ficha: str = "",
) -> list[PlanoAdjudicar]:
    por_fatia = {f["id"]: f for f in fatias}
    por_ancora = {a["id"]: a for a in ancoras}
    alvos: list[dict] = []
    for par in sorted(pares, key=lambda p: p["id"]):
        alvos.append({
            "alvo_id": par["id"], "alvo_tipo": "par", "fatias": [par["a"]["fatia_id"], par["b"]["fatia_id"]],
            "apontamento": bloco_apontamento_par(par), "detalhe": par,
        })
    for rel in relacoes_adjudicaveis(mapa, set(por_ancora)):
        alvos.append({
            "alvo_id": rel["id"], "alvo_tipo": "relacao", "fatias": [rel["fatia_id"]],
            "apontamento": bloco_apontamento_relacao(rel, por_ancora[rel["ancora_id"]]), "detalhe": rel,
        })
    for alvo in alvos:
        ausentes = [i for i in alvo["fatias"] if i not in por_fatia]
        if ausentes:
            raise ValueError(f"fatia {ausentes[0]} ausente de fatias.jsonl; refaça `fatiar` e `mapear`")
        alvo["fatias"] = sorted(set(alvo["fatias"]))
    uso = Counter(i for alvo in alvos for i in alvo["fatias"])
    sistema = sistema_adjudicar(ficha)
    digest = hashlib.sha256(sistema.encode("utf-8")).hexdigest()[:12]
    planos = []
    for alvo in alvos:
        blocos = [
            ex.bloco_texto(bloco_fatia(por_fatia[i]), cache=k == 0 and uso[i] > 1)
            for k, i in enumerate(alvo["fatias"])
        ]
        blocos.append(ex.bloco_texto(alvo["apontamento"]))
        cid = ex.custom_id("adj", alvo["alvo_tipo"], alvo["alvo_id"], settings.model, digest)
        params = ex.montar_params(settings, sistema, blocos)
        planos.append(PlanoAdjudicar(av.PlannedRequest(cid, alvo["alvo_id"], PASSADA, [alvo["alvo_id"]], params), alvo))
    return planos


def estimar_adjudicar(
    pares: list[dict], mapa: list[dict], fatias: list[dict], ancoras: list[dict], settings: av.Settings,
    ficha: str = "",
) -> dict[str, int]:
    planos = planejar_adjudicar(pares, mapa, fatias, ancoras, settings, ficha)
    return ex.estimar([p.req for p in planos], lambda req, n: SAIDA_ESTIMADA_TOKENS)


def consolidar_adjudicacoes(planos: list[PlanoAdjudicar], cp: av.Checkpoint) -> list[dict]:
    linhas = []
    for plano in planos:
        alvo = plano.alvo
        base = {"alvo_id": alvo["alvo_id"], "alvo_tipo": alvo["alvo_tipo"], "detalhe": alvo["detalhe"]}
        entrada = cp.done.get(plano.req.custom_id)
        if entrada is None:
            linhas.append({**base, "veredito": None, "correcao": "", "justificativa": "", "erro": "pendente"})
            continue
        if entrada.get("erro"):
            linhas.append({**base, "veredito": None, "correcao": "", "justificativa": "", "erro": entrada["erro"]})
            continue
        resultado = entrada["resultado"]
        veredito = str(resultado.get("veredito") or "").strip().upper()
        if veredito not in VEREDITOS:
            linhas.append({**base, "veredito": None, "correcao": "", "justificativa": "", "erro": "veredito_invalido"})
            continue
        linhas.append({
            **base, "veredito": veredito, "correcao": str(resultado.get("correcao") or ""),
            "justificativa": str(resultado.get("justificativa") or ""), "erro": None,
        })
    return linhas


def run_adjudicar(
    *,
    pares: list[dict],
    mapa: list[dict],
    fatias: list[dict],
    ancoras: list[dict],
    out: Path,
    settings: av.Settings,
    client: object,
    ficha: str = "",
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda texto: None,
) -> dict:
    rt = ex.abrir_runtime(out, "adjudicar", settings, client, sleep, log)
    planos = planejar_adjudicar(pares, mapa, fatias, ancoras, settings, ficha)
    restantes = ex.executar(rt, [p.req for p in planos], PASSADA)
    linhas = consolidar_adjudicacoes(planos, rt.cp)
    write_jsonl_atomic(out / ADJUDICACOES_NOME, linhas)
    contagem = Counter(linha["veredito"] or "SEM_VEREDITO" for linha in linhas)
    return {
        "linhas": linhas, "requisicoes": len(planos), "pendentes": len(restantes),
        "vereditos": dict(contagem), "incompleto": bool(restantes),
        "erros": sum(1 for linha in linhas if linha["erro"] and linha["erro"] != "pendente"),
    }
