"""Etapa extra: classificação dos pinos de teste que travam texto da tese contra as âncoras.

Entrada: ``inventario_pinos.csv`` (``;``, UTF-8 com ou sem BOM; colunas de ``scripts/inventario_pinos.py``).
Ficam os pinos com ``categoria`` em frase, negacao ou ressalva. Por padrão saem os pinos cujo ``alvo`` só
cita ``defesa.tex`` e/ou ``roteiro_defesa.md`` (slides e roteiro serão refeitos; nenhum arquivo de
``text/latex``); ``incluir_slides`` os mantém. Alvos truncados (``+N``) ou dinâmicos não são excluídos.

Os pinos são ordenados (arquivo do teste, linha, função, literal) e agrupados em lotes de ``tamanho_lote``
(padrão 40), uma requisição por lote, com as âncoras no bloco de sistema em cache. Cada pino recebe um
``pino_id`` estável (sha256 de ``arquivo_teste|funcao|linha|literal``). A resposta é ``{"pinos": [...]}``;
JSON inválido, resposta truncada, pino ausente ou conflito fora da lista ficam registrados em ``erro`` no
``pinos.jsonl`` (nunca descartados em silêncio). ``pinos_meta.json`` guarda as contagens da seleção.
"""

from __future__ import annotations

import csv
import hashlib
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from scripts.atestado import avaliar as av
from scripts.atestado.common import write_json_atomic, write_jsonl_atomic

from . import execucao as ex
from .comum import (
    CATEGORIAS_PINO,
    CONFLITOS_PINO,
    PINOS_META_NOME,
    PINOS_NOME,
    id_curto,
    inteiro_ou_none,
)
from .prompts import sistema_pinos, usuario_pinos

PASSADA = 4
TAMANHO_LOTE_PADRAO = 40
ARQUIVOS_SLIDES = ("defesa.tex", "roteiro_defesa.md")
SAIDA_POR_PINO_TOKENS = 120


def pino_id(arquivo_teste: str, funcao: str, linha: str, literal: str) -> str:
    return id_curto(arquivo_teste, funcao, str(linha), literal)


# ---------------------------------------------------------------------------
# Inventário e seleção
# ---------------------------------------------------------------------------
def ler_inventario(caminho: Path) -> list[dict]:
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"{caminho} não existe")
    with caminho.open(encoding="utf-8-sig", newline="") as arquivo:
        return [dict(linha) for linha in csv.DictReader(arquivo, delimiter=";")]


def alvo_so_slides(alvo: str) -> bool:
    """True se o ``alvo`` cita só ``defesa.tex`` e/ou ``roteiro_defesa.md`` (nenhum outro arquivo)."""
    texto = (alvo or "").replace("(inferido)", "").strip()
    itens = [i.strip() for i in texto.split("|") if i.strip()]
    if not itens or any(i.startswith("+") or i.startswith("(") for i in itens):
        return False
    return all(i.replace("\\", "/").rsplit("/", 1)[-1] in ARQUIVOS_SLIDES for i in itens)


def selecionar(linhas: list[dict], incluir_slides: bool = False) -> tuple[list[dict], dict]:
    """Pinos de conteúdo ordenados e sem repetição, com ``pino_id``; e as contagens da seleção."""
    unicos: dict[str, dict] = {}
    for linha in linhas:
        if (linha.get("categoria") or "").strip() not in CATEGORIAS_PINO:
            continue
        pino = {k: (v or "") for k, v in linha.items() if k is not None}
        pino["pino_id"] = pino_id(pino.get("arquivo_teste", ""), pino.get("funcao", ""),
                                  pino.get("linha", ""), pino.get("literal", ""))
        unicos.setdefault(pino["pino_id"], pino)
    pinos = sorted(
        unicos.values(),
        key=lambda p: (p["arquivo_teste"], inteiro_ou_none(p["linha"]) or 0, p["funcao"], p["literal"]),
    )
    excluidos = 0
    if not incluir_slides:
        mantidos = [p for p in pinos if not alvo_so_slides(p.get("alvo", ""))]
        excluidos = len(pinos) - len(mantidos)
        pinos = mantidos
    meta = {
        "linhas_inventario": len(linhas), "pinos_de_conteudo": len(unicos), "incluir_slides": incluir_slides,
        "excluidos_slides": excluidos, "pinos_enviados": len(pinos),
    }
    return pinos, meta


def agrupar(pinos: list[dict], tamanho: int = TAMANHO_LOTE_PADRAO) -> list[list[dict]]:
    tamanho = max(1, tamanho)
    return [pinos[i : i + tamanho] for i in range(0, len(pinos), tamanho)]


# ---------------------------------------------------------------------------
# Planejamento
# ---------------------------------------------------------------------------
def planejar_pinos(
    pinos: list[dict], ancoras: list[dict], settings: av.Settings, tamanho_lote: int = TAMANHO_LOTE_PADRAO
) -> list[tuple[av.PlannedRequest, list[dict]]]:
    sistema = sistema_pinos(ancoras)
    digest = hashlib.sha256(sistema.encode("utf-8")).hexdigest()[:12]
    planos = []
    for lote in agrupar(pinos, tamanho_lote):
        ids = [p["pino_id"] for p in lote]
        cid = ex.custom_id("pin", ",".join(ids), settings.model, digest)
        params = ex.montar_params(settings, sistema, [ex.bloco_texto(usuario_pinos(lote))])
        planos.append((av.PlannedRequest(cid, id_curto(*ids, n=12), PASSADA, ids, params), lote))
    return planos


def estimar_pinos(
    pinos: list[dict], ancoras: list[dict], settings: av.Settings, tamanho_lote: int = TAMANHO_LOTE_PADRAO
) -> dict[str, int]:
    """Saída aproximada: 120 tokens por pino."""
    planos = planejar_pinos(pinos, ancoras, settings, tamanho_lote)
    return ex.estimar([req for req, _ in planos], lambda req, n: SAIDA_POR_PINO_TOKENS * len(req.ids))


# ---------------------------------------------------------------------------
# Processamento
# ---------------------------------------------------------------------------
def _booleano(valor) -> bool:
    if isinstance(valor, bool):
        return valor
    return str(valor).strip().lower() in ("true", "sim", "1")


def _id_ou_none(valor) -> str | None:
    texto = str(valor).strip() if valor is not None else ""
    return texto if texto and texto.lower() not in ("null", "none") else None


def _linha_pino(pino: dict, erro: str | None, item: dict | None, ids_ancoras: set[str]) -> dict:
    base = {
        "pino_id": pino["pino_id"], "arquivo_teste": pino.get("arquivo_teste", ""),
        "funcao": pino.get("funcao", ""), "linha": inteiro_ou_none(pino.get("linha")),
        "tipo": pino.get("tipo", ""), "categoria": pino.get("categoria", ""), "alvo": pino.get("alvo", ""),
        "literal": pino.get("literal", ""),
    }
    vazio = {
        "afirmacao_travada": "", "duplica_ancora": None, "conflito": None, "ancora_conflito": None,
        "central_sem_ancora": False, "justificativa": "", "flags": [], "erro": erro,
    }
    if item is None:
        return {**base, **vazio}
    flags: list[str] = []
    conflito = str(item.get("conflito") or "").strip()
    if conflito not in CONFLITOS_PINO:
        return {**base, **vazio, "erro": "conflito_invalido"}
    duplica = _id_ou_none(item.get("duplica_ancora"))
    ancora = _id_ou_none(item.get("ancora_conflito"))
    if any(i is not None and i not in ids_ancoras for i in (duplica, ancora)):
        flags.append("ancora_desconhecida")
    return {
        **base, "afirmacao_travada": str(item.get("afirmacao_travada") or ""), "duplica_ancora": duplica,
        "conflito": conflito, "ancora_conflito": ancora, "central_sem_ancora": _booleano(item.get("central_sem_ancora")),
        "justificativa": str(item.get("justificativa") or ""), "flags": flags, "erro": None,
    }


def consolidar_pinos(
    planos: list[tuple[av.PlannedRequest, list[dict]]], cp: av.Checkpoint, ancoras: list[dict]
) -> list[dict]:
    """Uma linha por pino, na ordem dos lotes: classificação ou erro (``pendente``, JSON inválido...)."""
    ids_ancoras = {a["id"] for a in ancoras}
    linhas: list[dict] = []
    for req, lote in planos:
        entrada = cp.done.get(req.custom_id)
        if entrada is None or entrada.get("erro"):
            erro = "pendente" if entrada is None else entrada["erro"]
            linhas.extend(_linha_pino(p, erro, None, ids_ancoras) for p in lote)
            continue
        bruto = entrada["resultado"].get("pinos")
        if not isinstance(bruto, list):
            linhas.extend(_linha_pino(p, "pinos_ausente", None, ids_ancoras) for p in lote)
            continue
        por_id = {str(i.get("pino_id")): i for i in bruto if isinstance(i, dict)}
        for pino in lote:
            item = por_id.get(pino["pino_id"])
            linhas.append(_linha_pino(pino, None if item is not None else "ausente_na_resposta", item, ids_ancoras))
    return linhas


def run_pinos(
    *,
    pinos: list[dict],
    ancoras: list[dict],
    out: Path,
    settings: av.Settings,
    client: object,
    tamanho_lote: int = TAMANHO_LOTE_PADRAO,
    meta: dict | None = None,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda texto: None,
) -> dict:
    rt = ex.abrir_runtime(out, "pinos", settings, client, sleep, log)
    planos = planejar_pinos(pinos, ancoras, settings, tamanho_lote)
    restantes = ex.executar(rt, [req for req, _ in planos], PASSADA)
    linhas = consolidar_pinos(planos, rt.cp, ancoras)
    write_jsonl_atomic(out / PINOS_NOME, linhas)
    if meta is not None:
        write_json_atomic(out / PINOS_META_NOME, meta)
    return {
        "linhas": linhas, "requisicoes": len(planos), "pendentes": len(restantes),
        "conflitos": dict(Counter(linha["conflito"] for linha in linhas if linha["conflito"])),
        "erros": sum(1 for linha in linhas if linha["erro"] and linha["erro"] != "pendente"),
        "incompleto": bool(restantes),
    }
