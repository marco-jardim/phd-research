"""Vários leitores independentes na etapa ``mapear``: união dos mapas e concordância entre eles.

Cada leitor grava o seu mapa (``mapa.jsonl`` do Sonnet, ``mapa_opencode.jsonl`` do DeepSeek). ``cruzar``,
``adjudicar`` e ``relatorio`` aceitam ``--mapas a.jsonl,b.jsonl`` e trabalham sobre a união.

Com um só mapa nada muda: as linhas voltam como foram lidas (sem o campo ``leitores``). Com dois ou mais:

* uma linha por fatia (ordem do primeiro mapa que a traz); ``erro`` só se todos os leitores falharam nela;
* as relações e as afirmações são unidas e deduplicadas pelo ``id`` do item, isto é, a mesma fatia e o mesmo
  trecho normalizado (nas relações, também a mesma âncora e a mesma relação: dois leitores que classificam o
  mesmo trecho de forma diferente geram dois itens, e os dois vão à adjudicação). O item guardado é o do
  primeiro leitor; ``leitores`` lista todos os que o trouxeram, na ordem dos mapas. Os ids não mudam;
* o ``id`` e o ``trecho_valido`` seguem o cálculo de ``mapear``: a verificação literal já foi feita por leitor.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .comum import MAPA_NOME, MAPA_OPENCODE_NOME, RELACOES, ler_jsonl

NOMES_LEITORES = {MAPA_NOME: "sonnet", MAPA_OPENCODE_NOME: "opencode"}
CAMPOS_ITENS = ("relacoes", "afirmacoes")


def nome_do_leitor(arquivo: str) -> str:
    """``sonnet`` para ``mapa.jsonl``, ``opencode`` para ``mapa_opencode.jsonl``; senão o nome sem ``mapa_``."""
    if arquivo in NOMES_LEITORES:
        return NOMES_LEITORES[arquivo]
    stem = Path(arquivo).stem
    return stem[len("mapa_"):] if stem.startswith("mapa_") and len(stem) > len("mapa_") else stem


def separar_mapas(texto: str) -> list[str]:
    """Nomes de ``--mapas`` sem repetição, na ordem dada."""
    return list(dict.fromkeys(parte.strip() for parte in texto.split(",") if parte.strip()))


def carregar_mapas(out: Path, arquivos: list[str]) -> tuple[list[dict], list[str], str | None]:
    """Mapa unido, nomes dos leitores e mensagem de erro (``None`` se tudo bem)."""
    out = Path(out)
    if not arquivos:
        return [], [], "--mapas vazio: informe ao menos um arquivo de mapa"
    leitura: list[tuple[str, list[dict]]] = []
    for arquivo in arquivos:
        linhas = ler_jsonl(out / arquivo)
        if not linhas:
            return [], [], f"{out / arquivo} ausente ou vazio; rode `mapear` antes"
        leitura.append((nome_do_leitor(arquivo), linhas))
    nomes = [nome for nome, _ in leitura]
    if len(set(nomes)) != len(nomes):
        return [], [], f"--mapas com leitores repetidos: {', '.join(arquivos)}"
    return unir_mapas(leitura), nomes, None


def unir_mapas(mapas: list[tuple[str, list[dict]]]) -> list[dict]:
    """União de ``[(leitor, linhas)]``; um só mapa volta sem alteração."""
    if len(mapas) == 1:
        return mapas[0][1]
    unidas: dict[str, dict] = {}
    for leitor, linhas in mapas:
        for linha in linhas:
            fatia = linha["fatia_id"]
            destino = unidas.get(fatia)
            if destino is None:
                destino = unidas[fatia] = {
                    **{k: v for k, v in linha.items() if k not in CAMPOS_ITENS},
                    "relacoes": [], "afirmacoes": [], "_vistos": {c: {} for c in CAMPOS_ITENS},
                    "malformados": 0, "_erros": [],
                }
            destino["_erros"].append(linha.get("erro"))
            destino["malformados"] += int(linha.get("malformados") or 0)
            for campo in CAMPOS_ITENS:
                for item in linha.get(campo, []):
                    vistos = destino["_vistos"][campo]
                    existente = vistos.get(item["id"])
                    if existente is None:
                        copia = {**item, "leitores": [leitor]}
                        vistos[item["id"]] = copia
                        destino[campo].append(copia)
                    elif leitor not in existente["leitores"]:
                        existente["leitores"].append(leitor)
    resultado = []
    for linha in unidas.values():
        erros = linha.pop("_erros")
        linha.pop("_vistos")
        linha["erro"] = None if any(e is None for e in erros) else erros[0]
        resultado.append(linha)
    return resultado


# ---------------------------------------------------------------------------
# Concordância
# ---------------------------------------------------------------------------
def concordancia_por_ancora(mapa: list[dict], leitores: list[str], ids_ancoras: list[str]) -> list[dict]:
    """Uma linha por (âncora, relação) com contagem só de cada leitor e de vários (``ambos``).

    Só entram relações com trecho verificado, relação válida e âncora conhecida. ``ids_ancoras`` dá a ordem.
    """
    ordem = {a: i for i, a in enumerate(ids_ancoras)}
    contagens: dict[tuple[str, str], Counter] = {}
    for linha in mapa:
        for rel in linha.get("relacoes", []):
            if not rel.get("trecho_valido") or rel.get("relacao") not in RELACOES or rel.get("ancora_id") not in ordem:
                continue
            quem = rel.get("leitores") or []
            chave = "ambos" if len(quem) > 1 else (quem[0] if quem else "?")
            contagens.setdefault((rel["ancora_id"], rel["relacao"]), Counter())[chave] += 1
    linhas = []
    for (ancora, relacao) in sorted(contagens, key=lambda k: (ordem[k[0]], RELACOES.index(k[1]))):
        c = contagens[(ancora, relacao)]
        linhas.append({
            "ancora_id": ancora, "relacao": relacao,
            **{f"so_{leitor}": c[leitor] for leitor in leitores}, "ambos": c["ambos"],
        })
    return linhas
