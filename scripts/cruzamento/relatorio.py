"""Etapa 5: relatório em Markdown e matriz âncora × arquivo × relação (CSV).

Lê só o que as etapas anteriores gravaram em ``.local/cruzamento/`` (fatias, mapa, pares, adjudicações e
os checkpoints, de onde vêm os tokens de ``usage``). Não chama a API.

"Afirmações frequentes sem âncora" é uma heurística de contagem, não agrupamento semântico: uma
afirmação conta como sem âncora quando nenhuma relação da mesma fatia cobre a mesma linha ou o mesmo
trecho; os grupos são formados pelo primeiro tópico e pelos números do trecho (ou, sem números, por
tipo, polaridade e escopo) e entram na lista quando têm ao menos ``min_ocorrencias`` ocorrências.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from pathlib import Path

from scripts.atestado.avaliar import USAGE_FIELDS
from scripts.atestado.common import read_json, write_text_atomic

from .comum import (
    ADJUDICACOES_NOME,
    CONFLITOS_PINO,
    ETAPAS,
    FATIAS_NOME,
    MAPA_NOME,
    MATRIZ_NOME,
    PARES_NOME,
    PINOS_META_NOME,
    PINOS_NOME,
    RELACOES,
    RELATORIO_NOME,
    colapsar_espacos,
    formatar_milhar,
    ler_jsonl,
)
from .cruzar import PARES_PROBLEMAS_NOME
from .leitores import carregar_mapas, concordancia_por_ancora

TRECHO_MAX = 180
LIMITE_LISTA = 50


def _curto(texto: str, limite: int = TRECHO_MAX) -> str:
    t = " ".join(texto.split())
    return t if len(t) <= limite else t[: limite - 1] + "…"


def _local(item: dict) -> str:
    return f"{item['arquivo']}:{item['linha']}"


# ---------------------------------------------------------------------------
# Matriz e candidatas
# ---------------------------------------------------------------------------
def relacoes_validas(mapa: list[dict], ids_ancoras: set[str]) -> list[dict]:
    return [
        rel
        for linha in mapa
        for rel in linha.get("relacoes", [])
        if rel.get("trecho_valido") and rel.get("relacao") in RELACOES and rel.get("ancora_id") in ids_ancoras
    ]


def montar_matriz(mapa: list[dict], ancoras: list[dict]) -> list[dict]:
    """Uma linha por (âncora, arquivo) com contagem por relação; só relações com trecho verificado."""
    ordem = {a["id"]: i for i, a in enumerate(ancoras)}
    contagens: dict[tuple[str, str], Counter] = {}
    for rel in relacoes_validas(mapa, set(ordem)):
        contagens.setdefault((rel["ancora_id"], rel["arquivo"]), Counter())[rel["relacao"]] += 1
    linhas = []
    for (ancora, arquivo) in sorted(contagens, key=lambda k: (ordem[k[0]], k[1])):
        c = contagens[(ancora, arquivo)]
        linha = {"ancora_id": ancora, "arquivo": arquivo}
        linha.update({r: c[r] for r in RELACOES})
        linha["total"] = sum(c.values())
        linhas.append(linha)
    return linhas


def matriz_csv(linhas: list[dict]) -> str:
    campos = ["ancora_id", "arquivo", *RELACOES, "total"]
    buffer = io.StringIO()
    escritor = csv.DictWriter(buffer, fieldnames=campos, delimiter=";", lineterminator="\n")
    escritor.writeheader()
    escritor.writerows(linhas)
    return buffer.getvalue()


def _coberta(af: dict, relacoes_da_fatia: list[dict]) -> bool:
    trecho = colapsar_espacos(af["trecho"])
    for rel in relacoes_da_fatia:
        outro = colapsar_espacos(rel["trecho"])
        if rel.get("linha") == af.get("linha") or trecho in outro or outro in trecho:
            return True
    return False


def candidatas_a_ancora(mapa: list[dict], min_ocorrencias: int = 3) -> list[dict]:
    grupos: dict[tuple, list[dict]] = {}
    for linha in mapa:
        relacoes = [r for r in linha.get("relacoes", []) if r.get("trecho_valido")]
        for af in linha.get("afirmacoes", []):
            if not af.get("trecho_valido") or _coberta(af, relacoes):
                continue
            topico = (af.get("topicos") or ["outro"])[0]
            numeros = tuple(sorted(colapsar_espacos(n) for n in af.get("numeros") or []))
            if numeros:
                chave = (topico, numeros, af.get("polaridade"))
            else:
                chave = (topico, (), af.get("tipo"), af.get("polaridade"), colapsar_espacos(af.get("escopo") or "").lower())
            grupos.setdefault(chave, []).append(af)
    saida = []
    for chave, itens in grupos.items():
        if len(itens) < min_ocorrencias:
            continue
        saida.append({
            "topico": chave[0], "numeros": list(chave[1]), "ocorrencias": len(itens),
            "arquivos": len({i["arquivo"] for i in itens}), "escopo": itens[0].get("escopo") or "",
            "exemplo": itens[0]["trecho"], "locais": [_local(i) for i in itens],
        })
    return sorted(saida, key=lambda g: (-g["ocorrencias"], g["topico"], g["numeros"]))


# ---------------------------------------------------------------------------
# Custo
# ---------------------------------------------------------------------------
def uso_por_etapa(out: Path) -> dict[str, dict[str, dict[str, int]]]:
    """Tokens de ``usage`` por etapa e modo, lidos dos checkpoints (``{}`` se não há checkpoint)."""
    resultado: dict[str, dict[str, dict[str, int]]] = {}
    for etapa in ETAPAS:
        estado = read_json(out / f"{etapa}.checkpoint.json")
        if not estado:
            continue
        modos: dict[str, dict[str, int]] = {}
        for entrada in estado.get("requisicoes", {}).values():
            acumulado = modos.setdefault(entrada["modo"], {**dict.fromkeys(USAGE_FIELDS, 0), "requisicoes": 0})
            acumulado["requisicoes"] += 1
            for campo in USAGE_FIELDS:
                acumulado[campo] += int(entrada.get("uso", {}).get(campo, 0))
        resultado[etapa] = modos
    return resultado


CHAVE_NAO_REGISTRADA = "(não registrada)"


def uso_por_chave(out: Path) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    """Tokens de ``usage`` por NOME da variável da chave, etapa e modo (o valor da chave nunca é gravado)."""
    resultado: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    for etapa in ETAPAS:
        estado = read_json(out / f"{etapa}.checkpoint.json")
        if not estado:
            continue
        for entrada in estado.get("requisicoes", {}).values():
            chave = entrada.get("chave") or CHAVE_NAO_REGISTRADA
            acumulado = resultado.setdefault(chave, {}).setdefault(etapa, {}).setdefault(
                entrada["modo"], {**dict.fromkeys(USAGE_FIELDS, 0), "requisicoes": 0})
            acumulado["requisicoes"] += 1
            for campo in USAGE_FIELDS:
                acumulado[campo] += int(entrada.get("uso", {}).get(campo, 0))
    return resultado


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------
def _veredito(adj_por_id: dict[str, dict], alvo_id: str) -> str:
    linha = adj_por_id.get(alvo_id)
    if linha is None:
        return "não adjudicado"
    return linha["veredito"] or f"sem veredito ({linha['erro']})"


def secao_pinos(pinos: list[dict], meta: dict) -> list[str]:
    """Seção "Pinos": contagem por conflito, pinos em conflito e pinos centrais sem âncora."""
    L: list[str] = ["## Pinos", ""]
    if not pinos:
        L.extend(["- `pinos` não executado (sem `pinos.jsonl`)", ""])
        return L
    classificados = [p for p in pinos if not p.get("erro")]
    if "excluidos_slides" in meta:
        motivo = "incluídos" if meta.get("incluir_slides") else "excluídos"
        L.append(f"- Pinos cujo alvo só cita `defesa.tex` ou `roteiro_defesa.md` {motivo} "
                 f"(slides e roteiro serão refeitos): {meta['excluidos_slides']}")
    L.append(f"- Pinos no arquivo: {len(pinos)}; classificados: {len(classificados)}; "
             f"com erro ou pendentes: {dict(sorted(Counter(p['erro'] for p in pinos if p.get('erro')).items())) or 'nenhum'}")
    contagem = Counter(p["conflito"] for p in classificados)
    L.append("- Por conflito: " + ", ".join(f"{c} {contagem[c]}" for c in CONFLITOS_PINO))
    L.append(f"- Pinos que duplicam uma âncora: {sum(1 for p in classificados if p.get('duplica_ancora'))}")
    L.append("")
    em_conflito = [p for p in classificados if p["conflito"] != "nenhum"]
    L.append(f"### Pinos em conflito ({len(em_conflito)})")
    L.append("")
    if not em_conflito:
        L.append("- nenhum")
    for p in em_conflito[:LIMITE_LISTA]:
        L.append(f"- {p['conflito']}: `{p['arquivo_teste']}:{p['linha']}` \"{_curto(p['literal'])}\" "
                 f"(âncora {p.get('ancora_conflito') or '-'}): {_curto(p['justificativa'])}")
    if len(em_conflito) > LIMITE_LISTA:
        L.append(f"- ... e mais {len(em_conflito) - LIMITE_LISTA} em `pinos.jsonl`")
    L.append("")
    centrais = [p for p in classificados if p.get("central_sem_ancora")]
    L.append(f"### Pinos centrais sem âncora ({len(centrais)})")
    L.append("")
    if not centrais:
        L.append("- nenhum")
    for p in centrais[:LIMITE_LISTA]:
        L.append(f"- `{p['arquivo_teste']}:{p['linha']}` \"{_curto(p['literal'])}\": "
                 f"{_curto(p['afirmacao_travada'])}")
    if len(centrais) > LIMITE_LISTA:
        L.append(f"- ... e mais {len(centrais) - LIMITE_LISTA} em `pinos.jsonl`")
    L.append("")
    return L


def secao_concordancia(mapa: list[dict], leitores: list[str], ancoras: list[dict]) -> list[str]:
    """Seção "Concordância entre leitores": por âncora, relações só de cada leitor e de vários (ambos).

    Dois leitores concordam quando o item é o mesmo (mesma fatia, mesmo trecho normalizado, mesma âncora e
    mesma relação); o mesmo lugar citado com outro recorte de trecho conta como um item de cada leitor.
    """
    L: list[str] = ["## Concordância entre leitores", ""]
    L.append(f"Leitores: {', '.join(leitores)}. Só entram relações com trecho verificado na fatia; "
             "a lista nominal traz as relações não neutras achadas por um leitor só.")
    L.append("")
    linhas = concordancia_por_ancora(mapa, leitores, [a["id"] for a in ancoras])
    cabecalho = ["Relação", *(f"Só {leitor}" for leitor in leitores), "Ambos"]
    for ancora in ancoras:
        L.append(f"### {ancora['id']} ({ancora['origem']})")
        L.append("")
        proprias = [linha for linha in linhas if linha["ancora_id"] == ancora["id"]]
        if not proprias:
            L.append("- nenhuma relação")
            L.append("")
            continue
        L.append("| " + " | ".join(cabecalho) + " |")
        L.append("|---|" + "---:|" * (len(cabecalho) - 1))
        for linha in proprias:
            valores = [linha[f"so_{leitor}"] for leitor in leitores] + [linha["ambos"]]
            L.append(f"| {linha['relacao']} | " + " | ".join(str(v) for v in valores) + " |")
        L.append("")
        so_um = sorted(
            (
                (r, r["leitores"][0])
                for linha in mapa
                for r in linha.get("relacoes", [])
                if r.get("trecho_valido") and r.get("ancora_id") == ancora["id"] and r.get("relacao") in RELACOES
                and r.get("relacao") != "afirma" and len(r.get("leitores") or []) == 1
            ),
            key=lambda x: (x[1], x[0]["arquivo"], x[0]["linha"], x[0]["id"]),
        )
        for r, leitor in so_um[:LIMITE_LISTA]:
            L.append(f"- só {leitor}: {r['relacao']} `{_local(r)}` \"{_curto(r['trecho'])}\"")
        if len(so_um) > LIMITE_LISTA:
            L.append(f"- ... e mais {len(so_um) - LIMITE_LISTA} em `mapa*.jsonl`")
        if so_um:
            L.append("")
    return L


def construir_relatorio(
    fatias: list[dict], mapa: list[dict], pares: list[dict], problemas_pares: list[dict],
    adjudicacoes: list[dict], ancoras: list[dict], uso: dict, min_ocorrencias: int = 3,
    pinos: list[dict] | None = None, pinos_meta: dict | None = None, uso_chave: dict | None = None,
    leitores: list[str] | None = None,
) -> str:
    adj = {a["alvo_id"]: a for a in adjudicacoes}
    ids_ancoras = {a["id"] for a in ancoras}
    validas = relacoes_validas(mapa, ids_ancoras)
    L: list[str] = ["# Cruzamento da tese com as âncoras", ""]
    L.append(
        "Gerado por `scripts/cruzamento_ancoras.py relatorio` a partir de `.local/cruzamento/`. "
        "Não editar à mão. Só entram relações e afirmações com trecho literal verificado na fatia."
    )
    L.append("")

    # --- resumo ----------------------------------------------------------------
    fontes = Counter(f["fonte"] for f in fatias)
    erros = Counter(linha["erro"] for linha in mapa if linha.get("erro"))
    n_afirmacoes = sum(1 for linha in mapa for a in linha.get("afirmacoes", []) if a.get("trecho_valido"))
    L.append("## Resumo")
    L.append("")
    L.append(f"- Fatias: {len(fatias)} ({', '.join(f'{k} {v}' for k, v in sorted(fontes.items()))}); "
             f"mapeadas sem erro: {sum(1 for linha in mapa if not linha.get('erro'))}; "
             f"com erro ou pendentes: {sum(erros.values())}")
    por_relacao = Counter(v["relacao"] for v in validas)
    L.append(f"- Relações não neutras com trecho verificado: {len(validas)} "
             f"({', '.join(f'{r} {por_relacao[r]}' for r in RELACOES)})")
    L.append(f"- Afirmações com trecho verificado: {n_afirmacoes}")
    L.append(f"- Pares inconsistentes apontados: {len(pares)}")
    contagem = Counter(_veredito(adj, a["alvo_id"]) for a in adjudicacoes)
    L.append(f"- Adjudicações: {len(adjudicacoes)} ({', '.join(f'{k} {v}' for k, v in sorted(contagem.items())) or 'nenhuma'})")
    L.append("")

    # --- por âncora --------------------------------------------------------------
    L.append("## Ocorrências por âncora")
    L.append("")
    for ancora in ancoras:
        L.append(f"### {ancora['id']} ({ancora['origem']})")
        L.append("")
        L.append(ancora["texto"])
        if ancora["ressalva"]:
            L.append("")
            L.append(f"Ressalva: {ancora['ressalva']}")
        L.append("")
        proprias = [r for r in validas if r["ancora_id"] == ancora["id"]]
        if not proprias:
            L.append("- nenhuma ocorrência")
        for relacao in RELACOES:
            itens = sorted((r for r in proprias if r["relacao"] == relacao), key=lambda r: (r["arquivo"], r["linha"]))
            if not itens:
                continue
            if relacao == "afirma":
                L.append(f"- **afirma** ({len(itens)}): " + ", ".join(f"`{_local(r)}`" for r in itens))
                continue
            L.append(f"- **{relacao}** ({len(itens)}):")
            for r in itens:
                L.append(f"  - `{_local(r)}` [{_veredito(adj, r['id'])}] \"{_curto(r['trecho'])}\": {_curto(r['justificativa'])}")
        L.append("")

    # --- concordância entre leitores -------------------------------------------------
    if leitores and len(leitores) > 1:
        L.extend(secao_concordancia(mapa, leitores, ancoras))

    # --- contradições confirmadas ----------------------------------------------------
    L.append("## Contradições confirmadas")
    L.append("")
    confirmados = [a for a in adjudicacoes if a["veredito"] == "CONFIRMADO"]
    dos_pares = [a for a in confirmados if a["alvo_tipo"] == "par"]
    das_relacoes = [a for a in confirmados if a["alvo_tipo"] == "relacao"]
    L.append("### Entre trechos (por tópico)")
    L.append("")
    if not dos_pares:
        L.append("- nenhuma")
    por_topico: dict[str, list[dict]] = {}
    for a in dos_pares:
        por_topico.setdefault(a["detalhe"]["topico"], []).append(a)
    for topico in sorted(por_topico):
        L.append(f"#### {topico}")
        L.append("")
        for a in sorted(por_topico[topico], key=lambda x: (x["detalhe"]["a"]["arquivo"], x["detalhe"]["a"]["linha"])):
            par = a["detalhe"]
            L.append(f"- {par['tipo']}: `{_local(par['a'])}` \"{_curto(par['a']['trecho'])}\" × "
                     f"`{_local(par['b'])}` \"{_curto(par['b']['trecho'])}\"")
            L.append(f"  - correção: {_curto(a['correcao'], 300) or '-'}")
            L.append(f"  - justificativa: {_curto(a['justificativa'], 300) or '-'}")
        L.append("")
    L.append("### Contra âncoras (por âncora)")
    L.append("")
    if not das_relacoes:
        L.append("- nenhuma")
    por_ancora: dict[str, list[dict]] = {}
    for a in das_relacoes:
        por_ancora.setdefault(a["detalhe"]["ancora_id"], []).append(a)
    for ancora_id in sorted(por_ancora):
        L.append(f"#### {ancora_id}")
        L.append("")
        for a in sorted(por_ancora[ancora_id], key=lambda x: (x["detalhe"]["arquivo"], x["detalhe"]["linha"])):
            rel = a["detalhe"]
            L.append(f"- {rel['relacao']}: `{_local(rel)}` \"{_curto(rel['trecho'])}\"")
            L.append(f"  - correção: {_curto(a['correcao'], 300) or '-'}")
            L.append(f"  - justificativa: {_curto(a['justificativa'], 300) or '-'}")
        L.append("")
    dependentes = [a for a in adjudicacoes if a["veredito"] == "DEPENDE_DE_ESCOPO"]
    L.append("### Depende de escopo")
    L.append("")
    if not dependentes:
        L.append("- nenhum")
    for a in dependentes:
        d = a["detalhe"]
        if a["alvo_tipo"] == "par":
            L.append(f"- par ({d['topico']}, {d['tipo']}): `{_local(d['a'])}` × `{_local(d['b'])}`: {_curto(a['correcao'], 300) or '-'}")
        else:
            L.append(f"- {d['ancora_id']} {d['relacao']}: `{_local(d)}`: {_curto(a['correcao'], 300) or '-'}")
    L.append("")

    # --- candidatas ------------------------------------------------------------------
    candidatas = candidatas_a_ancora(mapa, min_ocorrencias)
    L.append("## Afirmações frequentes sem âncora (candidatas a âncora)")
    L.append("")
    L.append(f"Heurística de contagem (ver o cabeçalho de `scripts/cruzamento/relatorio.py`); mínimo de {min_ocorrencias} ocorrências.")
    L.append("")
    if not candidatas:
        L.append("- nenhuma")
    for g in candidatas[:LIMITE_LISTA]:
        numeros = f"; números: {', '.join(g['numeros'])}" if g["numeros"] else ""
        L.append(f"- {g['topico']}: {g['ocorrencias']} ocorrências em {g['arquivos']} arquivo(s){numeros}; "
                 f"escopo: {g['escopo'] or '-'}")
        L.append(f"  - exemplo: \"{_curto(g['exemplo'])}\"")
        L.append(f"  - onde: {', '.join(f'`{x}`' for x in g['locais'][:8])}")
    L.append("")

    # --- pinos -----------------------------------------------------------------------
    L.extend(secao_pinos(pinos or [], pinos_meta or {}))

    # --- custo -----------------------------------------------------------------------
    L.append("## Custo (tokens de `usage`)")
    L.append("")
    L.append("| Etapa | Modo | Requisições | Entrada | Escrita em cache | Leitura de cache | Saída |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    total = dict.fromkeys(("requisicoes", *USAGE_FIELDS), 0)
    for etapa in ETAPAS:
        for modo, u in sorted(uso.get(etapa, {}).items()):
            L.append(
                f"| {etapa} | {modo} | {formatar_milhar(u['requisicoes'])} | {formatar_milhar(u['input_tokens'])} | "
                f"{formatar_milhar(u['cache_creation_input_tokens'])} | {formatar_milhar(u['cache_read_input_tokens'])} | "
                f"{formatar_milhar(u['output_tokens'])} |"
            )
            for campo in total:
                total[campo] += u[campo]
    L.append(
        f"| **Total** | | {formatar_milhar(total['requisicoes'])} | {formatar_milhar(total['input_tokens'])} | "
        f"{formatar_milhar(total['cache_creation_input_tokens'])} | {formatar_milhar(total['cache_read_input_tokens'])} | "
        f"{formatar_milhar(total['output_tokens'])} |"
    )
    L.append("")
    if uso_chave:
        L.append("### Por chave (nome da variável de ambiente)")
        L.append("")
        L.append("| Chave | Etapa | Modo | Requisições | Entrada | Escrita em cache | Leitura de cache | Saída |")
        L.append("|---|---|---|---:|---:|---:|---:|---:|")
        for chave in sorted(uso_chave):
            soma = dict.fromkeys(("requisicoes", *USAGE_FIELDS), 0)
            for etapa in ETAPAS:
                for modo, u in sorted(uso_chave[chave].get(etapa, {}).items()):
                    L.append(
                        f"| {chave} | {etapa} | {modo} | {formatar_milhar(u['requisicoes'])} | "
                        f"{formatar_milhar(u['input_tokens'])} | {formatar_milhar(u['cache_creation_input_tokens'])} | "
                        f"{formatar_milhar(u['cache_read_input_tokens'])} | {formatar_milhar(u['output_tokens'])} |"
                    )
                    for campo in soma:
                        soma[campo] += u[campo]
            L.append(
                f"| {chave} | **Total** | | {formatar_milhar(soma['requisicoes'])} | "
                f"{formatar_milhar(soma['input_tokens'])} | {formatar_milhar(soma['cache_creation_input_tokens'])} | "
                f"{formatar_milhar(soma['cache_read_input_tokens'])} | {formatar_milhar(soma['output_tokens'])} |"
            )
        L.append("")

    # --- problemas --------------------------------------------------------------------
    L.append("## Problemas de execução e de verificação")
    L.append("")
    flags = Counter(f for linha in mapa for item in (*linha.get("relacoes", []), *linha.get("afirmacoes", []))
                    for f in item.get("flags", []))
    L.append(f"- Fatias com erro ou pendentes: {dict(sorted(erros.items())) or 'nenhuma'}")
    L.append(f"- Itens malformados descartados do mapa: {sum(linha.get('malformados', 0) for linha in mapa)}")
    L.append(f"- Marcas nos itens do mapa: {dict(sorted(flags.items())) or 'nenhuma'}")
    L.append(f"- Problemas em `cruzar`: {dict(sorted(Counter(p['motivo'] for p in problemas_pares).items())) or 'nenhum'}")
    erros_adj = Counter(a["erro"] for a in adjudicacoes if a.get("erro"))
    L.append(f"- Adjudicações com erro ou pendentes: {dict(sorted(erros_adj.items())) or 'nenhuma'}")
    L.append("")
    invalidos = [
        (linha["arquivo"], item.get("linha"), tipo, item["trecho"])
        for linha in mapa
        for tipo, itens in (("relação", linha.get("relacoes", [])), ("afirmação", linha.get("afirmacoes", [])))
        for item in itens
        if not item.get("trecho_valido")
    ]
    L.append(f"### Trechos inválidos ({len(invalidos)})")
    L.append("")
    if not invalidos:
        L.append("- nenhum")
    for arquivo, linha_modelo, tipo, trecho in invalidos[:LIMITE_LISTA]:
        L.append(f"- {tipo} em `{arquivo}` (linha dada pelo modelo: {linha_modelo}): \"{_curto(trecho)}\"")
    if len(invalidos) > LIMITE_LISTA:
        L.append(f"- ... e mais {len(invalidos) - LIMITE_LISTA} em `mapa.jsonl`")
    L.append("")
    return "\n".join(L).rstrip() + "\n"


def gerar_relatorio(
    out: Path, ancoras: list[dict], min_ocorrencias: int = 3, arquivos_mapas: tuple[str, ...] = (MAPA_NOME,),
) -> dict:
    """Com mais de um mapa em ``arquivos_mapas`` o relatório usa a união e ganha a seção de concordância."""
    out = Path(out)
    mapa, leitores, erro = carregar_mapas(out, list(arquivos_mapas))
    if erro is not None:
        raise ValueError(erro)
    texto = construir_relatorio(
        ler_jsonl(out / FATIAS_NOME), mapa, ler_jsonl(out / PARES_NOME), ler_jsonl(out / PARES_PROBLEMAS_NOME),
        ler_jsonl(out / ADJUDICACOES_NOME), ancoras, uso_por_etapa(out), min_ocorrencias,
        ler_jsonl(out / PINOS_NOME), read_json(out / PINOS_META_NOME) or {}, uso_por_chave(out),
        leitores=leitores,
    )
    matriz = montar_matriz(mapa, ancoras)
    write_text_atomic(out / RELATORIO_NOME, texto)
    write_text_atomic(out / MATRIZ_NOME, matriz_csv(matriz))
    return {"relatorio": out / RELATORIO_NOME, "matriz": out / MATRIZ_NOME, "linhas_matriz": len(matriz)}
