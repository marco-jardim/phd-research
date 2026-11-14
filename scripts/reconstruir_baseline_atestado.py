"""Reconstrói a linha de base da atestação (``piloto/tese/avaliacoes.jsonl``) a partir do ledger versionado.

Contexto (``.local/reatestamento_2026-11-10/DIAGNOSTICO.md``): em 2026-11-10 um ``avaliar --escopo tese
--chaves ...`` regravou ``avaliacoes.jsonl`` só com o subconjunto filtrado, e o ``avaliar`` completo seguinte
não reaproveitou nada (76 vereditos mudaram). O ledger versionado (``docs/atestado/ledger.csv``) é a
autoridade: para cada linha dele, este script procura a avaliação que a produziu, sem chamar API nenhuma.

Candidatas, nesta ordem de prioridade:

1. ``piloto/tese/avaliacoes.jsonl`` atual e a cópia dele em ``--backup-dir`` (se já houver);
2. ``piloto/tese/avaliacoes.pre_e1_lucena.jsonl``, ``avaliacoes.pre_maia_202611080438.jsonl`` e
   ``avaliacoes.pre_cap3.jsonl``;
3. ``avaliacoes*.jsonl`` em ``piloto/tese/backup_pre_reavaliacao_20261108_0034/``;
4. ``piloto/<capítulo>/avaliacoes.jsonl``;
5. respostas guardadas em ``checkpoint.json`` (da pasta da tese e do backup de 2026-11-08), remontadas com as
   funções do pipeline (``verificar.make_assessment``, ``avaliar.assemble_final``, que aplica
   ``combine_passes``, e ``adjudicar.apply_adjudication``). O checkpoint não guarda o modelo nem a hora da
   adjudicação: o modelo vem de ``metadados.json`` (``adjudicacao.modelo``) e a data, do ``enviado_em`` do lote.

Contextos cuja chave está hoje sem evidência (nível ``nenhum``) são montados primeiro como o ``avaliar`` os
monta (``NAO_VERIFICAVEL``, sem API). Linhas ``PROPRIO`` não passam por ``avaliacoes.jsonl``: o
``exportar-ledger`` as tira de ``artefatos.json``; aqui só se confere que estão lá.

Casamento: o id da candidata é o ``id`` ou o ``id_legado`` da linha do ledger (ou do contexto atual), a chave é a
mesma e ``ledger.hash_avaliacao(modelo, veredito, trecho)`` é igual ao do ledger. O modelo e a data das linhas
não adjudicadas vêm do ``metadados.json`` na exportação; aqui são os mais frequentes dessas linhas no próprio
ledger. Entre as candidatas que casam pelo hash vence a que reproduz também nível, adjudicação, modelo e data;
depois, a de maior prioridade. O checkpoint só é consultado quando os arquivos não dão casamento exato.

Saída: cada linha casada é gravada com ``avaliar.reused_row`` (origem ``baseline_ledger_main_20261110``), com a
impressão digital da evidência e o ``fonte_sha256`` atuais, para que o próximo ``avaliar`` a reaproveite; o campo
``reconstruido_de`` registra a candidata usada. A ordem é a de ``citacoes.jsonl``. A evidência atual é montada
pelo mesmo construtor do ``avaliar`` (só leitura do cache em ``evidencias/``; o índice de trechos
``*.chunks.json`` é regravado se estiver desatualizado, como no ``avaliar``).

O arquivo padrão de saída é novo (``.local/reatestamento_2026-11-10/avaliacoes_reconstruido.jsonl``).
``piloto/tese/avaliacoes.jsonl`` só é regravado com ``--aplicar``, e antes disso ``avaliacoes.jsonl``,
``checkpoint.json`` e ``metadados.json`` da pasta são copiados para ``--backup-dir`` (sem sobrescrever cópias
já feitas). ``--aplicar`` recusa reconstrução incompleta, salvo com ``--aceitar-parcial``.

Uso::

    python scripts/reconstruir_baseline_atestado.py
    python scripts/reconstruir_baseline_atestado.py --aplicar

Código de saída: 0 se toda linha não PROPRIO do ledger foi reproduzida (hash e demais campos); 2 se alguma
ficou sem correspondência ou casou só pelo hash (o arquivo parcial é gravado mesmo assim).

Conferência sugerida depois de ``--aplicar``: ``scripts/atestar_citacoes.py exportar-ledger --escopo tese
--saida <pasta separada>`` e diff vazio contra ``docs/atestado/ledger.csv``.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.atestado import adjudicar as adj
from scripts.atestado import avaliar as av
from scripts.atestado import ledger as led
from scripts.atestado.common import (
    DEFAULT_ADJ_MODEL,
    DEFAULT_BIB,
    DEFAULT_OUT,
    ESCOPO_TESE,
    read_json,
    read_jsonl,
    resolve_path,
    write_jsonl_atomic,
)
from scripts.atestado.evidencias import cache_path, classify_key, load_bib_refs
from scripts.atestado.prompts import build_evidence_block_info
from scripts.atestado.trechos import load_english_queries
from scripts.atestado.verificar import DIVERGENTE, make_assessment, needs_second_pass

ORIGEM = "baseline_ledger_main_20261110"
REATESTAMENTO = Path(".local") / "reatestamento_2026-11-10"
SAIDA_PADRAO = REATESTAMENTO / "avaliacoes_reconstruido.jsonl"
BACKUP_PADRAO = REATESTAMENTO / "acidental"
LEDGER_PADRAO = Path("docs") / "atestado" / "ledger.csv"
SNAPSHOTS_TESE = (
    "avaliacoes.pre_e1_lucena.jsonl",
    "avaliacoes.pre_maia_202611080438.jsonl",
    "avaliacoes.pre_cap3.jsonl",
)
BACKUP_REAVALIACAO = "backup_pre_reavaliacao_20261108_0034"
ARQUIVOS_BACKUP = ("avaliacoes.jsonl", "checkpoint.json", "metadados.json")
CAMPOS_SECUNDARIOS = ("veredito_final", "nivel_evidencia", "adjudicado", "modelo", "data")
FONTE_PIPELINE = "pipeline_sem_evidencia"


# ---------------------------------------------------------------------------
# Estruturas
# ---------------------------------------------------------------------------
@dataclass
class Evidencia:
    """Evidência atual de uma chave, montada como no ``avaliar``."""

    texto: str
    nivel: str
    fingerprint: str
    trechos: list[dict]
    fonte: str
    classe: str


@dataclass
class Candidata:
    fonte: str
    linha: dict
    # impressão digital da evidência sobre a qual a candidata foi avaliada (None se desconhecida)
    evidencia_avaliada: str | None


@dataclass
class Casamento:
    candidata: Candidata
    divergencias: list[str]  # campos do ledger, além do hash, que a candidata não reproduz


@dataclass
class Resultado:
    linhas: list[dict] = field(default_factory=list)
    por_fonte: Counter = field(default_factory=Counter)
    divergentes: list[tuple[dict, Casamento]] = field(default_factory=list)
    sem_correspondencia: list[tuple[dict, str, list[str]]] = field(default_factory=list)
    # (linha, candidata, fonte_sha256 da candidata igual ao atual)
    evidencia_diferente: list[tuple[dict, Candidata, bool]] = field(default_factory=list)
    local_divergente: list[tuple[dict, list[str]]] = field(default_factory=list)
    proprias: int = 0
    proprias_ausentes: list[dict] = field(default_factory=list)
    fora_do_ledger: int = 0
    meta_base: dict = field(default_factory=dict)
    meta_atual: dict = field(default_factory=dict)
    meta_variantes: dict[str, list[str]] = field(default_factory=dict)

    @property
    def completo(self) -> bool:
        return not self.sem_correspondencia and not self.divergentes


# ---------------------------------------------------------------------------
# Caminhos e rótulos
# ---------------------------------------------------------------------------
def rotulo(path: Path, out_dir: Path) -> str:
    """Caminho legível da fonte: relativo a ``out_dir`` ou à raiz do repositório quando possível."""
    resolved = path.resolve()
    for base in (out_dir.resolve(), ROOT):
        try:
            return resolved.relative_to(base).as_posix()
        except ValueError:
            continue
    return resolved.as_posix()


def _unicos_existentes(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        out.append(path)
    return out


def arquivos_candidatos(pilot: Path, out_dir: Path, backup_dir: Path) -> list[Path]:
    """``avaliacoes*.jsonl`` candidatos, em ordem de prioridade."""
    paths = [pilot / "avaliacoes.jsonl", backup_dir / "avaliacoes.jsonl"]
    paths += [pilot / name for name in SNAPSHOTS_TESE]
    backup = pilot / BACKUP_REAVALIACAO
    if backup.is_dir():
        paths += sorted(backup.rglob("avaliacoes*.jsonl"))
    piloto = out_dir / "piloto"
    if piloto.is_dir():
        paths += sorted(
            p / "avaliacoes.jsonl" for p in piloto.iterdir() if p.is_dir() and p.resolve() != pilot.resolve()
        )
    return _unicos_existentes(paths)


def checkpoints_candidatos(pilot: Path) -> list[Path]:
    paths = [pilot / "checkpoint.json"]
    backup = pilot / BACKUP_REAVALIACAO
    if backup.is_dir():
        paths += sorted(backup.rglob("checkpoint*.json"))
    return _unicos_existentes(paths)


# ---------------------------------------------------------------------------
# Evidência atual e metadados
# ---------------------------------------------------------------------------
def evidencias_atuais(
    chaves: set[str], contexts: list[dict], out_dir: Path, max_chars: int, refs: dict
) -> dict[str, Evidencia]:
    """Evidência por chave como o ``avaliar`` a monta (todos os contextos da chave, na ordem de ``citacoes.jsonl``)."""
    queries_en = load_english_queries(out_dir)
    by_key = av.group_by_key([c for c in contexts if c["key"] in chaves])
    result: dict[str, Evidencia] = {}
    for key in sorted(by_key):
        record = read_json(cache_path(out_dir, key)) or {"key": key}
        text, level, info = build_evidence_block_info(record, max_chars, by_key[key], out_dir, queries_en)
        result[key] = Evidencia(
            texto=text,
            nivel=level,
            fingerprint=av.evidence_fingerprint(text),
            trechos=info["trechos"],
            fonte=av.source_sha256(record, max_chars),
            classe=classify_key(refs.get(key)),
        )
    return result


def _mais_frequente(values: list[str]) -> str:
    return Counter(values).most_common(1)[0][0] if values else ""


def meta_do_ledger(ledger: list[dict], meta_atual: dict) -> tuple[dict, dict[str, list[str]]]:
    """Modelo e data que a exportação atribuiu às linhas não adjudicadas (vêm de ``metadados.json``).

    Devolve também os valores distintos encontrados (uma exportação só produz um de cada)."""
    base = [r for r in ledger if r["veredito_final"] != led.PROPRIO and r.get("adjudicado") != "sim"]
    modelos = [r.get("modelo", "") for r in base]
    datas = [r.get("data", "") for r in base]
    meta = {
        "modelo": _mais_frequente(modelos) or meta_atual.get("modelo", ""),
        "data_utc": _mais_frequente(datas) or (meta_atual.get("data_utc") or "")[:10],
    }
    return meta, {"modelo": sorted(set(modelos)), "data": sorted(set(datas))}


# ---------------------------------------------------------------------------
# Candidatas
# ---------------------------------------------------------------------------
def indexar_arquivos(paths: list[Path], out_dir: Path) -> dict[str, list[tuple[int, Candidata]]]:
    """Linhas com veredito final por id, com a prioridade do arquivo de origem."""
    index: dict[str, list[tuple[int, Candidata]]] = {}
    for prioridade, path in enumerate(paths):
        fonte = rotulo(path, out_dir)
        for row in read_jsonl(path):
            if not row.get("id") or not av.is_final_row(row):
                continue
            avaliada = row.get("evidencia_fingerprint_avaliada") or row.get("evidencia_fingerprint")
            index.setdefault(row["id"], []).append((prioridade, Candidata(fonte, row, avaliada)))
    return index


def ids_procurados(linha: dict, ctx: dict) -> list[str]:
    ids: list[str] = []
    for value in (linha["id"], linha.get("id_legado"), ctx.get("id_legado")):
        if value and value not in ids:
            ids.append(value)
    return ids


def candidatas_de_arquivo(index: dict[str, list[tuple[int, Candidata]]], ids: list[str]) -> list[Candidata]:
    """Por prioridade do arquivo; no mesmo arquivo, o id atual antes do legado (como no exportador)."""
    found = [(prio, ids.index(i), cand) for i in ids for prio, cand in index.get(i, [])]
    return [cand for _prio, _pos, cand in sorted(found, key=lambda t: (t[0], t[1]))]


def _passada(
    requisicoes: dict, passada: int, ctx_id: str, key: str, evidencia: str, modelo: str
) -> list[tuple[dict, bool]]:
    """Avaliações de uma passada que incluem o contexto, como em ``avaliar.collect_assessments``.

    O booleano diz se o ``custom_id`` se reproduz com a evidência atual (mesma evidência enviada)."""
    out: list[tuple[dict, bool]] = []
    for cid, entry in requisicoes.items():
        if entry.get("passada") != passada or entry.get("key") != key or ctx_id not in entry.get("ids", []):
            continue
        itens = {str(item.get("id")): item for item in entry.get("itens") or []}
        assessment = make_assessment(itens.get(ctx_id), evidencia)
        if entry.get("erro"):
            assessment["flags"].append(entry["erro"].split(":")[0])
        mesma = cid == av.make_custom_id(passada, key, entry["ids"], modelo, evidencia)
        out.append((assessment, mesma))
    return out


def _adjudicacoes(state: dict, ctx_id: str, key: str, evidencia: str) -> list[tuple[dict, dict, str | None, str | None]]:
    """(resposta, avaliação, id do lote, enviado_em) das adjudicações do contexto, sem as respostas com erro."""
    secao = state.get("adjudicacao") or {}
    lotes = SimpleNamespace(batches=secao.get("lotes") or [])
    enviado = {cid: b.get("enviado_em") for b in lotes.batches for cid in b.get("custom_ids", [])}
    out = []
    for cid, entry in (secao.get("requisicoes") or {}).items():
        if entry.get("erro") or entry.get("key") != key or ctx_id not in entry.get("ids", []):
            continue
        raw = {str(item.get("id")): item for item in entry.get("itens") or []}.get(ctx_id)
        if raw is None:
            continue
        batch_id = adj.batch_id_of(lotes, cid) if entry.get("modo") == "batch" else None
        out.append((raw, make_assessment(raw, evidencia), batch_id, enviado.get(cid)))
    return out


def candidatas_de_checkpoint(
    state: dict, fonte: str, ctx: dict, ev: Evidencia, modelo_avaliacao: str, modelo_adjudicacao: str
) -> list[Candidata]:
    """Linhas finais remontadas das respostas do checkpoint (passada 1, passada 2 e adjudicação)."""
    requisicoes = state.get("requisicoes") or {}
    key, ctx_id = ctx["key"], ctx["id"]
    segundas = _passada(requisicoes, 2, ctx_id, key, ev.texto, modelo_avaliacao)
    adjudicacoes: list | None = None
    out: list[Candidata] = []
    for first, mesma1 in _passada(requisicoes, 1, ctx_id, key, ev.texto, modelo_avaliacao):
        opcoes = segundas if needs_second_pass(first) else [(None, True)]
        for second, mesma2 in opcoes:
            mesma = mesma1 and mesma2
            # o checkpoint não guarda a evidência: impressão digital e fonte só valem se o custom_id se reproduz
            base = av.assemble_final(
                ctx, ev.classe, ev.nivel, first, second, ev.texto, ev.fingerprint if mesma else None,
                ev.trechos, ev.fonte if mesma else None,
            )
            if not av.is_final_row(base):
                continue
            avaliada = ev.fingerprint if mesma else None
            if base["veredito_final"] != DIVERGENTE:
                out.append(Candidata(fonte, base, avaliada))
                continue
            if adjudicacoes is None:
                adjudicacoes = _adjudicacoes(state, ctx_id, key, ev.texto)
            for raw, assessment, batch_id, em in adjudicacoes:
                row = copy.deepcopy(base)
                adj.apply_adjudication(row, assessment, raw, modelo_adjudicacao, batch_id, em)
                out.append(Candidata(fonte, row, avaliada))
    return out


def linha_pipeline_sem_evidencia(ctx: dict, ev: Evidencia) -> Candidata:
    """A linha que o ``avaliar`` grava para contexto sem evidência (sem API)."""
    row = av.assemble_final(ctx, ev.classe, ev.nivel, None, None, ev.texto, ev.fingerprint, ev.trechos, ev.fonte)
    return Candidata(FONTE_PIPELINE, row, ev.fingerprint)


# ---------------------------------------------------------------------------
# Casamento
# ---------------------------------------------------------------------------
def escolher(linha: dict, candidatas: list[Candidata], meta: dict) -> Casamento | None:
    """Primeira candidata que reproduz a linha inteira; senão, a primeira que reproduz o hash."""
    parcial: Casamento | None = None
    for cand in candidatas:
        if cand.linha.get("key") != linha["chave"]:
            continue
        projecao = led._row_from_evaluation({**cand.linha, "id": linha["id"]}, meta)
        if projecao["hash_avaliacao"] != linha["hash_avaliacao"]:
            continue
        divergencias = [c for c in CAMPOS_SECUNDARIOS if (projecao.get(c) or "") != (linha.get(c) or "")]
        if not divergencias:
            return Casamento(cand, [])
        if parcial is None:
            parcial = Casamento(cand, divergencias)
    return parcial


def linha_de_saida(ctx: dict, cand: Candidata, ev: Evidencia, origem: str) -> dict:
    """Contexto sem evidência: a linha do pipeline; demais: ``reused_row`` com a evidência atual.

    ``reconstruido_de`` guarda a candidata usada (preservada se a candidata já era uma linha reconstruída)."""
    if cand.fonte == FONTE_PIPELINE:
        row = dict(cand.linha)
    else:
        row = av.reused_row(ctx, cand.linha, origem, ev.texto, ev.fingerprint, ev.trechos, ev.fonte)
    row["reconstruido_de"] = cand.linha.get("reconstruido_de") or cand.fonte
    return row


def reconstruir(
    ledger_path: Path, out_dir: Path, backup_dir: Path, max_chars: int, bib: Path | None, origem: str = ORIGEM
) -> Resultado:
    res = Resultado()
    ledger = led.load_ledger(ledger_path)
    pilot = out_dir / "piloto" / ESCOPO_TESE
    contexts = read_jsonl(out_dir / "citacoes.jsonl")
    if not contexts:
        raise FileNotFoundError(f"{out_dir / 'citacoes.jsonl'} ausente ou vazio")
    by_id = {c["id"]: c for c in contexts}
    res.meta_atual = read_json(pilot / "metadados.json", default={}) or {}
    res.meta_base, res.meta_variantes = meta_do_ledger(ledger, res.meta_atual)
    modelo_adj = (res.meta_atual.get("adjudicacao") or {}).get("modelo") or DEFAULT_ADJ_MODEL

    avaliadas = [r for r in ledger if r["veredito_final"] != led.PROPRIO]
    proprias = [r for r in ledger if r["veredito_final"] == led.PROPRIO]
    res.proprias = len(proprias)
    artefatos = {a.get("id") for a in read_json(pilot / "artefatos.json", default=[]) or []}
    res.proprias_ausentes = [r for r in proprias if r["id"] not in artefatos and r.get("id_legado") not in artefatos]
    ledger_ids = {r["id"] for r in ledger}
    res.fora_do_ledger = sum(1 for c in contexts if c["id"] not in ledger_ids)

    refs = load_bib_refs(bib) if bib is not None and bib.is_file() else {}
    chaves = {r["chave"] for r in avaliadas}
    evidencias = evidencias_atuais(chaves, contexts, out_dir, max_chars, refs)
    index = indexar_arquivos(arquivos_candidatos(pilot, out_dir, backup_dir), out_dir)
    checkpoints = [(f"checkpoint:{rotulo(p, out_dir)}", read_json(p) or {}) for p in checkpoints_candidatos(pilot)]

    escolhidas: dict[str, dict] = {}
    for linha in avaliadas:
        ctx = by_id.get(linha["id"])
        if ctx is None:
            res.sem_correspondencia.append((linha, "contexto ausente de citacoes.jsonl", []))
            continue
        if ctx["key"] != linha["chave"]:
            res.sem_correspondencia.append((linha, f"chave do contexto atual é {ctx['key']}", []))
            continue
        local = [
            f"{campo}: ledger={linha.get(campo, '')!r} atual={ctx.get(chave_ctx, '')!r}"
            for campo, chave_ctx in (("arquivo", "file"), ("capitulo", "chapter"), ("id_legado", "id_legado"))
            if (linha.get(campo) or "") != (ctx.get(chave_ctx) or "")
        ]
        if local:
            res.local_divergente.append((linha, local))
        ev = evidencias[linha["chave"]]
        candidatas = [linha_pipeline_sem_evidencia(ctx, ev)] if ev.nivel == "nenhum" else []
        candidatas += candidatas_de_arquivo(index, ids_procurados(linha, ctx))
        casamento = escolher(linha, candidatas, res.meta_base)
        if casamento is None or casamento.divergencias:
            do_checkpoint = [
                cand
                for fonte, state in checkpoints
                for cand in candidatas_de_checkpoint(state, fonte, ctx, ev, res.meta_base["modelo"], modelo_adj)
            ]
            outro = escolher(linha, do_checkpoint, res.meta_base)
            if outro is not None and (casamento is None or not outro.divergencias):
                casamento = outro
            candidatas += do_checkpoint
        if casamento is None:
            vistos = sorted({
                str(c.linha.get("veredito_final")) for c in candidatas if c.linha.get("key") == linha["chave"]
            })
            motivo = "nenhuma candidata com o mesmo hash_avaliacao" if vistos else "nenhuma candidata pelo id"
            res.sem_correspondencia.append((linha, motivo, vistos))
            continue
        cand = casamento.candidata
        res.por_fonte[cand.fonte] += 1
        if casamento.divergencias:
            res.divergentes.append((linha, casamento))
        if cand.evidencia_avaliada != ev.fingerprint:
            res.evidencia_diferente.append((linha, cand, cand.linha.get("fonte_sha256") == ev.fonte))
        escolhidas[ctx["id"]] = linha_de_saida(ctx, cand, ev, origem)
    res.linhas = [escolhidas[c["id"]] for c in contexts if c["id"] in escolhidas]
    return res


# ---------------------------------------------------------------------------
# Aplicação e relatório
# ---------------------------------------------------------------------------
def fazer_backup(pilot: Path, backup_dir: Path) -> tuple[list[str], list[str]]:
    """Copia avaliacoes/checkpoint/metadados da pasta da tese para ``backup_dir``, sem sobrescrever cópias."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    copiados: list[str] = []
    mantidos: list[str] = []
    for name in ARQUIVOS_BACKUP:
        src, dst = pilot / name, backup_dir / name
        if dst.exists():
            mantidos.append(name)
        elif src.is_file():
            shutil.copy2(src, dst)
            copiados.append(name)
    return copiados, mantidos


def _descreve(linha: dict) -> str:
    return f"{linha['id']}  {linha['chave']}  {linha['arquivo']}  ledger={linha['veredito_final']}"


def imprimir(res: Resultado, ledger_path: Path, out_dir: Path) -> None:
    avaliadas = sum(res.por_fonte.values()) + len(res.sem_correspondencia)
    print(f"ledger: {rotulo(ledger_path, out_dir)} ({avaliadas + res.proprias} linhas; "
          f"{avaliadas} avaliadas, {res.proprias} PROPRIO)")
    base, atual = res.meta_base, res.meta_atual
    print(f"modelo/data das linhas não adjudicadas no ledger: {base['modelo']} / {base['data_utc']}")
    for campo, valores in res.meta_variantes.items():
        if len(valores) > 1:
            print(f"AVISO: o ledger tem mais de um {campo} nas linhas não adjudicadas: {', '.join(valores)}")
    atual_modelo, atual_data = atual.get("modelo", ""), (atual.get("data_utc") or "")[:10]
    if (atual_modelo, atual_data) != (base["modelo"], base["data_utc"]):
        print(f"AVISO: metadados.json atual tem {atual_modelo or '(vazio)'} / {atual_data or '(vazio)'}; o "
              "exportar-ledger usa esses valores nas linhas não adjudicadas e o diff contra o ledger não sairá vazio")

    print(f"casadas: {sum(res.por_fonte.values())}")
    for fonte, n in sorted(res.por_fonte.items(), key=lambda t: (-t[1], t[0])):
        print(f"  {fonte}: {n}")
    if res.evidencia_diferente:
        mesma_fonte = sum(1 for _linha, _cand, igual in res.evidencia_diferente if igual)
        print(f"avaliadas sobre evidência diferente da atual ou desconhecida: {len(res.evidencia_diferente)}, "
              f"{mesma_fonte} com fonte_sha256 igual ao atual (gravadas com a impressão digital atual; a "
              "anterior, quando conhecida, fica em evidencia_fingerprint_avaliada)")
        for linha, cand, igual in res.evidencia_diferente:
            print(f"  {_descreve(linha)}  fonte={cand.fonte}  mesma_fonte={'sim' if igual else 'nao'}")

    if res.divergentes:
        print(f"casadas só pelo hash, com outro campo diferente do ledger: {len(res.divergentes)}")
        for linha, casamento in res.divergentes:
            proj = led._row_from_evaluation({**casamento.candidata.linha, "id": linha["id"]}, base)
            campos = ", ".join(f"{c}: ledger={linha.get(c, '')!r} reconstruida={proj.get(c, '')!r}"
                               for c in casamento.divergencias)
            print(f"  {_descreve(linha)}  fonte={casamento.candidata.fonte}  {campos}")

    print(f"sem correspondência: {len(res.sem_correspondencia)}")
    for linha, motivo, vistos in res.sem_correspondencia:
        extra = f"  vereditos das candidatas: {', '.join(vistos)}" if vistos else ""
        print(f"  {_descreve(linha)}  motivo: {motivo}{extra}")

    print(f"PROPRIO: {res.proprias} (o exportar-ledger as tira de artefatos.json; não entram em avaliacoes.jsonl)")
    if res.proprias_ausentes:
        print(f"AVISO: PROPRIO ausentes do artefatos.json atual: {len(res.proprias_ausentes)}")
        for linha in res.proprias_ausentes:
            print(f"  {_descreve(linha)}")
    if res.local_divergente:
        print(f"AVISO: localização do contexto atual difere do ledger em {len(res.local_divergente)} linha(s) "
              "(o exportar-ledger usa a atual)")
        for linha, campos in res.local_divergente:
            print(f"  {linha['id']}  {linha['chave']}  " + "; ".join(campos))
    if res.fora_do_ledger:
        print(f"contextos de citacoes.jsonl fora do ledger: {res.fora_do_ledger} "
              "(não gravados; o próximo avaliar os avalia)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstrói piloto/tese/avaliacoes.jsonl a partir do ledger versionado, sem chamar API."
    )
    parser.add_argument("--ledger", default=str(LEDGER_PADRAO), help="ledger versionado (padrão docs/atestado/ledger.csv)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="diretório de trabalho (padrão .local/biblio)")
    parser.add_argument("--saida", default=str(SAIDA_PADRAO),
                        help="arquivo reconstruído (padrão .local/reatestamento_2026-11-10/avaliacoes_reconstruido.jsonl)")
    parser.add_argument("--backup-dir", default=str(BACKUP_PADRAO), dest="backup_dir",
                        help="cópia dos arquivos acidentais antes de --aplicar (padrão "
                             ".local/reatestamento_2026-11-10/acidental)")
    parser.add_argument("--max-chars", type=int, default=120000, dest="max_chars",
                        help="o mesmo valor usado em `avaliar` (a evidência montada tem de ser a mesma)")
    parser.add_argument("--bib", default=DEFAULT_BIB, help="só para a classe (literatura/cinzenta) de linhas sem evidência")
    parser.add_argument("--origem", default=ORIGEM, help=f"marca reaproveitado_de (padrão {ORIGEM})")
    parser.add_argument("--aplicar", action="store_true",
                        help="regrava piloto/tese/avaliacoes.jsonl (antes copia avaliacoes, checkpoint e metadados "
                             "para --backup-dir, se a cópia ainda não existir)")
    parser.add_argument("--aceitar-parcial", action="store_true", dest="aceitar_parcial",
                        help="com --aplicar, regrava mesmo com linhas sem correspondência ou casadas só pelo hash")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = resolve_path(args.out)
    pilot = out_dir / "piloto" / ESCOPO_TESE
    alvo = pilot / "avaliacoes.jsonl"
    saida = resolve_path(args.saida)
    backup_dir = resolve_path(args.backup_dir)
    ledger_path = resolve_path(args.ledger)
    if saida.resolve() == alvo.resolve():
        parser.error(f"--saida não pode ser {alvo}; para regravar a avaliação da tese use --aplicar")
    if args.aceitar_parcial and not args.aplicar:
        parser.error("--aceitar-parcial só vale com --aplicar")

    res = reconstruir(ledger_path, out_dir, backup_dir, args.max_chars, resolve_path(args.bib), args.origem)
    imprimir(res, ledger_path, out_dir)

    aplicar = args.aplicar
    if aplicar and not res.completo and not args.aceitar_parcial:
        print(f"--aplicar recusado: reconstrução incompleta; {alvo} não foi alterado "
              "(confira a lista acima ou use --aceitar-parcial)", file=sys.stderr)
        aplicar = False
    if aplicar:
        copiados, mantidos = fazer_backup(pilot, backup_dir)
        print(f"backup em {backup_dir}: copiados {', '.join(copiados) or 'nenhum'}; "
              f"já existentes (mantidos) {', '.join(mantidos) or 'nenhum'}")
    write_jsonl_atomic(saida, res.linhas)
    print(f"saída: {saida} ({len(res.linhas)} linhas)")
    if aplicar:
        write_jsonl_atomic(alvo, res.linhas)
        print(f"aplicado: {alvo} ({len(res.linhas)} linhas)")
    return 0 if res.completo else 2


if __name__ == "__main__":
    sys.exit(main())
