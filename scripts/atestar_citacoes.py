"""Atestação bibliográfica da tese (etapas 2 a 4), com piloto por capítulo.

Sequência de execução (a etapa 1 é ``scripts/citacoes_contexto.py``, que gera
``.local/biblio/citacoes.jsonl``). Caminhos relativos valem a partir da raiz do repositório.

    python scripts/citacoes_contexto.py
    python scripts/atestar_citacoes.py evidencias --capitulo "Introdução"
    python scripts/atestar_citacoes.py avaliar    --capitulo "Introdução"            # lote (padrão)
    python scripts/atestar_citacoes.py avaliar    --capitulo "Introdução" --modo sync --chaves chaveA,chaveB
    python scripts/atestar_citacoes.py relatorio  --capitulo "Introdução" --precos precos.yaml

1. ``evidencias``: usa antes de tudo os textos do autor em ``--fontes-locais`` (padrão ``.local/fontes``,
   ``<chave>.txt|pdf|xml|html``; ``<chave>_preprint.*`` só como reserva, sinalizado); sem texto local,
   recupera resumo e texto completo só de fontes abertas (OpenAlex, Europe PMC,
   SciELO, Unpaywall, arXiv), com checagem de identidade (DOI exato ou título >= 0,85) antes de
   aceitar qualquer texto; obras próprias recebem conferência de artefatos (``\\path{...}``).
   Cache por chave em ``.local/biblio/evidencias/<chave>.json`` (``--refresh`` refaz). Citações do
   Consensus em ``.local/biblio/consensus/<chave>.json`` entram como evidência extra. Variáveis:
   ``UNPAYWALL_EMAIL`` ou ``CROSSREF_MAILTO`` (e-mail de contato, usado no User-Agent e no Unpaywall).
   Escopo global: em vez de ``--capitulo``, ``--escopo tese`` seleciona todos os contextos da tese,
   grava em ``.local/biblio/piloto/tese/`` e planeja uma requisição por chave atravessando os capítulos
   (várias só acima de ``--max-contextos``, compartilhando o bloco de evidência em cache). Vereditos
   finais de execuções por capítulo (``piloto/*/avaliacoes.jsonl``) são copiados quando o contexto
   (chave + sentença normalizada) e a evidência enviada (fingerprint) não mudaram
   (``reaproveitado_de``); ``--sem-reaproveitar`` desliga. ``temperature`` só é enviada com ``--temperature``.
2. ``avaliar``: uma requisição por (chave, capítulo) ao Claude (padrão Sonnet 5.5) com cache de
   prompt; ``--modo batch`` (padrão) usa a Message Batches API e é retomável (o id do lote fica em
   ``checkpoint.json``; rode de novo para continuar a espera/baixar); ``--modo sync`` serve para
   depurar. Verificação mecânica dos trechos, segunda passada cega e veredito final. Chave da API:
   ``ANTHROPIC_API_KEY`` (ambiente ou ``.env``), nunca impressa nem gravada.
2b. ``adjudicar``: envia só os contextos com veredito final DIVERGENTE ao Claude Opus (padrão
   ``claude-opus-5-5``), uma requisição por chave, com a rubrica de adjudicação em cache, a mesma
   evidência das passadas de avaliação (use o mesmo ``--max-chars``) e as duas avaliações anteriores.
   Atualiza ``avaliacoes.jsonl`` no lugar (campo ``adjudicacao``; ``passada1``/``passada2`` ficam) e é
   retomável: os ids de lote ficam na seção ``adjudicacao`` de ``checkpoint.json`` e uma nova execução
   consulta o lote em vez de reenviá-lo (``--refazer-falhas`` reenvia o que terminou com erro).
   Nunca envia ``temperature``. Uma nova execução de ``avaliar`` regrava ``avaliacoes.jsonl`` e
   descarta a adjudicação gravada nele; rode ``adjudicar`` depois da avaliação final.
2c. Fontes com texto completo maior que ``--max-chars`` (padrão 120.000) não são mais cortadas no início:
   ``avaliar`` e ``adjudicar`` enviam o início do documento (3.000 caracteres) mais os trechos de ~1.500
   caracteres mais relevantes para os contextos da chave (BM25 sobre consulta com sentença, parágrafo,
   números, siglas, nomes próprios e termos técnicos), marcados como ``[trecho 12, p. 184]`` (``p.`` é a
   página do PDF, gravada em ``fulltext_paginas`` por ``evidencias`` para PDFs locais; refaça
   ``evidencias`` para registros antigos). Os trechos enviados ficam em ``trechos_enviados`` em cada
   linha de ``avaliacoes.jsonl``; a impressão digital da evidência muda com a seleção. O índice de trechos
   fica em ``evidencias/<chave>.chunks.json``. ``consultas`` (rode antes de ``avaliar``, com o mesmo
   ``--max-chars``) pede ao Claude Haiku, em lote retomável, a tradução da sentença e palavras-chave em
   inglês de cada contexto dessas fontes e as guarda em ``evidencias/_consultas_en.json``; sem elas a
   seleção usa só a consulta em português. Mudar as consultas muda a seleção, então os vereditos são
   refeitos.
3. ``relatorio``: ``relatorio.md`` com contagens, tabela por contexto, listas de revisão humana,
   uso de tokens e, com ``--precos`` (YAML: input, output, cache_write, cache_read em USD por milhão
   de tokens; batch_desconto como fração, ex. 0.5), custo estimado.

4. ``exportar-ledger --escopo tese``: grava o ledger versionado ``docs/atestado/ledger.csv`` (UTF-8, ``;``;
   colunas id, id_legado, chave, arquivo, capitulo, veredito_final, nivel_evidencia, adjudicado, modelo, data,
   hash_avaliacao; sem citações nem justificativas, que ficam em ``.local/``) a partir de
   ``piloto/tese/avaliacoes.jsonl`` e ``artefatos.json`` (obras próprias entram como ``PROPRIO``). Avaliações
   com o id antigo (chave + sentença) são mapeadas ao id novo de ``citacoes.jsonl`` pelo campo ``id_legado``.
   ``hash_avaliacao`` é o sha256 (16 primeiros hexadecimais) de modelo, veredito final e trecho da fonte
   (o trecho não é gravado). Contextos ainda sem veredito final (PENDENTE) não entram no ledger.
   Também reconcilia ``docs/atestado/pendencias.yaml`` (ids com NAO_SUSTENTA ou CONTRADIZ, cada um com
   ``decisao``; decisões e ``origem`` já registradas são preservadas, inclusive ao migrar ids antigos):

   * sem ``--aceitar-novas``, NAO_SUSTENTA/CONTRADIZ que não estão em ``pendencias.yaml`` NÃO são
     acrescentados: o ledger é gravado, os ids novos são listados e o código de saída é 3;
   * com ``--aceitar-novas``, os ids novos entram com ``decisao: pendente`` (código 0), e o teste de
     portão só passa depois que o autor registrar uma decisão diferente de ``pendente``;
   * ``--marcar-baseline`` (migração única) marca as entradas já existentes com
     ``origem: baseline-2026-11-08``, a única forma de ``pendente`` aceita pelo portão. Só vale para os ids
     congelados em ``tests/data/pendencias_baseline_ids.txt``: qualquer outro id (por exemplo, um aceito
     antes com ``--aceitar-novas``) faz o comando recusar a marcação, sem gravar nada, com código 4.

   Avaliações com id antigo só são mapeadas quando o id antigo corresponde a um único id novo; se a mesma
   chave aparece duas vezes na sentença, os dois contextos ficam sem atestado até nova avaliação.

   Códigos de saída: 0 sucesso; 2 sem avaliações; 3 problemas novos sem ``--aceitar-novas``;
   4 ``--marcar-baseline`` recusado.

Como atestar contextos alterados após uma edição do texto (o teste ``tests/test_atestado_ledger.py`` falha
com "contextos sem atestamento" para toda sentença nova ou reescrita; o ``id`` é a chave, a sentença com
``[cit]`` na posição da citação, o comando de citação e os argumentos opcionais, como a página):

    python scripts/citacoes_contexto.py                       # extrai os contextos atuais
    python scripts/atestar_citacoes.py evidencias --escopo tese
    python scripts/atestar_citacoes.py avaliar    --escopo tese   # reaproveita vereditos de contextos inalterados
    python scripts/atestar_citacoes.py adjudicar  --escopo tese   # só os DIVERGENTE
    python scripts/atestar_citacoes.py exportar-ledger --escopo tese
    python -m pytest tests/test_atestado_ledger.py -q

O reaproveitamento do ``avaliar`` (``reaproveitado_de``) mantém o custo baixo: só contextos novos ou com evidência
alterada vão à API. Isso inclui as linhas da própria ``piloto/tese/avaliacoes.jsonl`` gravadas com o id antigo
(casadas por ``id_legado``), que são lidas antes das pastas por capítulo e preservam a adjudicação.
Se o passo ``exportar-ledger`` sair com código 3 (problemas NOVOS), corrija o texto ou
rode de novo com ``--aceitar-novas`` e registre a decisão em ``pendencias.yaml`` antes de versionar o ledger.
Trechos de fonte com menos de 6 palavras e menos de 40 caracteres não sustentam SUSTENTA, SUSTENTA_PARCIALMENTE
nem CONTRADIZ (``trecho_curto``: veredito rebaixado a NAO_VERIFICAVEL, com segunda passada).

Saídas em ``.local/biblio/piloto/<capitulo>/``: ``evidencias_resumo.md``, ``classificacao.json``,
``artefatos.json``, ``checkpoint.json``, ``avaliacoes.jsonl``, ``metadados.json``, ``relatorio.md``.
Textos completos e resumos ficam só em ``.local/`` (fora do git); as saídas trazem apenas trechos curtos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.atestado import adjudicar as adj
from scripts.atestado import avaliar as avaliar_mod
from scripts.atestado import consultas as cons
from scripts.atestado import evidencias as ev
from scripts.atestado import ledger as led
from scripts.atestado import proprios as pr
from scripts.atestado import relatorio as rel
from scripts.atestado import trechos as tr
from scripts.atestado.common import (
    DEFAULT_ADJ_MODEL,
    DEFAULT_BIB,
    DEFAULT_MODEL,
    DEFAULT_OUT,
    DEFAULT_QUERY_MODEL,
    ESCOPO_TESE,
    TESE_LABEL,
    load_env_var,
    read_json,
    read_jsonl,
    resolve_path,
    slug,
    write_json_atomic,
    write_text_atomic,
)
from scripts.atestado.http import HttpClient


def split_csv(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def select_contexts(records: list[dict], capitulo: str | None, chaves: list[str]) -> list[dict]:
    """Contextos do capítulo (título exato); ``capitulo=None`` seleciona a tese inteira
    (texto, apêndices, tabelas e notas). ``chaves`` restringe a essas chaves."""
    selected = list(records) if capitulo is None else [r for r in records if r.get("chapter") == capitulo]
    if chaves:
        wanted = set(chaves)
        selected = [r for r in selected if r["key"] in wanted]
    return selected


def is_global(args: argparse.Namespace) -> bool:
    return getattr(args, "escopo", None) == ESCOPO_TESE


def scope_label(args: argparse.Namespace) -> str:
    return TESE_LABEL if is_global(args) else args.capitulo


def pilot_dir(out_dir: Path, capitulo: str | None) -> Path:
    return out_dir / "piloto" / (ESCOPO_TESE if capitulo is None else slug(capitulo))


def load_contexts(out_dir: Path, capitulo: str | None, chaves: list[str]) -> list[dict]:
    path = out_dir / "citacoes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} não existe; rode scripts/citacoes_contexto.py antes")
    contexts = select_contexts(read_jsonl(path), capitulo, chaves)
    if not contexts:
        raise ValueError("nenhum contexto na tese" if capitulo is None else f"nenhum contexto para o capítulo {capitulo!r}")
    return contexts


def scope_chapter(args: argparse.Namespace) -> str | None:
    """Capítulo escolhido, ou None no escopo global."""
    return None if is_global(args) else args.capitulo


def cmd_evidencias(args: argparse.Namespace) -> int:
    out_dir = resolve_path(args.out)
    contexts = load_contexts(out_dir, scope_chapter(args), split_csv(args.chaves))
    refs = ev.load_bib_refs(resolve_path(args.bib))
    proprios = set(split_csv(args.proprios))
    mailto = load_env_var("CROSSREF_MAILTO") or load_env_var("UNPAYWALL_EMAIL")
    email = load_env_var("UNPAYWALL_EMAIL") or load_env_var("CROSSREF_MAILTO")
    collector = ev.Collector(HttpClient(mailto, min_interval=args.intervalo), unpaywall_email=email,
                             fontes_locais=resolve_path(args.fontes_locais) if args.fontes_locais else None)
    pdir = pilot_dir(out_dir, scope_chapter(args))

    by_key: dict[str, list[dict]] = {}
    for ctx in contexts:
        by_key.setdefault(ctx["key"], []).append(ctx)

    rows: list[dict] = []
    classificacao: dict[str, dict] = {}
    artefatos: list[dict] = []
    tracked = pr.git_tracked_files(ROOT)
    for key in sorted(by_key):
        ref = refs.get(key)
        classe = ev.classify_key(ref, proprios)
        if classe == "proprio":
            checks = pr.check_artifacts(by_key[key], ROOT, tracked)
            artefatos.extend(checks)
            summary = ", ".join(f"{v}={n}" for v, n in sorted(_count(c["veredito"] for c in checks).items()))
            rows.append({"key": key, "classe": classe, "artefato": summary})
            classificacao[key] = {"classe": classe, "nivel": "artefato"}
            continue
        if ref is None:
            rows.append({"key": key, "classe": classe, "identidade_verificada": False, "metodo": "",
                         "erros": ["chave ausente do .bib"], "abstract": "", "fulltext": ""})
            classificacao[key] = {"classe": classe, "nivel": "nenhum"}
            continue
        record = ev.get_evidence(collector, ref, classe, out_dir, refresh=args.refresh)
        rows.append(record)
        classificacao[key] = {"classe": classe, "nivel": ev.evidence_level(record),
                              "fonte": ev.source_label(record),
                              "identidade_verificada": bool(record.get("identidade_verificada"))}
        print(f"{key}: {classe}, {classificacao[key]['nivel']}", flush=True)

    write_json_atomic(pdir / "classificacao.json", classificacao)
    write_json_atomic(pdir / "artefatos.json", artefatos)
    ev.write_resumo(pdir / "evidencias_resumo.md", rows, scope_label(args))
    print(f"resumo: {pdir / 'evidencias_resumo.md'}")
    return 0


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def cmd_avaliar(args: argparse.Namespace) -> int:
    out_dir = resolve_path(args.out)
    contexts = load_contexts(out_dir, scope_chapter(args), split_csv(args.chaves))
    refs = ev.load_bib_refs(resolve_path(args.bib))
    api_key = load_env_var("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY ausente (ambiente ou .env)", file=sys.stderr)
        return 2
    settings = avaliar_mod.Settings(
        model=args.modelo,
        max_tokens=args.max_tokens,
        temperature=0.0 if args.temperature is None else args.temperature,
        send_temperature=args.temperature is not None,
        modo=args.modo,
        max_chars=args.max_chars,
        max_contextos=args.max_contextos,
        concorrencia=args.concorrencia,
        refazer_falhas=args.refazer_falhas,
    )
    client = avaliar_mod.make_client(api_key)
    pdir = pilot_dir(out_dir, scope_chapter(args))
    reuse_root = out_dir / "piloto" if is_global(args) and not args.sem_reaproveitar else None
    result = avaliar_mod.run_avaliacao(
        capitulo=scope_label(args),
        escopo=ESCOPO_TESE if is_global(args) else "capitulo",
        reuse_root=reuse_root,
        contexts=contexts,
        refs=refs,
        proprios=set(split_csv(args.proprios)),
        out_dir=out_dir,
        pilot_dir=pdir,
        settings=settings,
        client=client,
        log=lambda text: print(text, flush=True),
    )
    counts = _count(row["veredito_final"] for row in result["rows"])
    print(f"status: {result['metadados']['status']}; vereditos: {counts}")
    print(f"saída: {pdir / 'avaliacoes.jsonl'}")
    return 0 if not result["incompleto"] else 3


def cmd_consultas(args: argparse.Namespace) -> int:
    out_dir = resolve_path(args.out)
    contexts = load_contexts(out_dir, scope_chapter(args), split_csv(args.chaves))
    api_key = load_env_var("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY ausente (ambiente ou .env)", file=sys.stderr)
        return 2
    settings = avaliar_mod.Settings(
        model=args.modelo,
        max_tokens=args.max_tokens,
        send_temperature=False,
        modo=args.modo,
        max_chars=args.max_chars,
        concorrencia=args.concorrencia,
        refazer_falhas=args.refazer_falhas,
    )
    pdir = pilot_dir(out_dir, scope_chapter(args))
    summary = cons.run_consultas(
        contexts=contexts,
        out_dir=out_dir,
        pilot_dir=pdir,
        settings=settings,
        client=avaliar_mod.make_client(api_key),
        log=lambda text: print(text, flush=True),
    )
    print(f"consultas: {summary}")
    print(f"saída: {tr.english_queries_path(out_dir)}")
    return 0 if not summary["incompleto"] else 3


def cmd_adjudicar(args: argparse.Namespace) -> int:
    out_dir = resolve_path(args.out)
    pdir = pilot_dir(out_dir, scope_chapter(args))
    rows = read_jsonl(pdir / "avaliacoes.jsonl")
    if not rows:
        print(f"{pdir / 'avaliacoes.jsonl'} ausente ou vazio; rode `avaliar` antes", file=sys.stderr)
        return 2
    if not any(r.get("veredito_final") == "DIVERGENTE" for r in rows):
        print("nenhum contexto DIVERGENTE para adjudicar")
        return 0
    contexts = load_contexts(out_dir, scope_chapter(args), [])
    api_key = load_env_var("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY ausente (ambiente ou .env)", file=sys.stderr)
        return 2
    settings = avaliar_mod.Settings(
        model=args.modelo,
        max_tokens=args.max_tokens,
        send_temperature=False,
        modo=args.modo,
        max_chars=args.max_chars,
        concorrencia=args.concorrencia,
        refazer_falhas=args.refazer_falhas,
    )
    result = adj.run_adjudicacao(
        capitulo=scope_label(args),
        escopo=ESCOPO_TESE if is_global(args) else "capitulo",
        contexts=contexts,
        out_dir=out_dir,
        pilot_dir=pdir,
        settings=settings,
        client=avaliar_mod.make_client(api_key),
        log=lambda text: print(text, flush=True),
    )
    counts = _count(row["veredito_final"] for row in result["rows"])
    print(f"adjudicados: {result['adjudicados']}; vereditos: {counts}")
    print(f"saída: {pdir / 'avaliacoes.jsonl'}")
    return 0 if not result["incompleto"] else 3


def cmd_relatorio(args: argparse.Namespace) -> int:
    out_dir = resolve_path(args.out)
    pdir = pilot_dir(out_dir, scope_chapter(args))
    rows = read_jsonl(pdir / "avaliacoes.jsonl")
    if not rows:
        print(f"{pdir / 'avaliacoes.jsonl'} ausente ou vazio; rode `avaliar` antes", file=sys.stderr)
        return 2
    meta = read_json(pdir / "metadados.json", default={})
    artefatos = read_json(pdir / "artefatos.json", default=[])
    classificacao = read_json(pdir / "classificacao.json", default={})
    precos = rel.load_precos(resolve_path(args.precos)) if args.precos else None
    text = rel.build_report(rows, meta, artefatos, classificacao, scope_label(args), precos,
                            escopo=ESCOPO_TESE if is_global(args) else "capitulo")
    write_text_atomic(pdir / "relatorio.md", text)
    print(f"relatório: {pdir / 'relatorio.md'}")
    return 0


def cmd_exportar_ledger(args: argparse.Namespace) -> int:
    out_dir = resolve_path(args.out)
    pdir = pilot_dir(out_dir, None)
    dest = resolve_path(args.saida)
    try:
        summary = led.exportar_ledger(
            pdir, out_dir, dest, aceitar_novas=args.aceitar_novas, marcar_baseline=args.marcar_baseline
        )
    except led.BaselineRecusada as exc:
        print(str(exc), file=sys.stderr)
        return 4
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"ledger: {dest / led.LEDGER_NAME} ({summary['linhas']} linhas)")
    print(f"pendências: {dest / led.PENDENCIAS_NAME} ({summary['pendencias']} abertas)")
    if summary["pendentes_sem_veredito"]:
        print(f"sem veredito final (fora do ledger): {len(summary['pendentes_sem_veredito'])}", file=sys.stderr)
    novas = summary["pendencias_novas"]
    if novas:
        print(f"problemas NOVOS ({len(novas)}): " + ", ".join(novas))
    if summary["pendencias_resolvidas"]:
        print(f"pendências resolvidas e removidas: {len(summary['pendencias_resolvidas'])}")
    if novas and not args.aceitar_novas:
        print(
            "NAO_SUSTENTA/CONTRADIZ novos não foram acrescentados a pendencias.yaml: corrija o texto ou "
            "rode de novo com --aceitar-novas (a decisão do autor continua exigida pelo teste de portão)",
            file=sys.stderr,
        )
        return 3
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Atestação bibliográfica da tese (piloto por capítulo).")
    sub = parser.add_subparsers(dest="comando", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument("--capitulo", help='título exato do capítulo, ex.: "Introdução"')
        group.add_argument("--escopo", choices=(ESCOPO_TESE,),
                           help="tese inteira (texto, apêndices, tabelas, notas): uma requisição por chave "
                                "atravessando os capítulos; saídas em piloto/tese/")
        p.add_argument("--out", default=DEFAULT_OUT, help="diretório de trabalho (padrão .local/biblio)")

    p_ev = sub.add_parser("evidencias", help="recupera evidências em fontes abertas")
    common(p_ev)
    p_ev.add_argument("--chaves", help="restringe a estas chaves (separadas por vírgula)")
    p_ev.add_argument("--proprios", help="chaves de obras próprias além das de autoria 'Jardim'")
    p_ev.add_argument("--bib", default=DEFAULT_BIB)
    p_ev.add_argument("--fontes-locais", default=".local/fontes", dest="fontes_locais",
                      help="pasta com textos fornecidos pelo autor (<chave>.txt/.pdf/.xml/.html); "
                           "têm prioridade sobre a rede (padrão .local/fontes; vazio desliga)")
    p_ev.add_argument("--refresh", action="store_true", help="ignora o cache de evidências")
    p_ev.add_argument("--intervalo", type=float, default=1.0, help="segundos mínimos entre requisições ao mesmo host")
    p_ev.set_defaults(func=cmd_evidencias)

    p_av = sub.add_parser("avaliar", help="avalia as citações com o Claude")
    common(p_av)
    p_av.add_argument("--modelo", default=DEFAULT_MODEL)
    p_av.add_argument("--modo", choices=("batch", "sync"), default="batch")
    p_av.add_argument("--max-chars", type=int, default=120000, dest="max_chars")
    p_av.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens")
    temp = p_av.add_mutually_exclusive_group()
    temp.add_argument("--temperature", type=float, default=None,
                      help="envia temperature à API (padrão: não enviar; o Sonnet atual a rejeita)")
    temp.add_argument("--omitir-temperature", action="store_true",
                      help="não envia temperature (já é o padrão; mantido por compatibilidade)")
    p_av.add_argument("--sem-reaproveitar", action="store_true", dest="sem_reaproveitar",
                      help="no escopo tese, não copia vereditos de execuções por capítulo")
    p_av.add_argument("--max-contextos", type=int, default=12, dest="max_contextos",
                      help="contextos por requisição (acima disso a chave vira mais de uma requisição)")
    p_av.add_argument("--concorrencia", type=int, default=2, help="threads no modo sync")
    p_av.add_argument("--chaves", help="restringe a estas chaves (útil para depurar em sync)")
    p_av.add_argument("--proprios")
    p_av.add_argument("--bib", default=DEFAULT_BIB)
    p_av.add_argument("--refazer-falhas", action="store_true", dest="refazer_falhas",
                      help="repete requisições concluídas com erro de resposta")
    p_av.set_defaults(func=cmd_avaliar)

    p_co = sub.add_parser("consultas", help="gera consultas em inglês (Haiku, lote) para a seleção de trechos")
    common(p_co)
    p_co.add_argument("--modelo", default=DEFAULT_QUERY_MODEL)
    p_co.add_argument("--modo", choices=("batch", "sync"), default="batch")
    p_co.add_argument("--max-chars", type=int, default=120000, dest="max_chars",
                      help="o mesmo valor de `avaliar`: só fontes com texto completo maior que isto precisam de consulta")
    p_co.add_argument("--max-tokens", type=int, default=600, dest="max_tokens")
    p_co.add_argument("--concorrencia", type=int, default=2, help="threads no modo sync")
    p_co.add_argument("--chaves", help="restringe a estas chaves (separadas por vírgula)")
    p_co.add_argument("--refazer-falhas", action="store_true", dest="refazer_falhas",
                      help="repete requisições concluídas com erro de resposta")
    p_co.set_defaults(func=cmd_consultas)

    p_adj = sub.add_parser("adjudicar", help="desempata os contextos DIVERGENTE com um modelo mais forte")
    common(p_adj)
    p_adj.add_argument("--modelo", default=DEFAULT_ADJ_MODEL)
    p_adj.add_argument("--modo", choices=("batch", "sync"), default="batch")
    p_adj.add_argument("--max-chars", type=int, default=120000, dest="max_chars",
                       help="o mesmo valor usado em `avaliar` (a evidência enviada tem de ser idêntica)")
    p_adj.add_argument("--max-tokens", type=int, default=8192, dest="max_tokens")
    p_adj.add_argument("--concorrencia", type=int, default=2, help="threads no modo sync")
    p_adj.add_argument("--refazer-falhas", action="store_true", dest="refazer_falhas",
                       help="reenvia requisições cujo lote terminou com erro ou cuja resposta veio inválida")
    p_adj.set_defaults(func=cmd_adjudicar)

    p_rel = sub.add_parser("relatorio", help="gera relatorio.md")
    common(p_rel)
    p_rel.add_argument("--precos", help="YAML com USD por milhão de tokens")
    p_rel.set_defaults(func=cmd_relatorio)

    p_led = sub.add_parser("exportar-ledger", help="exporta docs/atestado/ledger.csv e pendencias.yaml")
    p_led.add_argument("--escopo", choices=(ESCOPO_TESE,), required=True,
                       help="lê piloto/tese/avaliacoes.jsonl (e artefatos.json das obras próprias)")
    p_led.add_argument("--out", default=DEFAULT_OUT, help="diretório de trabalho (padrão .local/biblio)")
    p_led.add_argument("--saida", default="docs/atestado", help="pasta do ledger versionado (padrão docs/atestado)")
    p_led.add_argument("--aceitar-novas", action="store_true", dest="aceitar_novas",
                       help="acrescenta a pendencias.yaml os NAO_SUSTENTA/CONTRADIZ novos (decisao: pendente); "
                            "sem esta opção eles só são listados e o comando sai com código 3")
    p_led.add_argument("--marcar-baseline", action="store_true", dest="marcar_baseline",
                       help="migração única: marca as pendências existentes com origem: baseline-2026-11-08; "
                            "recusa (código 4, nada gravado) ids fora de tests/data/pendencias_baseline_ids.txt")
    p_led.set_defaults(func=cmd_exportar_ledger)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
