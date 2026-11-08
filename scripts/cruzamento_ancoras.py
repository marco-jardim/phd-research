"""Cruzamento sistemático da tese, dos slides e do roteiro com as âncoras (map-reduce com LLM).

Substitui a busca por grep por uma varredura em fatias pequenas. Todas as saídas ficam em
``.local/cruzamento/`` (ignorado pelo git). Caminhos relativos valem a partir da raiz do repositório.
Sequência de execução::

    python scripts/cruzamento_ancoras.py fatiar
    python scripts/cruzamento_ancoras.py mapear    --estimar                   # só estima; não chama a API
    python scripts/cruzamento_ancoras.py mapear    --modo sync --limite 3      # piloto em 3 fatias
    python scripts/cruzamento_ancoras.py mapear                                # lote (padrão), retomável
    python scripts/cruzamento_ancoras.py cruzar    --estimar
    python scripts/cruzamento_ancoras.py cruzar
    python scripts/cruzamento_ancoras.py adjudicar --estimar
    python scripts/cruzamento_ancoras.py adjudicar
    python scripts/cruzamento_ancoras.py relatorio

1. ``fatiar``: grava ``fatias.jsonl`` (grupos de parágrafos inteiros da mesma seção, até 1.200 palavras;
   um frame por fatia nos slides; uma seção ou bloco por fatia no roteiro). Só entra o que a árvore de
   ``\\input`` de ``tese.tex`` alcança; ``*_diff.tex`` e ``*_diff_work/`` ficam de fora. Determinístico.
2. ``mapear``: uma requisição por fatia ao Claude (padrão Sonnet 5.5). Devolve as relações não neutras com
   as âncoras (afirma, contradiz, enfraquece, exagera, omite_ressalva) e as afirmações substantivas da
   fatia (tipo, tópicos da taxonomia fixa, números, polaridade, escopo). Todo trecho é conferido
   literalmente contra a fatia; o que não existir fica marcado ``trecho_invalido``. Saída: ``mapa.jsonl``.
3. ``cruzar``: por tópico, o Sonnet aponta pares de afirmações inconsistentes entre arquivos ou seções
   (número divergente, polaridade oposta, escopo trocado, definição divergente, superlativo sem suporte).
   Tópicos com mais de 60 afirmações são divididos em lotes, conferidos entre si dois a dois.
   Saída: ``pares.jsonl`` e ``pares_problemas.jsonl``.
4. ``adjudicar``: cada par e cada relação contradiz/exagera vai ao Claude Opus (padrão ``claude-opus-5-5``)
   com as fatias completas: CONFIRMADO, FALSO_ALARME ou DEPENDE_DE_ESCOPO, com a correção mínima.
   Saída: ``adjudicacoes.jsonl``.
5. ``relatorio``: ``relatorio.md`` (por âncora, contradições confirmadas por tópico, candidatas a âncora,
   tokens por etapa) e ``matriz_ancoras.csv`` (âncora × arquivo × contagem por relação).

``mapear``, ``cruzar`` e ``adjudicar`` aceitam:

* ``--modo batch`` (padrão): Message Batches API, retomável (o id do lote fica em
  ``<etapa>.checkpoint.json``; rode de novo para continuar a espera e baixar). ``--modo sync``: pool de
  ``--concorrencia`` threads (padrão 8), retry com backoff exponencial em 429, 529 e erros de rede,
  checkpoint gravado a cada resposta e retomada sem reenviar o que já tem resposta.
* ``--estimar``: não chama a API; imprime requisições e tokens aproximados de entrada e saída
  (cerca de 3,5 caracteres por token; a saída é uma estimativa por etapa, não um limite).
* ``--refazer-falhas``: reenvia o que terminou com erro de resposta (JSON inválido, resposta truncada).

O cache de prompt (``cache_control``) fica no bloco de sistema (âncoras, taxonomia e terminologia) e nunca
se envia ``temperature``. Chave da API: ``ANTHROPIC_API_KEY`` (ambiente ou ``.env``), nunca impressa nem
gravada. Códigos de saída: 0 sucesso; 2 insumo ou chave ausente; 3 execução incompleta ou com erros.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.atestado import avaliar as av  # noqa: E402
from scripts.atestado.common import (  # noqa: E402
    DEFAULT_ADJ_MODEL,
    DEFAULT_MODEL,
    load_env_var,
    resolve_path,
    write_jsonl_atomic,
)
from scripts.cruzamento import adjudicar as adj  # noqa: E402
from scripts.cruzamento import cruzar as crz  # noqa: E402
from scripts.cruzamento import mapear as mp  # noqa: E402
from scripts.cruzamento import relatorio as rel  # noqa: E402
from scripts.cruzamento.comum import (  # noqa: E402
    DEFAULT_ANCORAS,
    DEFAULT_OUT,
    DEFAULT_ROTEIRO,
    DEFAULT_SLIDES,
    DEFAULT_TESE,
    FATIAS_NOME,
    MAPA_NOME,
    MAX_PALAVRAS_FATIA,
    PARES_NOME,
    carregar_ancoras,
    formatar_milhar,
    ler_jsonl,
)
from scripts.cruzamento.execucao import CONCORRENCIA_PADRAO, criar_settings  # noqa: E402
from scripts.cruzamento.fatiar import fatiar  # noqa: E402


def log(texto: str) -> None:
    print(texto, flush=True)


def _erro(texto: str) -> int:
    print(texto, file=sys.stderr)
    return 2


def _opcional(valor: str) -> Path | None:
    return resolve_path(valor) if valor else None


def _imprimir_estimativa(etapa: str, est: dict[str, int]) -> None:
    print(
        f"{etapa}: {formatar_milhar(est['requisicoes'])} requisições; "
        f"entrada ≈ {formatar_milhar(est['entrada'])} tokens; saída ≈ {formatar_milhar(est['saida'])} tokens "
        "(aproximado: 3,5 caracteres por token; a entrada conta o prefixo em cache por inteiro)"
    )


def _carregar_ancoras(args: argparse.Namespace):
    try:
        return carregar_ancoras(resolve_path(args.ancoras)), None
    except (FileNotFoundError, ValueError) as exc:
        return None, _erro(str(exc))


def _carregar_fatias(out: Path):
    fatias = ler_jsonl(out / FATIAS_NOME)
    if not fatias:
        return None, _erro(f"{out / FATIAS_NOME} ausente ou vazio; rode `fatiar` antes")
    return fatias, None


def _cliente():
    chave = load_env_var("ANTHROPIC_API_KEY")
    if not chave:
        return None, _erro("ANTHROPIC_API_KEY ausente (ambiente ou .env)")
    return av.make_client(chave), None


def _settings(args: argparse.Namespace):
    return criar_settings(args.modelo, args.modo, args.max_tokens, args.concorrencia, args.refazer_falhas)


# ---------------------------------------------------------------------------
# Subcomandos
# ---------------------------------------------------------------------------
def cmd_fatiar(args: argparse.Namespace) -> int:
    out = resolve_path(args.out)
    base = resolve_path(args.base)
    try:
        fatias = fatiar(base, resolve_path(args.tese), _opcional(args.slides), _opcional(args.roteiro),
                        args.max_palavras)
    except FileNotFoundError as exc:
        return _erro(f"arquivo ausente: {exc}")
    write_jsonl_atomic(out / FATIAS_NOME, fatias)
    por_fonte: dict[str, list[int]] = {}
    for fatia in fatias:
        por_fonte.setdefault(fatia["fonte"], []).append(fatia["palavras"])
    for fonte, palavras in sorted(por_fonte.items()):
        print(f"{fonte}: {len(palavras)} fatias, {formatar_milhar(sum(palavras))} palavras (maior: {max(palavras)})")
    print(f"saída: {out / FATIAS_NOME} ({len(fatias)} fatias)")
    return 0


def cmd_mapear(args: argparse.Namespace) -> int:
    out = resolve_path(args.out)
    fatias, erro = _carregar_fatias(out)
    if erro is not None:
        return erro
    if args.limite:
        fatias = fatias[: args.limite]
    ancoras, erro = _carregar_ancoras(args)
    if erro is not None:
        return erro
    settings = _settings(args)
    if args.estimar:
        _imprimir_estimativa("mapear", mp.estimar_mapear(fatias, ancoras, settings))
        return 0
    client, erro = _cliente()
    if erro is not None:
        return erro
    resultado = mp.run_mapear(fatias=fatias, ancoras=ancoras, out=out, settings=settings, client=client, log=log)
    print(f"mapear: {resultado['requisicoes']} requisições; pendentes {resultado['pendentes']}; "
          f"fatias com erro {resultado['erros']}; trechos inválidos {resultado['trechos_invalidos']}")
    print(f"saída: {out / MAPA_NOME}")
    return 3 if resultado["incompleto"] or resultado["erros"] else 0


def cmd_cruzar(args: argparse.Namespace) -> int:
    out = resolve_path(args.out)
    mapa = ler_jsonl(out / MAPA_NOME)
    if not mapa:
        return _erro(f"{out / MAPA_NOME} ausente ou vazio; rode `mapear` antes")
    settings = _settings(args)
    if args.estimar:
        _imprimir_estimativa("cruzar", crz.estimar_cruzar(mapa, settings, args.max_afirmacoes, args.incluir_outro))
        return 0
    client, erro = _cliente()
    if erro is not None:
        return erro
    resultado = crz.run_cruzar(mapa=mapa, out=out, settings=settings, client=client,
                               max_afirmacoes=args.max_afirmacoes, incluir_outro=args.incluir_outro, log=log)
    print(f"cruzar: {resultado['requisicoes']} requisições; pares {len(resultado['pares'])}; "
          f"problemas {len(resultado['problemas'])}; pendentes {resultado['pendentes']}")
    print(f"saída: {out / PARES_NOME}")
    return 3 if resultado["incompleto"] or resultado["problemas"] else 0


def cmd_adjudicar(args: argparse.Namespace) -> int:
    out = resolve_path(args.out)
    mapa = ler_jsonl(out / MAPA_NOME)
    if not mapa:
        return _erro(f"{out / MAPA_NOME} ausente ou vazio; rode `mapear` antes")
    pares_path = out / PARES_NOME
    if not pares_path.is_file():
        return _erro(f"{pares_path} ausente; rode `cruzar` antes")
    pares = ler_jsonl(pares_path)
    fatias, erro = _carregar_fatias(out)
    if erro is not None:
        return erro
    ancoras, erro = _carregar_ancoras(args)
    if erro is not None:
        return erro
    settings = _settings(args)
    try:
        if args.estimar:
            _imprimir_estimativa("adjudicar", adj.estimar_adjudicar(pares, mapa, fatias, ancoras, settings))
            return 0
        client, erro = _cliente()
        if erro is not None:
            return erro
        resultado = adj.run_adjudicar(pares=pares, mapa=mapa, fatias=fatias, ancoras=ancoras, out=out,
                                      settings=settings, client=client, log=log)
    except ValueError as exc:
        return _erro(str(exc))
    print(f"adjudicar: {resultado['requisicoes']} requisições; vereditos {resultado['vereditos']}; "
          f"pendentes {resultado['pendentes']}")
    print(f"saída: {out / 'adjudicacoes.jsonl'}")
    return 3 if resultado["incompleto"] or resultado["erros"] else 0


def cmd_relatorio(args: argparse.Namespace) -> int:
    out = resolve_path(args.out)
    ancoras, erro = _carregar_ancoras(args)
    if erro is not None:
        return erro
    try:
        resultado = rel.gerar_relatorio(out, ancoras, args.min_ocorrencias)
    except ValueError as exc:
        return _erro(str(exc))
    print(f"relatório: {resultado['relatorio']}")
    print(f"matriz: {resultado['matriz']} ({resultado['linhas_matriz']} linhas)")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cruzamento da tese com as âncoras (map-reduce com LLM).")
    sub = parser.add_subparsers(dest="comando", required=True)

    def comum(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", default=DEFAULT_OUT, help=f"diretório de trabalho (padrão {DEFAULT_OUT})")

    def api(p: argparse.ArgumentParser, modelo: str, max_tokens: int) -> None:
        p.add_argument("--modelo", default=modelo)
        p.add_argument("--modo", choices=("batch", "sync"), default="batch")
        p.add_argument("--concorrencia", type=int, default=CONCORRENCIA_PADRAO,
                       help=f"threads no modo sync (padrão {CONCORRENCIA_PADRAO})")
        p.add_argument("--max-tokens", type=int, default=max_tokens, dest="max_tokens")
        p.add_argument("--estimar", action="store_true",
                       help="não chama a API: imprime requisições e tokens aproximados")
        p.add_argument("--refazer-falhas", action="store_true", dest="refazer_falhas",
                       help="reenvia o que terminou com erro de resposta")

    p_fa = sub.add_parser("fatiar", help="gera fatias.jsonl")
    comum(p_fa)
    p_fa.add_argument("--tese", default=DEFAULT_TESE)
    p_fa.add_argument("--slides", default=DEFAULT_SLIDES, help="vazio desliga")
    p_fa.add_argument("--roteiro", default=DEFAULT_ROTEIRO, help="vazio desliga")
    p_fa.add_argument("--base", default=".", help="raiz dos caminhos relativos gravados")
    p_fa.add_argument("--max-palavras", type=int, default=MAX_PALAVRAS_FATIA, dest="max_palavras")
    p_fa.set_defaults(func=cmd_fatiar)

    p_ma = sub.add_parser("mapear", help="relações com as âncoras e afirmações por fatia")
    comum(p_ma)
    api(p_ma, DEFAULT_MODEL, 8192)
    p_ma.add_argument("--ancoras", default=DEFAULT_ANCORAS)
    p_ma.add_argument("--limite", type=int, default=0, help="só as N primeiras fatias (piloto)")
    p_ma.set_defaults(func=cmd_mapear)

    p_cr = sub.add_parser("cruzar", help="pares inconsistentes por tópico")
    comum(p_cr)
    api(p_cr, DEFAULT_MODEL, 4096)
    p_cr.add_argument("--max-afirmacoes", type=int, default=crz.MAX_AFIRMACOES_PADRAO, dest="max_afirmacoes",
                      help="afirmações por lote dentro de um tópico")
    p_cr.add_argument("--incluir-outro", action="store_true", dest="incluir_outro",
                      help='inclui o tópico "outro" (fora por padrão)')
    p_cr.set_defaults(func=cmd_cruzar)

    p_ad = sub.add_parser("adjudicar", help="adjudica pares e relações contradiz/exagera (Opus)")
    comum(p_ad)
    api(p_ad, DEFAULT_ADJ_MODEL, 2048)
    p_ad.add_argument("--ancoras", default=DEFAULT_ANCORAS)
    p_ad.set_defaults(func=cmd_adjudicar)

    p_re = sub.add_parser("relatorio", help="gera relatorio.md e matriz_ancoras.csv")
    comum(p_re)
    p_re.add_argument("--ancoras", default=DEFAULT_ANCORAS)
    p_re.add_argument("--min-ocorrencias", type=int, default=3, dest="min_ocorrencias",
                      help="ocorrências mínimas para listar uma afirmação frequente sem âncora")
    p_re.set_defaults(func=cmd_relatorio)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
