#!/usr/bin/env python
"""Cobertura de números da tese: quais números do texto compilado o registro cobre.

Percorre a árvore de ``text/latex/tese.tex`` (carregamento de ``linter_prosa``; sem
``*_diff*``, slides nem roteiro), extrai cada número da prosa, das legendas, das notas e
das tabelas (extração de ``portao_fidelidade``) e classifica:

- ``coberto``: o ``texto_pt`` de um valor do registro de invariantes
  (``tests/data/registro_invariantes.yaml``) está entre as exigências das ocorrências
  daquele arquivo (por nome de valor, ou texto literal igual a um ``texto_pt``);
- ``isento``: não é dado a registrar. Motivos: ``latex`` (dentro de matemática,
  ``\\texttt``, citação, ``\\ref``, medida etc.), ``data`` e ``ano`` (1900 a 2100 em
  contexto de data ou citação), ``rotulo`` (capítulo, seção, figura, tabela, equação,
  passo, estudo, objetivo, meta...; regra compartilhada com ``portao_fidelidade``), ``norma`` (ABNT NBR, ISO, Lei, Resolução, CID...) e
  ``identificador`` (dígito colado a letra: O3, H1, K2.5, COVID-19);
- ``descoberto``: o resto.

O número é comparado ao ``texto_pt`` pelo valor (``61.696`` = ``61696``), com sinal e
``%`` ignorados. Um ``texto_pt`` que não é só número (por exemplo ``1:249``) cobre os
números dentro de cada ocorrência literal dele no texto.

Saídas: ``.local/cobertura/numeros.csv`` (arquivo, linha, numero, classe, frase com até
200 caracteres) e um resumo no stdout: totais, por arquivo, porcentagem de cobertos entre
os não isentos e os 30 descobertos mais frequentes.

Uso::

    python scripts/cobertura_numeros.py [--min-cobertura P]

``--min-cobertura P`` (0 a 100) sai com código 1 se a porcentagem de cobertos entre os
não isentos ficar abaixo de P.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import linter_prosa as lp  # noqa: E402
import registro_invariantes as ri  # noqa: E402
from portao_fidelidade import RE_NUMERO, RE_SEP_MILHAR, normalizar_numero  # noqa: E402

SAIDA_PADRAO = ".local/cobertura/numeros.csv"
MAX_FRASE_CSV = 200
TOP_DESCOBERTOS = 30

CLASSES = ("coberto", "isento", "descoberto")


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------

@dataclass
class Numero:
    arquivo: str
    linha: int
    numero: str      # como está no texto (sem sinal), com "%" se houver
    classe: str      # coberto | isento | descoberto
    motivo: str      # só para isento (e "registro" para coberto)
    frase: str

    def frase_csv(self) -> str:
        return self.frase[:MAX_FRASE_CSV]

    def chave(self) -> str:
        """Forma canônica do valor (sem ``%``), para contar repetições."""
        return normalizar_numero(self.numero.rstrip("%"))


def eh_arquivo_compilado(rel: str) -> bool:
    """Texto que entra no PDF: sem ``*_diff*``, slides nem roteiro."""
    rel = rel.replace("\\", "/")
    nome = rel.rsplit("/", 1)[-1]
    if not rel.endswith(".tex") or "_diff" in nome or "roteiro" in nome:
        return False
    return not rel.startswith("text/slides/") and "/slides/" not in rel


# --------------------------------------------------------------------------
# Registro
# --------------------------------------------------------------------------

def textos_por_arquivo(reg: ri.Registro) -> dict[str, set[str]]:
    """``texto_pt`` exigidos pelas ocorrências, por arquivo (caminho com ``/``)."""
    conhecidos = {str(v["texto_pt"]) for _, v in reg.valores.values()
                  if v.get("texto_pt") is not None}
    saida: dict[str, set[str]] = defaultdict(set)
    for e in reg.entradas:
        for oc in e.get("ocorrencias") or []:
            arquivo = str(oc["arquivo"]).replace("\\", "/")
            for chave in ("exige", "exige_todos", "exige_algum"):
                for token in oc.get(chave) or []:
                    for alt in (token if isinstance(token, list) else [token]):
                        alt = str(alt)
                        if alt in reg.valores:
                            texto = reg.valores[alt][1].get("texto_pt")
                            if texto is not None:
                                saida[arquivo].add(str(texto))
                        elif alt in conhecidos:
                            saida[arquivo].add(alt)
    return saida


# --------------------------------------------------------------------------
# Extração
# --------------------------------------------------------------------------

_RE_NORM = re.compile(r"\{,\}|" + RE_SEP_MILHAR.pattern)
_RE_NUMERO_PURO = re.compile(r"[-\u2212+]?\d+(?:[.,]\d+)*")


def _normalizar_com_mapa(texto: str) -> tuple[str, list[int]]:
    """Troca ``{,}`` por ``,`` e tira o separador de milhar TeX, como o portão de
    fidelidade; ``mapa[i]`` é a posição, no texto original, do caractere ``i``."""
    partes: list[str] = []
    mapa: list[int] = []
    ultimo = 0
    for m in _RE_NORM.finditer(texto):
        partes.append(texto[ultimo:m.start()])
        mapa.extend(range(ultimo, m.start()))
        if m.group(0) == "{,}":
            partes.append(",")
            mapa.append(m.start())
        ultimo = m.end()
    partes.append(texto[ultimo:])
    mapa.extend(range(ultimo, len(texto)))
    mapa.append(len(texto))
    return "".join(partes), mapa


# --------------------------------------------------------------------------
# Isenções
# --------------------------------------------------------------------------

_RE_UNIDADE = re.compile(
    r"(?:cm|mm|pt|em|bp|sp)(?![A-Za-z])|\\(?:textwidth|linewidth|columnwidth|"
    r"textheight|paperwidth)")
_RE_ESTRUTURA = re.compile(
    r"\\(?:renewcommand\s*\{\\(?:arraystretch|tabcolsep|arrayrulewidth|baselinestretch)\}"
    r"|fontsize|linespread|setcounter|rule|cline|addtolength|setlength)"
    r"\s*(?:\[[^\]]*\]\s*)?(?:\{[^{}]*\}\s*){1,2}")
_RE_DATA = re.compile(
    r"(?<![\d/.,])\d{1,2}/\d{1,2}/(?:19|20)\d\d(?!\d)"
    r"|(?<![\d/.,])\d{1,2}/(?:19|20)\d\d(?!\d)")
_RE_ANO_ANTES = re.compile(
    r"(?:(?<!\w)(?:em|de|desde|até|entre|ano|anos|durante|após|antes|in|since|from)\s*"
    r"|[(\[]\s*)$", re.I)
_RE_ANO_PAR_ANTES = re.compile(r"(?<!\d)(?:19|20)\d\d\s*(?:e|a|ou|-|\u2013|--)\s*$", re.I)
_RE_ANO_PAR_DEPOIS = re.compile(r"\s*(?:e|a|ou|-|\u2013|--)\s*(?:19|20)\d\d(?!\d)", re.I)
_RE_ANO_FECHA = re.compile(r"\s*[)\]]")
_PALAVRAS_ROTULO = (
    r"cap[íi]tulos?|caps?\.|se[çc][ãa]o|se[çc][õo]es|subse[çc][ãa]o|subse[çc][õo]es|"
    r"tabelas?|tabs?\.|figuras?|figs?\.|quadros?|equa[çc][ãa]o|equa[çc][õo]es|eqs?\.|"
    r"ap[êe]ndices?|anexos?|passos?|etapas?|f[óo]rmulas?|p[áa]ginas?|pp?\.|linhas?|"
    r"itens?|algoritmos?|listagens?|hip[óo]teses?|estudos?|experimentos?|objetivos?|metas?")
_RE_ROTULO = re.compile(
    r"(?<!\w)(?:" + _PALAVRAS_ROTULO + r")\s*"
    r"(?:\d+(?:[.,]\d+)*\s*(?:,|e|ou|a|-|\u2013|--)\s*)*(?:[A-Z]{1,2}[.-])?$", re.I)
_RE_NORMA = re.compile(
    r"(?<!\w)(?:NBR|ISO|IEC|RDC|CID|Lei|Decreto|Portaria|Resolu[çc][ãa]o|"
    r"Instru[çc][ãa]o Normativa|Norma|RFC)"
    r"(?:[\s-]+(?:CNS|CNPq|MS|SUS|GM|Complementar|Federal|Geral|Estadual|"
    r"n[ºo°]\.?|n\.º|de|no))*[\s-]*(?:\d+(?:[.,]\d+)*[\s/.-]*)*$", re.I)


def eh_identificador(antes: str) -> bool:
    """O número que segue ``antes`` é dígito de identificador (O3, H1, K2.5, COVID-19)?

    Regra compartilhada com ``portao_fidelidade``: o trecho anterior termina em letra, ou
    em hífen precedido de letra.
    """
    return antes[-1:].isalpha() or (antes[-1:] == "-" and antes[-2:-1].isalpha())


def _motivo_isencao(arq: lp.Arquivo, pos: int, novo: str, ini: int, fim: int,
                    bruto: str, estruturas: list[tuple[int, int]],
                    datas: list[tuple[int, int]]) -> str | None:
    """Motivo da isenção do número em ``novo[ini:fim]`` (posição ``pos`` no arquivo)."""
    if arq.texto[pos].isspace():  # o linter branqueou: matemática, \texttt, \ref, ...
        return "latex"
    if _RE_UNIDADE.match(novo, fim) or any(a <= ini < b for a, b in estruturas):
        return "latex"
    if any(a <= ini < b for a, b in datas):
        return "data"
    antes = re.sub(r"\s+", " ", novo[max(0, ini - 120):ini])
    depois = novo[fim:fim + 40]
    if re.fullmatch(r"\d{4}", bruto) and 1900 <= int(bruto) <= 2100 and (
            _RE_ANO_ANTES.search(antes) or _RE_ANO_PAR_ANTES.search(antes)
            or _RE_ANO_FECHA.match(depois) or _RE_ANO_PAR_DEPOIS.match(depois)
            or antes.endswith("/") and antes[-2:-1].isdigit()):
        return "ano"
    if _RE_ROTULO.search(antes):
        return "rotulo"
    if _RE_NORMA.search(antes):
        return "norma"
    if eh_identificador(antes):
        return "identificador"
    return None


# --------------------------------------------------------------------------
# Classificação
# --------------------------------------------------------------------------

def _intervalos(padrao: re.Pattern, texto: str) -> list[tuple[int, int]]:
    return [m.span() for m in padrao.finditer(texto)]


class _Frases:
    """Frase (ou linha de tabela) que contém uma posição do arquivo."""

    def __init__(self, arq: lp.Arquivo, lim: str):
        self.arq, self.lim = arq, lim
        quebras = list(re.finditer(r"\n[ \t]*\n", lim))
        self.inicios = [0] + [m.end() for m in quebras]
        self.fins = [m.start() for m in quebras] + [len(lim)]
        self._cache: dict[int, list[tuple[int, int]]] = {}

    def de(self, pos: int) -> str:
        if self.arq.tipo(pos) == "tabela":
            ini = self.lim.rfind("\n", 0, pos) + 1
            fim = self.lim.find("\n", pos)
            return " ".join(self.lim[ini:fim if fim != -1 else len(self.lim)].split())
        k = bisect.bisect_right(self.inicios, pos) - 1
        if k not in self._cache:
            self._cache[k] = ri.spans_frases(self.lim[self.inicios[k]:self.fins[k]])
        rel = pos - self.inicios[k]
        for a, b in self._cache[k]:
            if a <= rel < b:
                return " ".join(self.lim[self.inicios[k] + a:self.inicios[k] + b].split())
        return " ".join(self.lim[self.inicios[k]:self.fins[k]].split())


def classificar_arquivo(arq: lp.Arquivo, cobertos: set[str]) -> list[Numero]:
    """Extrai e classifica os números de um arquivo preparado por ``linter_prosa``.

    ``cobertos``: ``texto_pt`` dos valores do registro exigidos para este arquivo.
    """
    lim = arq.limpo.replace("~", " ")
    novo, mapa = _normalizar_com_mapa(lim)
    puros = {normalizar_numero(t.lstrip("-\u2212+")) for t in cobertos
             if _RE_NUMERO_PURO.fullmatch(t)}
    compostos = [t for t in cobertos if not _RE_NUMERO_PURO.fullmatch(t)]
    spans_compostos = [m.span() for t in compostos for m in re.finditer(re.escape(t), novo)]
    estruturas = _intervalos(_RE_ESTRUTURA, novo)
    datas = _intervalos(_RE_DATA, novo)
    frases = _Frases(arq, lim)

    saida: list[Numero] = []
    for m in RE_NUMERO.finditer(novo):
        bruto = m.group("num")
        ini, fim = m.start("num"), m.end("num")
        pos = mapa[ini]
        motivo = _motivo_isencao(arq, pos, novo, ini, fim, bruto, estruturas, datas)
        if motivo:
            classe = "isento"
        elif normalizar_numero(bruto) in puros or any(a <= ini < b for a, b in spans_compostos):
            classe, motivo = "coberto", "registro"
        else:
            classe, motivo = "descoberto", ""
        saida.append(Numero(arq.rel, arq.linha(pos), bruto + ("%" if m.group("pct") else ""),
                            classe, motivo, frases.de(pos)))
    return saida


def analisar(tese: Path, reg: ri.Registro, base: Path) -> list[Numero]:
    """Todos os números do texto compilado, na ordem de leitura dos arquivos."""
    arvore = lp.carregar_arvore(Path(tese), Path(base))
    por_arquivo = textos_por_arquivo(reg)
    saida: list[Numero] = []
    for arq in arvore.arquivos.values():
        if eh_arquivo_compilado(arq.rel):
            saida.extend(classificar_arquivo(arq, por_arquivo.get(arq.rel, set())))
    return saida


# --------------------------------------------------------------------------
# Resumo
# --------------------------------------------------------------------------

def resumir(numeros: list[Numero]) -> dict:
    c = Counter(n.classe for n in numeros)
    total = len(numeros)
    nao_isentos = c["coberto"] + c["descoberto"]
    return {
        "total": total,
        "coberto": c["coberto"],
        "isento": c["isento"],
        "descoberto": c["descoberto"],
        "pct_geral": 100.0 * c["coberto"] / total if total else 100.0,
        "pct_nao_isentos": 100.0 * c["coberto"] / nao_isentos if nao_isentos else 100.0,
    }


def top_descobertos(numeros: list[Numero], n: int = TOP_DESCOBERTOS):
    """``(chave, ocorrências, exemplo)`` dos descobertos mais frequentes."""
    contagem: Counter = Counter()
    exemplo: dict[str, Numero] = {}
    for x in numeros:
        if x.classe == "descoberto":
            contagem[x.chave()] += 1
            exemplo.setdefault(x.chave(), x)
    ordem = sorted(contagem.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(k, v, exemplo[k]) for k, v in ordem[:n]]


def texto_resumo(numeros: list[Numero]) -> str:
    geral = resumir(numeros)
    linhas = [
        "Cobertura de números",
        f"Total: {geral['total']} | coberto: {geral['coberto']} | "
        f"isento: {geral['isento']} | descoberto: {geral['descoberto']}",
        f"Cobertura geral (cobertos/total): {geral['pct_geral']:.1f}%",
        f"Cobertos entre os não isentos: {geral['pct_nao_isentos']:.1f}%",
        "",
        "Isentos por motivo: " + (", ".join(
            f"{k}={v}" for k, v in sorted(Counter(
                n.motivo for n in numeros if n.classe == "isento").items())) or "nenhum"),
        "",
        "Por arquivo (total, coberto, isento, descoberto, % cobertos entre não isentos):",
    ]
    por_arquivo: dict[str, list[Numero]] = {}
    for n in numeros:
        por_arquivo.setdefault(n.arquivo, []).append(n)
    for arquivo, nums in por_arquivo.items():
        r = resumir(nums)
        linhas.append(f"  {arquivo}: {r['total']}, {r['coberto']}, {r['isento']}, "
                      f"{r['descoberto']}, {r['pct_nao_isentos']:.1f}%")
    linhas += ["", f"Descobertos mais frequentes (top {TOP_DESCOBERTOS}):"]
    for chave, k, ex in top_descobertos(numeros):
        linhas.append(f"  {k:4d}x  {ex.numero}  (ex.: {ex.arquivo}:{ex.linha})")
    if not any(n.classe == "descoberto" for n in numeros):
        linhas.append("  nenhum")
    return "\n".join(linhas)


def gravar_csv(numeros: list[Numero], destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "linha", "numero", "classe", "frase"])
        for n in numeros:
            w.writerow([n.arquivo, n.linha, n.numero, n.classe, n.frase_csv()])


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tese", default=str(lp.TESE_PADRAO))
    ap.add_argument("--registro", default=str(ri.REGISTRO))
    ap.add_argument("--raiz", default=str(lp.REPO),
                    help="base dos caminhos relativos (devem coincidir com o registro)")
    ap.add_argument("--saida", default=SAIDA_PADRAO, help="CSV de saída")
    ap.add_argument("--min-cobertura", type=float, default=None, metavar="P",
                    help="sai com 1 se os cobertos entre os não isentos ficarem abaixo "
                         "de P por cento")
    args = ap.parse_args(argv)

    raiz = Path(args.raiz)
    reg = ri.carregar_registro(Path(args.registro))
    numeros = analisar(Path(args.tese), reg, raiz)
    destino = Path(args.saida)
    if not destino.is_absolute():
        destino = raiz / destino
    gravar_csv(numeros, destino)
    print(texto_resumo(numeros))
    print(f"\nCSV: {destino}")
    if args.min_cobertura is not None:
        pct = resumir(numeros)["pct_nao_isentos"]
        if pct < args.min_cobertura:
            print(f"Cobertura {pct:.1f}% abaixo do mínimo {args.min_cobertura:.1f}%",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
