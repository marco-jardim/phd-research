"""Linter de prosa da tese (LaTeX) com baseline que só pode encolher.

Substitui a memória do revisor por verificações mecânicas derivadas de
``docs/support_text/STYLE_PROMPT_escrita_academica.md`` (o "guia") e do
``AGENTS.md`` (seção 1). Só biblioteca padrão.

Leitura da tese: segue ``\\input``/``\\include`` a partir de ``text/latex/tese.tex``
(``tese_rev_diff.tex`` é ignorado), acompanha os capítulos
(``\\begin{chapter}{...}`` e ``\\chapter{...}``), o arquivo e a linha, e neutraliza
(troca por espaços, preservando posições) comentários, ``verbatim``/``lstlisting``,
matemática, ``\\texttt``, ``\\path``, ``\\url``, ``\\href``, citações, ``\\ref``,
``\\label`` e afins. Cada trecho recebe um tipo: ``prosa``, ``legenda``, ``tabela``,
``nota`` ou ``titulo`` (argumento de ``\\chapter``/``\\section``; os títulos não contam
como primeira menção na regra R03). Em ``tese.tex`` só vale o que vem depois de
``\\begin{document}``, e o Abstract em inglês (entre os marcadores em
``ABSTRACT_INICIO``/``ABSTRACT_FIM``) é ignorado.

Regras: R01 a R10 (ver ``REGRAS``). ``siglas.tex`` e ``glossario.tex`` são definições:
só recebem as regras com ``aplica_definicoes`` (R01 e R08).

Baseline (``tests/data/linter_prosa_baseline.json``): contagem por (regra, arquivo).
O gate falha se uma contagem passar do baseline ou surgir violação em (regra, arquivo)
ausente dele.

Uso::

    python scripts/linter_prosa.py                     # relatório + confere o baseline
    python scripts/linter_prosa.py --gerar-baseline    # recria o baseline do zero
    python scripts/linter_prosa.py --atualizar-baseline  # só diminui contagens

O relatório sai em ``.local/linter/relatorio.md``.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

REPO = Path(__file__).resolve().parents[1]
TESE_PADRAO = REPO / "text" / "latex" / "tese.tex"
BASELINE_PADRAO = REPO / "tests" / "data" / "linter_prosa_baseline.json"
RELATORIO_PADRAO = REPO / ".local" / "linter" / "relatorio.md"

EXCLUIR_ARQUIVOS = {"tese_rev_diff.tex"}
ARQUIVOS_DEFINICOES = {"siglas.tex", "glossario.tex"}

# O Abstract em inglês de tese.tex não segue o guia (que é em português).
ABSTRACT_INICIO = re.compile(r"bf\s+Abstract\}")
ABSTRACT_FIM = re.compile(r"\\textbf\{Keywords:\}[^\n]*")

PALAVRAS_MAX_PARAGRAFO = 200  # R09
PALAVRAS_MIN_PROSA_CORRIDA = 10  # R10: parágrafo com menos palavras não é prosa corrida

SEVERIDADES = ("erro", "aviso")
I = re.IGNORECASE

# ---------------------------------------------------------------------------
# Texto LaTeX -> texto neutralizado
# ---------------------------------------------------------------------------
_ACENTOS = {"'": "\u0301", "`": "\u0300", "^": "\u0302", "~": "\u0303", '"': "\u0308"}
_RE_ACENTO = re.compile(r"\\(['`^~\"])\s*(?:\{\s*(\\i|[A-Za-z])\s*\}|(\\i|[A-Za-z]))")
_RE_CEDILHA = re.compile(r"\\c\s*(?:\{\s*([A-Za-z])\s*\}|\s+([A-Za-z]))")


def acentos_latex(s: str) -> str:
    r"""``\c{c}\~{a}\'{\i}`` -> ``ç ã í`` (Unicode NFC)."""

    def acento(m: re.Match) -> str:
        letra = m.group(2) or m.group(3)
        if letra == "\\i":
            letra = "i"
        return letra + _ACENTOS[m.group(1)]

    s = _RE_ACENTO.sub(acento, s)
    s = _RE_CEDILHA.sub(lambda m: (m.group(1) or m.group(2)) + "\u0327", s)
    return unicodedata.normalize("NFC", s)


def _branquear(texto: str, intervalos: Iterable[tuple[int, int]]) -> str:
    """Troca por espaço cada caractere (menos quebras de linha) dos intervalos."""
    partes = list(texto)
    for a, b in intervalos:
        for i in range(max(0, a), min(len(partes), b)):
            if partes[i] != "\n":
                partes[i] = " "
    return "".join(partes)


def _intervalos_comentarios(texto: str) -> list[tuple[int, int]]:
    """Do ``%`` não escapado até o fim de cada linha."""
    saida: list[tuple[int, int]] = []
    pos = 0
    for linha in texto.split("\n"):
        barras = 0
        for i, ch in enumerate(linha):
            if ch == "\\":
                barras += 1
                continue
            if ch == "%" and barras % 2 == 0:
                saida.append((pos + i, pos + len(linha)))
                break
            barras = 0
        pos += len(linha) + 1
    return saida


def _fecha(s: str, i: int, abre: str = "{", fecha: str = "}") -> int:
    """Índice do delimitador que fecha o aberto em ``s[i]``; -1 se não fechar."""
    prof = 0
    j = i
    while j < len(s):
        ch = s[j]
        if ch == "\\":
            j += 2
            continue
        if ch == abre:
            prof += 1
        elif ch == fecha:
            prof -= 1
            if prof == 0:
                return j
        j += 1
    return -1


def _pular_espacos(s: str, j: int) -> int:
    while j < len(s) and s[j] in " \t\n":
        j += 1
    return j


def _ler_args(s: str, j: int, n_chaves: int) -> tuple[list[tuple[int, int]], int]:
    """Até ``n_chaves`` grupos ``{...}`` a partir de ``j`` (pulando ``[...]`` antes de cada).

    Retorna os intervalos internos (sem as chaves) e o índice logo após o último lido.
    """
    grupos: list[tuple[int, int]] = []
    fim = j
    while len(grupos) < n_chaves:
        k = _pular_espacos(s, fim)
        while k < len(s) and s[k] == "[":
            e = _fecha(s, k, "[", "]")
            if e < 0:
                break
            k = _pular_espacos(s, e + 1)
        if k < len(s) and s[k] == "{":
            e = _fecha(s, k)
            if e < 0:
                break
            grupos.append((k + 1, e))
            fim = e + 1
        else:
            break
    return grupos, fim


def _mascarar_regex(texto: str, padrao: re.Pattern, entre_paragrafos: bool = False) -> str:
    """Branqueia os casamentos.

    Com ``entre_paragrafos=False`` (matemática), preserva os que atravessam linha em
    branco, pois um ``$`` solto não pode engolir parágrafos. Blocos literais
    (``verbatim``) passam ``True``: podem conter linhas em branco.
    """

    def sub(m: re.Match) -> str:
        trecho = m.group(0)
        if not entre_paragrafos and re.search(r"\n[ \t]*\n", trecho):
            return trecho
        return re.sub(r"[^\n]", " ", trecho)

    return padrao.sub(sub, texto)


_RE_VERBATIM = re.compile(
    r"\\begin\{(verbatim\*?|Verbatim\*?|lstlisting|minted)\}.*?\\end\{\1\}", re.S
)
_RE_VERB = re.compile(r"\\(?:verb|lstinline)\*?(?:\[[^\]\n]*\])?([^\sA-Za-z*])(.*?)\1")
_AMBIENTES_MATEMATICOS = (
    "equation|align|gather|multline|eqnarray|displaymath|math|flalign|alignat"
)
_RE_MATH = [
    re.compile(
        r"\\begin\{(" + _AMBIENTES_MATEMATICOS + r")(\*?)\}.*?\\end\{\1\2\}", re.S
    ),
    re.compile(r"(?<!\\)\$\$.*?(?<!\\)\$\$", re.S),
    re.compile(r"(?<!\\)\$(?:\\.|[^$\\\n]|\n(?![ \t]*\n))*\$", re.S),
    re.compile(r"(?<!\\)\\\(.*?(?<!\\)\\\)", re.S),
    re.compile(r"(?<!\\)\\\[.*?(?<!\\)\\\]", re.S),
]

# Comandos cujo(s) argumento(s) saem do texto: nome -> número de grupos {..}.
_COMANDOS_REMOVER = {
    "ref": 1, "eqref": 1, "autoref": 1, "cref": 1, "Cref": 1, "pageref": 1, "nameref": 1,
    "vref": 1, "label": 1, "url": 1, "nolinkurl": 1, "includegraphics": 1, "input": 1,
    "include": 1, "bibliography": 1, "bibliographystyle": 1, "addbibresource": 1,
    "vspace": 1, "hspace": 1, "setlength": 2, "index": 1, "cline": 1, "color": 1,
    "cellcolor": 1, "rowcolor": 1, "columncolor": 1, "addcontentsline": 3,
    "cite": 1, "citeonline": 1, "citep": 1, "citet": 1, "citeauthor": 1,
    "citeauthoronline": 1, "citeyear": 1, "citeyearpar": 1, "parencite": 1,
    "textcite": 1, "autocite": 1, "nocite": 1, "apud": 2, "apudonline": 2,
    "texttt": 1,  # rótulos de código ficam como estão (guia, seção 3.6)
}
# Comandos que mantêm um grupo: nome -> (grupos lidos, índice do grupo mantido).
_COMANDOS_MANTER = {
    "href": (2, 1), "textcolor": (2, 1), "multicolumn": (3, 2), "multirow": (3, 2),
    "resizebox": (3, 2), "scalebox": (2, 1), "parbox": (2, 1), "captionof": (2, 1),
}
# Argumento de coluna de ambientes tabulares.
_RE_BEGIN_TABULAR_1 = re.compile(r"\\begin\{(?:tabular|longtable|array|longtabu|minipage)\}")
_RE_BEGIN_TABULAR_2 = re.compile(r"\\begin\{(?:tabular\*|tabularx|tabulary)\}")
_RE_BEGIN_OPCIONAL = re.compile(r"\\begin\{[A-Za-z*]+\}[ \t]*(\[[^\]\n]*\])")

_AMBIENTES_TABULARES = ("tabular", "tabular*", "tabularx", "tabulary", "longtable", "longtabu", "array")
_RE_INCLUDE = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")


def _mascarar_comandos(
    texto: str,
    remover: dict[str, int],
    manter: dict[str, tuple[int, int]],
) -> str:
    nomes = sorted(set(remover) | set(manter), key=len, reverse=True)
    if not nomes:
        return texto
    rx = re.compile(r"\\(" + "|".join(re.escape(n) for n in nomes) + r")(?![A-Za-z])\*?")
    intervalos: list[tuple[int, int]] = []
    for m in rx.finditer(texto):
        nome = m.group(1)
        if nome in manter:
            n, idx = manter[nome]
            grupos, fim = _ler_args(texto, m.end(), n)
            if len(grupos) <= idx:
                intervalos.append((m.start(), fim))
                continue
            ka, kb = grupos[idx]
            intervalos.append((m.start(), ka))
            intervalos.append((kb, fim))
        else:
            _grupos, fim = _ler_args(texto, m.end(), remover[nome])
            intervalos.append((m.start(), fim))
    return _branquear(texto, intervalos)


def _mascarar_args_tabulares(texto: str) -> str:
    intervalos: list[tuple[int, int]] = []
    for rx, n in ((_RE_BEGIN_TABULAR_1, 1), (_RE_BEGIN_TABULAR_2, 2)):
        for m in rx.finditer(texto):
            _grupos, fim = _ler_args(texto, m.end(), n)
            intervalos.append((m.end(), fim))
    for m in _RE_BEGIN_OPCIONAL.finditer(texto):
        intervalos.append(m.span(1))
    return _branquear(texto, intervalos)


def _spans_ambiente(texto: str, nomes: Iterable[str]) -> list[tuple[int, int]]:
    """Conteúdo dos ambientes ``nomes`` (apenas o mais externo de cada aninhamento)."""
    rx = re.compile(r"\\(begin|end)\{(" + "|".join(re.escape(n) for n in nomes) + r")\}")
    pilhas: dict[str, list[int]] = {}
    spans: list[tuple[int, int]] = []
    for m in rx.finditer(texto):
        nome = m.group(2)
        if m.group(1) == "begin":
            pilhas.setdefault(nome, []).append(m.end())
        else:
            pilha = pilhas.get(nome)
            if pilha:
                ini = pilha.pop()
                if not pilha:
                    spans.append((ini, m.start()))
    return spans


def _spans_argumento(texto: str, rx: re.Pattern, n_chaves: int, idx: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in rx.finditer(texto):
        grupos, _fim = _ler_args(texto, m.end(), n_chaves)
        if len(grupos) > idx:
            spans.append(grupos[idx])
    return spans


_RE_TITULO = re.compile(r"\\(?:chapter|section|subsection|subsubsection|paragraph|subparagraph)(?![A-Za-z])\*?")
_RE_BEGIN_CHAPTER = re.compile(r"\\begin\{chapter\}")
_RE_CAPTION = re.compile(r"\\caption(?![A-Za-z])\*?")
_RE_FOOTNOTE = re.compile(r"\\(?:footnote|footnotetext)(?![A-Za-z])")
_RE_CAPTIONOF = re.compile(r"\\captionof(?![A-Za-z])")
_RE_ITALICO_CMD = re.compile(r"\\(?:textit|emph|textsl)(?![A-Za-z])\*?")
_RE_ITALICO_GRUPO = re.compile(r"\{\s*\\(?:itshape|it|em|sl|slshape)(?![A-Za-z])")
_RE_ASPAS = re.compile(r"``.{1,600}?''|\u201c.{1,600}?\u201d|\"[^\"\n]{1,400}\"", re.S)


@dataclass
class Arquivo:
    rel: str
    definicao: bool
    limpo: str  # sem comentários, preâmbulo e Abstract; usado nos trechos
    texto: str  # neutralizado (todas as máscaras)
    texto_path: str  # neutralizado, mas com ``\path{...}`` preservado
    includes: list[tuple[int, int, str]] = field(default_factory=list)
    spans: list[tuple[int, int, str]] = field(default_factory=list)  # (a, b, tipo)
    italicos: list[tuple[int, int]] = field(default_factory=list)
    aspas: list[tuple[int, int]] = field(default_factory=list)
    capitulos: list[tuple[int, str]] = field(default_factory=list)
    _quebras: list[int] = field(default_factory=list)
    _prosa: str | None = None
    _prosa_path: str | None = None

    def linha(self, pos: int) -> int:
        return bisect.bisect_left(self._quebras, pos) + 1

    def tipo(self, pos: int) -> str:
        melhor: tuple[int, str] | None = None
        for a, b, tipo in self.spans:
            if a <= pos < b and (melhor is None or b - a < melhor[0]):
                melhor = (b - a, tipo)
        return melhor[1] if melhor else "prosa"

    def em_italico(self, pos: int) -> bool:
        return any(a <= pos < b for a, b in self.italicos)

    def em_aspas(self, pos: int) -> bool:
        return any(a <= pos < b for a, b in self.aspas)

    def trecho(self, a: int, b: int) -> str:
        ini = max(0, a - 60)
        fim = min(len(self.limpo), max(b, a + 1) + 60)
        s = " ".join(self.limpo[ini:fim].split()).replace("`", "'")
        return s if len(s) <= 160 else s[:159] + "…"

    def _sem_nao_prosa(self, texto: str) -> str:
        alvo = [(a, b) for a, b, tipo in self.spans if tipo != "prosa"]
        return _branquear(texto, alvo)

    @property
    def prosa(self) -> str:
        if self._prosa is None:
            self._prosa = self._sem_nao_prosa(self.texto)
        return self._prosa

    @property
    def prosa_path(self) -> str:
        if self._prosa_path is None:
            self._prosa_path = self._sem_nao_prosa(self.texto_path)
        return self._prosa_path


def preparar_arquivo(caminho: Path, rel: str, definicao: bool) -> Arquivo:
    bruto = caminho.read_text(encoding="utf-8-sig")
    t = acentos_latex(bruto)
    t = _branquear(t, _intervalos_comentarios(t))
    # Preâmbulo (só em arquivos com \begin{document}) e Abstract em inglês.
    mascara: list[tuple[int, int]] = []
    doc = re.search(r"\\begin\{document\}", t)
    if doc:
        mascara.append((0, doc.end()))
    ini = ABSTRACT_INICIO.search(t)
    if ini:
        fim = ABSTRACT_FIM.search(t, ini.end())
        if fim:
            mascara.append((ini.start(), fim.end()))
    limpo = _branquear(t, mascara)

    sem_verb = _mascarar_regex(limpo, _RE_VERBATIM, entre_paragrafos=True)
    sem_verb = _mascarar_regex(sem_verb, _RE_VERB)
    sem_verb = sem_verb.replace("~", " ")
    includes = [(m.start(), m.end(), m.group(1).strip()) for m in _RE_INCLUDE.finditer(sem_verb)]

    sem_math = sem_verb
    for rx in _RE_MATH:
        sem_math = _mascarar_regex(sem_math, rx)

    spans: list[tuple[int, int, str]] = []
    for a, b in _spans_ambiente(sem_math, _AMBIENTES_TABULARES):
        spans.append((a, b, "tabela"))
    for a, b in _spans_ambiente(sem_math, ("tablenotes",)):
        spans.append((a, b, "nota"))
    for a, b in _spans_argumento(sem_math, _RE_FOOTNOTE, 1, 0):
        spans.append((a, b, "nota"))
    for a, b in _spans_argumento(sem_math, _RE_CAPTION, 1, 0):
        spans.append((a, b, "legenda"))
    for a, b in _spans_argumento(sem_math, _RE_CAPTIONOF, 2, 1):
        spans.append((a, b, "legenda"))
    titulos = _spans_argumento(sem_math, _RE_TITULO, 1, 0)
    titulos_cap = _spans_argumento(sem_math, _RE_BEGIN_CHAPTER, 1, 0)
    for a, b in titulos + titulos_cap:
        spans.append((a, b, "titulo"))

    capitulos: list[tuple[int, str]] = []
    for m in re.finditer(r"\\chapter(?![A-Za-z])\*?", sem_math):
        grupos, _fim = _ler_args(sem_math, m.end(), 1)
        titulo = sem_math[grupos[0][0] : grupos[0][1]] if grupos else ""
        capitulos.append((m.start(), " ".join(titulo.split())[:80]))
    for m in _RE_BEGIN_CHAPTER.finditer(sem_math):
        grupos, _fim = _ler_args(sem_math, m.end(), 1)
        titulo = sem_math[grupos[0][0] : grupos[0][1]] if grupos else ""
        capitulos.append((m.start(), " ".join(titulo.split())[:80]))
    capitulos.sort()

    italicos = _spans_argumento(sem_math, _RE_ITALICO_CMD, 1, 0)
    for m in _RE_ITALICO_GRUPO.finditer(sem_math):
        e = _fecha(sem_math, m.start())
        if e > 0:
            italicos.append((m.start() + 1, e))
    aspas = [m.span() for m in _RE_ASPAS.finditer(sem_math)]

    t_path = _mascarar_comandos(sem_math, _COMANDOS_REMOVER, _COMANDOS_MANTER)
    t_path = _mascarar_args_tabulares(t_path)
    texto = _mascarar_comandos(t_path, {"path": 1}, {})

    return Arquivo(
        rel=rel,
        definicao=definicao,
        limpo=limpo,
        texto=texto,
        texto_path=t_path,
        includes=includes,
        spans=spans,
        italicos=italicos,
        aspas=aspas,
        capitulos=capitulos,
        _quebras=[i for i, c in enumerate(limpo) if c == "\n"],
    )


# ---------------------------------------------------------------------------
# Árvore de inclusões
# ---------------------------------------------------------------------------
@dataclass
class Arvore:
    arquivos: dict[str, Arquivo] = field(default_factory=dict)
    # Trechos na ordem de leitura: (arquivo, início, fim) em posições de ``Arquivo.texto``.
    visitas: list[tuple[Arquivo, int, int]] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)


def resolver_include(alvo: str, dir_atual: Path, raiz_dir: Path) -> Path | None:
    for diretorio in (dir_atual, raiz_dir):
        for nome in (alvo, alvo + ".tex"):
            p = diretorio / nome
            if p.suffix.lower() == ".tex" and p.is_file():
                return p.resolve()
    return None


def _relativo(caminho: Path, base: Path) -> str:
    try:
        return caminho.relative_to(base).as_posix()
    except ValueError:
        return caminho.as_posix()


def _visitar(caminho: Path, raiz_dir: Path, base: Path, arvore: Arvore, pilha: list[str]) -> None:
    chave = str(caminho)
    if chave in pilha:
        arvore.avisos.append(f"{_relativo(caminho, base)}: inclusão circular ignorada")
        return
    arq = arvore.arquivos.get(chave)
    if arq is None:
        arq = preparar_arquivo(caminho, _relativo(caminho, base), caminho.name in ARQUIVOS_DEFINICOES)
        arvore.arquivos[chave] = arq
    pilha.append(chave)
    cursor = 0
    for ini, fim, alvo in arq.includes:
        arvore.visitas.append((arq, cursor, ini))
        filho = resolver_include(alvo, caminho.parent, raiz_dir)
        if filho is None:
            if not alvo.lower().endswith((".pgf", ".pdf", ".png", ".jpg", ".eps", ".svg")):
                arvore.avisos.append(f"{arq.rel}:{arq.linha(ini)}: inclusão não encontrada: {alvo}")
        elif filho.name in EXCLUIR_ARQUIVOS:
            arvore.avisos.append(f"{arq.rel}:{arq.linha(ini)}: inclusão ignorada: {filho.name}")
        else:
            _visitar(filho, raiz_dir, base, arvore, pilha)
        cursor = fim
    arvore.visitas.append((arq, cursor, len(arq.texto)))
    pilha.pop()


def carregar_arvore(tex_raiz: Path, base: Path | None = None) -> Arvore:
    """Lê ``tex_raiz`` e tudo o que ele inclui. ``base`` define os caminhos relativos."""
    tex_raiz = Path(tex_raiz).resolve()
    base = (Path(base) if base is not None else REPO).resolve()
    arvore = Arvore()
    _visitar(tex_raiz, tex_raiz.parent, base, arvore, [])
    return arvore


# ---------------------------------------------------------------------------
# Regras
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Achado:
    ini: int
    fim: int
    mensagem: str
    severidade: str


@dataclass(frozen=True)
class Violacao:
    regra: str
    severidade: str
    arquivo: str
    linha: int
    tipo: str
    mensagem: str
    trecho: str


@dataclass(frozen=True)
class Padrao:
    rx: re.Pattern
    mensagem: str
    severidade: str = "erro"
    excecao: Callable[[Arquivo, int, int], bool] | None = None


def _p(padrao: str, mensagem: str, sev: str = "erro", flags: int = I, exc=None) -> Padrao:
    return Padrao(re.compile(padrao, flags), mensagem, sev, exc)


def _antes(padrao: str, janela: int = 80) -> Callable[[Arquivo, int, int], bool]:
    """Exceção: o texto imediatamente antes do achado casa ``padrao`` (ancorar com ``$``)."""
    rx = re.compile(padrao, I)
    return lambda arq, a, b: bool(rx.search(arq.texto[max(0, a - janela) : a]))


def _depois(padrao: str, janela: int = 80) -> Callable[[Arquivo, int, int], bool]:
    """Exceção: o texto imediatamente depois do achado casa ``padrao`` (no início)."""
    rx = re.compile(padrao, I)
    return lambda arq, a, b: bool(rx.match(arq.texto[b : b + janela]))


def _contexto(frases: Iterable[str], lado: int = 60) -> Callable[[Arquivo, int, int], bool]:
    """Exceção: alguma das ``frases`` aparece em volta do achado."""
    frases_n = [" ".join(f.lower().split()) for f in frases]

    def f(arq: Arquivo, a: int, b: int) -> bool:
        janela = " ".join(arq.texto[max(0, a - lado) : b + lado].lower().split())
        return any(fr in janela for fr in frases_n)

    return f


def _aplicar_padroes(arq: Arquivo, padroes: Iterable[Padrao], aspas: bool = True) -> list[Achado]:
    saida: list[Achado] = []
    for p in padroes:
        for m in p.rx.finditer(arq.texto):
            a, b = m.span()
            if aspas and arq.em_aspas(a):
                continue
            if p.excecao is not None and p.excecao(arq, a, b):
                continue
            saida.append(Achado(a, b, p.mensagem, p.severidade))
    return saida


# --- R01: travessão (guia, seção 11 e checklist "Sem travessões") -------------
_RE_TRAVESSAO = re.compile(r"[\u2014\u2013]|\\text(?:em|en)dash(?![A-Za-z])|-{2,}")


def _celula_isolada(t: str, a: int, b: int) -> bool:
    """``--``/``---`` como único conteúdo de uma célula de tabular (marcador de vazio)."""
    esq = t[:a]
    quebra = esq.rfind("\\\\")
    fim_esq = max(esq.rfind("&") + 1, quebra + 2 if quebra >= 0 else 0, esq.rfind("\n") + 1)
    if esq[fim_esq:].strip():
        return False
    candidatos = [i for i in (t.find("&", b), t.find("\\\\", b), t.find("\n", b)) if i >= 0]
    fim_dir = min(candidatos) if candidatos else len(t)
    return not t[b:fim_dir].strip()


def regra_r01(arq: Arquivo) -> list[Achado]:
    saida: list[Achado] = []
    t = arq.texto
    for m in _RE_TRAVESSAO.finditer(t):
        s = m.group(0)
        if s.startswith("-") and arq.tipo(m.start()) == "tabela" and _celula_isolada(t, m.start(), m.end()):
            continue  # marcador de célula vazia em tabular, não prosa
        if s == "--":
            antes = t[m.start() - 1] if m.start() > 0 else ""
            depois = t[m.end()] if m.end() < len(t) else ""
            if (antes.isdigit() or antes == "%") and depois.isdigit():
                continue  # intervalo numérico: 10--20
        saida.append(
            Achado(
                m.start(),
                m.end(),
                "travessão em prosa: reescrever com vírgula, dois-pontos ou parênteses (guia, seção 11)",
                "erro",
            )
        )
    return saida


# --- R02: formas proibidas no corpo -------------------------------------------
# Fonte: guia, seção 3.4 (tabela de termos canônicos), 3.5.3 e 3.5.5 (termos
# banidos) e 3.6 (tabela "Formas proibidas no corpo" e "Verificações mecânicas").
# Citações diretas (entre aspas), \texttt, \path e \url ficam fora.

# Taxas epidemiológicas e "taxa de aprendizado" (guia 3.6/AGENTS: "taxa" é aviso).
TAXAS_PERMITIDAS = (
    "aprendizado", "incidência", "mortalidade", "notificação", "letalidade",
    "prevalência", "natalidade", "detecção", "cura", "abandono",
)
# Guia 3.6: "falso vínculo" só se fizer parte do nome definido da política NP.
FALSO_VINCULO_PERMITIDO = ("restrição de falsos vínculos", "política NP")


def _exc_recall(arq: Arquivo, a: int, b: int) -> bool:
    if arq.em_italico(a):
        return True
    return bool(re.search(r"precision[\s-]*$", arq.texto[max(0, a - 20) : a], I))


PADROES_R02: list[Padrao] = [
    _p(r"\bzonas?\s+cinz(?:as?|ent[ao]s?)\b",
       "«zona cinzenta/cinza» é proibida: usar \\textit{grey area} (guia 3.6)"),
    _p(r"\bgray\s+(?:area|zone)s?\b|\bgrey\s+zones?\b",
       "«grey zone/gray area» fora do nome do algoritmo: usar \\textit{grey area} (guia 3.6)"),
    _p(r"\báreas?\s+cinzas?\b",
       "«área cinza» só na glosa \\textit{grey area} (área cinza) (guia 3.5.4 e 3.6)",
       exc=_antes(r"\\textit\{\s*grey\s+areas?\s*\}\s*\(\s*$")),
    _p(r"\brelacionamentos?\s+de\s+registros?\b",
       "«relacionamento de registros» só na glosa \\textit{linkage} (relacionamento de registros) (guia 3.4)",
       exc=_antes(r"\\textit\{\s*linkage\s*\}\s*\(\s*$")),
    _p(r"\brelacionamentos?\s+(?:probabil[ií]stic|determin[ií]stic)\w*",
       "usar \\textit{linkage} probabilístico/determinístico (guia 3.4)"),
    _p(r"\bvincula(?:ção|ções)\b",
       "«vinculação» é proibida: usar \\textit{linkage} (guia 3.4 e 3.6)"),
    _p(r"(?<!gzcmd-)\brecord[\s-]+linkage\b|\blinkagem\b",
       "«record linkage»/«linkagem» fora de citação: usar \\textit{linkage} (guia 3.6)"),
    _p(r"\brevis[ãa]o\s+clerical\b",
       "«revisão clerical» é proibida: usar «revisão manual» (guia 3.6)"),
    _p(r"\bn[ãa]o[\s-]+v[ií]nculos?\b|\bn[ãa]o-par(?:es)?\b",
       "«não vínculo»/«não-par»: usar «não par» para a classe da referência (guia 3.6)"),
    _p(r"\bfalsos?\s+v[ií]nculos?\b",
       "«falso vínculo» é proibido, salvo no nome da política NP: usar «falso positivo» (guia 3.6)",
       exc=_contexto(FALSO_VINCULO_PERMITIDO)),
    _p(r"\bprecis[ãa]o\b",
       "«precisão» só em AUC-PR, «precisão média» e «curva precisão-sensibilidade»; VPP é o nome único (guia 3.5.1 e 3.6)",
       exc=_depois(r"\s*-\s*sensibilidade|\s+m[ée]dia\b")),
    _p(r"\bflorestas?\s+aleat[óo]rias?\b",
       "«floresta aleatória» só na glosa Random Forest (floresta aleatória) (guia 3.5.3 e 3.6)",
       flags=0, exc=_antes(r"Random\s+Forests?\s*\}?\s*\(\s*$")),
    _p(r"\bFlorestas?\s+[Aa]leat[óo]rias?\b",
       "«Floresta Aleatória» em maiúsculas é proibida: usar Random Forest (guia 3.5.3)", flags=0),
    _p(r"(?<![\w-])RL(?![\w-])",
       "sigla RL é proibida: usar «regressão logística» por extenso (guia 3.6)", flags=0),
    _p(r"\bSINAN\b|\bSINASC\b",
       "grafia oficial: Sinan e Sinasc (guia 3.6, siglas)", flags=0),
    _p(r"\bSIH\s*-\s*SUS\b", "grafia oficial: SIH/SUS, com barra (guia 3.6, siglas)"),
    _p(r"\bpar(?:es)?\s+duvidos[oa]s?\b",
       "«pares duvidosos» só ao relatar terceiros: usar \\textit{grey area} (guia 3.4 e 3.6)", "aviso"),
    _p(r"\bpadr[ãa]o[\s-]+ouro\b",
       "«padrão-ouro» só para estudos de terceiros: usar «referência operacional» (guia 3.4 e 3.6)", "aviso"),
    _p(r"\bgold[\s-]+standard\b",
       "«gold standard» é proibido: usar «referência operacional» (guia 3.4)"),
    _p(r"\btaxas?\b",
       "«taxa»: usar proporção, fração ou o nome da métrica; taxas epidemiológicas e de aprendizado são admitidas (guia 3.1/3.6)",
       "aviso",
       exc=_depois(r"\s+de\s+(?:" + "|".join(re.escape(x) for x in TAXAS_PERMITIDAS) + r")\b")),
    _p(r"\brecall\b",
       "«recall»: usar «sensibilidade»; só em \\textit{} ou em precision-recall (guia 3.5.1 e 3.5.5)",
       exc=_exc_recall),
    _p(r"\brevoca(?:ção|ções)\b", "«revocação» é banida: usar «sensibilidade» (guia 3.5.5)"),
    _p(r"\bcalibra(?:ção|ções)\s+sigmoida(?:l|is)\b|\bcalibra(?:ção|ções)\s+de\s+Platt\b",
       "usar \\textit{Platt scaling} (guia 3.5.2 e 3.5.5)"),
    _p(r"\bidentificador(?:es)?\s+únicos?\b", "usar «identificador unívoco» (guia 3.4)"),
    _p(r"\bbancos?\s+de\s+dados\b", "usar «bases de dados» (guia 3.4 e 3.6)"),
    _p(r"\bpar(?:es)?\s+incertos?\b", "«pares incertos» é proibido: usar «potenciais pares» ou \\textit{grey area} (guia 3.4)"),
    _p(r"\bPPV\b|\bpositive\s+predictive\s+value\b",
       "usar «valor preditivo positivo (VPP)» (guia 3.5.1 e 3.6)", flags=re.IGNORECASE),
    _p(r"\barcabou[çc]os?\b", "usar \\textit{framework} (guia 3.5.5)"),
    _p(r"\bmotor\s+de\s+perda\b|\bperdas?\s+esperadas?\b|\bexpected-loss\b",
       "usar \\textit{expected loss}/\\textit{expected loss engine} (guia 3.5.5)"),
    _p(r"\b(?:non-?match|NON-?MATCH|Non-?match|match|Match|MATCH)(?:es)?\b",
       "«match»/«nonmatch» só em \\texttt{MATCH}/\\texttt{NONMATCH} (guia 3.6)", flags=0),
    _p(r"\b(?:cut-?off|threshold)s?\b", "usar «limiar» ou «ponto de corte» (guia 3.5.1 e 3.6)"),
    _p(r"\bfalse\s+(?:positive|negative)s?\b", "usar «falso positivo»/«falso negativo» (guia 3.6)"),
]


def regra_r02(arq: Arquivo) -> list[Achado]:
    return _aplicar_padroes(arq, PADROES_R02)


# --- R04: "linkage" fora de \textit{} -----------------------------------------
NOMES_PROPRIOS_LINKAGE = (
    "Laboratório de Linkage e Análise de Dados Populacionais",
    "Laboratório de Linkage de Dados Populacionais",
)
_RE_LINKAGE_SOLTO = re.compile(r"(?<![\w\-_/.])linkage(?![\w\-/])", I)


def regra_r04(arq: Arquivo) -> list[Achado]:
    saida: list[Achado] = []
    nomes = [" ".join(n.lower().split()) for n in NOMES_PROPRIOS_LINKAGE]
    for m in _RE_LINKAGE_SOLTO.finditer(arq.texto):
        a, b = m.span()
        if arq.em_italico(a) or arq.em_aspas(a):
            continue
        janela = " ".join(arq.texto[max(0, a - 70) : b + 70].lower().split())
        if any(n in janela for n in nomes):
            continue
        saida.append(
            Achado(a, b, "«linkage» deve ficar em itálico: \\textit{linkage} (guia 3.4)", "erro")
        )
    return saida


# --- R05: primeira pessoa (guia, seção 11 e checklist "Impessoalidade") --------
PRIMEIRA_PESSOA_VERBOS = (
    "fizemos", "realizamos", "propomos", "observamos", "utilizamos", "adotamos",
    "aplicamos", "avaliamos", "verificamos", "analisamos", "encontramos", "obtivemos",
    "consideramos", "testamos", "selecionamos", "definimos", "calculamos", "construímos",
    "treinamos", "usamos", "implementamos", "elaboramos", "identificamos", "comparamos",
    "executamos", "descrevemos", "apresentamos", "mostramos", "demonstramos",
    "acreditamos", "entendemos", "buscamos", "optamos", "escolhemos", "tivemos",
    "pudemos", "vimos", "tomamos", "estimamos", "temos", "podemos", "devemos", "vamos",
    "estamos", "queremos", "precisamos", "sabemos", "notamos", "percebemos",
    "concluímos", "assumimos", "partimos", "chegamos", "recomendamos", "sugerimos",
    "esperamos", "ressaltamos", "destacamos", "reconhecemos", "reportamos", "registramos",
    "descartamos", "computamos", "medimos", "excluímos", "incluímos", "decidimos",
    "refizemos", "rodamos", "geramos", "criamos", "discutimos", "relatamos",
)
PADROES_R05: list[Padrao] = [
    _p(r"\b(?:eu|nós|nosso|nossa|nossos|nossas|conosco)\b",
       "primeira pessoa: a autora é impessoal (guia, seção 11)",
       exc=_depois(r"\s+cr[ií]ticos?\b")),  # "nós críticos" é vocabulário da área (guia 3.4)
    _p(r"\b(?:" + "|".join(PRIMEIRA_PESSOA_VERBOS) + r")\b",
       "primeira pessoa do plural: usar voz passiva/impessoal (guia, seção 11)"),
]


def regra_r05(arq: Arquivo) -> list[Achado]:
    return _aplicar_padroes(arq, PADROES_R05)


# --- R06: ponto decimal em prosa ------------------------------------------------
_RE_DECIMAL = re.compile(r"(?<![\w.,/\\])\d+\.\d{1,2}(?![\w/]|\.\d)")
# Versões, nomes de modelos e referências internas que usam ponto (guia, Seção
# "Nomes de modelos e versões"; AGENTS.md seção 1).
PADROES_VERSAO_MODELO = [
    re.compile(p, I)
    for p in (
        r"\b(?:Claude\s+)?(?:Haiku|Sonnet|Opus)\s+\d+(?:\.\d+)+",
        r"\bK\d+(?:\.\d+)+",
        r"\b(?:Kimi|DeepSeek|GLM|Qwen\d*|MiniMax|GPT|Gemini|Llama|Mistral|Grok|Gemma|Phi)[-\s]*[A-Za-z]?\d+(?:\.\d+)+",
        r"\bv\d+(?:\.\d+)+",
        r"\bvers(?:ão|ões)\s+v?\d+(?:\.\d+)+",
        r"\b[A-Za-z][A-Za-z0-9]*-\d+(?:\.\d+)+",
        r"\b(?:Python|Java|Node|XGBoost|LightGBM|CatBoost|pandas|numpy|scikit-learn|Windows|Ubuntu|LaTeX)\s+v?\d+(?:\.\d+)+",
        r"\b(?:Se(?:ção|ções)|Cap(?:ítulos?|s?\.)|Tabelas?|Tab\.|Figuras?|Fig\.|Quadros?|"
        r"Ap(?:êndices?)|Equa(?:ção|ções)|Eq\.|Algoritmos?|Teoremas?|Proposi(?:ção|ções)|"
        r"Corol[áa]rios?|Defini(?:ção|ções)|Lemas?|Anexos?|\u00a7)\s*\d+(?:\.\d+)+",
    )
]


def regra_r06(arq: Arquivo) -> list[Achado]:
    t = arq.texto
    permitidos: list[tuple[int, int]] = []
    for rx in PADROES_VERSAO_MODELO:
        permitidos.extend(m.span() for m in rx.finditer(t))
    saida: list[Achado] = []
    for m in _RE_DECIMAL.finditer(t):
        a, b = m.span()
        if any(pa <= a and b <= pb for pa, pb in permitidos):
            continue
        saida.append(
            Achado(a, b, "ponto decimal: usar vírgula (ex.: 0,95); milhar com ponto (guia, AGENTS seção 1)", "erro")
        )
    return saida


# --- R07: vícios de LLM e adjetivos promocionais (aviso) -----------------------
def _sem_citacao_imediata(arq: Arquivo, a: int, b: int) -> bool:
    """Exceção da autoridade vaga: há \\cite no restante da frase (guia 12.8)."""
    resto = arq.limpo[b : b + 300]
    fim = re.search(r"\.\s", resto)
    return "\\cite" in (resto[: fim.start()] if fim else resto)


def _vicios(frases: Iterable[str], mensagem: str, exc=None) -> list[Padrao]:
    saida = []
    for f in frases:
        regex = r"\b(?:" + f.replace(" ", r"\s+") + r")\b"
        saida.append(Padrao(re.compile(regex, I), mensagem, "aviso", exc))
    return saida


PADROES_R07: list[Padrao] = (
    # Guia 12.2: conectivos de enchimento sob vigilância.
    _vicios(
        ("Além disso", "Dito isso", "Por outro lado", "Vale notar", "De modo geral", "Em geral",
         "É importante (?:destacar|ressaltar)", "Cabe destacar que", "Convém mencionar que"),
        "conectivo de enchimento (guia 12.2)")
    # Guia 12.1: fechamentos formulares.
    + _vicios(("Em resumo", "Em síntese", "Assim sendo", "No fim das contas"),
              "fechamento formular (guia 12.1)")
    # Guia 12.7: frases de enchimento e cautela desnecessária.
    + _vicios(
        ("É importante considerar que", "É (?:crucial|fundamental) (?:notar|observar) que",
         "Vale (?:lembrar|ressaltar|destacar) que", "Cabe (?:ressaltar|destacar|mencionar) que",
         "Convém (?:notar|observar) que", "Não se pode deixar de mencionar",
         "É interessante (?:notar|observar)", "É importante notar"),
        "frase de enchimento (guia 12.7)")
    # Guia 12.8: autoridade vaga, salvo com \cite imediato.
    + _vicios(
        ("(?:Pesquisas|Estudos) (?:indicam|mostram|sugerem|demonstram) que",
         "A literatura (?:aponta|demonstra|reconhece) que",
         "(?:Especialistas|Autores) (?:afirmam|concordam) que",
         "É amplamente (?:reconhecido|aceito) que", "Há evidências de que",
         "Há consenso na literatura"),
        "autoridade vaga sem citação imediata (guia 12.8)", _sem_citacao_imediata)
    # Guia 12.9: linguagem processual de agente.
    + _vicios(
        ("(?:Vamos|Vou) (?:analisar|examinar|considerar|decompor)", "O próximo passo é",
         "A próxima etapa é", "Aqui está", "Segue abaixo", "A seguir apresentamos"),
        "linguagem processual de agente (guia 12.9)")
    # Guia 12.4 e 12.10: simetria mecânica e pares redundantes.
    + _vicios(("Por um lado", "amplo e abrangente", "relevante e significativo",
               "rigoroso e sistemático"),
              "paralelismo mecânico ou par redundante (guia 12.4 e 12.10)")
    # Guia 12.3: hedging duplo.
    + _vicios(("pode potencialmente",), "hedging duplo (guia 12.3)")
    # Guia 11: antipadrões.
    + _vicios(("isso prova que", "é certo que", "game-changer", "feature engineering"),
              "antipadrão (guia 11)")
    # Guia 11 e 12.5: adjetivos promocionais e vocabulário genérico.
    + _vicios(
        ("robust[oa]s?", "inovador(?:a|es|as)?", "poderos[oa]s?", "revolucionári[oa]s?",
         "notáve(?:l|is)", "crucial", "cruciais", "disruptiv[oa]s?", "abrangentes?",
         "de ponta", "sofisticad[oa]s?", "sem precedentes", "paradigmas?"),
        "adjetivo promocional ou genérico (guia 11 e 12.5)")
    + [Padrao(re.compile(r"(?<!\\)[!?]"),
              "exclamação ou pergunta retórica (guia 11)", "aviso", None)]
)


def regra_r07(arq: Arquivo) -> list[Achado]:
    return _aplicar_padroes(arq, PADROES_R07, aspas=False)


# --- R08: nomes oficiais dos sistemas (guia 3.6, tabela de siglas) -------------
SIGLAS_OFICIAIS = {
    "SIM": "Sistema de Informações sobre Mortalidade",
    "Sinan": "Sistema de Informação de Agravos de Notificação",
    "Sinasc": "Sistema de Informações sobre Nascidos Vivos",
    "SIH/SUS": "Sistema de Informações Hospitalares do SUS",
    "SUS": "Sistema Único de Saúde",
}
_INFO = r"Sistemas?\s+de\s+Informa(?:ção|ções)"
_EXPANSOES_R08 = [
    ("SIM", re.compile(_INFO + r"\s+(?:de|sobre|da|em)\s+(?:Mortalidade|[ÓO]bitos?)", I)),
    ("Sinan", re.compile(_INFO + r"\s+(?:de|sobre)\s+Agravos(?:\s+de)?\s+\w+", I)),
    ("Sinasc", re.compile(_INFO + r"\s+(?:de|sobre|dos|das|em)\s+(?:Nascidos?\s+Vivos?|Nascimentos?)", I)),
    ("SIH/SUS", re.compile(
        _INFO + r"\s+Hospitalar(?:es)?(?:\s+(?:do|no|de|dos)\s+(?:SUS|Sistema\s+Único\s+de\s+Saúde))?", I)),
    ("SUS", re.compile(r"Sistema\s+[ÚU]nico\s+de\s+Sa[úu]de", I)),
]


def regra_r08(arq: Arquivo) -> list[Achado]:
    saida: list[Achado] = []
    for sigla, rx in _EXPANSOES_R08:
        oficial = SIGLAS_OFICIAIS[sigla]
        for m in rx.finditer(arq.texto):
            achado = " ".join(m.group(0).split())
            if achado != oficial:
                saida.append(
                    Achado(
                        m.start(),
                        m.end(),
                        f"expansão de {sigla} fora da forma oficial «{oficial}» (guia 3.6): «{achado}»",
                        "erro",
                    )
                )
    return saida


# --- R09 e R10: parágrafos ------------------------------------------------------
_RE_ESTRUTURAL = re.compile(
    r"^\s*\\(item|begin|end|chapter|section|subsection|subsubsection|paragraph|subparagraph|"
    r"caption|newpage|clearpage|appendix)(?![A-Za-z])"
)


def paragrafos(arq: Arquivo) -> list[tuple[int, int]]:
    """Intervalos dos parágrafos de prosa (linha em branco ou comando estrutural separa)."""
    res: list[tuple[int, int]] = []
    ini: int | None = None
    fim = 0
    pos = 0
    for linha in arq.prosa.split("\n"):
        inicio_linha = pos
        pos += len(linha) + 1
        if not linha.strip():
            if ini is not None:
                res.append((ini, fim))
                ini = None
            continue
        recuo = len(linha) - len(linha.lstrip())
        estrutural = _RE_ESTRUTURAL.match(linha)
        if estrutural:
            if ini is not None:
                res.append((ini, fim))
                ini = None
            if estrutural.group(1) == "item":
                ini = inicio_linha + recuo
                fim = inicio_linha + len(linha)
            else:
                res.append((inicio_linha + recuo, inicio_linha + len(linha)))
            continue
        if ini is None:
            ini = inicio_linha + recuo
        fim = inicio_linha + len(linha)
    if ini is not None:
        res.append((ini, fim))
    return res


def contar_palavras(trecho: str) -> int:
    sem_comandos = re.sub(r"\\[A-Za-z]+\*?", " ", trecho)
    return len(re.findall(r"[^\W_]+(?:[-'’][^\W_]+)*", sem_comandos))


def regra_r09(arq: Arquivo) -> list[Achado]:
    saida: list[Achado] = []
    for a, b in paragrafos(arq):
        n = contar_palavras(arq.prosa[a:b])
        if n > PALAVRAS_MAX_PARAGRAFO:
            saida.append(
                Achado(a, a + 1, f"parágrafo com {n} palavras (limite {PALAVRAS_MAX_PARAGRAFO})", "aviso")
            )
    return saida


def regra_r10(arq: Arquivo) -> list[Achado]:
    saida: list[Achado] = []
    for a, b in paragrafos(arq):
        if contar_palavras(arq.prosa[a:b]) < PALAVRAS_MIN_PROSA_CORRIDA:
            continue
        for m in re.finditer(r"\\path(?![A-Za-z])", arq.prosa_path[a:b]):
            saida.append(
                Achado(a + m.start(), a + m.end(), "\\path{...} dentro de parágrafo de prosa corrida", "aviso")
            )
    return saida


# --- R03: glosa de primeira menção por capítulo -------------------------------
_RE_M_LINKAGE = re.compile(r"\\textit\{\s*linkage\s*\}", I)
_RE_G_LINKAGE = re.compile(r"\s*\(\s*relacionamento\s+de\s+registros\s*\)", I)
_RE_M_GREY = re.compile(r"\\textit\{\s*grey\s+areas?\s*\}", I)
_RE_G_GREY = re.compile(r"\s*\(\s*áreas?\s+cinzas?\s*\)", I)
_RE_M_RF = re.compile(r"(?:\\textit\{\s*)?Random\s+Forests?(?:\s*\})?")
_RE_G_RF = re.compile(r"\s*\(\s*florestas?\s+aleat[óo]rias?\s*\)", I)


def regra_r03(arvore: Arvore) -> list[Violacao]:
    saida: list[Violacao] = []
    capitulo: str | None = None
    viu_linkage = viu_grey = viu_rf = False
    for arq, ini, fim in arvore.visitas:
        # (posição, tipo do evento, título do capítulo ou casamento da menção)
        eventos: list[tuple[int, str, str | re.Match]] = []
        for pos, titulo in arq.capitulos:
            if ini <= pos < fim:
                eventos.append((pos, "capitulo", titulo))
        if not arq.definicao:
            for nome, rx in (("linkage", _RE_M_LINKAGE), ("grey", _RE_M_GREY), ("rf", _RE_M_RF)):
                for achado in rx.finditer(arq.texto, ini, fim):
                    eventos.append((achado.start(), nome, achado))
        eventos.sort(key=lambda e: e[0])
        for pos, nome, dado in eventos:
            if isinstance(dado, str):
                capitulo = dado
                viu_linkage = viu_grey = False
                continue
            m = dado
            if arq.tipo(pos) == "titulo":
                continue
            resto = arq.texto[m.end() : m.end() + 80]
            if nome == "linkage" and capitulo is not None and not viu_linkage:
                viu_linkage = True
                ok, msg = bool(_RE_G_LINKAGE.match(resto)), (
                    f"primeira menção de \\textit{{linkage}} no capítulo «{capitulo}» sem a glosa "
                    "(relacionamento de registros) (guia 3.4)")
            elif nome == "grey" and capitulo is not None and not viu_grey:
                viu_grey = True
                ok, msg = bool(_RE_G_GREY.match(resto)), (
                    f"primeira menção de \\textit{{grey area}} no capítulo «{capitulo}» sem a glosa "
                    "(área cinza) (guia 3.6)")
            elif nome == "rf" and not viu_rf:
                viu_rf = True
                ok, msg = bool(_RE_G_RF.match(resto)), (
                    "primeira menção de Random Forest na tese sem a glosa (floresta aleatória) (guia 3.5.4)")
            else:
                continue
            if not ok:
                saida.append(
                    Violacao("R03", "erro", arq.rel, arq.linha(pos), arq.tipo(pos), msg,
                             arq.trecho(pos, m.end()))
                )
    return saida


@dataclass(frozen=True)
class Regra:
    id: str
    titulo: str
    severidade: str  # padrão; cada achado pode trazer a sua
    aplica_definicoes: bool
    func: Callable | None  # por arquivo; None quando a regra é da árvore (R03)


REGRAS: dict[str, Regra] = {
    r.id: r
    for r in (
        Regra("R01", "travessão em prosa", "erro", True, regra_r01),
        Regra("R02", "formas proibidas no corpo (guia 3.4, 3.5, 3.6)", "erro", False, regra_r02),
        Regra("R03", "glosa de primeira menção por capítulo", "erro", False, None),
        Regra("R04", "linkage fora de itálico", "erro", False, regra_r04),
        Regra("R05", "primeira pessoa", "erro", False, regra_r05),
        Regra("R06", "ponto decimal em prosa", "erro", False, regra_r06),
        Regra("R07", "vícios de LLM e adjetivos promocionais (guia 11 e 12)", "aviso", False, regra_r07),
        Regra("R08", "nomes oficiais dos sistemas de informação", "erro", True, regra_r08),
        Regra("R09", f"parágrafo com mais de {PALAVRAS_MAX_PARAGRAFO} palavras", "aviso", False, regra_r09),
        Regra("R10", "\\path{...} em parágrafo de prosa corrida", "aviso", False, regra_r10),
    )
}


def lintar(arvore: Arvore, regras: Iterable[str] | None = None) -> list[Violacao]:
    """Todas as violações (ou só as de ``regras``), ordenadas por regra, arquivo e linha."""
    ativas = set(regras) if regras is not None else set(REGRAS)
    saida: list[Violacao] = []
    for arq in arvore.arquivos.values():
        for regra in REGRAS.values():
            if regra.func is None or regra.id not in ativas:
                continue
            if arq.definicao and not regra.aplica_definicoes:
                continue
            vistos: set[tuple[int, str]] = set()
            for ach in regra.func(arq):
                chave = (ach.ini, ach.mensagem)
                if chave in vistos:
                    continue
                vistos.add(chave)
                saida.append(
                    Violacao(regra.id, ach.severidade, arq.rel, arq.linha(ach.ini), arq.tipo(ach.ini),
                             ach.mensagem, arq.trecho(ach.ini, ach.fim))
                )
    if "R03" in ativas:
        saida.extend(regra_r03(arvore))
    saida.sort(key=lambda v: (v.regra, v.arquivo, v.linha, v.mensagem))
    return saida


# ---------------------------------------------------------------------------
# Baseline que só encolhe
# ---------------------------------------------------------------------------
def contar(violacoes: Iterable[Violacao]) -> dict[str, dict[str, int]]:
    """Contagem por regra e arquivo (erros e avisos juntos)."""
    cont: dict[str, dict[str, int]] = {}
    for v in violacoes:
        por_arq = cont.setdefault(v.regra, {})
        por_arq[v.arquivo] = por_arq.get(v.arquivo, 0) + 1
    return {r: dict(sorted(cont[r].items())) for r in sorted(cont)}


def ler_baseline(caminho: Path) -> dict:
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def montar_baseline(contagens: dict[str, dict[str, int]]) -> dict:
    return {
        "versao": 1,
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "contagens": contagens,
    }


def escrever_texto(caminho: Path, texto: str) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8", newline="\n") as f:
        f.write(texto)


def escrever_baseline(caminho: Path, baseline: dict) -> None:
    escrever_texto(caminho, json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def comparar(
    contagens: dict[str, dict[str, int]], baseline: dict[str, dict[str, int]]
) -> list[tuple[str, str, int, int]]:
    """(regra, arquivo, atual, limite) onde a contagem passa do baseline (ausente = 0)."""
    excessos = []
    for regra, por_arq in contagens.items():
        for arquivo, n in por_arq.items():
            limite = baseline.get(regra, {}).get(arquivo, 0)
            if n > limite:
                excessos.append((regra, arquivo, n, limite))
    return sorted(excessos)


def atualizar_baseline(
    contagens: dict[str, dict[str, int]], baseline: dict[str, dict[str, int]]
) -> tuple[dict[str, dict[str, int]] | None, list[tuple[str, str, int, int]]]:
    """Novas contagens do baseline, só com reduções; ``(None, aumentos)`` se houver aumento."""
    aumentos = comparar(contagens, baseline)
    if aumentos:
        return None, aumentos
    novo: dict[str, dict[str, int]] = {}
    for regra in sorted(baseline):
        for arquivo in sorted(baseline[regra]):
            atual = contagens.get(regra, {}).get(arquivo, 0)
            if atual > 0:
                novo.setdefault(regra, {})[arquivo] = min(baseline[regra][arquivo], atual)
    return novo, []


# ---------------------------------------------------------------------------
# Relatório e CLI
# ---------------------------------------------------------------------------
def formatar_relatorio(
    violacoes: list[Violacao],
    arvore: Arvore,
    tex_raiz: Path,
    baseline: dict[str, dict[str, int]] | None = None,
) -> str:
    cont = contar(violacoes)
    linhas = [
        "# Relatório do linter de prosa",
        "",
        f"Gerado em {datetime.now().isoformat(timespec='seconds')} a partir de `{Path(tex_raiz).name}`; "
        f"{len(arvore.arquivos)} arquivos lidos, {len(violacoes)} violações.",
        "",
        "## Resumo por regra",
        "",
        "| Regra | Descrição | Erros | Avisos | Total | Baseline |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for rid, regra in REGRAS.items():
        erros = sum(1 for v in violacoes if v.regra == rid and v.severidade == "erro")
        avisos = sum(1 for v in violacoes if v.regra == rid and v.severidade == "aviso")
        base = "n/d" if baseline is None else str(sum(baseline.get(rid, {}).values()))
        linhas.append(f"| {rid} | {regra.titulo} | {erros} | {avisos} | {erros + avisos} | {base} |")
    linhas += ["", "## Contagem por regra e arquivo", "", "| Regra | Arquivo | Contagem | Baseline |", "|---|---|---:|---:|"]
    for rid, por_arq in cont.items():
        for arquivo, n in por_arq.items():
            base = "n/d" if baseline is None else str(baseline.get(rid, {}).get(arquivo, 0))
            linhas.append(f"| {rid} | `{arquivo}` | {n} | {base} |")
    if baseline is not None:
        excessos = comparar(cont, baseline)
        linhas += ["", "## Acima do baseline", ""]
        if excessos:
            linhas += [f"- {r} em `{a}`: {n} (baseline {lim})" for r, a, n, lim in excessos]
        else:
            linhas.append("Nenhuma.")
    if arvore.avisos:
        linhas += ["", "## Avisos de leitura", ""] + [f"- {a}" for a in arvore.avisos]
    linhas += ["", "## Violações", ""]
    for rid, regra in REGRAS.items():
        do_rid = [v for v in violacoes if v.regra == rid]
        if not do_rid:
            continue
        linhas += [f"### {rid}: {regra.titulo}", ""]
        for v in do_rid:
            linhas.append(f"- `{v.arquivo}:{v.linha}` [{v.tipo}, {v.severidade}] {v.mensagem}: `{v.trecho}`")
        linhas.append("")
    return "\n".join(linhas) + "\n"


def _imprimir_excessos(excessos: list[tuple[str, str, int, int]]) -> None:
    for regra, arquivo, n, limite in excessos:
        print(f"  {regra} em {arquivo}: {n} (baseline {limite})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tex", type=Path, default=TESE_PADRAO, help="arquivo raiz (padrão: tese.tex)")
    ap.add_argument("--base", type=Path, default=REPO, help="raiz dos caminhos relativos")
    ap.add_argument("--baseline", type=Path, default=BASELINE_PADRAO)
    ap.add_argument("--relatorio", type=Path, default=RELATORIO_PADRAO)
    grupo = ap.add_mutually_exclusive_group()
    grupo.add_argument("--gerar-baseline", action="store_true", help="recria o baseline do zero")
    grupo.add_argument("--atualizar-baseline", action="store_true", help="só diminui contagens")
    args = ap.parse_args(argv)

    arvore = carregar_arvore(args.tex, args.base)
    violacoes = lintar(arvore)
    contagens = contar(violacoes)
    baseline = ler_baseline(args.baseline)["contagens"] if args.baseline.is_file() else None

    status = 0
    if args.gerar_baseline:
        escrever_baseline(args.baseline, montar_baseline(contagens))
        baseline = contagens
        print(f"baseline gerado: {args.baseline} ({len(violacoes)} violações)")
    elif args.atualizar_baseline:
        if baseline is None:
            print(f"baseline ausente: {args.baseline}; use --gerar-baseline", file=sys.stderr)
            return 2
        novo, aumentos = atualizar_baseline(contagens, baseline)
        if novo is None:
            print("recusado: há contagens acima do baseline:", file=sys.stderr)
            _imprimir_excessos(aumentos)
            status = 1
        else:
            escrever_baseline(args.baseline, montar_baseline(novo))
            baseline = novo
            print(f"baseline atualizado: {args.baseline}")
    elif baseline is None:
        print(f"baseline ausente: {args.baseline}; use --gerar-baseline", file=sys.stderr)
        status = 2
    else:
        excessos = comparar(contagens, baseline)
        if excessos:
            print("contagens acima do baseline:")
            _imprimir_excessos(excessos)
            status = 1
        else:
            print("dentro do baseline")

    escrever_texto(args.relatorio, formatar_relatorio(violacoes, arvore, args.tex, baseline))
    por_regra = {rid: sum(1 for v in violacoes if v.regra == rid) for rid in REGRAS}
    print("violações por regra: " + ", ".join(f"{r}={n}" for r, n in por_regra.items()))
    print(f"relatório: {args.relatorio}")
    return status


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console do Windows em cp1252
    raise SystemExit(main())
