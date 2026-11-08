"""Extrai os contextos de citação da tese (etapa 1 da atestação bibliográfica).

Percorre ``tese.tex`` seguindo ``\\input`` e ``\\include`` (inclusive tabelas),
ignora comentários LaTeX, localiza os comandos de citação (``\\cite``,
``\\citeonline``, ``\\apud`` etc.) e grava, para cada par (ocorrência, chave):
sentença, parágrafo, capítulo, seção e tipo de ambiente (prosa, nota, legenda,
tabela, item).

Saídas em ``--out`` (padrão ``.local/biblio``):

* ``citacoes.jsonl``: um registro por contexto de citação.
* ``referencias.csv``: uma linha por chave (bib e citadas), separador ``;``.
* ``resumo.md``: totais e listas de conferência, em português.

Convenções:

* ``key`` é a chave do .bib, a mesma usada como identificador no cache por
  chave de ``scripts/ref_audit`` (``data/cache/abstracts/<key>.json``).
* Em ``sentence_text`` e ``paragraph_text`` a citação do registro aparece como
  ``[cit]`` e as demais citações como ``[ref]``.
* ``id`` é ``sha256(chave + "|" + sentença normalizada)[:16]``; contextos com o
  mesmo ``id`` (mesma chave na mesma sentença) viram um único registro, e os
  demais locais ficam em ``outras_ocorrencias``.

Somente biblioteca padrão, sem rede.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

CITE_MARK = "\ue000"
MAX_PARAGRAPH = 2000

CITE_NAMES = {
    "cite",
    "citeonline",
    "citep",
    "citet",
    "citeauthor",
    "citeauthoronline",
    "citeyear",
    "citeyearpar",
    "parencite",
    "textcite",
    "autocite",
}
APUD_NAMES = {"apud", "apudonline"}

TABLE_ENVS = {
    "table",
    "table*",
    "longtable",
    "tabular",
    "tabular*",
    "tabularx",
    "tabulary",
    "longtabu",
    "sidewaystable",
    "threeparttable",
}
LIST_ENVS = {"itemize", "enumerate", "description"}
BREAK_ENVS = (
    TABLE_ENVS
    | LIST_ENVS
    | {
        "figure",
        "figure*",
        "sidewaysfigure",
        "chapter",
        "quote",
        "quotation",
        "abstract",
        "equation",
        "equation*",
        "align",
        "align*",
        "center",
        "minipage",
    }
)
ENV_ARGS = {
    "tabular": 1,
    "longtable": 1,
    "longtabu": 1,
    "tabular*": 2,
    "tabularx": 2,
    "tabulary": 2,
    "minipage": 1,
    "chapter": 1,
}
SECTION_LEVELS = {"chapter": 0, "section": 1, "subsection": 2, "subsubsection": 3}
SPAN_COMMANDS = {
    "footnote": "nota",
    "footnotetext": "nota",
    "caption": "legenda",
    "captionof": "legenda",
    "subcaption": "legenda",
}
NON_TEX_SUFFIXES = {".pgf", ".pdf", ".png", ".jpg", ".jpeg", ".eps", ".svg", ".tikz"}
SKIP_DIRS = {"figures"}

REF_COMMANDS = {"ref", "autoref", "pageref", "cref", "eqref", "nameref"}
DROP_ARGS = {
    "label": 1,
    "index": 1,
    "vspace": 1,
    "hspace": 1,
    "includegraphics": 1,
    "setlength": 2,
    "resizebox": 2,
    "scalebox": 1,
    "cline": 1,
    "color": 1,
    "textcolor": 1,
    "cellcolor": 1,
    "rowcolor": 1,
    "multicolumn": 2,
    "multirow": 2,
    "href": 1,
    "hypertarget": 1,
    "bibliography": 1,
    "bibliographystyle": 1,
    "addbibresource": 1,
}

ABBREVIATIONS = {
    "p",
    "pp",
    "e.g",
    "i.e",
    "eg",
    "ie",
    "fig",
    "figs",
    "tab",
    "tabs",
    "cap",
    "caps",
    "n",
    "v",
    "vs",
    "cf",
    "ed",
    "eds",
    "vol",
    "vols",
    "dr",
    "dra",
    "sr",
    "sra",
    "prof",
    "profa",
    "ibid",
    "op",
    "cit",
    "al",
    "art",
    "arts",
    "sec",
    "eq",
    "eqs",
    "apud",
    "ex",
    "org",
    "orgs",
    "trad",
    "aprox",
    "pág",
    "págs",
    "ltda",
    "univ",
}

_CMD_RE = re.compile(r"\\([A-Za-z]+)(\*?)")
_SCAN_RE = re.compile(r"\\\\|\\([A-Za-z]+)(\*?)")
_ENVNAME_RE = re.compile(r"\s*\{([^}]*)\}")
_INCLUDE_RE = re.compile(r"\\(?:input|include)\s*\{([^}]*)\}")
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n\s*")
_WS_RE = re.compile(r"[ \t]*(?:\r?\n[ \t]*)?")
_BOUNDARY_RE = re.compile(r"([.!?]+)([\"'”’)\]]*)\s+")
_LAST_WORD_RE = re.compile(r"\w+(?:\.\w+)*$")
_BIB_RE = re.compile(
    r"\\bibliography\s*\{([^}]*)\}|\\addbibresource\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}"
)
_ENTRY_RE = re.compile(r"@\s*([A-Za-z]+)\s*([{(])")
_FIELD_NAME_RE = re.compile(r"[\s,]*([A-Za-z][\w\-:.]*)\s*=\s*")
_BARE_RE = re.compile(r"[^,\s{}\"#=]+")
_SKIP_BIB_TYPES = {"comment", "string", "preamble"}


# ---------------------------------------------------------------------------
# Auxiliares de LaTeX
# ---------------------------------------------------------------------------
def strip_comment(line: str) -> str:
    """Remove o comentário LaTeX (``%`` não escapado) de uma linha."""
    backslashes = 0
    for i, ch in enumerate(line):
        if ch == "\\":
            backslashes += 1
            continue
        if ch == "%" and backslashes % 2 == 0:
            return line[:i]
        backslashes = 0
    return line


def brace_end(s: str, i: int, honor_escapes: bool = True) -> int:
    """Índice da chave que fecha a ``{`` em ``s[i]``; -1 se não fechar."""
    depth = 0
    j = i
    n = len(s)
    while j < n:
        ch = s[j]
        if honor_escapes and ch == "\\":
            j += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return -1


def bracket_end(s: str, i: int) -> int:
    """Índice do ``]`` que fecha o ``[`` em ``s[i]``; -1 se não fechar."""
    depth = 0
    j = i + 1
    n = len(s)
    while j < n:
        ch = s[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return -1
        elif ch == "]" and depth == 0:
            return j
        j += 1
    return -1


def skip_ws(s: str, i: int) -> int:
    """Pula espaços e, no máximo, uma quebra de linha (nunca linha em branco)."""
    return _WS_RE.match(s, i).end()


def skip_blank(s: str, i: int) -> int:
    """Pula apenas espaços e tabulações."""
    while i < len(s) and s[i] in " \t":
        i += 1
    return i


def parse_command_args(
    s: str, i: int, n_braces: int
) -> tuple[list[str], list[tuple[int, int]], int] | None:
    """Lê até dois ``[..]`` e ``n_braces`` grupos ``{..}`` a partir de ``i``.

    Retorna (opcionais, [(início, fim) do conteúdo de cada grupo], índice final)
    ou None se algum grupo obrigatório faltar.
    """
    opts: list[str] = []
    j = skip_ws(s, i)
    while j < len(s) and s[j] == "[" and len(opts) < 2:
        e = bracket_end(s, j)
        if e < 0:
            return None
        opts.append(s[j + 1 : e])
        j = skip_ws(s, e + 1)
    groups: list[tuple[int, int]] = []
    for k in range(n_braces):
        if j >= len(s) or s[j] != "{":
            return None
        e = brace_end(s, j)
        if e < 0:
            return None
        groups.append((j + 1, e))
        j = e + 1
        if k < n_braces - 1:
            j = skip_ws(s, j)
    return opts, groups, j


@dataclass
class CiteCall:
    """Um comando de citação encontrado no texto."""

    start: int
    end: int
    command: str
    args: list[str]
    keys: list[str]


def split_keys(raw: str) -> list[str]:
    """Separa chaves por vírgula, ignorando vazias e argumentos de macro (#1)."""
    keys = []
    for part in raw.split(","):
        key = re.sub(r"\s+", " ", part).strip()
        if key and not re.fullmatch(r"#\d", key):
            keys.append(key)
    return keys


def parse_cite(s: str, m: re.Match[str]) -> CiteCall | None:
    """Interpreta ``m`` (``\\nome*``) como citação; None se não for."""
    name = m.group(1).lower()
    if name in CITE_NAMES:
        n_braces = 1
    elif name in APUD_NAMES:
        n_braces = 2
    else:
        return None
    parsed = parse_command_args(s, m.end(), n_braces)
    if parsed is None:
        return None
    opts, groups, end = parsed
    keys: list[str] = []
    for cs, ce in groups:
        keys.extend(split_keys(s[cs:ce]))
    if not keys:
        return None
    return CiteCall(m.start(), end, name + m.group(2), opts, keys)


# ---------------------------------------------------------------------------
# Remoção de LaTeX com mapa de posições
# ---------------------------------------------------------------------------
@dataclass
class Stripped:
    """Texto sem LaTeX e mapas para o texto de origem."""

    text: str
    start_map: list[int]
    end_map: list[int]
    cites: list[CiteCall]
    cite_pos: list[int]


class _Stripper:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.out: list[str] = []
        self.smap: list[int] = []
        self.emap: list[int] = []
        self.cites: list[CiteCall] = []
        self.cite_pos: list[int] = []

    def emit(self, chunk: str, a: int, b: int) -> None:
        for ch in chunk:
            if ch.isspace():
                if not self.out or self.out[-1] == " ":
                    continue
                ch = " "
            self.out.append(ch)
            self.smap.append(a)
            self.emap.append(b)

    def run(self) -> Stripped:
        raw = self.raw
        n = len(raw)
        i = 0
        while i < n:
            ch = raw[i]
            if ch == "\\":
                i = self.backslash(i)
            elif ch == "{" and raw.startswith("{,}", i):
                self.emit(",", i, i + 3)
                i += 3
            elif ch in "{}$":
                i += 1
            elif ch == "~":
                self.emit(" ", i, i + 1)
                i += 1
            elif ch == "&":
                self.emit(" | ", i, i + 1)
                i += 1
            elif raw.startswith("``", i) or raw.startswith("''", i):
                self.emit('"', i, i + 2)
                i += 2
            else:
                self.emit(ch, i, i + 1)
                i += 1
        return Stripped("".join(self.out), self.smap, self.emap, self.cites, self.cite_pos)

    def backslash(self, i: int) -> int:
        raw = self.raw
        n = len(raw)
        if i + 1 >= n:
            return i + 1
        m = _CMD_RE.match(raw, i)
        if m is None:
            nxt = raw[i + 1]
            if nxt in "%&_#${}":
                self.emit(nxt, i, i + 2)
            elif nxt == "\\":
                j = i + 2
                if j < n and raw[j] == "[":
                    e = bracket_end(raw, j)
                    if e >= 0:
                        j = e + 1
                self.emit(" ", i, j)
                return j
            elif nxt in ",;:! \n":
                self.emit(" ", i, i + 2)
            return i + 2
        call = parse_cite(raw, m)
        if call is not None:
            self.cites.append(call)
            self.cite_pos.append(len(self.out))
            self.emit(CITE_MARK, call.start, call.end)
            return call.end
        return self.named(m)

    def named(self, m: re.Match[str]) -> int:
        raw = self.raw
        name = m.group(1).lower()
        end = m.end()
        if name in REF_COMMANDS:
            parsed = parse_command_args(raw, end, 1)
            if parsed is None:
                return end
            self.emit("N", m.start(), parsed[2])
            return parsed[2]
        if name in SPAN_COMMANDS:
            parsed = parse_command_args(raw, end, 2 if name == "captionof" else 1)
            return parsed[2] if parsed else end
        if name in ("begin", "end"):
            nm = _ENVNAME_RE.match(raw, end)
            return nm.end() if nm else end
        if name in DROP_ARGS:
            return self.drop_groups(end, DROP_ARGS[name])
        if name in ("ldots", "dots", "textellipsis"):
            self.emit("…", m.start(), end)
            return end
        return end

    def drop_groups(self, end: int, count: int) -> int:
        raw = self.raw
        n = len(raw)
        j = end
        if j < n and raw[j] == "[":
            e = bracket_end(raw, j)
            if e >= 0:
                j = e + 1
        for _ in range(count):
            k = skip_blank(raw, j)
            if k < n and raw[k] == "{":
                e = brace_end(raw, k)
                if e < 0:
                    break
                j = e + 1
            else:
                break
        return j


def strip_latex(raw: str) -> Stripped:
    """Remove LaTeX de ``raw``; citações viram o marcador ``CITE_MARK``."""
    return _Stripper(raw).run()


def plain_text(raw: str) -> str:
    """Texto simples de um trecho curto (títulos de capítulo e seção)."""
    return strip_latex(raw).text.replace(CITE_MARK, "").strip()


# ---------------------------------------------------------------------------
# Segmentação em sentenças
# ---------------------------------------------------------------------------
def _is_abbreviation(before: str) -> bool:
    m = _LAST_WORD_RE.search(before)
    if m is None:
        return False
    word = m.group(0)
    if word.lower() in ABBREVIATIONS:
        return True
    return len(word) == 1 and word.isupper()


def _is_sentence_start(ch: str) -> bool:
    return ch.isupper() or ch.isdigit() or ch in "\"“‘'(«"


def segment_sentences(text: str) -> list[tuple[int, int]]:
    """Divide ``text`` em sentenças; retorna intervalos [a, b) sem espaços nas pontas."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _BOUNDARY_RE.finditer(text):
        nxt_idx = m.end()
        if nxt_idx >= len(text) or not _is_sentence_start(text[nxt_idx]):
            continue
        if m.group(1) == "." and _is_abbreviation(text[: m.start(1)]):
            continue
        spans.append((start, m.end(2)))
        start = nxt_idx
    spans.append((start, len(text)))
    trimmed = []
    for a, b in spans:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if b > a:
            trimmed.append((a, b))
    return trimmed


def _sentence_index(sentences: list[tuple[int, int]], pos: int) -> int | None:
    for k, (a, b) in enumerate(sentences):
        if a <= pos < b:
            return k
    before = [k for k, (a, _b) in enumerate(sentences) if a <= pos]
    if before:
        return before[-1]
    return 0 if sentences else None


def render(text: str, a: int, b: int, cite_index: dict[int, int], target: int | None) -> str:
    """Trecho ``text[a:b]`` com a citação alvo como [cit] e as demais como [ref]."""
    parts = []
    for k in range(a, b):
        ch = text[k]
        if ch == CITE_MARK:
            parts.append("[cit]" if cite_index.get(k) == target else "[ref]")
        else:
            parts.append(ch)
    return "".join(parts)


def window_text(full: str, sent_start: int, sent_len: int, limit: int = MAX_PARAGRAPH) -> str:
    """Corta ``full`` em no máximo ``limit`` caracteres, mantendo a sentença."""
    if len(full) <= limit:
        return full
    if sent_len >= limit - 2:
        return full[sent_start : sent_start + limit - 1] + "…"
    start = sent_start - (limit - sent_len) // 2
    start = max(0, min(start, len(full) - limit))
    end = start + limit
    piece = full[start:end]
    if start > 0:
        piece = "…" + piece[1:]
    if end < len(full):
        piece = piece[:-1] + "…"
    return piece


# ---------------------------------------------------------------------------
# Documento: expansão de \input e \include
# ---------------------------------------------------------------------------
def rel_path(path: Path, base: Path) -> str:
    """Caminho relativo a ``base`` com barras diretas (absoluto se estiver fora)."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


@dataclass
class Document:
    """Texto achatado da tese com mapa de posição para arquivo e linha."""

    text: str
    piece_starts: list[int]
    piece_info: list[tuple[str, int]]
    root_dir: Path
    body_start: int
    body_end: int
    warnings: list[str] = field(default_factory=list)

    def locate(self, offset: int) -> tuple[str, int]:
        idx = bisect.bisect_right(self.piece_starts, offset) - 1
        return self.piece_info[max(idx, 0)]


def resolve_include(target: str, root_dir: Path) -> Path | None:
    """Caminho do .tex alvo de um \\input; None se for figura, .pgf ou pasta ignorada."""
    name = target.strip()
    if not name:
        return None
    path = Path(name)
    if any(part in SKIP_DIRS for part in path.parts):
        return None
    suffix = path.suffix.lower()
    if suffix in NON_TEX_SUFFIXES:
        return None
    if suffix != ".tex":
        path = Path(name + ".tex")
    return root_dir / path


class _Flattener:
    def __init__(self, root_dir: Path, base: Path) -> None:
        self.root_dir = root_dir
        self.base = base
        self.parts: list[str] = []
        self.starts: list[int] = []
        self.info: list[tuple[str, int]] = []
        self.length = 0
        self.warnings: list[str] = []

    def add(self, chunk: str, file_rel: str, line: int) -> None:
        if not chunk:
            return
        self.starts.append(self.length)
        self.info.append((file_rel, line))
        self.parts.append(chunk)
        self.length += len(chunk)

    def expand(self, path: Path, stack: tuple[Path, ...]) -> None:
        rel = rel_path(path, self.base)
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            self.warnings.append(f"não foi possível ler {rel}: {exc}")
            return
        stack = stack + (path.resolve(),)
        for lineno, line in enumerate(raw.splitlines(keepends=True), start=1):
            has_newline = line.endswith("\n")
            content = line.rstrip("\r\n")
            code = strip_comment(content)
            if len(code) < len(content) and not code.strip():
                continue
            if has_newline:
                code += "\n"
            pos = 0
            for m in _INCLUDE_RE.finditer(code):
                self.add(code[pos : m.start()], rel, lineno)
                self.include(m.group(1), stack, rel, lineno)
                pos = m.end()
            self.add(code[pos:], rel, lineno)

    def include(self, target: str, stack: tuple[Path, ...], rel: str, lineno: int) -> None:
        path = resolve_include(target, self.root_dir)
        if path is None:
            return
        if path.resolve() in stack:
            self.warnings.append(f"inclusão cíclica ignorada: {target} ({rel}:{lineno})")
            return
        if not path.is_file():
            self.warnings.append(f"arquivo incluído não encontrado: {target} ({rel}:{lineno})")
            return
        self.expand(path, stack)


def default_base(tex_path: Path) -> Path:
    """Diretório de referência dos caminhos: o atual, se contiver a tese; senão o da tese."""
    cwd = Path.cwd().resolve()
    try:
        tex_path.resolve().relative_to(cwd)
    except ValueError:
        return tex_path.resolve().parent
    return cwd


def build_document(tex_path: Path, base: Path | None = None) -> Document:
    """Achata ``tex_path`` seguindo \\input e \\include recursivamente."""
    tex_path = tex_path.resolve()
    base = base if base is not None else default_base(tex_path)
    flat = _Flattener(tex_path.parent, base)
    flat.expand(tex_path, ())
    text = "".join(flat.parts)
    begin = text.find("\\begin{document}")
    body_start = begin + len("\\begin{document}") if begin >= 0 else 0
    end = text.find("\\end{document}", body_start)
    body_end = end if end >= 0 else len(text)
    return Document(
        text, flat.starts, flat.info, tex_path.parent, body_start, body_end, flat.warnings
    )


# ---------------------------------------------------------------------------
# Estrutura: capítulos, seções, ambientes, notas e legendas
# ---------------------------------------------------------------------------
@dataclass
class Span:
    """Nota de rodapé ou legenda: intervalo completo e do conteúdo."""

    start: int
    end: int
    cstart: int
    cend: int
    kind: str


@dataclass
class EnvSpan:
    start: int
    end: int
    kind: str


@dataclass
class Marker:
    pos: int
    kind: str
    title: str


@dataclass
class Structure:
    spans: list[Span] = field(default_factory=list)
    envs: list[EnvSpan] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    breaks: list[tuple[int, int]] = field(default_factory=list)


def _consume_env_args(text: str, j: int, env: str) -> tuple[int, list[tuple[int, int]]]:
    """Consome [opt] e grupos de argumento logo após \\begin{env}."""
    k = j
    if env in BREAK_ENVS and k < len(text) and text[k] == "[":
        e = bracket_end(text, k)
        if e >= 0:
            k = e + 1
    groups: list[tuple[int, int]] = []
    for _ in range(ENV_ARGS.get(env, 0)):
        k2 = skip_blank(text, k)
        if k2 < len(text) and text[k2] == "{":
            e = brace_end(text, k2)
            if e < 0:
                break
            groups.append((k2 + 1, e))
            k = e + 1
        else:
            break
    return k, groups


def scan_structure(text: str, start: int, end: int) -> Structure:
    """Varre ``text[start:end]`` e coleta marcadores, ambientes, notas e quebras."""
    st = Structure()
    stack: list[tuple[str, int]] = []
    for m in _SCAN_RE.finditer(text, start, end):
        name = m.group(1)
        if name is None:
            st.breaks.append((m.start(), m.end()))
            continue
        if name in ("item", "par", "appendix"):
            stop = m.end()
            if name == "item":
                k = skip_blank(text, stop)
                if k < len(text) and text[k] == "[":
                    e = bracket_end(text, k)
                    if e >= 0:
                        stop = e + 1
            elif name == "appendix":
                st.markers.append(Marker(m.start(), "appendix", ""))
            st.breaks.append((m.start(), stop))
        elif name in SECTION_LEVELS:
            parsed = parse_command_args(text, m.end(), 1)
            if parsed is None:
                continue
            _opts, groups, stop = parsed
            title = plain_text(text[groups[0][0] : groups[0][1]])
            st.markers.append(Marker(m.start(), name, title))
            st.breaks.append((m.start(), stop))
        elif name == "begin":
            nm = _ENVNAME_RE.match(text, m.end())
            if nm is None:
                continue
            env = nm.group(1).strip()
            stop, groups = _consume_env_args(text, nm.end(), env)
            if env == "chapter" and groups:
                st.markers.append(
                    Marker(m.start(), "chapter", plain_text(text[groups[0][0] : groups[0][1]]))
                )
            if env in BREAK_ENVS:
                st.breaks.append((m.start(), stop))
            stack.append((env, m.start()))
        elif name == "end":
            nm = _ENVNAME_RE.match(text, m.end())
            if nm is None:
                continue
            env = nm.group(1).strip()
            if env in BREAK_ENVS:
                st.breaks.append((m.start(), nm.end()))
            _close_env(st, stack, env, nm.end())
        elif name.lower() in SPAN_COMMANDS:
            lname = name.lower()
            parsed = parse_command_args(text, m.end(), 2 if lname == "captionof" else 1)
            if parsed is None:
                continue
            _opts, groups, stop = parsed
            cs, ce = groups[-1]
            st.spans.append(Span(m.start(), stop, cs, ce, SPAN_COMMANDS[lname]))
    for match in _BLANK_LINE_RE.finditer(text, start, end):
        st.breaks.append((match.start(), match.end()))
    st.breaks = merge_intervals(st.breaks)
    st.markers.sort(key=lambda mk: mk.pos)
    st.spans.sort(key=lambda sp: sp.start)
    return st


def _close_env(st: Structure, stack: list[tuple[str, int]], env: str, end: int) -> None:
    for idx in range(len(stack) - 1, -1, -1):
        if stack[idx][0] == env:
            begin = stack[idx][1]
            del stack[idx:]
            if env in TABLE_ENVS:
                st.envs.append(EnvSpan(begin, end, "tabela"))
            elif env in LIST_ENVS:
                st.envs.append(EnvSpan(begin, end, "item"))
            return


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def gaps(start: int, end: int, breaks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Intervalos de [start, end) fora das quebras."""
    out: list[tuple[int, int]] = []
    pos = start
    for bs, be in breaks:
        if be <= pos or bs >= end:
            continue
        if bs > pos:
            out.append((pos, bs))
        pos = max(pos, be)
    if pos < end:
        out.append((pos, end))
    return out


def compute_states(
    markers: list[Marker],
) -> tuple[list[int], list[tuple[str, bool, tuple[str, ...]]]]:
    """Posições dos marcadores e o estado (capítulo, apêndice, seções) após cada um."""
    positions = [mk.pos for mk in markers]
    chapter = ""
    appendix = False
    sections: list[str] = []
    states: list[tuple[str, bool, tuple[str, ...]]] = [(chapter, appendix, ())]
    for mk in markers:
        if mk.kind == "appendix":
            appendix = True
        elif mk.kind == "chapter":
            chapter = mk.title
            sections = []
        else:
            level = SECTION_LEVELS[mk.kind]
            sections = sections[: level - 1] + [mk.title]
        states.append((chapter, appendix, tuple(sections)))
    return positions, states


# ---------------------------------------------------------------------------
# Unidades de texto e registros
# ---------------------------------------------------------------------------
@dataclass
class Unit:
    start: int
    end: int
    exclusions: list[tuple[int, int]]
    kind: str | None


@dataclass
class RawUnit:
    raw: str
    raw_offsets: list[int]
    flat_offsets: list[int]

    def to_flat(self, idx: int) -> int:
        k = max(bisect.bisect_right(self.raw_offsets, idx) - 1, 0)
        return self.flat_offsets[k] + (idx - self.raw_offsets[k])


def build_raw(text: str, unit: Unit) -> RawUnit:
    """Texto bruto da unidade sem os trechos excluídos (notas e legendas aninhadas)."""
    chunks: list[str] = []
    raw_offsets: list[int] = []
    flat_offsets: list[int] = []
    length = 0
    pos = unit.start

    def add(a: int, b: int) -> None:
        nonlocal length
        chunks.append(text[a:b])
        raw_offsets.append(length)
        flat_offsets.append(a)
        length += b - a

    for es, ee in sorted(unit.exclusions):
        es = max(es, pos)
        ee = min(ee, unit.end)
        if ee <= es:
            continue
        if es > pos:
            add(pos, es)
        pos = ee
    if pos < unit.end:
        add(pos, unit.end)
    if not raw_offsets:
        raw_offsets.append(0)
        flat_offsets.append(unit.start)
    return RawUnit("".join(chunks), raw_offsets, flat_offsets)


def build_units(st: Structure, start: int, end: int) -> list[Unit]:
    units: list[Unit] = []
    for sp in st.spans:
        inner = [
            (o.start, o.end)
            for o in st.spans
            if o is not sp and o.start >= sp.cstart and o.end <= sp.cend
        ]
        units.append(Unit(sp.cstart, sp.cend, inner, sp.kind))
    for a, b in gaps(start, end, st.breaks):
        overlapping = [(o.start, o.end) for o in st.spans if o.start < b and o.end > a]
        units.append(Unit(a, b, overlapping, None))
    return units


def env_kind(off: int, file: str, envs: list[EnvSpan]) -> str:
    best: EnvSpan | None = None
    for e in envs:
        if e.start <= off < e.end and (best is None or e.start > best.start):
            best = e
    kind = best.kind if best else "prosa"
    if kind == "prosa" and "tables" in PurePosixPath(file).parts:
        kind = "tabela"
    return kind


def normalise_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def make_id(key: str, sentence: str) -> str:
    digest = hashlib.sha256(f"{key}|{normalise_sentence(sentence)}".encode()).hexdigest()
    return digest[:16]


def _unit_records(
    doc: Document,
    st: Structure,
    unit: Unit,
    marker_pos: list[int],
    states: list[tuple[str, bool, tuple[str, ...]]],
) -> list[dict]:
    ru = build_raw(doc.text, unit)
    lowered = ru.raw.lower()
    if "cite" not in lowered and "apud" not in lowered:
        return []
    sr = strip_latex(ru.raw)
    if not sr.cites:
        return []
    sentences = segment_sentences(sr.text)
    cite_index = {pos: idx for idx, pos in enumerate(sr.cite_pos)}
    records: list[dict] = []
    for idx, call in enumerate(sr.cites):
        si = _sentence_index(sentences, sr.cite_pos[idx])
        if si is None:
            continue
        a, b = sentences[si]
        raw_a = sr.end_map[a - 1] if a > 0 else 0
        raw_b = sr.end_map[b - 1]
        sentence_latex = re.sub(r"\s+", " ", ru.raw[raw_a:raw_b]).strip()
        sentence_text = render(sr.text, a, b, cite_index, idx)
        norm_text = render(sr.text, a, b, cite_index, None)
        prefix = render(sr.text, 0, a, cite_index, idx)
        full = render(sr.text, 0, len(sr.text), cite_index, idx).strip()
        lead = len(prefix) - len(prefix.lstrip())
        paragraph_text = window_text(full, max(len(prefix) - lead, 0), len(sentence_text))
        off = ru.to_flat(call.start)
        file, line = doc.locate(off)
        mk = bisect.bisect_right(marker_pos, off)
        chapter, appendix, sections = states[mk]
        kind = unit.kind or env_kind(off, file, st.envs)
        for key in call.keys:
            records.append(
                {
                    "id": make_id(key, norm_text),
                    "key": key,
                    "cite_command": call.command,
                    "args": list(call.args),
                    "file": file,
                    "line": line,
                    "chapter": chapter,
                    "apendice": appendix,
                    "section_path": list(sections),
                    "environment": kind,
                    "sentence_latex": sentence_latex,
                    "sentence_text": sentence_text,
                    "paragraph_text": paragraph_text,
                    "co_cited": [k for k in call.keys if k != key],
                    "outras_ocorrencias": [],
                    "_off": off,
                }
            )
    return records


def merge_duplicates(records: list[dict]) -> list[dict]:
    """Junta registros de mesmo ``id`` (mesma chave e sentença) em um só."""
    seen: dict[str, dict] = {}
    merged: list[dict] = []
    for rec in records:
        first = seen.get(rec["id"])
        if first is None:
            seen[rec["id"]] = rec
            merged.append(rec)
        else:
            first["outras_ocorrencias"].append(f"{rec['file']}:{rec['line']}")
    return merged


def extract_citations(doc: Document) -> list[dict]:
    """Todos os contextos de citação do documento, ordenados por arquivo, linha e chave."""
    st = scan_structure(doc.text, doc.body_start, doc.body_end)
    marker_pos, states = compute_states(st.markers)
    records: list[dict] = []
    for unit in build_units(st, doc.body_start, doc.body_end):
        records.extend(_unit_records(doc, st, unit, marker_pos, states))
    records.sort(key=lambda r: (r["file"], r["line"], r["key"], r["_off"]))
    return merge_duplicates(records)


# ---------------------------------------------------------------------------
# BibTeX
# ---------------------------------------------------------------------------
@dataclass
class BibEntry:
    key: str
    entry_type: str
    fields: dict[str, str]


def _collapse(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _entry_end(text: str, open_idx: int) -> int:
    if text[open_idx] == "{":
        return brace_end(text, open_idx, honor_escapes=False)
    depth = 0
    for j in range(open_idx + 1, len(text)):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == ")" and depth <= 0:
            return j
    return -1


def _parse_value(body: str, i: int) -> tuple[str, int]:
    """Lê um valor de campo (chaves, aspas, número ou macro, com ``#``)."""
    parts: list[str] = []
    n = len(body)
    while True:
        while i < n and body[i].isspace():
            i += 1
        if i >= n:
            break
        ch = body[i]
        if ch == "{":
            e = brace_end(body, i, honor_escapes=False)
            if e < 0:
                parts.append(body[i + 1 :])
                i = n
            else:
                parts.append(body[i + 1 : e])
                i = e + 1
        elif ch == '"':
            j = i + 1
            depth = 0
            while j < n:
                c = body[j]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                elif c == '"' and depth == 0:
                    break
                j += 1
            parts.append(body[i + 1 : j])
            i = j + 1
        else:
            bare = _BARE_RE.match(body, i)
            if bare is None:
                break
            parts.append(bare.group(0))
            i = bare.end()
        k = i
        while k < n and body[k].isspace():
            k += 1
        if k < n and body[k] == "#":
            i = k + 1
            continue
        i = k
        break
    return _collapse("".join(parts)), i


def _parse_fields(body: str, i: int) -> dict[str, str]:
    fields: dict[str, str] = {}
    while i < len(body):
        m = _FIELD_NAME_RE.match(body, i)
        if m is None:
            break
        name = m.group(1).lower()
        value, i = _parse_value(body, m.end())
        fields.setdefault(name, value)
    return fields


def parse_bibtex(text: str) -> tuple[dict[str, BibEntry], list[str]]:
    """Lê entradas BibTeX; ignora @comment, @string e @preamble."""
    entries: dict[str, BibEntry] = {}
    warnings: list[str] = []
    pos = 0
    while True:
        m = _ENTRY_RE.search(text, pos)
        if m is None:
            break
        etype = m.group(1).lower()
        open_idx = m.end() - 1
        close_idx = _entry_end(text, open_idx)
        body_end = close_idx if close_idx >= 0 else len(text)
        pos = body_end + 1
        if etype in _SKIP_BIB_TYPES:
            continue
        body = text[open_idx + 1 : body_end]
        comma = body.find(",")
        if comma < 0:
            key, fields = body.strip(), {}
        else:
            key, fields = body[:comma].strip(), _parse_fields(body, comma + 1)
        if not key:
            continue
        if key in entries:
            warnings.append(f"chave duplicada no .bib (mantida a primeira): {key}")
            continue
        entries[key] = BibEntry(key, etype, fields)
    return entries, warnings


def normalise_doi(raw: str) -> str:
    """DOI sem prefixo de URL, em minúsculas."""
    doi = raw.replace("{", "").replace("}", "").strip()
    doi = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    return doi.strip().lower()


def plain_bib(value: str) -> str:
    return _collapse(value.replace("{", "").replace("}", ""))


def split_authors(raw: str) -> list[str]:
    """Separa autores por ``and`` fora de chaves."""
    names: list[str] = []
    depth = 0
    cur = 0
    low = raw.lower()
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif depth == 0 and ch.isspace() and low.startswith("and", i + 1):
            after = i + 4
            if after >= len(raw) or raw[after].isspace():
                names.append(raw[cur:i].strip())
                cur = after
                i = after
                continue
        i += 1
    names.append(raw[cur:].strip())
    return [n for n in names if n]


def _surname(name: str) -> str:
    name = name.strip()
    if name.startswith("{") and brace_end(name, 0, honor_escapes=False) == len(name) - 1:
        return plain_bib(name)
    if "," in name:
        return plain_bib(name.split(",", 1)[0])
    tokens = name.split()
    return plain_bib(tokens[-1]) if tokens else ""


def author_label(fields: dict[str, str]) -> str:
    """Sobrenome do primeiro autor, com ``et al.`` se houver mais de um."""
    raw = fields.get("author") or fields.get("editor") or fields.get("organization") or ""
    names = split_authors(raw)
    if not names:
        return ""
    label = _surname(names[0])
    if len(names) > 1:
        label += " et al."
    return label


def venue_of(fields: dict[str, str]) -> str:
    for name in ("journal", "journaltitle", "booktitle", "publisher", "howpublished", "institution"):
        if fields.get(name):
            return plain_bib(fields[name])
    return ""


def find_bib_files(doc: Document) -> list[Path]:
    """Arquivos .bib declarados por \\bibliography ou \\addbibresource."""
    paths: list[Path] = []
    for m in _BIB_RE.finditer(doc.text):
        raw = m.group(1) if m.group(1) is not None else m.group(2)
        for name in raw.split(","):
            name = name.strip()
            if not name:
                continue
            if not name.lower().endswith(".bib"):
                name += ".bib"
            path = doc.root_dir / name
            if path not in paths:
                paths.append(path)
    return paths


def load_bib(doc: Document) -> tuple[dict[str, BibEntry], list[Path]]:
    entries: dict[str, BibEntry] = {}
    files = find_bib_files(doc)
    for path in files:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            doc.warnings.append(f"não foi possível ler o .bib {path.name}: {exc}")
            continue
        parsed, warnings = parse_bibtex(text)
        doc.warnings.extend(f"{path.name}: {w}" for w in warnings)
        for key, entry in parsed.items():
            if key in entries:
                doc.warnings.append(f"chave repetida entre .bib (mantida a primeira): {key}")
            else:
                entries[key] = entry
    return entries, files


# ---------------------------------------------------------------------------
# Saídas
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "key",
    "entry_type",
    "year",
    "author",
    "title",
    "venue",
    "doi",
    "url",
    "n_contexts",
    "files",
    "status",
]


def build_reference_rows(entries: dict[str, BibEntry], records: list[dict]) -> list[dict]:
    counts: Counter[str] = Counter()
    files: dict[str, set[str]] = {}
    for rec in records:
        counts[rec["key"]] += 1
        found = files.setdefault(rec["key"], set())
        found.add(rec["file"])
        for other in rec["outras_ocorrencias"]:
            found.add(other.rsplit(":", 1)[0])
    rows = []
    for key in sorted(set(entries) | set(counts)):
        entry = entries.get(key)
        fields = entry.fields if entry else {}
        year = fields.get("year") or fields.get("date", "")[:4]
        if entry is None:
            status = "ausente_no_bib"
        elif counts[key] > 0:
            status = "citada"
        else:
            status = "nao_citada"
        rows.append(
            {
                "key": key,
                "entry_type": entry.entry_type if entry else "",
                "year": plain_bib(year),
                "author": author_label(fields),
                "title": plain_bib(fields.get("title", "")),
                "venue": venue_of(fields),
                "doi": normalise_doi(fields["doi"]) if fields.get("doi") else "",
                "url": plain_bib(fields.get("url", "")),
                "n_contexts": counts[key],
                "files": ",".join(sorted(files.get(key, set()))),
                "status": status,
            }
        )
    return rows


def write_jsonl(records: list[dict], path: Path) -> None:
    lines = []
    for rec in records:
        public = {k: v for k, v in rec.items() if not k.startswith("_")}
        lines.append(json.dumps(public, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, delimiter=";", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    tex_label: str,
    records: list[dict],
    rows: list[dict],
    entries: dict[str, BibEntry],
    warnings: list[str],
) -> str:
    occurrences = len(records) + sum(len(r["outras_ocorrencias"]) for r in records)
    cited = {r["key"] for r in records}
    in_bib = [row for row in rows if row["status"] != "ausente_no_bib"]
    with_doi = [row for row in in_bib if row["doi"]]
    url_only = [row for row in in_bib if not row["doi"] and row["url"]]
    neither = [row for row in in_bib if not row["doi"] and not row["url"]]
    missing = [row["key"] for row in rows if row["status"] == "ausente_no_bib"]
    uncited = [row["key"] for row in rows if row["status"] == "nao_citada"]

    lines = [
        "# Resumo da extração de contextos de citação",
        "",
        f"Gerado por `scripts/citacoes_contexto.py` a partir de `{tex_label}`.",
        "",
        "## Totais",
        "",
        f"- Ocorrências de citação (chave por comando): {occurrences}",
        f"- Contextos únicos (mesma chave e sentença contam uma vez): {len(records)}",
        f"- Chaves únicas citadas: {len(cited)}",
        f"- Entradas no .bib: {len(entries)}",
        f"- Entradas com DOI: {len(with_doi)}",
        f"- Entradas apenas com URL (sem DOI): {len(url_only)}",
        f"- Entradas sem DOI e sem URL: {len(neither)}",
        f"- Chaves citadas ausentes do .bib: {len(missing)}",
        f"- Entradas do .bib nunca citadas: {len(uncited)}",
        "",
        "## Entradas por tipo",
        "",
        "| Tipo | Entradas | Citadas |",
        "| --- | ---: | ---: |",
    ]
    by_type: Counter[str] = Counter(row["entry_type"] for row in in_bib)
    cited_by_type: Counter[str] = Counter(row["entry_type"] for row in in_bib if row["status"] == "citada")
    for etype in sorted(by_type):
        lines.append(f"| {etype} | {by_type[etype]} | {cited_by_type[etype]} |")

    lines += ["", "## Contextos por capítulo", ""]
    first_offset: dict[str, int] = {}
    per_chapter: Counter[str] = Counter()
    for rec in records:
        label = rec["chapter"] or "(sem capítulo)"
        if rec["apendice"]:
            label += " (apêndice)"
        per_chapter[label] += 1
        first_offset[label] = min(first_offset.get(label, rec["_off"]), rec["_off"])
    for label in sorted(per_chapter, key=lambda lb: first_offset[lb]):
        lines.append(f"- {label}: {per_chapter[label]}")
    if not per_chapter:
        lines.append("Nenhum contexto encontrado.")

    lines += ["", "## Chaves citadas ausentes do .bib", ""]
    lines += [f"- `{k}`" for k in missing] or ["Nenhuma."]
    lines += ["", "## Entradas do .bib nunca citadas", ""]
    lines += [f"- `{k}`" for k in uncited] or ["Nenhuma."]

    lines += ["", "## 20 chaves mais citadas", "", "| Chave | Contextos |", "| --- | ---: |"]
    top = sorted((row for row in rows if row["n_contexts"] > 0), key=lambda r: (-r["n_contexts"], r["key"]))
    for row in top[:20]:
        lines.append(f"| `{row['key']}` | {row['n_contexts']} |")

    if warnings:
        lines += ["", "## Avisos", ""]
        lines += [f"- {w}" for w in warnings]
    return "\n".join(lines) + "\n"


def run(tex_path: Path, out_dir: Path, base: Path | None = None) -> dict:
    """Executa a extração completa e grava as três saídas em ``out_dir``."""
    doc = build_document(tex_path, base)
    records = extract_citations(doc)
    entries, _files = load_bib(doc)
    rows = build_reference_rows(entries, records)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, out_dir / "citacoes.jsonl")
    write_csv(rows, out_dir / "referencias.csv")
    tex_base = base if base is not None else default_base(tex_path)
    summary = build_summary(rel_path(tex_path, tex_base), records, rows, entries, doc.warnings)
    (out_dir / "resumo.md").write_text(summary, encoding="utf-8", newline="\n")
    return {"records": records, "rows": rows, "entries": entries, "warnings": doc.warnings}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extrai contextos de citação da tese.")
    parser.add_argument("--tex", default="text/latex/tese.tex", help="arquivo raiz da tese")
    parser.add_argument("--out", default=".local/biblio", help="diretório de saída")
    args = parser.parse_args(argv)
    tex_path = Path(args.tex)
    if not tex_path.is_file():
        print(f"arquivo não encontrado: {tex_path}", file=sys.stderr)
        return 2
    result = run(tex_path, Path(args.out))
    for warning in result["warnings"]:
        print(f"aviso: {warning}", file=sys.stderr)
    cited = {r["key"] for r in result["records"]}
    print(f"contextos: {len(result['records'])}; chaves citadas: {len(cited)}; saída: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
