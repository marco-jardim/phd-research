"""Renomeação segura da sigla P (arranjo sequencial) para ASR, só na camada de apresentação.

Implementa o lote 1 de ``docs/plans/PLANO_RENOMEAR_P.md``. Como "P" é uma letra só, nada aqui usa
substituição global: cada ocorrência é inventariada com seu contexto, classificada por regras de zona e por
dois modelos (Sonnet, e Opus nas divergências), aprovada pela autora num manifesto CSV e só então trocada,
localizada pelo contexto (que tem de ser único no arquivo). A verificação compara com o ``HEAD``.
Sequência de execução (tudo em ``.local/renomear_sigla/``, ignorado pelo git)::

    python scripts/renomear_sigla.py inventariar --superficies texto,geradores,testes,figuras
    python scripts/renomear_sigla.py classificar --regras
    python scripts/renomear_sigla.py classificar --llm --estimar            # não chama a API
    python scripts/renomear_sigla.py classificar --llm --modo batch         # retomável
    python scripts/renomear_sigla.py adjudicar --estimar
    python scripts/renomear_sigla.py adjudicar
    python scripts/renomear_sigla.py manifesto                              # autora marca aprovado=sim
    python scripts/renomear_sigla.py aplicar --manifesto .local/renomear_sigla/manifesto.csv --lote text/latex/estudo4_gzcmd.tex
    python scripts/renomear_sigla.py verificar --base HEAD

Subcomandos:

1. ``inventariar``: snapshot ``ancoras_<commit>_<AAAAMMDDhhmm>.json`` com toda ocorrência candidata (regex
   ``(?<![A-Za-z0-9_])P(?![A-Za-z0-9_])`` e as variantes ``P-Platt`` e ``P-PAR2-grey``), alvo ou não. Os
   deslocamentos (``offset``) valem para o texto com EOL normalizado para LF. O ``id`` é
   ``sha256(arquivo|esq80|token|dir80)[:16]`` (sufixo ``-2``, ``-3`` só se o contexto se repetir no arquivo,
   caso em que ``aplicar`` recusa o span).
2. ``classificar --regras``: reaplica as regras de zona (math, ``\\label``/``\\ref``/``\\path``/``\\cite``,
   ``\\texttt``, comentários, acentos, strings Python por ``tokenize``) ao snapshot. ``--llm``: envia TODAS as
   candidatas (contexto de ±200 caracteres) ao Claude Sonnet 5.5; ``--estimar`` só estima; ``--modo batch``
   (padrão) usa a Message Batches API, ``sync`` um pool de threads. Sem ``temperature``, com cache de prompt,
   checkpoint ``llm.jsonl`` retomável (``os.replace`` com retry). A chave vem da variável ``--chave-env``
   (padrão ``ANTHROPIC_API_KEY``, ambiente ou ``.env``) e nunca é impressa nem gravada.
3. ``adjudicar``: divergência regra x LLM, regra ambígua ou confiança < 0,9 vão ao Claude Opus 5.5
   (``--por-requisicao`` padrão 2; ``--max-tokens`` padrão 8000 em ``adjudicar`` e ``classificar --llm``).
   ``--refazer-falhas`` reenvia só os itens com erro; os de ``resposta_truncada`` voltam um por requisição.
4. ``manifesto``: ``manifesto.csv`` (uma linha por candidata, ``aprovado`` vazio) e ``manifesto_resumo.txt``
   (por arquivo e casos ambíguos). Regerar preserva ``aprovado`` das linhas cujo contexto e classe não mudaram.
   Ocorrência NÃO alvo cujo contexto contém ``⟦ ⟧ ⏎`` entra com o contexto escapado (``\\u27e6`` etc.) e
   ``aplicavel=nao`` (``aplicar`` a recusa mesmo aprovada); ocorrência ALVO nessa situação aborta o comando.
5. ``aplicar``: só linhas ``aprovado=sim`` de classe alvo; localiza o contexto, que tem de ser único; se o
   arquivo mudou desde o snapshot, relocaliza pelo contexto; recusa o arquivo inteiro se algum span falhar;
   preserva EOL e codificação; reaplicar é inofensivo (``ja_aplicado``).
6. ``verificar``: contra ``--base`` (padrão ``HEAD``), por arquivo: (a) o texto atual é o da base com
   exatamente os spans aprovados trocados (e o diff token a token os explica); (b) o resíduo de ``P`` é a
   base menos os spans; (c) o multiconjunto de números por parágrafo (``contar_numeros`` de
   ``scripts/portao_fidelidade.py``) é igual ao da base.

Códigos de saída: 0 sucesso; 2 insumo ou chave ausente; 3 execução incompleta, recusa ou falha de verificação.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import difflib
import hashlib
import io
import json
import re
import subprocess
import sys
import time
import tokenize
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.atestado import avaliar as av  # noqa: E402
from scripts.atestado.common import (  # noqa: E402
    DEFAULT_ADJ_MODEL,
    DEFAULT_MODEL,
    load_env_var,
    read_json,
    read_jsonl,
    replace_with_retry,
    utc_now_iso,
    write_json_atomic,
    write_jsonl_atomic,
)
from scripts.rotulos_apresentacao import ROTULO_BRACO, ROTULO_MATH, substituicao_token  # noqa: E402

DEFAULT_OUT = ".local/renomear_sigla"
SNAPSHOT_VERSAO = 1
RAIO_LLM = 200
RAIO_ADJ = 400
LIMIAR_CONFIANCA = 0.9
MARCA_ABRE, MARCA_FECHA, MARCA_QUEBRA = "⟦", "⟧", "⏎"

CAND_RE = re.compile(r"(?<![A-Za-z0-9_])P(?:-Platt|-PAR2-grey)?(?![A-Za-z0-9_])")

CLASSES_ALVO = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")
CLASSES_NAO_ALVO = tuple(f"N{i}" for i in range(1, 13))
AMBIGUA = "AMB"
TODAS_CLASSES = CLASSES_ALVO + CLASSES_NAO_ALVO

SUPERFICIES = ("texto", "geradores", "testes", "figuras")
GERADORES_PADRAO = (
    "tabelas_persona_editorial",
    "tabelas_s8_editorial",
    "tabelas_s8a1_editorial",
    "equidade_arranjo",
    "tabela_referencia_arranjo",
    "tabelas_arranjo_editorial",
    "figura_curva_esforco",
)
AUTO_EXCLUIDOS = ("tests/test_renomear_sigla.py",)

# zona -> classe fixa (regra determinística; nunca trocar)
ZONAS_PROTEGIDAS = {
    "label_ref": "N3",
    "caminho": "N4",
    "cite": "N9",
    "texttt": "N9",
    "comentario": "N9",
    "verbatim": "N9",
    "docstring": "N9",
    "fora_de_escopo": "N12",
}

TITULOS = {"part", "chapter", "section", "subsection", "subsubsection", "paragraph", "subparagraph"}
CMD_PROT: dict[str, tuple[str, str]] = {}
for _n in ("label", "ref", "cref", "Cref", "autoref", "pageref", "eqref", "nameref", "vref", "crefrange"):
    CMD_PROT[_n] = ("label_ref", "N3")
for _n in ("path", "url", "input", "include", "includegraphics", "href", "bibliography", "bibliographystyle",
           "usepackage", "documentclass", "graphicspath", "includeonly"):
    CMD_PROT[_n] = ("caminho", "N4")
for _n in ("cite", "citeonline", "citeauthor", "citeauthoronline", "citeyear", "citeyearpar", "citet", "citep",
           "citealp", "citealt", "textcite", "parencite", "nocite", "Cite"):
    CMD_PROT[_n] = ("cite", "N9")

MATH_ENVS = {"equation", "equation*", "align", "align*", "alignat", "alignat*", "gather", "gather*", "multline",
             "multline*", "eqnarray", "eqnarray*", "displaymath", "math", "flalign", "flalign*"}
VERB_ENVS = {"verbatim", "verbatim*", "lstlisting", "minted", "comment", "Verbatim"}
CMD_RE = re.compile(r"\\([A-Za-z]+)(\*?)")


def log(texto: str) -> None:
    print(texto, flush=True)


def _erro(texto: str) -> int:
    print(texto, file=sys.stderr)
    return 2


def hash16(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Leitura de arquivos (EOL e codificação preservados)
# ---------------------------------------------------------------------------
def ler_bruto(path: Path) -> tuple[str, str]:
    """``(texto bruto, sha256 dos bytes)``; UTF-8 sem perda (BOM e EOL mantidos)."""
    dados = path.read_bytes()
    return dados.decode("utf-8"), hashlib.sha256(dados).hexdigest()


def normalizar_eol(texto: str) -> str:
    return texto.replace("\r\n", "\n")


def escrever_bruto(path: Path, texto: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(texto.encode("utf-8"))
    replace_with_retry(tmp, path)


def posix(rel: Path | str) -> str:
    return str(rel).replace("\\", "/")


def superficie_de(rel: str) -> str:
    rel = posix(rel)
    if rel.endswith(".pgf"):
        return "figuras"
    if rel.endswith(".py"):
        return "testes" if rel.startswith("tests/") else "geradores"
    return "texto"


def fora_de_escopo(rel: str) -> bool:
    rel = posix(rel)
    nome = rel.rsplit("/", 1)[-1]
    return (
        rel.startswith(("text/slides/", "docs/plans/", "docs/support_text/"))
        or "roteiro" in nome
        or nome == "tese_rev_diff.tex"
        or nome.endswith("_diff.tex")
        or "_diff_work/" in rel
        or (rel.startswith("gzcmd/") and nome.startswith("M") and nome.endswith(".md"))
    )


# ---------------------------------------------------------------------------
# Zonas de LaTeX
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Zona:
    ini: int
    fim: int
    nome: str


def _fecha_chave(t: str, i: int) -> int | None:
    """Índice da ``}`` que fecha a ``{`` em ``t[i]``, ou ``None`` se desbalanceado."""
    prof = 0
    j = i
    n = len(t)
    while j < n:
        c = t[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            prof += 1
        elif c == "}":
            prof -= 1
            if prof == 0:
                return j
        j += 1
    return None


def _fecha_colchete(t: str, i: int) -> int | None:
    prof = 0
    j = i
    n = len(t)
    while j < n:
        c = t[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            fim = _fecha_chave(t, j)
            if fim is None:
                return None
            j = fim + 1
            continue
        if c == "[":
            prof += 1
        elif c == "]":
            prof -= 1
            if prof == 0:
                return j
        j += 1
    return None


def _fecha_dolar(t: str, i: int) -> int | None:
    """Próximo ``$`` não escapado dentro do mesmo parágrafo."""
    j = i
    n = len(t)
    while j < n:
        c = t[j]
        if c == "\\":
            j += 2
            continue
        if c == "$":
            return j
        if c == "\n" and re.match(r"\n[ \t]*\n", t[j:j + 40]):
            return None
        j += 1
    return None


def zonas_tex(t: str) -> list[Zona]:
    """Zonas de math, argumentos protegidos, títulos, rótulos de ``\\item`` e comentários."""
    zonas: list[Zona] = []
    n = len(t)
    i = 0
    while i < n:
        c = t[i]
        if c == "\\":
            if i + 1 >= n:
                break
            d = t[i + 1]
            if d in "([":
                fecho = "\\)" if d == "(" else "\\]"
                fim = t.find(fecho, i + 2)
                if fim < 0:
                    i += 2
                    continue
                zonas.append(Zona(i + 2, fim, "math"))
                i = fim + 2
                continue
            if not d.isalpha():
                i += 2
                continue
            m = CMD_RE.match(t, i)
            nome = m.group(1)
            j = m.end()
            if nome == "begin":
                m2 = re.match(r"\{([A-Za-z]+\*?)\}", t[j:j + 40])
                if not m2:
                    i = j
                    continue
                env = m2.group(1)
                corpo_ini = j + m2.end()
                if env in MATH_ENVS or env in VERB_ENVS:
                    fim = t.find("\\end{" + env + "}", corpo_ini)
                    if fim < 0:
                        i = corpo_ini
                        continue
                    zonas.append(Zona(corpo_ini, fim, "math" if env in MATH_ENVS else "verbatim"))
                    i = fim + len(env) + 6
                    continue
                i = corpo_ini
                continue
            if nome == "verb":
                if j < n and not t[j].isspace():
                    delim = t[j]
                    fim = t.find(delim, j + 1)
                    if fim > 0:
                        zonas.append(Zona(j + 1, fim, "verbatim"))
                        i = fim + 1
                        continue
                i = j
                continue
            if nome == "item":
                m3 = re.match(r"[ \t]*\[", t[j:j + 40])
                if m3:
                    ab = j + m3.end() - 1
                    fim = _fecha_colchete(t, ab)
                    if fim is not None:
                        zonas.append(Zona(ab + 1, fim, "item"))
                i = j
                continue
            if nome in TITULOS:
                k = j
                while k < n and t[k] in " \t":
                    k += 1
                if k < n and t[k] == "[":
                    fim = _fecha_colchete(t, k)
                    if fim is not None:
                        zonas.append(Zona(k + 1, fim, "titulo"))
                        k = fim + 1
                while k < n and t[k] in " \t":
                    k += 1
                if k < n and t[k] == "{":
                    fim = _fecha_chave(t, k)
                    if fim is not None:
                        zonas.append(Zona(k + 1, fim, "titulo"))
                i = j
                continue
            if nome == "hyperref":
                k = j
                if k < n and t[k] == "[":
                    fim = _fecha_colchete(t, k)
                    if fim is not None:
                        zonas.append(Zona(k + 1, fim, "label_ref"))
                        k = fim + 1
                i = k
                continue
            if nome in CMD_PROT or nome == "texttt":
                k = j
                while k < n and t[k] in " \t\n":
                    k += 1
                while k < n and t[k] == "[":
                    fim = _fecha_colchete(t, k)
                    if fim is None:
                        break
                    k = fim + 1
                    while k < n and t[k] in " \t\n":
                        k += 1
                if k < n and t[k] == "{":
                    fim = _fecha_chave(t, k)
                    if fim is not None:
                        if nome == "texttt":
                            zona = "texttt_P" if t[k + 1:fim].strip() == "P" else "texttt"
                        else:
                            zona = CMD_PROT[nome][0]
                        zonas.append(Zona(k + 1, fim, zona))
                        i = fim + 1
                        continue
                i = j
                continue
            i = j
            continue
        if c == "%":
            fim = t.find("\n", i)
            fim = n if fim < 0 else fim
            zonas.append(Zona(i + 1, fim, "comentario"))
            i = fim
            continue
        if c == "$":
            if t.startswith("$$", i):
                fim = t.find("$$", i + 2)
                if fim < 0:
                    i += 2
                    continue
                zonas.append(Zona(i + 2, fim, "math"))
                i = fim + 2
                continue
            fim = _fecha_dolar(t, i + 1)
            if fim is None:
                i += 1
                continue
            zonas.append(Zona(i + 1, fim, "math"))
            i = fim + 1
            continue
        i += 1
    return zonas


def zona_em(zonas: list[Zona], pos: int) -> str:
    """Nome da zona mais interna que cobre ``pos`` (``prosa`` se nenhuma)."""
    melhor: Zona | None = None
    for z in zonas:
        if z.ini <= pos < z.fim and (melhor is None or z.ini >= melhor.ini):
            melhor = z
    return melhor.nome if melhor else "prosa"


def zona_math_exata(zonas: list[Zona], texto: str, pos: int) -> bool:
    """O math que contém ``pos`` tem por conteúdo exatamente ``P``."""
    for z in zonas:
        if z.nome == "math" and z.ini <= pos < z.fim:
            return texto[z.ini:z.fim].strip() == "P"
    return False


# ---------------------------------------------------------------------------
# Regras de classificação
# ---------------------------------------------------------------------------
A5_RE = re.compile(r"(?:,| com) \\{1,2}texttt\{MATCH\} ao humano")


def classe_por_token(token: str, dir_: str) -> str:
    if token == "P-Platt":
        return "A2"
    if token == "P-PAR2-grey":
        return "A3"
    if A5_RE.match(dir_):
        return "A5"
    return "A1"


def regra_tex(texto: str, ini: int, fim: int, token: str, zona: str, zonas: list[Zona]) -> tuple[str, str]:
    """``(classe, nota)`` de uma candidata em LaTeX ou PGF."""
    if zona in ZONAS_PROTEGIDAS:
        return ZONAS_PROTEGIDAS[zona], f"zona {zona}"
    esq = texto[max(0, ini - 40):ini]
    dir_ = texto[fim:fim + 80]
    prev = texto[ini - 1] if ini > 0 else ""
    prox = texto[fim] if fim < len(texto) else ""
    if prev == "\\":
        return "N11", "comando \\P"
    if dir_.startswith(("\\'", "\\`", "\\^", '\\"', "\\~")) or (prox and ord(prox) > 127 and prox.isalpha()) or (
        prev and ord(prev) > 127 and prev.isalpha()
    ):
        return "N2", "letra acentuada"
    if esq.endswith("(?"):
        return "N8", "grupo nomeado de regex"
    if zona == "math":
        if zona_math_exata(zonas, texto, ini):
            return AMBIGUA, "math com conteúdo exato P"
        if re.match(r"\s*\\cup\b", dir_) or re.search(r"\\cup\s*$", esq):
            return "A4", "união em math"
        if re.match(r"\s*(?:\\left\s*)?[(\[]", dir_):
            return "N1", "P( de probabilidade em math"
        return AMBIGUA, "P em math sem forma conhecida"
    if zona == "texttt_P":
        return AMBIGUA, "\\texttt{P} exato"
    if token == "P" and dir_.startswith("="):
        return "N7", "P= de precisão"
    if token == "P" and re.search(r"percentil", texto[max(0, ini - 40):fim + 40], re.I) and re.match(r"[ ~]?\d", dir_):
        return "N6", "percentil"
    if token == "P" and (
        (prev != "" and prev in "-/_") or esq.endswith("\\_") or dir_.startswith(("/", "\\_"))
    ):
        return "N4", "identificador ou caminho em prosa"
    if token == "P" and re.match(r"-[A-Za-z0-9]", dir_):
        return AMBIGUA, "P- seguido de outro termo"
    if token == "P" and dir_.startswith(".") and (
        re.search(r"[A-ZÀ-Ý][\w'’-]+,?(?:\s|~)+$", esq)
        or re.match(r"\.(?:\s|~)*[A-Z]\.", dir_)
        or re.search(r"\b[A-Z]\.(?:\s|~)*$", esq)
    ):
        return "N11", "inicial de nome"
    if zona == "item":
        return "A6", "rótulo de \\item"
    if zona == "titulo":
        return "A7", "título de seção"
    return classe_por_token(token, dir_), "prosa, legenda ou célula"


def literais_py(src: str) -> list[tuple[int, int, bool]]:
    """Literais de string de um módulo Python: ``(ini, fim, docstring)`` em offsets de ``src``."""
    starts = [0]
    for linha in src.split("\n")[:-1]:
        starts.append(starts[-1] + len(linha) + 1)
    tipos_meio = {tokenize.STRING}
    fmid = getattr(tokenize, "FSTRING_MIDDLE", None)
    if fmid is not None:
        tipos_meio.add(fmid)
    toks = []
    try:
        for tk in tokenize.generate_tokens(io.StringIO(src).readline):
            toks.append(tk)
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise ValueError(f"tokenize falhou: {type(exc).__name__}") from exc
    ignora = {tokenize.NL, tokenize.COMMENT}
    saida = []
    for idx, tk in enumerate(toks):
        if tk.type not in tipos_meio:
            continue
        a = starts[tk.start[0] - 1] + tk.start[1]
        b = starts[tk.end[0] - 1] + tk.end[1]
        doc = False
        if tk.type == tokenize.STRING:
            prev = next((toks[k] for k in range(idx - 1, -1, -1) if toks[k].type not in ignora), None)
            prox = next((toks[k] for k in range(idx + 1, len(toks)) if toks[k].type not in ignora), None)
            doc = (prev is None or prev.type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)) and (
                prox is not None and prox.type in (tokenize.NEWLINE, tokenize.ENDMARKER)
            )
        saida.append((a, b, doc))
    return saida


_DADOS = r"(?:braco|bloco|arm|variant|grupo|group|chave|key|decisao|selec\w*|filtro|coluna|col)"
_APRES = r"(?:rotulo|label|legenda|nome|texto|titulo|caption|display|descricao|apres\w*|nota|glos\w*)"


def regra_py(src: str, ini: int, fim: int, token: str, lit_ini: int, lit_fim: int, docstring: bool) -> tuple[str, str]:
    """Classe de uma candidata dentro de um literal de string Python."""
    if docstring:
        return "N9", "docstring"
    esq_lit = src[lit_ini:ini]
    dir_lit = src[fim:lit_fim]
    if esq_lit.endswith("(?"):
        return "N8", "grupo nomeado de regex"
    if re.search(r"[-_/]$", esq_lit) or dir_lit.startswith(("/", "\\_", "\\\\_")):
        return "N4", "identificador ou caminho"
    if re.search(rf"{_DADOS}\w*\s*(?:==|!=|=|\bin\b)\s*[\[(\{{]?\s*['\"]?\s*$", esq_lit):
        return "N9", "seletor de dados dentro da string"
    m = re.match(r"^[rRbBuUfF]*('''|\"\"\"|'|\")(.*)\1$", src[lit_ini:lit_fim], re.S)
    inner = m.group(2) if m else src[lit_ini:lit_fim]
    if re.search(r"[A-Za-z_]+/[A-Za-z_]", inner) and " " not in inner.strip():
        return "N4", "string com forma de caminho"
    if inner.strip() == token:
        antes = src[max(0, lit_ini - 80):lit_ini]
        depois = src[lit_fim:lit_fim + 30]
        if re.match(r"\s*:", depois):
            return "N9", "chave de dicionário"
        if re.search(r"(?:==|!=)\s*$", antes) or re.search(
            r"\.(?:eq|ne|isin|get|startswith|contains)\(\s*[\[(\{]?\s*$", antes
        ) or (re.search(r"\[\s*$", antes) and re.match(r"\s*\]", depois)):
            return "N9", "seletor de dados"
        m2 = re.search(r"([\w\"']+)\s*[:=]\s*$", antes)
        if m2:
            nome = m2.group(1).strip("\"'")
            if nome == token:
                return classe_por_token(token, ""), "valor de rótulo cuja chave é o próprio token"
            if re.fullmatch(_DADOS + r"\w*", nome, re.I):
                return "N9", f"valor de {nome}"
            if re.fullmatch(_APRES + r"\w*", nome, re.I):
                return classe_por_token(token, ""), f"valor de {nome}"
        return AMBIGUA, "literal exato sem contexto de uso"
    resto = inner.replace(token, "", 1)
    if re.search(r"[A-Za-zÀ-ÿ]{3,}", resto) and re.search(r"\s", inner):
        return classe_por_token(token, src[fim:lit_fim]), "rótulo ou frase de apresentação"
    return AMBIGUA, "string curta sem frase"


def candidatos_texto(rel: str, texto: str) -> list[dict]:
    """Candidatas de um arquivo (texto com EOL normalizado): ``ini, fim, token, zona, classe, nota``."""
    rel = posix(rel)
    saida: list[dict] = []
    if fora_de_escopo(rel):
        for m in CAND_RE.finditer(texto):
            saida.append({"ini": m.start(), "fim": m.end(), "token": m.group(0), "zona": "fora_de_escopo",
                          "classe": "N12", "nota": "fora do escopo da renomeação"})
        return saida
    if rel.endswith(".py"):
        for a, b, doc in literais_py(texto):
            for m in CAND_RE.finditer(texto, a, b):
                token = m.group(0)
                classe, nota = regra_py(texto, m.start(), m.end(), token, a, b, doc)
                saida.append({"ini": m.start(), "fim": m.end(), "token": token,
                              "zona": "docstring" if doc else "string_py", "classe": classe, "nota": nota})
        return saida
    zonas = zonas_tex(texto)
    for m in CAND_RE.finditer(texto):
        zona = zona_em(zonas, m.start())
        classe, nota = regra_tex(texto, m.start(), m.end(), m.group(0), zona, zonas)
        saida.append({"ini": m.start(), "fim": m.end(), "token": m.group(0), "zona": zona, "classe": classe,
                      "nota": nota})
    return saida


# ---------------------------------------------------------------------------
# Parágrafos, frases e números
# ---------------------------------------------------------------------------
def spans_paragrafos(texto: str) -> list[tuple[int, int]]:
    spans = []
    pos = 0
    for m in re.finditer(r"\n[ \t]*\n", texto):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(texto)))
    return spans


def contar_numeros_texto(texto: str) -> Counter:
    from scripts.portao_fidelidade import contar_numeros

    return contar_numeros(texto)


def numeros_dict(texto: str) -> dict[str, int]:
    return {str(k): v for k, v in sorted(contar_numeros_texto(texto).items(), key=lambda kv: str(kv[0]))}


def frase_em(par: str, pos: int) -> str:
    ini = 0
    for m in re.finditer(r"[.!?]\s+", par[:pos]):
        ini = m.end()
    m = re.search(r"[.!?](?:\s|$)", par[pos:])
    fim = pos + m.end() if m else len(par)
    return " ".join(par[ini:fim].split())


def git_commit(raiz: Path) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=raiz, capture_output=True, text=True,
                           timeout=30)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "semgit"
    except (OSError, subprocess.SubprocessError):
        return "semgit"


# ---------------------------------------------------------------------------
# inventariar
# ---------------------------------------------------------------------------
def arquivos_padrao(raiz: Path, superficies: list[str]) -> list[str]:
    arqs: list[str] = []
    if "texto" in superficies:
        arqs += [posix(p.relative_to(raiz)) for p in sorted((raiz / "text/latex").glob("*.tex"))
                 if p.name != "tese_rev_diff.tex"]
    if "geradores" in superficies:
        arqs += [f"scripts/{n}.py" for n in GERADORES_PADRAO if (raiz / f"scripts/{n}.py").is_file()]
    if "testes" in superficies:
        arqs += [posix(p.relative_to(raiz)) for p in sorted((raiz / "tests").glob("*.py"))
                 if posix(p.relative_to(raiz)) not in AUTO_EXCLUIDOS]
    if "figuras" in superficies:
        arqs += [posix(p.relative_to(raiz)) for p in sorted((raiz / "text/latex/figures").glob("*.pgf"))]
    return arqs


def construir_ocorrencias(rel: str, raw: str, sha: str, commit: str, superficie: str) -> list[dict]:
    norm = normalizar_eol(raw)
    cands = candidatos_texto(rel, norm)
    spans = spans_paragrafos(norm)
    starts = [s for s, _ in spans]
    vistos: Counter = Counter()
    linhas_nl = [m.start() for m in re.finditer("\n", norm)]
    num_cache: dict[int, dict[str, int]] = {}
    saida = []
    for c in cands:
        esq80 = norm[max(0, c["ini"] - 80):c["ini"]]
        dir80 = norm[c["fim"]:c["fim"] + 80]
        base_id = hash16("|".join([rel, esq80, c["token"], dir80]))
        vistos[base_id] += 1
        ident = base_id if vistos[base_id] == 1 else f"{base_id}-{vistos[base_id]}"
        k = max(0, bisect.bisect_right(starts, c["ini"]) - 1)
        a, b = spans[k]
        par = norm[a:b]
        if k not in num_cache:
            num_cache[k] = numeros_dict(par)
        saida.append({
            "id": ident,
            "arquivo": rel,
            "superficie": superficie,
            "zona": c["zona"],
            "offset": c["ini"],
            "linha": bisect.bisect_right(linhas_nl, c["ini"] - 1) + 1,
            "token": c["token"],
            "esq80": esq80,
            "dir80": dir80,
            "hash_frase": hash16(frase_em(par, c["ini"] - a)),
            "numeros_paragrafo": num_cache[k],
            "sha256_arquivo": sha,
            "commit": commit,
            "classe_regra": c["classe"],
            "nota_regra": c["nota"],
        })
    return saida


def inventariar(raiz: Path, arquivos: list[str], commit: str) -> dict:
    ocorrencias: list[dict] = []
    meta: dict[str, dict] = {}
    for rel in arquivos:
        path = raiz / rel
        if not path.is_file():
            log(f"aviso: arquivo ausente, ignorado: {rel}")
            continue
        raw, sha = ler_bruto(path)
        sup = superficie_de(rel)
        occ = construir_ocorrencias(rel, raw, sha, commit, sup)
        meta[rel] = {"sha256": sha, "superficie": sup, "eol": "crlf" if "\r\n" in raw else "lf",
                     "candidatas": len(occ)}
        ocorrencias += occ
    return {"versao": SNAPSHOT_VERSAO, "commit": commit, "criado_em": utc_now_iso(),
            "arquivos": meta, "ocorrencias": ocorrencias}


def caminho_snapshot(out: Path, commit: str, agora: datetime | None = None) -> Path:
    agora = agora or datetime.now()
    return out / f"ancoras_{commit}_{agora.strftime('%Y%m%d%H%M')}.json"


def snapshot_mais_recente(out: Path) -> Path | None:
    achados = sorted(out.glob("ancoras_*.json"), key=lambda p: (p.stat().st_mtime, p.name))
    return achados[-1] if achados else None


def carregar_snapshot(args: argparse.Namespace) -> tuple[Path | None, dict | None]:
    out = resolver(args.raiz, args.out)
    path = Path(args.snapshot) if getattr(args, "snapshot", "") else snapshot_mais_recente(out)
    if path is None or not path.is_file():
        return None, None
    return path, json.loads(path.read_text(encoding="utf-8"))


def resolver(raiz: str | Path, valor: str | Path) -> Path:
    p = Path(valor)
    return p if p.is_absolute() else Path(raiz) / p


def resumo_classes(ocorrencias: list[dict], campo: str = "classe_regra") -> str:
    cont = Counter(o[campo] for o in ocorrencias)
    return ", ".join(f"{k}={cont[k]}" for k in sorted(cont))


def cmd_inventariar(args: argparse.Namespace) -> int:
    raiz = Path(args.raiz).resolve()
    sups = [s.strip() for s in args.superficies.split(",") if s.strip()]
    if not sups or not set(sups) <= set(SUPERFICIES):
        return _erro(f"--superficies inválido: use itens de {', '.join(SUPERFICIES)}")
    if args.arquivos:
        pedidos = [posix(a.strip()) for a in args.arquivos.split(",") if a.strip()]
        arquivos = []
        for a in pedidos:
            p = Path(a)
            rel = posix(p.resolve().relative_to(raiz)) if p.is_absolute() else a
            if superficie_de(rel) in sups:
                arquivos.append(rel)
    else:
        arquivos = arquivos_padrao(raiz, sups)
    if not arquivos:
        return _erro("nenhum arquivo selecionado")
    commit = git_commit(raiz)
    try:
        snap = inventariar(raiz, arquivos, commit)
    except (ValueError, UnicodeDecodeError) as exc:
        return _erro(f"falha ao inventariar: {exc}")
    out = resolver(raiz, args.out)
    path = caminho_snapshot(out, commit)
    write_json_atomic(path, snap)
    occ = snap["ocorrencias"]
    por_sup = Counter(o["superficie"] for o in occ)
    log(f"snapshot: {path}")
    log(f"arquivos: {len(snap['arquivos'])}; candidatas: {len(occ)} ({', '.join(f'{k}={v}' for k, v in sorted(por_sup.items()))})")
    log(f"classes pela regra: {resumo_classes(occ)}")
    return 0


# ---------------------------------------------------------------------------
# classificar --regras
# ---------------------------------------------------------------------------
def reclassificar_regras(raiz: Path, snap: dict) -> dict:
    """Reaplica as regras às ocorrências do snapshot (arquivos intactos desde o inventário)."""
    por_arq: dict[str, list[dict]] = {}
    for o in snap["ocorrencias"]:
        por_arq.setdefault(o["arquivo"], []).append(o)
    mudou = 0
    for rel, lista in por_arq.items():
        raw, sha = ler_bruto(raiz / rel)
        if sha != snap["arquivos"][rel]["sha256"]:
            raise ValueError(f"{rel} mudou desde o snapshot; rode `inventariar` de novo")
        por_off = {c["ini"]: c for c in candidatos_texto(rel, normalizar_eol(raw))}
        for o in lista:
            c = por_off.get(o["offset"])
            if c is None or c["token"] != o["token"]:
                raise ValueError(f"{rel}: ocorrência {o['id']} não reencontrada no offset do snapshot")
            if (o["classe_regra"], o["zona"]) != (c["classe"], c["zona"]):
                mudou += 1
            o["classe_regra"], o["zona"], o["nota_regra"] = c["classe"], c["zona"], c["nota"]
    snap["reclassificado_em"] = utc_now_iso()
    return {"mudaram": mudou}


def cmd_classificar_regras(args: argparse.Namespace) -> int:
    raiz = Path(args.raiz).resolve()
    path, snap = carregar_snapshot(args)
    if snap is None:
        return _erro("snapshot ausente; rode `inventariar` antes")
    try:
        res = reclassificar_regras(raiz, snap)
    except (ValueError, FileNotFoundError) as exc:
        return _erro(str(exc))
    write_json_atomic(path, snap)
    log(f"snapshot: {path}; classes alteradas: {res['mudaram']}")
    log(f"classes pela regra: {resumo_classes(snap['ocorrencias'])}")
    por_zona = Counter(o["zona"] for o in snap["ocorrencias"])
    log("zonas: " + ", ".join(f"{k}={v}" for k, v in sorted(por_zona.items())))
    return 0


# ---------------------------------------------------------------------------
# LLM: prompts, requisições, execução (sync e batch), checkpoint
# ---------------------------------------------------------------------------
DEFINICAO_CLASSES = """\
Contexto: nesta tese, "P" é a sigla do arranjo sequencial recalibrado (braço P do registro interno 8c16a41).
Na camada de apresentação (texto exibido ao leitor, legendas, células de tabela, rótulos de figura e pinos de
teste que reproduzem esse texto) o nome passa a ser ASR. Dados, identificadores, chaves de seleção, \\label,
\\ref, caminhos e tabelas históricas continuam com P. Cada ocorrência vem entre ⟦ e ⟧ dentro do contexto.

Classes ALVO (o token marcado designa o arranjo e será trocado):
A1: "P" isolado em prosa, legenda, nota ou célula de tabela, nomeando o arranjo.
A2: "P-Platt" (variante com Platt scaling).
A3: "P-PAR2-grey" (variante restrita a PAR = 2).
A4: "P" em modo matemático na união $P\\cup B1$.
A5: "P" em "P, \\texttt{MATCH} ao humano" ou "P com \\texttt{MATCH} ao humano".
A6: "P" ou "P-Platt" como rótulo de \\item[...].
A7: "P" em título de seção (\\section, \\subsection, \\paragraph).
Classes NÃO ALVO (nunca trocar):
N1: P de probabilidade em modo matemático, como P(y=1|s) ou P(M).
N2: P de palavra acentuada ou com acento LaTeX (P\\'os).
N3: argumento de \\label, \\ref, \\cref, \\pageref.
N4: caminho, nome de arquivo, URL, \\input, \\includegraphics ou identificador com P entre separadores.
N5: parte de outra sigla (PAR, PPV, PRL, NP).
N6: percentil (P10, P90).
N7: "P=" de precisão em PGF ou saída de programa.
N8: grupo nomeado de expressão regular, como (?P<nome>...).
N9: código de dados ou identificador de programa (seletor braco == "P", chave de dicionário, coluna, \\cite, \\texttt de código, comentário, docstring).
N10: letra p minúscula ou outro P que não é o arranjo.
N11: inicial de nome, abreviatura, comando \\P ou "P." de referência.
N12: slides, roteiro ou arquivo fora do escopo.
Se o token não designar o arranjo, escolha a classe N que melhor o descreva. Em dúvida real, dê confiança baixa.
"""

SISTEMA_CLASSIFICAR = (
    "Você classifica ocorrências da letra isolada P em arquivos de uma tese de doutorado (LaTeX, PGF, Python), "
    "para uma renomeação segura.\n\n" + DEFINICAO_CLASSES + "\nResponda somente com um array JSON, um objeto por "
    'ocorrência, na mesma ordem: [{"id": "...", "classe": "A1", "confianca": 0.97, "justificativa": "frase curta"}]. '
    "A confiança vai de 0 a 1. Sem texto fora do array."
)

SISTEMA_ADJUDICAR = (
    "Você é o árbitro de uma classificação dupla (regra determinística e modelo) de ocorrências da letra P em "
    "arquivos de uma tese de doutorado, para uma renomeação segura.\n\n" + DEFINICAO_CLASSES + "\nPara cada "
    "ocorrência você recebe a classe da regra, a do primeiro modelo com a justificativa e um contexto maior. "
    "Decida a classe correta, mesmo que nenhum dos dois acerte. Responda somente com um array JSON, um objeto "
    'por ocorrência, na mesma ordem: [{"id": "...", "classe": "A1", "confianca": 0.97, "justificativa": "frase '
    'curta"}]. Sem texto fora do array.'
)


def contexto_marcado(norm: str, ini: int, fim: int, raio: int) -> str:
    return norm[max(0, ini - raio):ini] + MARCA_ABRE + norm[ini:fim] + MARCA_FECHA + norm[fim:fim + raio]


def item_texto(o: dict, ctx: str, extra: str = "") -> str:
    return (
        f"### id: {o['id']}\narquivo: {o['arquivo']}; superfície: {o['superficie']}; zona: {o['zona']}\n"
        f"{extra}contexto:\n<<<\n{ctx}\n>>>\n"
    )


@dataclass
class Item:
    id: str
    texto: str


@dataclass
class Requisicao:
    custom_id: str
    ids: list[str]
    params: dict


def custom_id_de(etapa: str, ids: list[str], modelo: str) -> str:
    return f"{etapa[:4]}-{hashlib.sha256('|'.join([etapa, modelo, *ids]).encode()).hexdigest()[:24]}"


MAX_TOKENS_PADRAO = 8000


def montar_requisicoes(itens: list[Item], sistema: str, modelo: str, etapa: str, por_req: int,
                       max_tokens_item: int = 150, max_tokens: int | None = None) -> list[Requisicao]:
    """``max_tokens`` fixa o teto de saída de cada requisição; sem ele, vale ``150 * itens + 300``."""
    reqs = []
    for k in range(0, len(itens), por_req):
        bloco = itens[k:k + por_req]
        usuario = (f"Classifique as {len(bloco)} ocorrências a seguir. Responda só com o array JSON.\n\n"
                   + "\n".join(i.texto for i in bloco))
        params = {
            "model": modelo,
            "max_tokens": max_tokens or (max_tokens_item * len(bloco) + 300),
            "system": [{"type": "text", "text": sistema, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": usuario}],
        }
        ids = [i.id for i in bloco]
        reqs.append(Requisicao(custom_id_de(etapa, ids, modelo), ids, params))
    return reqs


def estimar(reqs: list[Requisicao]) -> dict:
    sist = sum(len(r.params["system"][0]["text"]) for r in reqs[:1])
    usuario = sum(len(r.params["messages"][0]["content"]) for r in reqs)
    n_itens = sum(len(r.ids) for r in reqs)
    return {
        "requisicoes": len(reqs),
        "itens": n_itens,
        "entrada_cache": int(sist / 3.5) if reqs else 0,
        "entrada_usuario": int(usuario / 3.5),
        "entrada_total_sem_cache": int((sist * len(reqs) + usuario) / 3.5),
        "saida": 70 * n_itens,
    }


class Armazem:
    """Checkpoint JSONL por item (``id``), regravado atomicamente a cada resposta."""

    def __init__(self, path: Path, sleep: Callable[[float], None] = time.sleep) -> None:
        self.path = path
        self.sleep = sleep
        self.linhas: dict[str, dict] = {r["id"]: r for r in read_jsonl(path)}

    def gravar(self, rows: list[dict]) -> None:
        for r in rows:
            self.linhas[r["id"]] = r
        write_jsonl_atomic(self.path, self.linhas.values())

    def pendentes(self, ids: list[str], refazer: bool) -> list[str]:
        return [i for i in ids if i not in self.linhas or (refazer and self.linhas[i].get("erro"))]


def linhas_da_resposta(req: Requisicao, message, modelo: str, modo: str, meta: dict) -> list[dict]:
    """Uma linha de checkpoint por id da requisição (com ``erro`` quando a resposta não serve)."""
    base = {"modelo": modelo, "modo": modo, "data": utc_now_iso(), "req": req.custom_id, **meta,
            "uso": av.usage_of(message), "classe": None, "confianca": None, "justificativa": None, "erro": None}
    if getattr(message, "stop_reason", None) == "max_tokens":
        return [{**base, "id": i, "erro": "resposta_truncada"} for i in req.ids]
    try:
        itens = av.parse_items(av.message_text(message))
    except ValueError as exc:
        return [{**base, "id": i, "erro": f"json_invalido: {exc}"} for i in req.ids]
    por_id = {str(it.get("id")): it for it in itens}
    rows = []
    for i in req.ids:
        it = por_id.get(i)
        if it is None:
            rows.append({**base, "id": i, "erro": "ausente_na_resposta"})
            continue
        classe = str(it.get("classe", "")).strip().upper()
        try:
            conf = float(it.get("confianca"))
        except (TypeError, ValueError):
            conf = None
        if classe not in TODAS_CLASSES or conf is None:
            rows.append({**base, "id": i, "erro": "classe_ou_confianca_invalida",
                         "justificativa": str(it.get("justificativa", ""))[:300]})
            continue
        rows.append({**base, "id": i, "classe": classe, "confianca": max(0.0, min(1.0, conf)),
                     "justificativa": str(it.get("justificativa", ""))[:300]})
    return rows


def executar_llm(*, itens: list[Item], sistema: str, modelo: str, client, modo: str, armazem: Armazem,
                 etapa: str, por_req: int, concorrencia: int = 2, refazer: bool = False,
                 sleep: Callable[[float], None] = time.sleep, poll_s: float = 30.0,
                 espera_max_s: float = 25 * 3600.0, estado_lotes: Path | None = None,
                 log_fn: Callable[[str], None] = log, meta: dict | None = None,
                 max_tokens: int | None = None) -> dict:
    """Envia o que falta, grava o checkpoint e devolve ``{requisicoes, pendentes, erros}``.

    Com ``refazer`` só os itens que terminaram com erro são reenviados; os truncados
    (``resposta_truncada``) voltam um por requisição, para a resposta caber no teto.
    """
    meta = meta or {}
    ids_todos = [i.id for i in itens]
    faltam = set(armazem.pendentes(ids_todos, refazer))
    truncados = {i for i in faltam if (armazem.linhas.get(i) or {}).get("erro") == "resposta_truncada"}
    todo_itens = [i for i in itens if i.id in faltam and i.id not in truncados]
    reqs = montar_requisicoes(todo_itens, sistema, modelo, etapa, por_req, max_tokens=max_tokens)
    reqs += montar_requisicoes([i for i in itens if i.id in truncados], sistema, modelo, etapa, 1,
                               max_tokens=max_tokens)
    por_cid = {r.custom_id: r for r in reqs}
    enviadas = 0
    if reqs:
        if modo == "sync":
            enviadas = _rodar_sync(reqs, client, armazem, modelo, concorrencia, sleep, log_fn, meta)
        else:
            enviadas = _rodar_batch(reqs, por_cid, client, armazem, modelo, sleep, poll_s, espera_max_s,
                                    estado_lotes, log_fn, meta)
    ainda = armazem.pendentes(ids_todos, False)
    erros = [i for i in ids_todos if i in armazem.linhas and armazem.linhas[i].get("erro")]
    return {"requisicoes": enviadas, "pendentes": len(ainda), "erros": len(erros)}


def _rodar_sync(reqs, client, armazem, modelo, concorrencia, sleep, log_fn, meta) -> int:
    def chamar(r: Requisicao):
        return av.with_retry(lambda: client.messages.create(**r.params), sleep)

    n = 0
    with ThreadPoolExecutor(max_workers=max(1, concorrencia)) as pool:
        futuros = {pool.submit(chamar, r): r for r in reqs}
        for fut in as_completed(futuros):
            r = futuros[fut]
            try:
                msg = fut.result()
            except av.ApiCallError as exc:
                log_fn(f"falha em {r.custom_id}: {exc}")
                continue
            armazem.gravar(linhas_da_resposta(r, msg, modelo, "sync", meta))
            n += 1
    return n


def _rodar_batch(reqs, por_cid, client, armazem, modelo, sleep, poll_s, espera_max_s, estado_path,
                 log_fn, meta) -> int:
    estado_path = estado_path or armazem.path.with_suffix(".lotes.json")
    estado = read_json(estado_path, {"lotes": []})
    voando = {i for lote in estado["lotes"] if not lote.get("baixado") for ids in lote["custom_ids"].values()
              for i in ids}
    novas = [r for r in reqs if not set(r.ids) <= voando]
    if any(lote.get("id") is None for lote in estado["lotes"]):
        raise RuntimeError("envio de lote anterior sem id registrado: confira a conta e reconcilie "
                           f"{estado_path.name} manualmente; não reenviar")
    enviadas = 0
    if novas:
        entrada = {"id": None, "baixado": False, "custom_ids": {r.custom_id: r.ids for r in novas},
                   "enviado_em": utc_now_iso()}
        estado["lotes"].append(entrada)
        write_json_atomic(estado_path, estado)
        try:
            lote = av.with_retry(
                lambda: client.messages.batches.create(
                    requests=[{"custom_id": r.custom_id, "params": r.params} for r in novas]),
                sleep, attempts=1)
        except av.ApiCallError as exc:
            status = getattr(exc.__cause__, "status_code", None)
            if status is not None and 400 <= status < 500 and status not in (408, 409, 429):
                # recusa explícita: o lote não foi criado, então o envio pode ser repetido
                estado["lotes"].remove(entrada)
                write_json_atomic(estado_path, estado)
            raise
        entrada["id"] = lote.id
        write_json_atomic(estado_path, estado)
        enviadas = len(novas)
        log_fn(f"lote enviado: {lote.id} ({len(novas)} requisições)")
    for entrada in estado["lotes"]:
        if entrada.get("baixado"):
            continue
        espera, atraso = 0.0, poll_s
        while True:
            b = av.with_retry(lambda: client.messages.batches.retrieve(entrada["id"]), sleep)
            if b.processing_status == "ended":
                break
            if espera >= espera_max_s:
                log_fn(f"lote {entrada['id']} ainda em processamento; rode novamente para retomar")
                break
            sleep(atraso)
            espera += atraso
            atraso = min(atraso * 1.5, 300.0)
        if b.processing_status != "ended":
            continue
        ids_por_cid = entrada["custom_ids"]
        for item in client.messages.batches.results(entrada["id"]):
            ids = ids_por_cid.get(item.custom_id)
            if ids is None:
                continue
            r = por_cid.get(item.custom_id) or Requisicao(item.custom_id, ids, {})
            if item.result.type != "succeeded":
                tipo = getattr(getattr(item.result, "error", None), "type", None)
                armazem.gravar([{"id": i, "modelo": modelo, "modo": "batch", "data": utc_now_iso(),
                                 "req": item.custom_id, **meta, "classe": None, "confianca": None,
                                 "justificativa": None, "uso": {}, "erro": f"batch:{item.result.type}:{tipo}"}
                                for i in ids])
                continue
            armazem.gravar(linhas_da_resposta(r, item.result.message, modelo, "batch", meta))
        entrada["baixado"] = True
        write_json_atomic(estado_path, estado)
    return enviadas


def criar_cliente(chave_env: str, fabrica=None):
    """``(cliente, erro)``; o erro cita só o NOME da variável, nunca valor."""
    nome = chave_env.split(",")[0].strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", nome):
        return None, _erro("--chave-env inválido: use o nome de uma variável de ambiente")
    valor = load_env_var(nome)
    if not valor:
        return None, _erro(f"{nome} ausente (ambiente ou .env)")
    return (fabrica or av.make_client)(valor), None


def itens_do_snapshot(raiz: Path, snap: dict, raio: int, extras: dict[str, str] | None = None,
                      ids: set[str] | None = None) -> list[Item]:
    """Itens para o LLM com o contexto lido do arquivo (que tem de ser o do snapshot)."""
    cache: dict[str, str] = {}
    itens = []
    for o in snap["ocorrencias"]:
        if ids is not None and o["id"] not in ids:
            continue
        rel = o["arquivo"]
        if rel not in cache:
            raw, sha = ler_bruto(raiz / rel)
            if sha != snap["arquivos"][rel]["sha256"]:
                raise ValueError(f"{rel} mudou desde o snapshot; rode `inventariar` de novo")
            cache[rel] = normalizar_eol(raw)
        ini = o["offset"]
        fim = ini + len(o["token"])
        itens.append(Item(o["id"], item_texto(o, contexto_marcado(cache[rel], ini, fim, raio),
                                              (extras or {}).get(o["id"], ""))))
    return itens


def _imprimir_estimativa(etapa: str, est: dict) -> None:
    log(f"{etapa}: {est['requisicoes']} requisições para {est['itens']} ocorrências; "
        f"prefixo em cache ≈ {est['entrada_cache']} tokens; entrada variável ≈ {est['entrada_usuario']} tokens; "
        f"entrada total sem cache ≈ {est['entrada_total_sem_cache']} tokens; saída ≈ {est['saida']} tokens "
        "(aproximado: 3,5 caracteres por token)")


def _args_llm(args) -> dict:
    return {"modo": args.modo, "por_req": args.por_requisicao, "concorrencia": args.concorrencia,
            "refazer": args.refazer_falhas, "max_tokens": args.max_tokens}


def cmd_classificar_llm(args: argparse.Namespace, fabrica=None) -> int:
    raiz = Path(args.raiz).resolve()
    path, snap = carregar_snapshot(args)
    if snap is None:
        return _erro("snapshot ausente; rode `inventariar` antes")
    out = resolver(raiz, args.out)
    try:
        itens = itens_do_snapshot(raiz, snap, args.raio)
    except (ValueError, FileNotFoundError) as exc:
        return _erro(str(exc))
    if args.limite:
        itens = itens[: args.limite]
    arm = Armazem(out / "llm.jsonl")
    if args.estimar:
        faltam = set(arm.pendentes([i.id for i in itens], args.refazer_falhas))
        reqs = montar_requisicoes([i for i in itens if i.id in faltam], SISTEMA_CLASSIFICAR, args.modelo,
                                  "clas", args.por_requisicao, max_tokens=args.max_tokens)
        _imprimir_estimativa("classificar --llm", estimar(reqs))
        return 0
    client, erro = criar_cliente(args.chave_env, fabrica)
    if erro is not None:
        return erro
    res = executar_llm(itens=itens, sistema=SISTEMA_CLASSIFICAR, modelo=args.modelo, client=client,
                       armazem=arm, etapa="clas", estado_lotes=out / "llm.lotes.json",
                       meta={"etapa": "classificar", "snapshot": path.name}, **_args_llm(args))
    log(f"classificar --llm: {res['requisicoes']} requisições; pendentes {res['pendentes']}; erros {res['erros']}")
    log(f"saída: {arm.path}")
    return 3 if res["pendentes"] or res["erros"] else 0


def precisa_adjudicar(o: dict, llm: dict | None) -> bool:
    if llm is None or llm.get("erro") or llm.get("classe") is None:
        return True
    if o["classe_regra"] == AMBIGUA or llm["classe"] != o["classe_regra"]:
        return True
    return (llm.get("confianca") or 0.0) < LIMIAR_CONFIANCA


def cmd_adjudicar(args: argparse.Namespace, fabrica=None) -> int:
    raiz = Path(args.raiz).resolve()
    path, snap = carregar_snapshot(args)
    if snap is None:
        return _erro("snapshot ausente; rode `inventariar` antes")
    out = resolver(raiz, args.out)
    llm = {r["id"]: r for r in read_jsonl(out / "llm.jsonl")}
    if not llm:
        return _erro(f"{out / 'llm.jsonl'} ausente ou vazio; rode `classificar --llm` antes")
    ids = {o["id"] for o in snap["ocorrencias"] if precisa_adjudicar(o, llm.get(o["id"]))}
    extras = {}
    for o in snap["ocorrencias"]:
        if o["id"] in ids:
            r = llm.get(o["id"]) or {}
            extras[o["id"]] = (f"classe da regra: {o['classe_regra']} ({o.get('nota_regra', '')}); "
                               f"classe do modelo: {r.get('classe')} (confiança {r.get('confianca')}): "
                               f"{r.get('justificativa')}\n")
    try:
        itens = itens_do_snapshot(raiz, snap, args.raio, extras, ids)
    except (ValueError, FileNotFoundError) as exc:
        return _erro(str(exc))
    if args.limite:
        itens = itens[: args.limite]
    arm = Armazem(out / "adjudicacao.jsonl")
    log(f"adjudicar: {len(itens)} ocorrências divergentes, ambíguas ou com confiança < {LIMIAR_CONFIANCA}")
    if args.estimar:
        faltam = set(arm.pendentes([i.id for i in itens], args.refazer_falhas))
        reqs = montar_requisicoes([i for i in itens if i.id in faltam], SISTEMA_ADJUDICAR, args.modelo,
                                  "adju", args.por_requisicao, max_tokens=args.max_tokens)
        _imprimir_estimativa("adjudicar", estimar(reqs))
        return 0
    client, erro = criar_cliente(args.chave_env, fabrica)
    if erro is not None:
        return erro
    res = executar_llm(itens=itens, sistema=SISTEMA_ADJUDICAR, modelo=args.modelo, client=client,
                       armazem=arm, etapa="adju", estado_lotes=out / "adjudicacao.lotes.json",
                       meta={"etapa": "adjudicar", "snapshot": path.name}, **_args_llm(args))
    log(f"adjudicar: {res['requisicoes']} requisições; pendentes {res['pendentes']}; erros {res['erros']}")
    log(f"saída: {arm.path}")
    return 3 if res["pendentes"] or res["erros"] else 0


# ---------------------------------------------------------------------------
# manifesto
# ---------------------------------------------------------------------------
COLUNAS = ["id", "arquivo", "superficie", "zona", "contexto", "token", "classe_regra", "classe_llm",
           "classe_final", "substituicao", "aprovado", "aplicavel", "confianca_llm", "origem_final",
           "justificativa", "sha256_arquivo", "offset", "commit"]

ESCAPES_MARCADORES = {MARCA_ABRE: "\\u27e6", MARCA_FECHA: "\\u27e7", MARCA_QUEBRA: "\\u23ce"}


def tem_marcador(*partes: str) -> bool:
    return any(m in p for p in partes for m in ESCAPES_MARCADORES)


def escapar_marcadores(texto: str) -> str:
    """Troca ``⟦ ⟧ ⏎`` por ``\\u27e6 \\u27e7 \\u23ce`` literais (só para exibir contexto não aplicável)."""
    for marca, escape in ESCAPES_MARCADORES.items():
        texto = texto.replace(marca, escape)
    return texto


def fmt_contexto(esq: str, token: str, dir_: str) -> str:
    for parte in (esq, token, dir_):
        if any(m in parte for m in (MARCA_ABRE, MARCA_FECHA, MARCA_QUEBRA)):
            raise ValueError("o contexto contém um dos marcadores ⟦ ⟧ ⏎")
    return (esq + MARCA_ABRE + token + MARCA_FECHA + dir_).replace("\n", MARCA_QUEBRA)


def parse_contexto(celula: str) -> tuple[str, str, str]:
    s = celula.replace(MARCA_QUEBRA, "\n")
    a = s.find(MARCA_ABRE)
    b = s.find(MARCA_FECHA)
    if a < 0 or b < a or s.count(MARCA_ABRE) != 1 or s.count(MARCA_FECHA) != 1:
        raise ValueError("contexto sem marcadores ⟦token⟧ válidos")
    return s[:a], s[a + 1:b], s[b + 1:]


def decidir_classe(o: dict, llm: dict | None, adj: dict | None, so_regras: bool) -> tuple[str, str]:
    """``(classe_final, origem)``. Zona protegida nunca vira alvo."""
    regra = o["classe_regra"]
    r_adj = adj["classe"] if adj and not adj.get("erro") and adj.get("classe") else None
    r_llm = llm["classe"] if llm and not llm.get("erro") and llm.get("classe") else None
    if o["zona"] in ZONAS_PROTEGIDAS:
        discorda = (r_adj or r_llm or regra) in CLASSES_ALVO
        return regra, "zona_protegida" + ("_conflito" if discorda else "")
    if so_regras:
        return regra, "so_regras"
    if r_adj:
        return r_adj, "adjudicacao"
    if r_llm and r_llm == regra and regra != AMBIGUA and (llm.get("confianca") or 0.0) >= LIMIAR_CONFIANCA:
        return regra, "regra_e_llm"
    return AMBIGUA, "pendente"


def substituicao_da_linha(o: dict, classe_final: str) -> str:
    if classe_final not in CLASSES_ALVO:
        return ""
    return substituicao_token(o["token"], em_math=o["zona"] == "math")


def gravar_manifesto(destino: Path, linhas: list[dict]) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        w.writeheader()
        w.writerows(linhas)
    replace_with_retry(tmp, destino)


def ler_manifesto(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def construir_manifesto(snap: dict, llm: dict[str, dict], adj: dict[str, dict], anterior: dict[str, dict],
                        so_regras: bool = False) -> list[dict]:
    linhas = []
    for o in snap["ocorrencias"]:
        l_llm, l_adj = llm.get(o["id"]), adj.get(o["id"])
        final, origem = decidir_classe(o, l_llm, l_adj, so_regras)
        if tem_marcador(o["esq80"], o["token"], o["dir80"]):
            if final in CLASSES_ALVO:
                raise ValueError(
                    f"{o['id']} ({o['arquivo']}): ocorrência ALVO ({final}) com ⟦ ⟧ ou ⏎ no contexto; "
                    "o span não pode ser localizado com segurança, resolva à mão")
            # não alvo (ou ainda ambígua): entra escapada, só para leitura, e nunca é aplicável
            ctx = fmt_contexto(escapar_marcadores(o["esq80"]), escapar_marcadores(o["token"]),
                               escapar_marcadores(o["dir80"]))
            aplicavel = "nao"
        else:
            ctx = fmt_contexto(o["esq80"], o["token"], o["dir80"])
            aplicavel = "sim"
        subst = substituicao_da_linha(o, final)
        linha = {
            "id": o["id"], "arquivo": o["arquivo"], "superficie": o["superficie"], "zona": o["zona"],
            "contexto": ctx, "token": o["token"], "classe_regra": o["classe_regra"],
            "classe_llm": (l_llm or {}).get("classe") or "", "classe_final": final, "substituicao": subst,
            "aprovado": "", "aplicavel": aplicavel,
            "confianca_llm": "" if not l_llm or l_llm.get("confianca") is None else f"{l_llm['confianca']:.2f}",
            "origem_final": origem,
            "justificativa": ((l_adj or l_llm or {}).get("justificativa") or "").replace("\n", " "),
            "sha256_arquivo": o["sha256_arquivo"], "offset": o["offset"], "commit": o["commit"],
        }
        ant = anterior.get(o["id"])
        if ant and ant.get("contexto") == ctx and ant.get("classe_final") == final and ant.get("substituicao") == subst:
            linha["aprovado"] = ant.get("aprovado", "")
        linhas.append(linha)
    return linhas


def resumo_manifesto(linhas: list[dict]) -> str:
    por_arq: dict[str, Counter] = {}
    for l in linhas:
        por_arq.setdefault(l["arquivo"], Counter())[l["classe_final"]] += 1
    out = ["Resumo por arquivo (classe_final):"]
    for arq in sorted(por_arq):
        out.append(f"  {arq}: " + ", ".join(f"{k}={v}" for k, v in sorted(por_arq[arq].items())))
    alvos = sum(1 for l in linhas if l["classe_final"] in CLASSES_ALVO)
    out.append(f"Total: {len(linhas)} candidatas; alvo {alvos}; "
               f"não alvo {sum(1 for l in linhas if l['classe_final'] in CLASSES_NAO_ALVO)}; "
               f"ambíguas {sum(1 for l in linhas if l['classe_final'] == AMBIGUA)}")
    inaplicaveis = [l for l in linhas if l.get("aplicavel") == "nao"]
    if inaplicaveis:
        out.append("Contexto com marcadores ⟦ ⟧ ⏎ (escapados, aplicavel=nao): "
                   + ", ".join(f"{l['id']} ({l['arquivo']})" for l in inaplicaveis))
    conflitos = [l for l in linhas if l["origem_final"].endswith("_conflito")]
    if conflitos:
        out.append("Zona protegida com opinião alvo (mantida N): " + ", ".join(l["id"] for l in conflitos))
    amb = [l for l in linhas if l["classe_final"] == AMBIGUA]
    out.append("Casos ambíguos (decidir à mão; sem substituição até virarem classe alvo ou não alvo):")
    for l in amb:
        out.append(f"  {l['id']} {l['arquivo']} [{l['zona']}] regra={l['classe_regra']} llm={l['classe_llm'] or '-'}: "
                   f"{l['contexto'][:160]}")
    return "\n".join(out) + "\n"


def cmd_manifesto(args: argparse.Namespace) -> int:
    raiz = Path(args.raiz).resolve()
    path, snap = carregar_snapshot(args)
    if snap is None:
        return _erro("snapshot ausente; rode `inventariar` antes")
    out = resolver(raiz, args.out)
    llm = {r["id"]: r for r in read_jsonl(out / "llm.jsonl")}
    adj = {r["id"]: r for r in read_jsonl(out / "adjudicacao.jsonl")}
    if not args.so_regras and not llm:
        return _erro("sem llm.jsonl; rode `classificar --llm` ou use --so-regras (sem a dupla checagem)")
    destino = out / "manifesto.csv"
    anterior = {l["id"]: l for l in ler_manifesto(destino)} if destino.is_file() else {}
    try:
        linhas = construir_manifesto(snap, llm, adj, anterior, args.so_regras)
    except ValueError as exc:
        return _erro(str(exc))
    gravar_manifesto(destino, linhas)
    texto = resumo_manifesto(linhas)
    (out / "manifesto_resumo.txt").write_text(texto, encoding="utf-8")
    print(texto)
    log(f"manifesto: {destino} ({len(linhas)} linhas; preencha `aprovado` com sim nas linhas aprovadas)")
    return 0


# ---------------------------------------------------------------------------
# aplicar
# ---------------------------------------------------------------------------
def _nl(padrao: str) -> str:
    return padrao.replace("\\\n", "(?:\r\n|\n)")


def _alternativas(token: str) -> str:
    opcoes = {token}
    if token in ROTULO_BRACO:
        opcoes.add(ROTULO_BRACO[token])
    if token in ROTULO_MATH:
        opcoes.add(ROTULO_MATH[token])
    return "(?:" + "|".join(re.escape(x) for x in sorted(opcoes, key=len, reverse=True)) + ")"


def _parte_vizinha(txt: str) -> str:
    saida, pos = [], 0
    for m in CAND_RE.finditer(txt):
        saida.append(_nl(re.escape(txt[pos:m.start()])))
        saida.append(_alternativas(m.group(0)))
        pos = m.end()
    saida.append(_nl(re.escape(txt[pos:])))
    return "".join(saida)


def padrao_do_span(esq: str, token: str, dir_: str) -> re.Pattern:
    """Padrão (lookahead) com grupo ``t`` no token; vizinhos já trocados também casam."""
    return re.compile(f"(?=(?P<e>{_parte_vizinha(esq)})(?P<t>{_alternativas(token)})(?P<d>{_parte_vizinha(dir_)}))")


def aplicar_em_texto(texto: str, linhas: list[dict]) -> tuple[str, list[dict], list[dict]]:
    """Troca os spans ``linhas`` (``id, contexto, token, substituicao``) em ``texto``.

    Devolve ``(novo texto, resultados, erros)``; com qualquer erro o texto devolvido é o original.
    """
    resultados, erros, trocas = [], [], []
    for l in linhas:
        try:
            esq, token, dir_ = parse_contexto(l["contexto"])
        except ValueError as exc:
            erros.append({"id": l["id"], "erro": f"contexto_invalido: {exc}"})
            continue
        subst = l["substituicao"]
        achados = list(padrao_do_span(esq, token, dir_).finditer(texto))
        if not achados:
            erros.append({"id": l["id"], "erro": "contexto_ausente"})
            continue
        if len(achados) > 1:
            erros.append({"id": l["id"], "erro": f"contexto_ambiguo ({len(achados)} ocorrências)"})
            continue
        m = achados[0]
        ini, fim = m.start("t"), m.end("t")
        atual = texto[ini:fim]
        if atual == token:
            trocas.append((ini, fim, subst, l["id"]))
            resultados.append({"id": l["id"], "estado": "aplicado", "offset": ini})
        elif atual == subst:
            resultados.append({"id": l["id"], "estado": "ja_aplicado", "offset": ini})
        else:
            erros.append({"id": l["id"], "erro": f"estado_inconsistente ({atual!r})"})
    posicoes = Counter(t[0] for t in trocas)
    for ini, _fim, _s, ident in trocas:
        if posicoes[ini] > 1:
            erros.append({"id": ident, "erro": "span_duplicado"})
    if erros:
        return texto, resultados, erros
    novo = texto
    for ini, fim, subst, _ident in sorted(trocas, reverse=True):
        novo = novo[:ini] + subst + novo[fim:]
    return novo, resultados, erros


def linhas_aprovadas(linhas: list[dict]) -> tuple[list[dict], list[str]]:
    """Linhas ``aprovado=sim`` válidas e a lista de problemas (classe, substituição ou escopo)."""
    ok, problemas = [], []
    for l in linhas:
        if (l.get("aprovado") or "").strip().lower() != "sim":
            continue
        if (l.get("aplicavel") or "sim").strip().lower() != "sim":
            problemas.append(f"{l['id']}: aprovado, mas aplicavel=nao (contexto escapado, não localizável)")
            continue
        if l["classe_final"] not in CLASSES_ALVO:
            problemas.append(f"{l['id']}: aprovado, mas classe_final={l['classe_final']!r} não é alvo")
            continue
        if fora_de_escopo(l["arquivo"]) or l["zona"] in ZONAS_PROTEGIDAS:
            problemas.append(f"{l['id']}: aprovado em zona ou arquivo protegido ({l['zona']}, {l['arquivo']})")
            continue
        try:
            esperado = substituicao_token(l["token"], em_math=l["zona"] == "math")
        except ValueError as exc:
            problemas.append(f"{l['id']}: {exc}")
            continue
        if l["substituicao"] != esperado:
            problemas.append(f"{l['id']}: substituicao {l['substituicao']!r} difere do mapeamento ({esperado!r})")
            continue
        ok.append(l)
    return ok, problemas


def _bate_lote(rel: str, lote: list[str] | None) -> bool:
    if not lote:
        return True
    return any(rel == posix(x) or rel.endswith("/" + posix(x)) for x in lote)


def cmd_aplicar(args: argparse.Namespace) -> int:
    raiz = Path(args.raiz).resolve()
    destino = resolver(raiz, args.manifesto)
    if not destino.is_file():
        return _erro(f"manifesto ausente: {destino}")
    ok, problemas = linhas_aprovadas(ler_manifesto(destino))
    for p in problemas:
        print(f"recusado: {p}", file=sys.stderr)
    if problemas:
        print("manifesto com linhas aprovadas inválidas: nada foi alterado", file=sys.stderr)
        return 3
    lote = [x.strip() for x in args.lote.split(",") if x.strip()] if args.lote else None
    por_arq: dict[str, list[dict]] = {}
    for l in ok:
        if _bate_lote(l["arquivo"], lote):
            por_arq.setdefault(l["arquivo"], []).append(l)
    falhou = bool(problemas)
    total = 0
    for rel, linhas in sorted(por_arq.items()):
        path = (raiz / rel).resolve()
        try:
            path.relative_to(raiz)
        except ValueError:
            print(f"recusado: {rel} fora da raiz", file=sys.stderr)
            falhou = True
            continue
        if not path.is_file():
            print(f"recusado: {rel} ausente", file=sys.stderr)
            falhou = True
            continue
        raw, sha = ler_bruto(path)
        shas = {l["sha256_arquivo"] for l in linhas}
        if sha not in shas:
            log(f"{rel}: arquivo mudou desde o snapshot; relocalizando pelo contexto")
        novo, resultados, erros = aplicar_em_texto(raw, linhas)
        if erros:
            falhou = True
            for e in erros:
                print(f"recusado: {rel}: {e['id']}: {e['erro']}", file=sys.stderr)
            print(f"{rel}: NÃO alterado", file=sys.stderr)
            continue
        n_ap = sum(1 for r in resultados if r["estado"] == "aplicado")
        n_ja = sum(1 for r in resultados if r["estado"] == "ja_aplicado")
        if novo != raw and not args.dry_run:
            escrever_bruto(path, novo)
        total += n_ap
        log(f"{rel}: {n_ap} trocas{' (simulação)' if args.dry_run else ''}; {n_ja} já aplicadas")
    log(f"total de trocas: {total}")
    return 3 if falhou else 0


# ---------------------------------------------------------------------------
# verificar
# ---------------------------------------------------------------------------
_TOK_RE = re.compile(r"\w+|\s+|[^\w\s]", re.U)


def _par_normalizado(b: str, a: str) -> tuple[str, str]:
    i = 0
    while i < min(len(b), len(a)) and b[i] == a[i]:
        i += 1
    j = 0
    while j < min(len(b), len(a)) - i and b[len(b) - 1 - j] == a[len(a) - 1 - j]:
        j += 1
    return b[i:len(b) - j], a[i:len(a) - j]


def ops_token(base: str, atual: str) -> Counter:
    """Multiconjunto de ``(trecho da base, trecho do atual)`` das diferenças token a token."""
    tb, ta = _TOK_RE.findall(base), _TOK_RE.findall(atual)
    ops: Counter = Counter()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, tb, ta, autojunk=False).get_opcodes():
        if tag != "equal":
            ops[_par_normalizado("".join(tb[i1:i2]), "".join(ta[j1:j2]))] += 1
    return ops


def verificar_arquivo(rel: str, base: str, atual: str, linhas: list[dict], todas: list[dict],
                      completo: bool = False) -> dict:
    """Verifica um arquivo (textos com EOL normalizado)."""
    rel_ = {"arquivo": rel, "falhas": [], "avisos": []}
    esperado, resultados, erros = aplicar_em_texto(base, linhas)
    if erros:
        rel_["falhas"].append({"tipo": "span_nao_localizado_na_base", "detalhe": erros[:10]})
        esperado = base
    if atual != esperado:
        obs, exp = ops_token(base, atual), ops_token(base, esperado)
        fora = obs - exp
        faltam = exp - obs
        rel_["falhas"].append({
            "tipo": "diferenca_fora_do_span",
            "inesperadas": [{"base": b[:120], "atual": a[:120], "n": n} for (b, a), n in list(fora.items())[:10]],
            "nao_aplicadas": [{"base": b[:120], "atual": a[:120], "n": n} for (b, a), n in list(faltam.items())[:10]],
        })
    cb, ca = candidatos_texto(rel, base), candidatos_texto(rel, atual)
    esperado_res = len(cb) - len([r for r in resultados if r["estado"] in ("aplicado", "ja_aplicado")])
    rel_["residuo"] = len(ca)
    if len(ca) != esperado_res:
        rel_["falhas"].append({"tipo": "residuo_difere_da_base", "atual": len(ca), "esperado": esperado_res})
    if todas:
        esp_man = len(todas) - len(linhas)
        if len(ca) != esp_man:
            rel_["falhas"].append({"tipo": "residuo_difere_do_manifesto", "atual": len(ca), "esperado": esp_man})
    alvos = [c for c in ca if c["classe"] in CLASSES_ALVO]
    rel_["alvos_pendentes"] = len(alvos)
    var = [c for c in alvos if c["classe"] in ("A2", "A3", "A4", "A5")]
    if alvos:
        msg = {"tipo": "alvo_pendente", "n": len(alvos), "variantes_A2_A5": len(var)}
        (rel_["falhas"] if completo else rel_["avisos"]).append(msg)
    pb = [p for p in spans_paragrafos(base)]
    pa = [p for p in spans_paragrafos(atual)]
    if len(pb) != len(pa):
        rel_["falhas"].append({"tipo": "numero_de_paragrafos", "base": len(pb), "atual": len(pa)})
    else:
        difs = []
        for k, ((b1, b2), (a1, a2)) in enumerate(zip(pb, pa)):
            cb_, ca_ = contar_numeros_texto(base[b1:b2]), contar_numeros_texto(atual[a1:a2])
            if cb_ != ca_:
                difs.append({"paragrafo": k, "removidos": dict((cb_ - ca_).most_common(5)),
                             "acrescentados": dict((ca_ - cb_).most_common(5))})
        if difs:
            rel_["falhas"].append({"tipo": "numeros_por_paragrafo", "paragrafos": difs[:10], "total": len(difs)})
    rel_["ok"] = not rel_["falhas"]
    return rel_


def leitor_git(raiz: Path, ref: str) -> Callable[[str], str | None]:
    def ler(rel: str) -> str | None:
        r = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=raiz, capture_output=True, timeout=60)
        if r.returncode != 0:
            return None
        return normalizar_eol(r.stdout.decode("utf-8"))

    return ler


def verificar(raiz: Path, arquivos: list[str], manifesto: list[dict], leitor_base: Callable[[str], str | None],
              completo: bool = False) -> dict:
    ok_linhas, problemas = linhas_aprovadas(manifesto)
    por_arq_ok: dict[str, list[dict]] = {}
    for l in ok_linhas:
        por_arq_ok.setdefault(l["arquivo"], []).append(l)
    por_arq_todas: dict[str, list[dict]] = {}
    for l in manifesto:
        por_arq_todas.setdefault(l["arquivo"], []).append(l)
    relatorios = []
    for rel in arquivos:
        base = leitor_base(rel)
        if base is None:
            relatorios.append({"arquivo": rel, "ok": False,
                               "falhas": [{"tipo": "base_ausente"}], "avisos": []})
            continue
        atual = normalizar_eol(ler_bruto(raiz / rel)[0])
        relatorios.append(verificar_arquivo(rel, base, atual, por_arq_ok.get(rel, []),
                                            por_arq_todas.get(rel, []), completo))
    return {"problemas_manifesto": problemas, "arquivos": relatorios,
            "ok": not problemas and all(r["ok"] for r in relatorios)}


def cmd_verificar(args: argparse.Namespace, leitor=None) -> int:
    raiz = Path(args.raiz).resolve()
    destino = resolver(raiz, args.manifesto)
    if not destino.is_file():
        return _erro(f"manifesto ausente: {destino}")
    manifesto = ler_manifesto(destino)
    if args.arquivos:
        arquivos = [posix(a.strip()) for a in args.arquivos.split(",") if a.strip()]
    else:
        arquivos = sorted({l["arquivo"] for l in manifesto})
    rel = verificar(raiz, arquivos, manifesto, leitor or leitor_git(raiz, args.base), args.completo)
    out = resolver(raiz, args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "verificacao.json", rel)
    for p in rel["problemas_manifesto"]:
        print(f"manifesto: {p}", file=sys.stderr)
    for r in rel["arquivos"]:
        log(f"{r['arquivo']}: {'OK' if r['ok'] else 'FALHA'} (resíduo {r.get('residuo', '-')}; "
            f"alvos pendentes {r.get('alvos_pendentes', '-')})")
        for f in r["falhas"]:
            log(f"  falha: {json.dumps(f, ensure_ascii=False)[:600]}")
        for a in r["avisos"]:
            log(f"  aviso: {json.dumps(a, ensure_ascii=False)}")
    log(f"verificação: {'OK' if rel['ok'] else 'FALHOU'}; relatório: {out / 'verificacao.json'}")
    return 0 if rel["ok"] else 3


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Renomeação segura da sigla P (arranjo sequencial) para ASR na camada de apresentação.")
    sub = parser.add_subparsers(dest="comando", required=True)

    def comum(p: argparse.ArgumentParser, snapshot: bool = True) -> None:
        p.add_argument("--raiz", default=str(ROOT), help="raiz do repositório (padrão: a deste script)")
        p.add_argument("--out", default=DEFAULT_OUT, help=f"diretório de trabalho (padrão {DEFAULT_OUT})")
        if snapshot:
            p.add_argument("--snapshot", default="", help="snapshot (padrão: o mais recente em --out)")

    def api(p: argparse.ArgumentParser, modelo: str, raio: int, por_req: int = 8) -> None:
        p.add_argument("--modelo", default=modelo)
        p.add_argument("--modo", choices=("batch", "sync"), default="batch")
        p.add_argument("--concorrencia", type=int, default=2, help="threads no modo sync (padrão 2)")
        p.add_argument("--por-requisicao", type=int, default=por_req, dest="por_requisicao",
                       help=f"ocorrências por requisição (padrão {por_req})")
        p.add_argument("--max-tokens", type=int, default=MAX_TOKENS_PADRAO, dest="max_tokens",
                       help=f"teto de tokens de saída de cada requisição (padrão {MAX_TOKENS_PADRAO}); "
                            "a justificativa do árbitro cabe com folga")
        p.add_argument("--raio", type=int, default=raio, help=f"caracteres de contexto de cada lado (padrão {raio})")
        p.add_argument("--limite", type=int, default=0, help="só as N primeiras ocorrências (piloto)")
        p.add_argument("--estimar", action="store_true", help="não chama a API: imprime requisições e tokens")
        p.add_argument("--refazer-falhas", action="store_true", dest="refazer_falhas",
                       help="reenvia o que terminou com erro de resposta")
        p.add_argument("--chave-env", default="ANTHROPIC_API_KEY", dest="chave_env", metavar="NOME",
                       help="variável com a chave da Anthropic (ambiente ou .env); nunca impressa")

    p_in = sub.add_parser("inventariar", help="snapshot de todas as ocorrências candidatas")
    comum(p_in, snapshot=False)
    p_in.add_argument("--superficies", default=",".join(SUPERFICIES),
                      help=f"superfícies, separadas por vírgula: {', '.join(SUPERFICIES)}")
    p_in.add_argument("--arquivos", default="", help="restringe a estes arquivos (vírgula), dentro das superfícies")
    p_in.set_defaults(func=cmd_inventariar)

    p_cl = sub.add_parser("classificar", help="classifica as candidatas por regras ou por LLM")
    comum(p_cl)
    modo = p_cl.add_mutually_exclusive_group(required=True)
    modo.add_argument("--regras", action="store_true", help="reaplica as regras de zona ao snapshot")
    modo.add_argument("--llm", action="store_true", help="Claude Sonnet 5.5 em todas as candidatas")
    p_cl.add_argument("--modelo", default=DEFAULT_MODEL)
    p_cl.add_argument("--modo", choices=("batch", "sync"), default="batch")
    p_cl.add_argument("--concorrencia", type=int, default=2)
    p_cl.add_argument("--por-requisicao", type=int, default=8, dest="por_requisicao")
    p_cl.add_argument("--max-tokens", type=int, default=MAX_TOKENS_PADRAO, dest="max_tokens",
                      help=f"teto de tokens de saída de cada requisição (padrão {MAX_TOKENS_PADRAO})")
    p_cl.add_argument("--raio", type=int, default=RAIO_LLM)
    p_cl.add_argument("--limite", type=int, default=0)
    p_cl.add_argument("--estimar", action="store_true")
    p_cl.add_argument("--refazer-falhas", action="store_true", dest="refazer_falhas")
    p_cl.add_argument("--chave-env", default="ANTHROPIC_API_KEY", dest="chave_env", metavar="NOME")
    p_cl.set_defaults(func=lambda a: cmd_classificar_regras(a) if a.regras else cmd_classificar_llm(a))

    p_ad = sub.add_parser("adjudicar", help="Claude Opus 5.5 nas divergências regra x LLM ou confiança < 0,9")
    comum(p_ad)
    api(p_ad, DEFAULT_ADJ_MODEL, RAIO_ADJ, por_req=2)
    p_ad.set_defaults(func=cmd_adjudicar)

    p_ma = sub.add_parser("manifesto", help="gera manifesto.csv para aprovação e o resumo")
    comum(p_ma)
    p_ma.add_argument("--so-regras", action="store_true", dest="so_regras",
                      help="usa só a regra (sem LLM); abre mão da dupla checagem")
    p_ma.set_defaults(func=cmd_manifesto)

    p_ap = sub.add_parser("aplicar", help="troca só os spans com aprovado=sim")
    comum(p_ap, snapshot=False)
    p_ap.add_argument("--manifesto", default=f"{DEFAULT_OUT}/manifesto.csv")
    p_ap.add_argument("--lote", default="", help="arquivos do lote (vírgula); padrão: todos os aprovados")
    p_ap.add_argument("--dry-run", action="store_true", dest="dry_run", help="não escreve nada")
    p_ap.set_defaults(func=cmd_aplicar)

    p_ve = sub.add_parser("verificar", help="compara com a base: spans, resíduo e números por parágrafo")
    comum(p_ve, snapshot=False)
    p_ve.add_argument("--manifesto", default=f"{DEFAULT_OUT}/manifesto.csv")
    p_ve.add_argument("--base", default="HEAD", help="referência git da base (padrão HEAD)")
    p_ve.add_argument("--arquivos", default="", help="restringe a estes arquivos (vírgula)")
    p_ve.add_argument("--completo", action="store_true",
                      help="alvos ainda não trocados passam a ser falha (aceite final)")
    p_ve.set_defaults(func=cmd_verificar)
    return parser


def main(argv: list[str] | None = None) -> int:
    for fluxo in (sys.stdout, sys.stderr):
        if (getattr(fluxo, "encoding", "") or "").lower().replace("-", "") == "utf8":
            continue
        try:
            fluxo.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
