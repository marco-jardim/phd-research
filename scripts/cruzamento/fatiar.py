"""Etapa 1: fatiamento da tese, dos slides e do roteiro em contextos pequenos (sem API).

Cada fatia é um grupo de parágrafos inteiros da mesma seção e do mesmo arquivo, com no máximo
``max_palavras`` palavras (um parágrafo maior que o limite vira uma fatia só dele; nunca é cortado).
Nos slides, cada ``frame`` é uma fatia; no roteiro, cada seção ou bloco entre títulos de Markdown
(blocos maiores que o limite são divididos entre parágrafos).

Só entra o que a árvore de ``\\input`` de ``tese.tex`` alcança (reaproveita ``scripts/linter_prosa.py`` e
``scripts/indice_tese.py``, que incluem tabelas e notas); arquivos ``*_diff.tex`` e pastas ``*_diff_work``
são ignorados. O texto da fatia vem de ``Arquivo.limpo`` (comentários, preâmbulo e Abstract em inglês
em branco; acentos LaTeX resolvidos), com os números de linha do arquivo original.

Campos: ``id`` (sha256 de ``arquivo|linha_inicio|texto``, 16 hexadecimais), ``arquivo``, ``linha_inicio``,
``linha_fim``, ``secao`` (caminho de títulos), ``fonte`` (tese|slides|roteiro), ``palavras``, ``texto``
(quebras de linha preservadas; parágrafos separados por uma linha vazia) e ``linhas`` (número da linha do
arquivo para cada linha de ``texto``; 0 nas linhas vazias de separação).
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from scripts import indice_tese as it
from scripts import linter_prosa as lp

from .comum import MAX_PALAVRAS_FATIA, id_curto

Paragrafo = tuple[int, list[str]]  # (linha inicial, linhas)

RE_TITULO_MD = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
RE_FRAME = re.compile(r"\\begin\{frame\}")
RE_FIM_FRAME = re.compile(r"\\end\{frame\}")
RE_SECAO_SLIDES = re.compile(r"\\section\*?(?![A-Za-z])")


def _palavras(texto: str) -> int:
    return lp.contar_palavras(it.RE_AMBIENTE.sub(" ", texto))


def _relativo(caminho: Path, base: Path) -> str:
    try:
        return caminho.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return caminho.resolve().as_posix()


def arquivo_excluido(rel: str) -> bool:
    """Artefatos de diff de revisão: ``*_diff.tex``, ``*_diff_work/`` e os excluídos pelo linter."""
    partes = PurePosixPath(rel).parts
    nome = partes[-1] if partes else ""
    return (
        nome.endswith("_diff.tex")
        or nome in lp.EXCLUIR_ARQUIVOS
        or any(p.endswith("_diff_work") for p in partes[:-1])
    )


# ---------------------------------------------------------------------------
# Parágrafos e empacotamento
# ---------------------------------------------------------------------------
def paragrafos_do_trecho(texto: str, linha_inicial: int) -> list[Paragrafo]:
    """Parágrafos (blocos de linhas não vazias) de ``texto``, que começa na linha ``linha_inicial``."""
    res: list[Paragrafo] = []
    atual: list[str] = []
    inicio = linha_inicial
    for i, linha in enumerate(texto.split("\n")):
        if linha.strip():
            if not atual:
                inicio = linha_inicial + i
            atual.append(linha.rstrip())
        elif atual:
            res.append((inicio, atual))
            atual = []
    if atual:
        res.append((inicio, atual))
    return res


def empacotar(paragrafos: list[Paragrafo], max_palavras: int) -> list[list[Paragrafo]]:
    """Agrupa parágrafos consecutivos sem passar de ``max_palavras`` (parágrafos sem palavras saem)."""
    grupos: list[list[Paragrafo]] = []
    atual: list[Paragrafo] = []
    total = 0
    for paragrafo in paragrafos:
        n = _palavras("\n".join(paragrafo[1]))
        if n == 0:
            continue
        if atual and total + n > max_palavras:
            grupos.append(atual)
            atual, total = [], 0
        atual.append(paragrafo)
        total += n
    if atual:
        grupos.append(atual)
    return grupos


def montar_fatia(arquivo: str, secao: str, fonte: str, linhas_texto: list[str], numeros: list[int]) -> dict:
    texto = "\n".join(linhas_texto)
    reais = [n for n in numeros if n]
    inicio = reais[0]
    return {
        "id": id_curto(arquivo, str(inicio), texto),
        "arquivo": arquivo,
        "linha_inicio": inicio,
        "linha_fim": reais[-1],
        "secao": secao,
        "fonte": fonte,
        "palavras": _palavras(texto),
        "texto": texto,
        "linhas": numeros,
    }


def fatias_de_paragrafos(
    arquivo: str, secao: str, fonte: str, paragrafos: list[Paragrafo], max_palavras: int
) -> list[dict]:
    fatias = []
    for grupo in empacotar(paragrafos, max_palavras):
        texto: list[str] = []
        numeros: list[int] = []
        for k, (inicio, linhas) in enumerate(grupo):
            if k:
                texto.append("")
                numeros.append(0)
            for j, linha in enumerate(linhas):
                texto.append(linha)
                numeros.append(inicio + j)
        fatias.append(montar_fatia(arquivo, secao, fonte, texto, numeros))
    return fatias


# ---------------------------------------------------------------------------
# Tese
# ---------------------------------------------------------------------------
class _ConstrutorFatias(it._Construtor):
    """Construtor do índice que também guarda os trechos (arquivo, início, fim) de cada nó."""

    def __init__(self, arvore: lp.Arvore, base: Path) -> None:
        super().__init__(arvore, base)
        self.trechos: dict[int, list[tuple[lp.Arquivo, int, int]]] = {}

    def _adicionar(self, no, arq, a, b) -> None:
        self.trechos.setdefault(no.id, []).append((arq, a, b))
        super()._adicionar(no, arq, a, b)


def _caminho_secao(nos: list, no) -> str:
    rotulos = []
    atual = no
    while atual is not None:
        rotulos.append(atual.rotulo)
        atual = nos[atual.pai] if atual.pai is not None else None
    return " > ".join(reversed(rotulos))


def fatiar_tese(tese: Path, base: Path, max_palavras: int = MAX_PALAVRAS_FATIA) -> list[dict]:
    tese = Path(tese).resolve()
    base = Path(base).resolve()
    arvore = lp.carregar_arvore(tese, base)
    arvore.visitas = [v for v in arvore.visitas if not arquivo_excluido(v[0].rel)]
    construtor = _ConstrutorFatias(arvore, base)
    nos = construtor.executar()
    fatias: list[dict] = []
    for no in nos:
        secao = _caminho_secao(nos, no)
        rel_atual = None
        paragrafos: list[Paragrafo] = []

        def descarregar() -> None:
            if rel_atual is not None and paragrafos:
                fatias.extend(fatias_de_paragrafos(rel_atual, secao, "tese", paragrafos, max_palavras))

        for arq, a, b in construtor.trechos.get(no.id, []):
            if arq.rel != rel_atual:
                descarregar()
                rel_atual, paragrafos = arq.rel, []
            paragrafos.extend(paragrafos_do_trecho(arq.limpo[a:b], arq.linha(a)))
        descarregar()
    return fatias


# ---------------------------------------------------------------------------
# Slides (um frame por fatia)
# ---------------------------------------------------------------------------
def fatiar_slides(caminho: Path, base: Path) -> list[dict]:
    caminho = Path(caminho)
    bruto = caminho.read_text(encoding="utf-8-sig")
    limpo = lp._branquear(bruto, lp._intervalos_comentarios(bruto))
    rel = _relativo(caminho, base)
    secoes = []
    for m in RE_SECAO_SLIDES.finditer(limpo):
        grupo = it._ler_grupo(limpo, m.end())
        if grupo is not None:
            secoes.append((m.start(), it._limpar_titulo(grupo[0])))
    fatias: list[dict] = []
    numero_slide = 0
    for m in RE_FRAME.finditer(limpo):
        fim = RE_FIM_FRAME.search(limpo, m.end())
        if fim is None:
            continue
        numero_slide += 1
        titulo = ""
        resto = limpo[m.end():].lstrip(" \t")
        if resto[:1] in ("{", "["):
            grupo = it._ler_grupo(limpo, m.end())
            if grupo is not None:
                titulo = it._limpar_titulo(grupo[0])
        secao_atual = [t for pos, t in secoes if pos < m.start()]
        partes = ["Slides"]
        if secao_atual:
            partes.append(secao_atual[-1])
        partes.append(f"Slide {numero_slide}" + (f": {titulo}" if titulo else ""))
        inicio_linha = limpo.count("\n", 0, m.start()) + 1
        linhas = [linha.rstrip() for linha in limpo[m.start():fim.end()].split("\n")]
        numeros = [inicio_linha + i for i in range(len(linhas))]
        fatias.append(montar_fatia(rel, " > ".join(partes), "slides", linhas, numeros))
    return fatias


# ---------------------------------------------------------------------------
# Roteiro (Markdown: uma seção ou bloco entre títulos por fatia)
# ---------------------------------------------------------------------------
def fatiar_roteiro(caminho: Path, base: Path, max_palavras: int = MAX_PALAVRAS_FATIA) -> list[dict]:
    caminho = Path(caminho)
    rel = _relativo(caminho, base)
    linhas = caminho.read_text(encoding="utf-8-sig").split("\n")
    pilha: list[tuple[int, str]] = []
    blocos: list[tuple[str, list[tuple[int, str]]]] = [("Roteiro", [])]
    em_codigo = False
    for numero, linha in enumerate(linhas, 1):
        if linha.strip().startswith("```"):
            em_codigo = not em_codigo
        titulo = None if em_codigo else RE_TITULO_MD.match(linha)
        if titulo:
            nivel = len(titulo.group(1))
            while pilha and pilha[-1][0] >= nivel:
                pilha.pop()
            pilha.append((nivel, titulo.group(2)))
            blocos.append((" > ".join(t for _n, t in pilha), []))
        blocos[-1][1].append((numero, linha))
    fatias: list[dict] = []
    for secao, conteudo in blocos:
        com_texto = [1 for _n, linha in conteudo if linha.strip() and not RE_TITULO_MD.match(linha)]
        if not com_texto:
            continue
        paragrafos = paragrafos_do_trecho("\n".join(linha for _n, linha in conteudo), conteudo[0][0])
        fatias.extend(fatias_de_paragrafos(rel, secao, "roteiro", paragrafos, max_palavras))
    return fatias


# ---------------------------------------------------------------------------
# Conjunto
# ---------------------------------------------------------------------------
def fatiar(
    base: Path,
    tese: Path | None,
    slides: Path | None,
    roteiro: Path | None,
    max_palavras: int = MAX_PALAVRAS_FATIA,
) -> list[dict]:
    """Fatias da tese, dos slides e do roteiro, em ordem estável e sem ids repetidos."""
    base = Path(base)
    todas: list[dict] = []
    if tese is not None:
        todas.extend(fatiar_tese(tese, base, max_palavras))
    if slides is not None:
        todas.extend(fatiar_slides(slides, base))
    if roteiro is not None:
        todas.extend(fatiar_roteiro(roteiro, base, max_palavras))
    vistos: set[str] = set()
    unicas = []
    for fatia in todas:
        if fatia["id"] in vistos:
            continue
        vistos.add(fatia["id"])
        unicas.append(fatia)
    return unicas
