"""Contrato de testes do deck de defesa (text/slides/defesa.tex) e do roteiro de fala
(text/slides/roteiro_defesa.md).

Liga o deck novo à tese: estrutura (25 quadros principais e 14 de reserva), figuras,
remissões, terminologia do guia de estilo, glosas, ressalvas das âncoras aprovadas
(tests/data/registro_invariantes.yaml), números contra o texto da tese e orçamento de tempo.

As funções de extração (quadros por label, texto sem comandos, seções do roteiro, números,
contagem de palavras) têm teste com caso sintético, para que um erro de extração não vire
aprovação vazia.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import registro_invariantes as ri  # noqa: E402
from portao_fidelidade import normalizar_numero  # noqa: E402

DECK = ROOT / "text/slides/defesa.tex"
ROTEIRO = ROOT / "text/slides/roteiro_defesa.md"
SLIDES_DIR = DECK.parent
TESE_DIR = ROOT / "text/latex"
TESE_TEX = TESE_DIR / "tese.tex"
CAPITULO3 = TESE_DIR / "capitulo3.tex"

PRINCIPAIS = [f"S{i:02d}" for i in range(1, 26)]
RESERVAS = [f"R{i:02d}" for i in range(1, 15)]
TODOS = PRINCIPAIS + RESERVAS

# Cronometragem e orçamento de palavras (130 palavras por minuto).
MAX_SEGUNDOS = 2400
TOTAL_SEGUNDOS = 2280
MAX_PALAVRAS_FALA_TOTAL = 5200
FATOR_META = 1.25
MAX_PALAVRAS_RESPOSTA = 75
MAX_PALAVRAS_RESERVA = 85
# Alvo editorial de 110; o S06 mede 116 (corpo, nota de rodapé e ressalva). O menor valor redondo
# com folga de 5% sobre 116 (121,8) é 125.
MAX_PALAVRAS_QUADRO = 125


def ler(caminho: Path) -> str:
    return caminho.read_text(encoding="utf-8")


# ======================================================================================
# Extração: LaTeX
# ======================================================================================

def sem_comentarios(tex: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", tex)


def grupo(s: str, i: int) -> tuple[str, int]:
    """``s[i]`` é ``{``; devolve (conteúdo, índice seguinte ao ``}`` correspondente)."""
    assert s[i] == "{", f"grupo esperado em {s[i:i + 20]!r}"
    nivel = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            nivel += 1
        elif c == "}":
            nivel -= 1
            if nivel == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError(f"chave sem fechamento: {s[i:i + 40]!r}")


@dataclass
class Quadro:
    label: str
    titulo: str
    corpo: str          # LaTeX do corpo, sem comentários
    inicio: int         # posição de \begin{frame} no texto sem comentários
    apendice: bool = False
    fontes: list[str] = field(default_factory=list)

    @property
    def corpo_sem_fonte(self) -> str:
        return remover_fonte(self.corpo)[0]

    @property
    def ressalva(self) -> str:
        return " ".join(extrair_comando(self.corpo, "ressalva"))


RE_FRAME = re.compile(r"\\begin\{frame\}(?:\[(?P<opt>[^\]]*)\])?")


def extrair_comando(tex: str, nome: str) -> list[str]:
    """Conteúdo do último argumento ``{...}`` de cada ``\\nome`` (após ``[...]`` opcional)."""
    saida = []
    for m in re.finditer(r"\\" + nome + r"(?![A-Za-z])", tex):
        pos = m.end()
        if tex[pos:pos + 1] == "[":
            pos = tex.index("]", pos) + 1
        if tex[pos:pos + 1] == "{":
            conteudo, _ = grupo(tex, pos)
            saida.append(conteudo)
    return saida


def remover_fonte(tex: str) -> tuple[str, list[str]]:
    """Remove cada ``\\fonte[...]{...}``; devolve (texto sem fontes, conteúdos)."""
    partes, fontes, ultimo = [], [], 0
    for m in re.finditer(r"\\fonte(?![A-Za-z])", tex):
        if m.start() < ultimo:
            continue
        pos = m.end()
        if tex[pos:pos + 1] == "[":
            pos = tex.index("]", pos) + 1
        conteudo, fim = grupo(tex, pos)
        fontes.append(conteudo)
        partes.append(tex[ultimo:m.start()])
        ultimo = fim
    partes.append(tex[ultimo:])
    return "".join(partes), fontes


def extrair_quadros(tex: str) -> list[Quadro]:
    tex = sem_comentarios(tex)
    pos_apendice = tex.find(r"\appendix")
    quadros = []
    for m in RE_FRAME.finditer(tex):
        pos = m.end()
        titulo = ""
        if tex[pos:pos + 1] == "{":
            titulo, pos = grupo(tex, pos)
        fim = tex.index(r"\end{frame}", pos)
        corpo = tex[pos:fim]
        rotulo = re.search(r"label=([A-Za-z0-9_]+)", m.group("opt") or "")
        q = Quadro(label=rotulo.group(1) if rotulo else "", titulo=titulo, corpo=corpo,
                   inicio=m.start(), apendice=0 <= pos_apendice < m.start())
        q.fontes = remover_fonte(corpo)[1]
        quadros.append(q)
    return quadros


_FORMATACAO = ("textit", "textbf", "texttt", "emph", "textsc", "textrm")
_ACENTOS = {"'": "\u0301", "`": "\u0300", "^": "\u0302", "~": "\u0303", '"': "\u0308", "c": "\u0327"}


def _aplicar_args(tex: str, nome: str, padrao: str, opcional: bool = True) -> str:
    """Aplica ``padrao`` ('d' descarta, 'k' mantém) aos argumentos ``{...}`` de ``\\nome``."""
    saida, ultimo = [], 0
    for m in re.finditer(r"\\" + nome + r"(?![A-Za-z])", tex):
        if m.start() < ultimo:
            continue
        pos = m.end()
        if opcional:
            while tex[pos:pos + 1] == "[":
                pos = tex.index("]", pos) + 1
        mantido = []
        ok = True
        for acao in padrao:
            while tex[pos:pos + 1] == " ":
                pos += 1
            if tex[pos:pos + 1] != "{":
                ok = False
                break
            conteudo, pos = grupo(tex, pos)
            if acao == "k":
                mantido.append(conteudo)
        if not ok:
            continue
        saida.append(tex[ultimo:m.start()])
        saida.append(" ".join(mantido))
        ultimo = pos
    saida.append(tex[ultimo:])
    return "".join(saida)


def _texto_dos_nos(tikz: str) -> str:
    """Texto dos ``\\node`` de um tikzpicture (e dos ``label`` de nós de moldura)."""
    textos = []
    for m in re.finditer(r"\\node", tikz):
        i = m.end()
        nivel_ch = nivel_col = 0
        grupos = []
        while i < len(tikz):
            c = tikz[i]
            if c == "\\":
                i += 2
                continue
            if c == "{":
                conteudo, fim = grupo(tikz, i)
                if nivel_col == 0:
                    grupos.append(conteudo)
                i = fim
                continue
            if c == "[":
                nivel_col += 1
            elif c == "]":
                nivel_col -= 1
            elif c == ";" and nivel_col == 0:
                break
            i += 1
        trecho = tikz[m.end():i]
        textos.extend(re.findall(r"below:([^{}]*)", trecho))
        if grupos:
            textos.append(grupos[-1])
    return " ; ".join(textos)


def plano(tex: str) -> str:
    """Texto visível do LaTeX: sem comentários, comandos de layout, medidas, fórmulas
    convertidas em texto, conteúdo de figuras incluídas e coordenadas de tikz."""
    t = sem_comentarios(tex)
    t = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
               lambda m: _texto_dos_nos(m.group(0)), t, flags=re.S)
    t = _aplicar_args(t, "resizebox", "ddk")
    t = _aplicar_args(t, "multicolumn", "ddk")
    for nome, padrao in (("input", "d"), ("renewcommand", "dd"), ("vspace", "d"),
                         ("mapa", "d"), ("cxn", "k"), ("label", "d"), ("ressalva", "k")):
        t = _aplicar_args(t, nome, padrao)
    t = re.sub(r"\\begin\{(?:tabular|column|columns)\}(?:\[[^\]]*\])?(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})?",
               " ", t)
    t = re.sub(r"\\addlinespace(?:\[[^\]]*\])?|\\cmidrule(?:\([^)]*\))?\{[^}]*\}", " ", t)
    for nome in _FORMATACAO:
        t = _aplicar_args(t, nome, "k")

    def matematica(m: re.Match) -> str:
        interno = m.group(1)
        interno = re.sub(r"\\[A-Za-z]+", " ", interno)
        return re.sub(r"[_^]", "", interno)

    t = re.sub(r"\$([^$]*)\$", matematica, t)
    t = t.replace("{,}", ",").replace("{.}", ".")
    t = t.replace(r"\\", " ").replace("&", " ")
    t = t.replace("``", '"').replace("''", '"')
    t = re.sub(r"\\([%_#&$])", r"\1", t)
    t = t.replace("\\,", "")
    t = re.sub(r"\\[;:! ]", " ", t)
    t = re.sub(r"\\[A-Za-z]+\*?", " ", t)
    t = t.replace("~", " ").replace("{", "").replace("}", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def texto_quadro(q: Quadro, com_titulo: bool = True) -> str:
    """Texto visível do quadro (título, corpo, tabelas, ressalva), sem ``\\fonte``."""
    partes = [plano(q.titulo)] if com_titulo and q.titulo else []
    partes.append(plano(q.corpo_sem_fonte))
    return " ".join(p for p in partes if p)


# ======================================================================================
# Extração: roteiro (markdown)
# ======================================================================================

RE_TITULO_ROTEIRO = re.compile(r"^(#{2,3}) (?P<rotulo>[SR]\d\d)\. (?P<titulo>.*)$", re.M)


@dataclass
class SecaoRoteiro:
    rotulo: str
    titulo: str
    paragrafos: list[str]

    def _com(self, prefixo: str) -> list[str]:
        return [p for p in self.paragrafos if p.startswith(prefixo)]

    @property
    def transicoes(self) -> list[str]:
        return self._com("Transição:")

    @property
    def fontes(self) -> list[str]:
        return self._com("Fonte:")

    @property
    def perguntarem(self) -> list[str]:
        return self._com("Se perguntarem:")

    @property
    def fala(self) -> list[str]:
        meta = ("Transição:", "Se perguntarem:", "Fonte:")
        return [p for p in self.paragrafos if not p.startswith(meta)]


def secoes_roteiro(md: str) -> dict[str, SecaoRoteiro]:
    """Seções ``### Sxx.``/``### Rxx.``; cada uma vai até o próximo título."""
    blocos = re.split(r"(?m)^(?=#{1,3} )", md)
    saida: dict[str, SecaoRoteiro] = {}
    for bloco in blocos:
        m = RE_TITULO_ROTEIRO.match(bloco)
        if not m:
            continue
        corpo = bloco[m.end():].strip()
        paragrafos = [" ".join(p.split()) for p in re.split(r"\n\s*\n", corpo) if p.strip()]
        saida[m.group("rotulo")] = SecaoRoteiro(m.group("rotulo"), m.group("titulo"), paragrafos)
    return saida


@dataclass
class Pergunta:
    rotulo: str
    enunciado: str
    resposta: str
    apoio: str


RE_PERGUNTA = re.compile(r"^\*\*(?P<p>P\d+)\. (?P<enunciado>[^\n]*)\*\*\n(?P<corpo>.*?)(?=^\*\*P\d+\.|^#{1,3} |\Z)",
                         re.M | re.S)


def perguntas_roteiro(md: str) -> list[Pergunta]:
    saida = []
    for m in RE_PERGUNTA.finditer(md):
        linhas = [l.strip() for l in m.group("corpo").strip().splitlines() if l.strip()]
        resposta = [l for l in linhas if not l.startswith("Apoio:")]
        apoio = [l for l in linhas if l.startswith("Apoio:")]
        saida.append(Pergunta(m.group("p"), m.group("enunciado"), " ".join(resposta),
                              apoio[0] if apoio else ""))
    return saida


def sem_fonte_md(texto: str) -> str:
    """Remove trechos 'Fonte: ...' (linha inteira ou final de linha 'Apoio: ... Fonte: ...')
    e caminhos de arquivo entre crases."""
    linhas = []
    for l in texto.splitlines():
        if l.startswith("Fonte:"):
            continue
        l = re.sub(r"\s+Fonte:.*$", "", l) if l.startswith("Apoio:") else l
        linhas.append(l)
    return re.sub(r"`[^`\n]*`", " ", "\n".join(linhas))


def plano_md(texto: str) -> str:
    t = sem_fonte_md(texto)
    t = t.replace("*", "")
    return re.sub(r"\s+", " ", t).strip()


def tabela_cronometragem(md: str) -> dict[str, tuple[int, int]]:
    """``{quadro: (segundos, meta de palavras)}`` da tabela por quadro."""
    saida = {}
    for l in md.splitlines():
        cel = [c.strip() for c in l.strip().strip("|").split("|")]
        if len(cel) == 4 and re.fullmatch(r"S\d\d", cel[0]):
            saida[cel[0]] = (int(cel[2].replace(".", "")), int(cel[3].replace(".", "")))
    return saida


def linha_total(md: str) -> tuple[int, int]:
    for l in md.splitlines():
        cel = [c.strip() for c in l.strip().strip("|").split("|")]
        if len(cel) == 4 and cel[0] == "Total" and cel[1] == "":
            return int(cel[2].replace(".", "")), int(cel[3].replace(".", ""))
    raise AssertionError("linha 'Total' da tabela por quadro não encontrada")


# ======================================================================================
# Contagem de palavras (função única)
# ======================================================================================

def contar_palavras(texto: str) -> int:
    """Fichas separadas por espaço com ao menos um caractere alfanumérico."""
    return sum(1 for ficha in texto.split() if re.search(r"\w", ficha))


# ======================================================================================
# Números
# ======================================================================================

RE_NUM = re.compile(r"(?<![\w])\d+(?:[.,]\d+)*")
# Exceções explícitas ao confronto de números com a tese, com motivo e fonte.
EXCECOES_NUMEROS: dict[str, str] = {
    "3.12": ("roteiro, P20: 'Python 3.12' é a versão 3.12.4 da tese (text/latex/capitulo_dados.tex, "
             "linha 115, 'versão 3.12.4 na estação local'), abreviada na fala; sem valor novo"),
}


def _limpar_numeros(texto: str) -> str:
    t = texto.replace("{,}", ",").replace("{.}", ".").replace("\u2212", "-")
    t = re.sub(r"(?<=\d)(?:\\,|\u00a0|\u202f|\u2009)(?=\d{3}(?!\d))", "", t)
    return t.replace("~", " ")


def tokens_numericos(texto: str) -> list[str]:
    """Números do texto na forma normalizada, sem os isentos: identificadores (O3, SHA-256),
    inteiros de 0 a 12 e anos entre parênteses (citações)."""
    t = _limpar_numeros(texto)
    anos_citacao = set()
    for par in re.finditer(r"\([^()]*\)", t):
        for a in re.finditer(r"(?<!\d)(?:19|20)\d\d(?!\d)", par.group(0)):
            anos_citacao.add(par.start() + a.start())
    saida = []
    for m in re.finditer(r"\d+(?:[.,]\d+)*", t):
        antes = t[max(0, m.start() - 2):m.start()]
        if antes[-1:].isalpha() or (antes[-1:] == "-" and antes[-2:-1].isalpha()):
            continue  # identificador (O3, R05, SHA-256, GPL-3.0)
        if antes[-1:] in ("_", "^"):
            continue
        bruto = m.group(0)
        if re.fullmatch(r"\d+", bruto) and int(bruto) <= 12:
            continue
        if m.start() in anos_citacao and re.fullmatch(r"\d{4}", bruto):
            continue
        saida.append(normalizar_numero(bruto))
    return saida


def numeros_da_tese() -> set[str]:
    """Todos os números do texto da tese (``text/latex/**/*.tex`` exceto ``tese_rev_diff.tex``)
    e dos ``texto_pt`` do registro, na forma normalizada."""
    conjunto: set[str] = set()
    for caminho in sorted(TESE_DIR.rglob("*.tex")):
        if caminho.name == "tese_rev_diff.tex":
            continue
        t = _limpar_numeros(sem_comentarios(ler(caminho)))
        conjunto.update(normalizar_numero(m.group(0)) for m in re.finditer(r"\d+(?:[.,]\d+)*", t))
    conjunto.update(numeros_do_registro(ri.carregar_registro()))
    return conjunto


def numeros_do_registro(reg: ri.Registro) -> set[str]:
    conjunto: set[str] = set()
    for _, valor in reg.valores.values():
        texto = valor.get("texto_pt")
        if texto is None:
            continue
        t = _limpar_numeros(str(texto))
        conjunto.update(normalizar_numero(m.group(0)) for m in re.finditer(r"\d+(?:[.,]\d+)*", t))
    return conjunto


def numeros_ausentes(texto: str, tese: set[str]) -> list[str]:
    """Números de ``texto`` que não estão em ``tese`` nem nas exceções explícitas.
    Comparação pelo valor absoluto (o sinal é ignorado)."""
    return sorted({n for n in tokens_numericos(texto)
                   if n.lstrip("-+") not in tese and n not in EXCECOES_NUMEROS})


# ======================================================================================
# Terminologia
# ======================================================================================

def _lista(*pares):
    return [(re.compile(p, f), desc) for p, f, desc in pares]


PROIBIDOS = _lista(
    (r"[\u2013\u2014]", 0, "travessão Unicode"),
    (r"--", 0, "travessão ASCII (-- ou ---)"),
    (r"zona cinz", re.I, "'zona cinzenta' (usar grey area)"),
    (r"revis[ãa]o clerical", re.I, "'revisão clerical' (usar revisão manual)"),
    (r"n[ãa]o[ -]v[ií]nculo", re.I, "'não vínculo'"),
    (r"n[ãa]o-par\b", re.I, "'não-par' (usar 'não par')"),
    (r"padr[ãa]o-ouro", re.I, "'padrão-ouro' (usar referência operacional)"),
    (r"gold standard", re.I, "'gold standard'"),
    (r"\bPPV\b", 0, "'PPV' (usar VPP)"),
    (r"record linkage", re.I, "'record linkage' (usar linkage)"),
    (r"vincula[çc][ãa]o", re.I, "'vinculação'"),
    (r"perda esperada", re.I, "'perda esperada' (usar expected loss)"),
    (r"r[óo]tulo", re.I, "'rótulo'"),
    (r"semelhan[çc]a", re.I, "'semelhança' (usar similaridade)"),
    (r"0,485", 0, "valor retirado '0,485'"),
    (r"5,648", 0, "valor retirado '5,648'"),
    (r"(?<![\w-])RL(?![\w-])", 0, "sigla 'RL' (usar regressão logística)"),
    (r"\bnota\b(?!\s+final)", re.I, "'nota' sem 'final'"),
    (r"precis[ãa]o(?!\s+m[ée]dia)(?!-sensibilidade)", re.I,
     "'precisão' fora de 'precisão média' e 'precisão-sensibilidade'"),
    (r"(?<!\()[áa]rea cinza", re.I, "'área cinza' fora da glosa '(área cinza)'"),
    (r"(?<!\()relacionamento de registros", re.I,
     "'relacionamento de registros' fora da glosa"),
    (r"\barranjo P\b", 0, "'arranjo P' (usar ASR)"),
    (r"\bP (?:vig|conf)", 0, "'P vig'/'P conf' (usar ASR)"),
    # primeira pessoa (lista curta, sem falso positivo no texto atual)
    (r"\b(?:eu|meu|meus|minha|minhas|nós|nosso|nossa|nossos|nossas)\b", re.I,
     "primeira pessoa"),
    (r"\b(?:apresento|apresentamos|mostro|mostramos|proponho|propomos|concluímos|observamos)\b",
     re.I, "verbo em primeira pessoa"),
)


def termos_proibidos(texto: str) -> list[str]:
    achados = []
    for padrao, descricao in PROIBIDOS:
        for m in padrao.finditer(texto):
            ini = max(0, m.start() - 40)
            achados.append(f"{descricao}: ...{texto[ini:m.end() + 40]}...")
    return achados


# ======================================================================================
# Fixtures
# ======================================================================================

@pytest.fixture(scope="module")
def deck_tex() -> str:
    return ler(DECK)


@pytest.fixture(scope="module")
def quadros(deck_tex) -> dict[str, Quadro]:
    return {q.label: q for q in extrair_quadros(deck_tex)}


@pytest.fixture(scope="module")
def lista_quadros(deck_tex) -> list[Quadro]:
    return extrair_quadros(deck_tex)


@pytest.fixture(scope="module")
def roteiro() -> str:
    return ler(ROTEIRO)


@pytest.fixture(scope="module")
def secoes(roteiro) -> dict[str, SecaoRoteiro]:
    return secoes_roteiro(roteiro)


@pytest.fixture(scope="module")
def registro() -> ri.Registro:
    return ri.carregar_registro()


@pytest.fixture(scope="module")
def tese_numeros() -> set[str]:
    return numeros_da_tese()


# ======================================================================================
# Casos sintéticos das funções de extração
# ======================================================================================

TEX_SINTETICO = r"""
\documentclass{beamer}
\newcommand{\mapa}[1]{\gdef\x{#1}}
\begin{document}
% S01: comentário que não conta; 999 palavras
\begin{frame}[label=S01]{Título de 61.485 itens}
\mapa{3}
\small
Texto com \textit{grey area} (área cinza) e 11,2\% dos pares. % comentário 888
\itens{\item primeiro, 1.234 \item segundo}
\resizebox{!}{.55\textheight}{\input{figures/x.pgf}}
\begin{tabular}{@{}cL{.64\textwidth}@{}}
\cxn{1} & linha com $-52$ e $0{,}9$ \\
\end{tabular}
\ressalva{cautela \textit{post hoc}.}
\fonte{capitulo3.tex, seção ``Justificativa''; 777}
\end{frame}
\appendix
\begin{frame}[label=R01]{Reserva}
\begin{tikzpicture}[caixa/.style={minimum height=10mm}]
\node[caixa] (a) at (1.7,0) {primeiro nó};
\node[draw, fit=(a), label={[font=\small]below:rótulo da moldura}] (pp) {};
\draw (a) to (pp);
\end{tikzpicture}
\fonte{tese.tex}
\end{frame}
\end{document}
"""


def test_extrair_quadros_sintetico():
    qs = extrair_quadros(TEX_SINTETICO)
    assert [q.label for q in qs] == ["S01", "R01"]
    assert [q.apendice for q in qs] == [False, True]
    s01, r01 = qs
    assert s01.titulo == "Título de 61.485 itens"
    assert len(s01.fontes) == 1 and "777" in s01.fontes[0]
    assert "888" not in s01.corpo and "999" not in "".join(q.corpo for q in qs)
    assert s01.ressalva == r"cautela \textit{post hoc}."
    assert "777" not in s01.corpo_sem_fonte
    assert len(r01.fontes) == 1


def test_plano_sintetico_remove_layout_e_mantem_conteudo():
    s01 = extrair_quadros(TEX_SINTETICO)[0]
    p = texto_quadro(s01)
    assert "grey area (área cinza)" in p
    assert "11,2%" in p
    assert "-52 e 0,9" in p
    assert "cautela post hoc." in p
    # layout, medidas, figura incluída, \fonte e comentários não aparecem
    for fora in ("0.64", ".64", "textwidth", "textheight", "figures/x.pgf", "777", "888", "pgf"):
        assert fora not in p, fora
    assert not re.search(r"\\[A-Za-z]", p)


def test_plano_tikz_mantem_so_o_texto_dos_nos():
    r01 = extrair_quadros(TEX_SINTETICO)[1]
    p = texto_quadro(r01)
    assert "primeiro nó" in p and "rótulo da moldura" in p
    assert "1.7" not in p and "minimum" not in p and "draw" not in p


def test_grupo_balanceado_e_barra_invertida():
    conteudo, fim = grupo(r"{a {b} \} c} resto", 0)
    assert conteudo == r"a {b} \} c" and fim == len(r"{a {b} \} c}")
    with pytest.raises(ValueError):
        grupo("{aberto", 0)


MD_SINTETICO = """# Roteiro

## Fala-guia por quadro

### S01. Capa

Primeira fala com cinco palavras.

Segunda fala.

Transição: ponte para o seguinte.

Se perguntarem: R01.

Fonte: `tese.tex:1`.

### S02. Outro

Fala dois.

Transição: fim.

Fonte: `a.tex:2`.

## Reserva

### R01. Reserva um

Fala de reserva.

Fonte: `x.tex:3`.

## Perguntas prováveis

### Tema

**P1. Pergunta um?**

Resposta um com seis palavras aqui.

Apoio: S01, R01. Fonte: `x.tex:4`.

**P2. Pergunta dois?**

Resposta dois.

Apoio: sem quadro de apoio. Fonte: `y.tex:5`.

## Siglas e termos
"""


def test_secoes_roteiro_sintetico():
    s = secoes_roteiro(MD_SINTETICO)
    assert list(s) == ["S01", "S02", "R01"]
    assert s["S01"].fala == ["Primeira fala com cinco palavras.", "Segunda fala."]
    assert s["S01"].transicoes == ["Transição: ponte para o seguinte."]
    assert s["S01"].perguntarem == ["Se perguntarem: R01."]
    assert len(s["S01"].fontes) == 1 and len(s["S02"].fontes) == 1
    assert s["R01"].fala == ["Fala de reserva."]
    # a última seção de reserva não engole as perguntas
    assert "Pergunta" not in " ".join(s["R01"].paragrafos)


def test_perguntas_roteiro_sintetico():
    ps = perguntas_roteiro(MD_SINTETICO)
    assert [p.rotulo for p in ps] == ["P1", "P2"]
    assert ps[0].resposta == "Resposta um com seis palavras aqui."
    assert ps[0].apoio.startswith("Apoio: S01, R01.")
    assert contar_palavras(ps[0].resposta) == 6


def test_sem_fonte_md_sintetico():
    t = sem_fonte_md("Fala `a_b.tex:3` fim.\nFonte: `x.tex`, 61.485.\nApoio: S01. Fonte: 99.")
    assert "61.485" not in t and "99" not in t and "a_b" not in t
    assert "Fala" in t and "Apoio: S01." in t


def test_tabela_cronometragem_sintetica():
    md = "| S01 | Capa | 20 | 43 |\n| S02 | Outro | 1.080 | 2.340 |\n| Total | | 1.100 | 2.383 |\n"
    assert tabela_cronometragem(md) == {"S01": (20, 43), "S02": (1080, 2340)}
    assert linha_total(md) == (1100, 2383)


def test_contar_palavras_funcao_unica():
    assert contar_palavras("a b, c-d 1.234 | -- ") == 4
    assert contar_palavras("") == 0


# --- números ---------------------------------------------------------------------------

def test_tokens_numericos_isencoes():
    t = ("Em O3 e F4, R05, S09 e SHA-256; 7 casos; 12 itens; (Brasil, 2026); "
         "61.485 e 0{,}910 e 1\\,234 e $-21$ e 2.000 réplicas")
    assert tokens_numericos(t) == ["61485", "0.910", "1234", "21", "2000"]
    assert tokens_numericos("13 casos") == ["13"]
    assert tokens_numericos("Em 2019 a 2022") == ["2019", "2022"]  # ano fora de citação conta


def test_numero_ausente_da_tese_faz_a_checagem_falhar():
    tese = {"61485", "0.910"}
    sintetico = extrair_quadros(TEX_SINTETICO)[0]
    ausentes = numeros_ausentes(texto_quadro(sintetico), tese)
    # 1.234 e 11,2 não estão na "tese" sintética; 61.485 e 0,9 (inteiro menor) idem
    assert "1234" in ausentes and "11.2" in ausentes
    assert "61485" not in ausentes
    assert numeros_ausentes("Valor 61.485 e 0,910.", tese) == []
    assert numeros_ausentes("Valor 99.999.", tese) == ["99999"]
    # sinal ignorado
    assert numeros_ausentes("$-61.485$", tese) == []


def test_numeros_normalizam_virgula_e_ponto_de_milhar():
    assert tokens_numericos("1.172,5 e 39,3 e 84.368") == ["1172.5", "39.3", "84368"]


def test_numeros_do_registro_usa_texto_pt():
    reg = ri.Registro([{"id": "X", "valores": [{"nome": "v", "texto_pt": "61.485"},
                                               {"nome": "w", "texto_pt": "1:249"}]}])
    assert numeros_do_registro(reg) >= {"61485", "249"}


# --- terminologia ---------------------------------------------------------------------

@pytest.mark.parametrize("texto", [
    "a revisão clerical dos pares", "zona cinzenta", "pares não vínculo", "o não-vínculo",
    "um não-par", "o padrão-ouro", "o gold standard", "o PPV", "o record linkage",
    "a vinculação", "a perda esperada", "o rótulo", "a semelhança", "valor 0,485", "valor 5,648",
    "a sigla RL", "a nota do CdR", "a precisão do modelo", "a área cinza sem glosa",
    "o relacionamento de registros sem glosa", "o arranjo P", "P vig e P conf",
    "eu acho", "o nosso estudo", "mostramos que", "isto — aquilo", "isto – aquilo", "isto -- aquilo",
])
def test_termos_proibidos_detecta(texto):
    assert termos_proibidos(texto), texto


@pytest.mark.parametrize("texto", [
    "a nota final do CdR", "o CIDACS-RL", "a precisão média", "a curva precisão-sensibilidade",
    "grey area (área cinza)", "linkage (relacionamento de registros)", "anotados e notificação",
    "o arranjo sequencial (ASR)", "a regressão logística", "referência operacional",
])
def test_termos_proibidos_nao_acusa_texto_limpo(texto):
    assert termos_proibidos(texto) == []


# ======================================================================================
# A1. Estrutura do deck
# ======================================================================================

def test_a1_labels_s01_a_s25_e_r01_a_r14_unicos_e_em_ordem(lista_quadros):
    assert [q.label for q in lista_quadros] == TODOS


def test_a1_appendix_entre_s25_e_r01(deck_tex, quadros):
    t = sem_comentarios(deck_tex)
    assert t.count(r"\appendix") == 1
    pos = t.index(r"\appendix")
    assert quadros["S25"].inicio < pos < quadros["R01"].inicio
    assert [q.label for q in extrair_quadros(deck_tex) if not q.apendice] == PRINCIPAIS
    assert [q.label for q in extrair_quadros(deck_tex) if q.apendice] == RESERVAS


@pytest.mark.parametrize("label", TODOS)
def test_a1_cada_quadro_tem_exatamente_um_fonte(quadros, label):
    assert len(quadros[label].fontes) == 1, f"{label}: {len(quadros[label].fontes)} \\fonte"
    assert quadros[label].fontes[0].strip(), f"{label}: \\fonte vazio"


MAPA_ESPERADO = {"S11": [1], "S12": [2], "S13": [2], "S14": [3], "S15": [3], "S16": [3],
                 "S17": [4], "S18": [4], "S19": [5]}


def test_a1_mapa_so_nos_quadros_previstos(quadros):
    obtido = {l: [int(x) for x in extrair_comando(q.corpo, "mapa")] for l, q in quadros.items()}
    obtido = {l: v for l, v in obtido.items() if v}
    assert obtido == MAPA_ESPERADO


def test_a1_tema_default_sem_metropolis(deck_tex):
    assert r"\usetheme{default}" in sem_comentarios(deck_tex)
    assert "metropolis" not in deck_tex.lower()


def _normalizar_titulo(s: str) -> str:
    s = s.replace("\\\\", " ")
    s = re.sub(r"\\([`'^\"~])\s*\{?(\\i|[A-Za-z])\}?", _acento_titulo, s)
    for nome in _FORMATACAO:
        s = _aplicar_args(s, nome, "k")
    s = re.sub(r"\\[A-Za-z]+\*?", " ", s).replace("{", "").replace("}", "")
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", s).strip())


def _acento_titulo(m: re.Match) -> str:
    base = "i" if m.group(2) == "\\i" else m.group(2)
    return unicodedata.normalize("NFC", base + _ACENTOS[m.group(1)])


def titulo_da_tese() -> str:
    for linha in ler(TESE_TEX).splitlines():
        if linha.startswith(r"{\Large \bf Aprendizado de M"):
            conteudo, _ = grupo(linha, 0)
            return _normalizar_titulo(re.sub(r"^\\Large\s*\\bf\s*", "", conteudo))
    raise AssertionError("título não encontrado na folha de rosto de tese.tex")


def test_a1_titulo_do_deck_igual_ao_da_tese(deck_tex):
    titulo_deck = _normalizar_titulo(extrair_comando(sem_comentarios(deck_tex), "title")[0])
    assert titulo_deck == titulo_da_tese()
    assert titulo_deck.startswith("Aprendizado de Máquina Aplicado ao Pós-Processamento")


def test_a1_normalizar_titulo_sintetico():
    assert _normalizar_titulo(r"Sa\'ude\\do \textit{Linkage} Probabil\'istico") == \
        "Saúde do Linkage Probabilístico"


def objetivo_geral_da_tese() -> str:
    tex = sem_comentarios(ler(CAPITULO3))
    m = re.search(r"\\section\{Objetivo geral\}[^\n]*\n\s*\n(?P<p>[^\n]+)", tex)
    assert m, "seção 'Objetivo geral' não encontrada em capitulo3.tex"
    return " ".join(m.group("p").split())


def test_a1_s04_contem_literalmente_o_objetivo_geral(quadros):
    objetivo = objetivo_geral_da_tese()
    assert objetivo.startswith("Desenvolver e avaliar um pós-processamento")
    s04 = " ".join(quadros["S04"].corpo_sem_fonte.split())
    assert objetivo in s04, "S04 não contém o objetivo geral de capitulo3.tex"


# ======================================================================================
# A2. Figuras
# ======================================================================================

def _inputs(q: Quadro) -> list[str]:
    return [i.strip() for i in extrair_comando(q.corpo, "input")]


def test_a2_s14_inclui_a_figura_da_tese(quadros):
    assert _inputs(quadros["S14"]) == ["../latex/figures/curva_esforco.pgf"]
    assert (SLIDES_DIR / "../latex/figures/curva_esforco.pgf").resolve().is_file()


def test_a2_s15_inclui_a_figura_de_incerteza(quadros):
    assert _inputs(quadros["S15"]) == ["figures/incerteza_ordenacoes.pgf"]
    assert (SLIDES_DIR / "figures/incerteza_ordenacoes.pgf").is_file()


def test_a2_deck_nao_inclui_a_variante_antiga(quadros):
    todos = [i for q in quadros.values() for i in _inputs(q)]
    assert "figures/curva_esforco.pgf" not in todos, "variante antiga da curva de esforço no deck"
    assert sorted(todos) == sorted(["../latex/figures/curva_esforco.pgf",
                                    "figures/incerteza_ordenacoes.pgf"])


# ======================================================================================
# A3. Remissões
# ======================================================================================

RE_RESERVAS = re.compile(r"reservas?\s+(R\d\d(?:\s*(?:,|e)\s*R\d\d)*)")


def remissoes_reserva(texto: str) -> list[str]:
    return [r for m in RE_RESERVAS.finditer(texto) for r in re.findall(r"R\d\d", m.group(1))]


def test_a3_remissoes_sinteticas():
    assert remissoes_reserva("ver reservas R05, R10 e R11 e (reserva R03)") == ["R05", "R10", "R11", "R03"]
    assert remissoes_reserva("sem remissão") == []


def test_a3_reservas_citadas_no_deck_existem(quadros):
    total = 0
    for label, q in quadros.items():
        for r in remissoes_reserva(texto_quadro(q)):
            total += 1
            assert r in quadros and quadros[r].apendice, f"{label}: remissão a '{r}' sem quadro de reserva"
    assert total >= 15, f"poucas remissões encontradas ({total}); extração suspeita"


def test_a3_roteiro_tem_todas_as_secoes(secoes):
    assert [r for r in secoes if r.startswith("S")] == PRINCIPAIS
    assert [r for r in secoes if r.startswith("R")] == RESERVAS


@pytest.mark.parametrize("rotulo", PRINCIPAIS)
def test_a3_secao_s_tem_transicao_e_fonte(secoes, rotulo):
    s = secoes[rotulo]
    assert len(s.transicoes) == 1 and len(s.transicoes[0]) > len("Transição:") + 5, f"{rotulo}: Transição"
    assert len(s.fontes) == 1 and len(s.fontes[0]) > len("Fonte:") + 3, f"{rotulo}: Fonte"
    assert len(s.perguntarem) == 1, f"{rotulo}: Se perguntarem"


def test_a3_se_perguntarem_aponta_para_labels_existentes(secoes, quadros):
    total = 0
    for rotulo, s in secoes.items():
        for linha in s.perguntarem:
            for ref in re.findall(r"\b[SR]\d\d\b", linha):
                total += 1
                assert ref in quadros, f"{rotulo}, 'Se perguntarem': '{ref}' não é quadro do deck"
    assert total >= 20, f"poucas remissões encontradas ({total}); extração suspeita"


def test_a3_apoio_das_perguntas_aponta_para_labels_existentes(roteiro, quadros):
    perguntas = perguntas_roteiro(roteiro)
    assert len(perguntas) == 43
    assert [p.rotulo for p in perguntas] == [f"P{i}" for i in range(1, 44)]
    total = 0
    for p in perguntas:
        assert p.apoio, f"{p.rotulo}: sem linha 'Apoio:'"
        apoio = p.apoio.split("Fonte:")[0]
        for ref in re.findall(r"\b[SR]\d\d\b", apoio):
            total += 1
            assert ref in quadros, f"{p.rotulo}, 'Apoio': '{ref}' não é quadro do deck"
    assert total >= 40, f"poucas remissões encontradas ({total}); extração suspeita"


# ======================================================================================
# A4. Terminologia proibida
# ======================================================================================

def test_a4_deck_sem_termos_proibidos(quadros):
    problemas = []
    for label, q in quadros.items():
        problemas += [f"{label}: {a}" for a in termos_proibidos(texto_quadro(q))]
    assert not problemas, "\n".join(problemas)


def test_a4_roteiro_sem_termos_proibidos(roteiro):
    problemas = termos_proibidos(plano_md(roteiro))
    assert not problemas, "\n".join(problemas)


def test_a4_deck_sem_so_f4_nem_rrf_no_lugar(quadros):
    for label, q in quadros.items():
        t = texto_quadro(q).lower()
        assert "só f4" not in t, f"{label}: 'só F4'"
        assert "rrf no lugar" not in t, f"{label}: 'RRF no lugar'"


# ======================================================================================
# A5. Glosas
# ======================================================================================

def _ocorrencias(texto: str, expressao: str) -> int:
    return len(re.findall(expressao, texto, re.I))


def _roteiro_sem_siglas(roteiro: str) -> str:
    """A tabela 'Siglas e termos' é o próprio glossário: fica fora da contagem de glosas."""
    return roteiro.split("## Siglas e termos")[0]


GLOSAS = (("relacionamento de registros", "S02"), ("área cinza", "S03"),
          ("valor preditivo positivo", "S09"))


@pytest.mark.parametrize("expressao,rotulo", GLOSAS)
def test_a5_glosa_uma_vez_no_deck_e_no_quadro_certo(quadros, expressao, rotulo):
    por_quadro = {l: _ocorrencias(texto_quadro(q), expressao) for l, q in quadros.items()}
    assert sum(por_quadro.values()) == 1, f"'{expressao}' no deck: {por_quadro}"
    assert por_quadro[rotulo] == 1, f"'{expressao}' deveria estar em {rotulo}"


@pytest.mark.parametrize("expressao,rotulo", GLOSAS)
def test_a5_glosa_uma_vez_no_roteiro(roteiro, expressao, rotulo):
    texto = plano_md(_roteiro_sem_siglas(roteiro))
    assert _ocorrencias(texto, expressao) == 1, f"'{expressao}' no roteiro"


def test_a5_grey_area_vpp_e_asr_so_depois_da_glosa(quadros):
    for rotulo in PRINCIPAIS[:2]:  # S01, S02
        assert not re.search(r"grey area", texto_quadro(quadros[rotulo]), re.I), f"{rotulo}: grey area antes de S03"
    for rotulo in PRINCIPAIS[:8]:  # S01 a S08
        t = texto_quadro(quadros[rotulo])
        assert not re.search(r"\bVPP\b", t), f"{rotulo}: VPP antes de S09"
        assert not re.search(r"\bASR\b", t), f"{rotulo}: ASR antes de S09"


def test_a5_glosa_de_linkage_antecede_o_uso_no_deck(quadros):
    assert "(relacionamento de registros)" in texto_quadro(quadros["S02"])
    assert "(área cinza)" in texto_quadro(quadros["S03"])
    assert "valor preditivo positivo (VPP)" in texto_quadro(quadros["S09"])


# ======================================================================================
# A6. Âncoras por quadro
# ======================================================================================

# (âncora do registro, quadro, expressões em minúsculas no texto do quadro: título, corpo e
# \ressalva; sem comandos LaTeX)
ANCORAS = [
    ("C10", "S03", ["61.485", "grey area"]),
    ("C10", "S04", ["grey area", "preservando os vínculos aceitos pelo corte"]),
    ("C01r", "S06", ["post hoc", "coorte única", "revisor perfeito simulado", "a validar",
                     "sem ic que exclua zero", "seis variantes", "nenhuma superou"]),
    ("C12", "S06", ["post hoc", "coorte única", "revisor perfeito simulado", "a validar",
                    "sem ic que exclua zero", "seis variantes", "nenhuma superou"]),
    ("C03r", "S07", ["revisora única", "sem dupla revisão", "hipótese de trabalho"]),
    ("C15", "S11", ["equivalência não demonstrada"]),
    ("C16", "S12", ["otimistas", "inclui zero"]),
    ("C05r", "S12", ["otimistas", "inclui zero"]),
    ("C11", "S14", ["estimativa pontual", "sem ic que exclua zero"]),
    ("C12", "S15", ["nenhuma fusão superou o3", "seis variantes"]),
    ("C12", "S16", ["entre variantes"]),
    ("C13", "S17", ["16,0", "não deve aceitar vínculos sozinho"]),
    ("C13", "S18", ["não sustenta superioridade"]),
    ("C14", "S19", ["42,5", "não demonstra paridade", "raça/cor"]),
    ("C08r", "S22", ["o3 ou f4", "f3 quando houver as macd", "curva de esforço", "não examinados", "só apoia"]),
    ("C09r", "S22", ["o3 ou f4", "f3 quando houver as macd", "curva de esforço", "não examinados", "só apoia"]),
    ("C13", "S22", ["o3 ou f4", "f3 quando houver as macd", "curva de esforço", "não examinados", "só apoia"]),
    ("C02r", "S23", ["25", "dados à vista"]),
]


def expressoes_ausentes(texto: str, expressoes: list[str]) -> list[str]:
    t = " ".join(texto.lower().split())
    return [e for e in expressoes if e not in t]


def test_a6_expressoes_ausentes_sintetico():
    assert expressoes_ausentes("Texto com Post Hoc  e  coorte única", ["post hoc", "coorte única"]) == []
    assert expressoes_ausentes("Texto", ["post hoc"]) == ["post hoc"]


def test_a6_identificadores_existem_no_registro(registro):
    for ancora, _, _ in ANCORAS:
        assert ancora in registro.por_id, f"âncora {ancora} ausente de registro_invariantes.yaml"
    assert "C99" not in registro.por_id


@pytest.mark.parametrize("ancora,rotulo,expressoes", ANCORAS,
                         ids=[f"{a}-{r}" for a, r, _ in ANCORAS])
def test_a6_quadro_traz_as_expressoes_da_ancora(quadros, ancora, rotulo, expressoes):
    texto = texto_quadro(quadros[rotulo])
    ausentes = expressoes_ausentes(texto, expressoes)
    assert not ausentes, f"{rotulo} ({ancora}): faltam {ausentes}\ntexto do quadro: {texto}"


# ======================================================================================
# A7. Números
# ======================================================================================

def test_a7_a_tese_fornece_numeros(tese_numeros):
    # guarda contra extração vazia da tese
    assert len(tese_numeros) > 1000
    assert {"61485", "61696", "0.910", "1172.5", "23443"} <= tese_numeros


def test_a7_exececoes_sao_explicitas_e_justificadas():
    for numero, motivo in EXCECOES_NUMEROS.items():
        assert len(motivo) > 30, f"exceção {numero} sem motivo e fonte"


def test_a7_numeros_dos_quadros_estao_na_tese(quadros, tese_numeros):
    problemas = []
    total = 0
    for label, q in quadros.items():
        tokens = tokens_numericos(texto_quadro(q))
        total += len(tokens)
        problemas += [f"{label}: {n}" for n in numeros_ausentes(texto_quadro(q), tese_numeros)]
    assert total > 300, f"poucos números extraídos do deck ({total}); extração suspeita"
    assert not problemas, "números do deck fora da tese e do registro:\n" + "\n".join(problemas)


def _falas_para_numeros(secoes, roteiro) -> dict[str, str]:
    """Texto das falas: seções S e R (cabeçalho, fala, transição, 'Se perguntarem') e respostas."""
    saida = {}
    for rotulo, s in secoes.items():
        corpo = [p for p in s.paragrafos if not p.startswith("Fonte:")]
        saida[rotulo] = plano_md(s.titulo + "\n\n" + "\n\n".join(corpo))
    for p in perguntas_roteiro(roteiro):
        saida[p.rotulo] = plano_md(p.enunciado + "\n\n" + p.resposta)
    return saida


def test_a7_numeros_do_roteiro_estao_na_tese(secoes, roteiro, tese_numeros):
    falas = _falas_para_numeros(secoes, roteiro)
    assert len(falas) == 25 + 14 + 43
    problemas = []
    total = 0
    for rotulo, texto in falas.items():
        total += len(tokens_numericos(texto))
        problemas += [f"{rotulo}: {n}" for n in numeros_ausentes(texto, tese_numeros)]
    assert total > 300, f"poucos números extraídos do roteiro ({total}); extração suspeita"
    assert not problemas, "números do roteiro fora da tese e do registro:\n" + "\n".join(problemas)


# ======================================================================================
# A8. Orçamento
# ======================================================================================

def test_a8_cronometragem_soma_2280_s_e_cabe_em_2400(roteiro):
    tabela = tabela_cronometragem(roteiro)
    assert list(tabela) == PRINCIPAIS
    total = sum(s for s, _ in tabela.values())
    assert total == TOTAL_SEGUNDOS, f"soma dos quadros: {total} s"
    assert total <= MAX_SEGUNDOS
    assert linha_total(roteiro)[0] == total, "linha 'Total' da tabela diverge da soma"
    assert tabela["S09"][0] == 60 and tabela["S10"][0] == 60


def test_a8_meta_de_palavras_segue_130_por_minuto(roteiro):
    tabela = tabela_cronometragem(roteiro)
    for rotulo, (seg, meta) in tabela.items():
        assert abs(meta - seg * 130 / 60) <= 1, f"{rotulo}: meta {meta} para {seg} s"
    assert linha_total(roteiro)[1] == sum(m for _, m in tabela.values())


def palavras_fala_secao(s: SecaoRoteiro) -> int:
    """Fala mais transição (sem o rótulo 'Transição:')."""
    textos = s.fala + [t.removeprefix("Transição:") for t in s.transicoes]
    return contar_palavras(plano_md("\n\n".join(textos)))


def medidas_orcamento(roteiro: str, secoes: dict[str, SecaoRoteiro]) -> dict:
    tabela = tabela_cronometragem(roteiro)
    por_secao = {r: palavras_fala_secao(secoes[r]) for r in PRINCIPAIS}
    reserva = {r: contar_palavras(plano_md("\n\n".join(secoes[r].fala))) for r in RESERVAS}
    respostas = {p.rotulo: contar_palavras(plano_md(p.resposta)) for p in perguntas_roteiro(roteiro)}
    return {"meta": {r: m for r, (_, m) in tabela.items()}, "secao": por_secao,
            "reserva": reserva, "resposta": respostas, "total": sum(por_secao.values())}


def test_a8_orcamento_sintetico_acusa_estouro():
    s = secoes_roteiro(MD_SINTETICO)
    assert palavras_fala_secao(s["S01"]) == 5 + 2 + 4  # duas falas e a transição sem o rótulo
    assert contar_palavras(plano_md("\n\n".join(s["R01"].fala))) == 3


def test_a8_total_de_palavras_de_fala_cabe_em_40_minutos(roteiro, secoes):
    m = medidas_orcamento(roteiro, secoes)
    assert m["total"] <= MAX_PALAVRAS_FALA_TOTAL, f"total de fala e transição: {m['total']}"


def test_a8_nenhuma_secao_passa_de_125_por_cento_da_meta(roteiro, secoes):
    m = medidas_orcamento(roteiro, secoes)
    estouros = [f"{r}: {m['secao'][r]} palavras para meta {m['meta'][r]}"
                for r in PRINCIPAIS if m["secao"][r] > FATOR_META * m["meta"][r]]
    assert not estouros, "\n".join(estouros)


def test_a8_respostas_e_falas_de_reserva_cabem_no_limite(roteiro, secoes):
    m = medidas_orcamento(roteiro, secoes)
    assert len(m["resposta"]) == 43 and len(m["reserva"]) == 14
    longas = [f"{r}: {n} palavras (limite {MAX_PALAVRAS_RESPOSTA})"
              for r, n in m["resposta"].items() if n > MAX_PALAVRAS_RESPOSTA]
    longas += [f"{r}: {n} palavras (limite {MAX_PALAVRAS_RESERVA})"
               for r, n in m["reserva"].items() if n > MAX_PALAVRAS_RESERVA]
    assert not longas, "\n".join(longas)


def palavras_do_quadro(q: Quadro) -> int:
    """Corpo mais ``\\ressalva``; sem título, ``\\fonte`` nem figura incluída."""
    return contar_palavras(plano(q.corpo_sem_fonte))


def test_a8_palavras_do_quadro_sintetico():
    s01 = extrair_quadros(TEX_SINTETICO)[0]
    p = palavras_do_quadro(s01)
    assert p > 15
    assert contar_palavras(plano(s01.titulo)) == 4  # o título não entra na contagem do corpo
    assert "777" not in plano(s01.corpo_sem_fonte)


def test_a8_quadros_principais_cabem_em_125_palavras(quadros):
    estouros = [f"{l}: {palavras_do_quadro(quadros[l])} palavras" for l in PRINCIPAIS
                if palavras_do_quadro(quadros[l]) > MAX_PALAVRAS_QUADRO]
    assert not estouros, "\n".join(estouros)
