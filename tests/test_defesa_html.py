"""Contrato de testes do deck de defesa em HTML (text/slides/defesa.html).

Liga o deck HTML à tese: estrutura (28 quadros principais e 26 de reserva), arquivo
autossuficiente, recursos pedidos pela autora, figuras SVG em sincronia com o gerador, remissões,
terminologia do guia de estilo, glosas, ressalvas das âncoras aprovadas
(tests/data/registro_invariantes.yaml), números contra o texto da tese e orçamento de palavras.

As checagens de recursos (H3) são de presença no código do arquivo: teclas, botões, hash,
histórico, regras de impressão e modo de verificação. O comportamento em navegador (estouro por
quadro, navegação por teclado e botões, histórico, impressão em PDF) foi conferido à parte, em
Edge sem interface, e não é exercitado aqui.

O texto das figuras SVG não entra nos confrontos de terminologia, números e palavras: é conferido
pelo teste do gerador (tests/test_figuras_deck_svg.py) e pela sincronia de H4.

As funções genéricas (normalização e extração de números, conjunto de números da tese, contagem de
palavras, termos proibidos, título e objetivo da tese) vêm do contrato do deck Beamer
(tests/test_defesa_slides.py). As funções de extração do HTML têm teste com HTML sintético, e há casos
sintéticos que provam que H2, H6, H7 e H9 falham quando devem.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import registro_invariantes as ri  # noqa: E402
import test_defesa_slides as bz  # noqa: E402  (funções genéricas do contrato do deck Beamer)

from scripts import figuras_deck_svg as fsvg  # noqa: E402
from scripts import lembretes_deck as lem  # noqa: E402

DECK = ROOT / "text/slides/defesa.html"
TESE_DIR = bz.TESE_DIR
APENDICE_PACOTE = TESE_DIR / "apendice_pacote.tex"
APENDICE_GZCMD = TESE_DIR / "apendice_gzcmd.tex"

PRINCIPAIS = [f"S{i:02d}" for i in range(1, 29)]
RESERVAS = [f"R{i:02d}" for i in range(1, 27)]
TODOS = PRINCIPAIS + RESERVAS

OBJETIVO_ESPERADO = {"S11": 1, "S12": 2, "S13": 2, "S14": 3, "S15": 3, "S16": 3,
                     "S17": 4, "S18": 4, "S19": 4, "S20": 5}
FIGURAS_ESPERADAS = {"curva_esforco": "S14", "incerteza_ordenacoes": "S15", "curva_fusoes": "R20"}
MAX_PALAVRAS_QUADRO = 125


# ======================================================================================
# Extração: HTML (biblioteca padrão)
# ======================================================================================

BLOCOS = {"p", "li", "ul", "ol", "div", "h1", "h2", "h3", "h4", "table", "tr", "td", "th", "thead",
          "tbody", "caption", "section", "header", "footer", "blockquote", "figure", "figcaption",
          "br", "nav", "main", "button", "dl", "dt", "dd"}
_PAPEIS_POR_CLASSE = {"titulo": "titulo", "corpo": "corpo", "ressalva": "ressalva",
                      "fonte": "fonte", "sr": "sr", "lembrete": "lembrete"}
_PAPEIS_POR_TAG = {"svg": "svg", "script": "script", "style": "style", "code": "code"}
_CAMPOS = ("titulo", "corpo", "ressalva", "fonte")


@dataclass
class QuadroHtml:
    id: str
    tipo: str | None
    objetivo: int | None
    titulo: str = ""
    corpo: str = ""
    ressalva: str = ""
    fonte: str = ""
    sr: str = ""
    titulo_sc: str = ""      # "sc" = sem o conteúdo de <code>
    corpo_sc: str = ""
    ressalva_sc: str = ""
    visivel: str = ""
    svg_texto: str = ""
    hrefs: list[str] = field(default_factory=list)
    figuras: list[str] = field(default_factory=list)
    n_titulos: int = 0
    n_fontes: int = 0
    lembrete: str = ""       # texto do <aside class="lembrete"> (não entra em texto, texto_sc nem na palavra-limite)
    n_lembretes: int = 0

    @property
    def texto(self) -> str:
        """Título, corpo (com tabelas) e ressalva; sem fonte, SVG nem texto só para leitor de tela."""
        return " ".join(p for p in (self.titulo, self.corpo, self.ressalva) if p)

    @property
    def texto_sc(self) -> str:
        """O mesmo texto, sem o conteúdo de <code> (nomes de arquivo, de função, de valor)."""
        return " ".join(p for p in (self.titulo_sc, self.corpo_sc, self.ressalva_sc) if p)


@dataclass
class DocumentoHtml:
    lang: str | None
    titulo_doc: str
    quadros: list[QuadroHtml]
    n_style: int
    n_script: int
    script: str
    style: str
    hrefs_fora: list[str]

    @property
    def por_id(self) -> dict[str, QuadroHtml]:
        return {q.id: q for q in self.quadros}


def _limpar(texto: str) -> str:
    return unicodedata.normalize("NFC", " ".join(texto.split()))


class _Extrator(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lang = None
        self.titulo_doc: list[str] = []
        self.em_title = False
        self.quadros: list[QuadroHtml] = []
        self.atual: QuadroHtml | None = None
        self.bufs: dict[str, list[str]] = {}
        self.pilha: list[tuple[str, tuple[str, ...]]] = []
        self.ativos: Counter = Counter()
        self.n_style = self.n_script = 0
        self.script: list[str] = []
        self.style: list[str] = []
        self.hrefs_fora: list[str] = []

    # -- utilidades -------------------------------------------------------------------
    def _escrever(self, texto: str, so_espaco: bool = False) -> None:
        if self.atual is None:
            return
        if self.ativos["sr"] and not so_espaco:
            self.bufs["sr"].append(texto)
            return
        self.bufs["visivel"].append(texto)
        if self.ativos["lembrete"]:
            self.bufs["lembrete"].append(texto)
        for campo in _CAMPOS:
            if self.ativos[campo]:
                self.bufs[campo].append(texto)
                if not self.ativos["code"] and campo != "fonte":
                    self.bufs[campo + "_sc"].append(texto)

    def _papeis(self, tag: str, attrs: dict) -> tuple[str, ...]:
        papeis = []
        if tag in _PAPEIS_POR_TAG:
            papeis.append(_PAPEIS_POR_TAG[tag])
        for classe in (attrs.get("class") or "").split():
            if classe in _PAPEIS_POR_CLASSE:
                papeis.append(_PAPEIS_POR_CLASSE[classe])
        return tuple(papeis)

    def _abrir(self, tag: str, attrs_lista, vazio: bool) -> None:
        attrs = {k: (v or "") for k, v in attrs_lista}
        if tag == "html":
            self.lang = attrs.get("lang")
        if tag == "title" and self.atual is None:
            self.em_title = True
        if tag == "style" and not self.ativos["svg"]:
            self.n_style += 1
        if tag == "script" and not self.ativos["svg"]:
            self.n_script += 1
        if tag == "a" and "href" in attrs:
            (self.atual.hrefs if self.atual is not None else self.hrefs_fora).append(attrs["href"])
        if tag == "section" and "slide" in (attrs.get("class") or "").split() and self.atual is None:
            objetivo = attrs.get("data-objetivo")
            self.atual = QuadroHtml(id=attrs.get("id", ""), tipo=attrs.get("data-tipo"),
                                    objetivo=int(objetivo) if (objetivo or "").isdigit() else None)
            self.bufs = {c: [] for c in ("visivel", "sr", "svg_texto", "lembrete", *_CAMPOS,
                                          "titulo_sc", "corpo_sc", "ressalva_sc")}
            self.quadros.append(self.atual)
            self.pilha.append((tag, ("__quadro__",)))
            return
        papeis = self._papeis(tag, attrs)
        if self.atual is not None:
            for papel in papeis:
                if papel == "titulo":
                    self.atual.n_titulos += 1
                if papel == "fonte":
                    self.atual.n_fontes += 1
                if papel == "lembrete":
                    self.atual.n_lembretes += 1
            if tag in BLOCOS:
                self._escrever(" ", so_espaco=True)
        if vazio:
            return
        for papel in papeis:
            self.ativos[papel] += 1
        self.pilha.append((tag, papeis))

    def handle_starttag(self, tag, attrs):
        self._abrir(tag, attrs, vazio=tag in ("br", "hr", "img", "meta", "link", "input", "wbr"))

    def handle_startendtag(self, tag, attrs):
        self._abrir(tag, attrs, vazio=True)

    def handle_endtag(self, tag):
        if tag == "title":
            self.em_title = False
        indices = [i for i in range(len(self.pilha) - 1, -1, -1) if self.pilha[i][0] == tag]
        if not indices:
            return
        while len(self.pilha) > indices[0]:
            _, papeis = self.pilha.pop()
            if papeis == ("__quadro__",):
                self._fechar_quadro()
                continue
            for papel in papeis:
                self.ativos[papel] -= 1
        if self.atual is not None and tag in BLOCOS:
            self._escrever(" ", so_espaco=True)

    def _fechar_quadro(self) -> None:
        q, b = self.atual, self.bufs
        for campo in ("titulo", "corpo", "ressalva", "fonte", "sr", "visivel", "svg_texto", "lembrete",
                      "titulo_sc", "corpo_sc", "ressalva_sc"):
            setattr(q, campo, _limpar("".join(b[campo])))
        self.atual = None

    def handle_comment(self, data):
        achado = re.fullmatch(r"\s*FIGURA:(\w+)\s*", data)
        if achado and self.atual is not None:
            self.atual.figuras.append(achado.group(1))

    def handle_data(self, data):
        if self.em_title:
            self.titulo_doc.append(data)
        if self.ativos["script"] and not self.ativos["svg"]:
            self.script.append(data)
            return
        if self.ativos["style"] and not self.ativos["svg"]:
            self.style.append(data)
            return
        if self.atual is None:
            return
        if self.ativos["svg"]:
            self.bufs["svg_texto"].append(data + " ")
            return
        self._escrever(data)


def extrair_deck(html: str) -> DocumentoHtml:
    ex = _Extrator()
    ex.feed(html)
    ex.close()
    return DocumentoHtml(lang=ex.lang, titulo_doc=_limpar("".join(ex.titulo_doc)), quadros=ex.quadros,
                         n_style=ex.n_style, n_script=ex.n_script, script="".join(ex.script),
                         style="".join(ex.style), hrefs_fora=ex.hrefs_fora)


# ======================================================================================
# Funções das checagens
# ======================================================================================

NAMESPACE_SVG = 'xmlns="http://www.w3.org/2000/svg"'
PADROES_EXTERNOS = (
    (r"http://", "http://"), (r"https://", "https://"), (r"//cdn", "//cdn"),
    (r"<link\b", "<link"), (r"@import", "@import"), (r"\burl\(", "url("), (r"\bfetch\(", "fetch("),
    (r"XMLHttpRequest", "XMLHttpRequest"), (r"<img\b", "<img"), (r"<iframe\b", "<iframe"),
    (r"\bsrc\s*=", "src="),
)


def violacoes_autossuficiencia(html: str) -> list[str]:
    """Dependências externas e recursos proibidos no texto bruto do arquivo.

    O atributo ``xmlns="http://www.w3.org/2000/svg"`` dos SVG injetados pelo gerador é só o
    identificador do espaço de nomes (não é buscado na rede) e é retirado antes da varredura.
    """
    t = html.replace(NAMESPACE_SVG, "")
    achados = []
    for padrao, nome in PADROES_EXTERNOS:
        for m in re.finditer(padrao, t):
            ini = max(0, m.start() - 30)
            achados.append(f"{nome}: ...{t[ini:m.end() + 30]!r}")
    return achados


def contagem_glosas(quadros: list[QuadroHtml], expressao: str) -> dict[str, int]:
    return {q.id: len(re.findall(expressao, q.texto, re.I)) for q in quadros
            if re.search(expressao, q.texto, re.I)}


MESES = ("janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
         "setembro", "outubro", "novembro", "dezembro")
RE_DATA_EXTENSO = re.compile(r"(?<!\d)(\d{1,2})((?:\s*(?:,|e)\s*\d{1,2})*)\s+de\s+(" + "|".join(MESES)
                             + r")\s+de\s+(\d{4})")
RE_DATA_BARRA = re.compile(r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4})(?!\d)")
RE_VERSAO = re.compile(r"(?<![\w.])\d\.\d{1,2}\.\d{1,2}(?![\w.])")


def datas_do_texto(texto: str) -> list[tuple[int, int, int]]:
    """(dia, mês, ano) de cada data: 'D de mês de AAAA' (com 'D1 e D2') e 'DD/MM/AAAA'."""
    saida = []
    for m in RE_DATA_EXTENSO.finditer(texto):
        dias = [int(m.group(1))] + [int(d) for d in re.findall(r"\d{1,2}", m.group(2))]
        saida += [(d, MESES.index(m.group(3)) + 1, int(m.group(4))) for d in dias]
    saida += [(int(a), int(b), int(c)) for a, b, c in RE_DATA_BARRA.findall(texto)]
    return saida


def versoes_do_texto(texto: str) -> list[str]:
    return RE_VERSAO.findall(texto)


def _sem_datas_e_versoes(texto: str) -> str:
    t = RE_DATA_EXTENSO.sub(" ", texto)
    t = RE_DATA_BARRA.sub(" ", t)
    return RE_VERSAO.sub(" ", t)


# Exceções ao confronto de números com a tese: vazia. Qualquer número do deck que a tese e os
# ``texto_pt`` do registro não tragam falha o teste e é relatado.
EXCECOES_NUMEROS_HTML: dict[str, str] = {}


def numeros_ausentes_html(texto: str, tese: set[str],
                          excecoes: dict[str, str] | None = None) -> list[tuple[str, str]]:
    """(número normalizado, trecho) dos números de ``texto`` fora da tese, do registro e das exceções.

    Datas por extenso e versões de pacote saem antes (têm teste próprio); identificadores (O1, F4,
    R03, S09, v3.1, SHA-256), inteiros de 0 a 12 e anos de citação entre parênteses são isentos
    (regras de ``bz.tokens_numericos``).
    """
    excecoes = EXCECOES_NUMEROS_HTML if excecoes is None else excecoes
    t = _sem_datas_e_versoes(texto)
    achados = []
    limpo = bz._limpar_numeros(t)
    for n in bz.numeros_ausentes(t, tese):
        if n in excecoes:
            continue
        trecho = ""
        for m in re.finditer(r"\d+(?:[.,]\d+)*", limpo):
            if bz.normalizar_numero(m.group(0)) == n:
                trecho = limpo[max(0, m.start() - 40):m.end() + 40]
                break
        achados.append((n, trecho))
    return achados


def palavras_quadro(q: QuadroHtml) -> int:
    """Corpo mais ressalva; sem título, fonte, SVG nem texto só para leitor de tela."""
    return bz.contar_palavras(" ".join(p for p in (q.corpo, q.ressalva) if p))


EXPRESSOES_PROIBIDAS_EXTRA = (("só F4", re.compile(r"só f4", re.I)),
                              ("RRF no lugar", re.compile(r"rrf no lugar", re.I)))


def termos_proibidos_html(q: QuadroHtml) -> list[str]:
    texto = " ".join(p for p in (q.texto_sc, q.sr) if p)
    achados = [f"{q.id}: {a}" for a in bz.termos_proibidos(texto)]
    for nome, padrao in EXPRESSOES_PROIBIDAS_EXTRA:
        for m in padrao.finditer(texto):
            achados.append(f"{q.id}: '{nome}': ...{texto[max(0, m.start() - 40):m.end() + 40]}...")
    return achados


# ======================================================================================
# Fixtures
# ======================================================================================

@pytest.fixture(scope="module")
def html_bruto() -> str:
    return DECK.read_bytes().decode("utf-8")


@pytest.fixture(scope="module")
def deck(html_bruto) -> DocumentoHtml:
    return extrair_deck(html_bruto)


@pytest.fixture(scope="module")
def quadros(deck) -> dict[str, QuadroHtml]:
    return deck.por_id


@pytest.fixture(scope="module")
def registro() -> ri.Registro:
    return ri.carregar_registro()


@pytest.fixture(scope="module")
def tese_numeros() -> set[str]:
    return bz.numeros_da_tese()


# ======================================================================================
# Casos sintéticos da extração
# ======================================================================================

HTML_SINTETICO = """<!DOCTYPE html>
<html lang="pt-BR">
<head><meta charset="utf-8"><title>Título   do  deck</title>
<style>.slide { color: red; } /* 777 */</style></head>
<body>
<main id="palco">
<section class="slide" id="S01" data-tipo="principal" aria-label="rótulo 888">
<header><h1 class="titulo">Capa com <em>linkage</em></h1></header>
<div class="corpo"><p>Autor 61.485</p></div>
<footer class="pe"><p class="fonte">Fonte: tese.tex, 999.</p></footer>
</section>
<section class="slide" id="S02" data-tipo="principal" data-objetivo="3">
<header><h2 class="titulo">Segundo</h2><a class="voltar" href="#S01">← voltar</a></header>
<div class="corpo">
<p>Texto com <code>arquivo_1234.csv</code> e a reserva <a href="#R01">R01</a>.</p>
<ul><li>primeiro 1.234</li><li>segundo</li></ul>
<table><caption class="sr">legenda só para leitor 555</caption>
<tr><th>Fila</th><th>Pares</th></tr><tr><td>O1</td><td>19</td></tr></table>
<!-- FIGURA:curva_esforco -->
<figure class="figura"><svg viewBox="0 0 10 10"><style>.x{fill:red}</style><title>t</title>
<text x="1" y="1">texto da figura 4242</text></svg></figure>
<!-- /FIGURA:curva_esforco -->
<script>var oculto = 31337;</script>
</div>
<footer class="pe"><p class="ressalva"><span class="rotulo">Ressalva:</span> cautela <em>post hoc</em>.</p>
<p class="fonte">Fonte: x.tex.</p></footer>
</section>
</main>
<script>var k = 'ArrowRight';</script>
</body></html>
"""


def test_extrair_deck_sintetico_estrutura():
    d = extrair_deck(HTML_SINTETICO)
    assert d.lang == "pt-BR" and d.titulo_doc == "Título do deck"
    assert [q.id for q in d.quadros] == ["S01", "S02"]
    assert [q.tipo for q in d.quadros] == ["principal", "principal"]
    assert [q.objetivo for q in d.quadros] == [None, 3]
    assert [q.n_titulos for q in d.quadros] == [1, 1] and [q.n_fontes for q in d.quadros] == [1, 1]
    # <style> e <script> fora dos SVG: um de cada; o <style> do SVG não conta
    assert d.n_style == 1 and d.n_script == 2
    assert "ArrowRight" in d.script and "31337" in d.script and "777" in d.style


def test_extrair_deck_sintetico_textos():
    s01, s02 = extrair_deck(HTML_SINTETICO).quadros
    assert s01.titulo == "Capa com linkage" and s01.corpo == "Autor 61.485"
    assert s01.fonte == "Fonte: tese.tex, 999."
    assert "999" in s01.visivel  # a fonte faz parte do texto visível total, mas não de título, corpo e ressalva
    assert "999" not in s01.texto
    assert s02.titulo == "Segundo"
    assert s02.ressalva == "Ressalva: cautela post hoc."
    assert "primeiro 1.234 segundo" in s02.corpo and "Fila Pares O1 19" in s02.corpo
    # SVG, script, style e legenda só para leitor de tela ficam fora do texto visível
    for fora in ("4242", "31337", "777", "555", "x{fill"):
        assert fora not in s02.visivel and fora not in s02.texto, fora
    assert "4242" in s02.svg_texto and "555" in s02.sr
    # <code> fica no corpo, mas não no texto "sem código"
    assert "arquivo_1234.csv" in s02.corpo and "arquivo_1234.csv" not in s02.corpo_sc
    assert "888" not in s01.visivel  # atributo aria-label não é texto


def test_extrair_deck_sintetico_links_e_marcadores():
    s01, s02 = extrair_deck(HTML_SINTETICO).quadros
    assert s01.hrefs == [] and s02.hrefs == ["#S01", "#R01"]
    assert s02.figuras == ["curva_esforco"] and s01.figuras == []
    assert "← voltar" in s02.visivel and "voltar" not in s02.texto


def test_extrair_deck_sintetico_inline_nao_separa_palavras():
    d = extrair_deck('<section class="slide" id="S01"><div class="corpo"><p>F<sub>1</sub> e <var>k</var>'
                     '<em>x</em></p><p>duas</p><p>linhas</p></div></section>')
    assert d.quadros[0].corpo == "F1 e kx duas linhas"


# ======================================================================================
# H1. Estrutura
# ======================================================================================

def test_h1_ids_s01_a_s28_e_r01_a_r26_unicos_e_em_ordem(deck):
    ids = [q.id for q in deck.quadros]
    assert ids == TODOS, f"ids encontrados: {ids}"
    assert len(set(ids)) == len(ids) == 54


def test_h1_tipo_principal_nos_s_e_reserva_nos_r(deck):
    erros = [f"{q.id}: data-tipo={q.tipo!r}" for q in deck.quadros
             if q.tipo != ("principal" if q.id.startswith("S") else "reserva")]
    assert not erros, "\n".join(erros)


@pytest.mark.parametrize("rotulo", TODOS)
def test_h1_um_titulo_e_exatamente_uma_fonte(quadros, rotulo):
    q = quadros[rotulo]
    assert q.n_titulos == 1 and q.titulo, f"{rotulo}: {q.n_titulos} título(s); texto={q.titulo!r}"
    assert q.n_fontes == 1 and q.fonte.strip(), f"{rotulo}: {q.n_fontes} fonte(s); texto={q.fonte!r}"
    assert q.fonte.startswith("Fonte:") and len(q.fonte) > len("Fonte:") + 3, f"{rotulo}: fonte {q.fonte!r}"


def test_h1_data_objetivo_so_nos_quadros_previstos(deck):
    obtido = {q.id: q.objetivo for q in deck.quadros if q.objetivo is not None}
    assert obtido == OBJETIVO_ESPERADO, f"data-objetivo: {obtido}"


def test_h1_idioma_e_titulo_da_tese(deck, quadros):
    assert deck.lang == "pt-BR"
    titulo_tese = bz.titulo_da_tese()
    assert titulo_tese.startswith("Aprendizado de Máquina Aplicado ao Pós-Processamento")
    assert deck.titulo_doc == titulo_tese, f"<title>: {deck.titulo_doc!r}"
    assert quadros["S01"].titulo == titulo_tese, f"título da capa: {quadros['S01'].titulo!r}"


def test_h1_s04_contem_literalmente_o_objetivo_geral(quadros):
    objetivo = _limpar(bz.plano(bz.objetivo_geral_da_tese()))
    assert objetivo.startswith("Desenvolver e avaliar um pós-processamento")
    assert objetivo in quadros["S04"].corpo, f"S04 não contém o objetivo geral: {quadros['S04'].corpo!r}"


# ======================================================================================
# H2. Arquivo autossuficiente
# ======================================================================================

def test_h2_sem_dependencias_externas(html_bruto):
    achados = violacoes_autossuficiencia(html_bruto)
    assert not achados, "dependência externa no arquivo:\n" + "\n".join(achados)


def test_h2_links_so_internos(deck):
    externos = [(q.id, h) for q in deck.quadros for h in q.hrefs if not h.startswith("#")]
    externos += [("fora dos quadros", h) for h in deck.hrefs_fora if not h.startswith("#")]
    assert not externos, f"href externo: {externos}"


def test_h2_um_style_e_um_script_fora_dos_svg(deck):
    assert deck.n_style == 1, f"{deck.n_style} <style> fora dos SVG"
    assert deck.n_script == 1, f"{deck.n_script} <script> fora dos SVG"


def test_h2_utf8_sem_bom_e_sem_cr():
    bruto = DECK.read_bytes()
    assert not bruto.startswith(b"\xef\xbb\xbf"), "arquivo com BOM"
    assert b"\r" not in bruto, "arquivo com CR (fim de linha deve ser LF)"
    bruto.decode("utf-8")


@pytest.mark.parametrize("trecho,esperado", [
    ('<link rel="stylesheet" href="x.css">', "<link"),
    ("<style>@import 'a.css';</style>", "@import"),
    ('<img src="a.png">', "<img"),
    ('<img src="a.png">', "src="),
    ('<script src="https://cdn.exemplo.org/a.js"></script>', "https://"),
    ('<script src="//cdn.exemplo.org/a.js"></script>', "//cdn"),
    ("<p>http://exemplo.org</p>", "http://"),
    ("<style>a { background: url(x.png); }</style>", "url("),
    ("<script>fetch('/a')</script>", "fetch("),
    ("<script>new XMLHttpRequest()</script>", "XMLHttpRequest"),
    ('<iframe name="a"></iframe>', "<iframe"),
])
def test_h2_sintetico_acusa_cada_dependencia(trecho, esperado):
    achados = violacoes_autossuficiencia(f"<html><body>{trecho}</body></html>")
    assert any(a.startswith(esperado) for a in achados), (trecho, achados)


def test_h2_sintetico_aceita_o_espaco_de_nomes_do_svg_e_nada_mais():
    limpo = '<html><body><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"></svg><a href="#S01">a</a></body></html>'
    assert violacoes_autossuficiencia(limpo) == []
    assert violacoes_autossuficiencia(limpo.replace("</svg>", '<a href="http://x.org">x</a></svg>'))
    d = extrair_deck('<section class="slide" id="S01"><a href="https://x.org">x</a></section>')
    assert [h for h in d.quadros[0].hrefs if not h.startswith("#")] == ["https://x.org"]


# ======================================================================================
# H3. Recursos pedidos (presença no código do arquivo)
# ======================================================================================

@pytest.mark.parametrize("tecla", ["ArrowRight", "ArrowLeft", "PageDown", "PageUp", "Home", "End"])
def test_h3_tratamento_de_teclado(deck, tecla):
    assert "keydown" in deck.script, "sem ouvinte de keydown"
    assert re.search(rf"['\"]{tecla}['\"]", deck.script), f"tecla {tecla} não tratada no <script>"


def test_h3_glossario_por_tecla_botao_e_escape(deck, html_bruto):
    assert re.search(r"case\s+['\"]g['\"]", deck.script), "tecla g não tratada no <script>"
    assert "abrirGlossario" in deck.script and "fecharGlossario" in deck.script
    assert re.search(r"k === 'Escape' \|\| k === 'g'", deck.script), "Esc e g não fecham o glossário"
    assert re.search(r"<button\b[^>]*id=\"botao-glossario\"[^>]*>\s*Glossário\s*</button>", html_bruto), \
        "sem botão Glossário"
    assert re.search(r'<div id="glossario"[^>]*role="dialog"[^>]*\bhidden\b', html_bruto), "painel do glossário ausente"


def test_h3_glossario_fora_da_contagem_de_quadros_e_impresso_no_fim(deck, html_bruto):
    assert len(deck.quadros) == 54
    assert not any("glossario" in q.id for q in deck.quadros)
    m = re.search(r"@media\s+print\s*\{(?:(?!@media).)*?#glossario(?:(?!@media).)*", deck.style, re.S)
    assert m and re.search(r"#glossario[^}]*display:\s*block\s*!important", deck.style), \
        "glossário sem regra de impressão (display: block)"
    assert re.search(r"#glossario[^}]*break-before:\s*page", deck.style), "glossário sem quebra de página antes"


def test_h3_botoes_anterior_e_proximo(html_bruto):
    botoes = re.findall(r"<button\b([^>]*)>(.*?)</button>", html_bruto, re.S)
    assert botoes, "nenhum <button>"
    rotulos = [(atrs + " " + texto) for atrs, texto in botoes]
    assert any(re.search(r"Anterior", r) for r in rotulos), "sem botão Anterior"
    assert any(re.search(r"Pr[óo]ximo", r) for r in rotulos), "sem botão Próximo"
    for atrs, texto in botoes:
        assert "aria-label=" in atrs or texto.strip(), f"botão sem nome acessível: {atrs!r}"


def test_h3_contador_com_aria_live(html_bruto):
    assert re.search(r'<[a-z]+\b[^>]*id="contador"[^>]*aria-live="(?:polite|assertive)"', html_bruto) or \
        re.search(r'<[a-z]+\b[^>]*aria-live="(?:polite|assertive)"[^>]*id="contador"', html_bruto), \
        "contador sem aria-live"


def test_h3_hash_historico_e_hashchange(deck):
    for recurso in ("location.hash", "hashchange", "history.replaceState"):
        assert recurso in deck.script, f"{recurso} ausente do <script>"


def test_h3_regras_de_impressao(deck):
    css = deck.style
    assert re.search(r"@page\s*\{[^}]*size:\s*13\.333in\s+7\.5in", css), "@page sem size: 13.333in 7.5in"
    m = re.search(r"@media\s+print\s*\{", css)
    assert m, "sem @media print"
    bloco = css[m.end():]
    assert re.search(r"(?:break-after:\s*page|page-break-after:\s*always)", bloco), \
        "@media print sem quebra de página por quadro"


def test_h3_proporcao_16_por_9(deck):
    assert re.search(r"aspect-ratio:\s*16\s*/\s*9", deck.style), "sem aspect-ratio: 16 / 9"


def test_h3_modo_verificar_grava_data_estouro(deck):
    assert "verificar" in deck.script, "sem modo verificar no <script>"
    assert re.search(r"dataset\.estouro|setAttribute\(\s*['\"]data-estouro", deck.script), \
        "modo verificar sem gravação de data-estouro"


# ======================================================================================
# H4. Figuras
# ======================================================================================

def test_h4_marcadores_nos_quadros_certos(deck):
    obtido = {n: q.id for q in deck.quadros for n in q.figuras}
    assert obtido == FIGURAS_ESPERADAS, f"marcadores de figura: {obtido}"
    for q in deck.quadros:
        assert len(q.figuras) == len(set(q.figuras)), f"{q.id}: marcador duplicado"


def test_h4_figuras_em_sincronia_com_o_gerador():
    assert fsvg.verificar(DECK) == [], "figuras diferentes do gerador: rode scripts/figuras_deck_svg.py --injetar"


def test_h4_cada_figura_tem_um_unico_svg_e_sem_provisorio(deck, html_bruto):
    for nome in FIGURAS_ESPERADAS:
        m = re.search(rf"<!-- FIGURA:{nome} -->(.*?)<!-- /FIGURA:{nome} -->", html_bruto, re.S)
        assert m, f"marcadores de {nome} malformados"
        assert m.group(1).count("<svg") == 1 and 'class="figura figura-' + nome in m.group(1), nome
        assert "provisorio" not in m.group(1), f"{nome}: parágrafo provisório ainda presente"


# ======================================================================================
# H5. Remissões
# ======================================================================================

def test_h5_todo_href_aponta_para_id_existente(deck):
    ids = {q.id for q in deck.quadros}
    erros = [f"{q.id}: href={h!r}" for q in deck.quadros for h in q.hrefs
             if not (h.startswith("#") and h[1:] in ids)]
    erros += [f"fora dos quadros: href={h!r}" for h in deck.hrefs_fora if not (h.startswith("#") and h[1:] in ids)]
    assert not erros, "\n".join(erros)


def test_h5_mencoes_textuais_de_reserva_tem_link(deck):
    total = 0
    erros = []
    for q in deck.quadros:
        for r in bz.remissoes_reserva(q.texto):
            total += 1
            if f"#{r}" not in q.hrefs:
                erros.append(f"{q.id}: 'reserva {r}' no texto sem link para #{r} (links: {q.hrefs})")
    assert total >= 20, f"poucas menções de reserva encontradas ({total}); extração suspeita"
    assert not erros, "\n".join(erros)


def test_h5_reservas_novas_r15_a_r26_sao_citadas_por_quadro_principal(deck):
    citadas = {h[1:] for q in deck.quadros if q.id.startswith("S") for h in q.hrefs}
    ausentes = [f"R{i}" for i in range(15, 27) if f"R{i}" not in citadas]
    assert not ausentes, f"reservas sem remissão a partir de quadro principal: {ausentes}"


def test_h5_sintetico_mencao_sem_link_e_link_quebrado():
    d = extrair_deck('<section class="slide" id="S01"><div class="corpo"><p>ver reservas R05, R10 e R11 e '
                     '(reserva R03)</p><a href="#R05">R05</a></div></section>')
    q = d.quadros[0]
    assert bz.remissoes_reserva(q.texto) == ["R05", "R10", "R11", "R03"]
    assert [r for r in bz.remissoes_reserva(q.texto) if f"#{r}" not in q.hrefs] == ["R10", "R11", "R03"]


# ======================================================================================
# H6. Terminologia proibida
# ======================================================================================

def test_h6_deck_sem_termos_proibidos(deck):
    problemas = [a for q in deck.quadros for a in termos_proibidos_html(q)]
    assert not problemas, "\n".join(problemas)


@pytest.mark.parametrize("texto", ["a revisão clerical dos pares", "o não vínculo", "isto — aquilo",
                                   "isto – aquilo", "o PPV", "a zona cinzenta", "só F4 serve",
                                   "RRF no lugar de O3", "a nota do CdR"])
def test_h6_sintetico_detecta_termo_proibido(texto):
    d = extrair_deck(f'<section class="slide" id="S01"><div class="corpo"><p>{texto}</p></div></section>')
    assert termos_proibidos_html(d.quadros[0]), texto


def test_h6_sintetico_ignora_code_e_aceita_sinal_de_menos_e_nota_final():
    d = extrair_deck('<section class="slide" id="S01"><div class="corpo"><p>a nota final do CdR, '
                     '−21 e <code>nota_do_cdr</code>; linkage (relacionamento de registros)</p></div></section>')
    assert termos_proibidos_html(d.quadros[0]) == []


# ======================================================================================
# H7. Glosas
# ======================================================================================

GLOSAS_HTML = (("relacionamento de registros", "S02"), ("área cinza", "S03"),
               ("valor preditivo positivo", "S09"))


@pytest.mark.parametrize("expressao,rotulo", GLOSAS_HTML)
def test_h7_glosa_uma_vez_no_deck_e_no_quadro_certo(deck, expressao, rotulo):
    por_quadro = contagem_glosas(deck.quadros, expressao)
    assert sum(por_quadro.values()) == 1 and list(por_quadro) == [rotulo], \
        f"'{expressao}' no texto visível: {por_quadro} (esperado uma vez, em {rotulo})"


def test_h7_grey_area_vpp_e_asr_so_depois_da_glosa(quadros):
    for rotulo in PRINCIPAIS[:2]:
        q = quadros[rotulo]
        assert not re.search(r"grey area", q.texto + " " + q.svg_texto, re.I), f"{rotulo}: grey area antes de S03"
    for rotulo in PRINCIPAIS[:8]:
        q = quadros[rotulo]
        for sigla in ("VPP", "ASR"):
            assert not re.search(rf"\b{sigla}\b", q.texto + " " + q.svg_texto), f"{rotulo}: {sigla} antes de S09"


def test_h7_glosas_de_atributo_nao_contam(deck):
    q = deck.por_id["S02"]
    assert "relacionamento de registros" in q.titulo
    d = extrair_deck('<section class="slide" id="S01" aria-label="relacionamento de registros">'
                     '<div class="corpo"><p>nada</p></div></section>')
    assert contagem_glosas(d.quadros, "relacionamento de registros") == {}


def test_h7_lembrete_nao_conta_como_glosa_em_prosa():
    d = extrair_deck('<section class="slide" id="S02"><div class="corpo"><p>linkage (relacionamento de registros)</p></div>'
                     '<footer class="pe"><aside class="lembrete"><span class="rot">Lembrete:</span> '
                     '<span class="it"><strong>linkage</strong>: relacionamento de registros</span></aside></footer></section>')
    assert contagem_glosas(d.quadros, "relacionamento de registros") == {"S02": 1}
    assert "relacionamento de registros" in d.quadros[0].lembrete and d.quadros[0].n_lembretes == 1
    assert "relacionamento" not in d.quadros[0].texto.replace("relacionamento de registros", "", 1)


def test_h7_sintetico_glosa_duplicada_e_ausente():
    dup = extrair_deck('<section class="slide" id="S02"><h2 class="titulo">linkage (relacionamento de registros)</h2>'
                       '<div class="corpo"><p>outra vez (relacionamento de registros)</p></div></section>')
    assert contagem_glosas(dup.quadros, "relacionamento de registros") == {"S02": 2}
    nenhuma = extrair_deck('<section class="slide" id="S02"><div class="corpo"><p>nada</p></div></section>')
    assert contagem_glosas(nenhuma.quadros, "relacionamento de registros") == {}


# ======================================================================================
# H8. Âncoras por quadro
# ======================================================================================

# (âncora do registro, quadro, expressões em minúsculas no texto do quadro: título, corpo e
# ressalva). Parte da tabela do contrato do deck Beamer, com a renumeração do HTML (.tex S19 virou
# S20; S20 a S22 viraram S21 a S23; S23 virou S25). As linhas de S19, S24 e S27 são novas: a âncora
# de cada uma é a que o texto do quadro mais se aproxima (S19 e C13: LLM sem a nota final; S24 e C08r:
# a recomendação não muda; S26 e C10: a medida, a decomposição e o código aberto); as expressões foram
# conferidas no quadro antes de fixadas. S27 saiu desta tabela: o quadro do código aberto não corresponde
# a C10 (grey area e pós-processamento em componentes); suas expressões têm teste de fidelidade próprio
# (test_fidelidade_s27_e_s26_ao_apendice_do_pacote).
ANCORAS_HTML = [
    ("C10", "S03", ["61.485", "grey area", "referência operacional"]),
    ("C10", "S04", ["grey area", "preservando os vínculos aceitos pelo corte"]),
    ("C01r", "S06", ["post hoc", "coorte única", "revisor perfeito simulado", "a validar",
                     "sem ic que exclua zero", "seis variantes", "nenhuma superou"]),
    ("C12", "S06", ["post hoc", "coorte única", "revisor perfeito simulado", "a validar",
                    "sem ic que exclua zero", "seis variantes", "nenhuma superou"]),
    ("C03r", "S07", ["revisora única", "sem dupla revisão", "hipótese de trabalho"]),
    ("C15", "S11", ["equivalência não demonstrada"]),
    ("C16", "S12", ["depende da métrica", "inclui zero"]),
    ("C05r", "S12", ["otimistas", "não fundamentam adoção"]),
    ("C11", "S14", ["estimativa pontual", "sem ic que exclua zero"]),
    ("C12", "S15", ["nenhuma fusão superou o3", "seis variantes"]),
    ("C12", "S16", ["frente a o3 em nenhuma meta"]),
    ("C13", "S17", ["16,0", "não deve aceitar vínculos sozinho"]),
    ("C13", "S18", ["são hipóteses exploratórias"]),
    ("C13", "S19", ["não sugere vpp maior como aceitador", "não altera os 16,0%", "exploração posterior"]),
    ("C14", "S20", ["42,5", "não demonstra paridade", "não atributo demográfico"]),
    ("C08r", "S23", ["o3 ou f4", "f3 quando houver as macd", "curva de esforço", "não examinados", "só apoia"]),
    ("C09r", "S23", ["o3 ou f4", "f3 quando houver as macd", "curva de esforço", "não examinados", "só apoia"]),
    ("C13", "S23", ["o3 ou f4", "f3 quando houver as macd", "curva de esforço", "não examinados", "só apoia"]),
    ("C08r", "S24", ["não alteram a recomendação", "não alimentam os resultados", "o3 e f4 dependem de",
                     "hipótese para a coorte nova"]),
    ("C02r", "S25", ["25", "dados à vista"]),
    ("C10", "S26", ["36 abaixo dele", "entre ordenar a fila e automatizar a decisão"]),
]


def expressoes_ausentes(texto: str, expressoes: list[str]) -> list[str]:
    t = " ".join(texto.lower().split())
    return [e for e in expressoes if e not in t]


def test_h8_expressoes_ausentes_sintetico():
    assert expressoes_ausentes("Texto com Post Hoc  e  coorte única", ["post hoc", "coorte única"]) == []
    assert expressoes_ausentes("Texto", ["post hoc"]) == ["post hoc"]


def test_h8_identificadores_existem_no_registro(registro):
    for ancora, _, _ in ANCORAS_HTML:
        assert ancora in registro.por_id, f"âncora {ancora} ausente de registro_invariantes.yaml"
    assert "C99" not in registro.por_id


@pytest.mark.parametrize("ancora,rotulo,expressoes", ANCORAS_HTML, ids=[f"{a}-{r}" for a, r, _ in ANCORAS_HTML])
def test_h8_quadro_traz_as_expressoes_da_ancora(quadros, ancora, rotulo, expressoes):
    texto = quadros[rotulo].texto
    ausentes = expressoes_ausentes(texto, expressoes)
    assert not ausentes, f"{rotulo} ({ancora}): faltam {ausentes}\ntexto do quadro: {texto}"


def test_h8_le_titulo_corpo_e_ressalva_e_nao_o_lembrete():
    d = extrair_deck('<section class="slide" id="S19"><h2 class="titulo">Título</h2>'
                     '<div class="corpo"><p>corpo sem a expressão</p></div>'
                     '<footer class="pe"><aside class="lembrete"><span class="it"><strong>VPP</strong>: não altera os 16,0%'
                     '</span></aside><p class="ressalva">ressalva</p></footer></section>')
    q = d.quadros[0]
    assert "não altera os 16,0%" in q.lembrete
    assert expressoes_ausentes(q.texto, ["não altera os 16,0%"]) == ["não altera os 16,0%"]


def test_fidelidade_s27_e_s26_ao_apendice_do_pacote(quadros):
    """S27 saiu do H8 (não corresponde a C10): as expressões do quadro do código aberto são conferidas
    aqui, contra o apêndice do pacote (apendice_pacote.tex:65) e as explorações posteriores
    (apendice_gzcmd.tex:453), e S26 e S27 usam a redação da tese para a auditoria (conclusao.tex:65)."""
    texto = quadros["S27"].texto
    assert not expressoes_ausentes(texto, ["nem reproduz, por si só", "0.7.0", "experimental"]), texto
    pacote = " ".join(APENDICE_PACOTE.read_text(encoding="utf-8").split())
    gzcmd = " ".join(APENDICE_GZCMD.read_text(encoding="utf-8").split())
    assert "nem reproduz, por si só" in pacote
    assert "0.7.0" in gzcmd and "experimental" in gzcmd
    conclusao = " ".join((TESE_DIR / "conclusao.tex").read_text(encoding="utf-8").split())
    assert "exercício independente de auditoria das justificativas" in conclusao
    for rotulo in ("S26", "S27"):
        assert "exercício independente de auditoria das justificativas" in quadros[rotulo].texto, rotulo
        assert "sem auditoria independente" not in quadros[rotulo].texto, rotulo


# ======================================================================================
# H9. Números
# ======================================================================================

def test_h9_a_tese_fornece_numeros(tese_numeros):
    assert len(tese_numeros) > 1000
    assert {"61485", "61696", "0.910", "1172.5", "23443"} <= tese_numeros


def test_h9_excecoes_sao_explicitas_e_justificadas():
    for numero, motivo in EXCECOES_NUMEROS_HTML.items():
        assert len(motivo) > 30, f"exceção {numero} sem motivo e fonte"


def test_h9_numeros_dos_quadros_estao_na_tese(deck, tese_numeros):
    problemas = []
    total = 0
    for q in deck.quadros:
        total += len(bz.tokens_numericos(_sem_datas_e_versoes(q.texto_sc)))
        problemas += [f"{q.id}: {n} (...{trecho}...)" for n, trecho in
                      numeros_ausentes_html(q.texto_sc, tese_numeros)]
    assert total > 500, f"poucos números extraídos do deck ({total}); extração suspeita"
    assert not problemas, "números do deck fora da tese e do registro:\n" + "\n".join(problemas)


def test_h9_versoes_de_pacote_estao_nos_apendices(deck):
    tese = bz.sem_comentarios(APENDICE_PACOTE.read_text(encoding="utf-8")
                              + APENDICE_GZCMD.read_text(encoding="utf-8"))
    citadas = {v for q in deck.quadros for v in versoes_do_texto(q.texto_sc)}
    assert {"0.3.0", "0.4.0", "0.5.0", "0.6.0", "0.6.1", "0.7.0"} <= citadas, f"versões citadas: {sorted(citadas)}"
    ausentes = sorted(v for v in citadas if v not in tese)
    assert not ausentes, f"versões do deck que os apêndices não trazem: {ausentes}"


def formato_da_data_na_tese(dia: int, mes: int, ano: int, tese: str) -> str | None:
    """Formato em que a tese escreve a data, ou ``None``: 'extenso', 'dd/mm/aaaa' ou 'aaaa-mm-dd'."""
    nome = MESES[mes - 1]
    if re.search(rf"(?<!\d){dia}(?:\s*(?:,|e)\s*\d{{1,2}})*\s+de\s+{nome}\s+de\s+{ano}", tese) or \
            re.search(rf"(?<!\d)\d{{1,2}}\s*(?:,|e)\s*{dia}\s+de\s+{nome}\s+de\s+{ano}", tese):
        return "extenso"
    if re.search(rf"(?<!\d){dia:02d}/{mes:02d}/{ano}|(?<!\d){dia}/{mes}/{ano}", tese):
        return "dd/mm/aaaa"
    if re.search(rf"{ano}-{mes:02d}-{dia:02d}", tese):
        return "aaaa-mm-dd"
    return None


def test_h9_formato_da_data_sintetico():
    assert formato_da_data_na_tese(14, 11, 2026, "publicada em 14 de novembro de 2026.") == "extenso"
    assert formato_da_data_na_tese(7, 11, 2026, "versões de 6 e 7 de novembro de 2026") == "extenso"
    assert formato_da_data_na_tese(15, 11, 2026, "datado de 15/11/2026") == "dd/mm/aaaa"
    assert formato_da_data_na_tese(15, 11, 2026, "data 2026-11-15") == "aaaa-mm-dd"
    assert formato_da_data_na_tese(16, 11, 2026, "datado de 15/11/2026") is None
    assert datas_do_texto("6 e 7 de novembro de 2026 e 15/11/2026") == [(6, 11, 2026), (7, 11, 2026), (15, 11, 2026)]


def test_h9_datas_do_deck_estao_na_tese(deck):
    tese = bz.sem_comentarios("\n".join(c.read_text(encoding="utf-8") for c in sorted(TESE_DIR.rglob("*.tex"))
                                         if c.name != "tese_rev_diff.tex"))
    datas = {(q.id, d) for q in deck.quadros for d in datas_do_texto(q.texto_sc)}
    assert datas, "nenhuma data extraída do deck; extração suspeita"
    ausentes = sorted(f"{qid}: {d:02d}/{m:02d}/{a}" for qid, (d, m, a) in datas
                      if formato_da_data_na_tese(d, m, a, tese) is None)
    assert not ausentes, "datas do deck que a tese não traz (dia, mês e ano, em qualquer formato):\n" + "\n".join(ausentes)


def test_h9_sintetico_numero_ausente_da_tese_faz_a_checagem_falhar():
    tese = {"61485", "0.910"}
    d = extrair_deck('<section class="slide" id="S01"><div class="corpo"><p>Valor 61.485, 0,910 e 99.999 em O3, '
                     'R05, v3.1 e SHA-256; 7 casos; 0.6.1 de 14 de novembro de 2026.</p></div></section>')
    achados = numeros_ausentes_html(d.quadros[0].texto_sc, tese)
    assert [n for n, _ in achados] == ["99999"], achados
    assert "99.999" in achados[0][1]
    assert numeros_ausentes_html("Valor 99.999.", tese, {"99999": "exceção sintética com motivo e fonte longos"}) == []
    # número dentro de <code> não conta
    d2 = extrair_deck('<section class="slide" id="S01"><div class="corpo"><p><code>v 88.888</code> e 61.485</p></div></section>')
    assert numeros_ausentes_html(d2.quadros[0].texto_sc, tese) == []


def test_h9_sintetico_versoes_e_datas():
    assert versoes_do_texto("a 0.6.1 e 0.7.0-exp, v3.1, 1.000.000 e 3.12") == ["0.6.1", "0.7.0"]
    assert versoes_do_texto("pacote 0.4.1 e 0.4.2") == ["0.4.1", "0.4.2"]


# ======================================================================================
# H10. Orçamento
# ======================================================================================

def test_h10_palavras_do_quadro_sintetico():
    q = extrair_deck(HTML_SINTETICO).quadros[1]
    assert palavras_quadro(q) == bz.contar_palavras("Texto com arquivo_1234.csv e a reserva R01. primeiro 1.234 segundo "
                                                    "Fila Pares O1 19 Ressalva: cautela post hoc.")
    assert palavras_quadro(q) > 10


def test_h10_quadros_principais_cabem_em_125_palavras(deck):
    medidas = {q.id: palavras_quadro(q) for q in deck.quadros}
    assert all(medidas[r] > 0 for r in TODOS), f"quadro sem corpo: {[r for r in TODOS if not medidas[r]]}"
    estouros = [f"{r}: {medidas[r]} palavras (limite {MAX_PALAVRAS_QUADRO})" for r in PRINCIPAIS
                if medidas[r] > MAX_PALAVRAS_QUADRO]
    assert not estouros, "\n".join(estouros)


# ======================================================================================
# H11. Lembretes de siglas e termos e glossário (scripts/lembretes_deck.py)
# ======================================================================================

SEM_LEMBRETE = {"S01", "S28"}      # quadros sem termo de nível sigla ou termo fora do que o quadro define
LIMITE_PALAVRAS_DEFINICAO = 7
EXCECOES_PALAVRAS_DEFINICAO: dict[str, int] = {}


def _plano(fragmento: str) -> str:
    return lem.texto_puro(fragmento)


def _item(e) -> str:
    return f"{_plano(e.exibir)}: {_plano(e.defin)}"


def test_h11_glossario_e_lembretes_em_sincronia_com_o_script(html_bruto):
    assert lem.divergencias(html_bruto) == []
    assert lem.verificar_lembretes(html_bruto) == []


def test_h11_cada_quadro_tem_exatamente_um_lembrete_ou_nenhum_sem_termo(deck, html_bruto):
    selecao = lem.lembrete_do_quadro(html_bruto)
    esperado = {q: m + r for q, (m, r) in selecao.items()}
    assert set(esperado) == {q.id for q in deck.quadros}
    sem_termo = {q.id for q in deck.quadros if not esperado[q.id]}
    assert sem_termo <= SEM_LEMBRETE, f"quadros sem lembrete além de {sorted(SEM_LEMBRETE)}: {sorted(sem_termo)}"
    erros = [f"{q.id}: {q.n_lembretes} lembretes (termos: {len(esperado[q.id])})" for q in deck.quadros
             if q.n_lembretes != (1 if esperado[q.id] else 0)]
    assert not erros, "\n".join(erros)


def test_h11_todo_termo_sigla_ou_termo_nao_definido_no_quadro_esta_no_lembrete(deck, html_bruto):
    glossario = {e.chave: e for e in lem.ler_glossario(html_bruto)}
    erros = []
    for q in lem.quadros(html_bruto):
        bloco = html_bruto[q.ini:q.fim]
        texto = lem.texto_visivel(bloco)
        definidos = set(lem.definidos_no_quadro(bloco))
        lembrete = deck.por_id[q.id].lembrete
        retirados = set(lem.lembrete_do_quadro(html_bruto, list(glossario.values()))[q.id][1])
        for chave, e in glossario.items():
            if e.nivel not in lem.NO_LEMBRETE or chave in definidos or not re.search(lem.PADROES[chave], texto):
                continue
            if chave in retirados:
                assert f"{lem.DEMAIS[0]}: {lem.DEMAIS[1]}" in lembrete, f"{q.id}: retirou '{chave}' sem o item fixo"
            elif _item(e) not in lembrete:
                erros.append(f"{q.id}: '{chave}' ({e.nivel}) usado no quadro e ausente do lembrete")
    assert not erros, "\n".join(erros)


def test_h11_nenhum_termo_base_nem_termo_definido_no_quadro_aparece_no_lembrete(deck, html_bruto):
    glossario = {e.chave: e for e in lem.ler_glossario(html_bruto)}
    base = [e for e in glossario.values() if e.nivel == "base"]
    assert base, "o glossário não traz termos de nível base"
    erros = []
    for q in lem.quadros(html_bruto):
        bloco = html_bruto[q.ini:q.fim]
        lembrete = deck.por_id[q.id].lembrete
        erros += [f"{q.id}: termo base '{e.chave}' no lembrete" for e in base if _item(e) in lembrete]
        erros += [f"{q.id}: '{c}' definido no quadro e repetido no lembrete"
                  for c in lem.definidos_no_quadro(bloco) if _item(glossario[c]) in lembrete]
    assert not erros, "\n".join(erros)


def test_h11_dfn_aponta_para_entrada_existente_dentro_de_quadro_que_usa_o_termo(html_bruto):
    glossario = {e.chave for e in lem.ler_glossario(html_bruto)}
    usados = lem.termos_por_quadro(html_bruto)
    total = 0
    erros = []
    for q in lem.quadros(html_bruto):
        for c in lem.definidos_no_quadro(html_bruto[q.ini:q.fim]):
            total += 1
            if c not in glossario:
                erros.append(f"{q.id}: <dfn> para '{c}', fora do glossário")
            elif c not in usados[q.id]:
                erros.append(f"{q.id}: <dfn> de '{c}', que o quadro não usa")
    assert total >= 30, f"poucos <dfn> ({total})"
    assert not erros, "\n".join(erros)
    assert not re.findall(r"<dfn(?![^>]*data-termo=)", html_bruto), "<dfn> sem data-termo"


def test_h11_todo_quadro_traz_data_termos_com_os_tres_niveis(html_bruto):
    glossario = {e.chave: e for e in lem.ler_glossario(html_bruto)}
    detectados = lem.termos_por_quadro(html_bruto)
    niveis = set()
    for q in lem.quadros(html_bruto):
        atrib = re.search(r' data-termos="([^"]*)"', lem.ABERTURA.search(html_bruto[q.ini:q.fim]).group(0))
        assert atrib is not None, f"{q.id}: sem data-termos"
        termos = [t for t in atrib.group(1).split("|") if t]
        assert termos == detectados[q.id], f"{q.id}: data-termos difere dos termos detectados"
        niveis |= {glossario[t].nivel for t in termos}
    assert niveis == {"sigla", "termo", "base"}, niveis


def test_h11_glossario_sem_entrada_orfa_e_com_fonte_na_tese(html_bruto):
    glossario = lem.ler_glossario(html_bruto)
    usadas = {c for chaves in lem.termos_por_quadro(html_bruto, glossario).values() for c in chaves}
    assert not [e.chave for e in glossario if e.chave not in usadas], "entradas órfãs"
    assert not [e.chave for e in glossario if not e.fonte.strip()], "entradas sem fonte"
    assert {e.nivel for e in glossario} == {"sigla", "termo", "base"}
    for e in glossario:
        assert re.search(r"\.tex:\d+|definição geral|explicação geral", e.fonte), f"{e.chave}: fonte sem arquivo:linha ({e.fonte!r})"


def test_h11_definicoes_curtas_e_nao_circulares(html_bruto):
    erros = []
    for e in lem.ler_glossario(html_bruto):
        if lem.definicao_circular(e):
            erros.append(f"{e.chave}: definição circular ({_plano(e.defin)!r})")
        limite = EXCECOES_PALAVRAS_DEFINICAO.get(e.chave, LIMITE_PALAVRAS_DEFINICAO)
        if lem.palavras_da_definicao(e) > limite:
            erros.append(f"{e.chave}: {lem.palavras_da_definicao(e)} palavras (limite {limite}): {_plano(e.defin)!r}")
    assert not erros, "\n".join(erros)


def test_h11_fontes_do_glossario_apontam_para_linhas_existentes_da_tese(html_bruto):
    erros = []
    for e in lem.ler_glossario(html_bruto):
        for arquivo, linha in re.findall(r"([\w]+\.tex):(\d+)", e.fonte):
            caminho = TESE_DIR / arquivo
            n = len(caminho.read_text(encoding="utf-8").split("\n")) if caminho.exists() else 0
            if not (1 <= int(linha) <= n):
                erros.append(f"{e.chave}: {arquivo}:{linha} fora do arquivo ({n} linhas)")
    assert not erros, "\n".join(erros)


def test_h11_lembretes_e_glossario_sem_termos_proibidos(deck, html_bruto):
    textos = {q.id: q.lembrete for q in deck.quadros}
    textos["glossário"] = " ".join(f"{_plano(e.exibir)} {_plano(e.defin)}" for e in lem.ler_glossario(html_bruto))
    erros = [f"{k}: {a}" for k, t in textos.items() for a in bz.termos_proibidos(t)]
    assert not erros, "\n".join(erros)


def test_h11_numeros_do_glossario_estao_na_tese(html_bruto, tese_numeros):
    texto = " ".join(_plano(e.defin) for e in lem.ler_glossario(html_bruto))
    achados = numeros_ausentes_html(texto, tese_numeros)
    assert not achados, f"números do glossário fora da tese: {achados}"


def test_h11_glossario_em_ordem_alfabetica_e_com_ocorrencias_como_links(html_bruto):
    glossario = lem.ler_glossario(html_bruto)
    ordem = [lem.ordem_alfabetica(_plano(e.exibir)) for e in glossario]
    assert ordem == sorted(ordem)
    for e in glossario:
        m = re.search(rf"<!-- OCOR:{re.escape(e.chave)} -->(.*?)<!-- /OCOR:{re.escape(e.chave)} -->", html_bruto, re.S)
        assert m and re.findall(r'<a href="#([SR]\d\d)">', m.group(1)), f"{e.chave}: sem links de ocorrência"


def test_h11_lembrete_tem_a_letra_da_ressalva_sem_italico_e_dfn_herda_o_estilo(deck):
    css = deck.style

    def corpo_da_regra(seletor: str) -> str:
        m = re.search(rf"(?m)^{re.escape(seletor)}\s*\{{([^}}]*)\}}", css)
        assert m, f"regra {seletor} ausente da folha de estilo"
        return m.group(1)

    tamanho = lambda regra: re.search(r"font-size:\s*([\d.]+em)", regra).group(1)  # noqa: E731
    assert tamanho(corpo_da_regra(".lembrete")) == tamanho(corpo_da_regra(".ressalva"))
    assert re.search(r"font-style:\s*normal", corpo_da_regra(".lembrete"))
    assert re.search(r"font-weight:\s*(?:700|bold)", corpo_da_regra(".lembrete .rot"))
    assert re.search(r"font-style:\s*inherit", corpo_da_regra("dfn"))


def test_h11_painel_neste_quadro_antes_da_lista_alfabetica(deck, html_bruto):
    ini_neste = html_bruto.index('<section id="neste-quadro"')
    ini_lista = html_bruto.index('<dl id="lista-glossario">')
    assert ini_neste < ini_lista
    assert "montarNesteQuadro" in deck.script and "data-termos" in deck.script
    assert re.search(r"montarNesteQuadro\(\);\s*painelGlossario\.hidden = false", deck.script), \
        "o painel não monta 'Neste quadro' ao abrir"
    assert re.search(r"#neste-quadro[^}]*display:\s*none\s*!important", deck.style), "sem ocultar 'Neste quadro' na impressão"


def test_h11_orcamento_de_caracteres_respeitado_e_item_fixo_so_quando_retira(deck, html_bruto):
    glossario = lem.ler_glossario(html_bruto)
    por_chave = {e.chave: e for e in glossario}
    selecao = lem.lembrete_do_quadro(html_bruto, glossario)
    fixo = f"{lem.DEMAIS[0]}: {lem.DEMAIS[1]}"
    erros = []
    retiraram = 0
    for q in lem.quadros(html_bruto):
        mantidos, retirados = selecao[q.id]
        lembrete = deck.por_id[q.id].lembrete
        itens = [_item(por_chave[c]) for c in mantidos] + ([fixo] if retirados else [])
        orcamento = lem.orcamento_do_quadro(html_bruto[q.ini:q.fim])
        if lem.caracteres(itens) > orcamento:
            erros.append(f"{q.id}: {lem.caracteres(itens)} caracteres (orçamento {orcamento})")
        if bool(retirados) != (fixo in lembrete):
            erros.append(f"{q.id}: item fixo {'ausente' if retirados else 'indevido'}")
        if any(por_chave[c].nivel == "sigla" for c in retirados) and any(por_chave[c].nivel == "termo" for c in mantidos):
            erros.append(f"{q.id}: retirou sigla com termo ainda na lista")
        retiraram += bool(retirados)
    assert retiraram >= 1, "nenhum quadro precisou do orçamento: a regra não foi exercitada"
    assert not erros, "\n".join(erros)


def test_h11_orcamento_por_tipo_e_classe(html_bruto):
    esperado = {}
    for q in lem.quadros(html_bruto):
        bloco = html_bruto[q.ini:q.fim]
        classes = re.search(r'class="([^"]*)"', lem.ABERTURA.search(bloco).group(0)).group(1).split()
        denso = any(c in ("denso", "denso1", "denso2", "denso3") for c in classes)
        esperado[q.id] = 300 if q.id.startswith("S") else (260 if denso else 400)
        assert lem.orcamento_do_quadro(bloco) == esperado[q.id], q.id
    assert {260, 300, 400} <= set(esperado.values())


def test_h11_data_define_existe_no_glossario_e_o_termo_aparece_no_quadro(html_bruto):
    glossario = {e.chave for e in lem.ler_glossario(html_bruto)}
    usados = lem.termos_por_quadro(html_bruto)
    achados = {}
    for q in lem.quadros(html_bruto):
        for m in lem.DEFINE.finditer(html_bruto[q.ini:q.fim]):
            for c in m.group(1).split("|"):
                achados.setdefault(q.id, []).append(c)
                assert c in glossario, f"{q.id}: data-define de '{c}', fora do glossário"
                assert c in usados[q.id], f"{q.id}: data-define de '{c}', que o quadro não usa"
    assert {"S14", "R20"} <= set(achados), "figuras com legenda que define termos sem data-define"
    assert {"O1", "O2", "O3", "O4"} <= set(achados["S14"])
    for q in lem.quadros(html_bruto):
        figura = re.search(r'<figure class="figura"([^>]*)>', html_bruto[q.ini:q.fim])
        if figura and 'data-define="' in figura.group(1):
            lembrete = lem.lembrete_do_quadro(html_bruto)[q.id]
            assert not set(achados[q.id]) & set(lembrete[0] + lembrete[1]), f"{q.id}: termo da legenda repetido no lembrete"


def test_h11_definicao_nao_cita_identificador_interno(html_bruto):
    erros = [f"{e.chave}: {lem.identificadores_na_definicao(e)}" for e in lem.ler_glossario(html_bruto)
             if lem.identificadores_na_definicao(e)]
    assert not erros, "\n".join(erros)
    por = {e.chave: e for e in lem.ler_glossario(html_bruto)}
    assert "PAR" not in _plano(por["referência operacional"].defin)
    assert _plano(por["referência operacional"].defin) == "corte mais revisão manual de origem"
    assert "REFREC" not in _plano(por["bootstrap"].defin)


def test_h11_letra_do_lembrete_nas_classes_densas_nao_passa_da_do_corpo(deck):
    css = deck.style
    for classe in ("denso", "denso1", "denso2", "denso3"):
        corpo = re.search(rf"(?m)^\.slide\.{classe} \.corpo \{{[^}}]*?font-size:\s*([\d.]+)em", css)
        assert corpo, f"classe {classe} sem font-size do corpo"
        lembrete = re.search(rf"(?m)^\.slide\.{classe} \.lembrete \{{[^}}]*?font-size:\s*([\d.]+)em", css)
        tamanho_ressalva = float(re.search(r"(?m)^\.ressalva \{[^}]*?font-size:\s*([\d.]+)em", css).group(1))
        efetivo = float(lembrete.group(1)) if lembrete else tamanho_ressalva
        assert efetivo <= float(corpo.group(1)) + 1e-9, f"{classe}: lembrete {efetivo}em maior que o corpo {corpo.group(1)}em"
    assert re.search(r"(?m)^\.slide\.denso3 \.lembrete", css) and re.search(r"(?m)^\.slide\.denso2 \.lembrete", css)


# ======================================================================================
# H12. Estrutura do HTML dos quadros (marcação dentro de atributo, texto solto, rótulo do título)
# ======================================================================================

class _Estrutura(HTMLParser):
    VAZIOS = {"br", "hr", "img", "meta", "link", "input", "wbr", "path", "rect", "circle", "line", "stop"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.pilha: list[str] = []
        self.quadros: list[dict] = []
        self.atual: dict | None = None
        self.prof_quadro = 0

    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        if tag == "section" and "slide" in a.get("class", "").split() and self.atual is None:
            self.atual = {"id": a.get("id", ""), "aria": a.get("aria-label"), "filhos": [], "texto": [], "asides": 0}
            self.quadros.append(self.atual)
            self.prof_quadro = len(self.pilha)
        elif self.atual is not None:
            if len(self.pilha) == self.prof_quadro + 1:
                self.atual["filhos"].append(tag)
            if tag == "aside" and "lembrete" in a.get("class", "").split():
                self.atual["asides"] += 1
        if tag not in self.VAZIOS:
            self.pilha.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in self.pilha:
            while self.pilha and self.pilha.pop() != tag:
                pass
            if self.atual is not None and tag == "section" and len(self.pilha) == self.prof_quadro:
                self.atual = None

    def handle_data(self, data):
        if self.atual is not None and len(self.pilha) == self.prof_quadro + 1 and data.strip():
            self.atual["texto"].append(data.strip())


def violacoes_estrutura(html: str) -> list[str]:
    """Defeitos de marcação: marcação em valor de atributo, texto solto entre os blocos do quadro, blocos
    fora de header, div e footer, mais de um lembrete e aria-label diferente do título visível."""
    erros = []
    topo = html.split("<script>")[0]
    for m in re.finditer(r'\s([\w-]+)="([^"]*)"', topo):
        if "<" in m.group(2) or ">" in m.group(2):
            erros.append(f"atributo {m.group(1)} com marcação: {m.group(2)[:60]!r}")
    ex = _Estrutura()
    ex.feed(topo)
    ex.close()
    titulos = {q.id: q.titulo for q in extrair_deck(html).quadros}
    for q in ex.quadros:
        if q["filhos"] != ["header", "div", "footer"]:
            erros.append(f"{q['id']}: filhos diretos {q['filhos']} (esperado header, div, footer)")
        if q["texto"]:
            erros.append(f"{q['id']}: texto solto no quadro: {q['texto'][0][:60]!r}")
        if q["asides"] > 1:
            erros.append(f"{q['id']}: {q['asides']} lembretes")
        if q["aria"] is None or _limpar(q["aria"]) != titulos.get(q["id"]):
            erros.append(f"{q['id']}: aria-label difere do título visível")
        elif any(c in q["aria"] for c in '<>"'):
            erros.append(f"{q['id']}: aria-label com <, > ou aspas")
    return erros


def test_h12_estrutura_dos_quadros_sem_marcacao_em_atributo_nem_texto_solto(html_bruto):
    assert violacoes_estrutura(html_bruto) == []


def test_h12_numero_de_blocos_por_quadro(html_bruto):
    topo = html_bruto.split("<script>")[0]
    assert len(re.findall(r'<section class="slide', topo)) == 54
    ex = _Estrutura()
    ex.feed(topo)
    assert len(ex.quadros) == 54
    assert sum(q["filhos"].count("header") for q in ex.quadros) == 54
    assert sum(q["filhos"].count("footer") for q in ex.quadros) == 54
    assert sum(q["asides"] for q in ex.quadros) == len(re.findall(r'<aside class="lembrete"', topo)) <= 54


def test_h12_sintetico_dfn_dentro_de_aria_label_faz_o_teste_falhar():
    bom = ('<main><section class="slide" id="S01" aria-label="Título com SHAP"><header><h2 class="titulo">Título com '
           '<dfn data-termo="SHAP">SHAP</dfn></h2></header><div class="corpo"><p>x</p></div>'
           '<footer class="pe"><p class="fonte">Fonte</p></footer></section></main>')
    assert violacoes_estrutura(bom) == []
    ruim = bom.replace('aria-label="Título com SHAP"', 'aria-label="Título com <dfn data-termo="SHAP">SHAP</dfn>"')
    achados = violacoes_estrutura(ruim)
    assert any("marcação" in e for e in achados) or any("texto solto" in e for e in achados), achados
    assert achados


def test_h12_sintetico_texto_solto_aria_diferente_e_mais_de_um_lembrete():
    base = ('<main><section class="slide" id="S01" aria-label="Capa"><header><h2 class="titulo">Capa</h2></header>'
            '<div class="corpo"><p>x</p></div><footer class="pe">{rodape}</footer></section></main>')
    assert violacoes_estrutura(base.format(rodape="")) == []
    com_texto = base.replace("</header>", "</header>sobrou texto")
    assert any("texto solto" in e for e in violacoes_estrutura(com_texto.format(rodape="")))
    outro = base.replace('aria-label="Capa"', 'aria-label="Outra coisa"')
    assert any("aria-label difere" in e for e in violacoes_estrutura(outro.format(rodape="")))
    dois = '<aside class="lembrete">a</aside><aside class="lembrete">b</aside>'
    assert any("2 lembretes" in e for e in violacoes_estrutura(base.format(rodape=dois)))
    sem_div = base.replace('<div class="corpo"><p>x</p></div>', "")
    assert any("filhos diretos" in e for e in violacoes_estrutura(sem_div.format(rodape="")))


def test_h3_modo_verificar_cobre_sobreposicao_corpo_rodape_e_texto_solto(deck):
    assert re.search(r"corpo\.getBoundingClientRect\(\)\.bottom\s*>\s*pe\.getBoundingClientRect\(\)\.top", deck.script)
    assert re.search(r"cab\.getBoundingClientRect\(\)\.bottom\s*>\s*corpo\.getBoundingClientRect\(\)\.top", deck.script)
    assert "nodeType === 3" in deck.script
