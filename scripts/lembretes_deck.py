"""Lembretes de siglas e termos do deck de defesa (HTML) e ocorrências do glossário.

O glossário do deck (`<div id="glossario">`) é a fonte única das definições: uma entrada por termo
(`<dt>` com `data-chave`, `data-nivel` e `data-fonte`; `<dd class="def">` com a definição curta). Os
níveis são `sigla` (códigos, siglas e estrangeirismos), `termo` (termos em português próprios da tese
ou em sentido técnico) e `base` (vocabulário de base, repetido em quase todo quadro).

Para cada quadro, o script detecta os termos do glossário no texto visível (título, corpo, ressalva,
tabelas e texto das figuras SVG; sem a linha de fonte, sem texto só para leitor de tela e sem o próprio
lembrete) e:

* grava em `data-termos` da `<section>` TODOS os termos detectados (três níveis, separados por `|`),
  de onde o painel "Neste quadro" do glossário lê a lista, sem duplicar definições;
* reescreve o bloco entre `<!-- LEMBRETE -->` e `<!-- /LEMBRETE -->` só com os termos de nível `sigla`
  ou `termo` que o próprio quadro não define: o trecho que define o termo é marcado no HTML com
  `<dfn data-termo="chave">...</dfn>` (várias chaves separadas por `|`), e o script o pula;
* também pula os termos que a legenda de uma figura do quadro define: o contêiner da figura declara
  `data-define="chave|chave"` (o SVG não é editado);
* mostra TODOS os termos: quando a lista é longa, reduz a letra do lembrete em degraus (classe `compacto`,
  `mini` ou `micro` no `<aside>`, escolhida pelo tamanho da lista em caracteres; `LIMITES`, com piso
  por quadro em `DEGRAU_MINIMO`), nunca abaixo da letra da linha "Fonte:". Só nos quadros listados em `CORTES` (onde
  nem o menor degrau basta) há orçamento de caracteres: retira primeiro os itens de nível `termo`, do fim
  para o começo, e só depois os de nível `sigla`, e acrescenta o item fixo "demais termos: tecla g" (o painel
  "Neste quadro" segue completo);
* reescreve os links de ocorrência de cada entrada, entre `<!-- OCOR:chave -->` e
  `<!-- /OCOR:chave -->`.

Nenhum lembrete é editado à mão: `--injetar` regrava, `--verificar` acusa o que estiver
dessincronizado. A detecção usa expressões regulares por palavra inteira (`PADROES`, uma por chave do
glossário); os casos ambíguos são tratados ali de forma explícita: F1 fila contra F₁ métrica (o HTML
marca a métrica com `<var>F</var><sub>1</sub>`), C e CM (filas) contra letras soltas, `passo` (de
blocagem) contra "a cada passo" e "próximos passos", `meta` (de recuperação), `vigilância` e
`confirmação` como modos contra "vigilância do óbito", `desempate` pelo LLM contra o desempate
aleatório de O1.
"""

import argparse
import html as _html
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "text/slides/defesa.html"

ROTULO = "Lembrete:"
ARIA = "Lembrete de siglas e termos"
NIVEIS = ("sigla", "termo", "base")
NO_LEMBRETE = ("sigla", "termo")      # os níveis que entram no lembrete do quadro
# Degraus da letra do lembrete, conforme o tamanho da lista (texto visível sem o rótulo, itens separados
# por " · "): cada par é (classe do <aside>, máximo de caracteres do degrau); "" é a letra normal (.8em) e o
# último degrau não tem máximo. Os máximos foram calibrados medindo cada quadro no Chrome
# (scripts/verificar_deck_navegador.py --calibrar); o piso é a letra da linha "Fonte:" (.52em).
LIMITES = (("", 290), ("compacto", 340), ("mini", 385), ("micro", None))      # até duas linhas por degrau
DEGRAUS = ("", "compacto", "mini", "micro")
# Degrau mínimo de quadros cujo corpo não deixa espaço para o lembrete na letra que a lista pediria (medido no
# Chrome a 1280x720, 1920x1080 e 1024x768); vazio quando a lista já decide.
DEGRAU_MINIMO = {"S15": "micro", "S16": "compacto", "S19": "compacto", "S21": "micro"}
# Orçamento de caracteres só onde o menor degrau não basta (id do quadro -> máximo); vazio quando nenhum
# quadro precisa de corte. Onde há corte, o item fixo "demais termos: tecla g" avisa que o painel tem o resto.
CORTES = {}
DEMAIS = ("demais termos", "tecla g")      # item fixo, quando o orçamento retira termos
SEPARADOR = " · "

# ------------------------------------------------------------------ detecção

F_METRICA = "F\u2081"          # F₁, produzido a partir de <var>F</var><sub>1</sub>
NB = "\u2011"                   # hífen não separável

# chave do glossário -> expressão regular (sobre o texto visível normalizado)
PADROES = {
    # filas e arranjos
    "O1": r"\bO1\b", "O2": r"\bO2\b", "O3": r"\bO3\b", "O4": r"\bO4\b",
    "F1 (fila)": r"\bF1\b", "F2": r"\bF2\b", "F3": r"\bF3\b", "F4": r"\bF4\b",
    "A1 e A2": r"\bA[12]\b",
    # C e CM: só a fila (C seguido de "e CM", de dois-pontos ou de número; "filas C")
    "C (fila)": r"(?<![\w.\-])C(?=\s+e\s+CM\b|\s*:|\s+\d)|\bfilas C\b",
    "CM (fila)": r"\bCM\b",
    "ASR": r"\bASR\b",
    "cascata": r"(?i)\bcascatas?\b",
    "política v3 e v3.1": r"(?<![\w.])v3(?:\.1)?(?![\w])",
    "V1": r"\bV1\b", "V4": r"\bV4\b",
    "QW1": r"\bQW1\b",
    # instrumentos
    "CdR": r"\bCdR\b", "OpenRecLink": r"\bOpenRecLink\b", "GZ-CMD": r"GZ[-" + NB + r"]CMD",
    # saúde e dados
    "SIM": r"\bSIM\b", "Sinan": r"\bSinan\b(?![-" + NB + r"]TB)", "Sinan-TB": r"Sinan[-" + NB + r"]TB",
    "evento sentinela": r"(?i)evento sentinela",
    "subnotificação": r"(?i)\bsubnotifica\w*",
    "vigilância do óbito": r"(?i)vigilância do óbito",
    "vigilância (modo)": r"(?i)\bvigilância\b(?! do óbito)",
    "confirmação (modo)": r"(?i)\bconfirmação\b",
    "blocagem": r"(?i)\bblocagem\b",
    "passo": r"(?i)(?<!cada )(?<!próximos )(?<!próximo )\bpassos?\b",
    "PASSO": r"\bPASSO\b",
    "CAAE": r"\bCAAE\b", "LGPD": r"\bLGPD\b", "API": r"\bAPIs?\b", "PyPI": r"\bPyPI\b",
    "pair_id": r"\bpair_id\b",
    "REFREC": r"\bREFREC\b", "COMPREC": r"\bCOMPREC\b", "PAR": r"\bPAR\b",
    # método
    "linkage": r"(?i)(?<![-\w])linkage\b",
    "grey area": r"(?i)\bgrey area\b",
    "potenciais pares": r"(?i)\bpotenciais pares\b|\bpotencial par\b",
    "par verdadeiro e não par": r"(?i)\bpar(?:es)? verdadeiros?\b|\bnão par(?:es)?\b",
    "referência operacional": r"(?i)referência operacional",
    "revisão manual": r"(?i)revisão manual|revisão de origem|revisora? de origem",
    "ponto de corte": r"(?i)\bcortes?\b",
    "escore": r"(?i)\bescores?\b",
    "nota final": r"(?i)\bnota final\b|\bnotas finais\b",
    "subescores": r"(?i)\bsubescores?\b",
    "faixa intermediária da nota final": r"(?i)faixa intermediária",
    "curva de esforço": r"(?i)\bcurvas? de esforço\b",
    "revisor perfeito": r"(?i)revisor perfeito",
    "orçamento": r"(?i)\borçamentos?\b",
    "meta": r"(?i)\bmetas?\b(?= de \d|\s+\d| de recuperação|,? e \d)",
    "carga humana": r"(?i)\bcarga humana\b|\bmesma carga\b",
    "vínculo": r"(?i)\bvínculos?\b",
    "filtro": r"(?i)\bfiltro\b",
    "triagem": r"(?i)\btriagem\b",
    "desempate pelo LLM": r"(?i)\bdesempate\b(?! aleatório| de O1)",
    "desempate aleatório": r"(?i)desempate aleatório|desempate de O1",
    "dossiê": r"(?i)\bdossiês?\b",
    "labels": r"(?i)\blabels?\b",
    # estatística e aprendizado de máquina
    "IC": r"\bICs?\b", "VPP": r"\bVPP\b", "VPN": r"\bVPN\b",
    "sensibilidade": r"(?i)\bsensibilidade\b(?! automática)",
    "sensibilidade automática": r"(?i)sensibilidade automática",
    "FP": r"\bFPs?\b", "FN": r"\bFNs?\b",
    "AUC": r"\bAUC\b(?!-PR)", "AUC-PR": r"\bAUC-PR\b", "ROC": r"\bROC\b",
    "precisão média": r"(?i)precisão média|\bAP\b",
    "F1 (métrica)": F_METRICA,
    "Brier": r"\bBrier\b",
    "bootstrap": r"(?i)\bbootstrap\b",
    "post hoc": r"(?i)\bpost hoc\b",
    "fora da amostra": r"(?i)fora da amostra|\bOOF\b",
    "probabilidade calibrada": r"(?i)probabilidades? (?:calibradas?|recalibradas?)",
    "calibração isotônica": r"(?i)isotônica",
    "Platt scaling": r"(?i)\bplatt\b",
    "XGBoost": r"\bXGBoost\b", "Random Forest": r"\bRandom Forest\b", "Gradient Boosting": r"\bGradient Boosting\b",
    "SMOTE": r"\bSMOTE\b", "SMOTETomek": r"SMOTE-?Tomek", "SHAP": r"\bSHAP\b",
    "MLP": r"\bMLP\b", "SVM": r"\bSVM\b", "MACD": r"\bMACD\b",
    "expected loss": r"(?i)\bexpected loss\b",
    "bandit": r"(?i)\bbandit\b", "amostragem de Thompson": r"(?i)\bThompson\b",
    "fusão por posição recíproca": r"(?i)posição recíproca",
    "vazamento de alvo": r"(?i)vazamento de alvo",
    "pipeline": r"(?i)\bpipelines?\b",
    "partição por óbito": r"(?i)parti(?:ção|ções) (?:agrupadas? )?por (?:óbito|REFREC)|agrupad[oa]s? por (?:óbito|REFREC)",
    "partição por linha": r"(?i)parti(?:ção|ções) por linha",
    "validação cruzada": r"(?i)validação cruzada",
    "ablação": r"(?i)\bablação\b",
    "subamostragem": r"(?i)\bsubamostragem\b",
    "Fellegi e Sunter": r"\bFellegi\b",
    "Neyman e Pearson": r"\bNeyman\b",
    # LLM
    "LLM": r"\bLLMs?\b", "MATCH": r"\bMATCH\b", "NONMATCH": r"\bNONMATCH\b", "UNSURE": r"\bUNSURE\b",
}

_INLINE = ("em", "strong", "var", "sub", "sup", "code", "span", "a", "b", "i", "text", "tspan", "small", "dfn")


# ------------------------------------------------------------------ estruturas

@dataclass(frozen=True)
class Entrada:
    chave: str
    id: str
    nivel: str
    exibir: str      # HTML da forma exibida
    defin: str       # HTML da definição
    fonte: str       # arquivo:linha da tese


@dataclass(frozen=True)
class Quadro:
    id: str
    ini: int
    fim: int


SECAO = re.compile(r'<section class="slide[^"]*" id="(?P<id>[SR]\d\d)".*?</section>', re.S)
GLOSSARIO = re.compile(r'<dl id="lista-glossario">.*?</dl>', re.S)
ENTRADA = re.compile(r'<dt id="g-(?P<id>[^"]+)" data-chave="(?P<chave>[^"]+)" data-nivel="(?P<nivel>[^"]*)" '
                     r'data-fonte="(?P<fonte>[^"]*)">(?P<exibir>.*?)</dt>\s*<dd class="def">(?P<defin>.*?)</dd>', re.S)
ITEM = re.compile(r'<span class="it"><strong>(?P<exibir>.*?)</strong>: (?P<defin>.*?)</span>', re.S)
DFN = re.compile(r'<dfn data-termo="([^"]+)">')
DEFINE = re.compile(r' data-define="([^"]+)"')
ABERTURA = re.compile(r"<section [^>]*>")
BLOCO_LEMBRETE = re.compile(r"[ \t]*<!-- LEMBRETE -->.*?<!-- /LEMBRETE -->\n?", re.S)
ABERTURA_PE = re.compile(r'<footer class="pe">\n')


def ler(path=DECK):
    return Path(path).read_bytes().decode("utf-8")


def quadros(html):
    return [Quadro(m.group("id"), m.start(), m.end()) for m in SECAO.finditer(html)]


def ler_glossario(html):
    """Entradas do glossário, na ordem do arquivo."""
    m = GLOSSARIO.search(html)
    if m is None:
        raise ValueError("glossário ausente: falta <dl id=\"lista-glossario\">")
    return [Entrada(e.group("chave"), e.group("id"), e.group("nivel"), e.group("exibir"), e.group("defin"),
                    e.group("fonte")) for e in ENTRADA.finditer(m.group(0))]


def texto_puro(fragmento):
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]*>", "", fragmento))).strip()


# ------------------------------------------------------------------ texto visível de um quadro

def texto_visivel(bloco):
    """Texto visível do quadro, para detectar termos. Fora: lembrete, fonte, legenda e texto só para
    leitor de tela, link "voltar", marcador de objetivo, <title>/<desc>/<style> dos SVG e atributos."""
    b = bloco
    b = re.sub(r"<!-- LEMBRETE -->.*?<!-- /LEMBRETE -->", " ", b, flags=re.S)
    b = re.sub(r'<aside class="lembrete[^"]*".*?</aside>', " ", b, flags=re.S)
    b = re.sub(r'<p class="fonte">.*?</p>', " ", b, flags=re.S)
    b = re.sub(r'<caption class="sr">.*?</caption>', " ", b, flags=re.S)
    b = re.sub(r'<(\w+) class="sr">.*?</\1>', " ", b, flags=re.S)
    b = re.sub(r'<a class="voltar".*?</a>', " ", b, flags=re.S)
    b = re.sub(r'<div class="marcador".*?</div>', " ", b, flags=re.S)
    b = re.sub(r"<title\b.*?</title>|<desc\b.*?</desc>|<style\b.*?</style>", " ", b, flags=re.S)
    b = re.sub(r"<!--.*?-->", " ", b, flags=re.S)
    b = b.replace("<var>F</var><sub>1</sub>", F_METRICA)
    b = re.sub(r"</?(?:%s)\b[^>]*>" % "|".join(_INLINE), "", b)
    b = re.sub(r"<[^>]*>", " ", b)
    return re.sub(r"\s+", " ", _html.unescape(b)).strip()


def termos_do_texto(texto, chaves):
    """Chaves cujo padrão aparece no texto, na ordem da primeira aparição."""
    achados = []
    for chave in chaves:
        padrao = PADROES.get(chave)
        if padrao is None:
            raise KeyError(f"glossário sem padrão de detecção: {chave!r}")
        m = re.search(padrao, texto)
        if m:
            achados.append((m.start(), chave))
    return [c for _, c in sorted(achados)]


def termos_por_quadro(html, glossario=None):
    """Todos os termos do glossário que o quadro usa (três níveis), na ordem de aparição."""
    glossario = glossario if glossario is not None else ler_glossario(html)
    chaves = [e.chave for e in glossario]
    return {q.id: termos_do_texto(texto_visivel(html[q.ini:q.fim]), chaves) for q in quadros(html)}


def definidos_no_quadro(bloco):
    """Chaves que o próprio quadro define: <dfn data-termo> no texto e data-define no contêiner de uma
    figura (a legenda do SVG define o termo) ou de uma tabela cujo cabeçalho o define."""
    chaves = []
    for m in DFN.finditer(bloco):
        chaves += m.group(1).split("|")
    for m in DEFINE.finditer(bloco):
        chaves += m.group(1).split("|")
    return chaves


def termos_do_lembrete(html, glossario=None):
    """Termos de nível sigla ou termo que o quadro usa e não define, na ordem de aparição."""
    glossario = glossario if glossario is not None else ler_glossario(html)
    nivel = {e.chave: e.nivel for e in glossario}
    todos = termos_por_quadro(html, glossario)
    saida = {}
    for q in quadros(html):
        definidos = set(definidos_no_quadro(html[q.ini:q.fim]))
        saida[q.id] = [c for c in todos[q.id] if nivel[c] in NO_LEMBRETE and c not in definidos]
    return saida


# ------------------------------------------------------------------ orçamento

def id_do_quadro(bloco):
    return re.search(r' id="([SR]\d\d)"', ABERTURA.search(bloco).group(0)).group(1)


def orcamento_do_quadro(bloco):
    """Caracteres permitidos ao lembrete: None (sem corte) em todo quadro fora de CORTES."""
    return CORTES.get(id_do_quadro(bloco))


def degrau_do_lembrete(n, minimo=""):
    """Classe de letra do lembrete para uma lista de n caracteres ("" é a letra normal), nunca mais
    leve que o degrau mínimo do quadro."""
    for classe, maximo in LIMITES:
        if maximo is None or n <= maximo:
            return max(classe, minimo, key=DEGRAUS.index)
    return max(LIMITES[-1][0], minimo, key=DEGRAUS.index)


def texto_do_item(e):
    return f"{texto_puro(e.exibir)}: {texto_puro(e.defin)}"


def caracteres(textos):
    return len(SEPARADOR.join(textos))


def selecionar_itens(chaves, glossario, orcamento):
    """(mantidos, retirados): se a lista passa do orçamento, retira primeiro os itens de nível termo, do
    fim para o começo, e só depois os de nível sigla, do fim para o começo; o item fixo "demais termos:
    tecla g" entra na conta quando algo é retirado. A ordem de aparição dos que ficam é mantida."""
    por = {e.chave: e for e in glossario}
    mantidos, retirados = list(chaves), []
    if orcamento is None:
        return mantidos, retirados

    def custo():
        textos = [texto_do_item(por[c]) for c in mantidos]
        if retirados:
            textos.append(f"{DEMAIS[0]}: {DEMAIS[1]}")
        return caracteres(textos)

    for nivel in ("termo", "sigla"):
        while custo() > orcamento:
            candidatos = [c for c in mantidos if por[c].nivel == nivel]
            if not candidatos:
                break
            mantidos.remove(candidatos[-1])
            retirados.append(candidatos[-1])
    return mantidos, retirados


def lembrete_do_quadro(html, glossario=None):
    """id do quadro -> (itens mantidos, itens retirados pelo orçamento)."""
    glossario = glossario if glossario is not None else ler_glossario(html)
    todos = termos_do_lembrete(html, glossario)
    return {q.id: selecionar_itens(todos[q.id], glossario, orcamento_do_quadro(html[q.ini:q.fim]))
            for q in quadros(html)}


def textos_do_lembrete(mantidos, retirados, glossario):
    """Textos dos itens que o lembrete mostra (com o item fixo quando algo foi retirado)."""
    por = {e.chave: e for e in glossario}
    textos = [texto_do_item(por[c]) for c in mantidos]
    if retirados:
        textos.append(f"{DEMAIS[0]}: {DEMAIS[1]}")
    return textos


def degraus_do_quadro(html, glossario=None):
    """id do quadro -> classe de letra do lembrete ("" = normal), pelo tamanho da lista que ele mostra."""
    glossario = glossario if glossario is not None else ler_glossario(html)
    selecao = lembrete_do_quadro(html, glossario)
    return {q.id: degrau_do_lembrete(caracteres(textos_do_lembrete(*selecao[q.id], glossario)),
                                     DEGRAU_MINIMO.get(q.id, ""))
            for q in quadros(html)}


# ------------------------------------------------------------------ geração

def render_lembrete(chaves, glossario, eol="\n", retirou=False, degrau=""):
    por_chave = {e.chave: e for e in glossario}
    linhas = ["<!-- LEMBRETE -->"]
    itens = [f'<span class="it"><strong>{por_chave[c].exibir}</strong>: {por_chave[c].defin}</span>' for c in chaves]
    if retirou:
        itens.append(f'<span class="it"><strong>{DEMAIS[0]}</strong>: {DEMAIS[1]}</span>')
    if itens:
        classe = "lembrete" + (f" {degrau}" if degrau else "")
        linhas.append(f'<aside class="{classe}" aria-label="{ARIA}"><strong class="rot">{ROTULO}</strong> '
                      + " ".join(itens) + "</aside>")
    linhas.append("<!-- /LEMBRETE -->")
    return eol.join(linhas) + eol


def _ocorrencias(glossario, por_quadro, eol):
    saida = {}
    for e in glossario:
        ids = [q for q, ch in por_quadro.items() if e.chave in ch]
        links = ", ".join(f'<a href="#{q}">{q}</a>' for q in ids) if ids else "nenhum quadro"
        saida[e.chave] = f"<!-- OCOR:{e.chave} -->{links}<!-- /OCOR:{e.chave} -->"
    return saida


def _com_termos(bloco, termos):
    """Abertura da <section> com data-termos regravado (três níveis, separados por |)."""
    m = ABERTURA.search(bloco)
    tag = re.sub(r' data-termos="[^"]*"', "", m.group(0))
    tag = tag[:-1] + f' data-termos="{"|".join(termos)}">'
    return bloco[:m.start()] + tag + bloco[m.end():]


def aplicar(html):
    """HTML com data-termos, lembretes e ocorrências do glossário regravados."""
    eol = "\r\n" if "\r\n" in html else "\n"
    glossario = ler_glossario(html)
    por_quadro = termos_por_quadro(html, glossario)
    selecao = lembrete_do_quadro(html, glossario)
    degraus = degraus_do_quadro(html, glossario)
    for q in reversed(quadros(html)):
        bloco = html[q.ini:q.fim]
        novo = render_lembrete(selecao[q.id][0], glossario, eol, bool(selecao[q.id][1]), degraus[q.id])
        if BLOCO_LEMBRETE.search(bloco):
            bloco = BLOCO_LEMBRETE.sub(lambda _m: novo, bloco, count=1)
        else:
            m = ABERTURA_PE.search(bloco)
            if m is None:
                raise ValueError(f"{q.id}: sem <footer class=\"pe\"> para o lembrete")
            bloco = bloco[:m.end()] + novo + bloco[m.end():]
        bloco = _com_termos(bloco, por_quadro[q.id])
        html = html[:q.ini] + bloco + html[q.fim:]
    ocor = _ocorrencias(glossario, por_quadro, eol)
    for chave, novo in ocor.items():
        padrao = re.compile(rf"<!-- OCOR:{re.escape(chave)} -->.*?<!-- /OCOR:{re.escape(chave)} -->", re.S)
        if not padrao.search(html):
            raise ValueError(f"glossário: sem marcadores de ocorrência para {chave!r}")
        html = padrao.sub(lambda _m, n=novo: n, html, count=1)
    return html


def ordem_alfabetica(texto):
    s = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


def definicao_circular(e):
    """True se a definição começa pelo próprio termo ou repete o termo como primeira palavra."""
    exibido = texto_puro(e.exibir)
    base = re.sub(r"\s*\(.*\)$", "", exibido).strip().casefold()
    defin = texto_puro(e.defin).casefold()
    if not defin:
        return True
    primeira_def = re.split(r"[\s,;:]", defin)[0]
    primeira_termo = re.split(r"[\s,;:]", base)[0]
    return defin.startswith(base) or primeira_def == primeira_termo


IDENTIFICADORES_INTERNOS = ("REFREC", "COMPREC", "PAR", "PASSO", "pair_id")
DEFINEM_IDENTIFICADORES = ("REFREC", "COMPREC", "PAR", "PASSO", "pair_id")


def identificadores_na_definicao(e):
    """Identificadores internos que a definição cita, fora das entradas que os definem."""
    if e.chave in DEFINEM_IDENTIFICADORES:
        return []
    return [i for i in IDENTIFICADORES_INTERNOS if re.search(rf"(?<![\w]){re.escape(i)}(?![\w])", texto_puro(e.defin))]


def palavras_da_definicao(e):
    return len(texto_puro(e.defin).split())


def divergencias(html):
    """Defeitos do glossário e dos lembretes; lista vazia quando tudo está em sincronia."""
    erros = []
    try:
        glossario = ler_glossario(html)
    except ValueError as exc:
        return [str(exc)]
    chaves = [e.chave for e in glossario]
    if len(set(chaves)) != len(chaves):
        erros.append("glossário: chaves duplicadas")
    for c in chaves:
        if c not in PADROES:
            erros.append(f"glossário: entrada sem padrão de detecção: {c}")
    for c in PADROES:
        if c not in chaves:
            erros.append(f"padrão de detecção sem entrada no glossário: {c}")
    ordem = [ordem_alfabetica(texto_puro(e.exibir)) for e in glossario]
    if ordem != sorted(ordem):
        erros.append("glossário fora de ordem alfabética")
    for e in glossario:
        if not e.fonte.strip():
            erros.append(f"glossário: entrada sem fonte na tese: {e.chave}")
        if e.nivel not in NIVEIS:
            erros.append(f"glossário: nível inválido em {e.chave}: {e.nivel!r}")
        if definicao_circular(e):
            erros.append(f"glossário: definição circular em {e.chave}")
        if identificadores_na_definicao(e):
            erros.append(f"glossário: definição de {e.chave} cita identificador interno: {identificadores_na_definicao(e)}")
    if erros:
        return erros
    por_quadro = termos_por_quadro(html, glossario)
    usadas = {c for chs in por_quadro.values() for c in chs}
    for e in glossario:
        if e.chave not in usadas:
            erros.append(f"glossário: entrada órfã (nenhum quadro usa o termo): {e.chave}")
    for q in quadros(html):
        for c in definidos_no_quadro(html[q.ini:q.fim]):
            if c not in chaves:
                erros.append(f"{q.id}: <dfn> para termo fora do glossário: {c}")
            elif c not in por_quadro[q.id]:
                erros.append(f"{q.id}: <dfn> de termo que o quadro não usa: {c}")
    if aplicar(html) != html:
        erros.append("data-termos, lembretes ou ocorrências do glossário dessincronizados (rode --injetar)")
    return erros


def verificar_lembretes(html):
    """Defeitos por quadro: termo usado sem lembrete, item divergente do glossário, item sobrando,
    termo de nível base no lembrete, termo definido no quadro repetido no lembrete, ordem diferente da
    aparição, número de lembretes diferente de um, orçamento de caracteres estourado (só onde há corte),
    degrau de letra incoerente com o tamanho da lista, item "demais termos" ausente ou indevido, e
    data-termos diferente do detectado."""
    erros = []
    glossario = ler_glossario(html)
    por_chave = {e.chave: e for e in glossario}
    selecao = lembrete_do_quadro(html, glossario)
    degraus = degraus_do_quadro(html, glossario)
    todos = termos_por_quadro(html, glossario)
    for q in quadros(html):
        bloco = html[q.ini:q.fim]
        mantidos, retirados = selecao[q.id]
        n = len(re.findall(r'<aside class="lembrete[ "]', bloco))
        itens = [(m.group("exibir"), m.group("defin")) for m in ITEM.finditer(bloco)]
        if (mantidos or retirados) and n != 1:
            erros.append(f"{q.id}: {n} lembretes (esperado 1)")
        if not (mantidos or retirados) and n:
            erros.append(f"{q.id}: lembrete sem termo da lista")
        definidos = set(definidos_no_quadro(bloco))
        exibidos, textos = [], []
        tem_demais = False
        for exibir, defin in itens:
            if exibir == DEMAIS[0]:
                tem_demais = True
                if defin != DEMAIS[1]:
                    erros.append(f"{q.id}: item {DEMAIS[0]!r} com texto diferente do fixo")
                textos.append(f"{DEMAIS[0]}: {DEMAIS[1]}")
                continue
            achada = [c for c, e in por_chave.items() if e.exibir == exibir]
            if not achada:
                erros.append(f"{q.id}: item {texto_puro(exibir)!r} fora do glossário")
                continue
            chave = achada[0]
            exibidos.append(chave)
            textos.append(texto_do_item(por_chave[chave]))
            if por_chave[chave].defin != defin:
                erros.append(f"{q.id}: item {chave!r} diverge da definição do glossário")
            if por_chave[chave].nivel == "base":
                erros.append(f"{q.id}: termo de nível base no lembrete: {chave}")
            if chave in definidos:
                erros.append(f"{q.id}: item {chave!r} repete termo definido no próprio quadro")
        for c in mantidos:
            if c not in exibidos:
                erros.append(f"{q.id}: termo {c!r} usado sem item no lembrete")
        for c in exibidos:
            if c not in mantidos and c not in definidos and por_chave[c].nivel != "base":
                erros.append(f"{q.id}: item {c!r} sem uso no quadro ou retirado pelo orçamento")
        if [c for c in exibidos if c in mantidos] != [c for c in mantidos if c in exibidos]:
            erros.append(f"{q.id}: itens fora da ordem de aparição")
        if tem_demais != bool(retirados):
            erros.append(f"{q.id}: item {DEMAIS[0]!r} " + ("ausente" if retirados else "indevido"))
        orcamento = orcamento_do_quadro(bloco)
        if orcamento is not None and caracteres(textos) > orcamento:
            erros.append(f"{q.id}: lembrete com {caracteres(textos)} caracteres (orçamento {orcamento})")
        classe = re.search(r'<aside class="lembrete(?: ([a-z]+))?"', bloco)
        if classe is not None and (classe.group(1) or "") != degraus[q.id]:
            erros.append(f"{q.id}: degrau de letra {classe.group(1) or 'normal'!r} incoerente com a lista de "
                         f"{caracteres(textos)} caracteres (esperado {degraus[q.id] or 'normal'!r})")
        atrib = re.search(r' data-termos="([^"]*)"', ABERTURA.search(bloco).group(0))
        termos = atrib.group(1).split("|") if atrib and atrib.group(1) else []
        if termos != todos[q.id]:
            erros.append(f"{q.id}: data-termos difere dos termos detectados")
    return erros


# ------------------------------------------------------------------ vocabulário mínimo por bloco

ROTEIRO = ROOT / "text/slides/roteiro_defesa_html.md"
LINHA_BLOCO = re.compile(r"^\|\s*(?P<nome>[^|]+?)\s*\|\s*(?P<de>S\d\d) a (?P<ate>S\d\d)\s*\|", re.M)
EM_ITALICO = re.compile(r"<em\b[^>]*>.*</em>", re.S)
PRINCIPAL = re.compile(r"S\d\d")


def blocos_do_roteiro(md):
    """Nome do bloco -> quadros principais, das linhas "| Bloco | S01 a S06 | seg | min |" da tabela de
    tempo do roteiro (a linha "Total" fica de fora)."""
    saida = {}
    for m in LINHA_BLOCO.finditer(md):
        if m.group("nome") == "Total":
            continue
        de, ate = int(m.group("de")[1:]), int(m.group("ate")[1:])
        saida[m.group("nome")] = [f"S{i:02d}" for i in range(de, ate + 1)]
    return saida


def _termos_do_quadro(bloco):
    """Termos do quadro principal: data-termos na ordem do atributo, depois os de <dfn data-termo> que
    não estejam nele."""
    atrib = re.search(r' data-termos="([^"]*)"', ABERTURA.search(bloco).group(0))
    termos = atrib.group(1).split("|") if atrib and atrib.group(1) else []
    for c in definidos_no_quadro(bloco):
        if c not in termos:
            termos.append(c)
    return termos


def vocabulario_markdown(html, blocos):
    """Markdown da seção "Vocabulário mínimo por bloco" do roteiro: uma linha por bloco.

    Só quadros principais (S01..S28). "Apresentados" são os termos cujo primeiro quadro principal, na
    ordem do deck, pertence ao bloco (o termo consta de data-termos ou de <dfn data-termo>), com esse
    quadro entre colchetes. "Trazidos" são os termos presentes em quadros do bloco e apresentados em bloco
    anterior. Ambas as listas seguem a ordem de apresentação (quadro, depois posição em data-termos). Os
    termos que o glossário grafa em itálico (exibição inteira em <em>) saem entre asteriscos."""
    glossario = {e.chave: e for e in ler_glossario(html)}
    principais = [q for q in quadros(html) if PRINCIPAL.fullmatch(q.id)]
    ordem = [q.id for q in principais]
    termos = {q.id: _termos_do_quadro(html[q.ini:q.fim]) for q in principais}
    primeiro, sequencia = {}, []
    for qid in ordem:
        for c in termos[qid]:
            if c not in primeiro:
                primeiro[c] = qid
                sequencia.append((qid, c))

    def rotulo(c):
        e = glossario.get(c)
        return f"*{c}*" if e is not None and EM_ITALICO.fullmatch(e.exibir.strip()) else c

    def lista(itens):
        return "; ".join(f"{rotulo(c)} [{q}]" for q, c in itens) if itens else "nenhum"

    linhas = []
    for nome, ids in blocos.items():
        inicio = min(ordem.index(i) for i in ids)
        usados = {c for i in ids for c in termos[i]}
        novos = [(q, c) for q, c in sequencia if q in ids]
        trazidos = [(q, c) for q, c in sequencia if c in usados and ordem.index(q) < inicio]
        faixa = ids[0] if len(ids) == 1 else f"{ids[0]} a {ids[-1]}"
        linhas.append(f"- **{nome} ({faixa}).** Apresentados no bloco: {lista(novos)}. "
                      f"Trazidos de blocos anteriores: {lista(trazidos)}.")
    return "\n".join(linhas)


# ------------------------------------------------------------------ CLI

def injetar(path=DECK):
    html = ler(path)
    novo = aplicar(html)
    if novo != html:
        Path(path).write_bytes(novo.encode("utf-8"))
        return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grupo = ap.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--injetar", type=Path, metavar="HTML", help="Regrava data-termos, lembretes e ocorrências")
    grupo.add_argument("--verificar", type=Path, metavar="HTML", help="Código 1 se algo estiver dessincronizado")
    grupo.add_argument("--vocabulario", type=Path, metavar="HTML",
                       help="Imprime a seção 'Vocabulário mínimo por bloco' do roteiro, gerada do deck")
    ap.add_argument("--roteiro", type=Path, default=ROTEIRO, metavar="MD",
                    help="Roteiro de onde sai o mapa de blocos (tabela de tempo); só com --vocabulario")
    args = ap.parse_args(argv)
    if args.vocabulario:
        blocos = blocos_do_roteiro(Path(args.roteiro).read_bytes().decode("utf-8"))
        print(vocabulario_markdown(ler(args.vocabulario), blocos))
        return 0
    if args.injetar:
        print("regravado" if injetar(args.injetar) else "já em sincronia")
        return 0
    html = ler(args.verificar)
    erros = divergencias(html)
    if not erros or all("dessincronizados" in e for e in erros):
        erros = erros + verificar_lembretes(html)
    for e in erros:
        print(e)
    if not erros:
        print("lembretes e glossário em sincronia")
    return 1 if erros else 0


if __name__ == "__main__":
    sys.exit(main())
