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
* mostra TODOS os termos sempre que couberem: a letra do lembrete (classe `compacto` ou `mini` no `<aside>`,
  nunca abaixo de `mini`) e o eventual corte de cada quadro vêm da tabela `AJUSTES`, calibrada por medição no
  Chrome (`scripts/verificar_deck_navegador.py --calibrar`). Os itens de nível `sigla` vêm antes dos de nível
  `termo`, cada nível na ordem de aparição. Só onde a lista inteira não cabe nem em `mini` há corte: ficam os N
  primeiros itens e o item fixo "demais termos: tecla g" avisa que o painel "Neste quadro" tem o resto (o painel
  segue completo em todos os quadros);
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
# Letra do lembrete por quadro, em degraus: "" é a letra normal (.8em), "compacto" (.68em) e "mini" (.6em, o piso de
# legibilidade; não há degrau menor). A escolha é por MEDIÇÃO no Chrome, não por contagem de caracteres: o modo
# `--calibrar` de scripts/verificar_deck_navegador.py mede, a 1280x720, 1920x1080 e 1024x768, o maior degrau em
# que o quadro não estoura nem sobrepõe, deixando o lembrete ocupar as linhas que precisar, e imprime esta tabela
# pronta para colar. Sem a tabela o gerador não sabe o espaço de cada quadro; o modo padrão do script do navegador
# falha quando a tabela deixa de ser a ótima (por exemplo, depois de mudar o texto de um quadro).
DEGRAUS = ("", "compacto", "mini")
# quadro -> (degrau, itens mantidos). Itens mantidos None = lista inteira; um número = corte: ficam os N primeiros
# na ordem de prioridade (siglas antes de termos, cada nível na ordem de aparição), mais o item fixo
# "demais termos: tecla g". Regra da calibração: (a) mostrar o MAIOR NÚMERO de itens que couber em algum degrau (normal, compacto ou mini),
# no pior caso entre os três viewports, e, em empate no número de itens, o degrau de maior letra;
# (b) nas reservas, um degrau de densidade do corpo a menos (DESCIDAS) quando isso aumenta o número de itens;
# (c) corte, quando a lista inteira não cabe nem em mini.
# <AJUSTES>
AJUSTES = {
    "R01": ('compacto', None),
    "R02": ('', None),
    "R03": ('', None),
    "R04": ('', None),
    "R05": ('', None),
    "R06": ('compacto', None),
    "R07": ('', None),
    "R08": ('', None),
    "R09": ('compacto', None),
    "R10": ('', None),
    "R11": ('', None),
    "R12": ('', None),
    "R13": ('', None),
    "R14": ('', None),
    "R15": ('', None),
    "R16": ('compacto', None),
    "R17": ('', None),
    "R18": ('compacto', None),
    "R19": ('mini', None),
    "R20": ('', None),
    "R21": ('', None),
    "R22": ('', None),
    "R23": ('', None),
    "R24": ('', None),
    "R25": ('compacto', None),
    "R26": ('', None),
    "S02": ('', None),
    "S03": ('', None),
    "S04": ('', None),
    "S05": ('', None),
    "S06": ('', None),
    "S07": ('', None),
    "S08": ('', None),
    "S09": ('', None),
    "S10": ('', None),
    "S11": ('', None),
    "S12": ('', None),
    "S13": ('', None),
    "S14": ('', None),
    "S15": ('mini', 4),
    "S16": ('compacto', None),
    "S17": ('', None),
    "S18": ('', None),
    "S19": ('compacto', None),
    "S20": ('', None),
    "S21": ('mini', 8),
    "S22": ('', None),
    "S23": ('mini', 12),
    "S24": ('compacto', None),
    "S25": ('', None),
    "S26": ('', None),
    "S27": ('', None),
}
# </AJUSTES>
# Reservas cujo corpo desceu um degrau de densidade (classe da <section>, escrita à mão no HTML) para o lembrete
# caber inteiro: quadro -> classe de partida ("" = padrão). No máximo um degrau por quadro.
DESCIDAS = {"R25": "denso"}
DENSIDADES = ("", "denso", "denso1", "denso2", "denso3")
DEMAIS = ("demais termos", "tecla g")      # item fixo, quando há corte
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
    """Termos de nível sigla ou termo que o quadro usa e não define, em ordem de prioridade (siglas antes de termos)."""
    glossario = glossario if glossario is not None else ler_glossario(html)
    nivel = {e.chave: e.nivel for e in glossario}
    todos = termos_por_quadro(html, glossario)
    saida = {}
    for q in quadros(html):
        definidos = set(definidos_no_quadro(html[q.ini:q.fim]))
        saida[q.id] = por_prioridade([c for c in todos[q.id] if nivel[c] in NO_LEMBRETE and c not in definidos], nivel)
    return saida


# ------------------------------------------------------------------ orçamento

def id_do_quadro(bloco):
    return re.search(r' id="([SR]\d\d)"', ABERTURA.search(bloco).group(0)).group(1)


def ajuste_do_quadro(qid):
    """(degrau, itens mantidos) do quadro; sem entrada em AJUSTES, letra normal e lista inteira."""
    return AJUSTES.get(qid, ("", None))


def texto_do_item(e):
    return f"{texto_puro(e.exibir)}: {texto_puro(e.defin)}"


def caracteres(textos):
    return len(SEPARADOR.join(textos))


def selecionar_itens(chaves, mantidos_n):
    """(mantidos, retirados): `chaves` já está na ordem de prioridade; com corte (mantidos_n não nulo) ficam os
    N primeiros, o que retira os termos do fim para o começo e só depois as siglas, do fim para o começo."""
    if mantidos_n is None:
        return list(chaves), []
    return list(chaves[:mantidos_n]), list(chaves[mantidos_n:])


def por_prioridade(chaves, nivel):
    """Siglas antes de termos, cada nível na ordem de aparição."""
    return [c for n in NO_LEMBRETE for c in chaves if nivel[c] == n]


def lembrete_do_quadro(html, glossario=None):
    """id do quadro -> (itens mantidos, itens retirados pelo orçamento)."""
    glossario = glossario if glossario is not None else ler_glossario(html)
    todos = termos_do_lembrete(html, glossario)
    return {q.id: selecionar_itens(todos[q.id], ajuste_do_quadro(q.id)[1]) for q in quadros(html)}


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
    return {q.id: ajuste_do_quadro(q.id)[0] for q in quadros(html)}


def html_do_item(e):
    return f'<span class="it"><strong>{e.exibir}</strong>: {e.defin}</span>'


FIXO_HTML = f'<span class="it"><strong>{DEMAIS[0]}</strong>: {DEMAIS[1]}</span>'


def itens_completos(html, glossario=None):
    """id do quadro -> HTML de TODOS os itens do lembrete, em ordem de prioridade (sem corte e sem a tabela),
    para a calibração no navegador; só os quadros com termo no lembrete."""
    glossario = glossario if glossario is not None else ler_glossario(html)
    por = {e.chave: e for e in glossario}
    return {q: [html_do_item(por[c]) for c in chaves] for q, chaves in termos_do_lembrete(html, glossario).items() if chaves}


# ------------------------------------------------------------------ geração

def render_lembrete(chaves, glossario, eol="\n", retirou=False, degrau=""):
    por_chave = {e.chave: e for e in glossario}
    linhas = ["<!-- LEMBRETE -->"]
    itens = [html_do_item(por_chave[c]) for c in chaves]
    if retirou:
        itens.append(FIXO_HTML)
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
    aparição, número de lembretes diferente de um, degrau de letra ou corte incoerente com a tabela AJUSTES, item "demais termos" ausente ou indevido, e
    data-termos diferente do detectado."""
    erros = []
    glossario = ler_glossario(html)
    por_chave = {e.chave: e for e in glossario}
    selecao = lembrete_do_quadro(html, glossario)
    degraus = degraus_do_quadro(html, glossario)
    todos = termos_por_quadro(html, glossario)
    todos_lembrete = termos_do_lembrete(html, glossario)
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
        degrau, corte = ajuste_do_quadro(q.id)
        if degrau not in DEGRAUS:
            erros.append(f"{q.id}: degrau de letra {degrau!r} fora de {DEGRAUS}")
        if corte is not None and not 1 <= corte < len(todos_lembrete[q.id]):
            erros.append(f"{q.id}: corte de {corte} itens fora de 1..{len(todos_lembrete[q.id]) - 1} "
                         f"(lista de {len(todos_lembrete[q.id])}); recalibrar AJUSTES")
        classe = re.search(r'<aside class="lembrete(?: ([a-z]+))?"', bloco)
        if classe is not None and (classe.group(1) or "") != degraus[q.id]:
            erros.append(f"{q.id}: degrau de letra {classe.group(1) or 'normal'!r} incoerente com a tabela "
                         f"(esperado {degraus[q.id] or 'normal'!r})")
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
