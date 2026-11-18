"""Contrato de testes do roteiro de fala do deck de defesa em HTML (text/slides/roteiro_defesa_html.md).

Espelha o que ``tests/test_defesa_slides.py`` verifica para o roteiro do deck Beamer e liga o roteiro novo
ao deck ``text/slides/defesa.html`` (via ``extrair_deck`` de ``tests/test_defesa_html.py``) e à tese:

R1. Estrutura: uma seção por quadro, S01 a S28 e R01 a R26, na ordem e com o título do HTML;
    cada seção principal com tempo, fala, transição e perguntas prováveis; cada reserva com pergunta,
    origem e resposta de duas a quatro frases.
R2. Tempo: tabela por quadro e por bloco somando 2.400 s (40 min); meta de 130 palavras por minuto;
    palavras da fala dentro de 125% da meta de cada quadro e do total de 5.200 palavras.
R3. Remissões: toda reserva citada existe; a fala só cita reservas que o quadro do HTML liga; a
    "Origem" de cada reserva é o conjunto de quadros principais que a ligam; nenhuma reserva ligada
    fica sem menção; reserva citada nas perguntas e não ligada leva "pelo índice".
R4. Terminologia (lista do contrato do deck Beamer e do HTML), travessões, valores retirados e
    primeira pessoa (singular e plural).
R5. Glosas de *linkage*, *grey area*, VPP e Random Forest uma vez cada, no quadro certo, e siglas
    expandidas antes do primeiro uso; nomes de modelo por extenso.
R6. Números: todo número da fala está na tese ou no registro (mesma função do contrato do HTML), as
    datas e versões idem; o que não está no quadro correspondente consta da tabela "Rastreabilidade",
    com a linha da tese conferida (o número está na linha citada).
R7. Fidelidade: as ressalvas dos quadros e a mensagem central das âncoras aprovadas aparecem na fala.

Cada função de extração e de verificação tem caso sintético que prova que o defeito é acusado.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import test_defesa_html as hh  # noqa: E402  (extrair_deck, numeros_ausentes_html, datas e versões)
import test_defesa_slides as bz  # noqa: E402  (funções genéricas do contrato do deck Beamer)

ROTEIRO = ROOT / "text/slides/roteiro_defesa_html.md"
PRINCIPAIS = hh.PRINCIPAIS
RESERVAS = hh.RESERVAS
TODOS = hh.TODOS
TESE_DIR = bz.TESE_DIR

TOTAL_SEGUNDOS = 2400
MAX_PALAVRAS_FALA_TOTAL = 5200
FATOR_META = 1.25
MAX_PALAVRAS_RESERVA = 170
BLOCOS_ESPERADOS = {"Abertura": ("S01", "S06"), "Material e medida": ("S07", "S10"),
                    "Resultados": ("S11", "S20"), "Fechamento": ("S21", "S28")}
EXAMINADORES = ("Pinheiro", "Coeli", "Camargo Jr.", "Bastos", "Costa", "Qualquer membro")

PREFIXOS_META_S = ("Tempo:", "Transição:", "Perguntas prováveis:")
PREFIXOS_META_R = ("Pergunta:", "Origem:")


# ======================================================================================
# Extração
# ======================================================================================

def secoes(md: str) -> dict[str, bz.SecaoRoteiro]:
    return bz.secoes_roteiro(md)


def paragrafo(s: bz.SecaoRoteiro, prefixo: str) -> list[str]:
    return [p for p in s.paragrafos if p.startswith(prefixo)]


def fala_de(s: bz.SecaoRoteiro) -> list[str]:
    meta = PREFIXOS_META_S + PREFIXOS_META_R
    return [p for p in s.paragrafos if not p.startswith(meta)]


def texto_dito(s: bz.SecaoRoteiro) -> str:
    """Título, fala, transição e perguntas prováveis (sem 'Tempo' nem 'Origem')."""
    partes = [s.titulo] + [p for p in s.paragrafos if not p.startswith(("Tempo:", "Origem:"))]
    return bz.plano_md("\n\n".join(partes))


def texto_fala(s: bz.SecaoRoteiro) -> str:
    """Só a fala (sem transição, perguntas, tempo, origem)."""
    return bz.plano_md("\n\n".join(fala_de(s)))


def palavras_fala_secao(s: bz.SecaoRoteiro) -> int:
    textos = fala_de(s) + [t.removeprefix("Transição:") for t in paragrafo(s, "Transição:")]
    return bz.contar_palavras(bz.plano_md("\n\n".join(textos)))


def _norm_titulo(t: str) -> str:
    return unicodedata.normalize("NFC", " ".join(bz.plano_md(t).split()))


def titulo_confere(titulo_roteiro: str, titulo_html: str) -> bool:
    return _norm_titulo(titulo_roteiro) == unicodedata.normalize("NFC", " ".join(titulo_html.split()))


def tabela_blocos(md: str) -> dict[str, tuple[str, str, int, str]]:
    """``{bloco: (de, até, segundos, minutos)}`` mais a linha 'Total'."""
    saida = {}
    for linha in md.splitlines():
        cel = [c.strip() for c in linha.strip().strip("|").split("|")]
        if len(cel) == 4 and re.fullmatch(r"S\d\d a S\d\d", cel[1]):
            a, b = cel[1].split(" a ")
            saida[cel[0]] = (a, b, int(cel[2].replace(".", "")), cel[3])
    return saida


def frases(texto: str) -> list[str]:
    return [f for f in re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ*])", texto.strip()) if f]


RE_RESERVA = re.compile(r"\bR\d\d\b")
RE_QUADRO = re.compile(r"\b[SR]\d\d\b")


def reservas_em(texto: str) -> list[str]:
    return RE_RESERVA.findall(texto)


def reservas_sem_indice(perguntas: str, ligadas: set[str]) -> list[str]:
    """Reservas citadas nas perguntas que o quadro não liga e que não trazem ', pelo índice'."""
    erros = []
    for m in RE_RESERVA.finditer(perguntas):
        if m.group(0) in ligadas:
            continue
        if not perguntas[m.end():].startswith(", pelo índice"):
            erros.append(m.group(0))
    return erros


# ======================================================================================
# Terminologia, primeira pessoa e nomes de modelo
# ======================================================================================

PRIMEIRA_PESSOA = re.compile(
    r"\b(?:eu|me|mim|comigo|meu|meus|minha|minhas|nós|nosso|nossa|nossos|nossas|conosco|"
    r"agradeço|agradecemos|convido|convidamos|apresento|apresentamos|explico|explicamos|digo|dizemos|"
    r"encerro|encerramos|concluo|concluímos|parto|partimos|vamos|veremos|vejamos|passamos|passemos|"
    r"temos|podemos|devemos|fazemos|chegamos|sabemos|queremos|mostro|mostramos|proponho|propomos|"
    r"observamos|usamos|medimos|avaliamos|recomendamos|sustentamos|defendemos|registramos|"
    r"reimplementamos|estimamos|calculamos|comparamos|verificamos)\b", re.I)


def primeira_pessoa(texto: str) -> list[str]:
    return [f"...{texto[max(0, m.start() - 30):m.end() + 30]}..." for m in PRIMEIRA_PESSOA.finditer(texto)]


MODELOS = (  # (padrão que acusa nome incompleto, descrição)
    (re.compile(r"(?<!Claude )\bHaiku\b"), "'Haiku' sem 'Claude'"),
    (re.compile(r"\bHaiku\b(?! 5\.5)"), "'Haiku' sem '5.5'"),
    (re.compile(r"(?<!Claude )\bSonnet\b"), "'Sonnet' sem 'Claude'"),
    (re.compile(r"\bSonnet\b(?! 5\.5)"), "'Sonnet' sem '5.5'"),
    (re.compile(r"(?<!Claude )\bOpus\b"), "'Opus' sem 'Claude'"),
    (re.compile(r"\bOpus\b(?! 5\.5)"), "'Opus' sem '5.5'"),
    (re.compile(r"(?<!Claude )\bFable\b"), "'Fable' sem 'Claude'"),
    (re.compile(r"\bFable\b(?! 5\.1)"), "'Fable' sem '5.1'"),
    (re.compile(r"\bDeepSeek\b(?! V4\.1 Flash)"), "'DeepSeek' sem 'V4.1 Flash'"),
)


def modelos_incompletos(texto: str) -> list[str]:
    return [f"{desc}: ...{texto[max(0, m.start() - 20):m.end() + 20]}..." for padrao, desc in MODELOS
            for m in padrao.finditer(texto)]


def termos_proibidos_roteiro(md: str) -> list[str]:
    texto = bz.plano_md(md)
    achados = bz.termos_proibidos(texto)
    for nome, padrao in hh.EXPRESSOES_PROIBIDAS_EXTRA:
        achados += [f"'{nome}': ...{texto[max(0, m.start() - 40):m.end() + 40]}..." for m in padrao.finditer(texto)]
    return achados


# ======================================================================================
# Glosas
# ======================================================================================

GLOSAS = (("relacionamento de registros", "S02"), ("área cinza", "S03"),
          ("valor preditivo positivo", "S09"), ("floresta aleatória", "S12"))

EXPANSOES = (("SIM", "Sistema de Informações sobre Mortalidade"),
             ("Sinan", "Sistema de Informação de Agravos de Notificação"),
             ("CdR", "Comparador de Registros"),
             ("MACD", "Medidas Contínuas de Diferença de Datas"),
             ("ASR", "arranjo sequencial"),
             ("VPP", "valor preditivo positivo"),
             ("XGBoost", "árvores de decisão"))


def ocorrencias_por_secao(sec: dict[str, bz.SecaoRoteiro], expressao: str) -> dict[str, int]:
    saida = {}
    for rotulo, s in sec.items():
        n = len(re.findall(expressao, bz.plano_md(s.titulo + "\n\n" + "\n\n".join(s.paragrafos)), re.I))
        if n:
            saida[rotulo] = n
    return saida


def siglas_sem_expansao(texto_em_ordem: str, expansoes=EXPANSOES) -> list[str]:
    """Siglas cujo primeiro uso antecede a expansão (ou que não têm expansão no texto)."""
    erros = []
    for sigla, expansao in expansoes:
        uso = re.search(rf"\b{re.escape(sigla)}\b", texto_em_ordem)
        if not uso:
            continue
        exp = texto_em_ordem.find(expansao)
        if exp < 0 or exp > uso.start():
            erros.append(sigla)
    return erros


# ======================================================================================
# Números e rastreabilidade
# ======================================================================================

def tokens(texto: str) -> set[str]:
    return set(bz.tokens_numericos(hh._sem_datas_e_versoes(texto)))


def numeros_fora_do_quadro(dito: str, quadro: str) -> list[str]:
    return sorted(tokens(dito) - tokens(quadro))


def texto_do_quadro(q: hh.QuadroHtml) -> str:
    return " ".join(p for p in (q.titulo, q.corpo, q.ressalva, q.svg_texto) if p)


RE_FONTE = re.compile(r"`([^`:]+\.tex):(\d+)(?:-(\d+))?`")


def fontes_citadas(celula: str) -> list[tuple[str, int, int]]:
    return [(m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))) for m in RE_FONTE.finditer(celula)]


def rastreabilidade(md: str) -> list[tuple[str, str, str]]:
    """Linhas ``(quadro, números, fontes)`` da tabela 'Rastreabilidade'."""
    if "## Rastreabilidade" not in md:
        return []
    cauda = md.split("## Rastreabilidade", 1)[1]
    saida = []
    for linha in cauda.splitlines():
        cel = [c.strip() for c in linha.strip().strip("|").split("|")]
        if len(cel) == 3 and re.fullmatch(r"[SR]\d\d", cel[0]):
            saida.append((cel[0], cel[1], cel[2]))
    return saida


def numero_na_linha(numero: str, linhas: list[str]) -> bool:
    for linha in linhas:
        limpa = bz._limpar_numeros(linha)
        for m in re.finditer(r"\d+(?:[.,]\d+)*", limpa):
            if bz.normalizar_numero(m.group(0)) == numero:
                return True
    return False


def linhas_da_tese(arquivo: str, ini: int, fim: int) -> list[str]:
    caminho = TESE_DIR / arquivo
    if not caminho.exists():
        return []
    todas = caminho.read_text(encoding="utf-8").splitlines()
    return todas[ini - 1:fim]


def numeros_sem_apoio_na_linha(celula_numeros: str, celula_fontes: str) -> list[str]:
    """Números da célula que nenhuma das linhas citadas traz (ou fonte inexistente)."""
    linhas: list[str] = []
    for arquivo, ini, fim in fontes_citadas(celula_fontes):
        linhas += linhas_da_tese(arquivo, ini, fim)
    return [n for n in sorted(tokens(celula_numeros)) if not numero_na_linha(n, linhas)]


# ======================================================================================
# Fidelidade: expressões da fala (minúsculas, sem asteriscos)
# ======================================================================================

EXPRESSOES_FALA = {
    "S03": ["preservado", "não foi medido", "referência operacional", "61.485"],
    "S06": ["veio da ordem", "não da automação", "post hoc", "coorte única", "revisor perfeito simulado",
            "coorte nova", "seis variantes", "nenhuma fusão superou", "não exclui o zero"],
    "S07": ["revisora única", "sem dupla revisão", "hipótese de trabalho", "não são identificáveis",
            "delineamento"],
    "S08": ["preservado", "nenhum componente corrige"],
    "S09": ["revisor é perfeito simulado", "não é acerto"],
    "S11": ["equivalência não foi demonstrada", "efeito em dados reais não foi medido"],
    "S12": ["otimistas", "inclui zero", "não fundamentam"],
    "S13": ["mesma coorte", "não equivale a validação externa"],
    "S14": ["estimativa pontual", "post hoc", "sem intervalo de confiança que exclua o zero",
            "revisor perfeito simulado", "a ordem se inverte"],
    "S15": ["nenhuma fusão teve intervalo inteiramente abaixo de zero", "seis variantes",
            "sem correção para multiplicidade", "post hoc"],
    "S16": ["valores pontuais", "escolhidas entre variantes", "nenhuma fusão teve intervalo abaixo de zero frente a o1",
            "frente a o3, em nenhuma meta", "curva de esforço"],
    "S17": ["16,0", "subestimado", "não deve aceitar vínculos sozinho", "registro interno", "sem critério de sucesso",
            "leitura como resultado negativo é posterior", "revisor perfeito simulado"],
    "S18": ["não sustenta superioridade", "hipóteses exploratórias", "post hoc", "revisor perfeito simulado"],
    "S19": ["16,0", "não altera os 16,0", "exploração posterior", "uma realização", "717", "25 aceitos",
            "ordem de grandeza", "dezenas de não pares"],
    "S20": ["42,5", "95,0", "96,1", "não demonstra paridade", "raça/cor", "não foi examinada",
            "estrato do processo"],
    "S21": ["otimistas", "seis variantes", "post hoc", "coorte única", "revisor perfeito simulado"],
    "S22": ["revisora única", "não são identificáveis", "247", "revisor perfeito simulado", "post hoc"],
    "S23": ["o3", "f4", "f3 quando houver as macd", "curva de esforço", "não examinados", "só apoia",
            "coorte nova", "hipótese de trabalho"],
    "S24": ["um só extrato", "revisor perfeito simulado", "0.7.0", "experimental", "não foi publicada",
            "sem labels", "não alimentam os resultados", "dentro do ruído", "o3 e f4 dependem de",
            "altera a recomendação"],
    "S25": ["25%", "limite inferior", "dados à vista", "não foi testada", "dupla revisão", "não foi estimado"],
    "S26": ["não reivindica", "sem auditoria independente", "posterior ao registro"],
    "S27": ["não fornece os modelos ajustados", "reproduz, por si só", "auditoria", "experimental",
            "não foi publicada", "perfil histórico"],
}


def expressoes_ausentes(texto: str, expressoes: list[str]) -> list[str]:
    t = " ".join(texto.lower().split())
    return [e for e in expressoes if e not in t]


# ======================================================================================
# Fixtures
# ======================================================================================

@pytest.fixture(scope="module")
def roteiro() -> str:
    return ROTEIRO.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def sec(roteiro) -> dict[str, bz.SecaoRoteiro]:
    return secoes(roteiro)


@pytest.fixture(scope="module")
def deck() -> hh.DocumentoHtml:
    return hh.extrair_deck(hh.DECK.read_bytes().decode("utf-8"))


@pytest.fixture(scope="module")
def quadros(deck) -> dict[str, hh.QuadroHtml]:
    return deck.por_id


@pytest.fixture(scope="module")
def tese_numeros() -> set[str]:
    return bz.numeros_da_tese()


@pytest.fixture(scope="module")
def ligacoes(deck) -> dict[str, set[str]]:
    """Reservas ligadas por cada quadro (hrefs do HTML)."""
    return {q.id: {h[1:] for h in q.hrefs if h.startswith("#R")} for q in deck.quadros}


# ======================================================================================
# Casos sintéticos das extrações e verificações
# ======================================================================================

MD_SINT = """# Roteiro

## Cronometragem

| Bloco | Quadros | Segundos | Minutos |
| :- | :- | -: | -: |
| Abertura | S01 a S02 | 90 | 1,5 |
| Total | S01 a S02 | 90 | 1,5 |

## Fala-guia por quadro

### S01. Primeiro *quadro*

Tempo: 20 s (meta de 43 palavras).

Fala um com cinco palavras.

Transição: ponte curta.

Perguntas prováveis: Coeli: algo (R01); Bastos: outro (R02, pelo índice).

### S02. Segundo

Tempo: 70 s (meta de 152 palavras).

Fala dois.

Transição: fim.

Perguntas prováveis: nenhuma específica.

## Reserva

### R01. Reserva: um

Pergunta: Por quê?

Origem: S01.

Primeira frase. Segunda frase.

## Rastreabilidade

| Quadro | Número falado | Fonte |
| :- | :- | :- |
| S01 | 61.485 | `capitulo3.tex:11` |
"""


def test_sintetico_secoes_e_paragrafos():
    s = secoes(MD_SINT)
    assert list(s) == ["S01", "S02", "R01"]
    assert fala_de(s["S01"]) == ["Fala um com cinco palavras."]
    assert paragrafo(s["S01"], "Transição:") == ["Transição: ponte curta."]
    assert fala_de(s["R01"]) == ["Primeira frase. Segunda frase."]
    assert paragrafo(s["R01"], "Origem:") == ["Origem: S01."]
    assert "Rastreabilidade" not in " ".join(s["R01"].paragrafos)
    assert palavras_fala_secao(s["S01"]) == 5 + 2
    assert "Coeli" not in texto_fala(s["S01"]) and "Coeli" in texto_dito(s["S01"])


def test_sintetico_titulo_confere_ignora_asteriscos_e_acusa_diferenca():
    assert titulo_confere("Primeiro *quadro* com linkage", "Primeiro quadro com linkage")
    assert not titulo_confere("Primeiro quadro", "Primeiro quadro novo")
    assert not titulo_confere("Titulo sem acento", "Título sem acento")


def test_sintetico_tabela_de_tempos_e_blocos():
    assert bz.tabela_cronometragem("| S01 | Capa | 20 | 43 |\n| S02 | X | 70 | 152 |\n") == {"S01": (20, 43), "S02": (70, 152)}
    assert tabela_blocos(MD_SINT) == {"Abertura": ("S01", "S02", 90, "1,5"), "Total": ("S01", "S02", 90, "1,5")}


def test_sintetico_frases_e_contagem():
    assert len(frases("Primeira frase. Segunda frase. Terceira, com Jiao et al. (2021), e 0,5.")) == 3
    assert len(frases("Uma só frase com 4,5 e et al. (2021).")) == 1


def test_sintetico_reservas_sem_indice():
    assert reservas_sem_indice("Coeli: algo (R01); Bastos: outro (R02, pelo índice).", {"R01"}) == []
    assert reservas_sem_indice("Coeli: algo (R01); Bastos: outro (R02).", {"R01"}) == ["R02"]
    assert reservas_sem_indice("outro (R02, pelo índice)", set()) == []


@pytest.mark.parametrize("texto", ["eu apresento o resultado", "nós temos duas filas", "como mostramos antes",
                                   "vamos ao próximo quadro", "agradecemos à banca", "minha tese", "comigo", "me parece"])
def test_sintetico_primeira_pessoa_e_acusada(texto):
    assert primeira_pessoa(texto), texto


def test_sintetico_primeira_pessoa_aceita_voz_impessoal():
    assert primeira_pessoa("A tese sustenta que os passos 6 e 9 e nos passos de blocagem; registram-se os agradecimentos.") == []


@pytest.mark.parametrize("texto", ["o Haiku 5.5", "Claude Haiku", "Claude Sonnet", "o Opus 5.5", "Fable 5.1", "o DeepSeek"])
def test_sintetico_modelo_incompleto_e_acusado(texto):
    assert modelos_incompletos(texto), texto


def test_sintetico_modelos_completos_passam():
    assert modelos_incompletos("Claude Haiku 5.5, Claude Sonnet 5.5, Claude Opus 5.5, Claude Fable 5.1 e DeepSeek V4.1 Flash") == []


@pytest.mark.parametrize("texto", ["isto — aquilo", "isto – aquilo", "a zona cinzenta", "a revisão clerical", "o não vínculo",
                                   "o PPV", "precisão do modelo", "a nota do CdR", "só F4", "RRF no lugar de O3",
                                   "um F1 de 0,485", "o limiar −5,648", "(área cinza) e a área cinza", "relacionamento de registros"])
def test_sintetico_termo_proibido_e_acusado(texto):
    assert termos_proibidos_roteiro(texto), texto


def test_sintetico_termos_aceitos():
    assert termos_proibidos_roteiro("nota final do CdR, precisão média, *linkage* (relacionamento de registros), "
                                    "*grey area* (área cinza), −21 e curva precisão-sensibilidade") == []


def test_sintetico_siglas_e_glosas():
    ok = "Comparador de Registros, o CdR, e depois o CdR de novo."
    assert siglas_sem_expansao(ok, (("CdR", "Comparador de Registros"),)) == []
    assert siglas_sem_expansao("O CdR vem antes do Comparador de Registros.", (("CdR", "Comparador de Registros"),)) == ["CdR"]
    assert siglas_sem_expansao("O CdR sozinho.", (("CdR", "Comparador de Registros"),)) == ["CdR"]
    s = secoes(MD_SINT.replace("Fala dois.", "Fala dois (relacionamento de registros)."))
    assert ocorrencias_por_secao(s, "relacionamento de registros") == {"S02": 1}


def test_sintetico_numeros_fora_do_quadro_e_da_tese():
    assert numeros_fora_do_quadro("Foram 61.485 e 85,4% de 211 pares.", "O quadro traz 61.485 e 211.") == ["85.4"] or \
        numeros_fora_do_quadro("Foram 61.485 e 85,4% de 211 pares.", "O quadro traz 61.485 e 211.") == [bz.normalizar_numero("85,4")]
    assert numeros_fora_do_quadro("Só 61.485 e O3 e 7.", "61.485") == []
    tese = {"61485", "0.910"}
    assert hh.numeros_ausentes_html("Valor 61.485, 0,910 e 99.999.", tese)[0][0] == "99999"


def test_sintetico_fontes_e_numero_na_linha():
    assert fontes_citadas("`capitulo3.tex:11`; `tables/editorial/x.tex:3-5`") == [("capitulo3.tex", 11, 11), ("tables/editorial/x.tex", 3, 5)]
    assert fontes_citadas("sem fonte") == []
    assert numero_na_linha(bz.normalizar_numero("61.485"), ["tem 61{.}485 potenciais pares"])
    assert not numero_na_linha(bz.normalizar_numero("61.485"), ["tem 61.486 potenciais pares"])
    assert [r[0] for r in rastreabilidade(MD_SINT)] == ["S01"]
    assert numeros_sem_apoio_na_linha("99.999", "`arquivo_que_nao_existe.tex:1`") == [bz.normalizar_numero("99.999")]
    assert numeros_sem_apoio_na_linha("61.485", "`capitulo3.tex:11`") == []
    assert numeros_sem_apoio_na_linha("61.485", "`capitulo3.tex:1`") == [bz.normalizar_numero("61.485")]


def test_sintetico_expressoes_ausentes():
    assert expressoes_ausentes("Texto com Post Hoc e  coorte única", ["post hoc", "coorte única"]) == []
    assert expressoes_ausentes("Texto", ["post hoc"]) == ["post hoc"]


# ======================================================================================
# R1. Estrutura
# ======================================================================================

def test_r1_secoes_s01_a_s28_e_r01_a_r26_na_ordem(sec):
    assert list(sec) == TODOS, f"seções do roteiro: {list(sec)}"
    assert len(sec) == 54


@pytest.mark.parametrize("rotulo", TODOS)
def test_r1_titulo_igual_ao_do_html(sec, quadros, rotulo):
    assert titulo_confere(sec[rotulo].titulo, quadros[rotulo].titulo), \
        f"{rotulo}: roteiro {sec[rotulo].titulo!r} != HTML {quadros[rotulo].titulo!r}"


@pytest.mark.parametrize("rotulo", PRINCIPAIS)
def test_r1_secao_principal_tem_tempo_fala_transicao_e_perguntas(sec, rotulo):
    s = sec[rotulo]
    assert len(paragrafo(s, "Tempo:")) == 1, f"{rotulo}: Tempo"
    t = paragrafo(s, "Transição:")
    assert len(t) == 1 and len(t[0]) > len("Transição:") + 15, f"{rotulo}: Transição"
    p = paragrafo(s, "Perguntas prováveis:")
    assert len(p) == 1 and len(p[0]) > len("Perguntas prováveis:") + 8, f"{rotulo}: Perguntas prováveis"
    assert len(fala_de(s)) >= 1 and bz.contar_palavras(texto_fala(s)) >= 30


@pytest.mark.parametrize("rotulo", RESERVAS)
def test_r1_reserva_tem_pergunta_origem_e_resposta_de_duas_a_quatro_frases(sec, rotulo):
    s = sec[rotulo]
    assert len(paragrafo(s, "Pergunta:")) == 1 and paragrafo(s, "Pergunta:")[0].rstrip().endswith("?"), f"{rotulo}: Pergunta"
    assert len(paragrafo(s, "Origem:")) == 1, f"{rotulo}: Origem"
    resposta = fala_de(s)
    assert len(resposta) == 1, f"{rotulo}: {len(resposta)} parágrafos de resposta"
    n = len(frases(resposta[0]))
    assert 2 <= n <= 4, f"{rotulo}: {n} frases"
    assert bz.contar_palavras(texto_fala(s)) <= MAX_PALAVRAS_RESERVA, f"{rotulo}: {bz.contar_palavras(texto_fala(s))} palavras"


def test_r1_nomes_de_examinador_so_os_da_banca(sec):
    achados = set()
    for rotulo in PRINCIPAIS:
        p = paragrafo(sec[rotulo], "Perguntas prováveis:")[0].removeprefix("Perguntas prováveis:")
        for m in re.finditer(r"(?:^|[.;]\s+|,\s+e\s+)([A-ZÁÉÍÓÚ][\w.]*(?: [A-Z][\w.]*)?)\s*(?:,|:)", p.strip()):
            achados.add(m.group(1))
    nomes = {n for n in achados if n[0].isupper() and n.split()[0] in {"Pinheiro", "Coeli", "Camargo", "Bastos", "Costa", "Qualquer"}}
    assert nomes, "nenhum examinador extraído das perguntas; extração suspeita"
    for rotulo in PRINCIPAIS:
        p = paragrafo(sec[rotulo], "Perguntas prováveis:")[0]
        for nome in re.findall(r"\b(Pinheiro|Coeli|Camargo Jr\.|Bastos|Costa|Qualquer membro|Silva|Santos|Oliveira|Lucena)\b", p):
            assert nome in EXAMINADORES, f"{rotulo}: examinador fora da banca: {nome}"


# ======================================================================================
# R2. Tempo
# ======================================================================================

def test_r2_tempos_somam_40_minutos_por_quadro_e_por_bloco(roteiro):
    tabela = bz.tabela_cronometragem(roteiro)
    assert list(tabela) == PRINCIPAIS
    total = sum(s for s, _ in tabela.values())
    assert total == TOTAL_SEGUNDOS == 40 * 60, f"soma dos quadros: {total} s"
    assert bz.linha_total(roteiro)[0] == total
    blocos = tabela_blocos(roteiro)
    assert [b for b in blocos if b != "Total"] == list(BLOCOS_ESPERADOS)
    for nome, (a, b) in BLOCOS_ESPERADOS.items():
        seg = sum(s for k, (s, _) in tabela.items() if a <= k <= b)
        assert blocos[nome][:3] == (a, b, seg), f"{nome}: {blocos[nome]} (soma {seg})"
        assert blocos[nome][3] == f"{seg / 60:.1f}".replace(".", ","), f"{nome}: minutos {blocos[nome][3]}"
    assert blocos["Total"] == ("S01", "S28", TOTAL_SEGUNDOS, "40,0")
    assert sum(v[2] for k, v in blocos.items() if k != "Total") == TOTAL_SEGUNDOS


def test_r2_meta_de_palavras_segue_130_por_minuto_e_tempo_da_secao_confere(roteiro, sec):
    tabela = bz.tabela_cronometragem(roteiro)
    for rotulo, (seg, meta) in tabela.items():
        assert abs(meta - seg * 130 / 60) <= 1, f"{rotulo}: meta {meta} para {seg} s"
        assert paragrafo(sec[rotulo], "Tempo:")[0] == f"Tempo: {seg} s (meta de {meta} palavras).", \
            f"{rotulo}: {paragrafo(sec[rotulo], 'Tempo:')}"
    assert bz.linha_total(roteiro)[1] == sum(m for _, m in tabela.values())


def test_r2_novos_quadros_s19_s24_s27_sao_curtos(roteiro):
    tabela = bz.tabela_cronometragem(roteiro)
    assert [tabela[k][0] for k in ("S19", "S24", "S27")] == [60, 60, 60]
    assert tabela["S28"][0] == 20 and tabela["S01"][0] == 20


def test_r2_total_de_palavras_da_fala_cabe_em_40_minutos(sec):
    total = sum(palavras_fala_secao(sec[r]) for r in PRINCIPAIS)
    assert total <= MAX_PALAVRAS_FALA_TOTAL, f"total de fala e transição: {total}"
    assert total >= 0.8 * MAX_PALAVRAS_FALA_TOTAL, f"fala curta demais para 40 minutos: {total}"


def test_r2_nenhuma_secao_passa_de_125_por_cento_da_meta(roteiro, sec):
    tabela = bz.tabela_cronometragem(roteiro)
    estouros = [f"{r}: {palavras_fala_secao(sec[r])} palavras para meta {tabela[r][1]}"
                for r in PRINCIPAIS if palavras_fala_secao(sec[r]) > FATOR_META * tabela[r][1]]
    assert not estouros, "\n".join(estouros)


def test_r2_sintetico_orcamento_acusa_estouro():
    s = secoes(MD_SINT)
    assert palavras_fala_secao(s["S01"]) < 1.25 * 43
    longo = secoes(MD_SINT.replace("Fala um com cinco palavras.", "palavra " * 80))
    assert palavras_fala_secao(longo["S01"]) > 1.25 * 43


# ======================================================================================
# R3. Remissões
# ======================================================================================

def test_r3_todo_quadro_citado_existe(roteiro):
    texto = bz.sem_fonte_md(roteiro)
    citados = set(RE_QUADRO.findall(texto))
    assert citados <= set(TODOS), f"quadros citados e inexistentes: {sorted(citados - set(TODOS))}"
    assert {f"R{i:02d}" for i in range(1, 27)} <= citados | set(RESERVAS)
    assert len(reservas_em(texto)) >= 80, "poucas remissões; extração suspeita"


def test_r3_fala_so_cita_reservas_que_o_quadro_liga(sec, ligacoes):
    erros = []
    for rotulo in PRINCIPAIS + RESERVAS:
        citadas = set(reservas_em(texto_fala(sec[rotulo])))
        extras = citadas - ligacoes[rotulo]
        if extras:
            erros.append(f"{rotulo}: fala cita {sorted(extras)}, mas o quadro liga {sorted(ligacoes[rotulo])}")
    assert not erros, "\n".join(erros)


def test_r3_toda_reserva_ligada_pelo_quadro_e_mencionada(sec, ligacoes):
    erros = []
    for rotulo in PRINCIPAIS:
        mencionadas = set(reservas_em(texto_dito(sec[rotulo])))
        faltam = ligacoes[rotulo] - mencionadas
        if faltam:
            erros.append(f"{rotulo}: reservas ligadas e não mencionadas: {sorted(faltam)}")
    assert not erros, "\n".join(erros)


def test_r3_perguntas_citam_reserva_nao_ligada_so_com_pelo_indice(sec, ligacoes):
    erros = []
    for rotulo in PRINCIPAIS:
        p = paragrafo(sec[rotulo], "Perguntas prováveis:")[0]
        erros += [f"{rotulo}: {r} sem ', pelo índice'" for r in reservas_sem_indice(p, ligacoes[rotulo])]
    assert not erros, "\n".join(erros)


def test_r3_origem_da_reserva_e_o_conjunto_de_quadros_que_a_ligam(sec, ligacoes):
    erros = []
    for r in RESERVAS:
        esperado = sorted(s for s in PRINCIPAIS if r in ligacoes[s])
        dado = sorted(set(re.findall(r"\bS\d\d\b", paragrafo(sec[r], "Origem:")[0])))
        if dado != esperado:
            erros.append(f"{r}: origem {dado}, o HTML liga a partir de {esperado}")
    assert not erros, "\n".join(erros)


def test_r3_toda_reserva_tem_origem_em_quadro_principal(ligacoes):
    sem = [r for r in RESERVAS if not any(r in ligacoes[s] for s in PRINCIPAIS)]
    assert not sem, f"reservas sem remissão a partir de quadro principal: {sem}"


# ======================================================================================
# R4. Terminologia, travessões, primeira pessoa
# ======================================================================================

def test_r4_roteiro_sem_termos_proibidos(roteiro):
    problemas = termos_proibidos_roteiro(roteiro)
    assert not problemas, "\n".join(problemas)


def test_r4_roteiro_sem_primeira_pessoa(roteiro):
    problemas = primeira_pessoa(bz.plano_md(roteiro))
    assert not problemas, "\n".join(problemas)


def test_r4_roteiro_sem_travessao_nem_hifen_duplo(roteiro):
    assert not re.search("[\u2013\u2014]|--", roteiro)


def test_r4_nomes_de_modelo_por_extenso(roteiro):
    problemas = modelos_incompletos(bz.plano_md(roteiro))
    assert not problemas, "\n".join(problemas)
    assert "Claude Haiku 5.5" in roteiro and "DeepSeek V4.1 Flash" in roteiro


def test_r4_agradecimentos_sem_nomes_alem_da_orientadora(sec):
    fala = texto_fala(sec["S28"])
    assert "Rejane Sobrino Pinheiro" in fala
    nomes = re.findall(r"\b[A-ZÁÉÍÓÚ][a-záéíóúãõç]+ [A-ZÁÉÍÓÚ][a-záéíóúãõç]+(?: [A-ZÁÉÍÓÚ][a-záéíóúãõç]+)*\b", fala.replace("Rejane Sobrino Pinheiro", ""))
    permitidos = {"Rejane Sobrino", "Os agradecimentos", "A palavra"}
    assert not [n for n in nomes if n not in permitidos], f"nomes nos agradecimentos: {nomes}"


# ======================================================================================
# R5. Glosas e siglas
# ======================================================================================

@pytest.mark.parametrize("expressao,rotulo", GLOSAS)
def test_r5_glosa_uma_vez_no_roteiro_e_no_quadro_certo(sec, expressao, rotulo):
    por_secao = ocorrencias_por_secao(sec, expressao)
    assert sum(por_secao.values()) == 1 and list(por_secao) == [rotulo], \
        f"'{expressao}' no roteiro: {por_secao} (esperado uma vez, em {rotulo})"


def test_r5_glosa_de_linkage_esta_no_titulo_do_s02_e_nao_repete(roteiro, sec):
    assert "(relacionamento de registros)" in bz.plano_md(sec["S02"].titulo)
    assert len(re.findall("relacionamento de registros", bz.plano_md(roteiro))) == 1


def test_r5_grey_area_vpp_e_asr_so_depois_da_glosa(sec):
    for rotulo in PRINCIPAIS[:2]:
        assert not re.search(r"grey area", texto_dito(sec[rotulo]), re.I), f"{rotulo}: grey area antes de S03"
    for rotulo in PRINCIPAIS[:8]:
        t = texto_dito(sec[rotulo])
        for sigla in ("VPP", "ASR"):
            assert not re.search(rf"\b{sigla}\b", t), f"{rotulo}: {sigla} antes de S09"


def test_r5_siglas_expandidas_antes_do_primeiro_uso(sec):
    em_ordem = bz.plano_md("\n\n".join(texto_dito(sec[r]) for r in TODOS))
    assert siglas_sem_expansao(em_ordem) == []


# ======================================================================================
# R6. Números
# ======================================================================================

def test_r6_numeros_do_roteiro_estao_na_tese(sec, tese_numeros):
    problemas = []
    total = 0
    for rotulo in TODOS:
        texto = texto_dito(sec[rotulo])
        total += len(bz.tokens_numericos(hh._sem_datas_e_versoes(texto)))
        problemas += [f"{rotulo}: {n} (...{trecho}...)" for n, trecho in hh.numeros_ausentes_html(texto, tese_numeros)]
    assert total > 250, f"poucos números extraídos do roteiro ({total}); extração suspeita"
    assert not problemas, "números do roteiro fora da tese e do registro:\n" + "\n".join(problemas)


def test_r6_datas_e_versoes_do_roteiro_estao_na_tese(sec):
    tese = bz.sem_comentarios("\n".join(c.read_text(encoding="utf-8") for c in sorted(TESE_DIR.rglob("*.tex"))
                                         if c.name != "tese_rev_diff.tex"))
    apend = bz.sem_comentarios(hh.APENDICE_PACOTE.read_text(encoding="utf-8")
                               + hh.APENDICE_GZCMD.read_text(encoding="utf-8"))
    datas = {(r, d) for r in TODOS for d in hh.datas_do_texto(texto_dito(sec[r]))}
    assert datas, "nenhuma data extraída do roteiro; extração suspeita"
    ausentes = [f"{r}: {d:02d}/{m:02d}/{a}" for r, (d, m, a) in sorted(datas) if hh.formato_da_data_na_tese(d, m, a, tese) is None]
    assert not ausentes, "datas do roteiro que a tese não traz:\n" + "\n".join(ausentes)
    versoes = {v for r in TODOS for v in hh.versoes_do_texto(texto_dito(sec[r]))}
    assert {"0.6.1", "0.7.0"} <= versoes
    assert not [v for v in versoes if v not in apend], f"versões fora dos apêndices: {sorted(v for v in versoes if v not in apend)}"


def test_r6_numeros_fora_do_quadro_constam_da_rastreabilidade(roteiro, sec, quadros):
    linhas = rastreabilidade(roteiro)
    assert len(linhas) >= 10, f"poucas linhas na tabela de rastreabilidade ({len(linhas)})"
    erros = []
    declarados: dict[str, set[str]] = {}
    for q, numeros, _ in linhas:
        declarados.setdefault(q, set()).update(tokens(numeros))
    for rotulo in TODOS:
        fora = set(numeros_fora_do_quadro(texto_dito(sec[rotulo]), texto_do_quadro(quadros[rotulo])))
        faltam = fora - declarados.get(rotulo, set())
        if faltam:
            erros.append(f"{rotulo}: números falados fora do quadro e sem linha na rastreabilidade: {sorted(faltam)}")
        obsoletos = declarados.get(rotulo, set()) - tokens(texto_dito(sec[rotulo]))
        if obsoletos:
            erros.append(f"{rotulo}: rastreabilidade traz números que a fala não diz: {sorted(obsoletos)}")
    assert not erros, "\n".join(erros)


def test_r6_cada_linha_da_rastreabilidade_traz_o_numero_na_linha_citada(roteiro):
    erros = []
    for q, numeros, fontes in rastreabilidade(roteiro):
        if not fontes_citadas(fontes):
            erros.append(f"{q}: fonte sem arquivo:linha: {fontes!r}")
            continue
        erros += [f"{q}: {n} não está nas linhas citadas ({fontes})" for n in numeros_sem_apoio_na_linha(numeros, fontes)]
    assert not erros, "\n".join(erros)


def test_r6_rastreabilidade_so_cita_arquivos_da_tese_e_nunca_o_diff(roteiro):
    for _, _, fontes in rastreabilidade(roteiro):
        for arquivo, _, _ in fontes_citadas(fontes):
            assert (TESE_DIR / arquivo).exists() and "tese_rev_diff" not in arquivo, arquivo


# ======================================================================================
# R7. Fidelidade ao quadro e às âncoras
# ======================================================================================

@pytest.mark.parametrize("rotulo", sorted(EXPRESSOES_FALA))
def test_r7_fala_traz_as_ressalvas_do_quadro(sec, rotulo):
    ausentes = expressoes_ausentes(texto_fala(sec[rotulo]), EXPRESSOES_FALA[rotulo])
    assert not ausentes, f"{rotulo}: a fala omite {ausentes}\nfala: {texto_fala(sec[rotulo])}"


def test_r7_ressalvas_de_s14_a_s19_s21_s24_e_s27_estao_cobertas():
    for rotulo in ("S14", "S15", "S16", "S17", "S18", "S19", "S21", "S24", "S27"):
        assert rotulo in EXPRESSOES_FALA and len(EXPRESSOES_FALA[rotulo]) >= 4, rotulo


def test_r7_cada_ressalva_do_quadro_tem_expressao_no_html(quadros):
    """As expressões da fala não podem apontar para uma ressalva que o quadro nem traz."""
    for rotulo in ("S14", "S15", "S17", "S19", "S21", "S24", "S27"):
        texto = " ".join(quadros[rotulo].texto.lower().split())
        assert quadros[rotulo].ressalva, f"{rotulo}: quadro sem ressalva"
        assert "post hoc" in texto or "estimativa pontual" in texto or "exploração" in texto or "experimental" in texto or "revisor" in texto, rotulo


def test_r7_mensagem_central_das_ancoras(sec):
    s06 = texto_fala(sec["S06"]).lower()
    assert "veio da ordem" in s06 and "não da automação" in s06
    s21 = texto_fala(sec["S21"]).lower()
    assert "f3 e f4" in s21 and "reduziram" in s21 and "o1" in s21 and "metas de 9 e de 36" in s21
    assert "não sustenta que f3 ou f4 superem o3" in s21 or "não sustenta que f3 ou f4 superem o3" in " ".join(s21.split())
    s15 = texto_fala(sec["S15"]).lower()
    assert "nenhuma fusão teve intervalo inteiramente abaixo de zero" in s15 and "o3 isolada não tem marcador cheio" in s15
    s23 = texto_fala(sec["S23"]).lower()
    assert "isolada, em o3" in s23 and "em f4" in s23 and "f3 quando houver as macd" in s23
    assert "curva de esforço" in s23 and "coorte nova" in s23
    s24 = texto_fala(sec["S24"]).lower()
    assert "altera a recomendação" in s24 and "o1 é a mais forte" in s24


def test_r7_corte_sempre_preservado(sec):
    for rotulo in ("S03", "S08", "S23"):
        assert "preserv" in texto_fala(sec[rotulo]).lower(), rotulo
    for rotulo in PRINCIPAIS + RESERVAS:
        assert not re.search(r"corrig\w+ o openreclink|substitui o openreclink|dispensa o corte", texto_dito(sec[rotulo]), re.I) \
            or "não" in texto_dito(sec[rotulo]).lower(), rotulo


def test_r7_limitacoes_sao_de_delineamento_e_de_documentacao(sec):
    for rotulo in ("S07", "S22"):
        t = texto_fala(sec[rotulo]).lower()
        assert "delineamento" in t and "documentação" in t and "não do trabalho de revisão" in t.replace("e não do", "não do"), rotulo
    assert "limitação do delineamento" in texto_fala(sec["R01"]).lower() or "do delineamento e da documentação" in texto_fala(sec["R01"]).lower()


def test_r7_valores_retirados_ausentes(roteiro):
    assert "0,485" not in roteiro and "5,648" not in roteiro


def test_r7_s19_traz_o_texto_atualizado_do_quadro(quadros, sec):
    """O quadro S19 e a fala dizem que, no topo da ordem pela nota final, o modelo deu MATCH a dezenas de não pares."""
    assert "dezenas de não pares" in quadros["S19"].corpo and "dezenas de não pares" in texto_fala(sec["S19"])
    assert "reserva R15" in quadros["S19"].corpo and "reserva R11" in quadros["S19"].corpo
    assert "R15" in texto_fala(sec["S19"]) and "R11" in texto_fala(sec["S19"])


def test_r7_instrucoes_de_uso_do_deck(roteiro):
    uso = roteiro.split("## Uso do deck", 1)[1].split("## Cronometragem", 1)[0]
    for item in ("seta para a direita", "Próximo", "Anterior", "`i`", "`defesa.html#S14`", "Impressão em PDF", "`Home`", "`End`"):
        assert item in uso, f"instrução de uso sem '{item}'"

# ======================================================================================
# R8. Correções da revisão independente de fidelidade (2026-11-18)
# ======================================================================================

def test_r8_s07_razao_e_um_par_verdadeiro_para_cada_249_nao_pares(sec):
    t = texto_fala(sec["S07"])
    assert "um par verdadeiro para cada 249 não pares" in t
    assert "razão de 1 para 249 pares verdadeiros" not in t


def test_r8_s06_qualifica_fusoes_e_situa_automacao_na_vigilancia(sec):
    t = texto_fala(sec["S06"])
    assert "frente à probabilidade isolada, nenhuma fusão superou com intervalo inteiramente abaixo de zero" in t
    assert "na vigilância, aceitou abaixo do corte 4 pares verdadeiros e 21 falsos positivos" in t


def test_r8_s08_nao_diz_que_o_corte_permanece_em_todos_os_cenarios(sec):
    t = texto_fala(sec["S08"])
    assert "em todos os cenários" not in t
    assert "arranjo sequencial e no fluxo recomendado" in t
    assert "é avaliada sobre as decisões resultantes" in t


def test_r8_s09_metas_sao_o_que_o_asr_recuperou(sec):
    t = texto_fala(sec["S09"])
    assert "As metas são o que o arranjo sequencial, o ASR, recuperou abaixo do corte" in t
    assert "o ASR recuperou 23" in t and "recuperou 9" in t


def test_r8_s19_e_s24_cabem_em_110_por_cento_da_meta(roteiro, sec):
    tabela = bz.tabela_cronometragem(roteiro)
    for rotulo in ("S02", "S05", "S19", "S24"):
        assert palavras_fala_secao(sec[rotulo]) <= 1.10 * tabela[rotulo][1], \
            f"{rotulo}: {palavras_fala_secao(sec[rotulo])} palavras para meta {tabela[rotulo][1]}"
    assert "estratificada" in texto_fala(sec["S19"]) and "labels" in texto_fala(sec["S24"])


def test_r8_transicao_do_s23_anuncia_o_s24(sec):
    t = paragrafo(sec["S23"], "Transição:")[0]
    assert "explorações" in t and "refutação" in t and "não alteram" in t


def test_r8_r02_comeca_por_sim_e_inclui_exploracoes_posteriores(sec):
    t = texto_fala(sec["R02"])
    assert t.startswith("Sim.") and "explorações posteriores" in t


def test_r8_r24_nomeia_o_random_forest_com_smote_e_nao_o_xgboost_de_o3(sec):
    t = texto_fala(sec["R24"])
    assert "Random Forest com SMOTE ajustado na base inteira, que não é o XGBoost que gera O3" in t
    assert "Gini" in t


def test_r8_nonmatch_glosado_na_primeira_ocorrencia(sec):
    em_ordem = bz.plano_md("\n\n".join(texto_dito(sec[r]) for r in TODOS))
    primeira = em_ordem.find("NONMATCH")
    assert primeira >= 0 and "decisão de não par" in em_ordem[primeira:primeira + 40]


def test_r8_uso_do_deck_end_espaco_e_data_estouro(roteiro):
    uso = roteiro.split("## Uso do deck", 1)[1].split("## Cronometragem", 1)[0]
    assert "`End` ao último quadro da sequência, a reserva R26, e não ao S28" in uso
    assert "`defesa.html#S28`" in uso
    assert "A barra de espaço aciona o botão que estiver em foco" in uso
    assert "`data-estouro`" in uso
    assert not re.search(r"`PageDown`, barra de espaço", uso)


def test_r8_custo_computacional_nas_perguntas_do_s17_com_rastreabilidade(roteiro, sec):
    p = paragrafo(sec["S17"], "Perguntas prováveis:")[0]
    assert "Costa: custo computacional" in p and "US$ 0,94" in p and "US$ 0,11" in p and "755 chamadas" in p
    linhas = [r for r in rastreabilidade(roteiro) if r[0] == "S17"]
    assert any("0,94" in n for _, n, _ in linhas) and any("0,11" in n for _, n, _ in linhas)
