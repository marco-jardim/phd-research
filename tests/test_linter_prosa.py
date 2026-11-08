"""Testes do linter de prosa (``scripts/linter_prosa.py``).

Casos sintéticos em ``tmp_path`` para cada regra e o gate da tese real contra
``tests/data/linter_prosa_baseline.json`` (gerado por
``python scripts/linter_prosa.py --gerar-baseline --forcar``; baseline v2 com uma impressão
digital por erro; avisos não entram no gate).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import linter_prosa as lp

BASELINE_REAL = Path(__file__).parent / "data" / "linter_prosa_baseline.json"


def lint(tmp_path: Path, corpo: str, nome: str = "x.tex", extras: dict[str, str] | None = None):
    """Escreve ``corpo`` (e ``extras``) em ``tmp_path`` e devolve as violações."""
    (tmp_path / nome).write_text(corpo, encoding="utf-8")
    for rel, conteudo in (extras or {}).items():
        destino = tmp_path / rel
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(conteudo, encoding="utf-8")
    return lp.lintar(lp.carregar_arvore(tmp_path / nome, base=tmp_path))


def da_regra(violacoes, regra: str):
    return [v for v in violacoes if v.regra == regra]


def n_regra(tmp_path: Path, corpo: str, regra: str, **kw) -> int:
    return len(da_regra(lint(tmp_path, corpo, **kw), regra))


# ---------------------------------------------------------------------------
# R01 travessão
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "texto",
    [
        "Texto \u2014 com travessão.",
        "Texto \u2013 com travessão.",
        "Texto --- com travessão.",
        "Texto -- com travessão.",
        "palavra--outra palavra.",
        r"Texto\textemdash{}outro.",
    ],
)
def test_r01_positivos(tmp_path, texto):
    assert n_regra(tmp_path, texto, "R01") == 1


@pytest.mark.parametrize(
    "texto",
    [
        "Páginas 10--20 do livro.",
        "Óbitos de 2019--2022 no município.",
        r"Entre 5\%--10\% dos casos.",
        "% \u2014 comentário\nTexto limpo.",
        "Fórmula $a --- b$ no texto.",
        "\\begin{verbatim}\n\u2014 código\n\\end{verbatim}\nTexto.",
        r"Ver \path{a--b} no repositório.",
        r"Ver \url{http://x.org/a--b} no repositório.",
        r"Opção \texttt{--seed} fixa.",
        r"Ver \cite{a--b} no texto.",
    ],
)
def test_r01_negativos(tmp_path, texto):
    assert n_regra(tmp_path, texto, "R01") == 0


def test_r01_celula_isolada_de_tabular_nao_conta(tmp_path):
    vazio = "\\begin{tabular}{lll}\nA & 1 & -- \\\\ \\midrule\nB & --- & 2 \\\\\n\\end{tabular}\n"
    assert n_regra(tmp_path, vazio, "R01") == 0
    # travessão no meio do conteúdo da célula continua sendo violação
    misto = "\\begin{tabular}{ll}\nA & x -- y \\\\\n\\end{tabular}\n"
    assert n_regra(tmp_path, misto, "R01") == 1
    # fora de tabular, mesmo isolado, continua sendo violação
    assert n_regra(tmp_path, "Texto.\n\n--\n\nOutro texto.\n", "R01") == 1


@pytest.mark.parametrize("ambiente", ["verbatim", "lstlisting"])
def test_transcricao_literal_com_linhas_em_branco_fica_fora(tmp_path, ambiente):
    corpo = (
        "{\\fontsize{6}{7}\\selectfont\n"
        f"\\begin{{{ambiente}}}\n"
        "# Prompt \u2014 revisão clerical de record linkage\n\n"
        "Nós usamos a zona cinzenta com 0.95 e --- tabelas.\n\n"
        "| a | b |\n|---|---|\n"
        f"\\end{{{ambiente}}}\n}}\n\n"
        "Texto limpo depois.\n"
    )
    proibidas = {"R01", "R02", "R04", "R05", "R06", "R07"}
    assert [v for v in lint(tmp_path, corpo) if v.regra in proibidas] == []


# ---------------------------------------------------------------------------
# R02 formas proibidas
# ---------------------------------------------------------------------------
CASOS_R02 = [
    ("A zona cinzenta é ampla.", 1),
    ("A zona cinza é ampla.", 1),
    (r"A \textit{grey area} (área cinza) é ampla.", 0),
    ("A área cinza é ampla.", 1),
    (r"O \textit{linkage} (relacionamento de registros) integra bases.", 0),
    ("O relacionamento de registros integra bases.", 1),
    ("A vinculação das bases.", 1),
    ("O record linkage probabilístico.", 1),
    ("O pacote gzcmd-record-linkage é público.", 0),
    ("A revisão clerical foi feita.", 1),
    ("Um não vínculo foi observado.", 1),
    ("O falso vínculo ocorreu.", 1),
    ("A restrição de falsos vínculos define a política.", 0),
    ("A precisão do modelo.", 1),
    ("A precisão média foi calculada.", 0),
    ("A curva precisão-sensibilidade foi traçada.", 0),
    # precisão só como nome de métrica (contexto: número, %, VPP, F1, sensibilidade, métrica)
    ("Mediu-se a precisão de 0,95 no teste.", 1),
    ("A precisão foi 0,95 no teste.", 1),
    (r"A precisão chegou a 85\% no teste.", 1),
    ("A precisão e a sensibilidade foram altas.", 1),
    ("A precisão, a sensibilidade e o F1 foram altas.", 1),
    ("Foi medida a sensibilidade e a precisão.", 1),
    ("O F1 e a precisão foram altos.", 1),
    ("A precisão (VPP) foi alta.", 1),
    ("A métrica de precisão foi usada.", 1),
    ("A precisão como métrica foi usada.", 1),
    ("A precisão do classificador foi usada.", 1),
    ("Os pares foram medidos com precisão.", 0),
    ("A precisão dos dados de endereço é baixa.", 0),
    ("A precisão da data de nascimento varia.", 0),
    ("Usou-se Random Forest (floresta aleatória) no estudo.", 0),
    ("Usou-se floresta aleatória no estudo.", 1),
    ("Usou-se Floresta Aleatória no estudo.", 1),
    ("A sigla RL aparece.", 1),
    ("A ferramenta CIDACS-RL aparece.", 0),
    ("O SINAN registra.", 1),
    ("O SINASC registra.", 1),
    ("O Sinan registra.", 0),
    ("Dados do SIH-SUS.", 1),
    ("Dados do SIH/SUS.", 0),
    ("Os pares duvidosos foram revisados.", 1),
    ("O padrão-ouro foi usado.", 1),
    ("O gold standard foi usado.", 1),
    ("A taxa de erro foi baixa.", 1),
    ("A taxa de aprendizado foi fixada.", 0),
    ("A taxa de incidência de tuberculose.", 0),
    ("As taxas de mortalidade variam.", 0),
    ("O recall foi alto.", 1),
    (r"O \textit{recall} foi alto.", 0),
    ("A curva precision-recall foi traçada.", 0),
    ("A calibração sigmoidal foi usada.", 1),
    ("O identificador único do registro.", 1),
    ("Segundo o autor, ``zona cinzenta'' é termo usual.", 0),
    (r"O campo \texttt{recall} foi lido.", 0),
    (r"A precis\~{a}o do modelo.", 1),
    ("O cut-off do OpenRecLink.", 1),
    ("Os bancos de dados foram unidos.", 1),
    ("Foi usado o PPV do classificador.", 1),
]


@pytest.mark.parametrize("texto,esperado", CASOS_R02)
def test_r02(tmp_path, texto, esperado):
    assert n_regra(tmp_path, texto, "R02") == esperado


TERMOS_R02_COM_ACENTO = [
    "a área cinza do modelo",
    "o relacionamento de registros",
    "o relacionamento probabilístico",
    "o relacionamento determinístico",
    "a vinculação das bases",
    "as vinculações das bases",
    "a revisão clerical dos pares",
    "o não vínculo observado",
    "os não vínculos observados",
    "um não-par observado",
    "dois não-pares observados",
    "o falso vínculo ocorreu",
    "os falsos vínculos ocorreram",
    "a precisão do modelo",
    "a floresta aleatória treinada",
    "a Floresta Aleatória treinada",
    "o padrão-ouro adotado",
    "o par duvidoso revisado",
    "os pares duvidosos revisados",
    "o par incerto revisado",
    "os pares incertos revisados",
    "a taxa de erro do modelo",
    "a revocação do modelo",
    "a calibração sigmoidal do modelo",
    "a calibração de Platt do modelo",
    "o identificador único do registro",
    "os identificadores únicos dos registros",
    "o arcabouço metodológico",
    "o motor de perda do sistema",
    "a perda esperada do sistema",
]


@pytest.mark.parametrize("texto", TERMOS_R02_COM_ACENTO)
def test_r02_termos_com_acento(tmp_path, texto):
    assert n_regra(tmp_path, texto + ".", "R02") == 1


@pytest.mark.parametrize("texto", TERMOS_R02_COM_ACENTO)
def test_r02_termos_com_acento_em_macros_latex(tmp_path, texto):
    latex = (
        texto.replace("á", r"\'a").replace("ã", r"\~a").replace("ç", r"\c{c}")
        .replace("í", r"\'{\i}").replace("é", r"\'e").replace("ú", r"\'u")
        .replace("õ", r"\~o").replace("ó", r"\'o").replace("ê", r"\^e")
    )
    assert n_regra(tmp_path, latex + ".", "R02") == 1


def test_r02_severidades(tmp_path):
    avisos = lint(tmp_path, "Os pares duvidosos e o padrão-ouro e a taxa de erro.")
    assert {v.severidade for v in da_regra(avisos, "R02")} == {"aviso"}
    erros = lint(tmp_path, "O gold standard foi usado.")
    assert {v.severidade for v in da_regra(erros, "R02")} == {"erro"}


def test_r02_converte_acentos_latex(tmp_path):
    viols = lint(tmp_path, r"A zona cinzenta e a vincula\c{c}\~{a}o de bases.")
    assert n_regra(tmp_path, r"A zona cinzenta e a vincula\c{c}\~{a}o de bases.", "R02") == 2
    assert viols


# ---------------------------------------------------------------------------
# R03 glosa de primeira menção por capítulo
# ---------------------------------------------------------------------------
def cap(titulo: str, corpo: str) -> str:
    return "\\begin{chapter}{" + titulo + "}\n" + corpo + "\n\\end{chapter}\n"


def test_r03_linkage_sem_glosa(tmp_path):
    viols = da_regra(lint(tmp_path, cap("Um", r"O \textit{linkage} integra bases.")), "R03")
    assert len(viols) == 1
    assert viols[0].linha == 2
    assert viols[0].arquivo == "x.tex"


def test_r03_linkage_com_glosa(tmp_path):
    corpo = cap("Um", r"O \textit{linkage} (relacionamento de registros) integra bases.")
    assert n_regra(tmp_path, corpo, "R03") == 0


def test_r03_so_a_primeira_mencao_exige_glosa(tmp_path):
    corpo = cap(
        "Um",
        r"O \textit{linkage} (relacionamento de registros) integra bases."
        "\n\n" + r"O \textit{linkage} probabilístico decide pares.",
    )
    assert n_regra(tmp_path, corpo, "R03") == 0


def test_r03_cada_capitulo_recomeca(tmp_path):
    glosado = r"O \textit{linkage} (relacionamento de registros) integra bases."
    corpo = cap("Um", glosado) + cap("Dois", r"O \textit{linkage} decide pares.") + "\\chapter{Tres}\n" + glosado + "\n"
    viols = da_regra(lint(tmp_path, corpo), "R03")
    assert len(viols) == 1
    assert "Dois" in viols[0].mensagem


def test_r03_titulo_nao_conta_como_primeira_mencao(tmp_path):
    corpo = (
        r"\chapter{Sobre o \textit{linkage}}"
        "\n\n" + r"O \textit{linkage} (relacionamento de registros) integra bases." + "\n"
    )
    assert n_regra(tmp_path, corpo, "R03") == 0


def test_r03_antes_do_primeiro_capitulo_nao_conta(tmp_path):
    corpo = r"O \textit{linkage} abre o resumo." + "\n" + cap("Um", r"O \textit{linkage} (relacionamento de registros) integra.")
    assert n_regra(tmp_path, corpo, "R03") == 0


def test_r03_grey_area(tmp_path):
    sem = cap("Um", r"A \textit{grey area} concentra pares.")
    com = cap("Um", r"A \textit{grey area} (área cinza) concentra pares.")
    depois = cap("Um", r"A \textit{grey area} (área cinza) concentra pares. A \textit{grey area} cresce.")
    assert n_regra(tmp_path, sem, "R03") == 1
    assert n_regra(tmp_path, com, "R03") == 0
    assert n_regra(tmp_path, depois, "R03") == 0


def test_r03_random_forest_uma_vez_na_tese(tmp_path):
    sem = cap("Um", "Usou-se Random Forest no estudo.") + cap("Dois", "Usou-se Random Forest de novo.")
    assert n_regra(tmp_path, sem, "R03") == 1
    com = cap("Um", "Usou-se Random Forest (floresta aleatória) no estudo.") + cap("Dois", "Usou-se Random Forest de novo.")
    assert n_regra(tmp_path, com, "R03") == 0
    italico = cap("Um", r"Usou-se \textit{Random Forest} (floresta aleatória) no estudo.")
    assert n_regra(tmp_path, italico, "R03") == 0


def test_r03_inclusao_continua_o_mesmo_capitulo(tmp_path):
    raiz = (
        "\\begin{chapter}{A}\n"
        "\\input{sub}\n"
        r"O \textit{linkage} (relacionamento de registros) vem depois." "\n"
        "\\end{chapter}\n"
    )
    viols = da_regra(lint(tmp_path, raiz, extras={"sub.tex": r"O \textit{linkage} aparece antes." + "\n"}), "R03")
    assert [(v.arquivo, v.linha) for v in viols] == [("sub.tex", 1)]


def test_r03_nao_aplica_a_definicoes(tmp_path):
    raiz = "\\begin{chapter}{A}\n\\input{siglas.tex}\n\\end{chapter}\n"
    viols = lint(tmp_path, raiz, extras={"siglas.tex": r"X & \textit{linkage} e zona cinzenta \\" + "\n"})
    assert da_regra(viols, "R03") == []
    assert da_regra(viols, "R02") == []


# ---------------------------------------------------------------------------
# R04 linkage fora de itálico
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("O linkage probabilístico decide pares.", 1),
        ("O Linkage decide pares.", 1),
        (r"O \textit{linkage} decide pares.", 0),
        (r"O \emph{linkage} decide pares.", 0),
        (r"O {\itshape linkage} decide pares.", 0),
        ("Segundo o autor, ``o linkage decide'' pares.", 0),
        ("O Laboratório de Linkage e Análise de Dados Populacionais apoiou.", 0),
        ("O pacote gzcmd-record-linkage apoiou.", 0),
        (r"O \path{x/linkage} apoiou.", 0),
        ("Os linkages probabilísticos decidem pares.", 1),
        ("Dois Linkages foram feitos.", 1),
        (r"Os \textit{linkages} decidem pares.", 0),
        (r"Os \emph{linkages} decidem pares.", 0),
    ],
)
def test_r04(tmp_path, texto, esperado):
    assert n_regra(tmp_path, texto, "R04") == esperado


# ---------------------------------------------------------------------------
# R05 primeira pessoa
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("Fizemos o ajuste dos modelos.", 1),
        ("Nossa análise cobre os pares.", 1),
        ("Eu proponho um método.", 1),
        ("Propomos um método novo.", 1),
        ("Os nós críticos da linha de cuidado.", 0),
        ("Segundo o autor, ``nós fizemos'' o ajuste.", 0),
        ("Foi realizado o ajuste dos modelos.", 0),
        ("O ajuste foi realizado pelo autor.", 0),
    ],
)
def test_r05(tmp_path, texto, esperado):
    assert n_regra(tmp_path, texto, "R05") == esperado


# ---------------------------------------------------------------------------
# R06 ponto decimal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "texto",
    [
        "O valor foi 0.95 no total.",
        "A média de 3.5 casos.",
        r"Resultado de 12.5\% do total.",
        "A sensibilidade foi 0.941 no teste.",  # zero, ponto e três dígitos nunca é milhar
        "O escore 0.500 separa as classes.",
        "O escore (0.941) separa as classes.",
    ],
)
def test_r06_positivos(tmp_path, texto):
    assert n_regra(tmp_path, texto, "R06") == 1


@pytest.mark.parametrize(
    "texto",
    [
        "Foram 1.000 casos.",
        "Foram 84.368 pares.",
        "Foram 10.941 pares e 1.941 casos.",  # só "0.xxx" é decimal; "10.941" é milhar
        "Usou-se a versão v0.4.1 do pacote.",
        "Na versão 0.4 do pacote.",
        "Usou-se Kimi K2.5 no teste.",
        "Usou-se Claude Haiku 5.5 no teste.",
        "Usou-se DeepSeek V4.1 Flash no teste.",
        "Usou-se gpt-4.1 no teste.",
        "Ver Seção 3.2 do texto.",
        "O valor foi 0,95.",
        "O valor é $0.95$ no modelo.",
        "DOI 10.1590/abc no texto.",
        r"Ver \url{http://x.org/0.95} no texto.",
        r"Ver \path{data/v0.95.csv} no texto.",
    ],
)
def test_r06_negativos(tmp_path, texto):
    assert n_regra(tmp_path, texto, "R06") == 0


def test_r06_em_tabela_marca_o_tipo(tmp_path):
    corpo = "\\begin{tabular}{ll}\nA & 0.5 \\\\\n\\end{tabular}\n"
    viols = da_regra(lint(tmp_path, corpo), "R06")
    assert [v.tipo for v in viols] == ["tabela"]


# ---------------------------------------------------------------------------
# R07 vícios de LLM (aviso)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "texto",
    [
        "Além disso, o método funciona.",
        "Vale notar que o método funciona.",
        "Em resumo, o método funciona.",
        "Cabe ressaltar que o método funciona.",
        "Estudos mostram que o método funciona.",
        "Vamos analisar os resultados.",
        "Um método robusto foi usado.",
        "Uma abordagem inovadora foi usada.",
        "O método funciona!",
    ],
)
def test_r07_positivos(tmp_path, texto):
    viols = da_regra(lint(tmp_path, texto), "R07")
    assert len(viols) == 1
    assert viols[0].severidade == "aviso"


@pytest.mark.parametrize(
    "texto",
    [
        "Foram avaliados cinco classificadores.",
        r"Estudos mostram que o método funciona \cite{A2020}.",
    ],
)
def test_r07_negativos(tmp_path, texto):
    assert n_regra(tmp_path, texto, "R07") == 0


# ---------------------------------------------------------------------------
# R08 siglas oficiais
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("O Sistema de Informação sobre Mortalidade (SIM) registra.", 1),
        ("O Sistema de Informações sobre Mortalidade (SIM) registra.", 0),
        ("O Sistema de Informação de Agravos de Notificação (Sinan) registra.", 0),
        ("O Sistema de Informações de Agravos de Notificação (Sinan) registra.", 1),
        ("O Sistema de Informação de Agravos Notificáveis (Sinan) registra.", 1),
        ("O Sistema de Informações sobre Nascidos Vivos (Sinasc) registra.", 0),
        ("O Sistema de Informação sobre Nascidos Vivos (Sinasc) registra.", 1),
        ("O Sistema de Informações Hospitalares do SUS (SIH/SUS) registra.", 0),
        ("O Sistema de Informação Hospitalar do SUS (SIH/SUS) registra.", 1),
        ("O Sistema Único de Saúde (SUS) cobre.", 0),
        ("O sistema único de saúde (SUS) cobre.", 1),
    ],
)
def test_r08(tmp_path, texto, esperado):
    assert n_regra(tmp_path, texto, "R08") == esperado


def test_r08_e_r01_aplicam_a_definicoes_mas_r02_nao(tmp_path):
    raiz = "\\input{siglas.tex}\n"
    siglas = (
        "\\begin{longtable}{ll}\n"
        "SIM & Sistema de Informação sobre Mortalidade \\\\\n"
        "SINAN & travessão \u2014 aqui \\\\\n"
        "\\end{longtable}\n"
    )
    viols = lint(tmp_path, raiz, extras={"siglas.tex": siglas})
    assert [(v.regra, v.arquivo, v.tipo) for v in viols] == [
        ("R01", "siglas.tex", "tabela"),
        ("R08", "siglas.tex", "tabela"),
    ]


# ---------------------------------------------------------------------------
# R09 e R10 parágrafos
# ---------------------------------------------------------------------------
def test_r09_limite_de_200_palavras(tmp_path):
    assert n_regra(tmp_path, ("palavra " * 201).strip(), "R09") == 1
    assert n_regra(tmp_path, ("palavra " * 200).strip(), "R09") == 0


def test_r09_conta_o_paragrafo_em_varias_linhas(tmp_path):
    linhas = "\n".join(("palavra " * 50).strip() for _ in range(5))
    viols = da_regra(lint(tmp_path, linhas), "R09")
    assert len(viols) == 1
    assert viols[0].severidade == "aviso"
    assert "250" in viols[0].mensagem


def test_r09_separacoes(tmp_path):
    dois = ("palavra " * 120).strip() + "\n\n" + ("palavra " * 120).strip()
    assert n_regra(tmp_path, dois, "R09") == 0
    itens = "\\begin{itemize}\n\\item " + ("palavra " * 120).strip() + "\n\\item " + ("palavra " * 120).strip() + "\n\\end{itemize}\n"
    assert n_regra(tmp_path, itens, "R09") == 0


def test_r09_ignora_tabela_e_citacao(tmp_path):
    tabela = "\\begin{tabular}{l}\n" + ("palavra " * 250) + "\\\\\n\\end{tabular}\n"
    assert n_regra(tmp_path, tabela, "R09") == 0
    assert n_regra(tmp_path, ("palavra " * 200).strip() + " \\cite{a,b,c,d}", "R09") == 0


def test_r10_path_em_prosa_corrida(tmp_path):
    corpo = r"Os dados estão em \path{data/x.csv} e foram usados nos estudos de caso do capítulo."
    viols = da_regra(lint(tmp_path, corpo), "R10")
    assert len(viols) == 1
    assert viols[0].severidade == "aviso"


def test_r10_nao_conta_paragrafo_curto_tabela_nem_legenda(tmp_path):
    assert n_regra(tmp_path, r"Ver \path{data/x.csv}.", "R10") == 0
    tabela = "\\begin{tabular}{l}\n\\path{data/x.csv} e outros dados de apoio do estudo em questão aqui\\\\\n\\end{tabular}\n"
    assert n_regra(tmp_path, tabela, "R10") == 0
    legenda = "\\caption{Dados em \\path{data/x.csv} usados nos estudos de caso do capítulo de resultados.}\n"
    assert n_regra(tmp_path, legenda, "R10") == 0


# ---------------------------------------------------------------------------
# Leitura: remoção, tipos, inclusões
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "texto",
    [
        "\\begin{verbatim}\nzona cinzenta e 0.95 e Nós fizemos\n\\end{verbatim}\n",
        "\\begin{lstlisting}\nzona cinzenta\n\\end{lstlisting}\n",
        "% zona cinzenta em comentário\nTexto limpo.\n",
        "Fórmula $zona cinzenta = 0.95$ no texto.\n",
        "\\begin{equation}\nzona cinzenta = 0.95\n\\end{equation}\n",
        "Fórmula \\(zona cinzenta\\) no texto.\n",
        r"Ver \cite{zona cinzenta} e \ref{zona cinzenta} e \label{zona cinzenta}." "\n",
        r"Ver \path{zona cinzenta} e \url{zona cinzenta} e \texttt{zona cinzenta}." "\n",
        r"Ver \href{http://x.org/zona cinzenta}{texto limpo}." "\n",
        r"Ver \verb|zona cinzenta| no código." "\n",
    ],
)
def test_neutralizacao_remove_o_que_nao_e_prosa(tmp_path, texto):
    assert [v for v in lint(tmp_path, texto) if v.regra in {"R01", "R02", "R04", "R05", "R06"}] == []


def test_href_mantem_o_texto_visivel(tmp_path):
    assert n_regra(tmp_path, r"Ver \href{http://x.org}{a zona cinzenta}.", "R02") == 1


def test_tipos_de_trecho(tmp_path):
    corpo = (
        "A zona cinzenta em prosa.\n\n"
        "\\begin{table}\n\\caption{A zona cinzenta na legenda.}\n"
        "\\begin{tabular}{l}\nzona cinzenta \\\\\n\\end{tabular}\n"
        "\\begin{tablenotes}\n\\item A zona cinzenta na nota.\n\\end{tablenotes}\n"
        "\\end{table}\n\n"
        "Texto\\footnote{A zona cinzenta no rodapé.} segue.\n\n"
        "\\section{A zona cinzenta no título}\n"
    )
    tipos = [v.tipo for v in da_regra(lint(tmp_path, corpo), "R02")]
    assert tipos == ["prosa", "legenda", "tabela", "nota", "nota", "titulo"]


def test_preambulo_e_abstract_ficam_de_fora(tmp_path):
    corpo = (
        "\\newcommand{\\x}{zona cinzenta}\n% zona cinzenta\n"
        "\\begin{document}\n"
        "{\\LARGE\\bf Abstract}\\\\\nLinkage integrates the zona cinzenta.\n"
        "\\textbf{Keywords:} Linkage.\n\n"
        "A zona cinzenta em prosa.\n"
        "\\end{document}\n"
    )
    viols = lint(tmp_path, corpo)
    assert [(v.regra, v.linha) for v in viols] == [("R02", 8)]


def test_arvore_segue_inclusoes_e_ignora_rev_diff(tmp_path):
    raiz = "\\input{a}\n\\input{sub/b.tex}\n\\input{tese_rev_diff.tex}\n\\input{nao_existe}\n"
    extras = {
        "a.tex": "A zona cinzenta em a.\n",
        "sub/b.tex": "A zona cinzenta em b.\n",
        "tese_rev_diff.tex": "A zona cinzenta no diff.\n",
    }
    arvore = lp.carregar_arvore(_escrever(tmp_path, raiz, extras), base=tmp_path)
    viols = lp.lintar(arvore)
    assert sorted((v.arquivo, v.linha) for v in viols) == [("a.tex", 1), ("sub/b.tex", 1)]
    assert any("nao_existe" in a for a in arvore.avisos)
    assert not any(k.endswith("tese_rev_diff.tex") for k in arvore.arquivos)


def _escrever(tmp_path: Path, corpo: str, extras: dict[str, str]) -> Path:
    raiz = tmp_path / "x.tex"
    raiz.write_text(corpo, encoding="utf-8")
    for rel, conteudo in extras.items():
        destino = tmp_path / rel
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(conteudo, encoding="utf-8")
    return raiz


def test_trecho_tem_no_maximo_160_caracteres(tmp_path):
    corpo = ("palavra " * 60) + "zona cinzenta " + ("palavra " * 60) + "\n"
    viols = da_regra(lint(tmp_path, corpo), "R02")
    assert len(viols) == 1
    assert len(viols[0].trecho) <= 160
    assert "zona cinzenta" in viols[0].trecho


# ---------------------------------------------------------------------------
# Baseline que só encolhe
# ---------------------------------------------------------------------------
def test_fingerprint_independe_da_linha_e_distingue_trechos(tmp_path):
    a = lint(tmp_path, "A zona cinzenta é ampla.\n")
    b = lint(tmp_path, "\n\n\nA zona cinzenta é ampla.\n")  # mesma violação, três linhas abaixo
    assert (a[0].linha, b[0].linha) == (1, 4)
    assert a[0].fp == b[0].fp and len(a[0].fp) == 12
    c = lint(tmp_path, "A zona cinza é ampla.\n")  # outro trecho
    assert c[0].fp != a[0].fp
    assert lp.fingerprint("R02", "a.tex", "x") != lp.fingerprint("R02", "b.tex", "x")
    assert lp.fingerprint("R02", "a.tex", "x") != lp.fingerprint("R05", "a.tex", "x")


def test_fingerprint_normaliza_espacos_e_caixa(tmp_path):
    a = lint(tmp_path, "A zona   cinzenta\né ampla.\n")
    b = lint(tmp_path, "A zona cinzenta é ampla.\n")
    # a diferença está dentro do trecho casado ("zona   cinzenta\n" vs "zona cinzenta"):
    # o contexto colapsa espaços e quebras, logo a impressão digital é a mesma
    assert a[0].fp == b[0].fp


def test_montar_baseline_v2_so_com_erros(tmp_path):
    viols = lint(tmp_path, "A zona cinzenta \u2014 e o padrão-ouro.\n")  # R01 erro, R02 erro, R02 aviso
    assert {v.severidade for v in viols} == {"erro", "aviso"}
    base = lp.montar_baseline(viols)
    assert base["versao"] == 2
    assert set(base["violacoes"]) == {v.fp for v in viols if v.severidade == "erro"}
    entrada = next(iter(base["violacoes"].values()))
    assert set(entrada) == {"regra", "arquivo", "trecho"}
    assert entrada["arquivo"] == "x.tex"


def test_trocar_uma_violacao_por_outra_e_detectado(tmp_path):
    """Com contagens por (regra, arquivo) a troca passaria; com impressão digital, não."""
    base = lp.montar_baseline(lint(tmp_path, "A zona cinzenta é ampla.\n"))
    trocada = lint(tmp_path, "A zona cinza é ampla.\n")  # mesma regra, mesmo arquivo, mesma contagem
    assert len(trocada) == 1
    novas = lp.novas_violacoes(trocada, base)
    assert [v.fp for v in novas] == [trocada[0].fp]


def test_avisos_nunca_sao_novas_violacoes(tmp_path):
    viols = lint(tmp_path, "O padrão-ouro foi usado. Um método robusto!\n")
    assert viols and {v.severidade for v in viols} == {"aviso"}
    assert lp.novas_violacoes(viols, {"versao": 2, "violacoes": {}}) == []


SEPARADOR = "\n" * 60  # mais de 40 caracteres de espaço: uma violação não entra no contexto da outra


def test_contextos_identicos_no_mesmo_arquivo_tem_fingerprints_distintos(tmp_path):
    corpo = "A zona cinzenta é ampla." + SEPARADOR + "A zona cinzenta é ampla.\n"
    viols = lint(tmp_path, corpo)
    assert len(viols) == 2
    assert viols[0].contexto == viols[1].contexto  # mesmo contexto normalizado
    assert viols[0].fp != viols[1].fp
    base = lp.montar_baseline(viols)
    assert len(base["violacoes"]) == 2  # nenhuma colapsa
    assert lp.novas_violacoes(viols, base) == []
    # uma terceira repetição idêntica é violação nova
    tres = lint(tmp_path, corpo + SEPARADOR + "A zona cinzenta é ampla.\n")
    assert [v.fp for v in lp.novas_violacoes(tres, base)] == [tres[2].fp]
    # resolver a primeira repetição mantém o gate verde (o índice 1 continua no baseline)
    uma = lint(tmp_path, "A zona cinzenta é ampla.\n")
    assert lp.novas_violacoes(uma, base) == []
    novo, novas = lp.atualizar_baseline(uma, base)
    assert novas == [] and len(novo["violacoes"]) == 1


def test_fingerprint_ocorrencia_muda_o_hash():
    assert lp.fingerprint("R02", "a.tex", "x") == lp.fingerprint("R02", "a.tex", "x", 1)
    assert lp.fingerprint("R02", "a.tex", "x", 1) != lp.fingerprint("R02", "a.tex", "x", 2)


def test_novas_violacoes_ignora_mudanca_de_linha(tmp_path):
    base = lp.montar_baseline(lint(tmp_path, "A zona cinzenta é ampla.\n"))
    # o texto ao redor (±40 caracteres) não muda; só a linha
    movida = lint(tmp_path, "Linha nova." + SEPARADOR + "A zona cinzenta é ampla.\n")
    assert movida[0].linha == 61
    assert lp.novas_violacoes(movida, base) == []


def test_atualizar_baseline_remove_resolvidas_e_nao_acrescenta(tmp_path):
    duas = lint(tmp_path, "A zona cinzenta é ampla." + SEPARADOR + "A vinculação das bases.\n")
    base = lp.montar_baseline(duas)
    assert len(base["violacoes"]) == 2
    # uma resolvida: encolhe
    uma = lint(tmp_path, "A zona cinzenta é ampla.\n")
    novo, novas = lp.atualizar_baseline(uma, base)
    assert novas == []
    assert set(novo["violacoes"]) == {uma[0].fp}
    assert novo["versao"] == 2
    # uma resolvida e uma nova ao mesmo tempo: recusa (não troca)
    trocada = lint(tmp_path, "A zona cinzenta é ampla." + SEPARADOR + "O falso vínculo ocorreu.\n")
    novo, novas = lp.atualizar_baseline(trocada, base)
    assert novo is None
    assert [v.mensagem.startswith("«falso vínculo»") for v in novas] == [True]


def test_afrouxamentos_lista_so_o_que_o_base_nao_tinha():
    base = {"versao": 2, "violacoes": {"a": {}, "b": {}}}
    assert lp.afrouxamentos({"versao": 2, "violacoes": {"a": {}}}, base) == []
    assert lp.afrouxamentos({"versao": 2, "violacoes": {"a": {}, "b": {}}}, base) == []
    assert lp.afrouxamentos({"versao": 2, "violacoes": {"a": {}, "c": {}, "d": {}}}, base) == ["c", "d"]


def test_ler_baseline_recusa_versao_antiga(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"versao": 1, "contagens": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="versão 2"):
        lp.ler_baseline(p)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        cwd=repo, check=True, capture_output=True,
    )


@pytest.fixture
def repo_git(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git indisponível")
    repo = tmp_path / "repo"
    (repo / "tests" / "data").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/pq/revisao-2-redacao")
    return repo


def test_baseline_da_base_le_o_merge_base(repo_git):
    rel = lp.BASELINE_RELATIVO
    base = {"versao": 2, "violacoes": {"a": {"regra": "R01", "arquivo": "x", "trecho": "t"}}}
    (repo_git / rel).write_text(json.dumps(base), encoding="utf-8")
    _git(repo_git, "add", "-A")
    _git(repo_git, "commit", "-q", "-m", "base")
    _git(repo_git, "checkout", "-q", "-b", "feature")
    # o ramo de trabalho afrouxa o baseline (entrada nova) sem commitar: o merge-base não muda
    atual = {"versao": 2, "violacoes": {**base["violacoes"], "z": {"regra": "R02", "arquivo": "x", "trecho": "u"}}}
    (repo_git / rel).write_text(json.dumps(atual), encoding="utf-8")
    lido, motivo = lp.baseline_da_base(repo_git)
    assert lido == base, motivo
    assert lp.afrouxamentos(atual, lido) == ["z"]
    assert lp.afrouxamentos(base, lido) == []


def test_baseline_da_base_pula_sem_v2_ou_sem_ramo(repo_git):
    rel = lp.BASELINE_RELATIVO
    (repo_git / rel).write_text(json.dumps({"versao": 1, "contagens": {}}), encoding="utf-8")
    _git(repo_git, "add", "-A")
    _git(repo_git, "commit", "-q", "-m", "v1")
    lido, motivo = lp.baseline_da_base(repo_git)
    assert lido is None and "versão 2" in motivo
    lido, motivo = lp.baseline_da_base(repo_git, refs=("ramo/inexistente",))
    assert lido is None and "merge-base" in motivo


def test_baseline_da_base_fora_de_repositorio_git(tmp_path):
    vazio = tmp_path / "sem_git"
    vazio.mkdir()
    lido, motivo = lp.baseline_da_base(vazio)
    # dentro de tmp_path não há repositório git (a menos que o ambiente de teste o crie acima)
    if lido is not None or "git" not in motivo:
        pytest.skip("o diretório temporário está dentro de um repositório git")
    assert lido is None


def _cli(tmp_path: Path, *extra: str) -> int:
    return lp.main(
        [
            "--tex", str(tmp_path / "x.tex"),
            "--base", str(tmp_path),
            "--baseline", str(tmp_path / "out" / "baseline.json"),
            "--relatorio", str(tmp_path / "out" / "relatorio.md"),
            *extra,
        ]
    )


def test_cli_gera_baseline_e_relatorio(tmp_path):
    (tmp_path / "x.tex").write_text("A zona cinzenta \u2014 aqui.\n\nOutra vinculação.\n", encoding="utf-8")
    assert _cli(tmp_path, "--gerar-baseline", "--forcar") == 0
    baseline = json.loads((tmp_path / "out" / "baseline.json").read_text(encoding="utf-8"))
    assert baseline["versao"] == 2
    assert baseline["gerado_em"]
    assert sorted(e["regra"] for e in baseline["violacoes"].values()) == ["R01", "R02", "R02"]
    assert all(e["arquivo"] == "x.tex" and e["trecho"] for e in baseline["violacoes"].values())
    relatorio = (tmp_path / "out" / "relatorio.md").read_text(encoding="utf-8")
    assert "`x.tex:1` [prosa, erro]" in relatorio
    assert "`x.tex:3` [prosa, erro]" in relatorio
    assert "| 2 | 0 | 2 | 2 | 0 |" in relatorio  # resumo da R02: 2 erros, 0 avisos, baseline 2, 0 novos
    assert "| R02 | `x.tex` | 2 | 0 | 2 |" in relatorio


def test_cli_gerar_baseline_exige_forcar_e_avisa(tmp_path, capsys):
    (tmp_path / "x.tex").write_text("A zona cinzenta aqui.\n", encoding="utf-8")
    assert _cli(tmp_path, "--gerar-baseline") == 2
    assert not (tmp_path / "out" / "baseline.json").exists()
    assert "--forcar" in capsys.readouterr().err
    assert _cli(tmp_path, "--gerar-baseline", "--forcar") == 0
    assert "AVISO" in capsys.readouterr().err
    assert (tmp_path / "out" / "baseline.json").is_file()


def test_cli_confere_o_baseline_e_so_deixa_encolher(tmp_path):
    raiz = tmp_path / "x.tex"
    raiz.write_text("A zona cinzenta \u2014 aqui.\n", encoding="utf-8")
    assert _cli(tmp_path, "--gerar-baseline", "--forcar") == 0
    assert _cli(tmp_path) == 0

    # erro novo, afastado das violações já registradas
    raiz.write_text("A zona cinzenta \u2014 aqui." + SEPARADOR + "Outra vinculação.\n", encoding="utf-8")
    assert _cli(tmp_path) == 1
    relatorio = (tmp_path / "out" / "relatorio.md").read_text(encoding="utf-8")
    assert "## Erros novos (fora do baseline)" in relatorio
    assert "**novo**" in relatorio
    antes = (tmp_path / "out" / "baseline.json").read_text(encoding="utf-8")
    assert _cli(tmp_path, "--atualizar-baseline") == 1
    assert (tmp_path / "out" / "baseline.json").read_text(encoding="utf-8") == antes

    # trocar uma violação por outra também falha (a contagem seria a mesma)
    raiz.write_text("A zona cinzenta aqui." + SEPARADOR + "Outra vinculação.\n", encoding="utf-8")
    assert _cli(tmp_path) == 1

    # voltar ao texto original passa; resolver tudo e --atualizar-baseline remove as entradas
    raiz.write_text("A zona cinzenta \u2014 aqui.\n", encoding="utf-8")
    assert _cli(tmp_path) == 0
    raiz.write_text("Texto limpo.\n", encoding="utf-8")
    assert _cli(tmp_path, "--atualizar-baseline") == 0
    baseline = json.loads((tmp_path / "out" / "baseline.json").read_text(encoding="utf-8"))
    assert baseline["violacoes"] == {}
    assert baseline["versao"] == 2
    assert _cli(tmp_path) == 0


def test_cli_aviso_nao_falha_o_gate_mas_aparece_no_relatorio(tmp_path, capsys):
    raiz = tmp_path / "x.tex"
    raiz.write_text("Texto limpo.\n", encoding="utf-8")
    assert _cli(tmp_path, "--gerar-baseline", "--forcar") == 0
    raiz.write_text("O padrão-ouro foi usado. Um método robusto!\n", encoding="utf-8")
    capsys.readouterr()
    assert _cli(tmp_path) == 0
    assert "avisos informativos" in capsys.readouterr().out
    relatorio = (tmp_path / "out" / "relatorio.md").read_text(encoding="utf-8")
    assert "[prosa, aviso]" in relatorio


def test_cli_sem_baseline_falha(tmp_path):
    (tmp_path / "x.tex").write_text("Texto limpo.\n", encoding="utf-8")
    assert _cli(tmp_path) == 2


def test_cli_baseline_v1_e_recusado(tmp_path):
    (tmp_path / "x.tex").write_text("Texto limpo.\n", encoding="utf-8")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "baseline.json").write_text(
        json.dumps({"versao": 1, "contagens": {}}), encoding="utf-8"
    )
    assert _cli(tmp_path) == 2
    assert _cli(tmp_path, "--atualizar-baseline") == 2


# ---------------------------------------------------------------------------
# Gate da tese real
# ---------------------------------------------------------------------------
def test_tese_real_nao_piora_o_baseline():
    if not BASELINE_REAL.is_file():
        pytest.skip(
            f"baseline ausente ({BASELINE_REAL}); gere com: "
            "python scripts/linter_prosa.py --gerar-baseline --forcar"
        )
    arvore = lp.carregar_arvore(lp.TESE_PADRAO, base=lp.REPO)
    assert "text/latex/capitulo1.tex" in {a.rel for a in arvore.arquivos.values()}
    assert not any(a.rel.endswith("tese_rev_diff.tex") for a in arvore.arquivos.values())
    violacoes = lp.lintar(arvore)
    baseline = lp.ler_baseline(BASELINE_REAL)  # ValueError se não for a versão 2
    assert baseline["versao"] == 2
    novas = lp.novas_violacoes(violacoes, baseline)  # avisos nunca entram
    detalhe = [f"{v.regra} {v.arquivo}:{v.linha} {v.mensagem}: {v.trecho}" for v in novas[:20]]
    assert not novas, f"{len(novas)} erro(s) novo(s) fora do baseline:\n" + "\n".join(detalhe)


def test_baseline_nao_afrouxou():
    """O baseline só pode encolher frente ao do merge-base com ``pq/revisao-2-redacao``."""
    if not BASELINE_REAL.is_file():
        pytest.skip(f"baseline ausente ({BASELINE_REAL})")
    base, descricao = lp.baseline_da_base()
    if base is None:
        pytest.skip(f"sem baseline v2 de referência no git: {descricao}")
    atual = lp.ler_baseline(BASELINE_REAL)
    novas = lp.afrouxamentos(atual, base)
    detalhe = [
        f"{fp}: {atual['violacoes'][fp].get('regra')} {atual['violacoes'][fp].get('arquivo')}: "
        f"{atual['violacoes'][fp].get('trecho')}"
        for fp in novas[:20]
    ]
    assert not novas, (
        f"o baseline ganhou {len(novas)} entrada(s) ausente(s) do {descricao}; "
        "corrija a violação em vez de registrá-la:\n" + "\n".join(detalhe)
    )
