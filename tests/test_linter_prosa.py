"""Testes do linter de prosa (``scripts/linter_prosa.py``).

Casos sintéticos em ``tmp_path`` para cada regra e o gate da tese real contra
``tests/data/linter_prosa_baseline.json`` (gerado por
``python scripts/linter_prosa.py --gerar-baseline``).
"""

from __future__ import annotations

import json
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
    ],
)
def test_r06_positivos(tmp_path, texto):
    assert n_regra(tmp_path, texto, "R06") == 1


@pytest.mark.parametrize(
    "texto",
    [
        "Foram 1.000 casos.",
        "Foram 84.368 pares.",
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
def test_contar_agrupa_por_regra_e_arquivo(tmp_path):
    viols = lint(tmp_path, "A zona cinzenta \u2014 e a vinculação.\n")
    assert lp.contar(viols) == {"R01": {"x.tex": 1}, "R02": {"x.tex": 2}}


def test_comparar_detecta_excesso_e_violacao_nova():
    base = {"R01": {"a.tex": 2}}
    assert lp.comparar({"R01": {"a.tex": 2}}, base) == []
    assert lp.comparar({"R01": {"a.tex": 1}}, base) == []
    assert lp.comparar({"R01": {"a.tex": 3}}, base) == [("R01", "a.tex", 3, 2)]
    assert lp.comparar({"R01": {"b.tex": 1}}, base) == [("R01", "b.tex", 1, 0)]
    assert lp.comparar({"R02": {"a.tex": 1}}, base) == [("R02", "a.tex", 1, 0)]


def test_atualizar_baseline_so_diminui():
    base = {"R01": {"a.tex": 3, "b.tex": 1}, "R02": {"a.tex": 2}}
    novo, aumentos = lp.atualizar_baseline({"R01": {"a.tex": 2}, "R02": {"a.tex": 2}}, base)
    assert aumentos == []
    assert novo == {"R01": {"a.tex": 2}, "R02": {"a.tex": 2}}


def test_atualizar_baseline_recusa_aumento():
    base = {"R01": {"a.tex": 3}}
    novo, aumentos = lp.atualizar_baseline({"R01": {"a.tex": 4}, "R02": {"b.tex": 1}}, base)
    assert novo is None
    assert aumentos == [("R01", "a.tex", 4, 3), ("R02", "b.tex", 1, 0)]


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
    assert _cli(tmp_path, "--gerar-baseline") == 0
    baseline = json.loads((tmp_path / "out" / "baseline.json").read_text(encoding="utf-8"))
    assert baseline["versao"] == 1
    assert baseline["gerado_em"]
    assert baseline["contagens"] == {"R01": {"x.tex": 1}, "R02": {"x.tex": 2}}
    relatorio = (tmp_path / "out" / "relatorio.md").read_text(encoding="utf-8")
    assert "`x.tex:1` [prosa, erro]" in relatorio
    assert "`x.tex:3` [prosa, erro]" in relatorio
    assert "| R02 | `x.tex` | 2 | 2 |" in relatorio


def test_cli_confere_o_baseline_e_so_deixa_encolher(tmp_path):
    raiz = tmp_path / "x.tex"
    raiz.write_text("A zona cinzenta \u2014 aqui.\n", encoding="utf-8")
    assert _cli(tmp_path, "--gerar-baseline") == 0
    assert _cli(tmp_path) == 0

    raiz.write_text("A zona cinzenta \u2014 aqui.\nOutra vinculação.\n", encoding="utf-8")
    assert _cli(tmp_path) == 1
    antes = (tmp_path / "out" / "baseline.json").read_text(encoding="utf-8")
    assert _cli(tmp_path, "--atualizar-baseline") == 1
    assert (tmp_path / "out" / "baseline.json").read_text(encoding="utf-8") == antes

    raiz.write_text("Texto limpo.\n", encoding="utf-8")
    assert _cli(tmp_path, "--atualizar-baseline") == 0
    baseline = json.loads((tmp_path / "out" / "baseline.json").read_text(encoding="utf-8"))
    assert baseline["contagens"] == {}
    assert _cli(tmp_path) == 0


def test_cli_sem_baseline_falha(tmp_path):
    (tmp_path / "x.tex").write_text("Texto limpo.\n", encoding="utf-8")
    assert _cli(tmp_path) == 2


# ---------------------------------------------------------------------------
# Gate da tese real
# ---------------------------------------------------------------------------
def test_tese_real_nao_piora_o_baseline():
    if not BASELINE_REAL.is_file():
        pytest.skip(
            f"baseline ausente ({BASELINE_REAL}); gere com: python scripts/linter_prosa.py --gerar-baseline"
        )
    arvore = lp.carregar_arvore(lp.TESE_PADRAO, base=lp.REPO)
    assert "text/latex/capitulo1.tex" in {a.rel for a in arvore.arquivos.values()}
    assert not any(a.rel.endswith("tese_rev_diff.tex") for a in arvore.arquivos.values())
    violacoes = lp.lintar(arvore)
    baseline = lp.ler_baseline(BASELINE_REAL)
    assert baseline["versao"] == 1
    excessos = lp.comparar(lp.contar(violacoes), baseline["contagens"])
    detalhe = []
    for regra, arquivo, atual, limite in excessos:
        detalhe.append(f"{regra} em {arquivo}: {atual} (baseline {limite})")
        detalhe += [
            f"    {v.arquivo}:{v.linha} {v.mensagem}: {v.trecho}"
            for v in violacoes
            if v.regra == regra and v.arquivo == arquivo
        ][:5]
    assert not excessos, "o linter piorou frente ao baseline:\n" + "\n".join(detalhe)
