"""Restrições editoriais da segunda passada, sem depender de PDFs regenerados."""

from pathlib import Path
import re

import pytest


LATEX = Path(__file__).resolve().parents[1] / "text" / "latex"
PARAGRAPHS = [
    ("capitulo1.tex", "O percurso da tese abrange"),
    ("capitulo_dados.tex", "O novo pseudônimo é calculado"),
    ("capitulo_dados.tex", "A matriz restrita não foi depositada"),
    ("capitulo_dados.tex", r"O arquivo \path{qw3/qw3_analysis_frame.csv}"),
    ("capitulo_dados.tex", r"O relatório \path{k_anonymity_report.json}"),
    ("capitulo_dados.tex", r"O \texttt{pair\_id} histórico"),
    ("estudo1_comparador.tex", "Os pesos internos do comparador permanecem fixos"),
    ("estudo3_estrategias.tex", "Entre as configurações originais da Tabela"),
    ("estudo3_estrategias.tex", "O custo operacional também favoreceu"),
    ("estudo3_estrategias.tex", "A organização reprodutível das configurações"),
    ("estudo3_estrategias.tex", "As estratégias de vigilância e confirmação descritas"),
    ("estudo4_gzcmd.tex", "Na cascata v3.1 sobre a base inteira, a maior recuperação"),
    ("estudo4_gzcmd.tex", "A calibração por âncoras preservou a capacidade"),
    ("estudo5_equidade.tex", "A incompatibilidade geral entre paridade"),
    ("estudo5_equidade.tex", "A construção do padrão-ouro limita a inferência"),
    ("conclusao.tex", "O terceiro objetivo comparou estratégias operacionais"),
]
FORMULAS = (
    "fica fora do alcance",
    "permanece por",
    "alcance restrito",
    "deixa indeterminado",
)


def paragraph(filename, anchor):
    matches = [
        part.strip()
        for part in (LATEX / filename).read_text(encoding="utf-8").split("\n\n")
        if part.strip().startswith(anchor)
    ]
    assert len(matches) == 1, (filename, anchor, len(matches))
    return matches[0]


@pytest.mark.parametrize("filename,anchor", PARAGRAPHS)
def test_paragrafos_tratados_sem_tres_naos_no_mesmo_periodo(filename, anchor):
    text = paragraph(filename, anchor)
    assert not re.search("[—–]", text)
    # Não separar decimais, caminhos ou nomes de arquivo como se fossem períodos.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ])", text)
    for sentence in sentences:
        assert not re.search(r"\bnão\b.*\bnão\b.*\bnão\b", sentence, re.I), sentence


@pytest.mark.parametrize(
    "filename",
    sorted(
        {path.name for path in LATEX.glob("capitulo*.tex")}
        | {path.name for path in LATEX.glob("estudo*.tex")}
        | {"conclusao.tex"}
    ),
)
def test_formulas_no_maximo_uma_vez_por_capitulo(filename):
    text = (LATEX / filename).read_text(encoding="utf-8").lower()
    for formula in FORMULAS:
        assert text.count(formula) <= 1, (filename, formula)


def test_repeticao_da_magnitude_substituida_por_remissao_com_custos():
    text = (LATEX / "estudo4_gzcmd.tex").read_text(encoding="utf-8")
    magnitude = "de dois a quatro vínculos a mais"
    assert text.count(magnitude) == 1
    results, discussion = text.split(r"\label{sec:e4-reruns}", 1)[1].split(
        "Na cascata v3.1 sobre a base inteira, a maior recuperação", 1
    )
    assert magnitude in results
    discussion = "Na cascata v3.1 sobre a base inteira, a maior recuperação" + discussion
    discussion = discussion.split("\n\n", 1)[0]
    assert r"Seção~\ref{sec:e4-reruns}" in discussion
    assert "mais inconclusivos e encaminhamentos humanos" in discussion
    assert r"vinte vezes mais caro por \textit{token}" in discussion
    assert "não constitui resultado do arranjo" in discussion
    assert "não permite atribuir o limite da revisão apenas ao tamanho do modelo" in discussion


def test_ressalvas_de_fidelidade_permanecem_explicitas():
    calibration = paragraph("estudo4_gzcmd.tex", "A calibração por âncoras preservou a capacidade")
    assert "não demonstrou calibração absoluta na área cinza da nota" in calibration
    assert "não corrigiria o viés" in calibration
    assert "não de impossibilidade geral do método" in calibration
    equity = paragraph("estudo5_equidade.tex", "A construção do padrão-ouro limita a inferência")
    assert "a possibilidade de incorporação das evidências à referência não foi quantificada" in equity
    assert "não comprova exposição nos passos exportados" in equity
    assert "menos rejeições não demonstram melhora de equidade" in equity
    assert "A validação em outras coortes e a observação da revisão humana são necessárias" in equity
    conclusion = paragraph("conclusao.tex", "O terceiro objetivo comparou estratégias operacionais")
    assert r"não fundamentam a adoção dos \textit{pipelines} avaliados" in conclusion
    assert "Tampouco demonstram ganho epidemiológico do arranjo" in conclusion
    assert "seu efeito não se transfere automaticamente ao XGBoost" in conclusion
