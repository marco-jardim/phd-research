"""Estado das negações após o fechamento conservador, sem impor sua redução."""

from pathlib import Path
import re

import pytest

from terminologia import conta, contem, normalizar_terminologia


LATEX = Path(__file__).resolve().parents[1] / "text" / "latex"
PARAGRAPHS = [
    ("capitulo1.tex", r"O Capítulo~\ref{cap:referencial} traz o referencial teórico"),
    ("capitulo_dados.tex", "O novo pseudônimo é o HMAC-SHA256"),
    ("capitulo_dados.tex", "A matriz restrita não foi depositada"),
    # W5: a auditoria do frame de equidade saiu do capítulo e vive no apêndice do pacote.
    ("apendice_pacote.tex", r"O arquivo \path{qw3/qw3_analysis_frame.csv}"),
    ("capitulo_dados.tex", r"O relatório \path{k_anonymity_report.json}"),
    ("capitulo_dados.tex", r"O \texttt{pair\_id} histórico"),
    ("estudo1_comparador.tex", "Os pesos internos do comparador permanecem fixos"),
    # Revisão 2: o detalhe do Estudo 3 (validação cruzada, pipelines, contribuição) foi para o apêndice.
    ("apendice_estrategias.tex", "Entre as configurações originais da Tabela"),
    ("estudo3_estrategias.tex", "A razão hipotética de revisões por vínculo também favoreceu"),
    ("apendice_estrategias.tex", "A organização reprodutível das configurações"),
    ("apendice_estrategias.tex", "As estratégias de vigilância e confirmação descritas"),
    # Revisão 2: o parágrafo do Sonnet na cascata foi para o Apêndice do Estudo 4 (A.4, custos).
    ("apendice_gzcmd.tex", "Na cascata v3.1 sobre a base inteira, Claude Sonnet"),
    ("estudo4_gzcmd.tex", "A calibração por âncoras preservou a capacidade"),
    ("estudo5_equidade.tex", "A paridade simultânea de sensibilidade e VPP adotada"),
    ("estudo5_equidade.tex", "A construção da referência operacional limita a inferência"),
    # Conclusão reescrita (2026-11-08): o parágrafo dos Estudos 2 e 3 abre a resposta ao objetivo de classificadores.
    ("conclusao.tex", "A ordenação dos classificadores depende da métrica"),
    ("estudo1_comparador.tex", "As duas implementações não são equivalentes"),
]
FORMULAS = (
    "fica fora do alcance",
    "permanece por",
    "alcance restrito",
    "deixa indeterminado",
)

# Número de ocorrências dos quatro negadores por frase consecutiva.
# Retrato aprovado pelo autor, não meta de redução: valores 3 são deliberados.
EXPECTED_NEGATIONS = {
    r"O Capítulo~\ref{cap:referencial} traz o referencial teórico": [0, 0, 0],  # revisão 2: mapa dos capítulos com três frases, sem negação
    "O novo pseudônimo é o HMAC-SHA256": [0, 4, 1],
    "A matriz restrita não foi depositada": [3, 1],
    r"O arquivo \path{qw3/qw3_analysis_frame.csv}": [0, 0, 0, 1, 3],
    r"O relatório \path{k_anonymity_report.json}": [1, 1],
    r"O \texttt{pair\_id} histórico": [2, 0, 1, 3],
    "Os pesos internos do comparador permanecem fixos": [0, 1, 1, 1],  # estudo1 revisado (revisão 2)
    "Entre as configurações originais da Tabela": [0, 1, 1, 1],
    "A razão hipotética de revisões por vínculo também favoreceu": [0, 1, 1, 0, 1],  # estudo3 revisado (revisão 2): Coeli2021 (NAO_SUSTENTA) retirado, frase final com "sem medir"
    "A organização reprodutível das configurações": [1, 1, 0],
    "As estratégias de vigilância e confirmação descritas": [1, 0, 1],
    "Na cascata v3.1 sobre a base inteira, Claude Sonnet": [0, 2],
    "A calibração por âncoras preservou a capacidade": [1, 0, 1, 1, 1, 1, 2],
    "A paridade simultânea de sensibilidade e VPP adotada": [1, 1, 1, 0, 0, 1, 0],
    "A construção da referência operacional limita a inferência": [0, 1, 2, 1, 0, 0, 0],
    "A ordenação dos classificadores depende da métrica": [1, 1, 1, 0, 0, 2, 0],
    "As duas implementações não são equivalentes": [1, 1, 3, 1, 0],
}


def paragraph(filename, anchor):
    matches = [
        part.strip()
        for part in (LATEX / filename).read_text(encoding="utf-8").split("\n\n")
        # migrado: terminologia 3.6 (âncora comparada após normalizar; o parágrafo devolvido é o texto bruto)
        if normalizar_terminologia(part.strip(), arquivo=f"text/latex/{filename}").startswith(
            normalizar_terminologia(anchor, arquivo=f"text/latex/{filename}")
        )
    ]
    assert len(matches) == 1, (filename, anchor, len(matches))
    return matches[0]


def negations_by_sentence(text):
    """Contar não, nem, tampouco e sem em cada frase, na ordem do parágrafo."""
    # Não separar decimais, caminhos ou nomes de arquivo como se fossem períodos.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ])", text)
    return [len(re.findall(r"\b(?:não|nem|tampouco|sem)\b", sentence, re.I)) for sentence in sentences]


@pytest.mark.parametrize("filename,anchor", PARAGRAPHS)
def test_estado_das_negacoes_em_frases_consecutivas(filename, anchor):
    text = paragraph(filename, anchor)
    assert not re.search("[—–]", text)
    assert negations_by_sentence(text) == EXPECTED_NEGATIONS[anchor]


def test_todos_os_paragrafos_tem_contagens_documentadas():
    assert {anchor for _, anchor in PARAGRAPHS} == set(EXPECTED_NEGATIONS)


def test_contagem_inclui_os_quatro_negadores_e_mantem_a_ordem_das_frases():
    text = "Não ocorreu, nem ocorrerá, tampouco se repetirá sem revisão. A rotina terminou. Sem acesso, não se reproduz."
    assert negations_by_sentence(text) == [4, 0, 2]


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


def test_repeticao_da_magnitude_mantida_por_decisao_do_autor():
    # Revisão 2: a magnitude aparece uma vez no capítulo (resultados) e uma vez no apêndice dos custos (A.4),
    # em vez de duas vezes no capítulo; nos dois lugares é a mesma grandeza, "pares verdadeiros" a mais.
    text = (LATEX / "estudo4_gzcmd.tex").read_text(encoding="utf-8")
    appendix = (LATEX / "apendice_gzcmd.tex").read_text(encoding="utf-8")
    magnitude = "de dois a quatro pares verdadeiros a mais"
    recovered = magnitude
    arquivo = "text/latex/estudo4_gzcmd.tex"
    arquivo_ap = "text/latex/apendice_gzcmd.tex"
    assert conta(text, magnitude, arquivo=arquivo) == 1
    assert conta(appendix, recovered, arquivo=arquivo_ap) == 1
    results = text.split(r"\label{sec:e4-reruns}", 1)[1].split(r"\label{sec:e4-arranjo}", 1)[0]
    assert contem(results, magnitude, arquivo=arquivo)
    discussion = "Na cascata v3.1 sobre a base inteira, Claude Sonnet" + appendix.split(
        "Na cascata v3.1 sobre a base inteira, Claude Sonnet", 1)[1]
    discussion = discussion.split("\n\n", 1)[0]
    assert contem(discussion, recovered, arquivo=arquivo_ap)
    assert "mais inconclusivos e encaminhamentos humanos" in discussion
    assert r"vinte vezes mais caro por \textit{token}" in discussion
    assert "não constitui resultado do arranjo" in discussion
    assert "não permite atribuir o limite da revisão apenas ao tamanho do modelo" in discussion


def test_ressalvas_de_fidelidade_permanecem_explicitas():
    calibration = paragraph("estudo4_gzcmd.tex", "A calibração por âncoras preservou a capacidade")
    assert contem(calibration, "não demonstrou calibração absoluta na faixa intermediária da nota final", arquivo="text/latex/estudo4_gzcmd.tex")  # revisão 2: faixa intermediária da nota final
    assert "não corrigiria o viés" in calibration
    assert "não de impossibilidade geral do método" in calibration
    equity = paragraph("estudo5_equidade.tex", "A construção da referência operacional limita a inferência")
    assert "a possível incorporação dessas evidências à referência não foi quantificada" in equity
    # A glosa editorial não muda a ressalva: conferir o período integral após
    # normalizar apenas a expansão explícita do nome do comparador.
    assert "CdR (Comparador de Registros)" in equity
    equity = equity.replace("CdR (Comparador de Registros)", "CdR")
    assert r"O corte define PAR~=~1, com viés de verificação; adota-se como hipótese de trabalho que o CdR precedeu a revisão, com nota e subescores disponíveis à revisora, e a possível incorporação dessas evidências à referência não foi quantificada (Seção~\ref{sec:padrao-ouro})." in equity
    assert "adota-se como hipótese de trabalho que o CdR precedeu a revisão, com nota e subescores disponíveis à revisora" in equity
    assert "menos rejeições não demonstram melhora de equidade" in equity
    assert "A validação em outras coortes e a observação da revisão humana são necessárias" in equity
    conclusion = paragraph("conclusao.tex", "A ordenação dos classificadores depende da métrica")
    assert r"não fundamentam a adoção dos \textit{pipelines} (sequências operacionais) avaliados" in conclusion
    assert r"Esses estudos não fundamentam a adoção dos \textit{pipelines} (sequências operacionais) avaliados, e o GZ-CMD e o arranjo P, que preserva o corte, não usam os indicadores." in conclusion
    assert "impede transferir o resultado ao XGBoost" in conclusion
    comparator = paragraph("estudo1_comparador.tex", "As duas implementações não são equivalentes")
    assert "As duas implementações não são equivalentes: os subescores e a nota final analisados nesta tese são os da implementação Python." in comparator
    assert "As divergências não alteram os estudos seguintes, que tomam essa nota como dada, mas limitam a afirmação de que a reimplementação reproduz o protótipo Java." in comparator


@pytest.mark.parametrize("filename,expected", [
    ("estudo4_gzcmd.tex", "A redução de FP de P-Platt, portanto, não demonstra melhora intrínseca do revisor: muda o lote selecionado. Essas proporções, estimadas em coortes que incluem pares mais fáceis, não foram validadas prospectivamente para a região de aplicação."),
    ("estudo2_classificadores.tex", "A análise original usou validação cruzada estratificada com cinco partições da base completa, sem repetições"),
    ("estudo4_gzcmd.tex", "A análise de estabilidade descreve a estabilidade interna entre réplicas, não generalização"),
    ("estudo4_gzcmd.tex", r"a análise de estabilidade (Tabela~\ref{tab:cap8-robustez-arranjo}), não confirmatória, não justificam promover as análises alternativas a resultado principal sem novo registro"),
    ("estudo4_gzcmd.tex", "Na análise de estabilidade, o treinamento foi executado antes da aprovação da verificação de integridade do código"),
    ("capitulo2.tex", "A técnica é menos sujeita ao sobreajuste e dispõe de mecanismos intrínsecos para avaliação da importância de variáveis."),
    ("capitulo3.tex", "de modo que constitui hipótese a validar em outra coorte, sem pressupor superioridade operacional nem transferência dos parâmetros a outras bases."),
])
def test_repassada_final_preserva_termos_e_nexos(filename, expected):
    text = (LATEX / filename).read_text(encoding="utf-8")
    # migrado: terminologia 3.6
    assert contem(text, expected, arquivo=f"text/latex/{filename}") if "vínculo" in expected else expected in text
