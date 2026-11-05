from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "text/latex"


def abstracts():
    text = (LATEX / "tese.tex").read_text(encoding="utf-8")
    for start in ("O \\textit{linkage} integra sistemas sem", "\\textit{Linkage} integrates systems without"):
        begin = text.index(start)
        yield text[begin:text.index(r"\vspace{15pt}", begin)].strip()


def test_abstracts_space_limit_and_conservative_operational_comparison():
    resumo, abstract = abstracts()
    for text in (resumo, abstract):
        assert len(text.split()) <= 500
        assert len(text.split("\n\n")) == 4
        assert "Zenodo" in text and "211" in text
        assert not re.search("—|–", text)
    assert "211 preservados por construção" in resumo
    assert "Nos 36 PAR~=~2" in resumo
    assert "pre-specified protocol versioned in the repository (not a public registration)" in abstract
    assert "61,696" in abstract and "85.4" in abstract
    assert "37.5" in abstract and "94.5" in abstract and "86.3" in abstract
    assert "Zenodo deposit covers the whole-dataset cascade" in abstract
    assert "not the Python package" in abstract
    assert "lost 13 and nine" in abstract and "lost 27 and 17" in abstract
    assert "referring 19 and 17" in abstract and "referring one and 11" in abstract
    assert "42.5\\% versus 45.0\\%" in abstract
    assert "86.3\\% for females and 93.3\\% for males" in abstract
    assert "13 e nove" in resumo and "27 e 17" in resumo
    assert "estratégia de máxima sensibilidade" in resumo
    assert "oito contra oito" not in resumo and "eight versus eight" not in abstract


def test_every_study5_input_table_is_referenced_in_chapter():
    chapter = (LATEX / "estudo5_equidade.tex").read_text(encoding="utf-8")
    for name in re.findall(r"\\input\{(tables/[^}]+)\}", chapter):
        table = (LATEX / name).read_text(encoding="utf-8")
        labels = re.findall(r"\\label\{([^}]+)\}", table)
        assert labels, name
        assert any(r"\ref{" + label + "}" in chapter for label in labels), name


def test_reuse_caveat_present_in_method_limitations_and_conclusion():
    study = (LATEX / "estudo4_gzcmd.tex").read_text(encoding="utf-8")
    method = study.split(r"\label{sec:e4-arranjo-metodo}")[1].split(r"\section{Resultados}")[0]
    limits = study.split(r"\label{sec:e4-limitacoes}")[1]
    conclusion = (LATEX / "conclusao.tex").read_text(encoding="utf-8")
    for text in (method, limits, conclusion):
        assert "reutilizam majoritariamente" in text
        assert r"$P\cup B1$" in text and "14" in text
        assert "não é prospectiva" in text


def test_final_editorial_contract_and_table_placement():
    study = (LATEX / "estudo4_gzcmd.tex").read_text(encoding="utf-8")
    discussion = study.split(r"\section{Discussão}")[1].split(r"\subsection{Limitações}")[0]
    limits = study.split(r"\subsection{Limitações}")[1]
    for text in (discussion, limits):
        assert "vem inteiramente" not in text
        assert "quatro contra dez" not in text
        assert r"\ref{tab:cap8-arranjo-corte-par2}" in text
    conclusion = (LATEX / "conclusao.tex").read_text(encoding="utf-8")
    assert "permanece análise de sensibilidade" in conclusion
    assert "permanece sensibilidade" not in conclusion
    chapter = (LATEX / "estudo5_equidade.tex").read_text(encoding="utf-8")
    cutoff = chapter.index(r"\subsection{Comparação com o corte 6,0}")
    for mode in ("vigilancia", "confirmacao"):
        name = f"tables/tab_equidade_arranjo_rejeicoes_cascata_v31_{mode}.tex"
        assert chapter.index(r"\input{" + name + "}") < cutoff
    assert "Não é incluída" in chapter
    assert "tab_equidade_v31_rejeicoes_cascata.tex" in chapter


def test_canonical_names_and_operational_area_cinza_definition():
    for name in ("estudo1_comparador.tex", "estudo2_classificadores.tex", "estudo3_estrategias.tex",
                 "tese.tex", "conclusao.tex", "glossario.tex"):
        text = (LATEX / name).read_text(encoding="utf-8")
        assert "Floresta Aleatória" not in text
        assert r"\textit{Random Forest}" not in text
    formal = (LATEX / "apendice_formalizacao.tex").read_text(encoding="utf-8")
    assert r"\textit{labels}" not in formal
    for name in ("capitulo1.tex", "glossario.tex"):
        text = (LATEX / name).read_text(encoding="utf-8")
        assert "Nos resultados desta tese" in text
        assert r"\ref{sec:area-cinza}" in text
    intro = (LATEX / "capitulo1.tex").read_text(encoding="utf-8")
    assert "20 minutos por par, adotada no Estudo~4" in intro
