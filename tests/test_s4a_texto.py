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
    assert "not the arrangement" in abstract


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
