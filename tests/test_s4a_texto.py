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
    assert "211 vínculos" in resumo and "PAR é o rótulo do padrão-ouro" in resumo
    assert "1 indica aceitação pelo corte; 2, recuperação pela revisão manual" in resumo
    assert "preregistered in the repository, not in a public registry" in abstract
    assert "61,696" in abstract and "85.4" in abstract and "86.3" in abstract
    assert "Zenodo deposit covers the whole-dataset cascade" in abstract
    assert "arrangement is implemented in the thesis repository" in abstract
    assert "P lost 13 and 27" in abstract and "without recalibration lost nine and 17" in abstract
    assert "96 and 17 human referrals" in abstract
    assert "86.3\\% for females (95\\% CI: 76.6 to 92.4)" in abstract
    assert "93.3\\% for males (88.3 to 96.2)" in abstract
    assert "P perdeu 13 e 27" in resumo and "comparador perdeu nove e 17" in resumo
    assert "vazamento de alvo e seleção de limiar no teste" in resumo
    assert "oito contra oito" not in resumo and "eight versus eight" not in abstract
    assert "quatro PAR~=~2 contra dez" in resumo
    assert "four PAR~=~2 links versus ten" in abstract
    for text in (resumo, abstract):
        assert "P-Platt" in text and "Claude Haiku 5.5" in text
        assert "16" in text and "75" in text and "36" in text
        assert "24 óbitos" not in text and "24 additional deaths" not in text
    assert "2.156" in resumo and "2.522" in resumo and "limite inferior" in resumo
    assert "2,156" in abstract and "2,522" in abstract and "lower bound" in abstract
    assert "intervalos sobrepostos" in resumo and "intervals overlapping" in abstract
    assert "não sustentou essa dominância" in resumo and "did not sustain dominance" in abstract
    assert "hipótese permanece condicionada à carga" in resumo
    assert "hypothesis remains conditional on workload" in abstract


def test_every_study5_input_table_is_referenced_in_chapter():
    chapter = (LATEX / "estudo5_equidade.tex").read_text(encoding="utf-8")
    active = re.sub(r"(?m)^%.*$", "", chapter)
    for name in re.findall(r"\\input\{(tables/[^}]+)\}", active):
        table = (LATEX / name).read_text(encoding="utf-8")
        labels = re.findall(r"\\label\{([^}]+)\}", table)
        assert labels, name
        assert any(r"\ref{" + label + "}" in chapter for label in labels), name


def test_reuse_caveat_present_in_method_limitations_and_conclusion():
    study = (LATEX / "estudo4_gzcmd.tex").read_text(encoding="utf-8")
    method = study.split(r"\label{sec:e4-arranjo-metodo}")[1].split(r"\section{Resultados}")[0]
    limits = study.split(r"\label{sec:e4-limitacoes}")[1]
    conclusion = (LATEX / "conclusao.tex").read_text(encoding="utf-8")
    assert "reutilizam majoritariamente" in method
    assert r"$P\cup B1$" in method and "14" in method
    for text in (method, limits, conclusion):
        assert "não é prospectiva" in text
    for text in (limits, conclusion):
        assert r"\ref{sec:e4-arranjo-metodo}" in text
        assert "reutilização" in text


def test_final_editorial_contract_and_table_placement():
    study = (LATEX / "estudo4_gzcmd.tex").read_text(encoding="utf-8")
    discussion = study.split(r"\section{Discussão}")[1].split(r"\subsection{Limitações}")[0]
    limits = study.split(r"\subsection{Limitações}")[1]
    for text in (discussion, limits):
        assert "vem inteiramente" not in text
        assert "quatro contra dez" in text
        assert r"\ref{tab:cap8-arranjo-corte-par2}" in text
    conclusion = (LATEX / "conclusao.tex").read_text(encoding="utf-8")
    assert "permanece análise de sensibilidade" in conclusion
    assert "permanece sensibilidade" not in conclusion
    assert "cascata, apresentada como comparação" in conclusion
    assert "78,5\\% e 53,4\\%" in conclusion
    assert "34 e zero falsos positivos" in conclusion
    assert "restrição de 0,2\\% foi inviável" in conclusion
    assert "Claude Sonnet 5.5 acrescentou de dois a quatro vínculos" in conclusion
    assert "vinte vezes maior por" in conclusion
    chapter = (LATEX / "estudo5_equidade.tex").read_text(encoding="utf-8")
    cutoff = chapter.index(r"\subsection{Comparação com o corte 6,0}")
    for mode in ("vigilancia", "confirmacao"):
        name = f"tables/editorial/tab_equidade_arranjo_rejeicoes_cascata_v31_{mode}.tex"
        assert chapter.index(r"\input{" + name + "}") < cutoff
    assert "Não é incluída" in chapter
    assert "tab_equidade_v31_rejeicoes_cascata.tex" in chapter


def test_canonical_names_and_operational_area_cinza_definition():
    for name in ("estudo1_comparador.tex", "estudo2_classificadores.tex", "estudo3_estrategias.tex",
                 "tese.tex", "conclusao.tex", "glossario.tex", "capitulo2.tex",
                 "apendice_busca.tex", "siglas.tex"):
        text = (LATEX / name).read_text(encoding="utf-8")
        assert "Floresta Aleatória" not in text
        assert r"\textit{Random Forest}" not in text
    formal = (LATEX / "apendice_formalizacao.tex").read_text(encoding="utf-8")
    assert r"\textit{labels}" not in formal
    assert "Essas regiões não redefinem a área cinza operacional" in formal
    assert r"$[5,9)$" in formal and r"\ref{sec:area-cinza}" in formal
    for name in ("capitulo1.tex", "glossario.tex"):
        text = (LATEX / name).read_text(encoding="utf-8")
        assert "Nos resultados desta tese" in text
        assert r"\ref{sec:area-cinza}" in text
    intro = (LATEX / "capitulo1.tex").read_text(encoding="utf-8")
    assert "20 minutos por par, adotada no Estudo~4" in intro
