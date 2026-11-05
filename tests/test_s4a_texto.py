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
        assert not re.search("—|–", text)
        assert "P-Platt" in text and "XGBoost" in text
        assert len(re.findall(r'\d+(?:[.,]\d+)*', text)) <= 12
    assert r"PAR é o \textit{label} da referência" in resumo
    assert "registered internally before evaluation" in abstract
    assert "preregistered" not in abstract
    assert "ordenação calibrada recuperaria 22" in resumo
    assert "ordering would recover 22" in abstract
    assert "viés de verificação" in resumo and "incorporação" in resumo
    assert "Verification bias" in abstract and "incorporation" in abstract
    assert "intervalos sobrepostos" not in resumo and "intervals overlapping" not in abstract
    assert "16,0" in resumo and "16.0" in abstract
    assert "curva de esforço" in resumo and "effort-versus-recovered-links curve" in abstract
    assert "orçamento humano" in resumo and "human budget" in abstract
    assert "validação registrada em dados novos" in resumo
    assert "registered validation on new data" in abstract


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
        assert "quatro contra dez" not in text
        assert r"\ref{sec:e4-mesma-carga}" in text or r"\ref{tab:cap8-arranjo-corte-par2}" in text
    conclusion = (LATEX / "conclusao.tex").read_text(encoding="utf-8")
    assert "P e O3 recuperariam 23 e 22" in conclusion
    assert "21 FP e utilizou 344" in conclusion
    assert "revisor perfeito" in conclusion
    assert "não constitui dominância" in conclusion
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
    assert r"\textit{labels}" in formal
    assert "Essas regiões não redefinem a área cinza da nota" in formal
    assert r"$[5,9)$" in formal and r"\ref{sec:area-cinza}" in formal
    for name in ("capitulo1.tex", "glossario.tex"):
        text = (LATEX / name).read_text(encoding="utf-8")
        assert "área cinza da nota" in text
        assert "área cinza" in text
    intro = (LATEX / "capitulo1.tex").read_text(encoding="utf-8")
    assert "conversão em horas utiliza um parâmetro temporal" in intro
    assert "não carga histórica medida" in intro
