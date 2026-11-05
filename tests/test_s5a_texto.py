"""Contrato editorial S5 e proveniência da camada de apresentação."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "text/latex"


def test_main_arm_robustness_and_method_deviations_are_explicit():
    study = (LATEX / "estudo4_gzcmd.tex").read_text(encoding="utf-8")
    for text in (
        "P foi dominado por P-Platt na avaliação original",
        "P permanece o resultado principal registrado",
        "dominância de P-Platt sobre P não foi sustentada",
        "7/20", "4/20", "5/20", "0/20", "20/20", "19/20",
        "0,427", "0,65", "US\\$~0,6255", "US\\$~0,8703",
        "16,0\\%", "9,06 FP", "66,7\\%", "57,1\\%",
        "22 por", "quatro por", "um, seis e seis",
        "99,97\\%", "99,99\\%", "não existe no histórico do git",
        "69 chamadas", "dois ou três PAR", "01:22 UTC", "04:13 UTC",
        "não generalização", r"\input{tables/tab_cap8_robustez_arranjo.tex}",
        r"\ref{tab:cap8-robustez-arranjo}",
    ):
        assert text in study, text
    deviations = (ROOT / "gzcmd/M_arranjo_corte_desvios.md").read_text(encoding="utf-8")
    for text in ("8c16a41", "83c7842", "parameter_change: false", "no sensitivity API calls",
                 "1e-14", "especificidade", "antes do registro", "não generalização"):
        assert text in deviations, text


def test_objectives_and_conclusion_do_not_recommend_failed_recalibration():
    objectives = (LATEX / "capitulo3.tex").read_text(encoding="utf-8")
    assert "potenciais pares não decididos pelo corte" in objectives
    assert "Desenvolver e avaliar o GZ-CMD" in objectives
    assert "prática do serviço (corte seguido de revisão manual)" in objectives
    assert "origem do registro" not in objectives
    assert "O objetivo específico 4 mantém" not in objectives
    conclusion = (LATEX / "conclusao.tex").read_text(encoding="utf-8")
    assert "única vantagem" in conclusion and "limite inferior" in conclusion
    assert "não adotar a recalibração isotônica abaixo do corte" in conclusion
    assert "validação em dados novos antes da adoção" in conclusion
    assert "Recomenda-se a adoção de dois" not in conclusion
    assert "quantifica-se o custo de revisão manual" not in conclusion
    assert "sem demonstrar manutenção integral" not in conclusion
    for path in [LATEX / name for name in ("capitulo1.tex", "capitulo3.tex", "conclusao.tex",
                                           "estudo4_gzcmd.tex", "tese.tex", "glossario.tex")]:
        assert not re.search("—|–", path.read_text(encoding="utf-8"))
