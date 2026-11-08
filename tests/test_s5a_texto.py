"""Contrato editorial S5 e proveniência da camada de apresentação."""
from pathlib import Path
import csv
import re

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "text/latex"


def test_main_arm_robustness_and_method_deviations_are_explicit():
    study = (LATEX / "estudo4_gzcmd.tex").read_text(encoding="utf-8")
    appendix = (LATEX / "apendice_gzcmd.tex").read_text(encoding="utf-8")
    for text in (
        "P foi dominado por P-Platt na avaliação original",
        # Revisão 2 (D2): P é o braço do registro interno, não o "resultado principal".
        "P é o braço do registro interno e não o resultado principal do capítulo",
        "dominância de P-Platt sobre P não foi sustentada",
        "7/20", "4/20", "5/20", "0/20", "20/20", "19/20",
        "0,427", "0,65",
        "16,0\\%", "9,06 FP", "66,7\\%", "57,1\\%",
        "22 por", "quatro por", "um, seis e seis",
        "99,97\\%", "99,99\\%", "está ausente do histórico do git",
        "dois ou três PAR", "01:22:17 UTC", "02:17:13 UTC",
        "não generalização", r"\input{tables/editorial/tab_cap8_robustez_arranjo.tex}",
        r"\ref{tab:cap8-robustez-arranjo}",
    ):
        assert text in study, text
    assert "P permanece o resultado principal do registro interno" not in study
    # Revisão 2: custos das chamadas e suplemento foram para o apêndice do Estudo 4.
    for text in ("US\\$~0,6255", "US\\$~0,8703", "O suplemento, com 69 potenciais pares"):
        assert text in appendix, text
    deviations = (ROOT / "gzcmd/M_arranjo_corte_desvios.md").read_text(encoding="utf-8")
    for text in ("8c16a41", "83c7842", "parameter_change: false", "no sensitivity API calls",
                 "1e-14", "especificidade", "antes do registro", "não generalização"):
        assert text in deviations, text


def test_objectives_and_conclusion_do_not_recommend_failed_recalibration():
    objectives = (LATEX / "capitulo3.tex").read_text(encoding="utf-8")
    assert "região abaixo do ponto de corte do OpenRecLink" in objectives
    assert "Desenvolver e avaliar um pós-processamento" in objectives
    assert "resulta do corte do OpenRecLink e da revisão manual de origem" in objectives
    assert "prática do serviço" not in objectives
    assert "origem do registro" not in objectives
    assert "O objetivo específico 4 mantém" not in objectives
    conclusion = (LATEX / "conclusao.tex").read_text(encoding="utf-8")
    assert "pior em todo desfecho" not in conclusion
    assert "não reproduz 13 e 27 dos 36" in conclusion
    assert "Recomenda-se a fusão como ordenação da fila de revisão" in conclusion
    assert "sua vantagem sobre a probabilidade calibrada não foi demonstrada" in conclusion
    assert "coorte nova com referência cega" in conclusion
    hypothesis = conclusion.split("\n\n")[1]
    assert "probabilidade calibrada fora da amostra do classificador (O3)" in hypothesis
    assert r"25\% menos revisões que O1" in hypothesis
    assert "limite inferior" in hypothesis
    assert "em cada uma das duas metas" in hypothesis
    assert "$2/36$" not in hypothesis
    assert "validação registrada em dados novos" in conclusion
    assert "Recomenda-se a adoção de dois" not in conclusion
    recommendation = conclusion.split(r"\label{sec:concl-recomendacoes}")[1].split(r"\section{Trabalhos futuros}")[0]
    assert "recalibração isotônica" not in recommendation
    assert "validação" in recommendation and "hipótese" in recommendation
    assert "quantifica-se o custo de revisão manual" not in conclusion
    assert "sem demonstrar manutenção integral" not in conclusion
    for path in [LATEX / name for name in ("capitulo1.tex", "capitulo3.tex", "conclusao.tex",
                                           "estudo4_gzcmd.tex", "tese.tex", "glossario.tex")]:
        assert not re.search("—|–", path.read_text(encoding="utf-8"))


def test_small_band_warning_follows_csv_when_editorial_instruction_differs():
    with (ROOT / "data/arranjo_corte_taxas_crossfit.csv").open(encoding="utf-8") as stream:
        rates = list(csv.DictReader(stream))
    folds = [int(row["fold"]) for row in rates if row["modo"] == "vigilancia"
             and row["banda"] == "grey_low" and float(row["e_fn"]) == 1.0]
    assert folds == [0, 1, 3, 4]
    study = (LATEX / "estudo4_gzcmd.tex").read_text(encoding="utf-8")
    assert r"quatro \textit{folds} na vigilância" in study
    assert r"\texttt{grey\_low} em dois folds" not in study
