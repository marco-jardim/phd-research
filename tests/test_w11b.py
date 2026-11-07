"""Contratos W11b: síntese exploratória com números dos agregados públicos."""

import csv
from pathlib import Path
import re

import pytest

from test_w10b import APPROVED

ROOT = Path(__file__).resolve().parents[1]


def text(name):
    return (ROOT / name).read_text(encoding="utf-8")


def rows(name):
    with (ROOT / "data" / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def answer(number):
    questions = text("text/slides/roteiro_defesa.md").split("## Perguntas prováveis")[1]
    match = re.search(rf"### {number}\.[^\n]*\n\n(.*?)\n\nFonte:", questions, re.S)
    assert match, number
    return match[1]


def test_tail_numbers_and_verdict_match_context_aggregates():
    context = {(r["variante"], r["metrica"]): float(r["valor"])
               for r in rows("contexto_obito.csv")}
    count = f'{int(context["O3_contexto_MACD", "revisoes_36"]):,}'.replace(",", ".")
    assert count + " revisões" in answer(87)
    bootstrap = {(r["comparacao"], int(r["meta_nominal"])): r
                 for r in rows("contexto_robustez_bootstrap.csv")}
    supported = [(comparison, target) for (comparison, target), row in bootstrap.items()
                 if comparison.startswith("CM-")
                 and float(row["diferenca_revisoes_ic95_sup"]) < 0]
    assert supported == [("CM-O1", 36)]
    for phrase in ("somente contra O1", "meta de 36", "mesmos 36 vínculos",
                   "secundária pré-registrada", "Não demonstrou vantagem sobre F4"):
        assert phrase in answer(88)


def test_person_propagation_is_limited_to_published_recovery():
    row, = [r for r in rows("propagacao_pessoa.csv")
            if r["regra"] == "R1_UNIAO" and r["passo"] == "TOTAL"]
    assert int(row["par2"]) == int(row["propagados"]) == 2
    assert "dois vínculos recuperados, alcance restrito" in answer(87)


def test_llm_evidence_vpp_and_caution_track_same_baseline():
    metrics = {(r["modo"], r["variante"], r["tipo"], r["metrica"]): float(r["valor"])
               for r in rows("llm_evidencia.csv")}
    vpp = metrics["vigilancia", "O3-post", "aceitacao", "VPP"]
    assert f'{100 * vpp:.1f}'.replace(".", ",") + "%" in answer(87)
    for mode in ("vigilancia", "confirmacao"):
        assert metrics[mode, "O3-post", "aceitacao", "VPP"] > metrics[
            mode, "O3", "aceitacao", "VPP"]
        assert metrics[mode, "O3-post", "aceitacao", "FP"] > 0
    conclusion = text("text/latex/conclusao.tex")
    for phrase in ("sem dispensar revisão humana", "cobertura parcial e selecionada",
                   "não equivalem à validação aninhada integral"):
        assert phrase in conclusion


def test_expanded_dossier_uses_qualitative_fallback_without_public_number():
    body = answer(87).split("Dossiê ampliado:", 1)[1].split("São explorações", 1)[0]
    assert "o critério de utilidade pré-registrado não foi atingido" in body
    assert not re.search(r"\d", body)
    assert "a ampliação não demonstrou melhora uniforme da decisão" in text(
        "text/latex/conclusao.tex")


def test_v4_candidate_is_prospective_and_does_not_replace_approved_recommendation():
    conclusion = text("text/latex/conclusao.tex")
    assert conclusion.count(APPROVED) == 1
    future = conclusion.split(r"\label{sec:concl-trabalhos-futuros}")[1]
    for phrase in ("mais promissores para a cauda", "não demonstrou vantagem sobre a F4",
                   "antes do acesso aos dados novos", "comparação primária",
                   "comparações secundárias", "nem substitui a recomendação da F4",
                   "distintas das comparações fixadas no registro da v4 candidata",
                   "parâmetros da parada permanecem pendentes de escolha"):
        assert phrase in future
    for name in ("text/latex/conclusao.tex", "text/latex/capitulo1.tex",
                 "text/latex/glossario.tex", "text/slides/roteiro_defesa.md"):
        assert "gzcmd/M_v4_candidata.md" in text(name)
    glossary = text("text/latex/glossario.tex")
    for term in ("Contexto por óbito (atributos)", "v4 candidata", "LLM como evidência"):
        assert rf"\item[{term}]" in glossary
    assert "Não é versão já validada" in answer(89)


@pytest.mark.parametrize("number", [87, 88, 89])
def test_new_answers_are_short_and_follow_style(number):
    body = answer(number)
    assert len(body.split()) <= 70
    assert not re.search(r"[\u2013\u2014]|zona cinzenta", body)
    # Ponto só pode separar milhares, nunca casas decimais.
    numbers_with_dots = re.findall(r"(?<!\w)\d[\d.]*\.\d+", body)
    assert all(re.fullmatch(r"\d{1,3}(?:\.\d{3})+", value)
               for value in numbers_with_dots)


def test_protocol_backup_mentions_v4_without_expanding_final_recommendation():
    frames = re.findall(r"\\begin\{frame\}(.*?)\\end\{frame\}",
                        text("text/slides/defesa.tex"), re.S)
    assert len(frames) == 49
    assert "v4 candidata:" in frames[-1]
    assert "secundárias pré-registradas para a coorte nova" in frames[-1]
    assert len(frames[28].split(r"\fonte")[0].split()) <= 210
