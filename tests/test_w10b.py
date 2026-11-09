"""Contratos editoriais W10b contra os agregados publicados, sem dados privados."""

import csv
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "text/latex"
# Recomendação aprovada (decisão da autora de 2026-11-08, conclusão reescrita): preservar o corte e ordenar abaixo dele
# pela probabilidade calibrada, isolada (O3) ou fundida (F4; F3 com as MACD), pela curva de esforço; não "só F4".
APPROVED = (
    r"Recomenda-se, a validar em coorte nova com referência cega, preservar o corte do "
    r"OpenRecLink e, abaixo dele, ordenar a fila de revisão manual pela probabilidade "
    r"calibrada fora da amostra, isolada (O3) ou fundida ao escore do OpenRecLink por "
    r"posição recíproca (F4; F3 quando houver as MACD), escolhendo entre elas pela curva "
    r"de esforço."
)


def text(path):
    return (ROOT / path).read_text(encoding="utf-8")


def rows(name):
    with (ROOT / "data" / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pt(value, places=1):
    return f"{float(value):,.{places}f}".translate(str.maketrans(".,", ",."))


def test_approved_recommendation_is_literal_unique_and_in_recommendations():
    conclusion = text("text/latex/conclusao.tex")
    assert conclusion.count(APPROVED) == 1
    recommendations = conclusion.split(r"\label{sec:concl-recomendacoes}")[1].split(
        r"\section{Validação e próximos passos}")[0]
    assert APPROVED in recommendations
    # As ressalvas dos IC e do estimando estão na seção de limitações da conclusão reescrita.
    assert "condicionais às probabilidades" in conclusion
    assert "o que altera o estimando" in conclusion
    assert "a diferença entre O3 e as fusões não foi demonstrada" in recommendations
    # Capítulo 1 reescrito (âncora C08r): a recomendação é O3 isolada ou fundida (F4; F3 com as MACD),
    # pela curva de esforço; o texto antigo "sem vantagem demonstrada" foi substituído.
    intro = text("text/latex/capitulo1.tex")
    assert "Nenhuma ordenação domina em todas as cargas" in intro
    assert "não têm vantagem demonstrada" in intro


def test_bootstrap_claims_match_published_csv():
    bootstrap = {(r["comparacao"], int(r["meta_nominal"])): r
                 for r in rows("fusao_robustez_bootstrap.csv")}
    for target, median in ((9, -38), (36, -1203)):
        row = bootstrap["F4-O1", target]
        assert float(row["diferenca_revisoes_mediana"]) == median
        assert float(row["diferenca_revisoes_ic95_inf"]) < float(
            row["diferenca_revisoes_ic95_sup"]) < 0
        assert pt(-median, 0) in text("text/latex/conclusao.tex")
    assert float(bootstrap["F4-O1", 23]["diferenca_revisoes_ic95_sup"]) > 0
    for target in (9, 23, 36):
        row = bootstrap["F4-O3", target]
        assert float(row["diferenca_revisoes_ic95_inf"]) <= 0 <= float(
            row["diferenca_revisoes_ic95_sup"])


# W5: o bloco limítrofe e o filtro literal foram movidos do capítulo de dados para o apêndice de origem.
@pytest.mark.parametrize("name", ["conclusao.tex", "apendice_linkage_origem.tex", "glossario.tex"])
def test_boundary_precision_is_explained_and_matches_metadata(name):
    metadata = json.loads(text("data/fusao_robustez_metadata.json"))["faixa"]
    assert metadata["operational_n"] == 2522
    assert metadata["operational_PAR2"] == 36
    assert metadata["literal_n"] == 2495
    assert metadata["literal_PAR2"] == 31
    assert metadata["operational_min_score"] >= -5.6485
    assert metadata["operational_min_score"] < metadata["literal_threshold"]
    body = text(f"text/latex/{name}").replace("{,}", ",")
    # A conclusão reescrita traz só o limiar arredondado e as contagens do filtro literal; as casas extras
    # do bloco limítrofe ficam no apêndice de origem e no glossário.
    fragments = (("-5,648", "2.495 potenciais pares e 31 pares verdadeiros", "2.522 potenciais pares de maior escore")
                 if name == "conclusao.tex"
                 else ("-5,6485", "-5,64846", "literal", "2.495 candidatos", "31 PAR"))
    for fragment in fragments:
        assert fragment in body


def test_model_numbers_and_costs_are_backed_by_aggregate_tables():
    script = text("text/slides/roteiro_defesa.md")
    aggregate = [r for r in rows("probe_frontier_agregado.csv")
                 if r["modo"] == "vigilancia" and r["estrato"] == "TOTAL_AMOSTRA"
                 and r["passada"] in ("historico", "principal")]
    costs = rows("probe_frontier_custos.csv")
    names = {"claude-haiku-5-5": "Claude Haiku 5.5",
             "claude-sonnet-5-5": "Claude Sonnet 5.5",
             "claude-opus-5-5": "Claude Opus 5.5",
             "claude-fable-5-1": "Claude Fable 5.1"}
    assert len(aggregate) == 4
    for row in aggregate:
        model = row["model"]
        scope = "coorte_historica" if "sonnet" in model else "probe"
        cost, = [r for r in costs if r["model"] == model and r["escopo"] == scope]
        expected = (f"| {names[model]} | {int(float(row['VP']))}/{int(float(row['FP']))} "
                    f"| {pt(100 * float(row['VPP']))} | {pt(cost['usd_por_par_revisado'], 4)} |")
        assert expected in script
    historical = [r for r in rows("probe_haiku_sonnet_coorte.csv")
                  if r["coorte"] == "original" and r["faixa"] == "TOTAL"]
    for row in historical:
        assert pt(100 * float(row["agreement"])) + "%" in script
        assert pt(row["n"], 0) in script


def test_reference_turning_points_and_remaining_losses_match_csv():
    script = text("text/slides/roteiro_defesa.md")
    baseline = next(r for r in rows("probe_frontier_agregado.csv")
                    if r["model"] == "claude-haiku-5-5" and r["modo"] == "vigilancia"
                    and r["estrato"] == "TOTAL_AMOSTRA" and r["passada"] == "historico")
    fraction = (0.5 * float(baseline["MATCH"]) - float(baseline["VP"])) / float(baseline["FP"])
    assert pt(100 * fraction, 1) + "%" in script
    metrics = rows("vies_referencia.csv")
    for metric in ("O3-O1_margem25_64", "F4-O1_revisoes_25", "F4-O1_revisoes_100"):
        scenarios = [r for r in metrics if r["metrica"] == metric]
        assert scenarios
        assert all(float(r["mediana"]) < 0 for r in scenarios)
    for mode in ("vigilancia", "confirmacao"):
        remaining, = [r for r in metrics if r["modo"] == mode and r["metrica"] == "perdidos"
                      and float(r["fp_abaixo"]) == float(r["fp_acima"]) == 0
                      and float(r["par2_nao_vinculo"]) == 0.2]
        assert pt(remaining["valor_esperado"]) in script
    turns = rows("vies_referencia_viradas.csv")
    c1, = [r for r in turns if r["modo"] == "vigilancia" and r["eixo"] == "fp_uniforme"
           and r["conclusao"] == "C1_VPP_menor_50" and float(r["par2_nao_vinculo"]) == 0]
    assert float(c1["fracao_virada"]) == 0.5
    # Conclusão reescrita: as ressalvas de viés da referência permanecem no Estudo 4 (Seção da referência).
    study = text("text/latex/estudo4_gzcmd.tex")
    for fragment in ("pela mediana", "posição de precaução", "cerca de 40", "antes da defesa"):
        assert fragment in study


def test_new_answers_remain_contiguous_short_and_slide_stays_under_budget():
    script = text("text/slides/roteiro_defesa.md")
    questions = script.split("## Perguntas prováveis")[1].split("## Figuras e critérios")[0]
    assert [int(n) for n in re.findall(r"^### (\d+)\.", questions, re.M)] == list(range(1, 90))
    answers = re.findall(r"### (\d+)\.[^\n]*\n\n(.*?)\n\nFonte:", questions, re.S)
    assert len(answers) == 89
    assert all(len(answer.split()) <= 70 for _, answer in answers)
    frames = re.findall(r"\\begin\{frame\}(.*?)\\end\{frame\}", text("text/slides/defesa.tex"), re.S)
    assert len(frames[28].split(r"\fonte")[0].split()) <= 210
    for fragment in ("fusão RRF", "referência cega", "vantagem sobre a probabilidade não demonstrada"):
        assert fragment in frames[28]
    assert "Revisão cega independente não foi possível antes da defesa" in frames[25]
