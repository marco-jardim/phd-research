"""Conformidade W12b: recomendações, denominadores e fontes da síntese."""

import re

from test_s4a_texto import abstracts
from test_w11b import answer, rows, text


def test_abstracts_recommend_fusion_without_changing_hypothesis_or_limits():
    pt, en = abstracts()
    for body in (pt, en):
        assert len(body.split()) <= 500
        assert len(re.findall(r"\d+(?:[.,]\d+)*", body)) <= 18
        assert "OpenRecLink" in body
    for fragment in ("não da política de decisão", "fusão por posição recíproca",
                     "coorte nova com referência cega", "sua vantagem sobre a probabilidade calibrada não foi demonstrada",
                     "nenhuma das estratégias de decisão comparadas domina"):
        assert fragment in pt
    for fragment in ("not from the decision policy", "reciprocal rank fusion",
                     "new cohort with a blinded reference", "its advantage over the calibrated probability was not demonstrated",
                     "none of the compared decision strategies dominates", "nine links from the queue head"):
        assert fragment in en
    assert "por probabilidade ou escore" not in pt
    assert "by probability or score" not in en


def test_dominance_scope_and_recommendation_are_consistent_in_slides_and_script():
    for name in ("text/slides/defesa.tex", "text/slides/roteiro_defesa.md"):
        source = text(name)
        assert not re.search(r"Na vigilância, nenhuma ordenação domina", source)
        assert "Na vigilância, entre O1 a O4 e P, nenhuma ordenação domina" in source
        assert "Backup: mecanismos, cenários humanos e registro interno" in source
        assert "por probabilidade ou escore" not in source
    assert "vazamento reduzido pela seleção restrita" in answer(34)
    assert "Experimento 1 do Estudo 4" in answer(34)


def test_model_comparison_uses_acceptance_stratum_not_total_sample():
    aggregate = rows("probe_frontier_agregado.csv")
    names = ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1")
    percentages = []
    for name in names:
        row, = [r for r in aggregate if r["model"] == name and r["modo"] == "vigilancia"
                and r["estrato"] == "S1" and r["passada"] in ("historico", "principal")]
        percentages.append(f'{100 * float(row["VPP"]):.1f}'.replace(".", ",") + "%")
    assert ", ".join(percentages[:2]) + " (21 resultados), " + " e ".join(percentages[2:]) in answer(84)
    assert "entre as 25 aceitações do Haiku" in answer(84)
    script = text("text/slides/roteiro_defesa.md")
    for fragment in ("V1 a V4 correspondem aos estratos S1 a S4 no CSV",
                     "V3, os 24 PAR=0 não aceitos pelo Haiku e aceitos pelo Sonnet",
                     "V4, a amostra de 88 dos demais potenciais pares"):
        assert fragment in script


def test_conclusion_preserves_denominators_and_glossary_defines_new_models():
    conclusion = text("text/latex/conclusao.tex")
    for fragment in ("configuração de sensibilidade do NB02", "Random Forest com pesos extremos, limiar 0,3",
                     r"28,9\% (13 em 45)", r"42,3\% (11 em 26)",
                     "por exemplo após mudança de estado civil", "Soundex do primeiro e do último fragmentos",
                     "Experimento~1 do Estudo~4", "propagação por pessoa recuperou somente dois vínculos"):
        assert fragment in conclusion
    glossary = text("text/latex/glossario.tex")
    for name in ("Claude Opus 5.5", "Claude Fable 5.1"):
        assert rf"\item[{name}]" in glossary
    candidate = glossary.split(r"\item[v4 candidata]", 1)[1].split(r"\item[", 1)[0]
    assert "F4 (fusão por posição recíproca" in candidate
    assert "data/sensibilidade_o3_macd.csv" in text("text/slides/roteiro_defesa.md")
    assert "docs/plans/" not in text("text/slides/roteiro_defesa.md").split("### 85.", 1)[1].split("### 86.", 1)[0]
