"""Conformidade W12b: recomendações, denominadores e fontes da síntese."""

import re

from resumo_numeros import TETO_NUMEROS_RESULTADO, numeros_de_resultado
from test_s4a_texto import abstracts
from test_w11b import answer, rows, text


def test_abstracts_recommend_fusion_without_changing_hypothesis_or_limits():
    pt, en = abstracts()
    for body in (pt, en):
        assert len(body.split()) <= 500
        # Decisão da autora de 2026-11-08: teto de 24 números de RESULTADO (regra em resumo_numeros.py).
        assert len(numeros_de_resultado(body)) <= TETO_NUMEROS_RESULTADO
        assert "OpenRecLink" in body
    # Redação nova do Resumo e do Abstract (revisão 2): a fusão por posição recíproca (F3) e a referência cega
    # constam da conclusão e do Estudo 4; o resumo fixa a ordenação como origem da redução, a vantagem das fusões
    # sobre O3 não demonstrada, a validação em coorte nova e a ausência de dominância.
    # Resumo reescrito no estilo do grupo (decisão da autora de 2026-11-10): a frase da origem da redução foi
    # unificada, "As fusões com o escore" encurtou e o MATCH ao humano ("nove com 29 revisões") saiu do Resumo.
    for fragment in ("não veio da automação nem do LLM, mas da ordem de revisão dos pares",
                     "As fusões com o escore (F3 e F4)",
                     "a validar em coorte nova", "mas nenhum IC frente a O3 ficou abaixo de zero",
                     "pois nenhuma dominou em todas as cargas"):
        assert fragment in pt
    for fragment in ("came from the order in which pairs are reviewed, rather than from automating the decisions or from the LLM",
                     "Fusions of the calibrated probability with the score (F3 and F4)",
                     "to be validated in a new cohort", "but no CI relative to O3 fell below zero",
                     "since no ordering was dominant in every workload"):
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
    # Conclusão reescrita: esses denominadores e glosas permanecem nos capítulos e apêndices de origem.
    sources = {"apendice_estrategias.tex": ("Random Forest com pesos extremos da estratégia de vigilância (NB02) no limiar de 0,3",),
               "estudo4_gzcmd.tex": (r"28,9\% (13 pares verdadeiros em 45 aceitações)",
                                     r"42,3\% (11 pares verdadeiros em 26 aceitações)",
                                     "recuperou dois dos 36 pares verdadeiros (PAR~=~2) sem propagar PAR~=~0"),
               "estudo5_equidade.tex": ("por exemplo após mudança de estado civil",
                                        "Soundex do primeiro e do último fragmentos"),
               "estudo1_comparador.tex": ("Experimento~1 do Estudo~4",)}
    for name, fragments in sources.items():
        for fragment in fragments:
            assert fragment in text(f"text/latex/{name}"), (name, fragment)
    assert "LLM não deve aceitar vínculos sozinho" in conclusion
    glossary = text("text/latex/glossario.tex")
    for name in ("Claude Opus 5.5", "Claude Fable 5.1"):
        assert rf"\item[{name}]" in glossary
    candidate = glossary.split(r"\item[v4 candidata]", 1)[1].split(r"\item[", 1)[0]
    assert "F4 (fusão por posição recíproca" in candidate
    assert "data/sensibilidade_o3_macd.csv" in text("text/slides/roteiro_defesa.md")
    assert "docs/plans/" not in text("text/slides/roteiro_defesa.md").split("### 85.", 1)[1].split("### 86.", 1)[0]
