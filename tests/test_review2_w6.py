"""Conformidade W6: redação conservadora e apresentação dos agregados existentes."""
import csv
from pathlib import Path
import re

import pandas as pd
import pytest

from scripts.perfil_obitos_agregado import table
from scripts.tabelas_review2_w3 import historical_tables

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "text/latex"


def text(name):
    return (LATEX / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", [
    "estudo3_estrategias.tex", "estudo4_gzcmd.tex", "conclusao.tex",
    "tables/editorial/tab_cap8_referencia.tex",
])
def test_origin_is_not_attributed_to_health_service(name):
    source = text(name)
    for obsolete in ("prática do serviço", "revisão manual do serviço",
                     "revisão do serviço", "recuperados pelo serviço",
                     "rejeitados ou não examinados pelo serviço", "conhecidos do serviço"):
        assert obsolete not in source
    assert "origem" in source


def test_objectives_and_contribution_do_not_claim_audited_policy_or_wrong_incompatibility():
    objectives = text("capitulo3.tex")
    assert "critérios que podem ser satisfeitos simultaneamente" in objectives
    assert r"\textit{equalized odds}" in objectives
    assert "estratos de processo" in objectives
    for name in ("capitulo1.tex", "capitulo2.tex", "capitulo3.tex", "apendice_algoritmo.tex"):
        source = text(name)
        assert "política auditável" not in source
        assert "decisões auditáveis" not in source
        if name != "capitulo3.tex":  # C3-04: frase retirada do objetivo
            assert "sem exercício independente de auditoria das justificativas" in source
    opening = text("conclusao.tex").split(r"\section{Resposta aos objetivos}")[0]
    assert "e não da automação das decisões pela política de quatro ações" in opening


def test_falsification_contrasts_and_partition_ranges_match_unmodified_csvs():
    comparison = pd.read_csv(ROOT / "data/comparador_mesma_carga.csv")
    for mode, k, difference in (("vigilancia", 96, 1), ("confirmacao", 17, 3)):
        row = comparison.loc[comparison.modo.eq(mode) & comparison.k.eq(k)
                             & comparison.ordenacao.eq("O3_p_cal_ml")].iloc[0]
        assert row.P_par2_recuperados - row.par2_topk == difference
    stability = pd.read_csv(ROOT / "data/robustez_arranjo_resumo.csv")
    for mode, low, high in (("vigilancia", 9, 21), ("confirmacao", 21, 29)):
        row = stability.loc[stability.tipo.eq("desfecho") & stability.bloco.eq("P")
                            & stability.braco.eq("P") & stability.faixa.eq("TODAS")
                            & stability.metrica.eq("perdidos") & stability.modo.eq(mode)].iloc[0]
        assert (row.minimo, row.maximo) == (low, high)
    study = text("estudo4_gzcmd.tex")
    criterion = next(paragraph for paragraph in study.split("\n\n")
                     if paragraph.startswith("Como critério prospectivo de falsificação"))
    assert "ordenação e orçamento humano definidos antes de examinar os novos desfechos" in criterion
    assert "referência cega independente" in criterion
    assert r"pelo menos 25\% menos revisões que O1" in criterion
    assert r"limite inferior do intervalo de 95\% entre partições predefinidas" in criterion
    assert "em cada uma das duas metas" in criterion
    assert "os 23 PAR~=~2 da vigilância e os nove da confirmação" in criterion
    assert "P supera O3 por um par verdadeiro na carga 96 da vigilância e por três na carga 17 da confirmação" in criterion
    assert "de 9 a 21 perdas na vigilância e de 21 a 29 na confirmação" in criterion
    assert "Nenhuma dessas avaliações foi realizada" in criterion
    for name in ("defesa.tex", "roteiro_defesa.md"):
        slides = (ROOT / "text/slides" / name).read_text(encoding="utf-8")
        assert "Ganho das quatro ações sobre duas ações (O3)" in slides
        assert "de 9 a 21 e de 21 a 29" in slides


def test_f1_comparison_has_false_positives_and_explicit_calculation():
    conclusion = text("conclusao.tex")
    # Conclusão reescrita: a comparação de F1 fica como "variou de 0,880 a 0,918, diferença que não permite ordenar os métodos";
    # o cálculo por VP e FP e o intervalo de 0,012 a 0,027 deixaram de constar (nenhum capítulo os repete) por concisão.
    assert "o $F_1$ médio variou de 0,880 a 0,918, diferença que não permite ordenar os métodos" in conclusion
    assert "abaixo da variação" not in conclusion
    # Revisão 2: o trecho da variação entre controles A e A0 do Estudo 3 está no apêndice das estratégias.
    for name in ("apendice_estrategias.tex",):
        source = text(name)
        assert "um ou dois verdadeiros positivos" in source
        assert re.search(r"XGBoost e Random Forest entre os controles A(?: \([^)]*\))? e A0", source)
        assert "cálculo da tese" in source or "calculada na tese" in source
        assert "partição original do comparativo" in source
        assert r"\ref{sec:sensibilidade-qw1}" in source


def test_glossary_records_historical_alias_shared_agents_and_internal_holdout():
    for name in ("glossario.tex", "siglas.tex"):
        assert "em versões anteriores do texto" in text(name)
    source = text("glossario.tex")
    assert "compartilham modelo e dossiê e diferem por uma frase de papel" in source
    assert r"Conjunto de avaliação interno (\textit{holdout})" in source
    assert "sem implicar validação externa" in source
    assert "articulação de regras, classificador e revisores" in text("capitulo1.tex")


def test_summary_period_and_numeric_ceiling_are_preserved():
    source = text("tese.tex")
    for start, period in ((r"O \textit{linkage} integra", "óbitos de 2019 a 2022"),
                          ("Linkage integrates systems", "deaths from 2019 to 2022")):
        begin = source.index(start)
        abstract = source[begin:source.index(r"\vspace{15pt}", begin)]
        assert period in abstract
        assert len(re.findall(r"\d+(?:[.,]\d+)*", abstract)) <= 18
        assert "211" not in abstract
        assert len(abstract.split()) <= 500
    assert "À Daniela Medina Coeli Bastos" in source


def test_unimplemented_steps_are_disclosed_in_algorithm_title():
    source = text("apendice_algoritmo.tex")
    title = next(line for line in source.splitlines() if "desenho avaliado do GZ-CMD" in line)
    assert "As etapas de margem estreita e cardinalidade foram previstas, mas não implementadas" in title
    assert r"\ref{sec:e4-regras-guarda}" in title


def test_profile_generator_formats_thousands_without_changing_aggregates():
    frame = pd.read_csv(ROOT / "data/perfil_obitos_agregado.csv")
    rendered = table(frame)
    assert rendered == text("tables/editorial/tab_perfil_obitos_agregado.tex")
    for value in (48862, 12834, 15286, 3841):
        assert f"{value:,}".replace(",", ".") in rendered
        assert not re.search(rf"(?<![0-9]){value}(?![0-9])", rendered)


def test_score_band_precision_follows_count_only_source(tmp_path):
    with (ROOT / "data/tab_score_bands_proveniencia.csv").open(encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["band", "n", "vinculos"]
    historical_tables(tmp_path)
    rendered = (tmp_path / "tab_score_bands.tex").read_text(encoding="utf-8")
    assert rendered == text("tables/editorial/tab_score_bands.tex")
    assert not re.search(r"\d+,\d{2}(?!\d)", rendered)
    assert "sem indicar diferença demonstrada" in text("tables/editorial/tab_qw1_deep_learning.tex")


def test_remaining_editorial_caveats_and_reductions():
    study = text("estudo4_gzcmd.tex")
    assert "Essa proporção de referência é descritiva" in study
    # Revisão 2: a validação das âncoras foi para o apêndice do Estudo 4.
    assert "As âncoras superestimam a probabilidade nas faixas $[7,9)$" in text("apendice_gzcmd.tex")
    assert "Usar $1/240$ em vez de 0,005 não altera decisões" in study
    assert "cobre $C_{LLM}$ de 0,005 a 0,25" in study
    for name in ("estudo4_gzcmd.tex", "apendice_pacote.tex", "capitulo_dados.tex"):
        assert not re.search(r"\bbandas?\b", text(name))
    # Conclusão reescrita: a versão do prompt dos braços selecionados e as limitações do filtro permanecem no Estudo 4.
    assert "a versão do prompt não está registrada nos arquivos originais dos braços selecionados" in study
    assert "a tese não participou dessas etapas" not in text("capitulo1.tex")
    assert "sem distingui-la de abstenção generalizada" in text("capitulo1.tex")
    assert "não mede concordância entre registros" in text("estudo1_comparador.tex")
    assert "7 é o limiar mínimo da nota final no filtro de homonímia histórico v3" in text("capitulo_dados.tex")
    for value in ("0{,}5", "2{,}0"):
        assert r"$\beta=" + value + "$" in text("apendice_ablacao.tex")
    roteiro = (ROOT / "text/slides/roteiro_defesa.md").read_text(encoding="utf-8")
    assert "capitulo_dados.tex:115" not in roteiro
    assert "não identificada nos originais nem nos agregados" in roteiro


def test_queue_codes_have_ordering_definitions_in_each_conclusion_paragraph():
    """Cada código de fila (O1 a O4) é definido onde aparece.

    Na tese.tex (resumo), em cada parágrafo. Na conclusão reescrita (2026-11-08), que define as filas na primeira
    menção do capítulo e depois usa só o código, exige-se a definição no parágrafo da primeira menção.
    """
    definitions = {
        "O1": ("escore do OpenRecLink",),
        "O2": ("nota final do CdR", "nota do CdR"),
        "O3": ("probabilidade calibrada fora da amostra do classificador",
               "probabilidade isotônica fora da amostra"),
        "O4": ("probabilidade recalibrada de P",),
    }
    # Forma de definição usada na conclusão: "ordenação pela probabilidade calibrada fora da amostra (O3)".
    definitions_conclusion = dict(definitions, O3=definitions["O3"] + ("probabilidade calibrada fora da amostra (O3)",))
    for name in ("conclusao.tex", "tese.tex"):
        source = text(name)
        known = definitions_conclusion if name == "conclusao.tex" else definitions
        for code in set(re.findall(r"\bO[1-4]\b", source)):
            assert any(term in source for term in known[code]), (name, code)
        if name == "conclusao.tex":
            for code in sorted(set(re.findall(r"\bO[1-4]\b", source))):
                first = next(p for p in source.split("\n\n") if re.search(rf"\b{code}\b", p))
                assert any(term in first for term in definitions_conclusion[code]), (name, code, first)
            continue
        for paragraph in source.split("\n\n"):
            for code in set(re.findall(r"\bO[1-4]\b", paragraph)):
                terms = definitions[code]
                if code == "O3":
                    # W10a usa a forma curta no cenário de erro da referência;
                    # a definição completa fora da amostra continua obrigatória.
                    terms += ("ordenação pela probabilidade calibrada (O3)",)
                assert any(term in paragraph for term in terms), (name, code, paragraph)


def test_macd_birth_and_identical_name_definition_agree_with_glossary():
    study = next(paragraph for paragraph in text("estudo4_gzcmd.tex").split("\n\n")
                 if paragraph.startswith("As Medidas Contínuas de Diferença de Datas"))
    glossary = next(line for line in text("glossario.tex").splitlines()
                    if line.startswith(r"\item[MACD]"))
    for source in (study, glossary):
        assert "atributos derivados das datas de nascimento" in source.lower()
        assert "nome idêntico" in source
        assert "Nome idêntico exige concordância dos fragmentos e proporção de fragmentos iguais de pelo menos 0,95" in source
        assert "pelo menos cinco anos" in source
        assert "Não incluem diferença" in source and "óbito e notificação" in source


def test_haiku_default_reasoning_is_not_described_as_disabled():
    study = text("estudo4_gzcmd.tex")
    clause = study.split(r"Claude Haiku 5.5 utilizou \texttt{max\_tokens}=2500", 1)[1].split("Sua geração", 1)[0]
    assert "na configuração padrão do modelo, sem configuração explícita de raciocínio" in clause
    assert "a API aplica raciocínio adaptativo por padrão" in clause
    assert "os registros não discriminam tokens de raciocínio" in clause
    for obsolete in ("sem raciocínio", "sem thinking", "sem pensamento estendido"):
        assert obsolete not in clause
