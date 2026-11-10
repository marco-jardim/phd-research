"""Contratos da revisão por persona: glosas, colisões e números preservados."""
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import re

import pandas as pd
import pytest

from scripts.tabelas_persona_editorial import SUPPLEMENTAL, generate, transform

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "text/latex"
TABLES = LATEX / "tables"
EDITORIAL = TABLES / "editorial"


def text(name):
    return (LATEX / name).read_text(encoding="utf-8")


def test_algorithm_and_cost_factor_have_distinct_symbols():
    algorithm = text("apendice_algoritmo.tex")
    assert "$P" not in algorithm and r"\in P$" not in algorithm
    assert r"Potenciais pares $\mathcal{U}$" in algorithm
    assert r"$\mathcal{U}^{-}$" in algorithm
    formal = text("apendice_formalizacao.tex")
    assert r"$\bar C=C_{FP}C_{FN}/(C_{FP}+C_{FN})$" in formal
    assert r"O fator $\bar C$" in formal
    assert "$H=C_" not in formal


def test_pubmed_designations_do_not_collide_with_primary_arm():
    source = text("apendice_busca.tex")
    assert not re.search(r"\bP[1-6]\b", source)
    for i in range(1, 7):
        assert f"PubMed & PM{i}," in source


def test_primary_arm_and_comparisons_are_unambiguous():
    source = text("estudo4_gzcmd.tex")
    assert r"comparador \textit{a posteriori}" not in source
    assert "comparador de mesma carga" not in source
    assert "A comparação na mesma carga" in source
    assert r"braço \textit{a posteriori} preserva o corte e aplica a cascata sobre a base inteira abaixo dele, sem recalibração" in source
    assert "ASR-Platt é a sensibilidade" not in source
    # Revisão 2: P-Platt é definida na legenda da estabilidade e na descrição do arranjo (método).
    assert r"ASR-Platt (análise de sensibilidade de ASR com \textit{Platt scaling} abaixo do corte)" in source
    assert r"ASR-Platt, com \textit{Platt scaling} no lugar da isotônica abaixo do corte, foi reclassificada por emenda como análise de sensibilidade" in source
    assert "percentil 97,5" in source and "percentis 10 a 90" in source
    assert "P97,5" not in source and "P10 a P90" not in source
    assert r"constante $k=60$" not in source
    assert "treino sem subamostragem" in source
    # Revisão 2: a partição por COMPREC do Experimento 1 está no apêndice do Estudo 4, com Sinan-TB.
    assert "na partição agrupada por COMPREC (registro do Sinan-TB)" in text("apendice_gzcmd.tex")
    assert "(três vizinhos)" in text("estudo2_classificadores.tex")


def test_fusions_and_model_strata_are_glossed_at_use():
    source = text("estudo4_gzcmd.tex")
    assert "F3 (fusão por posição recíproca de O3, O1 e O3 com MACD)" in source
    assert source.index("F1 e F2 intercalam as filas") < source.index("F2, F3 e F4 superaram")
    assert "metas de 9, 23 e 36 PAR~=~2" in source
    for term in ("V1 (aceitações do Haiku)", "V2 (pares verdadeiros não aceitos pelo Haiku)",
                 "V3 (PAR~=~0 com decisões discordantes entre Haiku e Sonnet)"):
        assert term in source
    assert "depois de conhecido o resultado da cascata publicada aplicada abaixo do corte" in text("estudo5_equidade.tex")


def test_supplemental_tables_regenerate_without_historical_changes(tmp_path):
    before = {p.name: p.read_bytes() for p in TABLES.glob("*.tex")}
    generate(tmp_path)
    for source in tmp_path.glob("*.tex"):
        assert source.read_bytes() == (EDITORIAL / source.name).read_bytes()
    assert before == {p.name: p.read_bytes() for p in TABLES.glob("*.tex")}
    with pytest.raises(ValueError, match="históricas"):
        generate(TABLES)


# Linhas de tab_qw3_fairness_rejections_corte6, na ordem da tabela: (dimension, metric, group_a, group_b).
CORTE6_LINHAS = (
    ("nome_qtd_quartis", "ppv", "Q3", "Q4"), ("nome_qtd_quartis", "ppv", "Q2", "Q4"),
    ("nome_qtd_quartis", "ppv", "Q1", "Q4"), ("bandeira", "ppv", "demais_situacoes", "encerrada_obito"),
    ("faixa_idade", "ppv", "0-17", "60-79"), ("faixa_idade", "ppv", "0-17", "40-59"),
    ("faixa_idade", "ppv", "0-17", "80+"), ("faixa_idade", "ppv", "0-17", "18-39"),
    ("faixa_idade", "ppv", "18-39", "60-79"), ("faixa_idade", "ppv", "40-59", "60-79"),
    ("faixa_idade", "tpr", "18-39", "40-59"), ("sexo", "ppv", "F", "M"),
)


def _or_corte6_do_artefato():
    frame = pd.read_csv(ROOT / "data/equidade_v31_pairwise_tests.csv")
    frame = frame[frame.decisao.eq("corte6")]
    values = []
    for dimension, metric, group_a, group_b in CORTE6_LINHAS:
        row = frame[frame.dimension.eq(dimension) & frame.metric.eq(metric)
                    & frame.group_a.eq(group_a) & frame.group_b.eq(group_b)]
        assert len(row) == 1, (dimension, metric, group_a, group_b)
        values.append(row.odds_ratio.iloc[0])
    return values


def _or_duas_casas(value):
    rounded = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if rounded == 0:
        return r"$<$0{,}01"
    return f"{rounded:.2f}".replace(".", "{,}")


def _ultima_coluna(source):
    return [line.rsplit(" & ", 1)[1].rstrip().removesuffix("\\\\").strip()
            for line in source.splitlines() if " & " in line and not line.startswith("Dimens")]


def _sem_ultima_coluna(source):
    return "\n".join(re.sub(r" & [^&]*\\\\\s*$", r" \\\\", line) if " & " in line else line
                     for line in source.splitlines())


@pytest.mark.parametrize("name", SUPPLEMENTAL)
def test_supplemental_data_cells_preserve_numeric_values(name):
    def cells(source):
        result = []
        for line in source.splitlines():
            if " & " not in line or line.startswith(("Dimens", "Conclusão")):
                continue
            # A primeira coluna contém descrições editoriais, não estimativas.
            for cell in line.split(" & ")[1:]:
                if not re.search(r"[A-Za-zÀ-ÿ]", cell):
                    result.append(re.findall(r"\d+(?:[.,]\d+)*", cell))
        return result
    historical = (TABLES / (name + ".tex")).read_text(encoding="utf-8")
    editorial = (EDITORIAL / (name + ".tex")).read_text(encoding="utf-8")
    if name == "tab_qw3_fairness_rejections_corte6":
        # A coluna OR passa a duas casas na camada editorial; as demais células são idênticas à histórica
        # e a OR editorial vem do artefato, não da célula histórica.
        assert cells(_sem_ultima_coluna(historical)) == cells(_sem_ultima_coluna(editorial))
        assert _ultima_coluna(editorial) == [_or_duas_casas(v) for v in _or_corte6_do_artefato()]
        return
    assert cells(historical) == cells(editorial)


@pytest.mark.parametrize("path", list(EDITORIAL.glob("*.tex")), ids=lambda p: p.stem)
def test_persona_transform_is_idempotent(path):
    original = path.read_text(encoding="utf-8")
    once = transform(path.name, original)
    assert transform(path.name, once) == once


def test_new_inputs_keep_existing_labels_resolvable():
    for filename in ("estudo4_gzcmd.tex", "estudo5_equidade.tex", "apendice_sensibilidade.tex"):
        for name in SUPPLEMENTAL:
            assert "\\input{tables/" + name + ".tex}" not in text(filename)
        for target in re.findall(r"\\input\{(tables/editorial/[^}]+)\}", text(filename)):
            assert (LATEX / target).exists(), target


def test_table_glosses_and_metric_typography():
    for filename in ("tab_ablation_top10", "tab_qw1_deep_learning", "tab_cv_5fold",
                     "tab_imbalance_sensitivity", "tab_pareto_frontier"):
        source = (EDITORIAL / (filename + ".tex")).read_text(encoding="utf-8")
        assert not re.search(r"\bF1\b", source)
    for filename in ("tab_cap8_revisores", "tab_e4_desempenho_faixas_oof", "tab_cap8_ancoras_decisao"):
        source = (EDITORIAL / (filename + ".tex")).read_text(encoding="utf-8")
        assert "$s" not in source and not re.search(r"\bs[<\\]", source)
        assert "$v" in source
    for path in EDITORIAL.glob("*equidade*.tex"):
        assert "grupo A menos grupo B" not in path.read_text(encoding="utf-8")
    assert "chaves S1 a S4" not in (EDITORIAL / "tab_cap8_tamanho_modelo.tex").read_text(encoding="utf-8")
