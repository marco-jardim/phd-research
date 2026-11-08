"""Só são permitidas casas decimais, notas e substituições terminológicas.

Transformações: especificidade com duas casas (CSV), nota de exclusão do passo
11, gabarito -> padrão-ouro, corte SCORE -> corte do escore OpenRecLink,
MISSING -> sem informação, carga mínima derivada do CSV, teto identificado
como teto e nota do limiar 8. Nenhum outro valor tabular pode mudar.
"""
from pathlib import Path
from collections import Counter
import re

import pandas as pd
import pytest

from scripts import equidade_arranjo as historical
from scripts.tabelas_arranjo_editorial import generate, render_reference, step_ppv_note
from scripts.tabelas_review2_w3 import equity_tables
from terminologia import normalizar_terminologia

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "text/latex/tables"
EDITORIAL = TABLES / "editorial"
EQUITY_GLOSS = ("P: arranjo principal, que preserva o corte do OpenRecLink e aplica a política com recalibração isotônica abaixo dele. "
                "CdR: Comparador de Registros. REFREC identifica o registro do SIM. "
                "Cascata v3.1: política de quatro ações aplicada à base inteira, incluindo revisão humana.")


def rows(text):
    return [line.split(" & ") for line in text.splitlines()
            if " & " in line and not line.startswith("\\multicolumn")]


def numbers(text):
    return Counter(re.findall(r"\d+(?:[.,]\d+)*", text.replace("{,}", ",")))


def test_editorial_versions_reproduce_without_overwriting_historical(tmp_path):
    before = {p.name: p.read_bytes() for p in TABLES.glob("*.tex")}
    generate(ROOT / "data", tmp_path)
    equity_tables(tmp_path)
    assert len(list(tmp_path.glob("*.tex"))) == 10
    for path in tmp_path.glob("*.tex"):
        assert path.read_bytes() == (EDITORIAL / path.name).read_bytes()
    assert before == {p.name: p.read_bytes() for p in TABLES.glob("*.tex")}
    with pytest.raises(ValueError, match="históricas"):
        generate(ROOT / "data", TABLES)


def test_equity_numeric_cells_only_change_specificity_precision():
    original_text = (TABLES / "tab_equidade_arranjo.tex").read_text(encoding="utf-8")
    editorial_text = (EDITORIAL / "tab_equidade_arranjo.tex").read_text(encoding="utf-8")
    old = rows(original_text)
    new = rows(editorial_text)
    frame = pd.read_csv(ROOT / "data/equidade_arranjo_global.csv")
    assert len(old) == len(new)
    labels = {label: decision for decision, label in historical.LABELS.items()}
    allowed_numbers = numbers(original_text)
    for original, current in zip(old, new):
        assert len(original) == len(current)
        if original[0] in labels:
            row = frame.loc[frame.decisao.eq(labels[original[0]])].iloc[0]
            expected = f"{100 * row.espec:.2f}".replace(".", "{,}")
            assert current[7] == expected
            allowed_numbers.subtract(numbers(original[7]))
            allowed_numbers.update(numbers(expected))
            assert original[:7] == current[:7] and original[8:] == current[8:]
        else:
            assert original == current
    text = (EDITORIAL / "tab_equidade_arranjo.tex").read_text(encoding="utf-8")
    assert "99{,}97" in text and "99{,}99" in text
    assert "não para os testes pareados de VPP" in text
    assert "referência operacional" in text and "corte do escore OpenRecLink" in text
    assert "gabarito" not in text and "corte SCORE" not in text
    # Os números novos da nota vêm dos denominadores e grupos dos CSVs.
    note = step_ppv_note(*[pd.read_csv(ROOT / "data" / f"equidade_arranjo_{name}.csv")
                           for name in ("maxdiff", "group_metrics", "pairwise_tests")])
    allowed_numbers.update(numbers(note))
    allowed_numbers.update(numbers(EQUITY_GLOSS))
    assert numbers(editorial_text) == +allowed_numbers


def test_group_and_rejection_tables_preserve_all_cells_and_only_translate_note():
    for path in EDITORIAL.glob("tab_equidade_arranjo_*.tex"):
        original = (TABLES / path.name).read_text(encoding="utf-8")
        current = path.read_text(encoding="utf-8")
        renamed = original.replace("grupo A menos grupo B", "grupo 1 menos grupo 2").replace("Grupo A & Grupo B", "Grupo 1 & Grupo 2")
        assert rows(renamed) == rows(current)
        if "grupos" in path.name:
            assert "ao menos 30 observações no denominador de cada métrica" in current
            assert "antes de Benjamini-Hochberg" in current
            assert r"\path{data/equidade_arranjo_group_metrics.csv}" in current
            assert "como na análise anterior" not in current
        else:
            expected = renamed.replace(r"\end{table}", r"\par\footnotesize " + EQUITY_GLOSS + "\n" + r"\end{table}")
            assert current == expected


def test_reference_cells_only_add_csv_workload_bound_and_identify_ceiling():
    old = rows((TABLES / "tab_cap8_referencia.tex").read_text(encoding="utf-8"))
    new = rows((EDITORIAL / "tab_cap8_referencia.tex").read_text(encoding="utf-8"))
    carga = pd.read_csv(ROOT / "data/carga_abaixo_corte.csv")
    total = carga.loc[carga.passo.eq("TOTAL")].set_index("limiar")
    bound = f"{int(total.loc['por_passo', 'pares_acima_limiar']):,}".replace(",", ".")
    ceiling = f"{int(total.loc['por_passo', 'total_abaixo_corte']):,}".replace(",", ".")
    assert len(old) == len(new)
    for original, current in zip(old, new):
        if original[0] == "Corte + revisão manual do serviço":
            assert current[2] == "cenário $\\geq " + bound + "$"
            assert normalizar_terminologia(current[0]) == normalizar_terminologia("Corte + revisão clerical de origem")  # migrado: terminologia 3.6
            assert original[1] == current[1] and original[3:] == current[3:]
        elif original[0] == "Corte + revisão humana integral (cenário)":
            assert current[2] == ceiling + " (teto)"
            assert original[:2] == current[:2] and original[3:] == current[3:]
        elif original[0] == "CdR, revisão humana da área cinza":
            assert normalizar_terminologia(current[0]) == normalizar_terminologia("CdR, revisão humana da área cinza da nota")  # migrado: terminologia 3.6
            assert original[1:] == current[1:]
        elif original[0] == "CdR, limiar da nota em 8":
            assert current[0] == r"CdR, decisão de limiar (nota $\geq 8$)"
            assert original[1:] == current[1:]
        else:
            assert original == current
    text = (EDITORIAL / "tab_cap8_referencia.tex").read_text(encoding="utf-8")
    assert "2.522" in text and "não limiar escolhido para P" in text
    original_text = (TABLES / "tab_cap8_referencia.tex").read_text(encoding="utf-8")
    # As células permanecem intactas; a nota agora explicita os três cenários S9.
    global_bound = f"{int(total.loc['global', 'pares_acima_limiar']):,}".replace(",", ".")
    # S3 explicita também a etapa anterior à qual o registro interno se refere.
    assert numbers(text) == numbers(original_text) + Counter([bound, bound, global_bound, "8", "8", "3", "3", "0,95", "0,95", "0,99", "1,00", "2", "5,9"]) - Counter(["5,0"])
    assert r"\path{data/cenarios_sensibilidade_humana.csv}" in text
    assert "sensibilidade humana 0,95, não medida" in text
    sources = [pd.read_csv(ROOT / "data" / name) for name in (
        "referencia_comparador_pos_orl.csv", "arranjo_corte_metricas.csv", "funil_gzcmd_final_metricas.csv")]
    carga.loc[carga.limiar.eq("por_passo") & carga.passo.eq("TOTAL"), "pares_acima_limiar"] = 1234
    assert "$\\geq 1.234$" in render_reference(*sources, carga)


def test_chapters_include_editorial_not_historical_versions():
    for name in ("estudo4_gzcmd.tex", "estudo5_equidade.tex"):
        text = (ROOT / "text/latex" / name).read_text(encoding="utf-8")
        text = re.sub(r"(?m)^%.*$", "", text)
        inputs = re.findall(r"\\input\{([^}]+)\}", text)
        for table in inputs:
            if "tab_equidade_arranjo" in table or "tab_cap8_referencia" in table or "tab_cap8_arranjo_corte" in table:
                assert table.startswith("tables/editorial/")
