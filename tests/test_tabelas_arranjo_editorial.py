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
from scripts.tabelas_arranjo_editorial import generate, render_reference, replace_once, step_ppv_note
from scripts.tabelas_persona_editorial import rotulos_braco
from scripts.tabelas_review2_w3 import equity_tables

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "text/latex/tables"
EDITORIAL = TABLES / "editorial"
EQUITY_GLOSS = ("ASR: arranjo sequencial do registro interno, que preserva o corte do OpenRecLink e aplica a política com recalibração isotônica abaixo dele. "
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
        # .gitattributes fixa eol=lf em tables/editorial/*.tex; alguns geradores gravam CRLF no Windows.
        # O conteúdo tem de coincidir byte a byte, exceto pelo fim de linha.
        assert path.read_bytes().replace(b"\r\n", b"\n") == (EDITORIAL / path.name).read_bytes().replace(b"\r\n", b"\n")
    assert before == {p.name: p.read_bytes() for p in TABLES.glob("*.tex")}
    with pytest.raises(ValueError, match="históricas"):
        generate(ROOT / "data", TABLES)


def test_equity_numeric_cells_only_change_specificity_precision():
    # A tabela histórica segue com P; a editorial exibe ASR (rotulos_braco), sem outra mudança de célula.
    original_text = rotulos_braco((TABLES / "tab_equidade_arranjo.tex").read_text(encoding="utf-8"))
    editorial_text = (EDITORIAL / "tab_equidade_arranjo.tex").read_text(encoding="utf-8")
    old = rows(original_text)
    new = rows(editorial_text)
    frame = pd.read_csv(ROOT / "data/equidade_arranjo_global.csv")
    assert len(old) == len(new)
    labels = {rotulos_braco(label): decision for decision, label in historical.LABELS.items()}
    assert "Arranjo ASR (vigil.)" in labels and "Arranjo ASR (confirm.)" in labels
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
        renamed = rotulos_braco(
            original.replace("grupo A menos grupo B", "grupo 1 menos grupo 2").replace("Grupo A & Grupo B", "Grupo 1 & Grupo 2"))
        if "grupos" in path.name:
            # Revisão 2: a coluna conta pares verdadeiros (classe segundo a referência), não vínculos aceitos.
            renamed = replace_once(renamed, " & Vínculos & ", r" & \shortstack{Pares\\verdadeiros} & ")
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
    # A histórica segue com P; a editorial exibe ASR (rotulos_braco) e "Braço do registro interno" (decisão D2).
    old = rows(rotulos_braco((TABLES / "tab_cap8_referencia.tex").read_text(encoding="utf-8")))
    new = rows((EDITORIAL / "tab_cap8_referencia.tex").read_text(encoding="utf-8"))
    carga = pd.read_csv(ROOT / "data/carga_abaixo_corte.csv")
    total = carga.loc[carga.passo.eq("TOTAL")].set_index("limiar")
    bound = f"{int(total.loc['por_passo', 'pares_acima_limiar']):,}".replace(",", ".")
    ceiling = f"{int(total.loc['por_passo', 'total_abaixo_corte']):,}".replace(",", ".")
    assert len(old) == len(new)
    for original, current in zip(old, new):
        if original[0] == "Estratégia":
            # Cabeçalho: a coluna conta pares verdadeiros (classe segundo a referência), não vínculos aceitos;
            # a apresentação encurta e quebra os títulos em duas linhas (a nota define "Perdidos").
            assert original[6] == "Vínculos perdidos" and current[6] == "Perdidos"
            assert current[2:6] == [r"\shortstack[r]{Humano\\(pares)}", r"\shortstack[r]{LLM\\(pares)}",
                                    r"\shortstack[r]{Sens. auto.\\(\%)}", r"\shortstack[r]{FP\\auto.}"]
            assert current[7] == r"\shortstack[r]{Sens. com\\humano (\%)} \\"
            assert original[:2] == current[:2]
        elif original[0] == "Corte + revisão manual do serviço":
            assert current[2] == "cenário $\\geq " + bound + "$"
            assert current[0] == "Corte + revisão manual de origem"  # revisão 2: sem "revisão clerical"
            assert original[1] == current[1] and original[3:] == current[3:]
        elif original[0] == "Corte + revisão humana integral (cenário)":
            assert current[2] == ceiling + " (teto)"
            assert original[:2] == current[:2] and original[3:] == current[3:]
        elif original[0] == "CdR, revisão humana da área cinza":
            # Revisão 2: [5,9) é a "faixa intermediária da nota final"; "área cinza" é só a região abaixo do corte.
            assert current[0] == "CdR, revisão humana da faixa intermediária da nota final"
            assert original[1:] == current[1:]
        elif original[0] == "CdR, limiar da nota em 8":
            assert current[0] == r"CdR, decisão de limiar (nota $\geq 8$)"
            assert original[1:] == current[1:]
        else:
            assert original == current
    text = (EDITORIAL / "tab_cap8_referencia.tex").read_text(encoding="utf-8")
    assert "2.522" in text and "não limiar escolhido para ASR" in text
    historical_text = (TABLES / "tab_cap8_referencia.tex").read_text(encoding="utf-8")
    original_text = rotulos_braco(historical_text)
    # Decisão D2: o arranjo é o braço do registro interno, não o resultado principal (só rótulo, sem célula numérica).
    assert r"\multicolumn{8}{l}{\textit{Resultado principal}}" in historical_text
    assert r"\multicolumn{8}{l}{\textit{Braço do registro interno}}" in text
    assert "Resultado principal" not in text and "resultado principal" not in text
    # As células permanecem intactas; a nota agora explicita os três cenários S9.
    global_bound = f"{int(total.loc['global', 'pares_acima_limiar']):,}".replace(",", ".")
    # S3 explicita também a etapa anterior à qual o registro interno se refere.
    # A especificação das colunas (larguras) e o \tabcolsep são layout, não resultado.
    assert r"\resizebox" not in text and r"\raggedright\arraybackslash}p{" in text
    layout = re.compile(r"(?m)^(?:\\begin\{tabular\}\{.*|\\setlength\{\\tabcolsep\}.*)$")
    assert numbers(layout.sub("", text)) == numbers(layout.sub("", original_text)) + Counter([bound, bound, global_bound, "8", "8", "3", "3", "0,95", "0,95", "0,99", "1,00", "2", "5,9"]) - Counter(["5,0"])
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
