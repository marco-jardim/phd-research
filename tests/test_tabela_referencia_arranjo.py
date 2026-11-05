from pathlib import Path

import pandas as pd

from scripts.tabela_referencia_arranjo import linhas, render

ROOT = Path(__file__).resolve().parents[1]


def sources():
    return [pd.read_csv(ROOT / "data" / name) for name in (
        "referencia_comparador_pos_orl.csv", "arranjo_corte_metricas.csv",
        "funil_gzcmd_final_metricas.csv")]


def test_all_reference_columns_and_losses_come_from_aggregates():
    data = sources()
    rows = linhas(*data)
    assert len(rows) == 12
    principal = [r for r in rows if r[0] == "Resultado principal"]
    assert principal[0][3:] == ("96", "344", "87,0", "21", "13", "94,4")
    assert principal[1][3:] == ("17", "41", "88,7", "4", "27", "89,0")
    assert rows[2][3] == "até 61.485"
    assert [r[7] for r in rows[-3:]] == ["154", "120", "3"]
    changed = data[1].copy()
    changed.loc[changed.braco.eq("P"), "perdidos"] = 999
    assert [r[7] for r in linhas(data[0], changed, data[2]) if r[0] == "Resultado principal"] == ["999", "999"]


def test_generated_reference_is_current_and_preserves_label():
    expected = render(*sources())
    assert r"\label{tab:cap8-referencia}" in expected
    assert "Vínculos perdidos" in expected
    assert "volume examinado" in expected.lower()
    assert (ROOT / "text/latex/tables/tab_cap8_referencia.tex").read_text(encoding="utf-8") == expected
