from pathlib import Path

import pandas as pd

from scripts.tabela_referencia_arranjo import linhas, notas, render

ROOT = Path(__file__).resolve().parents[1]


def sources():
    return [pd.read_csv(ROOT / "data" / name) for name in (
        "referencia_comparador_pos_orl.csv", "arranjo_corte_metricas.csv",
        "funil_gzcmd_final_metricas.csv")]


def test_all_reference_columns_and_losses_come_from_aggregates():
    data = sources()
    rows = linhas(*data)
    assert len(rows) == 12
    assert rows[0][3] == "0" and rows[1][3] == "n.d."
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


def test_note_results_follow_source_and_manual_sensitivity_is_not_accuracy():
    ref, arr, _ = sources()
    text = notas(ref, arr)
    assert "100,0\\%" in text and "não de acurácia observada" in text
    assert "211 vínculos" in text and "36" in text and "151 vínculos" in text
    assert "123 pelo corte" in text and "28 pela revisão" in text
    assert "127 e 127" in text and "113 e 65" in text
    assert "61.485" in text and "93+0{,}95" in text and "/247" in text
    ref.loc[ref.estrategia.eq("orl_corte_servico"), "vp_automaticos"] = 999
    ref.loc[ref.estrategia.eq("pos_orl_revisao_humana_area_cinza"), "vinculos_area_cinza"] = 888
    arr.loc[arr.braco.eq("P"), "aceitos"] = 777
    changed = notas(ref, arr)
    assert "999 vínculos" in changed and "888 vínculos" in changed and "777 e 777" in changed
