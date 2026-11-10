from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from scripts.fig_varredura_custos import (
    build_np_figure, limit_label, np_figure_data, pct_pt, plot_np_figure,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tables():
    return (pd.read_csv(ROOT / "data/varredura_np_confirmacao.csv"),
            pd.read_csv(ROOT / "data/comparacao_np_confirmacao.csv"))


def test_pct_pt_rounds_half_up_with_decimal_comma():
    assert pct_pt(0.5384615384615384) == "53,8"
    assert pct_pt(0.9204) == "92,0"
    assert pct_pt(0.125) == "12,5"
    assert pct_pt(0.00125, 2) == "0,13"
    assert pct_pt(0.0125, 1) == "1,3"
    assert [limit_label(a) for a in (.0005, .001, .002, .005, .01)] == ["0,05%", "0,1%", "0,2%", "0,5%", "1%"]


def test_annotated_values_come_from_the_csv(tables):
    table, comparison = tables
    data = np_figure_data(table, comparison)
    assert [(d["auto_label"], d["human_label"]) for d in data["dumbbell"]] == [
        ("53,8", "93,5"), ("58,0", "92,0"), ("62,7", "90,1"), ("81,7", "87,6")]
    assert [d["direct"] for d in data["dumbbell"]] == [349, 280, 241, 21]
    assert [d["llm"] for d in data["dumbbell"]] == [0, 69, 108, 328]
    assert data["minimum_label"] == "0,61"
    assert (data["n_guardas"], data["n_min_vinculos"], data["fp_guardas_label"]) == (63, 110, "0,67")
    assert data["cost"]["sens_label"] == "87,7"
    assert (data["cost"]["llm"], data["cost"]["direct"]) == (44, 121)
    # Same values, recomputed straight from the CSV rows.
    viable = table[table.viavel & table.orcamento_humano.eq(float("inf"))].sort_values("e_fn_h")
    assert [pct_pt(v) for v in viable.sens_esperada_automatica] == [d["auto_label"] for d in data["dumbbell"]]
    assert [pct_pt(v) for v in viable.sens_esperada_com_humano] == [d["human_label"] for d in data["dumbbell"]]


def test_viability_matrix_states(tables):
    table, comparison = tables
    data = np_figure_data(table, comparison)
    assert data["budgets"][-1] == float("inf") and len(data["budgets"]) == 5
    guard = [c for c in data["columns"] if c["error"] is None]
    opened = [c for c in data["columns"] if c["error"] is not None]
    assert [c["label"] for c in guard] == ["0,05%", "0,1%", "0,2%", "0,5%"]
    assert all(set(c["states"]) == {"guardas"} for c in guard)
    assert [c["needed"] for c in opened] == [349, 280, 241, 21]
    # Budgets 25, 50, 100, 200, unlimited; a cell is viable iff the budget covers the direct-review need.
    for col in opened:
        expected = ["viavel" if b >= col["needed"] else "orcamento" for b in data["budgets"]]
        assert col["states"] == expected
    assert opened[3]["states"] == ["viavel"] * 5
    assert opened[2]["states"] == ["orcamento"] * 4 + ["viavel"]


def test_figure_texts_show_the_csv_values(tables):
    table, comparison = tables
    fig, _ = build_np_figure(table, comparison)
    try:
        shown = {t.get_text() for ax in fig.axes for t in ax.texts}
        shown |= {t.get_text() for ax in fig.axes for t in ax.get_xticklabels()}
        for label in ("53,8", "93,5", "58,0", "92,0", "62,7", "90,1", "81,7", "87,6", "349", "280", "241", "21", "165"):
            assert label in shown, label
        assert any("0,61%" in text for text in shown)
        assert fig.get_size_inches()[1] / fig.get_size_inches()[0] < 1.0
    finally:
        plt.close(fig)


def test_plot_np_figure_creates_deterministic_files(tables, tmp_path):
    table, comparison = tables
    first, second = tmp_path / "a" / "fig", tmp_path / "b" / "fig"
    plot_np_figure(table, comparison, first)
    plot_np_figure(table, comparison, second)
    for extension in (".pdf", ".png"):
        assert first.with_suffix(extension).stat().st_size > 0
        assert first.with_suffix(extension).read_bytes() == second.with_suffix(extension).read_bytes()
    assert b"/CreationDate" not in first.with_suffix(".pdf").read_bytes()
