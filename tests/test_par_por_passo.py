from pathlib import Path

import pandas as pd
import pytest

from scripts.par_por_passo import aggregate, generate
from local_data import DATA_ROOT, requires_files

ROOT = Path(__file__).resolve().parents[1]


def test_aggregation_counts_pending_once_and_has_no_pair_attributes():
    frame = pd.DataFrame({"passo": [11] * 4, "par": [2, 2, 2, 0],
                          "final": ["MATCH", "UNSURE", "DEFERRED", "NONMATCH"],
                          "pair_id": ["a", "b", "c", "d"], "NOME": ["private"] * 4})
    table = aggregate(frame, "vigilancia")
    assert list(table.columns) == ["modo", "passo", "par", "decisao", "n"]
    assert len(table) == 9 and table.n.sum() == 4
    assert table.loc[table.par.eq(2) & table.decisao.eq("HUMANO"), "n"].item() == 2
    frame.loc[0, "final"] = "UNKNOWN"
    with pytest.raises(ValueError, match="Unknown final"):
        aggregate(frame, "vigilancia")


@requires_files(DATA_ROOT / "data/COMPARADORSEMIDENT.csv",
                ROOT / ".local/arranjo_corte/policy_vigilancia_P.csv",
                ROOT / ".local/arranjo_corte/policy_confirmacao_P.csv")
def test_versioned_counts_reproduce_local_observed_decisions():
    expected = generate(DATA_ROOT)
    actual = pd.read_csv(ROOT / "data/par_por_passo.csv")
    pd.testing.assert_frame_equal(actual, expected)


def test_PAR2_by_step_and_cutoff_denominators_match_artifacts():
    table = pd.read_csv(ROOT / "data/par_por_passo.csv")
    assert list(table.columns) == ["modo", "passo", "par", "decisao", "n"]
    assert len(table) == 54
    for mode, expected in [("vigilancia", [0, 2, 2]), ("confirmacao", [1, 3, 4])]:
        rows = table.loc[table.modo.eq(mode)]
        assert rows.n.sum() == 61696
        by_par = rows.groupby("par").n.sum()
        assert by_par.to_dict() == {0: 61449, 1: 211, 2: 36}
        accepted = rows.loc[rows.par.eq(2) & rows.decisao.eq("MATCH")].set_index("passo").n
        assert accepted.loc[[6, 9, 11]].tolist() == expected
        assert rows.loc[rows.passo.eq(11) & rows.par.eq(2), "n"].sum() == 25
        cut = rows.loc[rows.par.eq(1) & rows.decisao.eq("MATCH")].set_index("passo").n
        assert cut.to_dict() == {6: 76, 9: 120, 11: 15}
        totals = rows.loc[rows.par.isin([1, 2])].groupby("passo").n.sum()
        assert totals.to_dict() == {6: 80, 9: 127, 11: 40}
        assert (cut / totals * 100).loc[[6, 9, 11]].tolist() == pytest.approx([95, 120/127*100, 37.5])
        primary = pd.read_csv(ROOT / "data/arranjo_corte_par2.csv")
        reference = primary.loc[primary.modo.eq(mode) & primary.braco.eq("P")].iloc[0]
        counts = rows.loc[rows.par.eq(2)].groupby("decisao").n.sum()
        assert counts.to_dict() == {"MATCH": reference.aceitos, "HUMANO": reference.humano,
                                    "NONMATCH": reference.perdidos}
