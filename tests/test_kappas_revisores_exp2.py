import pandas as pd
import pytest

from scripts.kappas_revisores_exp2 import calcular_kappas


def test_kappa_filters_missing_separately_from_unsure():
    df = pd.DataFrame({
        "pair_id": ["a", "b", "c", "d", "e"],
        "A": ["MATCH", "NONMATCH", "UNSURE", "MATCH", "MATCH"],
        "B": ["MATCH", "NONMATCH", "NONMATCH", None, "NONMATCH"],
        "y": [1, 0, 0, 1, 0],
        "nota_final": [9, 5, 6, 9, 8],
    })
    result = calcular_kappas(df, "teste")
    with_unsure, without_unsure = result.iloc[0], result.iloc[1]
    assert with_unsure["n"] == 4
    assert with_unsure["n_ausentes"] == 1
    assert with_unsure["kappa"] == pytest.approx(3 / 11)
    assert without_unsure["n"] == 3
    assert without_unsure["n_unsure_excluidos"] == 1
    assert without_unsure["kappa"] == pytest.approx(2 / 5)
    assert without_unsure["discordancias_match_nonmatch"] == 1


def test_rejects_unknown_decision():
    df = pd.DataFrame({"pair_id": ["a"], "A": ["ERROR"], "B": ["MATCH"]})
    with pytest.raises(ValueError, match="Decisões inválidas"):
        calcular_kappas(df, "teste")


def test_rejects_duplicate_pair():
    df = pd.DataFrame({"pair_id": ["a", "a"], "A": ["MATCH"] * 2})
    with pytest.raises(ValueError, match="unívoco"):
        calcular_kappas(df, "teste")
