import math

import pandas as pd
import pytest

from scripts.m8_auditoria_blocagem import chapman, loglinear, overlap, precondition, yield_by_step
from scripts.m8_chaves_llm import proxy_mask


def synthetic():
    return pd.DataFrame({"COMPREC": ["a", "b", "a", "c", "d"],
                         "REFREC": ["1", "2", "1", "3", "4"],
                         "PASSO": ["6", "6", "9", "9", "9"],
                         "y": [True, False, True, True, False],
                         "nota final": [6., 2., 6., 10., 4.]})


def test_yield_and_overlap():
    df = synthetic()
    result = yield_by_step(df).set_index("passo")
    assert result.loc["6", "pares_distintos"] == 2
    assert result.loc["6", "vinculos_so_neste_passo"] == 0
    assert result.loc["9", "vinculos_so_neste_passo"] == 1
    assert result.loc["9", "vinculos_tambem_outro_passo"] == 1
    assert result.loc["9", "taxa_vinculo_pares"] == pytest.approx(2 / 3)
    assert result.loc["9", "fracao_vinculos_cinza_nota"] == .5
    assert result.loc["9", "fracao_vinculos_nota_ge9"] == .5
    offdiag = overlap(df).query("passo_a != passo_b")
    assert offdiag["vinculos_em_comum"].eq(1).all()
    assert precondition(df)["pares_multiplos_passos"] == 1


def test_chapman_known_answer():
    result = chapman(20, 30, 10)
    expected = 21 * 31 / 11 - 1
    variance = 21 * 31 * 10 * 20 / (11 ** 2 * 12)
    assert result["total_estimado"] == pytest.approx(expected)
    assert result["variancia"] == pytest.approx(variance)
    assert result["nao_capturados"] == pytest.approx(expected - 40)
    assert result["ic95_superior"] == pytest.approx(expected + 1.96 * math.sqrt(variance))
    with pytest.raises(ValueError, match="sobreposição"):
        chapman(20, 30, 0)


def test_loglinear_known_missing_cell():
    # Tabela completa independente: mu(perfil)=10*2**sum(perfil); mu(000)=10.
    import itertools

    captures = {"a": set(), "b": set(), "c": set()}
    counter = 0
    for profile in list(itertools.product([0, 1], repeat=3))[1:]:
        for _ in range(10 * 2 ** sum(profile)):
            counter += 1
            for key, captured in zip(captures, profile):
                if captured:
                    captures[key].add(counter)
    result = loglinear(captures)
    assert result["nao_capturados"] == pytest.approx(10, rel=1e-6)
    assert result["ic95_inferior"] < result["total_estimado"] < result["ic95_superior"]


def test_proxy_dnf_and_allowlist():
    df = pd.DataFrame({"NOME prim frag igual": [1., 0., float("nan")],
                       "DTNASC dt iguais": [0., 1., 0.]})
    rule = {"proxy_dnf": [[{"campo": "NOME prim frag igual", "op": ">=", "valor": 1}],
                          [{"campo": "DTNASC dt iguais", "op": "==", "valor": 1}]]}
    assert proxy_mask(df, rule, set(df)).tolist() == [True, True, False]
    with pytest.raises(ValueError, match="allowlist"):
        proxy_mask(df, rule, {"NOME prim frag igual"})


def test_no_recapture_is_not_zero_missing():
    from scripts.m8_auditoria_blocagem import capture_table

    result = capture_table(synthetic().drop(index=2))
    assert result.loc[0, "status"] == "inviável"
    assert pd.isna(result.loc[0, "total_estimado"])
