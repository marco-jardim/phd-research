"""Testes portáveis da atualização, sem extrato privado ou chamadas externas."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from scripts.llm_evidencia import (
    ROOT, acceptance, crossfit_update, probability_bands, select_oldest,
)


def fixture_arrays():
    p = np.full(12, 0.2)
    y = np.tile([1, 1, 0, 0, 0, 0], 2)
    d = np.tile(["MATCH", "UNSURE", "MATCH", "NONMATCH", "NONMATCH", ""], 2)
    folds = np.repeat([0, 1], 6)
    groups = np.arange(12)
    return p, y, d, folds, groups


def test_exact_laplace_rv_and_unanswered_are_unchanged():
    p, y, d, folds, groups = fixture_arrays()
    post, table = crossfit_update(p, y, d, folds, groups)
    row = table[(table.fold == 0) & (table.faixa == 3) & (table.decisao == "MATCH")].iloc[0]
    # Training: 2 positives (MATCH=1), 3 answered negatives (MATCH=1).
    assert row.n_vinculos == 2 and row.n_nao_vinculos == 3
    assert row.RV == pytest.approx((2 / 5) / (2 / 6))
    assert post[0] == pytest.approx(0.2 * 1.2 / (0.8 + 0.2 * 1.2))
    assert np.array_equal(post[d == ""], p[d == ""])
    assert post[3] < p[3]


def test_heldout_labels_never_enter_rv_or_own_posterior():
    p, y, d, folds, groups = fixture_arrays()
    accesses = []
    before, table_before = crossfit_update(p, y, d, folds, groups,
                                          audit=lambda train, test: accesses.append((train, test)))
    y[folds == 0] = 1 - y[folds == 0]
    after, table_after = crossfit_update(p, y, d, folds, groups)
    assert np.array_equal(before[folds == 0], after[folds == 0])
    pd.testing.assert_frame_equal(table_before[table_before.fold == 0], table_after[table_after.fold == 0])
    assert all(set(train).isdisjoint(test) for train, test in accesses)


def test_group_leakage_rejected():
    p, y, d, folds, groups = fixture_arrays()
    groups[-1] = groups[0]
    with pytest.raises(ValueError, match="REFREC"):
        crossfit_update(p, y, d, folds, groups)


def test_empty_class_fallback_and_probability_endpoints():
    p, y, d, folds, groups = fixture_arrays()
    y[folds == 1] = 0
    p[0], p[1] = 0, 1
    post, table = crossfit_update(p, y, d, folds, groups)
    assert np.array_equal(post[folds == 0], p[folds == 0])
    assert table[table.fold == 0].fallback_neutro.all()
    assert np.isfinite(post).all()


def test_probability_edges_and_reject_invalid():
    assert probability_bands([0, .01, .05, .1, .5, 1]).tolist() == [0, 1, 2, 3, 4, 4]
    with pytest.raises(ValueError):
        probability_bands([np.nan])


def test_oldest_run_wins_and_conflicts_counted():
    old = datetime(2026, 1, 1, tzinfo=timezone.utc)
    new = datetime(2026, 2, 1, tzinfo=timezone.utc)
    selected, conflicts = select_oldest([
        (new, "a", [{"pair_id": "synthetic", "decision": "NONMATCH"}]),
        (old, "z", [{"pair_id": "synthetic", "decision": "MATCH"}]),
        (old, "z", [{"pair_id": "other", "decision": "UNSURE"}]),
    ])
    assert selected == {"synthetic": "MATCH", "other": "UNSURE"}
    assert conflicts == 1


def test_acceptance_is_probability_not_llm_decision_and_handles_empty():
    p = np.array([.5, .6, .1])
    y, below = np.array([1, 0, 1]), np.array([True, True, False])
    assert acceptance(p, y, below) == {"aceitacoes": 2, "VP": 1, "FP": 1, "VPP": .5}
    assert acceptance(p * 0, y, below)["VPP"] is None


def test_invalid_decision_rejected():
    p, y, d, folds, groups = fixture_arrays()
    d[0] = "INVALID"
    with pytest.raises(ValueError, match="decisões"):
        crossfit_update(p, y, d, folds, groups)


def test_exported_outputs_are_aggregate_and_internally_consistent():
    import json

    summary = pd.read_csv(ROOT / "data/llm_evidencia.csv")
    rv = pd.read_csv(ROOT / "data/llm_evidencia_rv.csv")
    meta = json.loads((ROOT / "data/llm_evidencia_metadata.json").read_text(encoding="utf-8"))
    assert list(summary) == ["modo", "variante", "tipo", "metrica", "valor"]
    assert not any("pair_id" in column or "REFREC" in column for column in rv)
    assert len(rv) == 2 * 5 * 5 * 3
    assert np.isfinite(rv.RV).all() and (rv.RV > 0).all()
    assert meta["api_calls"] == 0 and meta["folds_reproduced"]
    for mode, coverage in meta["coverage"].items():
        assert coverage["fracao_par2"] == coverage["par2_com_resposta"] / 36
        assert rv[rv.modo.eq(mode)].n_aplicados.sum() == sum(
            row["unique_responses"] for row in meta["sources"]
            if row.get("modo") == mode and "unique_responses" in row)
        for variant in ("O3", "O3-post", "P-LLM"):
            values = summary[summary.modo.eq(mode) & summary.variante.eq(variant)
                             & summary.tipo.eq("aceitacao")].set_index("metrica").valor
            assert values.VP + values.FP == values.aceitacoes
            assert values.VPP == pytest.approx(values.VP / values.aceitacoes)
