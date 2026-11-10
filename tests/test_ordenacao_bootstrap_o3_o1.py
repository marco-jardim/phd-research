"""Contratos sintéticos do bootstrap O3-O1 (sem o extrato real)."""

import numpy as np
import pandas as pd

from scripts.fusao_robustez import SEED, TARGETS, bootstrap_once
from scripts.ordenacao_bootstrap_o3_o1 import (
    COLUMNS, bootstrap_o3_o1_once, run_bootstrap, summarize_o3_o1,
)


def synthetic(n=300, groups=60, positives=40, seed=7):
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=bool)
    labels[rng.choice(n, positives, replace=False)] = True
    codes = np.sort(rng.integers(0, groups, n))
    queues = {"O1": rng.permutation(n), "O3": rng.permutation(n)}
    # O3_MACD só é exigido por fused_orders, usado no bootstrap original.
    queues["O3_MACD"] = rng.permutation(n)
    return queues, labels, codes, groups


def test_same_draws_and_efforts_as_existing_bootstrap():
    queues, labels, codes, groups = synthetic()
    a, b = np.random.default_rng(SEED), np.random.default_rng(SEED)
    for _ in range(5):
        new = bootstrap_o3_o1_once(queues, labels, codes, groups, a)
        old = bootstrap_once(queues, labels, codes, groups, b)
        assert new[1:] == old[1:]
        for name in ("O1", "O3"):
            assert np.array_equal(new[0][name], old[0][name])


def test_summary_columns_and_signs():
    effort = {"O1": np.array([10, 20, 40]), "O3": np.array([5, 10, 50])}
    table = summarize_o3_o1([(effort, 36, 100)] * 3)
    assert list(table.columns) == list(COLUMNS)
    assert table.comparacao.eq("O3-O1").all()
    assert table.meta_nominal.tolist() == list(TARGETS)
    first = table.iloc[0]
    assert first.diferenca_revisoes_mediana == -5
    assert first.reducao_percentual_mediana == 50
    assert first.proporcao_fusao_menor == 1
    assert table.iloc[2].proporcao_fusao_menor == 0 and table.iloc[2].diferenca_revisoes_mediana == 10
    assert table.proporcao_F4_menor_ambas.isna().all()


def test_existing_csv_columns_match_and_are_unchanged_by_import():
    existing = pd.read_csv("data/fusao_robustez_bootstrap.csv", nrows=0)
    assert list(existing.columns) == list(COLUMNS)


def test_run_bootstrap_is_deterministic_and_ic_ordered():
    queues, labels, codes, groups = synthetic()
    r1 = run_bootstrap(queues, labels, codes, groups, replicas=50)
    r2 = run_bootstrap(queues, labels, codes, groups, replicas=50)
    t1, t2 = summarize_o3_o1(r1), summarize_o3_o1(r2)
    pd.testing.assert_frame_equal(t1, t2)
    assert (t1.diferenca_revisoes_ic95_inf <= t1.diferenca_revisoes_mediana).all()
    assert (t1.diferenca_revisoes_mediana <= t1.diferenca_revisoes_ic95_sup).all()
    assert (t1.replicas_validas == len(r1)).all()


def test_replicates_without_positives_are_dropped():
    queues, _, codes, groups = synthetic()
    assert bootstrap_o3_o1_once(queues, np.zeros(300, dtype=bool), codes, groups,
                                np.random.default_rng(SEED)) is None
    assert run_bootstrap(queues, np.zeros(300, dtype=bool), codes, groups, replicas=3) == []
