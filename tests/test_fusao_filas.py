"""Contratos portáveis dos agregados exploratórios post hoc e teste local opt-in."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from local_data import DATA_ROOT, requires_files

from scripts.fusao_filas import (
    BASES,
    DEFINITIONS,
    METRICS,
    ROOT,
    TARGETS,
    TOP_K,
    VARIANTS,
    aggregate,
    load_queues,
    metrics,
    round_robin,
    rrf,
    thompson,
    validate_order,
    verify_baselines,
)


@pytest.mark.parametrize("n", [7, 61485])
def test_deduplication_complete_permutations(n):
    rng = np.random.default_rng(14)
    queues = [rng.permutation(n) for _ in range(3)]
    labels = np.zeros(n, dtype=bool)
    labels[: min(n, 36)] = True
    orders = [
        round_robin(queues[:2]),
        round_robin(queues),
        rrf(queues),
        rrf(queues[:2]),
    ]
    orders += [thompson(queues, labels, 0, window) for window in (200, None)]
    for order in orders:
        validate_order(order, n)
        assert len(np.unique(order)) == n
        curve = labels[order].cumsum()
        assert np.all(np.diff(curve) >= 0)
        assert curve[-1] == labels.sum()


def test_round_robin_and_rrf_ties():
    queues = [np.array([0, 1, 2, 3]), np.array([0, 2, 1, 3])]
    assert round_robin(queues).tolist() == [0, 2, 1, 3]
    assert rrf(queues).tolist() == [0, 1, 2, 3]


def test_adaptive_no_future_label_access():
    # O primeiro resultado não pode afetar a primeira decisão; depois de um
    # prefixo comum de recompensas, o prefixo de decisões também é idêntico.
    queues = [np.arange(100), np.arange(99, -1, -1), np.roll(np.arange(100), 50)]
    labels = np.zeros(100, dtype=bool)
    first = thompson(queues, labels, 19, 200)
    changed = labels.copy()
    changed[first[40:]] = True
    second = thompson(queues, changed, 19, 200)
    assert np.array_equal(first[:41], second[:41])
    assert np.array_equal(first, thompson(queues, labels, 19, 200))


def test_auc_and_quantile_definition():
    early = np.full(61485, 36)
    late = np.zeros(61485, dtype=int)
    late[3039:] = 36
    assert metrics(early)["auc_recuperacao_3040"] == 3039 / 3040
    assert metrics(late)["auc_recuperacao_3040"] == 0
    rows = pd.DataFrame(aggregate("A1", [early, late])).set_index("metrica")
    assert rows.loc["revisoes_36", "media"] == 1520.5
    assert rows.loc["revisoes_36", "p10"] == pytest.approx(304.9)


@pytest.mark.parametrize("window", [2, None])
def test_posterior_uses_only_selected_arm_and_its_window(monkeypatch, window):
    observed = []

    class FixedRng:
        def beta(self, alpha, beta):
            observed.append((alpha.copy(), beta.copy()))
            return np.array([0.9, 0.2, 0.1])

    monkeypatch.setattr(np.random, "default_rng", lambda seed: FixedRng())
    labels = np.array([True, False, True, True, False])
    queues = [np.arange(5)] * 3
    assert thompson(queues, labels, 0, window).tolist() == list(range(5))
    for step, (alpha, beta) in enumerate(observed):
        past = labels[max(0, step - window) if window is not None else 0 : step]
        assert alpha.tolist() == [1 + int(past.sum()), 1, 1]
        assert beta.tolist() == [1 + int((~past).sum()), 1, 1]


def test_aggregate_artifacts_and_reproduction():
    table = pd.read_csv(ROOT / "data/fusao_filas.csv")
    curve = pd.read_csv(ROOT / "data/fusao_filas_curva.csv")
    meta = json.loads(
        (ROOT / "data/fusao_filas_metadata.json").read_text(encoding="utf-8")
    )
    assert list(table) == ["fila", "metrica", "media", "p10", "p90"]
    assert list(curve) == ["fila", "k", "par2_acumulados"]
    assert len(table) == len(VARIANTS) * len(METRICS)
    assert not table.duplicated(["fila", "metrica"]).any()
    assert set(table.fila) == set(VARIANTS)
    assert meta["analysis"] == "exploratory_post_hoc" and not meta["registered"]
    assert meta["cpu_only"] and meta["llm_calls"] == 0
    assert meta["definitions"] == DEFINITIONS
    assert meta["fixed_a_priori_for_this_run"]
    assert meta["prior"] == {"alpha": 1, "beta": 1}
    assert meta["repetitions"] in (50, 200)
    assert meta["seeds"] == list(range(meta["repetitions"]))
    assert meta["targets"] == list(TARGETS) and meta["top_k"] == list(TOP_K)
    assert meta["baseline_reproduction"]["passed"]
    assert meta["below_cutoff"] == 61485 and meta["par2"] == 36
    curves = {}
    for name in VARIANTS:
        rows = curve.loc[curve.fila.eq(name)]
        assert np.array_equal(rows.k, np.arange(1, 61486))
        counts = rows.par2_acumulados.to_numpy()
        assert np.all(np.diff(counts) >= -1e-12)
        assert counts[-1] == 36 and counts.min() >= 0 and counts.max() <= 36
        values = table.loc[table.fila.eq(name)].set_index("metrica")
        assert set(values.index) == set(METRICS)
        assert (values.p10 <= values.p90).all()
        for k in TOP_K:
            assert values.loc[f"par2_top_{k}", "media"] == pytest.approx(counts[k - 1])
        assert values.loc["auc_recuperacao_3040", "media"] == pytest.approx(
            counts[:3039].sum() / (3040 * 36)
        )
        if name in BASES:
            curves[name] = counts
        if not name.startswith("A"):
            assert np.array_equal(values.media, values.p10)
            assert np.array_equal(values.media, values.p90)
            for metric, value in metrics(counts).items():
                assert values.loc[metric, "media"] == pytest.approx(value)
    assert verify_baselines(curves)["passed"]
    for name, expected in meta["output_sha256"].items():
        actual = hashlib.sha256(
            (ROOT / "data" / name).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        assert actual == expected


@requires_files(
    DATA_ROOT / "data/COMPARADORSEMIDENT.csv",
    DATA_ROOT / "data/pcal_ml_oof.csv",
    DATA_ROOT / "data/pcal_ml_oof.json",
    ROOT / "data/ablation_comprec_allowlist_metadata.json",
)
def test_local_reproduction_of_individual_queues():
    queues, labels, _, _ = load_queues(DATA_ROOT)
    curves = {name: labels[q].cumsum() for name, q in queues.items()}
    assert verify_baselines(curves)["passed"]
