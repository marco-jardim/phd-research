"""Contratos sintéticos portáveis e reprodução local opt-in da robustez."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from local_data import DATA_ROOT, requires_files

from scripts.fusao_robustez import (
    ROOT, SEED, TARGETS, bootstrap_once, fused_orders, random_tied_orders,
    resampled_queues, reviewable_mask, reviews, summarize_bootstrap, tie_audit, verify_reference,
)
from scripts.fusao_filas import load_queues, rrf, validate_order


def test_bootstrap_expands_cluster_copies_and_recomputes_rrf():
    queues = {"O1": np.array([0, 2, 1, 3]), "O3": np.array([2, 1, 3, 0]),
              "O3_MACD": np.array([3, 2, 0, 1])}
    expanded, original = resampled_queues(queues, np.array([2, 0, 2, 1]))
    assert original.tolist() == [0, 0, 2, 2, 3]
    assert expanded["O1"].tolist() == [0, 1, 2, 3, 4]
    assert expanded["O3"].tolist() == [2, 3, 4, 0, 1]
    for q in fused_orders(expanded).values():
        validate_order(q, 5)
    assert np.array_equal(fused_orders(expanded)["F4"], rrf([expanded["O3"], expanded["O1"]]))


def test_cluster_sampling_and_fractional_targets():
    queues = {n: np.arange(8) for n in ("O1", "O3", "O3_MACD")}
    labels = np.ones(8, dtype=bool)

    class FixedRng:
        def integers(self, low, high, size):
            assert (low, high, size) == (0, 3, 3)
            return np.array([0, 0, 2])

    run = bootstrap_once(queues, labels, np.array([0, 0, 1, 1, 1, 2, 2, 2]), 3, FixedRng())
    assert run[1:] == (7, 7)  # dois grupos 0, nenhum grupo 1, um grupo 2
    assert run[0]["F4"].tolist() == [2, 5, 7]
    empty = bootstrap_once(queues, ~labels, np.array([0, 0, 1, 1, 1, 2, 2, 2]), 3, FixedRng())
    assert empty is None


def test_bootstrap_fixed_targets_when_36_or_more():
    queues = {n: np.arange(40) for n in ("O1", "O3", "O3_MACD")}
    run = bootstrap_once(queues, np.ones(40, dtype=bool), np.arange(40), 40, np.random.default_rng(SEED))
    assert run[0]["F4"].tolist() == list(TARGETS)


def test_difference_and_reduction_signs_and_quantiles():
    effort = {"O1": np.array([10, 20, 40]), "O3": np.array([20, 40, 80]),
              "F4": np.array([5, 10, 20]), "F3": np.array([15, 30, 60])}
    table = summarize_bootstrap([(effort, 36, 100)] * 3)
    row = table.loc[(table.comparacao == "F4-O1") & (table.meta_nominal == 9)].iloc[0]
    assert row.diferenca_revisoes_mediana == -5
    assert row.reducao_percentual_mediana == 50
    assert row.proporcao_fusao_menor == row.proporcao_F4_menor_ambas == 1
    with pytest.raises(ValueError, match="Meta"):
        reviews({"O1": np.arange(2)}, np.array([True, False]), [2])


def test_random_ties_seed_and_no_label_access():
    scores = {"O1": np.ones(30), "O3": np.ones(30)}
    q, original = random_tied_orders(scores, np.random.default_rng(SEED))
    q2, original2 = random_tied_orders(scores, np.random.default_rng(SEED))
    assert np.array_equal(q, q2) and np.array_equal(original, original2)
    assert not np.array_equal(original[q], np.arange(30))
    audit = tie_audit(np.array([1., 1., 0.]), np.array([True, False, True]), np.array([0, 1]))
    assert audit["par2_em_empates"] == 1
    assert audit["posicao_relativa_media_par2_em_empates"] == 0


def test_operational_band_preserves_boundary_tie_block():
    scores = np.array([-5.647, -5.64846544302, -5.64846544302, -5.92366642673])
    assert reviewable_mask(scores).tolist() == [True, True, True, False]
    assert (scores >= -5.648).tolist() == [True, False, False, False]


def test_aggregate_outputs_portable():
    table = pd.read_csv(ROOT / "data/fusao_robustez_bootstrap.csv")
    faixa = pd.read_csv(ROOT / "data/fusao_robustez_faixa.csv")
    meta = json.loads((ROOT / "data/fusao_robustez_metadata.json").read_text(encoding="utf-8"))
    topo = json.loads((ROOT / "data/fusao_robustez_topo.json").read_text(encoding="utf-8"))
    assert len(table) == 12 and len(faixa) == 12
    assert not table.duplicated(["comparacao", "meta_nominal"]).any()
    assert meta["seed"] == SEED and meta["replicas"] in (500, 2000)
    assert meta["baseline_reproduction"]["passed"]
    assert meta["cpu_only"] and meta["llm_calls"] == 0
    assert meta["faixa"]["literal_n"] == 2495 and meta["faixa"]["literal_PAR2"] == 31
    assert meta["faixa"]["operational_n"] == 2522 and meta["faixa"]["operational_PAR2"] == 36
    assert (table.replicas_validas == meta["replicas"] - meta["replicas_sem_PAR2"]).all()
    assert ((table.proporcao_fusao_menor >= 0) & (table.proporcao_fusao_menor <= 1)).all()
    for name in ("diferenca_revisoes", "reducao_percentual"):
        assert (table[f"{name}_ic95_inf"] <= table[f"{name}_mediana"]).all()
        assert (table[f"{name}_mediana"] <= table[f"{name}_ic95_sup"]).all()
    assert topo["n"] == 17 and topo["par2"] == 16
    assert set(topo["empates"]) == {"O1", "O3", "F4_RRF"}
    assert topo["desempate_aleatorio"]["seed"] == SEED + 1
    assert topo["desempate_aleatorio"]["repeticoes"] == 200
    for path, expected in meta["output_sha256"].items():
        content = (ROOT / "data" / path).read_bytes().replace(b"\r\n", b"\n")
        assert expected in {hashlib.sha256(content).hexdigest(),
                            hashlib.sha256(content.replace(b"\n", b"\r\n")).hexdigest()}
    assert "pair_id" not in json.dumps(topo)


@requires_files(DATA_ROOT / "data/COMPARADORSEMIDENT.csv", DATA_ROOT / "data/pcal_ml_oof.csv",
                DATA_ROOT / "data/pcal_ml_oof.json")
def test_local_reproduction():
    queues, labels, _, _ = load_queues(DATA_ROOT)
    assert verify_reference(queues, labels)["passed"]
