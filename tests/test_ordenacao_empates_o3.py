"""Contratos da sensibilidade de O3 ao desempate (sintéticos + extrato local)."""

import json

import numpy as np
import pandas as pd
import pytest
from local_data import DATA_ROOT, requires_files

from scripts.comparador_mesma_carga import MODES
from scripts.ordenacao_empates_o3 import (
    COLUMNS,
    OUT_CSV,
    REFERENCE,
    REFERENCE_MAP,
    SEED,
    ReproductionError,
    generate,
    quantities,
    random_tie_counts,
    tie_summary,
    verify_original,
)

ORIGINAL = {
    ("vigilancia", "k_min_equivalente_P"): 125,
    ("vigilancia", "k_min_36"): 23443,
    ("vigilancia", "par2_topk_K1_humana"): 22,
    ("confirmacao", "k_min_equivalente_P"): 29,
    ("confirmacao", "k_min_36"): 23443,
    ("confirmacao", "par2_topk_K1_humana"): 6,
}


def test_seed_is_the_registered_one():
    assert SEED == 20261108


def test_random_ties_keep_block_order_and_are_deterministic():
    score = np.array([0.9, 0.5, 0.5, 0.5, 0.1, 0.1])
    par2 = np.array([False, True, False, False, True, False])
    a = random_tie_counts(score, par2, np.random.default_rng(1))
    b = random_tie_counts(score, par2, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    # Blocos não mudam de posição: o primeiro é 0,9; o bloco 0,5 ocupa 2..4.
    assert a[0] == 0 and a[3] == 1 and a[-1] == 2
    for seed in range(20):
        c = random_tie_counts(score, par2, np.random.default_rng(seed))
        assert c[0] == 0 and c[3] == 1 and c[-1] == 2


def test_tie_summary_counts_only_blocks_with_two_or_more():
    assert tie_summary(np.array([3.0, 2.0, 2.0, 1.0, 1.0, 1.0])) == (2, 3)
    assert tie_summary(np.array([3.0, 2.0, 1.0])) == (0, 1)


def test_quantities_use_comparator_targets():
    counts = np.zeros(30000, dtype=int)
    counts[1000:] = 36
    counts[200:1000] = 20
    counts[:200] = 5
    result = quantities(counts)
    for mode, p in MODES.items():
        recovered = p["accepted"] + p["referred"]
        assert result[(mode, "k_min_equivalente_P")] == int(np.argmax(counts >= recovered)) + 1
        assert result[(mode, "k_min_36")] == 1001
        assert result[(mode, "par2_topk_K1_humana")] == int(counts[p["human"] - 1])
    with pytest.raises(ValueError):
        quantities(np.full(30000, 35))


def test_verify_original_fails_when_it_does_not_match():
    rows = []
    for (mode, name), value in ORIGINAL.items():
        load, _ = REFERENCE_MAP[name]
        rows.append({"ordenacao": "O3_p_cal_ml", "modo": mode, "carga": load,
                     "k": value, "par2_topk": value})
    reference = pd.DataFrame(rows)
    verify_original(ORIGINAL, reference)
    reference.loc[0, "k"] += 1
    with pytest.raises(ReproductionError):
        verify_original(ORIGINAL, reference)


def test_reference_csv_holds_the_published_values():
    reference = pd.read_csv(REFERENCE)
    verify_original(ORIGINAL, reference)


@requires_files(
    DATA_ROOT / "data/COMPARADORSEMIDENT.csv",
    DATA_ROOT / "data/pcal_ml_oof.csv",
    DATA_ROOT / "data/pcal_ml_oof.json",
)
def test_small_run_reproduces_original_and_is_deterministic():
    first, _ = generate(DATA_ROOT, replicas=5, seed=SEED, write=False)
    second, _ = generate(DATA_ROOT, replicas=5, seed=SEED, write=False)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == list(COLUMNS)
    got = {(r.modo, r.grandeza): r.valor_original for r in first.itertuples()}
    assert got == ORIGINAL
    assert (first.n_replicas == 5).all() and (first.seed == SEED).all()


def test_versioned_csv_columns_and_order():
    if not OUT_CSV.exists():
        pytest.skip("data/ordenacao_empates_o3.csv ainda não gerado")
    table = pd.read_csv(OUT_CSV)
    assert list(table.columns) == list(COLUMNS)
    assert len(table) == len(ORIGINAL)
    assert (table.minimo <= table.p2_5).all()
    assert (table.p2_5 <= table.mediana).all()
    assert (table.mediana <= table.p97_5).all()
    assert (table.p97_5 <= table.maximo).all()
    assert (table.minimo <= table.mediana).all() and (table.mediana <= table.maximo).all()
    got = {(r.modo, r.grandeza): r.valor_original for r in table.itertuples()}
    assert got == ORIGINAL
    assert (table.seed == SEED).all()
    assert table.blocos_empatados.nunique() == 1 and table.maior_bloco.nunique() == 1
    forbidden = {"pair_id", "COMPREC", "REFREC", "SCORE", "PAR"}
    assert not forbidden & set(table.columns)
    metadata = json.loads(
        OUT_CSV.with_name(OUT_CSV.stem + "_metadata.json").read_text(encoding="utf-8"))
    assert metadata["seed"] == SEED and metadata["n_replicas"] == int(table.n_replicas.iloc[0])
