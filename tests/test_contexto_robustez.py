"""Contratos portáveis: sem extrato privado nem retreino nos testes."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts import contexto_robustez as analysis
from scripts import fusao_robustez as fusao


def synthetic_queues(n):
    return {name: np.arange(n) for name in ("O1", "O3", "O3_MACD", "C", "CM")}


def test_cluster_expansion_and_hybrid_targets_match_imported_f4():
    queues = synthetic_queues(8)
    labels = np.ones(8, dtype=bool)
    codes = np.array([0, 0, 1, 1, 1, 2, 2, 2])

    class FixedRng:
        def integers(self, low, high, size):
            assert (low, high, size) == (0, 3, 3)
            return np.array([0, 0, 2])

    run = analysis.bootstrap_once(queues, labels, codes, 3, FixedRng())
    original = fusao.bootstrap_once(queues, labels, codes, 3, FixedRng())
    assert run[1:] == (7, 7)
    assert run[0]["CM"].tolist() == [2, 5, 7]
    for name in fusao.NAMES:
        np.testing.assert_array_equal(run[0][name], original[0][name])
    assert analysis.bootstrap_once(queues, ~labels, codes, 3, FixedRng()) is None


@pytest.mark.parametrize("total", [36, 40, 80])
def test_hybrid_uses_fixed_targets_at_or_above_36(total):
    run = analysis.bootstrap_once(synthetic_queues(total), np.ones(total, dtype=bool),
                                  np.arange(total), total, np.random.default_rng(analysis.SEED))
    assert run[0]["CM"].tolist() == [9, 23, 36]


def test_seeded_bootstrap_reuses_f4_exactly_with_nontrivial_orders():
    rng = np.random.default_rng(41)
    queues = {name: rng.permutation(80) for name in synthetic_queues(80)}
    labels = np.arange(80) % 3 == 0
    codes = np.repeat(np.arange(20), 4)
    for seed in range(5):
        run = analysis.bootstrap_once(queues, labels, codes, 20, np.random.default_rng(seed))
        old = fusao.bootstrap_once(queues, labels, codes, 20, np.random.default_rng(seed))
        assert run[1:] == old[1:]
        for name in fusao.NAMES:
            np.testing.assert_array_equal(run[0][name], old[0][name])


def test_difference_signs_percentile_interval_and_strict_probability():
    efforts = []
    for value in (5, 10, 15):
        efforts.append(({"CM": np.array([value] * 3), "C": np.array([15] * 3),
                         "O1": np.array([10] * 3), "O3": np.array([20] * 3),
                         "F4": np.array([12] * 3)}, 36, 100))
    table = analysis.summarize_bootstrap(efforts)
    assert len(table) == 15
    row = table.loc[table.comparacao.eq("CM-O1") & table.meta_nominal.eq(9)].iloc[0]
    assert row.diferenca_revisoes_mediana == 0
    assert row.diferenca_revisoes_ic95_inf == pytest.approx(-4.75)
    assert row.diferenca_revisoes_ic95_sup == pytest.approx(4.75)
    assert row.proporcao_diferenca_menor_zero == pytest.approx(1 / 3)


def test_cache_alignment_and_probability_validation():
    ids, folds = np.array(["a", "b"]), np.array([0, 1])
    analysis.validate_cache(ids, np.array([0., 1.]), folds, ids, folds)
    with pytest.raises(ValueError, match="desalinhado"):
        analysis.validate_cache(ids[::-1], np.array([0., 1.]), folds, ids, folds)
    with pytest.raises(ValueError, match="folds"):
        analysis.validate_cache(ids, np.array([0., 1.]), folds[::-1], ids, folds)
    for p in (np.array([np.nan, 1.]), np.array([0.]), np.array([-1., 1.])):
        with pytest.raises(ValueError, match="Probabilidades"):
            analysis.validate_cache(ids, p, folds, ids, folds)


def test_checkpoint_replace_retries_windows_lock(tmp_path, monkeypatch):
    source, destination = tmp_path / "temp", tmp_path / "cache"
    source.write_text("checkpoint", encoding="utf-8")
    replace = analysis.os.replace
    attempts, delays = [], []

    def locked_once(src, dst):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("locked")
        replace(src, dst)

    monkeypatch.setattr(analysis.os, "replace", locked_once)
    monkeypatch.setattr(analysis.time, "sleep", delays.append)
    analysis.replace_retry(source, destination)
    assert delays == [0.1, 0.2]
    assert destination.read_text(encoding="utf-8") == "checkpoint"


def test_portable_aggregate_outputs_and_provenance():
    root = analysis.ROOT
    table = pd.read_csv(root / "data/contexto_robustez_bootstrap.csv")
    band = pd.read_csv(root / "data/contexto_robustez_faixa.csv")
    meta = json.loads((root / "data/contexto_robustez_metadata.json").read_text(encoding="utf-8"))
    assert len(table) == 15 and len(band) == 18
    assert not table.duplicated(["comparacao", "meta_nominal"]).any()
    assert set(table.comparacao) == {f"{a}-{b}" for a, b in analysis.COMPARISONS}
    assert set(band.fila) == set(analysis.NAMES)
    assert meta["seed"] == 20261107 and meta["fit_seed"] == 42
    assert meta["replicas"] in (500, 2000)
    assert meta["replicas"] == (500 if meta["reduced_to_500"] else 2000)
    assert "fixed 9/23/36 if total>=36" in meta["targets_rule"]
    assert meta["cpu_only"] and meta["llm_calls"] == meta["bootstrap_model_fits"] == 0
    assert meta["faixa"]["n"] == 2522 and meta["faixa"]["PAR2"] == 36
    assert meta["faixa"]["baseline_F4_table_reproduced"]
    assert meta["baseline_reproduction"]["passed"]
    assert meta["grupos_REFREC"] == 1457 and meta["below_cutoff"] == 61485
    assert meta["PAR2"] == 36 and meta["cpu_limit_minutes"] == 60
    assert meta["inner_seeds"] == [43, 44, 45, 46, 47]
    assert meta["folds"] == 5 and meta["split_by"] == "refrec"
    if meta["replicas"] == 2000:
        assert meta["F4_bootstrap_reproduced"]
    for name, expected in analysis.EXPECTED.items():
        assert meta["context_reproduction"][name] == {
            "reviews": expected, "all_summary_metrics": True, "full_curve": True}
        assert meta["oof_cache"][name]["private_path"].startswith(".local/")
    assert (table.replicas_validas == meta["replicas"] - meta["replicas_sem_PAR2"]).all()
    assert table.proporcao_diferenca_menor_zero.between(0, 1).all()
    assert (table.diferenca_revisoes_ic95_inf <= table.diferenca_revisoes_mediana).all()
    assert (table.diferenca_revisoes_mediana <= table.diferenca_revisoes_ic95_sup).all()
    # Versioned aggregates contain no pair identifier, labels, scores or field values.
    assert set(table.columns) == {"comparacao", "meta_nominal", "fracao_se_menos_36", "replicas_validas",
                                  "diferenca_revisoes_mediana", "diferenca_revisoes_ic95_inf",
                                  "diferenca_revisoes_ic95_sup", "proporcao_diferenca_menor_zero"}
    assert set(band.columns) == {"fila", "meta", "revisoes", "diferenca_O1", "diferenca_O3", "diferenca_F4",
                                 "reducao_percentual_O1", "reducao_percentual_O3", "reducao_percentual_F4"}
    old = pd.read_csv(root / "data/fusao_robustez_faixa.csv")
    pd.testing.assert_frame_equal(band.loc[band.fila.isin(fusao.NAMES), old.columns].reset_index(drop=True), old)
    for name, expected in meta["output_sha256"].items():
        content = (root / "data" / name).read_bytes()
        assert b"\r\n" not in content
        assert hashlib.sha256(content).hexdigest() == expected
    for name, expected in meta["code_sha256"].items():
        # Portable even if unrelated historical source files check out as CRLF.
        raw = (root / name).read_bytes().replace(b"\r\n", b"\n")
        assert expected in {hashlib.sha256(raw).hexdigest(),
                            hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest()}
