import json

import numpy as np
import pandas as pd
import pytest
from local_data import DATA_ROOT, requires_files

from scripts.comparador_mesma_carga import (
    CURVE_COLUMNS,
    MAX_K,
    MODES,
    OUTPUTS,
    ROOT,
    SUMMARY_COLUMNS,
    cumulative_counts,
    generate,
    load_principal,
    metrics,
)


def test_deterministic_ties_and_inclusive_topk():
    ids = np.array(["c", "b", "a", "d"])
    scores = np.array([3.0, 2.0, 2.0, 1.0])
    labels = np.array([False, True, False, True])
    counts = cumulative_counts(ids, scores, labels)
    np.testing.assert_array_equal(counts, [0, 0, 1, 2])
    permutation = [2, 0, 3, 1]
    np.testing.assert_array_equal(
        counts,
        cumulative_counts(ids[permutation], scores[permutation], labels[permutation]),
    )
    assert metrics("test", 2, counts)["par2_topk"] == 0
    row = metrics("test", 3, counts)
    assert row["nao_vinculos_topk"] == 2
    assert row["sens_humano_perfeito"] == pytest.approx(212 / 247)
    assert row["sens_humano_05"] == pytest.approx(211.95 / 247)


def test_missing_principal_is_explicit(tmp_path):
    result, status = load_principal(tmp_path / "missing.npz", None, None, None, None)
    assert result is None
    assert status == {"available": False, "reason": "oof.npz absent"}


def test_aggregate_columns_and_monotonicity():
    summary = pd.read_csv(ROOT / "data" / OUTPUTS[0])
    curve = pd.read_csv(ROOT / "data" / OUTPUTS[1])
    metadata = json.loads((ROOT / "data" / OUTPUTS[2]).read_text(encoding="utf-8"))
    assert list(summary) == SUMMARY_COLUMNS
    assert list(curve) == CURVE_COLUMNS
    forbidden = {
        "pair_id",
        "COMPREC",
        "REFREC",
        "SCORE",
        "nota final",
        "PAR",
        "R_DTNASC",
    }
    for table in (summary, curve):
        assert not forbidden.intersection(table.columns)
        np.testing.assert_array_equal(
            table.par2_topk + table.nao_vinculos_topk, table.k
        )
        np.testing.assert_allclose(
            table.sens_humano_perfeito, (211 + table.par2_topk) / 247
        )
        np.testing.assert_allclose(
            table.sens_humano_05, (211 + 0.95 * table.par2_topk) / 247
        )
    for _, region in curve.groupby("ordenacao"):
        np.testing.assert_array_equal(region.k, np.arange(1, MAX_K + 1))
        for column in CURVE_COLUMNS[2:]:
            assert (region[column].diff().dropna() >= 0).all()
    o1 = curve.loc[curve.ordenacao.eq("O1_SCORE") & curve.k.eq(MAX_K)].iloc[0]
    assert o1.par2_topk == 36
    burden = pd.read_csv(ROOT / "data/carga_abaixo_corte.csv")
    global_row = burden.loc[burden.limiar.eq("global") & burden.passo.eq("TOTAL")].iloc[
        0
    ]
    assert global_row.pares_acima_limiar == o1.k
    assert global_row.par2_acima_limiar == o1.par2_topk
    assert global_row.nao_vinculos_acima_limiar == o1.nao_vinculos_topk
    assert metadata["registered"] is False
    assert metadata["analysis"] == "exploratory_post_hoc"
    assert metadata["O4"]["available"] == ("O4_P_recalibrada" in set(curve.ordenacao))
    assert "including all ties" in metadata["inclusion_convention"]
    assert summary.comparador_fp_revisor_perfeito.eq(0).all()
    for mode, p in MODES.items():
        region = summary.loc[summary.modo.eq(mode)]
        assert region.P_par2_recuperados.eq(p["accepted"] + p["referred"]).all()
        assert region.P_fp.eq(p["fp"]).all()
        assert region.loc[region.carga.eq("K1_humana"), "k"].eq(p["human"]).all()
        assert (
            region.loc[region.carga.eq("K2_total"), "k"].eq(p["human"] + p["llm"]).all()
        )
    for _, row in summary.loc[summary.carga.eq("k_min_equivalente_P")].iterrows():
        assert row.par2_topk == row.P_par2_recuperados
    for _, row in summary.loc[summary.carga.eq("k_min_36")].iterrows():
        assert row.par2_topk == 36


@requires_files(
    DATA_ROOT / "data/COMPARADORSEMIDENT.csv",
    DATA_ROOT / "data/pcal_ml_oof.csv",
    DATA_ROOT / "data/pcal_ml_oof.json",
)
def test_private_regeneration_identical_and_boundary(tmp_path):
    from gzcmd.arranjo_corte import below_cutoff
    from scripts.comparador_mesma_carga import load_raw, numeric

    metadata = json.loads((ROOT / "data" / OUTPUTS[2]).read_text(encoding="utf-8"))
    oof = ROOT / ".local/arranjo_corte/oof.npz"
    if metadata["O4"]["available"] and not oof.exists():
        pytest.skip("Private O4 cache unavailable for identical regeneration")
    _, _, regenerated = generate(
        DATA_ROOT, tmp_path, oof_path=oof, analysis_date=metadata["date"]
    )
    for name in OUTPUTS:
        expected = (ROOT / "data" / name).read_bytes().replace(b"\r\n", b"\n")
        assert (tmp_path / name).read_bytes() == expected
    assert regenerated["universe"]["nonlinks"] == 61449
    raw = load_raw(DATA_ROOT / "data/COMPARADORSEMIDENT.csv")
    mask = below_cutoff(raw)
    score, par = numeric(raw.SCORE), numeric(raw.PAR)
    threshold = score.loc[mask & par.eq(2)].min()
    # >= is intentional: the threshold-based service bound expands ties.
    assert int((mask & score.ge(threshold)).sum()) == MAX_K
    assert int((mask & score.gt(threshold)).sum()) < MAX_K


@pytest.mark.parametrize("corruption", ["pair_id", "fold", "B1", "P"])
def test_principal_rejects_identity_or_probability_mismatch(tmp_path, corruption):
    from gzcmd.arranjo_corte import sha256

    source, original_path = tmp_path / "source.csv", tmp_path / "original.csv"
    source.write_text("source", encoding="utf-8")
    original_path.write_text("original", encoding="utf-8")
    original = pd.DataFrame(
        {"pair_id": ["a", "b"], "fold": [0, 1], "p_cal_ml": [0.1, 0.2]}
    )
    s1 = tmp_path / "s1.json"
    s1.write_text(
        json.dumps(
            {
                "input_fingerprints": {
                    "data/COMPARADORSEMIDENT.csv": sha256(source),
                    "data/pcal_ml_oof.csv": sha256(original_path),
                    "gzcmd/arranjo_corte.py": sha256(ROOT / "gzcmd/arranjo_corte.py"),
                    "gzcmd/ml_oof.py": sha256(ROOT / "gzcmd/ml_oof.py"),
                }
            }
        ),
        encoding="utf-8",
    )
    cache = {
        "pair_id": np.array(["b", "a"]),
        "fold": np.array([1, 0]),
        "B1": np.array([0.2, 0.1]),
        "P": np.array([0.3, 0.4]),
    }
    path = tmp_path / "oof.npz"
    np.savez(path, **cache)
    result, status = load_principal(path, original, source, original_path, s1)
    assert status["available"]
    np.testing.assert_array_equal(result, [0.4, 0.3])
    cache[corruption][0] = {"pair_id": "z", "fold": 4, "B1": 0.8, "P": np.nan}[
        corruption
    ]
    np.savez(path, **cache)
    result, status = load_principal(path, original, source, original_path, s1)
    assert result is None
    assert not status["available"]
