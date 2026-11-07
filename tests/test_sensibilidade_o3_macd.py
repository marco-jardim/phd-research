"""Contratos portáveis dos agregados e reprodução local do controle O3."""
import json
import hashlib

import numpy as np
import pandas as pd
import pytest

from scripts.sensibilidade_o3_macd import (
    METRICS, PREFIX, ROOT, TARGETS, TOP_K, VARIANTS,
    check_control, fit_control, prepare,
)
from local_data import DATA_ROOT, requires_files


def test_csv_schema_targets_and_privacy():
    table = pd.read_csv(ROOT / "data" / f"{PREFIX}.csv")
    curve = pd.read_csv(ROOT / "data" / f"{PREFIX}_curva.csv")
    metadata = json.loads((ROOT / "data" / f"{PREFIX}_metadata.json").read_text(encoding="utf-8"))
    assert list(table) == ["variante", "metrica", "valor"]
    assert list(curve) == ["variante", "k", "par2_acumulados"]
    assert set(table.variante) == set(VARIANTS)
    assert not table.duplicated(["variante", "metrica"]).any()
    assert metadata["targets"] == list(TARGETS)
    assert metadata["top_k"] == list(TOP_K)
    assert metadata["below_cutoff"] == 61485
    assert metadata["par2"] == 36
    assert metadata["llm_calls"] == 0
    assert metadata["cpu_only"] and not metadata["registered"]
    assert len(metadata["features"]["O3_controle"]) == 30
    assert len(metadata["features"]["O3_MACD"]) == 39
    assert metadata["control_reproduction"]["max_abs_difference"] <= 1e-6
    assert metadata["control_reproduction"]["queues_identical"]
    assert metadata["control_reproduction"]["folds_identical"]
    for name, expected in metadata["output_sha256"].items():
        # Git pode normalizar LF/CRLF: o hash canônico é sempre o de LF.
        actual = hashlib.sha256((ROOT / "data" / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert actual == expected
    for variant in VARIANTS:
        values = table.loc[table.variante.eq(variant)].set_index("metrica").valor
        assert set(values.index) == set(METRICS)
        counts = curve.loc[curve.variante.eq(variant)]
        assert np.array_equal(counts.k, np.arange(1, 61486))
        assert counts.par2_acumulados.is_monotonic_increasing
        assert counts.par2_acumulados.iloc[-1] == 36
        for target in TARGETS:
            assert values[f"revisoes_{target}"] == counts.loc[counts.par2_acumulados.ge(target), "k"].iloc[0]
        for k in TOP_K:
            assert values[f"par2_top_{k}"] == counts.loc[counts.k.eq(k), "par2_acumulados"].iloc[0]
        for metric in ("auc_inteira", "ap_inteira", "auc_abaixo", "ap_abaixo"):
            assert 0 <= values[metric] <= 1
    o1 = table.loc[table.variante.eq("O1_SCORE")].set_index("metrica").valor
    assert [o1[f"revisoes_{target}"] for target in TARGETS] == [51, 501, 2521]


@requires_files(DATA_ROOT / "data/COMPARADORSEMIDENT.csv", DATA_ROOT / "data/pcal_ml_oof.csv",
                ROOT / "data/ablation_comprec_allowlist_metadata.json")
def test_control_reproduces_original_oof():
    data = prepare(DATA_ROOT)
    _, _, reproduction = fit_control(data)
    assert reproduction["max_abs_difference"] <= 1e-6
    assert reproduction["correlation"] > 1 - 1e-12
    assert reproduction["queues_identical"] and reproduction["folds_identical"]


@pytest.mark.parametrize("scores,folds", [([0.1, 0.2], [1, 0]), ([0.2, 0.1], [0, 1])])
def test_control_rejects_fold_mismatch(scores, folds):
    data = {"original": pd.DataFrame({"p_cal_ml": [0.1, 0.2], "fold": [0, 1]}),
            "ids": np.array(["a", "b"]), "below": np.array([True, True])}
    with pytest.raises(RuntimeError, match="Controle não reproduziu"):
        check_control(data, np.array(scores), np.array(folds))
