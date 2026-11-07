"""Contratos portáveis: contexto sem rótulos e agregados reproduzíveis."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts import contexto_obito as analysis
from scripts.fusao_filas import rrf


def fixture_inputs():
    x = pd.DataFrame({"nota final": [10., 10., 4., 7., 3.]}, index=[9, 4, 8, 1, 6])
    groups = {"refrec": pd.Series(["a", "a", "a", "b", "c"], index=x.index),
              "comprec": pd.Series(["x", "y", "x", "z", "z"], index=x.index)}
    return x, groups


def test_context_definition_ties_singletons_and_signed_differences():
    x, groups = fixture_inputs()
    context = analysis.context_features(x, groups)
    assert list(context) == list(analysis.CONTEXT_FEATURES)
    assert context.index.equals(x.index)
    assert context.ctx_refrec_n_candidatos.tolist() == [3, 3, 3, 1, 1]
    assert context.ctx_refrec_posicao.tolist() == [1, 1, 3, 1, 1]
    assert context.ctx_refrec_delta_melhor.tolist() == [0, 0, -6, 0, 0]
    assert context.ctx_refrec_delta_segundo.tolist() == [0, 0, -6, 0, 0]
    assert context.ctx_refrec_eh_melhor.tolist() == [1, 1, 0, 1, 1]
    assert context.ctx_comprec_n_candidatos.tolist() == [2, 1, 2, 2, 2]
    assert context.ctx_comprec_posicao.tolist() == [1, 1, 2, 1, 2]
    assert context.ctx_comprec_delta_melhor.tolist() == [0, 0, -6, 0, -4]
    assert context.ctx_comprec_delta_segundo.tolist() == [6, 0, 0, 4, 0]
    assert context.ctx_comprec_eh_melhor.tolist() == [1, 1, 0, 1, 0]
    assert np.isfinite(context.to_numpy()).all()


def test_context_invariant_to_label_score_identifier_values_and_row_order():
    x, groups = fixture_inputs()
    before = analysis.context_features(x, groups)
    changed = x.assign(PAR=[2, 0, 0, 2, 1], SCORE=[100, -20, 4, 5, -300],
                       C_SITUENCE=[1, 2, 3, 4, 5], REFREC=[9, 2, 7, 8, 1])
    renamed = {name: key.map(lambda s: "renamed_" + s) for name, key in groups.items()}
    pd.testing.assert_frame_equal(before, analysis.context_features(changed, renamed))
    reverse = changed.iloc[::-1]
    shuffled = analysis.context_features(reverse, {name: key.loc[reverse.index] for name, key in renamed.items()})
    pd.testing.assert_frame_equal(before, shuffled.loc[x.index])


def test_augmentation_preserves_original_features_and_ignores_forbidden_raw_fields():
    x, groups = fixture_inputs()
    macd = x.assign(macd_exemplo=[0., 1., 0., 1., -1.])
    raw = pd.DataFrame({"REFREC": " " + groups["refrec"] + " ",
                        "COMPREC": groups["comprec"], "PAR": [0, 1, 2, 0, 2],
                        "SCORE": [1., 2., 3., 4., 5.], "C_SITUENCE": [1, 1, 0, 0, 0]})
    data = {"raw": raw, "x": {"O3_controle": x.copy(), "O3_MACD": macd.copy()}}
    analysis.augment(data)
    expected = analysis.context_features(x, groups)
    pd.testing.assert_frame_equal(data["x"]["O3_controle"], x)
    pd.testing.assert_frame_equal(data["x"]["O3_MACD"], macd)
    pd.testing.assert_frame_equal(data["x"]["O3_contexto"], pd.concat([x, expected], axis=1))
    pd.testing.assert_frame_equal(data["x"]["O3_contexto_MACD"], pd.concat([macd, expected], axis=1))
    raw[["PAR", "SCORE", "C_SITUENCE"]] = 999
    analysis.augment(data)
    pd.testing.assert_frame_equal(data["x"]["O3_contexto"], pd.concat([x, expected], axis=1))


@pytest.mark.parametrize("invalid", ["missing", "empty", "misaligned", "nonfinite", "ambiguous"])
def test_reject_invalid_context_inputs(invalid):
    x, groups = fixture_inputs()
    if invalid == "missing":
        groups["refrec"].iloc[0] = None
    elif invalid == "empty":
        groups["comprec"].iloc[0] = " "
    elif invalid == "misaligned":
        groups["refrec"] = groups["refrec"].iloc[::-1]
    elif invalid == "nonfinite":
        x.iloc[0, 0] = np.inf
    else:
        x["nota_final"] = x["nota final"]
    with pytest.raises(ValueError):
        analysis.context_features(x, groups)


def test_strict_control_rejects_permissive_original_tolerance(monkeypatch):
    monkeypatch.setattr(analysis, "fit_control", lambda data: (None, None, {"max_abs_difference": 1e-10}))
    with pytest.raises(RuntimeError, match="precisão de máquina"):
        analysis.strict_control({})


def test_rrf_reuses_imported_order_preserves_ties_and_excludes_above_cutoff():
    ids = np.array(["d", "a", "c", "b", "e"])
    below = np.array([True, True, False, True, True])
    first = np.array([2., 4., 100., 3., 1.])
    second = np.array([3., 1., -100., 2., 4.])
    p = analysis.fused_scores(ids, below, [first, second])
    assert np.isnan(p[~below]).all()
    lexical = np.sort(ids[below])
    queues = [np.searchsorted(lexical, analysis.queue(ids, s, below)) for s in (first, second)]
    expected = lexical[rrf(queues)]
    assert np.array_equal(analysis.queue(ids, p, below), expected)
    assert p[0] == p[3] and p[1] == p[4]
    assert p[0] < p[1]


def test_fixed_specification_features_and_portable_output_contracts():
    root = analysis.ROOT
    meta = json.loads((root / "data/contexto_obito_metadata.json").read_text(encoding="utf-8"))
    spec = json.loads((root / "data/contexto_obito_spec.json").read_text(encoding="utf-8"))
    assert meta["definition"] == spec["definition"] == analysis.DEFINITION
    assert meta["specification"] == spec
    assert spec["fixed_before_data_and_fits_utc"] < meta["date_utc"]
    assert meta["cpu_only"] and meta["llm_calls"] == 0 and not meta["registered"]
    assert meta["targets"] == list(analysis.TARGETS) and meta["top_k"] == list(analysis.TOP_K)
    for name, digest in meta["output_sha256"].items():
        assert hashlib.sha256((root / "data" / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest
    if meta["status"] == "stopped_runtime_estimate":
        assert meta["estimate"]["estimated_minutes"] > meta["limit_minutes"] == 45
        assert meta["estimate"]["pilot_fits"] == len(meta["estimate"]["pilot_seconds"]) == 4
        assert meta["completed_nested_fits"] == 0
        assert meta["context_features"] == list(analysis.CONTEXT_FEATURES)
        assert not (root / "data/contexto_obito.csv").exists()
        assert not (root / "data/contexto_obito_curva.csv").exists()
        return
    assert meta["status"] == "completed"
    table = pd.read_csv(root / "data/contexto_obito.csv")
    curve = pd.read_csv(root / "data/contexto_obito_curva.csv")
    assert list(table) == ["variante", "metrica", "valor"]
    assert list(curve) == ["variante", "k", "par2_acumulados"]
    expected_variants = set(analysis.VARIANTS) - set(meta.get("skipped_variants", {}))
    assert set(table.variante) == expected_variants
    assert not table.duplicated(["variante", "metrica"]).any()
    assert meta["seed"] == 42 and meta["inner_seeds"] == [43, 44, 45, 46, 47]
    assert meta["split_by"] == "refrec" and meta["folds"] == 5
    assert meta["control_reproduction"]["max_abs_difference"] <= np.finfo(float).eps
    assert meta["control_reproduction"]["folds_identical"]
    assert meta["control_reproduction"]["queues_identical"]
    assert meta["baseline_reproduction"]["passed"]
    assert meta["estimate"]["model_fits"] == 90
    assert meta["cpu_seconds"] <= meta["cpu_limit_minutes"] * 60
    assert meta["reference_macd"].startswith("stored aggregate")
    assert meta["execution_order"][0]["variant"] == "O3_contexto"
    checkpoint = json.loads((root / "data/contexto_obito_etapa1.json").read_text(encoding="utf-8"))
    assert checkpoint["variant"] == "O3_contexto"
    assert "O3_contexto_MACD" not in {r["variante"] for r in checkpoint["metrics"]}
    assert "F4_contexto" in {r["variante"] for r in checkpoint["metrics"]}
    if "O3_contexto_MACD" not in meta.get("skipped_variants", {}):
        assert checkpoint["completed_utc"] < meta["execution_order"][1]["started_utc"]
    else:
        assert not meta["estimate"]["include_context_macd"] or meta["context_macd_projection_cpu_minutes"] > meta["cpu_limit_minutes"]
    assert [len(meta["features"][n]) for n in ("O3_controle", "O3_MACD", "O3_contexto", "O3_contexto_MACD")] == [30, 39, 40, 49]
    assert meta["features"]["O3_contexto"] == meta["features"]["O3_controle"] + list(analysis.CONTEXT_FEATURES)
    assert meta["features"]["O3_contexto_MACD"] == meta["features"]["O3_MACD"] + list(analysis.CONTEXT_FEATURES)
    for name in expected_variants:
        values = table.loc[table.variante.eq(name)].set_index("metrica").valor
        counts = curve.loc[curve.variante.eq(name)].sort_values("k")
        assert set(values.index) == set(analysis.METRICS)
        assert np.array_equal(counts.k, np.arange(1, 61486))
        assert counts.par2_acumulados.is_monotonic_increasing
        assert counts.par2_acumulados.iloc[-1] == 36
        for target in analysis.TARGETS:
            assert values[f"revisoes_{target}"] == counts.loc[counts.par2_acumulados.ge(target), "k"].iloc[0]
        for k in analysis.TOP_K:
            assert values[f"par2_top_{k}"] == counts.loc[counts.k.eq(k), "par2_acumulados"].iloc[0]
        assert 0 <= values.auc_abaixo <= 1 and 0 <= values.ap_abaixo <= 1
    assert analysis.verify_references(table, curve)["passed"]


def test_runtime_stop_records_no_fabricated_metrics(tmp_path):
    spec = {"definition": analysis.DEFINITION, "registered": False}
    analysis.write_json(tmp_path / "contexto_obito_spec.json", spec)
    estimate = {"pilot_seconds": [1., 2., 3., 8.], "estimated_minutes": 48.,
                "model_fits": 120, "pilot_fits": 4, "safety_factor": 3}
    metadata = analysis.record_estimate_stop(tmp_path, spec, estimate, report_path=tmp_path / "report.md")
    assert metadata["status"] == "stopped_runtime_estimate"
    assert metadata["completed_nested_fits"] == 0
    assert metadata["estimate"] == estimate
    assert not (tmp_path / "contexto_obito.csv").exists()
    assert not (tmp_path / "contexto_obito_curva.csv").exists()
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Não calculado" in report and "48,000000" in report


@pytest.mark.parametrize("limit,include_macd", [(30, False), (40, True)])
def test_cpu_projection_preserves_priority_context_before_optional_macd(monkeypatch, limit, include_macd):
    clocks = iter([0., 2., 2., 6., 6., 16., 16., 26.])
    monkeypatch.setattr(analysis.time, "process_time", lambda: next(clocks))
    monkeypatch.setattr(analysis, "estimate_minutes", lambda data: {"pilot_seconds": [2.]})
    data = {"x": {n: pd.DataFrame() for n in ("O3_controle", "O3_MACD", "O3_contexto", "O3_contexto_MACD")},
            "y": np.array([0, 1]), "groups": np.array(["a", "b"])}
    plan = analysis.estimate_plan(data, limit)
    assert plan["model_fits"] == 90
    assert plan["projected_total_cpu_minutes"] == pytest.approx((2 + 10 + 10) * 30 * 3 / 60 + 26 / 60)
    assert plan["include_context_macd"] == include_macd
    assert list(plan["pilot_cpu_by_matrix"]) == list(data["x"])
