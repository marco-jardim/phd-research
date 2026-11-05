from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gzcmd.arranjo_corte import (
    MODEL, RATE_COLUMNS, REMOVED_FIELDS, SUMMARY_COLUMNS, below_cutoff,
    cache_matches, crossfit_rates, estimate_rates, nested_cutoff_probabilities,
    numeric_features, pending_cost, summarize_actions,
)
from gzcmd.config import load_config
from gzcmd.ml_oof import nested_oof_probabilities, pair_ids
from gzcmd.reliability import load_band_rates

ROOT = Path(__file__).resolve().parents[1]


class Model:
    def fit(self, frame, labels):
        self.fitted = set(frame.index)

    def predict_proba(self, frame):
        assert self.fitted.isdisjoint(frame.index)
        p = np.clip(frame.signal.to_numpy(), .001, .999)
        return np.column_stack([1 - p, p])


def synthetic():
    groups = np.repeat(np.arange(100), 4)
    x = pd.DataFrame({"signal": np.arange(400) / 400})
    y = np.tile([0, 1, 0, 1], 100)
    return x, y, groups


def test_all_mask_exactly_reproduces_nested_oof(monkeypatch):
    from gzcmd.arranjo_corte import LogitPlatt

    def forbidden_platt(*args, **kwargs):
        raise AssertionError("Isotonic reproduction must not fit Platt")

    monkeypatch.setattr(LogitPlatt, "fit", forbidden_platt)
    x, y, groups = synthetic()
    expected, expected_fold = nested_oof_probabilities(
        x, y, groups=groups, split_by="refrec", model_factory=lambda *_: Model())
    observed, fold, metadata = nested_cutoff_probabilities(
        x, y, np.ones(len(y), dtype=bool), groups,
        model_factory=lambda *_: Model(), include_a3=False, include_platt=False)
    np.testing.assert_array_equal(observed["P"], expected)
    np.testing.assert_array_equal(observed["B1"], expected)
    np.testing.assert_array_equal(fold, expected_fold)
    assert all(m["principal"] == "isotonic" for m in metadata)


def test_actual_all_mask_matches_published_refrec_oof(monkeypatch):
    from gzcmd.arranjo_corte import LogitPlatt

    def forbidden_platt(*args, **kwargs):
        raise AssertionError("Real isotonic reproduction must not fit Platt")

    monkeypatch.setattr(LogitPlatt, "fit", forbidden_platt)
    from scripts.arranjo_corte_s1_simular import load_raw
    raw = load_raw(ROOT / "data/COMPARADORSEMIDENT.csv")
    x = numeric_features(raw)
    y = pd.to_numeric(raw.PAR).isin([1, 2]).astype(int).to_numpy()
    probabilities, fold, _ = nested_cutoff_probabilities(
        x, y, np.ones(len(y), dtype=bool), raw.REFREC.str.strip().to_numpy(),
        include_a3=False, include_platt=False)
    expected = pd.read_csv(ROOT / "data/pcal_ml_oof.csv").set_index("pair_id").loc[pair_ids(raw)]
    np.testing.assert_array_equal(fold, expected.fold.to_numpy())
    np.testing.assert_allclose(probabilities["P"], expected.p_cal_ml.to_numpy(), rtol=0, atol=1e-14)


def test_audit_no_outer_or_above_cutoff_calibrator_leakage():
    x, y, groups = synthetic()
    below = np.arange(len(y)) % 4 != 0
    events = []

    def audit(fold, stage, fit, predict, outer_test):
        assert set(groups[fit]).isdisjoint(groups[outer_test])
        assert set(groups[fit]).isdisjoint(groups[predict])
        if stage in {"calibrator", "A3_calibrator", "A3_inner"}:
            assert below[fit].all()
        events.append(stage)

    probabilities, _, _ = nested_cutoff_probabilities(
        x, y, below, groups, model_factory=lambda *_: Model(), audit=audit)
    assert events.count("calibrator") == 5
    assert events.count("A3_inner") == 25
    assert np.isfinite(probabilities["P"]).all()


def test_insufficient_par2_forces_principal_platt():
    x, y, groups = synthetic()
    par2 = np.zeros(len(y), dtype=bool)
    par2[np.flatnonzero(y)[:8]] = True
    probabilities, _, metadata = nested_cutoff_probabilities(
        x, y, np.ones(len(y), dtype=bool), groups, par2=par2,
        model_factory=lambda *_: Model(), include_a3=False)
    np.testing.assert_array_equal(probabilities["P"], probabilities["P-Platt"])
    assert all(m["principal"] == "platt" and m["calibration_par2"] < 10 for m in metadata)


def test_real_cutoff_partition():
    from scripts.arranjo_corte_s1_simular import load_raw
    raw = load_raw(ROOT / "data/COMPARADORSEMIDENT.csv")
    mask = below_cutoff(raw)
    assert int(mask.sum()) == 61485
    assert int((~mask).sum()) == 211
    altered = raw.copy()
    altered.loc[altered.index[0], "SCORE"] = "999"
    altered.loc[altered.index[0], "PAR"] = "0"
    with pytest.raises(ValueError, match="partition"):
        below_cutoff(altered)


@pytest.mark.parametrize("mode", ["vigilancia", "confirmacao"])
def test_single_fold_rates_equal_current_m3(mode):
    from scripts.arranjo_corte_s1_simular import cohort_sources, load_raw
    from gzcmd.bands import BandAssigner
    raw = load_raw(ROOT / "data/COMPARADORSEMIDENT.csv")
    cfg = load_config(ROOT / "gzcmd/gzcmd_v3_config.yaml")
    decisions = {}
    for source in cohort_sources(ROOT):
        if source["mode"] == mode:
            run = json.loads((ROOT / source["path"]).read_text(encoding="utf-8"))
            decisions.update({r["pair_id"]: r["decision"] for r in run["results"]})
    frame = pd.DataFrame({"pair_id": pair_ids(raw), "fold": 0,
                          "band": BandAssigner.from_config(cfg).assign(numeric_features(raw)["nota final"]),
                          "y": pd.to_numeric(raw.PAR).isin([1, 2]).astype(int)})
    frame["decision"] = frame.pair_id.map(decisions)
    frame = frame.loc[frame.decision.notna()]
    expected = load_band_rates(cfg.llm_review.error_rates_by_band, mode=mode,
                               source=ROOT / "gzcmd/error_rates_by_band_v4.yaml")
    assert crossfit_rates(frame, cfg.llm_review.error_rates_by_band, [0])[0] == expected


def test_rate_crossfit_excludes_own_fold_and_fixed_fallbacks():
    fixed = {"grey_low": {"e_fp": .08, "e_fn": .12, "e_unsure": .03}}
    frame = pd.DataFrame({"band": ["grey_low"] * 20, "fold": [0] * 10 + [1] * 10,
                          "y": [0] * 10 + [1] * 10, "decision": ["MATCH"] * 10 + ["UNSURE"] * 10})
    rates = crossfit_rates(frame, fixed, [0, 1])
    assert asdict(rates[0]["grey_low"]) == {"e_fp": .08, "e_fn": 0., "e_unsure": 1.}
    assert asdict(rates[1]["grey_low"]) == {"e_fp": 1., "e_fn": .12, "e_unsure": 0.}
    assert asdict(estimate_rates(frame.iloc[:9], fixed)["grey_low"]) == {
        "e_fp": .08, "e_fn": .12, "e_unsure": .03}


@pytest.mark.parametrize("field", ["prompt_sha256", "model", "dossier_sha256", "removed_fields"])
def test_cache_refuses_different_identity(field):
    run = {"mode": "vigilancia", "model": MODEL, "prompt_sha256": "prompt",
           "dossier_sha256": {"a": "dossier"}, "removed_fields": list(REMOVED_FIELDS),
           "complete": True, "prompt_version": "v4"}
    kwargs = {"mode": "vigilancia", "pair_id": "a", "prompt_sha256": "prompt", "dossier_sha256": "dossier"}
    assert cache_matches(run, run, **kwargs)
    altered = deepcopy(run)
    altered[field] = {} if field == "dossier_sha256" else "different"
    assert not cache_matches(altered, altered, **kwargs)


def test_outputs_have_no_labels_or_outcome_metrics():
    table = pd.DataFrame({"band": ["high", "high"], "action": ["MATCH", "NONMATCH"],
                          "PAR": [1, 0], "TARGET": [1, 0], "y": [1, 0]})
    summary = summarize_actions(table, "vigilancia", "P")
    assert list(summary) == SUMMARY_COLUMNS
    assert int(summary.n.sum()) == 2
    assert not {"PAR", "TARGET", "y", "FP", "FN", "sensitivity"}.intersection(RATE_COLUMNS)
    for name, columns in (("arranjo_corte_s1_resumo.csv", SUMMARY_COLUMNS),
                          ("arranjo_corte_taxas_crossfit.csv", RATE_COLUMNS)):
        path = ROOT / "data" / name
        if path.exists():
            assert list(pd.read_csv(path, nrows=0)) == columns
    for path in (ROOT / ".local/arranjo_corte").glob("pendentes_*.csv"):
        assert list(pd.read_csv(path, nrows=0)) == ["pair_id"]


def test_prereg_hash_mismatch_refuses_before_training():
    from scripts.arranjo_corte_s1_simular import verify_prereg
    with pytest.raises(ValueError, match="SHA256"):
        verify_prereg(ROOT, "wrong")


def test_published_profile_code_unchanged():
    import subprocess
    protected = ["gzcmd/ml_oof.py", "gzcmd/runner.py", "gzcmd/calibration.py",
                 "gzcmd/guardrails.py", "gzcmd/gzcmd_v3_config.yaml",
                 "gzcmd/error_rates_by_band_v4.yaml", "gzcmd/gzcmd_v3_policy_engine.py"]
    assert subprocess.check_output(["git", "diff", "b09e979", "--", *protected], cwd=ROOT) == b""


def test_cost_rule_includes_retry_margin():
    assert pending_cost(500) == pytest.approx(1.58202)
    assert pending_cost(501) > pending_cost(500)


def test_platt_perfect_separation_has_finite_parameters():
    from gzcmd.arranjo_corte import LogitPlatt
    p = np.concatenate([np.zeros(80), np.ones(20)])
    y = np.concatenate([np.zeros(80, dtype=int), np.ones(20, dtype=int)])
    calibrator = LogitPlatt().fit(p, y)
    assert calibrator.model is not None
    assert np.isfinite([calibrator.model.intercept, calibrator.model.slope]).all()
    prediction = calibrator.predict(p)
    assert np.isfinite(prediction).all()
    assert ((prediction > 0) & (prediction < 1)).all()
    np.testing.assert_allclose(prediction[:80], 1 / 82, atol=1e-5)
    np.testing.assert_allclose(prediction[80:], 21 / 22, atol=1e-5)


def test_s2_gate_refuses_stop_or_changed_pending(tmp_path, monkeypatch):
    import scripts.arranjo_corte_s2_executar as launcher
    from gzcmd.arranjo_corte import sha256
    local, data = tmp_path / "local", tmp_path / "data"
    local.mkdir()
    data.mkdir()
    monkeypatch.setattr(launcher, "LOCAL", local)
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    for mode in ("vigilancia", "confirmacao"):
        (local / f"pendentes_{mode}.csv").write_text("pair_id\n", encoding="utf-8")
    for name in ("arranjo_corte_taxas_crossfit.csv", "arranjo_corte_s1_resumo.csv"):
        (data / name).write_text("header\n", encoding="utf-8")
    metadata = {"pending_sha256": {m: sha256(local / f"pendentes_{m}.csv") for m in launcher.MODES},
                "pending_total": 0, "pending_P_union_B1_by_mode": {m: 0 for m in launcher.MODES},
                "stop_rule_triggered": False,
                "frozen_rates_sha256": sha256(data / "arranjo_corte_taxas_crossfit.csv"),
                "summary_sha256": sha256(data / "arranjo_corte_s1_resumo.csv")}
    assert launcher.gate(metadata) == {m: 0 for m in launcher.MODES}
    with pytest.raises(ValueError, match="STOP"):
        launcher.gate({**metadata, "stop_rule_triggered": True})
    (local / "pendentes_vigilancia.csv").write_text("pair_id\na\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Pending IDs"):
        launcher.gate(metadata)
