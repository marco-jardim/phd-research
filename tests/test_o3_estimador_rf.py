"""Contratos sintéticos da sensibilidade de O3 ao estimador (sem o extrato real)."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from gzcmd.ml_oof import METHOD, SEED, _classifier, fit_xgb_oof, nested_oof_probabilities
from scripts.comparador_mesma_carga import MODES, load_raw
from scripts.fusao_robustez import SEED as BOOTSTRAP_SEED
from scripts.o3_estimador_rf import (
    COLUMNS,
    COMPARISONS,
    ESTIMATORS,
    EXPECTED,
    METHOD_RF,
    OUT_OOF,
    REFERENCE_NAMES,
    RF_PARAMS,
    ReproductionError,
    bootstrap_runs,
    build_universe,
    check_xgb_reproduction,
    estimate_minutes,
    evaluate_ordering,
    features_from_raw,
    long_table,
    oof_from_raw,
    oof_probabilities,
    rf_classifier,
    summarize_comparisons,
    verify_bootstrap_reference,
    verify_reference,
    write_oof_rf,
)
from scripts.ordenacao_bootstrap_o3_o1 import bootstrap_o3_o1_once
from scripts.ordenacao_empates_o3 import REFERENCE_MAP


def synthetic_xy(n=360, groups=72, seed=3):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    y[:12] = 1
    x = pd.DataFrame(rng.normal(size=(n, 4)) + y[:, None], columns=list("abcd"))
    g = np.array([f"g{i % groups}" for i in range(n)])
    return x, y, g


def synthetic_raw(n=360, groups=72, seed=3):
    x, y, g = synthetic_xy(n, groups, seed)
    raw = pd.DataFrame({
        "COMPREC": [str(i) for i in range(n)], "REFREC": g, "PASSO": ["6"] * n,
        "PAR": np.where(y == 1, "2", "0"),
        "NOME prim frag igual": x.a.round(4).astype(str).str.replace(".", ",", regex=False),
        "NOME qtd frag iguais": x.b.round(4).astype(str).str.replace(".", ",", regex=False),
        "DTNASC dt iguais": x.c.round(4).astype(str).str.replace(".", ",", regex=False),
        "nota final": x.d.round(4).astype(str).str.replace(".", ",", regex=False),
    })
    return raw


def synthetic_ranking(n=300, seed=11, positives=36, tied=False):
    rng = np.random.default_rng(seed)
    ids = np.array([f"{i:016x}" for i in rng.permutation(n)])
    below = np.ones(n, dtype=bool)
    below[:20] = False
    par2 = np.zeros(n, dtype=bool)
    par2[20 + rng.choice(n - 20, positives, replace=False)] = True
    score = np.zeros(n) if tied else rng.random(n) + par2 * 0.5
    groups = np.array([f"g{i % 50}" for i in range(n)])
    return ids, below, par2, groups, score


def reference_frame(name, values):
    rows = []
    for (mode, grandeza), value in values.items():
        load, column = REFERENCE_MAP[grandeza]
        row = {"ordenacao": REFERENCE_NAMES[name], "modo": mode, "carga": load, "k": 0, "par2_topk": 0}
        row[column] = value
        rows.append(row)
    return pd.DataFrame(rows)


# ---- estimador: padrão inalterado, RF só quando pedido ------------------------


def test_default_path_is_unchanged_xgboost():
    x, y, g = synthetic_xy()
    default = nested_oof_probabilities(x, y, split_by="refrec", groups=g)
    explicit = nested_oof_probabilities(x, y, split_by="refrec", groups=g, model_factory=_classifier)
    via_param = oof_probabilities(x, y, g, "xgb")
    for other in (explicit, via_param):
        assert np.array_equal(default[0], other[0])
        assert np.array_equal(default[1], other[1])
    assert ESTIMATORS["xgb"]["factory"] is None  # sem factory: ml_oof usa o padrão
    assert ESTIMATORS["xgb"]["method"] == METHOD == "nested-xgb-isotonic-5x5-v1"
    assert ESTIMATORS["rf"]["method"] == METHOD_RF == "nested-rf-isotonic-5x5-v1"


def test_xgb_fingerprint_and_probabilities_match_fit_xgb_oof_on_synthetic_fixture(tmp_path):
    raw = synthetic_raw()
    source = tmp_path / "extrato.csv"
    raw.to_csv(source, sep=";", index=False)
    result = fit_xgb_oof(source, output_path=tmp_path / "oof.csv", split_by="refrec")
    metadata = json.loads((tmp_path / "oof.json").read_text(encoding="utf-8"))
    assert metadata["fingerprint"]["method"] == METHOD
    assert metadata["fingerprint"]["seed"] == SEED == 42
    assert metadata["fingerprint"]["split_by"] == "refrec"
    x, y, g = features_from_raw(load_raw(source))
    assert metadata["features"] == list(x.columns)
    p, folds = oof_from_raw(load_raw(source), "xgb")
    assert np.array_equal(p, result.p_cal_ml.to_numpy())
    assert np.array_equal(folds, result.fold.to_numpy())
    # a reprodução exigida pelo script passa com tolerância 1e-9 neste fixture
    assert check_xgb_reproduction(p, folds, result.p_cal_ml, result.fold)["passed"]


def test_random_forest_only_when_requested_and_matches_nb01_params():
    x, y, g = synthetic_xy()
    params = rf_classifier(y, SEED).get_params()
    assert (params["n_estimators"], params["max_depth"], params["class_weight"],
            params["random_state"]) == (200, 10, "balanced", 42)
    p_xgb, f_xgb = oof_probabilities(x, y, g, "xgb")
    p_rf, f_rf = oof_probabilities(x, y, g, "rf")
    assert np.array_equal(f_xgb, f_rf)  # mesmas partições externas
    assert not np.allclose(p_xgb, p_rf)
    assert np.isfinite(p_rf).all() and ((p_rf >= 0) & (p_rf <= 1)).all()
    with pytest.raises(ValueError):
        oof_probabilities(x, y, g, "lightgbm")


def test_random_forest_oof_is_deterministic():
    x, y, g = synthetic_xy()
    first, _ = oof_probabilities(x, y, g, "rf")
    second, _ = oof_probabilities(x, y, g, "rf")
    assert np.array_equal(first, second)


def test_random_forest_predict_proba_is_bitwise_deterministic_and_restores_n_jobs():
    x, y, _ = synthetic_xy(n=2000, groups=400)
    model = rf_classifier(y, SEED).fit(x, y)
    first = model.predict_proba(x)
    for _ in range(5):
        assert np.array_equal(first, model.predict_proba(x))
    assert model.n_jobs == -1  # ajuste paralelo preservado; só a predição é serial
    assert rf_classifier(y, SEED).get_params()["n_jobs"] == -1
    serial = RandomForestClassifier(**RF_PARAMS, random_state=SEED, n_jobs=1).fit(x, y)
    assert np.array_equal(first, serial.predict_proba(x))  # fit paralelo = fit serial


def test_xgb_reproduction_check_fails_when_probabilities_or_folds_differ():
    p = np.array([0.1, 0.2, 0.3])
    folds = np.array([0, 1, 2])
    assert check_xgb_reproduction(p, folds, p + 5e-10, folds)["passed"]
    with pytest.raises(ReproductionError):
        check_xgb_reproduction(p, folds, p + 1e-6, folds)
    with pytest.raises(ReproductionError):
        check_xgb_reproduction(p, folds, p, np.array([0, 2, 1]))


# ---- métricas e referências -----------------------------------------------------


def test_evaluate_ordering_is_consistent_with_brute_force():
    ids, below, par2, _, score = synthetic_ranking()
    values = evaluate_ordering(ids, below, par2, score)
    by_id = np.argsort(ids[below], kind="stable")
    order = by_id[np.argsort(-score[below][by_id], kind="stable")]
    ranked = par2[below][order]
    positions = np.flatnonzero(ranked) + 1
    for mode, p in MODES.items():
        recovered = p["accepted"] + p["referred"]
        assert values[(mode, "k_min_equivalente_P")] == positions[recovered - 1]
        assert values[(mode, "k_min_36")] == positions[35]
        assert values[(mode, "par2_topk_K1_humana")] == ranked[: p["human"]].sum()
    assert 0.5 < values[("abaixo_corte", "auc_abaixo")] <= 1
    assert 0 < values[("abaixo_corte", "ap_abaixo")] <= 1


def test_ties_are_broken_by_lexical_pair_id_exactly_k_rows():
    ids, below, par2, _, score = synthetic_ranking(tied=True)
    values = evaluate_ordering(ids, below, par2, score)
    ranked = par2[below][np.argsort(ids[below])]
    assert values[("vigilancia", "par2_topk_K1_humana")] == ranked[:96].sum()


def test_verify_reference_accepts_expected_and_rejects_changes():
    for name in ("O1", "O3_XGB"):
        reference = reference_frame(name, EXPECTED[name])
        assert verify_reference(name, dict(EXPECTED[name]), reference)
        wrong = dict(EXPECTED[name])
        wrong[("vigilancia", "k_min_36")] += 1
        with pytest.raises(ReproductionError):
            verify_reference(name, wrong, reference)
        stale = reference_frame(name, {**EXPECTED[name], ("confirmacao", "k_min_36"): 1})
        with pytest.raises(ReproductionError):
            verify_reference(name, dict(EXPECTED[name]), stale)


def test_expected_values_are_the_thesis_ones():
    assert EXPECTED["O3_XGB"][("vigilancia", "k_min_equivalente_P")] == 125
    assert EXPECTED["O3_XGB"][("confirmacao", "k_min_equivalente_P")] == 29
    assert EXPECTED["O3_XGB"][("vigilancia", "k_min_36")] == 23443
    assert EXPECTED["O3_XGB"][("vigilancia", "par2_topk_K1_humana")] == 22
    assert EXPECTED["O3_XGB"][("confirmacao", "par2_topk_K1_humana")] == 6


# ---- bootstrap -------------------------------------------------------------------


def synthetic_queues(seed=5):
    ids, below, par2, groups, _ = synthetic_ranking()
    rng = np.random.default_rng(seed)
    scores = {name: rng.random(len(ids)) for name in ("O1", "O3_XGB", "O3_RF")}
    return build_universe(ids, below, par2, groups, scores)


def test_build_universe_returns_complete_permutations():
    queues, labels, codes, group_count = synthetic_queues()
    n = len(labels)
    assert n == 280 and labels.sum() == 36
    for q in queues.values():
        assert np.array_equal(np.sort(q), np.arange(n))
    assert len(codes) == n and codes.max() + 1 == group_count


def test_bootstrap_uses_the_same_draws_as_the_existing_o3_o1_bootstrap():
    queues, labels, codes, group_count = synthetic_queues()
    runs = bootstrap_runs(queues, labels, codes, group_count, 6, BOOTSTRAP_SEED)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for run in runs:
        old = bootstrap_o3_o1_once(
            {"O3": queues["O3_XGB"], "O1": queues["O1"]}, labels, codes, group_count, rng)
        assert run[1:] == old[1:]
        assert np.array_equal(run[0]["O3_XGB"], old[0]["O3"])
        assert np.array_equal(run[0]["O1"], old[0]["O1"])


def test_bootstrap_summary_is_deterministic_paired_and_signed():
    queues, labels, codes, group_count = synthetic_queues()
    queues["O3_RF"] = queues["O3_XGB"].copy()
    first = summarize_comparisons(bootstrap_runs(queues, labels, codes, group_count, 20, 1))
    second = summarize_comparisons(bootstrap_runs(queues, labels, codes, group_count, 20, 1))
    pd.testing.assert_frame_equal(first, second)
    assert set(first.comparacao) == {f"{a}-{b}" for a, b in COMPARISONS}
    same = first.loc[first.comparacao.eq("O3_RF-O3_XGB")]
    assert (same.diferenca_revisoes_mediana == 0).all()
    assert (same.proporcao_fusao_menor == 0).all()
    other = first.loc[first.comparacao.eq("O3_RF-O1")]
    assert (other.diferenca_revisoes_ic95_inf <= other.diferenca_revisoes_ic95_sup).all()


def test_bootstrap_reference_is_only_checked_with_default_replicas_and_seed(tmp_path):
    table = pd.DataFrame({"comparacao": ["O3_XGB-O1"]})
    assert not verify_bootstrap_reference(table, tmp_path / "ausente.csv", 2000, BOOTSTRAP_SEED)["checked"]
    assert not verify_bootstrap_reference(table, tmp_path / "ausente.csv", 10, BOOTSTRAP_SEED)["checked"]


# ---- saída ------------------------------------------------------------------------


def test_long_table_has_fixed_columns_and_no_missing_values():
    ids, below, par2, _, score = synthetic_ranking()
    metrics = {"O3_RF": evaluate_ordering(ids, below, par2, score)}
    queues, labels, codes, group_count = synthetic_queues()
    bootstrap = summarize_comparisons(bootstrap_runs(queues, labels, codes, group_count, 10, 1))
    table = long_table(metrics, bootstrap, None)
    assert tuple(table.columns) == COLUMNS
    assert set(table.bloco) == {"ordenacao", "bootstrap"}
    assert table.valor.notna().all() and (table.valor != "").all()
    boot = table.loc[table.bloco.eq("bootstrap") & table.estatistica.eq("proporcao_primeiro_menor")]
    assert len(boot) == len(COMPARISONS) * 3


def test_oof_rf_file_has_the_original_format_and_no_pii(tmp_path):
    original = pd.DataFrame({"pair_id": ["a", "b", "c"], "nota": ["1.0", "2.0", "3.0"],
                             "p_cal_platt": ["0.1", "0.2", "0.3"], "p_cal_ml": [0.0, 0.5, 1.0],
                             "fold": [0, 1, 2]})
    path = tmp_path / "pcal_ml_oof_rf.csv"
    fingerprint = {"method": METHOD_RF}
    write_oof_rf(path, original, np.array([0.2, 0.4, 0.6]), np.array([0, 1, 2]),
                 ["f1"], 3, fingerprint)
    saved = pd.read_csv(path)
    assert list(saved.columns) == ["pair_id", "nota", "p_cal_platt", "p_cal_ml", "fold"]
    assert saved.p_cal_ml.tolist() == [0.2, 0.4, 0.6]
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["fingerprint"]["method"] == "nested-rf-isotonic-5x5-v1"
    assert OUT_OOF != "data/pcal_ml_oof.csv"  # nunca sobrescreve o original


def test_estimate_minutes_reports_both_estimators():
    x, y, g = synthetic_xy()
    estimate = estimate_minutes(x, y, g, bootstrap_pilot=0.01, replicas=100)
    assert set(estimate["pilot_fit_seconds"]) == {"xgb", "rf"}
    assert estimate["nested_fits_per_estimator"] == 30
    assert estimate["estimated_minutes"]["total"] > 0
