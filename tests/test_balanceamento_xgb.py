"""Contratos sintéticos da sensibilidade ao desbalanceamento (RF e XGBoost); sem o extrato real."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from scripts.balanceamento_xgb import (
    COLUMNS,
    ESTIMATORS,
    STRATEGIES,
    ReproductionError,
    Strategy,
    aggregate,
    check_reproduction,
    class_ratio,
    estimator_description,
    fingerprints_match,
    load_partial,
    make_partitions,
    nadeau_bengio,
    resample_train,
    rf_factory,
    run_experiment,
    run_partition,
    vpp_zero_counts,
    xgb_factory,
)

SMOTE03 = next(s for s in STRATEGIES if s.key == "SMOTE 0.3")
CW_SMOTE = next(s for s in STRATEGIES if s.key == "Class weight + SMOTE 0.3")
NONE = next(s for s in STRATEGIES if s.key == "No balancing")
SUBSET = tuple(s for s in STRATEGIES if s.key in (
    "No balancing", "Class weight=balanced", "SMOTE 0.3", "SMOTETomek 0.3"))


def synthetic_xy(n=900, seed=11):
    # Prevalência ~4-5% (minoria/maioria < 0,3, exigido por SMOTE 0,3) e ~35 positivos
    # por treino de 3 partições, bastante para k_neighbors=3.
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.02).astype(int)
    y[:24] = 1
    assert 0.03 < y.mean() < 0.06
    X = rng.normal(size=(n, 6)) + 0.9 * y[:, None] * np.array([1, 1, 0.5, 0.5, 0, 0])
    return X, y


class Stub:
    """Estimador falso que registra o que viu; probabilidade = sigmoide da 1ª coluna."""

    def __init__(self, log, name, weighted, y_fit):
        self.log, self.name = log, name
        self.log.setdefault("factory", []).append((name, weighted, class_ratio(y_fit)))

    def fit(self, X, y):
        self.log.setdefault("fit", []).append((self.name, X.copy(), y.copy()))
        return self

    def predict_proba(self, X):
        self.log.setdefault("predict", []).append((self.name, X.copy()))
        p = 1.0 / (1.0 + np.exp(-X[:, 0]))
        return np.column_stack([1 - p, p])


def stub_estimators(log, names=("rf", "xgb")):
    return {n: (lambda y_fit, weighted, seed, n=n: Stub(log, n, weighted, y_fit)) for n in names}


def historical_from(records, strategies):
    frame = pd.DataFrame(records)
    frame = frame[(frame.estimador == "rf") & (frame.rep == 0)]
    rows = []
    for s in strategies:
        sub = frame[frame.estrategia == s.key]
        rows.append({
            "strategy": s.key,
            "prec_mean": sub.vpp.mean(), "prec_std": sub.vpp.std(ddof=1),
            "rec_mean": sub.sensibilidade.mean(), "rec_std": sub.sensibilidade.std(ddof=1),
            "f1_mean": sub.f1.mean(), "f1_std": sub.f1.std(ddof=1), "n_folds": len(sub),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- reamostragem


def test_resampling_only_touches_training_rows():
    X, y = synthetic_xy()
    train, test = next(StratifiedKFold(3, shuffle=True, random_state=1).split(X, y))
    log = {}
    run_partition(X, y, train, test, 42, stub_estimators(log, ("rf",)), strategies=(SMOTE03,))
    (_, X_fit, y_fit), = log["fit"]
    (_, X_pred), = log["predict"]
    assert len(X_fit) > len(train)  # linhas sintéticas só no treino
    assert np.array_equal(X_fit[: len(train)], X[train])
    assert np.array_equal(X_pred, X[test])  # teste intacto
    fitted = {tuple(r) for r in X_fit}
    assert not any(tuple(r) in fitted for r in X[test])  # nenhuma linha de teste no ajuste
    X_train = X[train]  # indexação avançada cria cópia nova a cada acesso
    assert resample_train(X_train, y[train], NONE, 42)[0] is X_train


def test_weighted_strategies_use_class_ratio_of_fitted_training_labels():
    X, y = synthetic_xy()
    train, test = next(StratifiedKFold(3, shuffle=True, random_state=1).split(X, y))
    log = {}
    run_partition(X, y, train, test, 42, stub_estimators(log, ("xgb",)),
                  strategies=(NONE, STRATEGIES[1], CW_SMOTE))
    flags = {w: r for _, w, r in log["factory"]}
    assert flags[True] == pytest.approx(1 / 0.3, rel=0.02)  # razão após SMOTE 0,3
    ratios = [(w, r) for _, w, r in log["factory"]]
    assert ratios[0][0] is False and ratios[1][0] is True
    assert ratios[1][1] == pytest.approx(class_ratio(y[train]))  # ponderação sem reamostragem


def test_scale_pos_weight_and_class_weight():
    y = np.array([0] * 90 + [1] * 10)
    assert class_ratio(y) == 9.0
    assert xgb_factory(y, True, 7).get_params()["scale_pos_weight"] == pytest.approx(9.0)
    assert xgb_factory(y, False, 7).get_params()["scale_pos_weight"] == 1.0
    model = xgb_factory(y, False, 7).get_params()
    assert (model["n_estimators"], model["max_depth"], model["learning_rate"]) == (300, 6, 0.05)
    assert rf_factory(y, True, 7).get_params()["class_weight"] == "balanced"
    assert rf_factory(y, False, 7).get_params()["class_weight"] is None
    assert rf_factory(y, False, 7).get_params()["n_estimators"] == 150


# --------------------------------------------------------------------------- partições


def test_partitions_are_shared_and_follow_historical_seed():
    X, y = synthetic_xy()
    parts = make_partitions(y, reps=2, seed=42, n_splits=3)
    assert len(parts) == 6
    again = make_partitions(y, reps=2, seed=42, n_splits=3)
    assert all(np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3])
               for a, b in zip(parts, again))
    hist = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X, y))
    for (rep, fold, train, test), (h_train, h_test) in zip(parts[:3], hist):
        assert rep == 0 and np.array_equal(test, h_test) and np.array_equal(train, h_train)
    assert not np.array_equal(parts[0][3], parts[3][3])  # repetição 1 usa outra seed


def test_both_estimators_see_identical_partitions_and_resamples():
    X, y = synthetic_xy()
    log = {}
    run_experiment(X, y, reps=2, n_splits=3, estimators=stub_estimators(log),
                   strategies=SUBSET, historical=None)
    by_name = {n: [(a.tobytes(), b.tobytes()) for m, a, b in log["fit"] if m == n]
               for n in ("rf", "xgb")}
    assert by_name["rf"] == by_name["xgb"]
    preds = {n: [a.tobytes() for m, a in log["predict"] if m == n] for n in ("rf", "xgb")}
    assert preds["rf"] == preds["xgb"]
    assert len(by_name["rf"]) == 2 * 3 * len(SUBSET)


# --------------------------------------------------------------------------- Nadeau e Bengio


def test_nadeau_bengio_matches_hand_calculation():
    d = [0.1, 0.3, 0.2, 0.4, 0.0]
    # média 0,2; desvios -0,1, 0,1, 0, 0,2, -0,2; soma de quadrados 0,10; s^2 = 0,10/4 = 0,025
    # fator 1/5 + 1/4 = 0,45; variância corrigida da média 0,01125; erro-padrão sqrt(0,01125)
    se = math.sqrt(0.01125)
    t_crit = 2.7764451052  # t_{0,975; 4}, tabela
    r = nadeau_bengio(d, n_test=1.0, n_train=4.0)
    assert r["diferenca_media"] == pytest.approx(0.2)
    assert r["dp_diferenca"] == pytest.approx(math.sqrt(0.025))
    assert r["se"] == pytest.approx(se, rel=1e-12)
    assert r["t"] == pytest.approx(0.2 / se, rel=1e-12)
    assert r["gl"] == 4
    assert r["ic95_inf"] == pytest.approx(0.2 - t_crit * se, rel=1e-8)
    assert r["ic95_sup"] == pytest.approx(0.2 + t_crit * se, rel=1e-8)
    # p bilateral de t = 1,8856 com 4 gl, por outra rota (tabela t: 0,10 < p < 0,20)
    assert 0.10 < r["p"] < 0.20
    # o IC corrigido é mais largo que o IC ingênuo (sem o termo n_test/n_train)
    naive = nadeau_bengio(d, n_test=0.0, n_train=1.0)
    assert (r["ic95_sup"] - r["ic95_inf"]) > (naive["ic95_sup"] - naive["ic95_inf"])


def test_nadeau_bengio_degenerate_cases():
    zero = nadeau_bengio([0.0, 0.0, 0.0], 1.0, 4.0)
    assert zero["t"] == 0.0 and zero["p"] == 1.0 and zero["ic95_inf"] == 0.0
    const = nadeau_bengio([0.1, 0.1, 0.1], 1.0, 4.0)
    assert math.isinf(const["t"]) and const["p"] == 0.0
    with pytest.raises(ValueError):
        nadeau_bengio([0.1], 1.0, 4.0)


# --------------------------------------------------------------------------- reprodução


def test_reproduction_passes_and_fails_without_adjusting():
    X, y = synthetic_xy()
    records, _ = run_experiment(X, y, reps=1, n_splits=3, estimators={"rf": ESTIMATORS["rf"]},
                                strategies=SUBSET, historical=None)
    hist = historical_from(records, SUBSET)
    ok = check_reproduction(records, hist, strategies=SUBSET)
    assert ok["passou"] and ok["max_diferenca_abs"] <= 1e-12
    bad = hist.copy()
    bad.loc[0, "f1_mean"] += 0.01
    with pytest.raises(ReproductionError, match="No balancing"):
        check_reproduction(records, bad, strategies=SUBSET)
    missing = hist.iloc[1:]
    with pytest.raises(ReproductionError, match="ausente"):
        check_reproduction(records, missing, strategies=SUBSET)


def test_run_aborts_before_any_xgboost_fit_when_history_does_not_reproduce():
    X, y = synthetic_xy()
    log = {}
    estimators = stub_estimators(log)
    records, _ = run_experiment(X, y, reps=1, n_splits=3, estimators={"rf": estimators["rf"]},
                                strategies=SUBSET, historical=None)
    bad = historical_from(records, SUBSET)
    bad["f1_mean"] += 0.5
    log.clear()
    with pytest.raises(ReproductionError):
        run_experiment(X, y, reps=2, n_splits=3, estimators=estimators, strategies=SUBSET,
                       historical=bad)
    assert {m for m, *_ in log["fit"]} == {"rf"}  # nenhum ajuste do XGBoost


def test_run_reuses_checkpoint_records():
    X, y = synthetic_xy()
    log = {}
    first, _ = run_experiment(X, y, reps=1, n_splits=3, estimators=stub_estimators(log),
                              strategies=SUBSET, historical=None)
    count = len(log["fit"])
    again, _ = run_experiment(X, y, reps=1, n_splits=3, estimators=stub_estimators(log),
                              strategies=SUBSET, historical=None, records=first)
    assert len(log["fit"]) == count and len(again) == len(first)


# --------------------------------------------------------------------------- determinismo e saída


def test_two_runs_are_bit_identical_and_output_is_aggregate_only():
    X, y = synthetic_xy()
    kwargs = dict(reps=2, n_splits=3, strategies=SUBSET, historical=None)
    first, _ = run_experiment(X, y, **kwargs)
    second, _ = run_experiment(X, y, **kwargs)
    assert first == second  # floats idênticos bit a bit
    table, ratio = aggregate(first, strategies=SUBSET)
    again, _ = aggregate(second, strategies=SUBSET)
    assert table.equals(again)
    assert ratio == pytest.approx(0.5)  # 3 partições: teste/treino = 1/2
    assert list(table.columns) == list(COLUMNS)
    assert not {"pair_id", "COMPREC", "REFREC"} & set(table.columns)
    assert set(table.tipo) == {"desempenho", "dif_sem_balanceamento", "dif_xgb_menos_rf"}
    perf = table[table.tipo == "desempenho"]
    assert len(perf) == 2 * len(SUBSET) * 4 and (perf.n_particoes == 6).all()


def test_paired_difference_uses_matching_partitions():
    X, y = synthetic_xy()
    records, _ = run_experiment(X, y, reps=2, n_splits=3, strategies=SUBSET, historical=None)
    table, ratio = aggregate(records, strategies=SUBSET)
    frame = pd.DataFrame(records).set_index(["estimador", "estrategia", "rep", "fold"])
    a = frame.loc[("xgb", "SMOTE 0.3")].f1.sort_index()
    b = frame.loc[("rf", "SMOTE 0.3")].f1.sort_index()
    row = table[(table.tipo == "dif_xgb_menos_rf") & (table.estrategia == "SMOTE 0.3")
                & (table.metrica == "F1")].iloc[0]
    expected = nadeau_bengio((a - b).to_numpy(), ratio, 1.0)
    assert row.diferenca_media == pytest.approx(expected["diferenca_media"], abs=1e-15)
    assert row.ic95_inf == pytest.approx(expected["ic95_inf"], abs=1e-15)
    assert row.gl == 5
    base = table[(table.tipo == "dif_sem_balanceamento") & (table.estimador == "xgb")]
    assert "No balancing" not in set(base.estrategia)


def test_metadata_params_are_strict_json_even_with_xgboost_missing_nan():
    y = np.array([0] * 90 + [1] * 10)
    assert math.isnan(xgb_factory(y, False, 1).get_params()["missing"])  # origem do NaN
    description = estimator_description(y)
    text = json.dumps(description, allow_nan=False)  # não pode levantar
    assert description["xgb"]["missing"] == "nan"  # texto, nunca 0
    assert description["xgb"]["scale_pos_weight"] == 1.0 and "rf" in json.loads(text)


def test_undefined_metric_stops_aggregation_and_vpp_zero_is_counted():
    X, y = synthetic_xy()
    records, _ = run_experiment(X, y, reps=1, n_splits=3, strategies=SUBSET, historical=None)
    broken = [dict(r) for r in records]
    broken[0]["f1"] = float("nan")
    with pytest.raises(ValueError, match="indefinida"):
        aggregate(broken, strategies=SUBSET)  # NaN não é ignorado em silêncio
    forced = [dict(r) for r in records]
    forced[0]["vpp"] = 0.0
    forced[1]["vpp"] = 0.0
    counts = vpp_zero_counts(forced)
    assert sum(sum(v.values()) for v in counts.values()) >= 2
    table, _ = aggregate(forced, strategies=SUBSET)  # VPP 0 entra nas médias, sem NaN
    perf = table[table.tipo == "desempenho"][["media", "dp"]]
    assert np.isfinite(perf.to_numpy(dtype=float)).all()
    json.dumps(counts, allow_nan=False)


def test_checkpoint_survives_script_change_but_not_a_different_analysis(tmp_path):
    X, y = synthetic_xy()
    records, _ = run_experiment(X, y, reps=1, n_splits=3, strategies=SUBSET, historical=None)
    base = {"seed": 42, "reps": 1, "n_splits": 3, "n_rows": 900, "n_pos": 41,
            "strategies": [s.key for s in SUBSET], "script_sha256": "antigo", "tolerance": 1e-4}
    path = tmp_path / "parcial.json"
    path.write_text(json.dumps({"fingerprint": base, "records": records}), encoding="utf-8")
    assert load_partial(path, {**base, "script_sha256": "novo"}) == records  # reaproveita
    assert load_partial(path, {**base, "seed": 43}) == []
    assert load_partial(path, {**base, "reps": 5}) == []
    assert fingerprints_match(base, {**base, "script_sha256": "novo"})
    assert load_partial(tmp_path / "ausente.json", base) == []


def test_strategy_list_matches_historical_artifact():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data/imbalance_sensitivity.csv"
    if not path.exists():
        pytest.skip("artefato histórico ausente")
    hist = pd.read_csv(path)
    assert [s.key for s in STRATEGIES] == list(hist.strategy)
    assert isinstance(STRATEGIES[0], Strategy)
