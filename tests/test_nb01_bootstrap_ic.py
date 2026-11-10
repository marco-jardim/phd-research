"""Contratos sintéticos do bootstrap do NB01 (sem o extrato real)."""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
)

from scripts.feature_engineering import engineer_features, load_data
from scripts.nb01_bootstrap_ic import (
    COLUMNS, DIFF_METRICS, DIVERGENCIA_AMBIENTE, JSON_KEYS, LR_SOLVERS, MODELS,
    POINT_METRICS, ReproducaoFalhou, analyze, bootstrap, group_weights,
    groups_of_test_rows, load_extract, optimizer_range, point_estimates, prepare,
    rerun_metrics, summarize, weighted_metrics,
)

ROOT = Path(__file__).resolve().parents[1]

# Extrato sintético no formato do real: ';', cabeçalho com sufixos DBF,
# vírgula decimal, vazios e uma coluna de texto. Nenhum dado real.
HEADER = ["COMPREC,C,12,0", "REFREC,C,12,0", "PASSO,N,2,0", "nota final,N,6,2",
          "DTNASC dt ap 1digi", "NOME qtd frag iguais", "C_SITUENCE",
          "R_DTNASC", "C_DTNASC", "TEXTO", "PAR"]
ROWS = [["000123", "R1", "1", "12,5", "0,8", "1", "3", "19800101", "19810101", "AAA", "2"],
        ["000124", "R1", "2", "7,25", "0", "0,5", "", "19700512", "", "BBB", "0"],
        ["000125", "R2", "1", "3", "1", "0,25", "4", "", "19900101", "CCC", "1"]]


def write_extract(tmp_path):
    path = tmp_path / "extrato.csv"
    lines = [";".join(r) for r in [HEADER, *ROWS]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def legacy_load(path):
    """load_data original com texto em object, como no pandas que gerou o NB01."""
    try:
        pd.get_option("future.infer_string")
    except pd.errors.OptionError:
        return load_data(path)
    with pd.option_context("future.infer_string", False):
        return load_data(path)


def test_load_extract_numeric_numpy_dtypes(tmp_path):
    df = load_extract(write_extract(tmp_path))
    assert list(df.columns) == [h.split(",")[0] for h in HEADER]
    assert all(isinstance(d, np.dtype) and d.kind in "if" for d in df.dtypes)
    assert df["nota final"].tolist() == [12.5, 7.25, 3.0]
    assert df["COMPREC"].tolist() == [123, 124, 125]
    assert df["TEXTO"].isna().all()
    assert df["C_SITUENCE"].isna().tolist() == [False, True, False]


def test_load_extract_matches_legacy_load_data(tmp_path):
    path = write_extract(tmp_path)
    new, old = load_extract(path), legacy_load(path)
    assert list(new.columns) == list(old.columns)
    for col in new.columns:
        assert new[col].dtype == old[col].dtype, col
        np.testing.assert_array_equal(new[col].to_numpy(), old[col].to_numpy())


def test_engineer_features_runs_on_loaded_extract(tmp_path):
    # Caminho que quebrava com string Arrow (DTNASC dt ap 1digi * 0.8).
    X, y = engineer_features(load_extract(write_extract(tmp_path)))
    assert X.shape == (3, 58)
    assert X["dtnasc_total"].to_numpy() == pytest.approx([0.64, 0.0, 0.8])
    assert X["diff_ano_nasc"].tolist() == [1, 99, 99]
    assert y.tolist() == [1, 0, 1]


def test_groups_of_test_rows_uses_refrec_by_position(tmp_path):
    groups = groups_of_test_rows(write_extract(tmp_path), np.array([2, 0]))
    assert groups.tolist() == ["R2", "R1"]


def synthetic(n=1500, groups=500, positives=60, seed=3):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=bool)
    y[rng.choice(n, positives, replace=False)] = True
    codes = rng.integers(0, groups, n)
    preps = {}
    for i, name in enumerate(MODELS):
        noise = 0.15 + 0.05 * i
        score = np.clip(y * 0.6 + rng.normal(0.2, noise, n), 0, 1)
        score = np.round(score, 2)  # empates de escore
        preps[name] = prepare(y, score >= 0.5, score)
    return preps, y, codes, groups


def test_weights_resample_whole_groups():
    codes = np.repeat(np.arange(20), 5)
    w = group_weights(np.random.default_rng(1), codes, 20)
    for g in range(20):
        assert len(set(w[codes == g])) == 1  # todas as linhas do óbito juntas
    assert w[::5].sum() == 20  # um peso por óbito; soma = número de óbitos


def test_weighted_equals_duplicated_rows():
    preps, y, codes, groups = synthetic()
    w = group_weights(np.random.default_rng(5), codes, groups)
    index = np.repeat(np.arange(len(y)), w)  # linhas de cada óbito duplicadas
    for name in ("Random Forest", "SVM (RBF)"):
        prep = preps[name]
        yd, pd_ = prep["y"][index], prep["pred"][index]
        got = weighted_metrics(prep, w)
        assert got["VPP"] == pytest.approx(precision_score(yd, pd_, zero_division=0))
        assert got["Sensibilidade"] == pytest.approx(recall_score(yd, pd_, zero_division=0))
        assert got["F1"] == pytest.approx(f1_score(yd, pd_, zero_division=0))
        assert got["AUC-PR"] == pytest.approx(
            average_precision_score(yd, prep["scores"][index]), abs=1e-12)


def test_point_estimate_matches_sklearn():
    preps, y, *_ = synthetic()
    for prep in preps.values():
        est = point_estimates(prep)
        assert est["AUC-PR"] == pytest.approx(
            average_precision_score(prep["y"], prep["scores"]), abs=1e-12)
        assert est["F1"] == pytest.approx(f1_score(prep["y"], prep["pred"]))


def test_deterministic_and_seed_sensitive():
    preps, _, codes, groups = synthetic()
    a, _ = bootstrap(preps, codes, groups, 30, seed=11)
    b, _ = bootstrap(preps, codes, groups, 30, seed=11)
    c, _ = bootstrap(preps, codes, groups, 30, seed=12)
    for name in preps:
        for metric in POINT_METRICS:
            assert np.array_equal(a[name][metric], b[name][metric])
    assert not np.array_equal(a["Random Forest"]["F1"], c["Random Forest"]["F1"])


def test_ci_contains_point_estimate_and_columns():
    preps, _, codes, groups = synthetic()
    point = {n: point_estimates(p) for n, p in preps.items()}
    reps, valid = bootstrap(preps, codes, groups, 300, seed=7)
    table = summarize(point, reps, 300, valid, 7)
    assert list(table.columns) == list(COLUMNS)
    n_pairs = len(list(itertools.combinations(MODELS, 2)))
    assert len(table) == len(MODELS) * len(POINT_METRICS) + n_pairs * len(DIFF_METRICS)
    assert (table.ic95_inf <= table.ponto_rerun).all()
    assert (table.ponto_rerun <= table.ic95_sup).all()
    diffs = table[table.tipo == "diferenca_pareada"]
    assert diffs.proporcao_replicas_diferenca_maior_0.between(0, 1).all()
    assert table[table.tipo == "classificador"].proporcao_replicas_diferenca_maior_0.isna().all()


def test_replicates_without_positives_are_dropped():
    # um único óbito positivo entre muitos: réplicas sem ele são descartadas
    n = 200
    y = np.zeros(n, dtype=bool)
    y[0] = True
    codes = np.arange(n)
    score = np.where(y, 0.9, 0.1)
    preps = {"A": prepare(y, score >= 0.5, score), "B": prepare(y, score >= 0.5, score)}
    reps, valid = bootstrap(preps, codes, n, 200, seed=1)
    assert 0 < valid < 200
    assert len(reps["A"]["F1"]) == len(reps["B"]["F1"]) == valid


def synthetic_run(n=600, groups=200, positives=40, seed=4):
    """Predições sintéticas dos 7 modelos e um nb01_results.json coerente com elas."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=bool)
    y[rng.choice(n, positives, replace=False)] = True
    preds = {}
    for i, name in enumerate(MODELS):
        score = np.clip(y * 0.6 + rng.normal(0.2, 0.15 + 0.05 * i, n), 0, 1)
        preds[name] = (y.astype(int), (score >= 0.5).astype(int), score)
    captured = {"preds": preds, "n_test": n, "pos_test": positives}
    results = [{"Modelo": name, **{JSON_KEYS[m]: round(v, 4)
                                   for m, v in rerun_metrics(*preds[name]).items()}}
               for name in MODELS]
    reference = {"n_test": n, "pos_test": positives, "all_results": results}
    groups_raw = rng.integers(0, groups, n).astype(str)
    return captured, reference, groups_raw


def perturb(reference, name, key="Precision", delta=0.01):
    for row in reference["all_results"]:
        if row["Modelo"] == name:
            row[key] = round(row[key] + delta, 4)


def test_all_reproduce_when_reference_matches():
    captured, reference, groups_raw = synthetic_run()
    result = analyze(captured, reference, groups_raw, 20, seed=1)
    assert all(s["reproduz"] for s in result["status"].values())
    assert result["table"].reproduz.all()
    assert (result["table"].motivo == "").all()


def test_listed_model_diverging_does_not_abort_and_keeps_both_values():
    captured, reference, groups_raw = synthetic_run()
    lr = "Logistic Regression"
    assert lr in DIVERGENCIA_AMBIENTE and "Stacking Ensemble" in DIVERGENCIA_AMBIENTE
    perturb(reference, lr, "Precision", 0.01)
    perturb(reference, lr, "F1-Score", -0.02)
    result = analyze(captured, reference, groups_raw, 20, seed=1)
    status = result["status"][lr]
    assert status["reproduz"] is False and status["motivo"] == DIVERGENCIA_AMBIENTE[lr]
    record = next(r for r in result["records"] if r["classificador"] == lr and r["metrica"] == "VPP")
    assert record["publicado"] != pytest.approx(record["rerun"], abs=1e-4)
    assert record["diferenca_rerun_menos_publicado"] == pytest.approx(
        record["rerun"] - record["publicado"])
    table = result["table"]
    rows = table[(table.tipo == "classificador") & (table.classificador_ou_par == lr)]
    vpp = rows.set_index("metrica").loc["VPP"]
    assert vpp.ponto_publicado == pytest.approx(record["publicado"])
    assert vpp.ponto_rerun == pytest.approx(record["rerun"])
    assert vpp.diferenca_rerun_menos_publicado == pytest.approx(vpp.ponto_rerun - vpp.ponto_publicado)
    assert not rows.reproduz.any() and (rows.motivo == DIVERGENCIA_AMBIENTE[lr]).all()
    assert rows.ic95_inf.notna().all()  # IC do rerun calculado mesmo sem reproduzir
    pairs = table[(table.tipo == "diferenca_pareada") & table.classificador_ou_par.str.contains(lr)]
    assert len(pairs) and not pairs.reproduz.any()
    others = table[(table.tipo == "classificador") & (table.classificador_ou_par != lr)]
    assert others.reproduz.all()


def test_unlisted_model_diverging_aborts():
    captured, reference, groups_raw = synthetic_run()
    assert "Random Forest" not in DIVERGENCIA_AMBIENTE
    perturb(reference, "Random Forest", "Recall", 0.01)
    with pytest.raises(ReproducaoFalhou, match="Random Forest Sensibilidade"):
        analyze(captured, reference, groups_raw, 20, seed=1)


def test_partition_mismatch_or_missing_model_aborts():
    captured, reference, groups_raw = synthetic_run()
    reference["pos_test"] += 1
    with pytest.raises(ReproducaoFalhou, match="pos_test"):
        analyze(captured, reference, groups_raw, 20, seed=1)
    captured, reference, groups_raw = synthetic_run()
    del captured["preds"]["Stacking Ensemble"]  # na lista, mas ausente: aborta
    with pytest.raises(ReproducaoFalhou, match="Stacking Ensemble"):
        analyze(captured, reference, groups_raw, 20, seed=1)


def test_optimizer_range_present_and_consistent():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4)) * np.array([1, 10, 100, 0.1])  # escalas distintas
    y = (X[:, 0] + 0.05 * X[:, 1] + rng.normal(0, 1, 400) > 1.5).astype(int)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X[:300], y[:300])
    rerun = lr.predict_proba(X[300:])[:, 1]
    out = optimizer_range(lr, X[:300], y[:300], X[300:], y[300:], rerun_proba=rerun)
    assert tuple(out["solvers"]) == LR_SOLVERS
    assert out["parametros_fixos_do_nb01"]["class_weight"] == "balanced"
    assert out["parametros_fixos_do_nb01"]["max_iter"] == 1000
    assert "solver" not in out["parametros_fixos_do_nb01"]
    assert out["solvers"]["lbfgs"]["max_abs_dif_proba_vs_rerun"] == 0.0
    for metric in (*POINT_METRICS, "FP"):
        span = out["faixa"][metric]
        values = [out["solvers"][s][metric] for s in LR_SOLVERS]
        assert span["minimo"] == min(values) and span["maximo"] == max(values)
    for s in LR_SOLVERS:
        m = out["solvers"][s]
        assert m["TP"] + m["FN"] == int(y[300:].sum())
        assert {"VPP", "Sensibilidade", "F1", "AUC-PR", "AUC-ROC", "n_iter"} <= set(m)


def test_generated_csv_is_coherent_if_present():
    path = ROOT / "data" / "nb01_bootstrap_ic.csv"
    if not path.exists():
        pytest.skip("artefato ainda não gerado")
    table = pd.read_csv(path)
    assert list(table.columns) == list(COLUMNS)
    assert (table.ic95_inf <= table.ponto_rerun + 1e-12).all()
    assert (table.ponto_rerun <= table.ic95_sup + 1e-12).all()
    not_reproduced = table.loc[table.tipo == "classificador"].loc[lambda t: ~t.reproduz]
    assert set(not_reproduced.classificador_ou_par) <= set(DIVERGENCIA_AMBIENTE)
    assert not_reproduced.motivo.notna().all()


def test_generated_metadata_has_decision_and_optimizer_range_if_present():
    path = ROOT / "data" / "nb01_bootstrap_ic_metadata.json"
    if not path.exists():
        pytest.skip("artefato ainda não gerado")
    import json

    meta = json.loads(path.read_text(encoding="utf-8"))
    assert set(meta["lr_faixa_otimizadores"]["solvers"]) == set(LR_SOLVERS)
    assert set(POINT_METRICS) <= set(meta["lr_faixa_otimizadores"]["faixa"])
    assert {"python", "numpy", "scipy", "sklearn", "pandas", "imblearn"} <= set(meta["versions"])
    assert meta["decisao"].startswith("pontos publicados no texto; IC do rerun no ambiente atual")
    assert set(meta["divergencia_ambiente"]["modelos"]) == set(DIVERGENCIA_AMBIENTE)
