"""Regressões da seleção histórica e da seleção opt-in sem campos brutos."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from gzcmd import cli, eval as evaluation
from gzcmd.classifier import (
    COMPARATOR_FEATURES, ClassifierConfig, GZCMDClassifier, get_feature_columns,
)
from scripts.exp1_refazer_allowlist import add_f2, compare_reproduction, comparison_table
from scripts.exp1_testes_t import comparar

ROOT = Path(__file__).resolve().parents[1]
# Snapshot dos nomes históricos, independente do seletor implementado.
HISTORICAL_OFF = """CODMUNRES local igual
CODMUNRES local prox
CODMUNRES uf igual
CODMUNRES uf prox
CODMUNRES_local_igual
CODMUNRES_local_prox
CODMUNRES_uf_igual
CODMUNRES_uf_prox
COMPREC,C,12,0
C_CODMUNRES,C,6,0
C_DTDIAG,C,10,0
C_DTNASC,C,8,0
C_DTNASC2,C,8,0
C_DTNOTI,C,10,0
C_IDLINHA,C,10,0
C_IDMUNI2,C,6,0
C_IDMUNIAT,C,6,0
C_IDPESSOA,C,10,0
C_IDUNID,C,15,0
C_IDUNIDAT,C,15,0
C_NOVOIDPESSOA,C,10,0
C_SITUENCE,C,10,0
DTNASC dt ap 1digi
DTNASC dt iguais
DTNASC dt inv ano
DTNASC dt inv dia
DTNASC dt inv mes
DTNASC_dt_ap_1digi
DTNASC_dt_iguais
DTNASC_dt_inv_ano
DTNASC_dt_inv_dia
DTNASC_dt_inv_mes
ENDERECO compl prox
ENDERECO numero igual
ENDERECO texto prox
ENDERECO tokens jacc
ENDERECO via igual
ENDERECO via prox
ENDERECO_compl_prox
ENDERECO_numero_igual
ENDERECO_texto_prox
ENDERECO_tokens_jacc
ENDERECO_via_igual
ENDERECO_via_prox
NOME prim frag igual
NOME qtd frag abrev
NOME qtd frag comuns
NOME qtd frag iguais
NOME qtd frag muito parec
NOME qtd frag raros
NOME ult frag igual
NOMEMAE prim frag igual
NOMEMAE qtd frag abrev
NOMEMAE qtd frag comuns
NOMEMAE qtd frag iguais
NOMEMAE qtd frag muito parec
NOMEMAE qtd frag raros
NOMEMAE ult frag igual
NOMEMAE_prim_frag_igual
NOMEMAE_qtd_frag_abrev
NOMEMAE_qtd_frag_comuns
NOMEMAE_qtd_frag_iguais
NOMEMAE_qtd_frag_muito_parec
NOMEMAE_qtd_frag_raros
NOMEMAE_ult_frag_igual
NOME_prim_frag_igual
NOME_qtd_frag_abrev
NOME_qtd_frag_comuns
NOME_qtd_frag_iguais
NOME_qtd_frag_muito_parec
NOME_qtd_frag_raros
NOME_ult_frag_igual
REFREC,C,12,0
R_CODMUNRES,C,6,0
R_DTNASC,C,8,0
R_DTNASC2,C,8,0
R_DTOBITO,C,10,0
R_IDLINHA,C,10,0
SCORE,C,15,0
diff_ano
dtnasc_all_zero
dtnasc_score_total
endereco_score_total
endereco_zero
mae_missing
mae_score_total
municipio_score
nome_score_total
nota final
nota_final
score_regras""".splitlines()
MACD = [
    "macd_nasc_close", "macd_nasc_day_match", "macd_nasc_diff_capped",
    "macd_nasc_month_match", "macd_nasc_partial_overlap", "macd_nasc_very_close",
    "macd_nasc_year_match", "macd_nome_perf_x_date_far", "macd_nome_perf_x_year_diff",
]


def frame(enabled: bool = True) -> pd.DataFrame:
    names = HISTORICAL_OFF + (MACD if enabled else [])
    out = pd.DataFrame({name: np.arange(20, dtype=float) for name in names})
    for name in ["COMPREC", "REFREC", "PAR", "TARGET", "PASSO", "guardrail", "p_cal"]:
        out[name] = np.arange(20) % 2
    out["text_column"] = "texto sintético"
    return out


@pytest.mark.parametrize("enabled,count", [(True, 100), (False, 91)])
def test_historical_snapshot_and_default_unchanged(enabled, count):
    expected = sorted(HISTORICAL_OFF + (MACD if enabled else []))
    selected = get_feature_columns(frame(enabled))
    assert len(selected) == count
    assert selected == expected
    assert selected == get_feature_columns(frame(enabled), feature_set="historical")


@pytest.mark.parametrize("enabled,count", [(True, 39), (False, 30)])
def test_allowlist_exact_names_and_no_raw_fields(enabled, count):
    # MACD existe no frame em ambos os casos: o opt-out precisa removê-la.
    selected = get_feature_columns(frame(), feature_set="allowlist", macd_enabled=enabled)
    source = json.loads((ROOT / "data/pcal_ml_oof.json").read_text(encoding="utf-8"))["features"]
    assert list(COMPARATOR_FEATURES) == source
    assert selected == sorted(source + (MACD if enabled else []))
    assert len(selected) == count
    prohibited = ("SCORE", "COMPREC", "REFREC", "TARGET", "C_SITUENCE", "ID")
    assert not any(token in name.upper() for name in selected for token in prohibited)
    # "muito parec" e "macd_nasc_partial_overlap" são atributos permitidos,
    # não o campo bruto PAR: conferir o nome do campo antes dos metadados.
    assert not any(name.split(",", 1)[0] == "PAR" for name in selected)
    assert not any(name.startswith(("C_", "R_")) for name in selected)
    assert "nota_final" not in selected


def test_allowlist_numeric_only_and_invalid_selection():
    data = frame()
    data["nota final"] = "não numérica"
    assert "nota final" not in get_feature_columns(data, feature_set="allowlist")
    with pytest.raises(ValueError, match="feature_set"):
        get_feature_columns(data, feature_set="invalid")


def test_classifier_fit_and_persistence_opt_in(tmp_path):
    config = ClassifierConfig(n_estimators=3, max_depth=2, calibration_cv=2, n_jobs=1)
    clf = GZCMDClassifier(config, feature_set="allowlist", macd_enabled=False).fit(frame())
    assert clf.features_ == sorted(COMPARATOR_FEATURES)
    path = tmp_path / "classifier.joblib"
    clf.save(path)
    loaded = GZCMDClassifier.load(path)
    assert loaded.feature_set == "allowlist"
    assert loaded.macd_enabled is False
    np.testing.assert_allclose(clf.predict_proba(frame()), loaded.predict_proba(frame()))
    state = joblib.load(path)
    del state["feature_set"], state["macd_enabled"]
    joblib.dump(state, path)
    legacy = GZCMDClassifier.load(path)
    assert legacy.feature_set == "historical"
    np.testing.assert_allclose(clf.predict_proba(frame()), legacy.predict_proba(frame()))


def test_csv_evaluation_passes_allowlist_and_macd(monkeypatch):
    captured = {}
    data = frame()
    monkeypatch.setattr(evaluation, "load_comparador_csv", lambda *a, **kw: data)

    def capture(df, **kwargs):
        captured.update(kwargs)
        assert df is data
        return pd.DataFrame()

    monkeypatch.setattr(evaluation, "evaluate_v3_dataframe", capture)
    evaluation.evaluate_v3_csv(
        input_csv="unused.csv", config_path=ROOT / "gzcmd/gzcmd_v3_config.yaml",
        modes=None, split_by="comprec", seeds=[42], test_size=0.3,
        group_stratify=True, calibration="ml_rf", macd_enabled=False,
        feature_set="allowlist",
        policy_version="v3",
    )
    assert captured["feature_set"] == "allowlist"
    assert captured["macd_enabled"] is False
    assert captured["policy_version"] == "v3"


@pytest.mark.parametrize("flag,expected", [([], "historical"), (["--feature-allowlist"], "allowlist")])
def test_cli_feature_option(monkeypatch, flag, expected):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({
            "mode": ["vigilancia"], "split_by": ["comprec"], "calibration": ["ml_rf"],
            "macd_enabled": [False], "exp_precision": [1.0], "exp_recall": [0.9],
            "exp_f1": [0.95], "exp_fbeta": [0.92], "auto_coverage": [0.9], "llm_used": [10],
        })

    monkeypatch.setattr(cli, "evaluate_v3_csv", capture)
    assert cli.main(["eval", "--config", str(ROOT / "gzcmd/gzcmd_v3_config.yaml"),
                     "--calibration", "ml_rf", "--no-macd", "--policy-version", "v3", *flag]) == 0
    assert captured["feature_set"] == expected
    assert captured["macd_enabled"] is False
    assert captured["policy_version"] == "v3"


def test_eval_v3_override_and_rf_selection_reach_actual_paths(monkeypatch):
    from gzcmd.config import load_config

    data = frame()
    data["nota final"] = 6.0
    captured = {}
    original_builder = evaluation.build_engine_from_config

    def guards(df, **kwargs):
        captured["guards"] = kwargs
        return pd.DataFrame({"guardrail": pd.Series(pd.NA, index=df.index)})

    def builder(cfg, **kwargs):
        captured["engine"] = kwargs
        return original_builder(cfg, **kwargs)

    class ClassifierSpy:
        def __init__(self, **kwargs):
            captured["classifier"] = kwargs

        def fit(self, train):
            captured["training_indices"] = train.index.tolist()

        def predict_proba(self, test):
            assert set(test.index).isdisjoint(captured["training_indices"])
            return np.tile([0.9, 0.1], (len(test), 1))

    monkeypatch.setattr(evaluation, "apply_guardrails", guards)
    monkeypatch.setattr(evaluation, "build_engine_from_config", builder)
    monkeypatch.setattr(evaluation, "GZCMDClassifier", ClassifierSpy)
    monkeypatch.setattr(evaluation, "split_train_test_indices", lambda *a, **kw: (np.arange(10), np.arange(10, 20)))
    result = evaluation.evaluate_v3_dataframe(
        data, cfg=load_config(ROOT / "gzcmd/gzcmd_v3_config.yaml"), modes=["vigilancia"],
        split_by="comprec", seeds=[42], test_size=0.3, group_stratify=True,
        calibration="ml_rf", macd_enabled=False, feature_set="allowlist", policy_version="v3",
    )
    assert len(result) == 1
    assert captured["guards"]["nota_always_match"] == 10.0
    assert captured["guards"]["nota_always_nonmatch"] == 3.0
    assert captured["guards"]["policy_version"] == "v3"
    assert captured["engine"]["policy_version"] == "v3"
    assert captured["classifier"] == {"feature_set": "allowlist", "macd_enabled": False}


def test_custom_t_paths_preserve_procedure(tmp_path):
    on = pd.DataFrame({"mode": ["vigilancia"] * 5, "seed": range(42, 47)})
    off = on.copy()
    for metric in ["exp_f1", "exp_precision", "exp_recall", "review_selected"]:
        on[metric] = [1, 3, 5, 7, 10]
        off[metric] = [2, 4, 6, 9, 12]
    a, b = tmp_path / "on.csv", tmp_path / "off.csv"
    on.to_csv(a, sep=";", index=False)
    off.to_csv(b, sep=";", index=False)
    result = comparar(on_path=a, off_path=b)
    assert result["n_rodadas"].eq(5).all()
    assert result["diferenca_media"].eq(-1.4).all()
    assert result["p_pareado"].between(0, 1).all()


def test_reproduction_checks_values_and_schema():
    old = pd.DataFrame({"mode": ["vigilancia"], "seed": [42], "value": [0.9]})
    assert compare_reproduction(old, old) == {"seed": 0.0, "value": 0.0}
    assert compare_reproduction(old, old.assign(guardrails_enabled=True)) == {"seed": 0.0, "value": 0.0}
    with pytest.raises(ValueError, match="guardrails habilitados"):
        compare_reproduction(old, old.assign(guardrails_enabled=False))
    new = old.assign(value=0.8)
    assert compare_reproduction(old, new)["value"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="não numérica"):
        compare_reproduction(old, old.assign(mode="confirmacao"))
    with pytest.raises(ValueError, match="Esquema"):
        compare_reproduction(old, old.drop(columns="value"))


def test_f2_both_modes_and_reduction_definition():
    data = pd.read_csv(ROOT / "data/ablation_comprec_macd.csv", sep=";")
    f2 = add_f2(data)
    vigilance = f2["mode"].eq("vigilancia")
    np.testing.assert_allclose(f2.loc[vigilance, "exp_f2"], f2.loc[vigilance, "exp_fbeta"])
    assert not np.allclose(f2.loc[~vigilance, "exp_f2"], f2.loc[~vigilance, "exp_fbeta"])
    off = pd.read_csv(ROOT / "data/ablation_comprec_nomacd.csv", sep=";")
    frames = {"macd": data, "nomacd": off}
    result = comparison_table(frames, frames)
    assert result["difference_allowlist_minus_historical"].eq(0).all()
    reduction = result.query("metric == 'review_reduction_percent'")
    assert len(reduction) == 2
    assert reduction["historical_mean"].between(14, 18).all()
