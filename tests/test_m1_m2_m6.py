from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gzcmd.guardrails import apply_guardrails
from gzcmd.config import load_config
from gzcmd.cli import build_parser
from gzcmd.ml_oof import nested_oof_probabilities, numeric_features
from gzcmd.policy_engine import Budget, Costs, HumanBudget, HumanError, LLMError, PolicyEngineV3
from gzcmd.reliability import load_band_rates
from gzcmd.runner import build_engine_from_config
from scripts.funil_gzcmd_v2 import loss_metrics, replay


def test_yaml_and_cli_expose_new_and_reproducible_profiles(tmp_path) -> None:
    cfg = load_config("gzcmd/gzcmd_v3_config.yaml")
    assert cfg.calibration.p_cal == "ml_xgb"
    assert cfg.guardrails.policy_version == "v3.1"
    assert cfg.decision_policy.modes["vigilancia"].human_review_cost == 1
    assert cfg.decision_policy.modes["vigilancia"].llm_review_cost == .005
    assert cfg.decision_policy.modes["vigilancia"].human_e_fn == .05
    assert cfg.decision_policy.modes["vigilancia"].human_max_pairs_per_window is None
    assert cfg.decision_policy.modes["confirmacao"].false_positive_cost == 100
    minimal = tmp_path / "custom.yaml"
    minimal.write_text("""
version: 3
bands:
  definitions:
    all: {min: 0, max: 999}
decision_policy:
  modes:
    vigilancia:
      costs: {false_positive: 10, false_negative: 50, llm_review: 0.25, human_review: 2.5}
      llm_budget: {max_calls_per_window: 2}
llm_review:
  enabled: false
""", encoding="utf-8")
    assert load_config(minimal).decision_policy.modes["vigilancia"].human_review_cost == 2.5
    parser = build_parser()
    old = parser.parse_args(["run", "--mode", "vigilancia", "--p-cal", "platt", "--policy-version", "v3"])
    assert (old.p_cal, old.policy_version) == ("platt", "v3")
    new = parser.parse_args(["run", "--mode", "confirmacao", "--p-cal", "ml_xgb", "--reuse-ml-oof"])
    assert new.reuse_ml_oof


def test_nested_oof_actual_fit_predict_boundaries_and_determinism() -> None:
    x = pd.DataFrame({"row": np.arange(100), "signal": np.arange(100) % 3})
    y = np.tile([0, 1], 50)
    events, fits, predictions, calibration_fits = [], [], [], []

    class SpyModel:
        def fit(self, frame, labels):
            self.train = set(frame.index)
            fits.append(self.train)

        def predict_proba(self, frame):
            test = set(frame.index)
            assert self.train.isdisjoint(test), "in-sample classifier prediction"
            predictions.append(test)
            p = (frame.row.to_numpy() + 0.5) / 101
            return np.column_stack([1 - p, p])

    class SpyCalibrator:
        def fit(self, p, labels):
            self.train = set(np.rint(p * 101 - 0.5).astype(int))
            calibration_fits.append(self.train)
            assert len(p) == 80

        def predict(self, p):
            test = set(np.rint(p * 101 - 0.5).astype(int))
            assert self.train.isdisjoint(test), "in-sample isotonic prediction"
            return p

    def audit(fold, phase, train, predicted, outer_test):
        assert set(train).isdisjoint(predicted)
        assert set(train).isdisjoint(outer_test)
        if phase == "inner":
            assert set(predicted).isdisjoint(outer_test)
        events.append((fold, phase, set(train), set(predicted), set(outer_test)))

    p, fold = nested_oof_probabilities(x, y, model_factory=lambda *_: SpyModel(),
                                      calibrator_factory=SpyCalibrator, audit=audit)
    assert len(fits) == len(predictions) == len(events) == 30
    assert len(calibration_fits) == 5
    assert np.bincount(fold).tolist() == [20] * 5
    for outer_fold in range(5):
        test = set(np.flatnonzero(fold == outer_fold))
        assert all(test.isdisjoint(train) for f, _, train, _, _ in events if f == outer_fold)
        assert calibration_fits[outer_fold].isdisjoint(test)
    again, again_fold = nested_oof_probabilities(x, y, model_factory=lambda *_: SpyModel(),
                                                calibrator_factory=SpyCalibrator)
    np.testing.assert_array_equal(p, again)
    np.testing.assert_array_equal(fold, again_fold)


def test_numeric_features_are_comparator_only_and_decimal_comma_safe() -> None:
    x = numeric_features(pd.DataFrame({"NOME qtd frag iguais": ["0,95", ""],
                                      "nota final": ["9,1", "5,3"], "PAR": ["1", "0"],
                                      "COMPREC": ["123", "456"], "R_DTNASC": ["20000101", "19990101"],
                                      "TARGET": [1, 0]}))
    assert x.columns.tolist() == ["NOME qtd frag iguais", "nota final"]
    assert x.iloc[0, 0] == 0.95
    assert x.iloc[1, 0] == 0.0


def test_v31_guards_fail_closed_and_never_discard_grey() -> None:
    df = pd.DataFrame({"nota final": [9.0, 4.99, 5.0, 8.99, 10.0, 9.5],
                       "NOME qtd frag iguais": [1] * 6, "NOME prim frag igual": [1] * 6,
                       "NOME ult frag igual": [1] * 6, "DTNASC dt iguais": [1, 1, 0, 0, np.nan, 1],
                       "CODMUNRES local igual": [1] * 6, "p_cal": [.9, 1, 1, 1, .99, .89],
                       "R_DTOBITO_dt": [pd.NaT] * 3 + [pd.Timestamp("2020-01-01")] + [pd.NaT, pd.Timestamp("2020-01-01")],
                       "C_DTDIAG_dt": [pd.NaT] * 3 + [pd.Timestamp("2021-01-01")] + [pd.NaT, pd.Timestamp("2021-01-01")]})
    g = apply_guardrails(df, policy_version="v3.1")
    assert g.guardrail.iloc[0] == "ALWAYS_MATCH"
    assert g.guardrail.iloc[1] == "ALWAYS_NONMATCH"
    assert not g.guardrail.iloc[2:4].eq("ALWAYS_NONMATCH").any()
    assert g.guardrail.iloc[4] == "ALWAYS_MATCH"  # exact DOB is no longer mandatory
    assert pd.isna(g.guardrail.iloc[5])  # high p<.9 left to Bayesian argmin
    assert g.reason.iloc[5] == "high_probability_inconsistent"


@pytest.mark.parametrize("budget", [0, 2, 100])
def test_nested_cascade_negative_evr_and_budget(budget: int) -> None:
    df = pd.DataFrame({"nota final": [4.99, 5.0, 8.99, 9.0, 9.5],
                       "p_cal": [0.01, 0.0, 1.0, 0.99, 0.5], "band": ["all"] * 5,
                       "guardrail": ["ALWAYS_NONMATCH", pd.NA, pd.NA, "ALWAYS_MATCH", "FORCE_REVIEW"]})
    engine = PolicyEngineV3(Costs(10, 50, .25), {"all": LLMError(0, 0)}, Budget(budget), policy_version="v3.1", human_error=HumanError(0, 0))
    out = engine.triage(df)
    assert out.action.iloc[0] == "NONMATCH"
    assert out.action.iloc[3] == "MATCH"
    assert out.action.iloc[1] == "NONMATCH"  # explicit Bayesian rule, not ALWAYS_NONMATCH
    assert out.action.iloc[2] == "MATCH"  # near-certainty is no longer DEFERRED
    llm = set(out.index[out.action.eq("LLM_REVIEW")])
    area = set(out.index[out.review_area])
    assert llm <= area <= set(out.index)
    assert area == {1, 2, 4}
    assert len(llm) == min(budget, 1)
    assert int(out.action.eq("DEFERRED").sum()) == int(budget == 0)
    assert (out.loc[[1, 2], "evr"] < 0).all()
    if budget:
        assert llm == {4}


def test_nested_rejects_nonmatch_guard_outside_a_minus() -> None:
    engine = PolicyEngineV3(Costs(10, 50, .25), {"a": LLMError(0, 0)}, Budget(5), policy_version="v3.1")
    with pytest.raises(ValueError, match="only below 5"):
        engine.triage(pd.DataFrame({"nota final": [6], "p_cal": [.01], "band": ["a"], "guardrail": ["ALWAYS_NONMATCH"]}))


def test_human_loss_and_offline_missing_response_are_not_false_negatives() -> None:
    costs = Costs(10, 50, .25, human_review=2)
    assert costs.total_loss(fp=2, fn=3, unsure=4, deferred=5) == 188
    engine = PolicyEngineV3(costs, {"a": LLMError(0, 0, e_unsure=.5)}, Budget(1))
    assert engine._losses(np.array([.2]), np.array(["a"]))[2][0] == 1.25
    out = pd.DataFrame({"action": ["MATCH", "NONMATCH", "LLM_REVIEW", "DEFERRED", "LLM_REVIEW"],
                        "y": [0, 1, 1, 1, 0], "p_cal": [.9, .1, .5, .5, .2],
                        "band": ["a"] * 5}, index=["a", "b", "c", "d", "e"])
    r = replay(out, {"c": "UNSURE", "e": "MATCH"}, old_batch={"c"})
    assert r.final.iloc[-1] == "SEM_RESPOSTA_LLM"  # new pair cannot borrow an unrelated run
    metric = loss_metrics(r, costs, {"a": LLMError(.1, .2, .1)}, HumanError(0, 0))
    assert metric["FP"] == metric["FN"] == metric["UNSURE"] == 1
    assert metric["DEFERRED"] == 1  # unanswered LLM is expected, not human
    assert metric["sem_resposta_LLM"] == 1
    assert metric["perda_observada"] == 64.25
    assert metric["perda_esperada"] == pytest.approx(3.25)
    assert metric["perda"] == pytest.approx(67.5)
    assert metric["esperada"]
    assert metric["sens_inferior"] == 0
    assert metric["sens_superior"] == 2 / 3
    assert metric["espec_superior"] == .5


@pytest.mark.parametrize("budget,expected", [
    (0, ["NONMATCH", "MATCH", "DEFERRED"]),
    (1, ["NONMATCH", "MATCH", "LLM_REVIEW"]),
    (2, ["LLM_REVIEW", "MATCH", "LLM_REVIEW"]),
    (3, ["LLM_REVIEW", "LLM_REVIEW", "LLM_REVIEW"]),
])
def test_bayesian_gain_and_budget_fallback_to_all_non_llm_choices(budget, expected) -> None:
    df = pd.DataFrame({"nota final": [6.0] * 3, "p_cal": [.019, .95, .5], "band": ["a"] * 3})
    engine = PolicyEngineV3(Costs(10, 50, .25), {"a": LLMError(0, 0)}, Budget(budget), policy_version="v3.1", human_error=HumanError(0, 0))
    out = engine.triage(df)
    assert out.action.tolist() == expected
    np.testing.assert_allclose(out.evr, [.70, .25, .75])
    np.testing.assert_allclose(out.base_loss, [.95, .5, 1.0])
    assert out.base_choice.tolist() == ["NONMATCH", "MATCH", "DEFERRED"]
    if budget == 0:
        assert out.decision_reason.eq("llm_budget_fallback").all()


def test_human_argmin_and_llm_ties_and_abstention_cost() -> None:
    df = pd.DataFrame({"nota final": [6.0, 6.0, 6.0], "p_cal": [.5, .005, .5], "band": ["bad", "perfect", "unsure"]})
    rates = {"bad": LLMError(.5, .5), "perfect": LLMError(0, 0), "unsure": LLMError(0, 0, 1)}
    out = PolicyEngineV3(Costs(10, 50, .25), rates, Budget(10), policy_version="v3.1", human_error=HumanError(0, 0)).triage(df)
    assert out.action.tolist() == ["DEFERRED", "NONMATCH", "DEFERRED"]
    assert out.loss_llm.iloc[2] == 1.25
    assert out.evr.iloc[1] == 0  # tie does not consume API budget
    assert not out.review_requested.any()


def test_m3_mode_rates_and_n_less_than_ten_fallback(tmp_path) -> None:
    import yaml
    fixed = {"small": {"e_fp": .1, "e_fn": .2}, "enough": {"e_fp": .1, "e_fn": .2}}
    block = {"error_rates_by_band": {b: {"e_fp": .3, "e_fn": .4} for b in fixed},
             "estimates_by_band": {"small": {"n": 9, "u": .8}, "enough": {"n": 10, "u": .25}}}
    path = tmp_path / "rates.yaml"
    path.write_text(yaml.safe_dump({"by_mode": {"vigilancia": {"pooled": block}}}), encoding="utf-8")
    rates = load_band_rates(fixed, mode="vigilancia", source=path)
    assert rates["small"] == LLMError(.1, .2, 0)
    assert rates["enough"] == LLMError(.3, .4, .25)
    cfg = load_config("gzcmd/gzcmd_v3_config.yaml")
    real = load_band_rates(cfg.llm_review.error_rates_by_band, mode="vigilancia", source=cfg.llm_review.estimated_rates_path)
    assert set(real) == set(cfg.llm_review.error_rates_by_band)
    assert all(0 <= r.e_unsure <= 1 for r in real.values())
    new_engine = build_engine_from_config(cfg, mode="vigilancia", policy_version="v3.1")
    old_engine = build_engine_from_config(cfg, mode="vigilancia", policy_version="v3")
    assert new_engine.llm_error_by_band["grey_mid"] == real["grey_mid"]
    assert old_engine.llm_error_by_band["grey_mid"].e_fn == .1
    assert old_engine.llm_error_by_band["grey_mid"].e_unsure == 0
    assert old_engine.costs.llm_review == .25
    assert old_engine.human_error == HumanError(0, 0)


def test_expected_sensitivity_distinguishes_automatic_and_perfect_human() -> None:
    out = pd.DataFrame({"action": ["LLM_REVIEW"] * 2, "y": [1, 0],
                        "p_cal": [.8, .2], "band": ["a", "a"]}, index=["a", "b"])
    pending = replay(out, {}, set())
    metrics = loss_metrics(pending, Costs(10, 50, .25), {"a": LLMError(.1, .2, .1)}, HumanError(0, 0))
    assert metrics["DEFERRED"] == 0
    assert metrics["sem_resposta_LLM"] == 2
    assert metrics["sens_observada"] == 0
    assert metrics["sens_esperada"] == pytest.approx(.7)
    assert metrics["sens_esperada_com_humano"] == pytest.approx(.8)
    assert metrics["espec_esperada"] == pytest.approx(.8)
    assert metrics["espec_esperada_com_humano"] == pytest.approx(.9)


@pytest.mark.parametrize("llm_cap,expected", [
    (0, ["DEFERRED", "NONMATCH", "NONMATCH"]),
    (1, ["LLM_REVIEW", "DEFERRED", "NONMATCH"]),
])
def test_human_budget_gain_and_fallback(llm_cap, expected) -> None:
    frame = pd.DataFrame({"nota final": [6.] * 3, "p_cal": [.5, .4, .05], "band": ["a", "b", "bad"]})
    rates = {"a": LLMError(.2, .2), "b": LLMError(.3, .3), "bad": LLMError(.9, .9)}
    human = HumanBudget(1)
    engine = PolicyEngineV3(Costs(10, 10, .1), rates, Budget(llm_cap), policy_version="v3.1",
                           human_error=HumanError(0, 0), human_budget=human)
    out = engine.triage(frame)
    assert out.action.tolist() == expected
    assert human.used == 1
    assert engine.budget.llm_used == llm_cap
    if llm_cap:
        assert out.decision_reason.iloc[0] == "human_budget_fallback_llm"
    else:
        assert out.decision_reason.iloc[1] == "human_budget_fallback_auto"


def test_human_budget_used_across_batches_and_zero_capacity() -> None:
    frame = pd.DataFrame({"nota final": [6.], "p_cal": [.5], "band": ["a"]})
    human = HumanBudget(1, used=1)
    engine = PolicyEngineV3(Costs(10, 10, .1), {"a": LLMError(.2, .2)}, Budget(1), policy_version="v3.1",
                           human_error=HumanError(0, 0), human_budget=human)
    out = engine.triage(frame)
    assert out.action.iloc[0] == "LLM_REVIEW"
    assert human.used == 1
    out2 = engine.triage(frame)
    assert out2.action.iloc[0] == "MATCH"  # both budgets exhausted
    assert human.used == 1


def test_imperfect_human_loss_and_unsure_expected_error() -> None:
    costs = Costs(10, 50, .005)
    human = HumanError(.05, .05)
    frame = pd.DataFrame({"nota final": [6.], "p_cal": [.2], "band": ["a"]})
    rates = {"a": LLMError(.1, .1)}
    imperfect = PolicyEngineV3(costs, rates, Budget(1), policy_version="v3.1", human_error=human).triage(frame)
    perfect = PolicyEngineV3(costs, rates, Budget(1), policy_version="v3.1", human_error=HumanError(0, 0)).triage(frame)
    assert imperfect.loss_human.iloc[0] == pytest.approx(1.9)
    assert imperfect.action.iloc[0] == "LLM_REVIEW"
    assert perfect.action.iloc[0] == "DEFERRED"
    abstains = PolicyEngineV3(costs, {"a": LLMError(0, 0, .5)}, Budget(1), policy_version="v3.1", human_error=human).triage(frame)
    assert abstains.loss_llm.iloc[0] == pytest.approx(.955)
    known = frame.rename(columns={"nota final": "nota"}).assign(action="LLM_REVIEW", y=1)
    response = replay(known, {0: "UNSURE"}, {0})
    metrics = loss_metrics(response, costs, rates, human)
    assert metrics["perda_observada"] == pytest.approx(1.005)
    assert metrics["perda_esperada"] == pytest.approx(.9)
    assert metrics["perda"] == pytest.approx(1.905)
    assert metrics["sens_esperada_com_humano"] == .95
    assert costs.total_loss(fp=0, fn=0, unsure=1, deferred=0, expected_human_error_loss=.9) == 1.9


def test_invalid_human_parameters_and_configuration(tmp_path) -> None:
    with pytest.raises(ValueError):
        HumanBudget(-1)
    with pytest.raises(ValueError):
        HumanError(.05, 1.1)
    import yaml
    from pathlib import Path
    config = yaml.safe_load(Path("gzcmd/gzcmd_v3_config.yaml").read_text(encoding="utf-8"))
    config["decision_policy"]["modes"]["vigilancia"]["human_budget"]["max_pairs_per_window"] = 25
    config["decision_policy"]["modes"]["vigilancia"]["human_error"] = {"e_fp": .02, "e_fn": .1}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.decision_policy.modes["vigilancia"].human_max_pairs_per_window == 25
    assert cfg.decision_policy.modes["vigilancia"].human_e_fn == .1


def test_sweep_metrics_include_human_errors_and_unsure_load() -> None:
    from gzcmd.cost_evaluation import expected_funnel_metrics
    frame = pd.DataFrame({"nota final": [6.] * 2, "p_cal": [.5, .5], "band": ["a", "a"],
                          "TARGET": [1, 1], "guardrail": [pd.NA, pd.NA],
                          "action": ["LLM_REVIEW", "DEFERRED"], "selected_loss": [2., 3.]})
    m = expected_funnel_metrics(frame, {"a": LLMError(.1, .2, .1)}, HumanError(.05, .05))
    assert m["humano_direto"] == 1
    assert m["humano_total_esperado"] == pytest.approx(1.1)
    assert m["sens_esperada_automatica"] == pytest.approx(.35)
    assert m["sens_esperada_com_humano"] == pytest.approx((.7 + .1 * .95 + .95) / 2)
    assert m["perda_esperada_total"] == 5


def test_sweep_grid_and_alternate_rates_path(tmp_path) -> None:
    import yaml
    from scripts.fig_varredura_custos import sweep
    cfg = load_config("gzcmd/gzcmd_v3_config.yaml")
    block = {"estimates_by_band": {"grey_mid": {"n": 10, "u": .1}},
             "error_rates_by_band": {"grey_mid": {"e_fp": .1, "e_fn": .2}}}
    rates_path = tmp_path / "reestimated.yaml"
    rates_path.write_text(yaml.safe_dump({"by_mode": {mode: {"pooled": block} for mode in ("vigilancia", "confirmacao")}}), encoding="utf-8")
    frame = pd.DataFrame({"nota final": [6.] * 3, "p_cal": [.4, .7, .99],
                          "band": ["grey_mid"] * 3, "TARGET": [1, 0, 1], "guardrail": [pd.NA] * 3})
    tables = sweep(frame, cfg, rates_path)
    for table in tables.values():
        assert len(table) == 60
        assert set(table.C_llm) == {.005, .05, .25}
        assert set(table.e_fn_h) == {0., .02, .05, .1}
        assert set(table.orcamento_humano) == {25., 50., 100., 200., np.inf}
        assert table.sens_esperada_automatica.between(0, 1).all()
        assert table.sens_esperada_com_humano.between(0, 1).all()
