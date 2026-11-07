"""Agregados reproduzidos; testes não retreinam nem chamam LLM."""
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.banca_b_recuperados_por_par import summarize
from scripts import banca_b_recuperados_por_par as recovered_module
from scripts.banca_c_agregados import read_csv, write_csv
from scripts.equidade_denominador_minimo import generate, recompute
from scripts.robustez_perda_ponderada import compute

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_weighted_loss_reproduces_artifact_and_marginal_medians(tmp_path):
    config = yaml.safe_load((ROOT / "gzcmd/gzcmd_v3_config.yaml").read_text(encoding="utf-8"))
    rows = compute(DATA, config)
    path = tmp_path / "loss.csv"
    write_csv(path, rows)
    assert path.read_bytes() == (DATA / "robustez_perda_ponderada.csv").read_bytes()
    for row in rows:
        assert row["perda_das_medianas"] == pytest.approx(
            row["C_FP"] * row["fp"] + row["C_FN"] * row["perdidos"] +
            row["C_H"] * row["pares_humano"] + row["C_LLM"] * row["pares_llm"])
        assert row["delta_p_max_maximo"] >= row["delta_p_max_mediana"]
        assert 0 <= row["jaccard_llm_minimo"] <= row["jaccard_llm_mediana"] <= 1
    vigilance = {row["bloco"]: row for row in rows if row["modo"] == "vigilancia"}
    assert vigilance["N"]["diferenca_P_menos_Platt"] == pytest.approx(144.105)
    assert vigilance["P"]["diferenca_P_menos_Platt"] == pytest.approx(94.41)
    assert vigilance["N"]["soma_ponderada_medianas_diferencas_pareadas"] == pytest.approx(147.605)
    assert vigilance["P"]["soma_ponderada_medianas_diferencas_pareadas"] == pytest.approx(47.8525)


def test_equity_filters_before_bh_and_reproduces_aggregates(tmp_path):
    source = pd.read_csv(DATA / "equidade_arranjo_pairwise_tests.csv")
    corrected = recompute(source)
    saved = pd.read_csv(DATA / "equidade_arranjo_den30_pairwise_tests.csv")
    pd.testing.assert_frame_equal(corrected, saved, check_dtype=False)
    assert corrected.n1.ge(30).all() and corrected.n2.ge(30).all()
    assert [int(corrected.loc[corrected.decisao.eq(d), "reject_fdr"].sum())
            for d in ("arranjo_P_vigilancia", "arranjo_P_confirmacao",
                      "cascata_v31_vigilancia", "cascata_v31_confirmacao")] == [3, 2, 6, 1]
    # Uma comparação minúscula com p muito baixo não integra a família BH.
    synthetic = pd.DataFrame(dict(decisao=["a"] * 3, n1=[3, 30, 40], n2=[50] * 3,
                                  p_value_raw=[.0001, .03, .2], p_value_fdr=[0.] * 3,
                                  reject_fdr=[True] * 3))
    result = recompute(synthetic)
    assert result.p_value_fdr.tolist() == pytest.approx([.06, .2])
    assert not result.reject_fdr.any()
    for path in DATA.glob("equidade_arranjo_*.csv"):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    editorial = tmp_path / "editorial"
    generate(tmp_path, editorial)
    assert len(list(editorial.glob("tab_den30_equidade_arranjo*.tex"))) == 5
    assert not list(editorial.glob("tab_equidade_arranjo_*.tex"))
    for path in editorial.glob("*.tex"):
        assert path.read_bytes() == (ROOT / "text/latex/tables/editorial" / path.name).read_bytes()
    for decision in ("arranjo_P_vigilancia", "cascata_v31_vigilancia"):
        rendered = (editorial / f"tab_den30_equidade_arranjo_rejeicoes_{decision}.tex").read_text(encoding="utf-8")
        assert rendered.count(r"\resizebox{\linewidth}{!}{%") == 1
        assert r"\end{tabular}}" in rendered
        assert rendered.index(r"\caption{") < rendered.index(r"\resizebox{")
        assert rendered.index(r"\end{tabular}}") < rendered.index(r"\par\footnotesize Fonte:")


def test_par_aggregate_does_not_invent_prediction_membership():
    raw = pd.DataFrame({"PAR": ["1"] * 211 + ["2"] * 36,
                        "REFREC": [str(i // 2) for i in range(247)]})
    result = summarize(raw)
    assert result["vinculos_por_par"] == {"1": 211, "2": 36}
    assert result["obitos_distintos_247"] == 124
    assert result["status"] == "pendente_predicoes_historicas_RF_SMOTE"
    assert "recuperados" not in result


def test_par_aggregate_reproduces_available_private_source():
    path = DATA / "COMPARADORSEMIDENT.csv"
    if not path.exists():
        pytest.skip("Extrato privado não disponível")
    raw = pd.read_csv(path, sep=";", dtype=str)
    raw.columns = [c.split(",")[0] for c in raw.columns]
    saved = json.loads((DATA / "banca_b_recuperados_por_par.json").read_text(encoding="utf-8"))
    available = summarize(raw)
    for key in ("vinculos", "obitos_distintos_247", "vinculos_por_par"):
        assert available[key] == saved[key]


def test_saved_predictions_are_controlled_and_only_counts_exported(monkeypatch):
    # Controle sintético da interface; não identifica integrantes históricos.
    raw = pd.DataFrame({"PAR": ["0"] * 18682, "REFREC": [str(i) for i in range(18682)],
                        "PASSO": ["6"] * 18682, "SCORE": ["7,0240"] * 18682,
                        "nota final": ["8"] * 18682})
    raw.loc[:73, "PAR"] = "1"
    raw.loc[18509:, "PAR"] = "1"
    raw.loc[20:26, "PAR"] = "2"
    raw.loc[:26, "nota final"] = "7"
    raw.loc[:9, "SCORE"] = "7"
    def engineered(numeric):
        return numeric[["nota final"]], numeric.PAR.isin([1, 2]).astype(int)
    def split(X, y, random_state):
        assert random_state == 42
        return X.iloc[18509:], X.iloc[:18509], y.iloc[18509:], y.iloc[:18509]
    monkeypatch.setattr(recovered_module, "engineer_features", engineered)
    monkeypatch.setattr(recovered_module, "split_data", split)
    predictions = pd.DataFrame({"probabilidade": [.1] * 18509})
    predictions.loc[:66, "probabilidade"] = .8
    predictions.loc[74:76, "probabilidade"] = .8
    result = summarize(raw, predictions)
    assert result["controle"]["recuperados"] == 27
    assert result["recuperados"]["por_par"] == {"1": 20, "2": 7}
    assert result["recuperados"]["abaixo_corte_orl"] == 10
    assert result["recuperados"]["acima_ou_no_corte_orl"] == 17
    assert result["recuperados"]["obitos_distintos"] == 27
    assert result["vp"]["por_par"] == {"1": 60, "2": 7}
    assert result["fp"]["por_par"] == {"0": 3, "1": 0, "2": 0}
    assert "REFREC" not in json.dumps(result)
    with pytest.raises(ValueError, match="cobrir exatamente"):
        summarize(raw, predictions.iloc[:-1])
    predictions.loc[0, "probabilidade"] = 2
    with pytest.raises(ValueError, match="Probabilidades"):
        summarize(raw, predictions)


def test_new_csv_artifacts_contain_only_aggregate_columns():
    for name in ("robustez_perda_ponderada.csv", "equidade_arranjo_den30_pairwise_tests.csv",
                 "equidade_arranjo_den30_rejeicoes.csv"):
        assert not {"pair_id", "REFREC", "COMPREC", "NOME", "R_DTOBITO"}.intersection(read_csv(DATA / name)[0])


def test_historical_retraining_aggregate_controls_and_metadata():
    saved = json.loads((DATA / "banca_b_recuperados_por_par.json").read_text(encoding="utf-8"))
    assert saved["status"] == "completo" and saved["treino_executado"]
    assert saved["controle"] == dict(test_n=18509, test_positivos=74, tp=67, fp=3, recuperados=27)
    assert saved["obitos_distintos_247"] == 204
    assert saved["seed"] == 42 and saved["tempo_execucao_segundos"] > 0
    assert saved["fp"]["por_par"] == {"0": 3, "1": 0, "2": 0}
    assert saved["vp"]["por_par"] == {"1": 62, "2": 5}
    assert saved["recuperados"]["por_par"] == {"1": 25, "2": 2}
    assert [saved[group]["obitos_distintos"] for group in ("vp", "recuperados", "fp")] == [60, 24, 3]
    assert saved["recuperados"]["acima_ou_no_corte_orl"] == 25
    assert saved["recuperados"]["abaixo_corte_orl"] == 2
    for group, n in (("vp", 67), ("recuperados", 27), ("fp", 3)):
        row = saved[group]
        assert row["n"] == n == sum(row["por_par"].values())
        assert row["acima_ou_no_corte_orl"] + row["abaixo_corte_orl"] == n
        assert row["corte_desconhecido"] == 0
        assert 0 < row["obitos_distintos"] <= n
    import hashlib
    for name, digest in saved["fontes_sha256"].items():
        if name == "data/COMPARADORSEMIDENT.csv" and not (ROOT / name).exists():
            continue
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest


def test_new_generated_aggregates_and_tables_use_lf():
    paths = [DATA / name for name in (
        "banca_b_recuperados_por_par.json", "custos_equivalentes_exploratorios.csv",
        "robustez_perda_ponderada.csv", "equidade_arranjo_den30_pairwise_tests.csv",
        "equidade_arranjo_den30_rejeicoes.csv", "equidade_arranjo_den30_recomputacao.json")]
    paths += list((ROOT / "text/latex/tables/editorial").glob("tab_den30_equidade_arranjo*.tex"))
    assert len(paths) == 11
    for path in paths:
        assert b"\r" not in path.read_bytes(), path.name
