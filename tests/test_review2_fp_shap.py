"""Aggregate-only FP cardinality and seed-42 SHAP-family regression tests."""
import json
import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.fp_cardinalidade import counts
from scripts.shap_por_familia import aggregate, duplicate_audit, family
from scripts.feature_engineering import engineer_features, split_data
from scripts.sensibilidade_sem_obito_refrec import carregar
from scripts.review2_proveniencia import references

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_cardinality_counts_overlap_once_and_only_against_preaccepted_records():
    accepted = pd.DataFrame({"COMPREC": ["C1", "C2"], "REFREC": ["R1", "R2"]})
    fp = pd.DataFrame({"COMPREC": ["C1", "C1", "X", "X"],
                       "REFREC": ["R2", "Y", "R1", "Y"]})
    assert counts(fp, accepted) == {
        "fp_total": 4, "comprec_ja_aceito": 2, "refrec_ja_aceito": 2,
        "ambos_ja_aceitos": 1, "comprec_ou_refrec_ja_aceito": 3, "nenhum_ja_aceito": 1}


def test_cardinality_published_counts_and_absence_of_pair_rows():
    value = json.loads((DATA / "fp_cardinalidade.json").read_text(encoding="utf-8"))
    assert value["n_referencia"] == 211
    assert value["modos"]["vigilancia"] == {
        "fp_total": 21, "comprec_ja_aceito": 0, "refrec_ja_aceito": 1,
        "ambos_ja_aceitos": 0, "comprec_ou_refrec_ja_aceito": 1, "nenhum_ja_aceito": 20}
    assert value["modos"]["confirmacao"] == {
        "fp_total": 4, "comprec_ja_aceito": 0, "refrec_ja_aceito": 0,
        "ambos_ja_aceitos": 0, "comprec_ou_refrec_ja_aceito": 0, "nenhum_ja_aceito": 4}
    assert "pair_id" not in json.dumps(value)


def test_family_allocation_preserves_every_attribute_and_total():
    source = pd.read_csv(DATA / "shap_importance.csv")
    result = aggregate(source)
    published = pd.read_csv(DATA / "shap_por_familia.csv")
    pd.testing.assert_frame_equal(result, published)
    assert result.n_atributos.sum() == 58
    assert result.shap_mean_abs_soma.sum() == pytest.approx(source.shap_mean_abs.sum())
    assert result.fracao_total.sum() == pytest.approx(1)
    assert family("score_recall") == "outros"  # Mixed-domain score, not a transform of nota.
    assert family("nome_x_dtnasc") == "outros"
    assert family("nome_squared") == "nome"
    assert result.set_index("familia").loc["nota e derivados", "shap_mean_abs_soma"] == pytest.approx(0.07205093890388912)


def test_family_rejects_unknown_or_duplicated_attributes():
    with pytest.raises(ValueError, match="Unclassified"):
        family("unknown_feature")
    with pytest.raises(ValueError, match="unique"):
        aggregate(pd.DataFrame({"feature": ["nota final"] * 2, "shap_mean_abs": [1, 2]}))


def test_seed42_heldout_duplicate_groups_are_exact_and_reproducible():
    X, y = engineer_features(carregar(DATA / "COMPARADORSEMIDENT.csv"))
    _, heldout, _, labels = split_data(X, y, random_state=42)
    actual = duplicate_audit(heldout)
    published = json.loads((DATA / "atributos_duplicados.json").read_text(encoding="utf-8"))
    for key, value in actual.items():
        assert published[key] == value
    assert (len(heldout), int(labels.sum())) == (18509, 74)
    assert actual["grupos_duplicatas_exatas_teste"] == [
        ["NOME prim frag igual", "nome_prim_ok"],
        ["DTNASC dt iguais", "dtnasc_squared"],
        ["CODMUNRES local igual", "local_ok"],
        ["dtnasc_ok", "max_score_dtnasc"],
    ]
    assert actual["duplicatas_exatas_redundantes"] == 4
    assert actual["dimensao_apos_remover_duplicatas_exatas_teste"] == 54
    assert all(t["monotona_nao_decrescente_no_teste"] for t in actual["transformacoes"])


def test_false_claim_of_exact_duplicate_fails_instead_of_using_tolerance():
    frame = pd.DataFrame({"DTNASC dt iguais": [0.0, 1.0], "dtnasc_squared": [0.0, 1.0 - 1e-12],
                          "dtnasc_ok": [0, 1], "max_score_dtnasc": [0, 1]})
    with pytest.raises(ValueError, match="not exact"):
        duplicate_audit(frame)


def test_config_annotation_changes_no_historical_parameter():
    import subprocess

    historical = subprocess.check_output(["git", "show", "eaf00b5:gzcmd/gzcmd_v3_config.yaml"], cwd=ROOT)
    current = (ROOT / "gzcmd/gzcmd_v3_config.yaml").read_text(encoding="utf-8")
    assert yaml.safe_load(historical) == yaml.safe_load(current)
    documented = (ROOT / "gzcmd/gzcmd_v31_p_config.yaml").read_text(encoding="utf-8")
    assert "RESERVADO, NÃO IMPLEMENTADO" in documented
    profile = yaml.safe_load(documented)
    assert profile["arranjo_P_protocolo"]["consumido_pelo_runner_generico"] is False
    assert profile["evaluation"]["seeds"] == [42]
    assert profile["evaluation"]["split_by"] == "refrec"
    for model in profile["llm_review"]["model"].values():
        assert model["max_tokens"] == 2500
        assert "temperature" not in model
    assert yaml.safe_load(current)["guardrails"]["policy_version"] == "v3.1"
    assert yaml.safe_load(current)["calibration"]["p_cal"] == "ml_xgb"


def test_supplemental_provenance_reports_current_hashes_without_rewriting_history():
    target = "gzcmd/llm_dossier.py"
    assert list(references({"code": {target: "a" * 64}}, target)) == [(f"/code/{target}", "a" * 64)]
    audit = json.loads((DATA / "review2_proveniencia.json").read_text(encoding="utf-8"))
    assert audit["manifestos_historicos_regravados"] == []
    assert audit["preregistros_alterados"] is False
    assert {r["manifesto"] for r in audit["fontes"]} == {
        "arranjo_corte_metadata.json", "arranjo_corte_s1_metadata.json",
        "filtro_llm_metadata.json", "robustez_arranjo_metadata.json"}
    for row in audit["fontes"]:
        assert hashlib.sha256((ROOT / row["fonte"]).read_bytes()).hexdigest() == row["sha256_atual"]
        assert row["identico"] == (row["sha256_atual"] == row["sha256_historico"])
