"""Verifica a fronteira entre os campos de origem e o dossiê de revisão."""

import pandas as pd
import pytest

from gzcmd.llm_dossier import build_dossier, make_pair_id


def test_dossier_excludes_nominal_fields_and_internal_identifiers():
    row = pd.Series({
        "COMPREC": 12345,
        "REFREC": 6789,
        "PASSO": 11,
        "NOME": "NOME NOMINAL DE TESTE",
        "NOMEMAE": "MAE NOMINAL DE TESTE",
        "ENDERECO": "ENDERECO NOMINAL DE TESTE",
        "R_DTNASC": "01011980",
        "SEXO": "F",
        "BAIRRO": "BAIRRO DE TESTE",
        "nota final": 7.5,
        "NOME qtd frag iguais": 0.5,
        "DTNASC dt iguais": 1,
        "nome_score_total": 0.75,
        "mae_missing": True,
        "macd_nasc_diff_capped": 30,
        "p_cal": 0.6,
        "band": "grey",
        "base_choice": "LLM_REVIEW",
        "evr": 0.2,
        "guardrail": "date_check",
        "guardrail_reason": "date_difference",
    })

    dossier = build_dossier(row)
    payload = dossier.to_dict()
    assert set(payload) == {"schema_version", "pair_id", "features", "model_outputs"}
    assert payload["pair_id"] == make_pair_id(12345, 6789, 11)
    assert payload["features"]["nota_final"] == 7.5
    assert payload["features"]["dob"] == {"DTNASC dt iguais": 1.0}
    assert payload["features"]["flags"] == {"mae_missing": True}
    assert payload["features"]["macd"] == {"macd_nasc_diff_capped": 30.0}
    assert payload["model_outputs"] == {
        "p_cal": 0.6,
        "band": "grey",
        "base_choice": "LLM_REVIEW",
        "evr": 0.2,
        "guardrail": "date_check",
        "guardrail_reason": "date_difference",
    }
    serialized = dossier.to_json()
    for excluded in [
        "COMPREC", "REFREC", "PASSO", "NOME NOMINAL DE TESTE",
        "MAE NOMINAL DE TESTE", "ENDERECO NOMINAL DE TESTE", "01011980",
        "SEXO", "BAIRRO DE TESTE",
    ]:
        assert excluded not in serialized


@pytest.mark.parametrize("column", ["observacao", "extra", "auxiliar,String"])
@pytest.mark.parametrize(
    "sentinel",
    ["PESSOA SINTETICA SOB COLUNA INOCUA", "1979-12-31"],
)
def test_dossier_excludes_pii_values_under_innocuous_columns(column, sentinel):
    row = pd.Series({"nota final": 7.5, "p_cal": 0.6})
    baseline = build_dossier(row, pair_id="synthetic-pair").to_dict()
    row[column] = sentinel

    dossier = build_dossier(row, pair_id="synthetic-pair")

    assert dossier.to_dict() == baseline
    assert sentinel not in dossier.to_json()


def test_historical_extract_sends_exactly_fourteen_real_subscores():
    groups = {
        "name": ["NOME qtd frag iguais", "NOME prim frag igual", "NOME ult frag igual"],
        "mother": ["NOMEMAE qtd frag iguais", "NOMEMAE prim frag igual", "NOMEMAE ult frag igual"],
        "dob": ["DTNASC dt iguais"],
        "address": ["ENDERECO via igual", "ENDERECO via prox", "ENDERECO numero igual",
                    "ENDERECO compl prox", "ENDERECO texto prox", "ENDERECO tokens jacc"],
    }
    sent = [column for columns in groups.values() for column in columns]
    row = pd.Series({column: (i + 1) / 20 for i, column in enumerate(sent)})
    row["CODMUNRES local igual"] = 0.75
    for absent in ("NOME qtd frag raros", "NOME qtd frag comuns", "NOME qtd frag muito parec",
                   "NOME qtd frag abrev", "NOMEMAE qtd frag raros", "NOMEMAE qtd frag comuns",
                   "NOMEMAE qtd frag muito parec", "NOMEMAE qtd frag abrev", "DTNASC dt ap 1digi",
                   "DTNASC dt inv dia", "DTNASC dt inv mes", "DTNASC dt inv ano",
                   "CODMUNRES uf igual", "CODMUNRES uf prox", "CODMUNRES local prox"):
        row[absent] = 0.9
    features = build_dossier(row).to_dict()["features"]
    assert len(row) == 29
    assert sum(len(features[group]) for group in groups) + 1 == 14
    for group, columns in groups.items():
        assert features[group] == {column: row[column] for column in columns}
    assert features["municipality"] == 0.75


def test_nonexistent_historical_columns_are_not_a_dossier_extension():
    row = pd.Series({"nota final": 7.5})
    baseline = build_dossier(row).to_dict()
    for column in ("NOME prim ult frag igual", "NOMEMAE prim ult frag igual",
                   "DTNASC ano igual", "DTNASC mes igual", "DTNASC dia igual"):
        row[column] = 1.0
    assert build_dossier(row).to_dict() == baseline
