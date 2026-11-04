"""Verifica a fronteira entre os campos de origem e o dossiê de revisão."""

import pandas as pd

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
        "DTNASC ano igual": 1,
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
