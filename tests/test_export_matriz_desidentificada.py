"""Fronteira de informação da matriz, incluindo os valores serializados."""

import hashlib
import hmac
import json

import pandas as pd
import pytest

from gzcmd.llm_dossier import make_pair_id
from scripts import export_matriz_desidentificada as export
from scripts.feature_engineering import SCORE_COLS

TEST_KEY = "a1" * 32


@pytest.fixture
def source():
    # Cardinalidades dos IDs: 4 e 5; os campos autorizados têm 1, 3 ou 6.
    frame = pd.DataFrame({col: ["0,25"] * 6 for col in SCORE_COLS})
    frame["COMPREC"] = ["COMP_S1_91", "COMP_S1_92", "COMP_S1_93",
                        "COMP_S1_94", "COMP_S1_95", "COMP_S1_91"]
    frame["REFREC"] = ["REF_S1_81", "REF_S1_82", "REF_S1_83",
                       "REF_S1_84", "REF_S1_81", "REF_S1_82"]
    frame["PASSO"] = ["6", "6", "9", "9", "11", "11"]
    frame["PAR"] = ["0", "1", "2", "0", "0", "0"]
    for col, value in {
        "R_DTNASC": "19870523", "C_DTNASC": "23/05/1987",
        "R_DTOBITO": "20210814", "C_SEXO": "F_SENTINELA",
        "R_BAIRES": "BAIRRO_SENTINELA", "C_IDPESSOA": "PESSOA_SENTINELA",
        "observacao": "NOME_NOMINAL_SENTINELA",
    }.items():
        frame[col] = value
    return frame


def matrix_for(source, monkeypatch):
    monkeypatch.setenv("PAIR_ID_HMAC_KEY", TEST_KEY)
    return export.build_matrix(
        source, export.load_hmac_key(), expected_rows=6, expected_links=2
    )


def test_allowlist_and_serialized_values(source, monkeypatch):
    matrix = matrix_for(source, monkeypatch)
    assert list(matrix.columns) == export.OUTPUT_COLS
    assert len(matrix.columns) == 32
    assert len(SCORE_COLS) == 29
    assert "CODMUNRES uf prox" not in matrix
    assert matrix["PAR"].tolist() == [0, 1, 2, 0, 0, 0]
    assert matrix["nota final"].eq(0.25).all()
    forbidden_cols = set(source.columns) - set(SCORE_COLS) - {"PASSO", "PAR"}
    sentinels = {str(v) for col in forbidden_cols for v in source[col]}
    serialized_json = matrix.to_json(orient="records", force_ascii=False)
    serialized_csv = matrix.to_csv(index=False)
    serialized_values = {
        str(value) for row in json.loads(serialized_json) for value in row.values()
    }
    assert sentinels.isdisjoint(serialized_values)
    for sentinel in sentinels | {TEST_KEY}:
        assert sentinel not in serialized_json
        assert sentinel not in serialized_csv


def test_hmac_deterministic_secret_dependent_and_truncated(source, monkeypatch):
    matrix = matrix_for(source, monkeypatch)
    row = source.iloc[0]
    raw = f"{row.COMPREC}|{row.REFREC}|{row.PASSO}".encode()
    expected = hmac.new(bytes.fromhex(TEST_KEY), raw, hashlib.sha256).hexdigest()[:16]
    assert matrix.iloc[0].pair_id_hmac == expected
    pd.testing.assert_frame_equal(matrix, matrix_for(source, monkeypatch))
    assert matrix["pair_id_hmac"].str.len().eq(16).all()
    assert matrix["pair_id_hmac"].is_unique
    assert expected != make_pair_id(row.COMPREC, row.REFREC, row.PASSO)
    assert expected != export.make_pair_id_hmac(
        row.COMPREC, row.REFREC, row.PASSO, bytes.fromhex("b2" * 32)
    )


def test_env_and_dotenv_key_loading(tmp_path, monkeypatch):
    monkeypatch.delenv("PAIR_ID_HMAC_KEY", raising=False)
    env_file = tmp_path / ".env"
    with pytest.raises(ValueError, match="PAIR_ID_HMAC_KEY ausente"):
        export.load_hmac_key(env_file)
    env_file.write_text(
        f'OTHER=ignored\nPAIR_ID_HMAC_KEY="{TEST_KEY}"\n', encoding="utf-8"
    )
    assert export.load_hmac_key(env_file) == bytes.fromhex(TEST_KEY)
    monkeypatch.setenv("PAIR_ID_HMAC_KEY", "b2" * 32)
    assert export.load_hmac_key(env_file) == bytes.fromhex("b2" * 32)
    monkeypatch.setenv("PAIR_ID_HMAC_KEY", "segredo_invalido_nao_exibir")
    with pytest.raises(ValueError, match="64 caracteres") as exc:
        export.load_hmac_key(env_file)
    assert "segredo_invalido_nao_exibir" not in str(exc.value)


@pytest.mark.parametrize("date", ["19870523", "23/05/1987"])
def test_rejects_date_under_allowed_column(source, monkeypatch, date):
    source.loc[0, SCORE_COLS[0]] = date
    with pytest.raises(ValueError, match="data detectada"):
        matrix_for(source, monkeypatch)


def test_rejects_identifier_cardinality_under_allowed_column(source, monkeypatch):
    source[SCORE_COLS[0]] = ["0", "1", "2", "3", "0", "1"]
    with pytest.raises(ValueError, match="Cardinalidade"):
        matrix_for(source, monkeypatch)


@pytest.mark.parametrize(
    "problem", ["missing", "duplicate", "label", "text", "links", "rows"]
)
def test_rejects_invalid_input(source, monkeypatch, problem):
    if problem == "missing":
        source = source.drop(columns=[SCORE_COLS[0]])
    elif problem == "duplicate":
        source.loc[5, ["COMPREC", "REFREC", "PASSO"]] = source.loc[
            0, ["COMPREC", "REFREC", "PASSO"]
        ]
    elif problem == "label":
        source.loc[0, "PAR"] = "3"
    elif problem == "text":
        source.loc[0, SCORE_COLS[0]] = "NOME_NOMINAL_SENTINELA"
    elif problem == "links":
        source["PAR"] = "0"
    elif problem == "rows":
        source = source.iloc[:-1]
    with pytest.raises(ValueError):
        matrix_for(source, monkeypatch)


def test_output_schema_rejects_added_column(source, monkeypatch):
    matrix = matrix_for(source, monkeypatch)
    matrix["observacao"] = "NOME_NOMINAL_SENTINELA"
    with pytest.raises(ValueError, match="lista permitida"):
        export.validate_matrix(matrix, source)


def test_csv_dbf_headers_and_metadata(source, monkeypatch, tmp_path):
    monkeypatch.setenv("PAIR_ID_HMAC_KEY", TEST_KEY)
    # O CLI continua fixo no extrato completo; só o construtor é adaptado no teste.
    original = export.build_matrix

    def synthetic_matrix(frame, key):
        return original(frame, key, expected_rows=6, expected_links=2)

    monkeypatch.setattr(export, "build_matrix", synthetic_matrix)
    source.columns = [f"{col},C,12,0" for col in source.columns]
    extrato = tmp_path / "extrato.csv"
    saida = tmp_path / "export" / "matriz_desidentificada.csv"
    source.to_csv(extrato, sep=";", index=False)
    metadata = export.export_matrix(extrato, saida)
    assert metadata["n"] == 6
    assert metadata["vinculos"] == 2
    assert metadata["colunas"] == export.OUTPUT_COLS
    assert metadata["sha256"] == hashlib.sha256(saida.read_bytes()).hexdigest()
    meta_path = saida.with_name("matriz_desidentificada_meta.json")
    assert json.loads(meta_path.read_text(encoding="utf-8")) == metadata
    assert TEST_KEY not in meta_path.read_text(encoding="utf-8")
    assert TEST_KEY not in saida.read_text(encoding="utf-8")
