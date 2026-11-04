"""A auditoria deve produzir apenas contagens, inclusive para células raras."""
import importlib.util
from pathlib import Path

import pandas as pd


spec = importlib.util.spec_from_file_location(
    "zenodo_conteudo", Path(__file__).resolve().parents[1] / "scripts/verificar_zenodo_conteudo.py"
)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_k_statistics_includes_missing_and_only_observed_categories():
    df = pd.DataFrame({
        "idade": pd.Categorical(["adulto"] * 6 + [None], categories=["adulto", "idoso"]),
        "y": [0] * 5 + [1, 1],
    })
    result = audit.k_statistics(df, ["idade", "y"])
    assert result == {
        "esquema_k": "idade|y", "k_min_qw3": 1, "n_celulas_qw3": 3,
        "n_celulas_k_menor_5": 2, "n_linhas_k_menor_5": 2,
    }
    assert "adulto" not in str(result)


def test_header_only_does_not_consume_data_values():
    raw = b'pair_id;SEXO\nprivate-id;F\n'
    header = audit.frame(raw, header_only=True)
    assert list(header.columns) == ["pair_id", "SEXO"]
    assert header.empty
    audit.rows.clear()
    audit.inventory("1.0.0", "benchmark_sim_sinan_tb.csv", raw, header_only=True)
    assert audit.rows[0]["n_linhas"] == ""
    assert "private-id" not in str(audit.rows)


def test_inventory_pair_ids_exports_only_cardinality(capsys):
    audit.rows.clear()
    audit.inventory("teste", "pares.csv", b'pair_id,score\nprivate-a,0.1\nprivate-a,0.2\nprivate-b,0.3\n')
    assert audit.rows[0]["n_pair_id_distintos"] == 2
    assert audit.rows[0]["n_linhas"] == 3
    assert "private-" not in str(audit.rows)
    assert "private-" not in capsys.readouterr().out
