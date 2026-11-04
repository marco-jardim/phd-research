"""A auditoria usa a proveniência canônica e verifica a coluna de vínculos."""

import pandas as pd
import pytest

from scripts import audit_tables


@pytest.mark.parametrize("alterar_vinculos", [False, True])
def test_score_bands_proveniencia(tmp_path, monkeypatch, alterar_vinculos):
    df = pd.DataFrame({
        "band": ["0-3", "3-5", "5-6", "6-7", "7-8", "8-9", "9-10", "10+"],
        "n": [5336, 34645, 14263, 6175, 1080, 102, 44, 51],
        "vinculos": [0, 3, 17, 57, 43, 34, 43, 50],
    })
    if alterar_vinculos:
        df.loc[0, "vinculos"] = 1
    df.to_csv(tmp_path / "tab_score_bands_proveniencia.csv", index=False)
    monkeypatch.setattr(audit_tables, "DATA_DIR", tmp_path)
    monkeypatch.setattr(audit_tables, "_results", [])

    audit_tables.audit_score_bands(9)

    result = audit_tables._results[0]
    assert result["source"] == "data/tab_score_bands_proveniencia.csv"
    assert result["status"] == ("DIVERGENCIA" if alterar_vinculos else "OK")
    assert len(result["checks"]) == 16
    assert len(result["divergences"]) == int(alterar_vinculos)
