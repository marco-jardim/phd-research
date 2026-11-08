"""Regressões das tabelas editoriais de vínculos no fechamento da banca."""

from pathlib import Path
import re

from scripts import tabelas_banca_b_editorial as editorial
from terminologia import contem


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "text/latex/tables"


def test_perfil_editorial_preserva_celulas_e_fonte_historica(monkeypatch, tmp_path):
    historical = TABLES / "tab_perfil_recuperados.tex"
    original = historical.read_bytes()
    monkeypatch.setattr(editorial, "OUT", tmp_path)
    editorial.tabela_perfil_recuperados()
    generated = (tmp_path / historical.name).read_bytes()
    assert historical.read_bytes() == original
    assert generated == (TABLES / "editorial" / historical.name).read_bytes()
    source = original.decode("utf-8")
    current = generated.decode("utf-8")
    expected = re.findall(r"^.* & .*", source, re.M)
    assert re.findall(r"^.* & .*", current, re.M) == expected
    assert "data/perfil_obitos_agregado.csv" in current
    assert "data/epidemiological_profile.json" in current
    assert "Valores históricos arredondados" in current
    assert contem(current, "Perfil dos vínculos recuperados")  # migrado: terminologia 3.6
    assert "Versão editorial da tabela histórica; células preservadas." not in current
    assert r"\input{tables/editorial/tab_perfil_recuperados.tex}" in (
        ROOT / "text/latex/estudo3_estrategias.tex"
    ).read_text(encoding="utf-8")


def test_impacto_editorial_usa_vinculos_e_custo_inverso_vpp(monkeypatch, tmp_path):
    monkeypatch.setattr(editorial, "OUT", tmp_path)
    editorial.tabela_impacto()
    generated = (tmp_path / "tab_impacto_epidemiologico.tex").read_bytes()
    assert generated == (TABLES / "editorial/tab_impacto_epidemiologico.tex").read_bytes()
    text = generated.decode("utf-8")
    assert contem(text, "vínculos detectados")  # migrado: terminologia 3.6
    assert contem(text, "custo por vínculo (inverso do VPP)")  # migrado: terminologia 3.6
    assert "óbitos detectados" not in text and "custo por óbito" not in text
