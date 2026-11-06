"""Regressões das tabelas editoriais de vínculos no fechamento da banca."""

from pathlib import Path
import re

from scripts import tabelas_banca_b_editorial as editorial


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
    expected = [row for row in re.findall(r"^.* & .*", source, re.M)
                if not row.startswith(("Idade (", "Sexo masculino", "Nota final (média)"))]
    assert re.findall(r"^.* & .*", current, re.M) == expected
    assert "As médias de idade e da nota e a distribuição por sexo não são apresentadas" in current
    assert "não há agregado versionado" in current
    assert "Perfil dos vínculos recuperados" in current
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
    assert "vínculos detectados" in text
    assert "custo por vínculo (inverso do VPP)" in text
    assert "óbitos detectados" not in text and "custo por óbito" not in text
