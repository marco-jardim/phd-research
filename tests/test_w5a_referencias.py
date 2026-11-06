"""Correções bibliográficas documentadas e expansão histórica, sem build."""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "text/latex"


def entries(name):
    text = (LATEX / name).read_text(encoding="utf-8")
    result = {}
    for block in re.split(r"(?m)(?=^@)", text):
        # Diretivas BibTeX sem chave (por exemplo @comment) não são entradas.
        match = re.match(r"@\w+\{([^,]+),", block)
        if match:
            result[match[1]] = block
    return result


@pytest.mark.parametrize("name", ["referencias.bib", "referencias_abnt.bib"])
def test_verified_authorship_links_and_access_dates(name):
    data = entries(name)
    if "Selig2010vigilancia" in data:
        entry = data["Selig2010vigilancia"]
        assert "Kritski, Afranio Lineu" in entry
        assert "Carvalho, Regina Maria Guedes de" in entry
        assert "10.1590/S0034-89102010000600012" in entry
        assert "Ricardo Montes" not in entry
    expected = {
        "Arruda2026internacoesTB": "http://hdl.handle.net/11422/28461",
        "Ferreira2021tuberculoseAPS": "https://objdig.ufrj.br/96/teses/918619.pdf",
        "Brasil2024tb": "https://www.gov.br/saude/pt-br/centrais-de-conteudo/publicacoes/boletins/epidemiologicos/especiais/2024/boletim-epidemiologico-de-tuberculose-numero-especial-mar-2024.pdf",
    }
    for key, url in expected.items():
        if key in data:
            assert "url = {" + url + "}" in data[key]
            assert "note = {Acesso em: 6 nov. 2026}" in data[key]
    if "Arruda2026internacoesTB" in data:
        assert "Arruda, Josefa Kayllanne da Silva" in data["Arruda2026internacoesTB"]
        assert "pages =" not in data["Arruda2026internacoesTB"]
    assert "7 nov. 2026" not in (LATEX / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["capitulo1.tex", "estudo4_gzcmd.tex"])
def test_expansion_is_historical_and_points_to_existing_glossary(name):
    expansion = r"\textit{Grey-Zone: Calibrated Mixture-of-experts + Conformal Deferral + Constrained Matching}"
    source = (LATEX / name).read_text(encoding="utf-8")
    line = next(line for line in source.splitlines() if expansion in line)
    assert "nome histórico cuja correspondência com os componentes implementados é discutida no Glossário" in line
    assert not any(dash in line for dash in ("—", "–", "--"))
    glossary = (LATEX / "glossario.tex").read_text(encoding="utf-8")
    assert expansion in glossary
    assert "sem predição conformal nem garantia de cobertura" in glossary
    assert "não impõe cardinalidade" in glossary
