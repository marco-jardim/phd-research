"""Keep the historical name distinct from implemented statistical guarantees."""

from pathlib import Path

import pytest


LATEX = Path(__file__).resolve().parents[1] / "text" / "latex"
EXPANSION = (
    "Grey-Zone: Calibrated Mixture-of-experts + Conformal Deferral + "
    "Constrained Matching"
)


@pytest.mark.parametrize("filename", ["glossario.tex", "siglas.tex"])
def test_gzcmd_entry_records_historical_name_and_implementation_limits(filename):
    text = (LATEX / filename).read_text(encoding="utf-8")
    entry = next(line for line in text.splitlines() if line.startswith(
        r"\item[GZ-CMD]" if filename == "glossario.tex" else "GZ-CMD "
    ))
    assert EXPANSION in entry
    for term in (
        "nome histórico",
        "Platt scaling",
        "isotônica",
        "sem mecanismo de seleção aprendido",
        "sem predição conformal nem garantia de cobertura",
        "guardas determinísticas",
        "orçamentos de revisão",
        "não impõe cardinalidade",
    ):
        assert term in entry
    assert "N:1" in entry
    assert not any(dash in entry for dash in ("—", "–", "--"))
