from pathlib import Path

from scripts.figura_curva_esforco import ARM_SLIDES, ARM_THESIS, WIDTH, generate

ROOT = Path(__file__).resolve().parents[1]
CURVE = ROOT / "data/comparador_mesma_carga_curva.csv"
COMPARATOR = ROOT / "data/comparador_mesma_carga.csv"
THESIS_FIGURE = ROOT / "text/latex/figures/curva_esforco.pgf"
SLIDES_FIGURE = ROOT / "text/slides/figures/curva_esforco.pgf"


def build(tmp_path, arm):
    return generate(CURVE, COMPARATOR, tmp_path / f"{arm}.pgf", arm).read_text(encoding="utf-8")


def test_thesis_figure_uses_full_names(tmp_path):
    content = build(tmp_path, ARM_THESIS)
    for expected in ("nota final do CdR", "fora da amostra", "escore do OpenRecLink",
                     "falsos positivos",
                     "Pares verdadeiros recuperados abaixo do corte, revisor perfeito"):
        assert expected in content
    for forbidden in ("nota do CdR", "(OOF)", "PAR=2", " FP"):
        assert forbidden not in content


def test_thesis_output_matches_versioned_figure(tmp_path):
    output = generate(CURVE, COMPARATOR, tmp_path / "t.pgf", ARM_THESIS)
    assert output.read_bytes() == THESIS_FIGURE.read_bytes()


def test_legacy_slides_output_is_unchanged(tmp_path):
    output = generate(CURVE, COMPARATOR, tmp_path / "s.pgf", ARM_SLIDES)
    assert output.read_bytes() == SLIDES_FIGURE.read_bytes()
    legacy = output.read_text(encoding="utf-8")
    assert "PAR=2" in legacy and "(OOF)" in legacy and "21 FP" in legacy


def test_confirmation_annotation_is_anchored_inside_the_plot_area(tmp_path):
    content = build(tmp_path, ARM_THESIS)
    anchored = [line for line in content.splitlines() if line.startswith(r"\pgftransformshift")]
    assert anchored == [rf"\pgftransformshift{{\pgfpoint{{{WIDTH:.5f}cm}}{{0.17000cm}}}}"]
    node = [line for line in content.splitlines() if "anotacaoconfirmacao}{" in line and "pgfnode" in line]
    assert len(node) == 1 and node[0].startswith(r"\pgfnode{rectangle}{east}")
    assert "4 falsos positivos, +41 LLM" in node[0]
    # o conector parte da âncora oeste do texto
    assert r"\pgfpointanchor{anotacaoconfirmacao}{west}" in content
