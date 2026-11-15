import csv
import re
from pathlib import Path

import pytest

from scripts import figura_incerteza_ordenacoes as fig

ROOT = Path(__file__).resolve().parents[1]
FIGURE = ROOT / "text/slides/figures/incerteza_ordenacoes.pgf"
FIELDS = ["comparacao", "meta_nominal", "fracao_se_menos_36", "replicas_validas",
          "proporcao_fusao_menor", "diferenca_revisoes_mediana",
          "diferenca_revisoes_ic95_inf", "diferenca_revisoes_ic95_sup"]


def make_rows(comparison, values):
    """values: meta -> (mediana, inferior, superior)."""
    return [{"comparacao": comparison, "meta_nominal": str(goal), "fracao_se_menos_36": "0",
             "replicas_validas": "2000", "proporcao_fusao_menor": "0.5",
             "diferenca_revisoes_mediana": str(m), "diferenca_revisoes_ic95_inf": str(lo),
             "diferenca_revisoes_ic95_sup": str(hi)} for goal, (m, lo, hi) in values.items()]


def synthetic(o3=None, f3=None, f4=None):
    o3 = o3 or {9: (-2, -5, -1), 23: (-20, -50, -10), 36: (-30, -80, -5)}
    f3 = f3 or {9: (-3, -6, -1), 23: (-25, -60, -12), 36: (-40, -90, -6)}
    f4 = f4 or {9: (-4, -7, -2), 23: (-30, -70, -14), 36: (-50, -95, -8)}
    return make_rows("O3-O1", o3), make_rows("F3-O1", f3) + make_rows("F4-O1", f4)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_generator_is_deterministic_and_lf(tmp_path):
    ordenacao = ROOT / "data/ordenacao_bootstrap_o3_o1.csv"
    fusao = ROOT / "data/fusao_robustez_bootstrap.csv"
    first = fig.generate(ordenacao, fusao, tmp_path / "a/fig.pgf").read_bytes()
    second = fig.generate(ordenacao, fusao, tmp_path / "b/fig.pgf").read_bytes()
    assert first == second
    assert b"\r" not in first


def test_values_come_from_the_csvs(tmp_path):
    o3, fusion = synthetic()
    base = fig.render(o3, fusion)
    assert base == fig.render(*synthetic())
    changed = synthetic(o3={9: (-2, -5, -1), 23: (-20, -50, -10), 36: (-31, -80, -5)})
    assert fig.render(*changed) != base
    changed = synthetic(f4={9: (-4, -7, -2), 23: (-30, -70, -14), 36: (-50, -95, -9)})
    assert fig.render(*changed) != base
    # os CSVs em disco passam pelo mesmo caminho de leitura
    ordenacao = write_csv(tmp_path / "o.csv", o3)
    fusao = write_csv(tmp_path / "f.csv", fusion)
    output = fig.generate(ordenacao, fusao, tmp_path / "fig.pgf")
    assert output.read_text(encoding="utf-8") == base


def test_panels_use_the_three_contrasts_and_goals():
    panels = fig.load_panels(*synthetic())
    assert [goal for goal, _ in panels] == [9, 23, 36]
    for _, rows in panels:
        assert [label for label, _, _ in rows] == ["O3", "F3", "F4"]
    assert panels[0][1][2][2] == (-4.0, -7.0, -2.0)


def test_missing_contrast_or_goal_raises():
    o3, fusion = synthetic()
    with pytest.raises(ValueError, match="ausente"):
        fig.render([r for r in o3 if r["meta_nominal"] != "23"], fusion)
    with pytest.raises(ValueError, match="ausente"):
        fig.render(o3, [r for r in fusion if r["comparacao"] != "F4-O1"])
    with pytest.raises(ValueError, match="ausente"):
        fig.render(o3, [r for r in fusion if not (r["comparacao"] == "F3-O1" and r["meta_nominal"] == "9")])
    with pytest.raises(ValueError, match="duplicado"):
        fig.render(o3, fusion + [fusion[0]])
    with pytest.raises(ValueError, match="inconsistente"):
        fig.render(make_rows("O3-O1", {9: (-2, -1, -5), 23: (-20, -50, -10), 36: (-30, -80, -5)}), fusion)


def test_marker_fill_follows_sign_of_upper_limit():
    assert fig.marker_filled(-0.5) is True
    assert fig.marker_filled(0.0) is False
    assert fig.marker_filled(13.0) is False
    o3, fusion = synthetic()
    assert "\\pgfsetfillcolor{white}" not in fig.render(o3, fusion)
    mixed, fusion = synthetic(o3={9: (-2, -5, 1), 23: (-20, -50, -10), 36: (-30, -80, -5)})
    content = fig.render(mixed, fusion)
    assert content.count("\\pgfsetfillcolor{white}") == 1
    assert content.count("\\pgfusepath{fill,stroke}") == 1
    assert content.count("\\pgfsetfillcolor{blue}") == 2


def test_truncation_rule_and_annotation():
    values = [(579.5, -1962.0, 23566.3), (-894.0, -2243.0, -164.0), (-1203.0, -2049.0, -63.0)]
    assert fig.truncated_limits(values) == {(0, 2)}
    assert fig.truncated_limits([(-21.0, -52.0, 13.0), (-35.0, -59.0, -14.0), (-38.0, -61.0, -18.0)]) == set()
    low, high = fig.axis_range(values, {(0, 2)})
    assert high < 1000 and low > -2500
    outlier = synthetic(o3={9: (-2, -5, -1), 23: (-20, -50, -10), 36: (580, -1962, 23566.3)},
                        f3={9: (-3, -6, -1), 23: (-25, -60, -12), 36: (-894, -2243, -164)},
                        f4={9: (-4, -7, -2), 23: (-30, -70, -14), 36: (-1203, -2049, -63)})
    content = fig.render(*outlier)
    assert content.count("até +23.566") == 1
    assert content.count(r"\pgfpathclose\pgfusepath{fill}") == 1
    assert "até" not in fig.render(*synthetic())
    # truncamento no limite inferior também é anotado, com sinal de menos
    negative = synthetic(o3={9: (-2, -5, -1), 23: (-20, -50, -10), 36: (-30, -80000, -5)})
    assert "até $-$80.000" in fig.render(*negative)


def test_text_is_portuguese_without_dashes():
    content = fig.render(*synthetic())
    for goal in (9, 23, 36):
        assert f"Meta de {goal} pares verdadeiros" in content
    assert "Diferença de revisões frente a O1 (mediana e IC 95\\%)" in content
    for label in ("O3", "F3", "F4"):
        assert f"{{\\scriptsize {label}}}" in content
    assert "\u2014" not in content and "\u2013" not in content
    real = FIGURE.read_text(encoding="utf-8") if FIGURE.exists() else ""
    assert "\u2014" not in real and "\u2013" not in real


def test_number_format_uses_dot_thousands_and_math_minus():
    assert fig.thousands(23566.3) == "23.566"
    assert fig.signed(-1006) == "$-$1.006"
    assert fig.signed(327.02, plus=True) == "+327"
    assert fig.signed(0) == "0"


def test_cli_custom_destination(tmp_path):
    target = tmp_path / "x/fig.pgf"
    fig.main(["--output", str(target)])
    assert target.read_bytes() == fig.render(
        fig.read_rows(ROOT / "data/ordenacao_bootstrap_o3_o1.csv"),
        fig.read_rows(ROOT / "data/fusao_robustez_bootstrap.csv")).encode("utf-8")


def test_cli_default_destination_is_the_slides_figure(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    for name in ("ordenacao_bootstrap_o3_o1.csv", "fusao_robustez_bootstrap.csv"):
        (tmp_path / "data" / name).write_bytes((ROOT / "data" / name).read_bytes())
    monkeypatch.setattr(fig, "ROOT", tmp_path)
    fig.main([])
    assert (tmp_path / "text/slides/figures/incerteza_ordenacoes.pgf").read_bytes() == FIGURE.read_bytes()


def test_versioned_figure_matches_generator(tmp_path):
    output = fig.generate(ROOT / "data/ordenacao_bootstrap_o3_o1.csv",
                          ROOT / "data/fusao_robustez_bootstrap.csv", tmp_path / "fig.pgf")
    assert output.read_bytes() == FIGURE.read_bytes()
    assert b"\r\n" not in FIGURE.read_bytes()
    assert "text/slides/figures/incerteza_ordenacoes.pgf text eol=lf" in (ROOT / ".gitattributes").read_text(encoding="utf-8")


def test_real_values_match_csv_and_expected():
    panels = {goal: {label: data for label, _, data in rows}
              for goal, rows in fig.load_panels(
                  fig.read_rows(ROOT / "data/ordenacao_bootstrap_o3_o1.csv"),
                  fig.read_rows(ROOT / "data/fusao_robustez_bootstrap.csv"))}
    assert round(panels[9]["O3"][0]) == -21 and round(panels[36]["O3"][2]) == 23566
    assert [round(v) for v in panels[36]["F4"]] == [-1203, -2049, -63]
    assert panels[36]["O3"][2] > 3 * 2243

def declared_box(content):
    match = re.search(r"\\pgfpathrectanglecorners\{\\pgfpoint\{(-?[\d.]+)cm\}\{(-?[\d.]+)cm\}\}"
                      r"\{\\pgfpoint\{(-?[\d.]+)cm\}\{(-?[\d.]+)cm\}\}\\pgfusepath\{use as bounding box\}", content)
    assert match, "a figura não declara caixa delimitadora"
    return tuple(float(value) for value in match.groups())


def test_bounding_box_is_declared_and_covers_axis_label_and_titles():
    for content in (fig.render(*synthetic()), FIGURE.read_text(encoding="utf-8")):
        xmin, ymin, xmax, ymax = declared_box(content)
        # a caixa é declarada antes de qualquer desenho
        assert content.splitlines()[2].endswith(r"\pgfusepath{use as bounding box}")
        total_width = fig.LEFT_MARGIN + 3 * fig.PANEL_WIDTH + 2 * fig.PANEL_GAP
        # rótulo único do eixo, de uma linha, centrado em AXIS_LABEL_Y
        axis = re.findall(r"\\pgftext\[at=\{\\pgfpoint\{([\d.]+)cm\}\{(-?[\d.]+)cm\}\}\]\{\\scriptsize Diferença", content)
        assert len(axis) == 1
        assert float(axis[0][1]) == fig.AXIS_LABEL_Y
        assert ymin <= fig.AXIS_LABEL_Y - fig.AXIS_LABEL_HALF_HEIGHT
        # título dos painéis: centro em PANEL_HEIGHT + TITLE_OFFSET, com meia altura de texto
        title_y = max(float(y) for y in re.findall(
            r"\\pgftext\[at=\{\\pgfpoint\{[\d.]+cm\}\{([\d.]+)cm\}\}\]\{\\scriptsize Meta de", content))
        assert title_y == pytest.approx(fig.PANEL_HEIGHT + fig.TITLE_OFFSET, abs=1e-5)
        assert ymax >= title_y + fig.TITLE_HALF_HEIGHT
        # rótulos das linhas à esquerda e último painel à direita
        assert xmin <= fig.LEFT_MARGIN - fig.ROW_LABEL_GAP - fig.ROW_LABEL_WIDTH
        assert xmax >= total_width + fig.TICK_LABEL_HALF_WIDTH
        # nenhum desenho de painel fica abaixo da caixa: marcas e números do eixo
        assert ymin < fig.TICK_LABEL_Y < fig.PANEL_HEIGHT


def test_single_axis_label_is_centered_under_the_three_panels():
    content = fig.render(*synthetic())
    assert content.count(fig.AXIS_LABEL) == 1
    assert "parbox" not in content
    total_width = fig.LEFT_MARGIN + 3 * fig.PANEL_WIDTH + 2 * fig.PANEL_GAP
    x = float(re.search(r"\\pgftext\[at=\{\\pgfpoint\{([\d.]+)cm\}\{[-\d.]+cm\}\}\]\{\\scriptsize Diferença", content).group(1))
    assert x == pytest.approx((fig.LEFT_MARGIN + total_width) / 2, abs=1e-5)
    assert fig.PANEL_WIDTH > fig.PANEL_HEIGHT


def test_truncation_annotation_stays_off_the_panel_border_and_marker():
    outlier = synthetic(o3={9: (-2, -5, -1), 23: (-20, -50, -10), 36: (580, -1962, 23566.3)},
                        f3={9: (-3, -6, -1), 23: (-25, -60, -12), 36: (-894, -2243, -164)},
                        f4={9: (-4, -7, -2), 23: (-30, -70, -14), 36: (-1203, -2049, -63)})
    content = fig.render(*outlier)
    match = re.search(r"\\pgftext\[right,at=\{\\pgfpoint\{([\d.]+)cm\}\{([\d.]+)cm\}\}\]\{\\scriptsize até \+23\.566", content)
    assert match
    right_border = fig.LEFT_MARGIN + 3 * fig.PANEL_WIDTH + 2 * fig.PANEL_GAP
    assert float(match.group(1)) == pytest.approx(right_border - fig.ANNOTATION_INSET, abs=1e-5)
    # o texto fica acima da linha do IC (marcador, raio de 2 pt) e abaixo do topo do painel
    row_y = fig.PANEL_HEIGHT * 3 / 4
    assert row_y + 0.2 < float(match.group(2)) < fig.PANEL_HEIGHT - 0.1
