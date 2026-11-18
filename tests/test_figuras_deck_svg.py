import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scripts import figuras_deck_svg as fig

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "text/slides/figures"
SVG_NS = "{http://www.w3.org/2000/svg}"
FORBIDDEN = ("nota do CdR", "OOF", "PAR=2", " FP", "\u2013", "\u2014")
CONTRAST_FIELDS = ["comparacao", "meta_nominal", "fracao_se_menos_36", "replicas_validas",
                   "proporcao_fusao_menor", "diferenca_revisoes_mediana",
                   "diferenca_revisoes_ic95_inf", "diferenca_revisoes_ic95_sup"]


# ---------------------------------------------------------------- dados sintéticos

def curve_rows(offset=0, n=40):
    rows = []
    for index, name in enumerate(("O1_SCORE", "O2_CdR", "O3_p_cal_ml", "O4_P_recalibrada")):
        for k in range(1, n + 1):
            rows.append({"ordenacao": name, "k": str(k), "par2_topk": str(min(36, k // (index + 2) + offset))})
    return rows


def comparator_rows(fp="21", llm="344"):
    def row(mode, k, y, fp_, llm_):
        return {"ordenacao": "O3_p_cal_ml", "modo": mode, "carga": "K1_humana",
                "P_pares_humano": str(k), "P_par2_recuperados": str(y), "P_fp": fp_, "P_pares_llm": llm_}
    return [row("vigilancia", 30, 12, fp, llm), row("confirmacao", 8, 3, "4", "41")]


def contrast_rows(comparison, values):
    return [{"comparacao": comparison, "meta_nominal": str(goal), "fracao_se_menos_36": "0",
             "replicas_validas": "2000", "proporcao_fusao_menor": "0.5",
             "diferenca_revisoes_mediana": str(m), "diferenca_revisoes_ic95_inf": str(lo),
             "diferenca_revisoes_ic95_sup": str(hi)} for goal, (m, lo, hi) in values.items()]


def contrasts(o3=None, f3=None, f4=None):
    o3 = o3 or {9: (-2, -5, -1), 23: (-20, -50, -10), 36: (-30, -80, -5)}
    f3 = f3 or {9: (-3, -6, -1), 23: (-25, -60, -12), 36: (-40, -90, -6)}
    f4 = f4 or {9: (-4, -7, -2), 23: (-30, -70, -14), 36: (-50, -95, -8)}
    return contrast_rows("O3-O1", o3), contrast_rows("F3-O1", f3) + contrast_rows("F4-O1", f4)


def fila_rows(n=30, bump=0, omit=None):
    rows = []
    for index, name in enumerate(("O1", "O3", "F3", "F4", "A1")):
        if name == omit:
            continue
        for k in range(1, n + 1):
            rows.append({"fila": name, "k": str(k),
                         "par2_acumulados": str(float(min(36, k // (index + 1) + (bump if name == "F4" else 0))))})
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture(scope="module")
def real():
    return fig.render_all()


def parse(svg):
    return ET.fromstring(svg)


# ---------------------------------------------------------------- determinismo e arquivos

def test_generator_is_deterministic_lf_utf8_without_bom(tmp_path):
    first = fig.generate(tmp_path / "a")
    second = fig.generate(tmp_path / "b")
    assert [p.name for p in first] == ["curva_esforco.svg", "incerteza_ordenacoes.svg", "curva_fusoes.svg"]
    for left, right in zip(first, second):
        data = left.read_bytes()
        assert data == right.read_bytes()
        assert b"\r" not in data and not data.startswith(b"\xef\xbb\xbf")
        data.decode("utf-8")
        assert data.endswith(b"</svg>\n")


def test_coordinates_have_at_most_two_decimals(real):
    for svg in real.values():
        geometry = re.sub(r"<(text|title|desc)\b[^>]*>.*?</\1>", "", svg, flags=re.DOTALL)
        assert not re.search(r"\d\.\d{3,}", geometry)


@pytest.mark.parametrize("name", fig.FIGURAS)
def test_versioned_files_match_generator_byte_for_byte(name, real):
    assert (FIGURES_DIR / f"{name}.svg").read_bytes() == real[name].encode("utf-8")


# ---------------------------------------------------------------- valores vêm dos CSVs

def test_effort_curve_values_come_from_the_csvs():
    base = fig.render_curva_esforco(curve_rows(), comparator_rows())
    assert base == fig.render_curva_esforco(curve_rows(), comparator_rows())
    assert fig.render_curva_esforco(curve_rows(offset=1), comparator_rows()) != base
    changed = fig.render_curva_esforco(curve_rows(), comparator_rows(fp="22", llm="345"))
    assert "30, 12), 22 falsos positivos, +345 LLM" in changed and changed != base
    assert "ASR vig.: (30, 12), 21 falsos positivos, +344 LLM" in base
    assert "ASR conf.: (8, 3), 4 falsos positivos, +41 LLM" in base


def test_incerteza_values_come_from_the_csvs(tmp_path):
    o3, fusion = contrasts()
    base = fig.render_incerteza(o3, fusion)
    changed = contrasts(o3={9: (-2, -5, -1), 23: (-20, -50, -10), 36: (-31, -80, -5)})
    assert fig.render_incerteza(*changed) != base
    changed = contrasts(f4={9: (-4, -7, -2), 23: (-30, -70, -14), 36: (-50, -95, -9)})
    assert fig.render_incerteza(*changed) != base
    # os CSVs em disco passam pelo mesmo caminho
    paths = {"curva": write_csv(tmp_path / "c.csv", curve_rows()),
             "comparador": write_csv(tmp_path / "m.csv", comparator_rows()),
             "ordenacao": write_csv(tmp_path / "o.csv", o3), "fusao": write_csv(tmp_path / "f.csv", fusion)}
    assert fig.render_incerteza(fig.pgf_inc.read_rows(paths["ordenacao"]),
                                fig.pgf_inc.read_rows(paths["fusao"])) == base
    assert fig.render_curva_esforco(fig.pgf_curva.read_rows(paths["curva"]),
                                    fig.pgf_curva.read_rows(paths["comparador"])) == \
        fig.render_curva_esforco(curve_rows(), comparator_rows())


def test_fusion_curve_values_come_from_the_csv():
    base = fig.render_curva_fusoes(fila_rows(), xmax=30)
    assert fig.render_curva_fusoes(fila_rows(bump=1), xmax=30) != base
    assert fig.render_curva_fusoes(fila_rows(), xmax=30) == base


# ---------------------------------------------------------------- erros explícitos

def test_incerteza_missing_contrast_goal_or_inconsistent_ci_raises():
    o3, fusion = contrasts()
    with pytest.raises(ValueError, match="ausente"):
        fig.render_incerteza([r for r in o3 if r["meta_nominal"] != "23"], fusion)
    with pytest.raises(ValueError, match="ausente"):
        fig.render_incerteza(o3, [r for r in fusion if r["comparacao"] != "F4-O1"])
    with pytest.raises(ValueError, match="ausente"):
        fig.render_incerteza(o3, [r for r in fusion if not (r["comparacao"] == "F3-O1" and r["meta_nominal"] == "9")])
    with pytest.raises(ValueError, match="duplicado"):
        fig.render_incerteza(o3, fusion + [fusion[0]])
    with pytest.raises(ValueError, match="inconsistente"):
        fig.render_incerteza(contrast_rows("O3-O1", {9: (-2, -1, -5), 23: (-20, -50, -10), 36: (-30, -80, -5)}),
                             fusion)


def test_effort_curve_missing_order_or_point_raises():
    with pytest.raises(ValueError, match="ausente"):
        fig.render_curva_esforco([r for r in curve_rows() if r["ordenacao"] != "O2_CdR"], comparator_rows())
    with pytest.raises(ValueError, match="sem ponto único"):
        fig.render_curva_esforco(curve_rows(), [r for r in comparator_rows() if r["modo"] != "confirmacao"])
    with pytest.raises(ValueError, match="sem ponto único"):
        fig.render_curva_esforco(curve_rows(), comparator_rows() + comparator_rows()[:1])


def test_fusion_curve_missing_queue_or_short_series_raises():
    with pytest.raises(ValueError, match="Fila ausente: F3"):
        fig.render_curva_fusoes(fila_rows(omit="F3"), xmax=30)
    with pytest.raises(ValueError, match="termina antes"):
        fig.render_curva_fusoes(fila_rows(n=20), xmax=30)
    broken = fila_rows()
    broken[5]["par2_acumulados"] = "99.0"
    with pytest.raises(ValueError, match="monotônica"):
        fig.render_curva_fusoes(broken, xmax=30)


# ---------------------------------------------------------------- marcadores e truncamento

def test_marker_fill_follows_sign_of_upper_limit():
    o3, fusion = contrasts()
    svg = fig.render_incerteza(o3, fusion)
    assert svg.count('class="cheio"') == 9 and 'class="vazado"' not in svg
    hollow = contrasts(f3={9: (-3, -6, 0), 23: (-25, -60, -12), 36: (-40, -90, -6)},
                       f4={9: (-4, -7, -2), 23: (-30, -70, 14), 36: (-50, -95, -8)})
    svg = fig.render_incerteza(*hollow)
    assert svg.count('class="vazado"') == 2 and svg.count('class="cheio"') == 7


def test_real_incerteza_markers_and_truncation(real):
    svg = real["incerteza_ordenacoes"]
    assert svg.count('class="cheio"') == 4 and svg.count('class="vazado"') == 5
    assert svg.count("até +") == 1 and "até +23.566" in svg


def test_truncation_is_annotated_on_the_side_of_the_outlier():
    o3, fusion = contrasts(o3={9: (-2, -5, -1), 23: (-20, -50, -10), 36: (-30, -80, 9000)})
    svg = fig.render_incerteza(o3, fusion)
    assert "até +9.000" in svg
    o3, fusion = contrasts(o3={9: (-2, -5, -1), 23: (-20, -50, -10), 36: (-30, -9000, -5)})
    svg = fig.render_incerteza(o3, fusion)
    assert f"até {fig.MINUS}9.000" in svg
    o3, fusion = contrasts()
    assert "até" not in fig.render_incerteza(o3, fusion)


# ---------------------------------------------------------------- textos

REQUIRED = {
    "curva_esforco": ["Pares verdadeiros recuperados abaixo do corte, revisor perfeito",
                      "Revisões humanas (escala logarítmica)", "O1: escore do OpenRecLink",
                      "O2: nota final do CdR", "O3: probabilidade calibrada fora da amostra",
                      "O4: probabilidade recalibrada de ASR", "Eixo limitado a 2.522 revisões",
                      "ASR vig.: (96, 23), 21 falsos positivos, +344 LLM",
                      "ASR conf.: (17, 9), 4 falsos positivos, +41 LLM"],
    "incerteza_ordenacoes": ["Diferença de revisões frente a O1 (mediana e IC 95%)",
                             "Meta de 9 pares verdadeiros", "Meta de 23 pares verdadeiros",
                             "Meta de 36 pares verdadeiros", "até +23.566", f"{fig.MINUS}2.000"],
    "curva_fusoes": ["Pares verdadeiros recuperados abaixo do corte, revisor perfeito",
                     "Revisões humanas (escala logarítmica)", "O1: escore do OpenRecLink",
                     "O3: probabilidade calibrada fora da amostra", "F3: fusão de O3, O1 e O3 com MACD",
                     "F4: fusão de O3 e O1", "Eixo limitado a 2.522 revisões"],
}


@pytest.mark.parametrize("name", fig.FIGURAS)
def test_required_texts_present_and_forbidden_terms_absent(name, real):
    svg = real[name]
    for expected in REQUIRED[name]:
        assert expected in svg, expected
    for forbidden in FORBIDDEN:
        assert forbidden not in svg, forbidden
    visible = "".join(t.text or "" for t in parse(svg).iter(SVG_NS + "text"))
    assert not re.search(r"-\d", visible)  # sinal de menos matemático nos rótulos


def test_fusion_figure_has_no_asr_annotation(real):
    assert "ASR" not in real["curva_fusoes"]


def test_text_sizes_are_at_least_15_and_decimal_style_is_brazilian(real):
    for svg in real.values():
        sizes = [float(v) for v in re.findall(r'font-size="([\d.]+)"', svg)]
        assert sizes and min(sizes) >= 15
        assert not re.search(r"\d,\d{3}(?!\d)", "".join(t.text or "" for t in parse(svg).iter(SVG_NS + "text")))


# ---------------------------------------------------------------- estrutura do SVG

@pytest.mark.parametrize("name", fig.FIGURAS)
def test_svg_is_well_formed_self_contained_and_accessible(name, real):
    svg = real[name]
    root = parse(svg)
    assert root.tag == SVG_NS + "svg"
    assert root.get("viewBox") and root.get("width") == "100%" and root.get("height") is None
    assert root.get("role") == "img"
    assert root.get("class") == f"figura figura-{name}"
    assert root.get("font-family") == "inherit"
    title, desc = root.find(SVG_NS + "title"), root.find(SVG_NS + "desc")
    assert title is not None and title.text and desc is not None and len(desc.text) > 60
    tags = {element.tag.replace(SVG_NS, "") for element in root.iter()}
    assert not tags & {"script", "image", "foreignObject", "use", "link"}
    lowered = svg.lower()
    for marker in ("href", "http://", "https://", "url(", "@import", "<script", "<image"):
        if marker == "http://":
            assert lowered.count(marker) == 1  # só o namespace do SVG
        else:
            assert marker not in lowered
    ids = [e.get("id") for e in root.iter() if e.get("id")]
    assert ids and all(i.startswith(name + "-") for i in ids)
    for style in root.iter(SVG_NS + "style"):
        assert all(sel.strip().startswith(f".figura-{name}") for sel in style.text.split("{")[0].split(","))


def test_background_is_transparent_and_curves_use_distinct_dashes(real):
    for svg in real.values():
        assert "background" not in svg
    effort = real["curva_esforco"]
    dashes = set(re.findall(r'stroke-dasharray="([^"]*)"', effort))
    assert {fig.DASHED, fig.DOTTED, fig.DASHDOT} <= dashes
    fusion = real["curva_fusoes"]
    assert {fig.SHORTDASH, fig.DOTTED, fig.DASHDOTDOT} <= set(re.findall(r'stroke-dasharray="([^"]*)"', fusion))
    # F3 e F4 não repetem cor nem traço de nenhuma série O1 a O4 da curva de esforço.
    own_o = set(fig.CURVE_STYLE.values())
    assert fig.FUSION_STYLE["F3"] not in own_o and fig.FUSION_STYLE["F4"] not in own_o
    assert {c for c, _ in own_o}.isdisjoint({fig.FUSION_STYLE["F3"][0], fig.FUSION_STYLE["F4"][0]})
    assert {d for _, d in own_o}.isdisjoint({fig.FUSION_STYLE["F3"][1], fig.FUSION_STYLE["F4"][1]})


def test_effort_curves_are_steps_and_reach_the_axis_limits(real):
    root = parse(real["curva_esforco"])
    paths = [p.get("d") for p in root.iter(SVG_NS + "path")
             if p.get("stroke-width") == "3" and p.get("d", "").count("V") > 5]
    assert len(paths) == 4
    for d in paths:
        assert re.fullmatch(r"M[\d.]+,[\d.]+(H[\d.]+V[\d.]+)*H970", d)


# ---------------------------------------------------------------- conferência com a tese

def thesis_points():
    firsts, at96 = {}, {}
    for row in fig.stream_rows(fig.FILAS_CSV):
        name = row["fila"]
        if name not in ("O1", "O3", "F3", "F4"):
            continue
        k, y = int(row["k"]), float(row["par2_acumulados"])
        if k == 96:
            at96[name] = y
        if y >= 36 and name not in firsts:
            firsts[name] = k
    return at96, firsts


def test_fusion_curve_matches_the_thesis_checkpoints():
    at96, firsts = thesis_points()
    assert at96 == {"O1": 19.0, "O3": 22.0, "F3": 25.0, "F4": 18.0}
    assert firsts == {"O1": 2521, "F3": 1777, "F4": 1387, "O3": 23443}
    names = {row["fila"] for row in fig.stream_rows(fig.FILAS_CSV)}
    assert {"O1", "O3", "F3", "F4"} <= names
    series = fig.load_filas(fig.stream_rows_filtered(fig.FILAS_CSV))
    assert {name: dict(series[name])[96] for name in series} == {"O1": 19, "O3": 22, "F3": 25, "F4": 18}
    assert all(len(points) == 2522 for points in series.values())


def test_fusion_note_reports_the_o3_arrival_outside_the_axis(real):
    assert "O3 chega aos 36 pares verdadeiros em 23.443 revisões" in real["curva_fusoes"]


# ---------------------------------------------------------------- injeção em HTML

@pytest.fixture(scope="module")
def svgs():
    return {"curva_esforco": fig.render_curva_esforco(curve_rows(), comparator_rows()),
            "incerteza_ordenacoes": fig.render_incerteza(*contrasts()),
            "curva_fusoes": fig.render_curva_fusoes(fila_rows(), xmax=30)}


HTML = ("<html>\n<body>\n<h1>Deck</h1>\n  <!-- FIGURA:curva_esforco -->\nVELHO\n  <!-- /FIGURA:curva_esforco -->\n"
        "<p>texto <b>intacto</b></p>\n<!-- FIGURA:incerteza_ordenacoes -->\n<!-- /FIGURA:incerteza_ordenacoes -->\n"
        "<footer>fim</footer>\n</body>\n</html>\n")


def test_injetar_replaces_between_markers_and_keeps_the_rest(tmp_path, svgs):
    path = tmp_path / "deck.html"
    path.write_text(HTML, encoding="utf-8", newline="")
    done = fig.injetar(path, svgs)
    assert done == ["curva_esforco", "incerteza_ordenacoes"]  # curva_fusoes sem marcador só fica de fora
    out = path.read_text(encoding="utf-8")
    assert svgs["curva_esforco"] in out and svgs["incerteza_ordenacoes"] in out
    assert "VELHO" not in out and svgs["curva_fusoes"] not in out
    for kept in ("<h1>Deck</h1>", "<p>texto <b>intacto</b></p>", "<footer>fim</footer>",
                 "  <!-- FIGURA:curva_esforco -->\n", "  <!-- /FIGURA:curva_esforco -->\n",
                 "<!-- FIGURA:incerteza_ordenacoes -->\n", "<!-- /FIGURA:incerteza_ordenacoes -->\n"):
        assert kept in out
    # idempotente
    before = path.read_bytes()
    fig.injetar(path, svgs)
    assert path.read_bytes() == before


def test_injetar_preserves_crlf_html(tmp_path, svgs):
    path = tmp_path / "crlf.html"
    path.write_bytes(HTML.replace("\n", "\r\n").encode("utf-8"))
    fig.injetar(path, svgs)
    data = path.read_bytes()
    assert b"\n" not in data.replace(b"\r\n", b"")
    assert fig.verificar(path, svgs) == []


def test_injetar_rejects_opening_marker_without_closing(tmp_path, svgs):
    path = tmp_path / "bad.html"
    broken = "<p>a</p>\n<!-- FIGURA:curva_esforco -->\nVELHO\n<p>b</p>\n"
    path.write_text(broken, encoding="utf-8", newline="")
    with pytest.raises(ValueError, match="curva_esforco"):
        fig.injetar(path, svgs)
    with pytest.raises(ValueError, match="curva_esforco"):
        fig.verificar(path, svgs)
    assert path.read_text(encoding="utf-8") == broken  # nada foi gravado


def test_verificar_detects_out_of_sync_figures(tmp_path, svgs):
    path = tmp_path / "deck.html"
    path.write_text(HTML, encoding="utf-8", newline="")
    assert fig.verificar(path, svgs) == ["curva_esforco", "incerteza_ordenacoes"]
    fig.injetar(path, svgs)
    assert fig.verificar(path, svgs) == []
    stale = dict(svgs, incerteza_ordenacoes=svgs["incerteza_ordenacoes"].replace("Meta de 9", "Meta de 8"))
    assert fig.verificar(path, stale) == ["incerteza_ordenacoes"]


def test_cli_exit_codes(tmp_path, monkeypatch, svgs):
    monkeypatch.setattr(fig, "render_all", lambda *a, **k: svgs)
    path = tmp_path / "deck.html"
    path.write_text(HTML, encoding="utf-8", newline="")
    assert fig.main(["--verificar", str(path)]) == 1
    assert fig.main(["--injetar", str(path)]) == 0
    assert fig.main(["--verificar", str(path)]) == 0
