"""Gera as figuras SVG do deck de defesa (HTML) a partir dos CSVs das figuras PGF.

Três figuras determinísticas, sem dados por par: curva_esforco, incerteza_ordenacoes e
curva_fusoes. As funções de dados vêm dos geradores PGF (importadas, não copiadas).
`injetar` cola cada SVG entre as linhas `<!-- FIGURA:<nome> -->` e `<!-- /FIGURA:<nome> -->`
do HTML; `verificar` acusa os trechos dessincronizados.
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import figura_curva_esforco as pgf_curva  # noqa: E402
from scripts import figura_incerteza_ordenacoes as pgf_inc  # noqa: E402

CURVA_CSV = ROOT / "data/comparador_mesma_carga_curva.csv"
COMPARADOR_CSV = ROOT / "data/comparador_mesma_carga.csv"
ORDENACAO_CSV = ROOT / "data/ordenacao_bootstrap_o3_o1.csv"
FUSAO_CSV = ROOT / "data/fusao_robustez_bootstrap.csv"
FILAS_CSV = ROOT / "data/fusao_filas_curva.csv"
DESTINO = ROOT / "text/slides/figures"
FIGURAS = ("curva_esforco", "incerteza_ordenacoes", "curva_fusoes")

MINUS = "\u2212"
ARM = "ASR"
XMAX_FUSOES = 2522  # limite do eixo da curva das fusões (o mesmo da curva de esforço)
FILAS_FUSOES = (
    ("O1", "O1: escore do OpenRecLink"),
    ("O3", "O3: probabilidade calibrada fora da amostra"),
    ("F3", "F3: fusão de O3, O1 e O3 com MACD"),
    ("F4", "F4: fusão de O3 e O1"),
)

# Paleta de Okabe e Ito; O1 usa a cor do texto da página (linha de base).
BASE = "currentColor"
BLUE, ORANGE, GREEN, VERMILION, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
SOLID, DASHED, DOTTED, DASHDOT = "", "13 9", "0.1 8", "13 8 0.1 8"
# Traços próprios das fusões: nenhuma série O1 a O4 usa a cor nem o traço de F3 ou de F4.
SHORTDASH, DASHDOTDOT = "5 6", "13 8 0.1 8 0.1 8"
CURVE_STYLE = {  # série -> (cor, padrão de traço)
    "O1_SCORE": (BASE, SOLID), "O2_CdR": (ORANGE, DASHED),
    "O3_p_cal_ml": (BLUE, DOTTED), "O4_P_recalibrada": (GREEN, DASHDOT),
}
CURVE_LABEL = {
    "O1_SCORE": "O1: escore do OpenRecLink",
    "O2_CdR": "O2: nota final do CdR",
    "O3_p_cal_ml": "O3: probabilidade calibrada fora da amostra",
    "O4_P_recalibrada": f"O4: probabilidade recalibrada de {ARM}",
}
FUSION_STYLE = {"O1": (BASE, SOLID), "O3": (BLUE, DOTTED), "F3": (VERMILION, SHORTDASH), "F4": (PURPLE, DASHDOTDOT)}
CONTRAST_COLOR = {"O3": BLUE, "F3": VERMILION, "F4": PURPLE}

TITLE = "Pares verdadeiros recuperados abaixo do corte, revisor perfeito"
X_AXIS_LABEL = "Revisões humanas (escala logarítmica)"
FONT_TITLE, FONT_AXIS, FONT_TICK, FONT_LEGEND, FONT_NOTE = 20, 18, 16, 16, 15
STROKE_CURVE, STROKE_FRAME = 3, 1.5


def fmt(value):
    text = f"{round(value, 2):.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def pt(number):
    """Inteiro com ponto de milhar."""
    return f"{round(number):,}".replace(",", ".")


def signed(value, plus=False):
    rounded = round(value)
    if rounded < 0:
        return MINUS + pt(-rounded)
    return ("+" if plus and rounded > 0 else "") + pt(rounded)


def text(x, y, value, size, anchor="start", weight=None):
    extra = f' font-weight="{weight}"' if weight else ""
    anchor_attr = "" if anchor == "start" else f' text-anchor="{anchor}"'
    return (f'<text x="{fmt(x)}" y="{fmt(y)}" font-size="{size}"{anchor_attr}{extra}>'
            f'{escape(value)}</text>')


def line_path(d, color, width, dash=SOLID, extra=""):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"'
            f'{dash_attr} stroke-linecap="round" stroke-linejoin="round"{extra}/>')


def svg_document(name, width, height, title, desc, body):
    """Envelope `<svg>`; todos os ids levam o nome da figura como prefixo."""
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" class="figura figura-{name}" '
            f'viewBox="0 0 {width} {height}" width="100%" role="img" '
            f'aria-labelledby="{name}-titulo {name}-desc" font-family="inherit">')
    parts = [head, f'<title id="{name}-titulo">{escape(title)}</title>',
             f'<desc id="{name}-desc">{escape(desc)}</desc>']
    parts.extend(body)
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------- curvas em degrau

def step_path(series, px, py):
    """Caminho em degrau: só os pontos em que a recuperação muda (colineares omitidos)."""
    d = [f"M{fmt(px(series[0][0]))},{fmt(py(series[0][1]))}"]
    last_y = series[0][1]
    for k, y in series[1:]:
        if y != last_y:
            d.append(f"H{fmt(px(k))}V{fmt(py(y))}")
            last_y = y
    d.append(f"H{fmt(px(series[-1][0]))}")
    return "".join(d)


class Frame:
    """Moldura de uma curva de esforço: eixo x logarítmico de 1 a xmax, eixo y de 0 a ymax."""
    LEFT, RIGHT, TOP, BOTTOM = 70.0, 970.0, 46.0, 290.0

    def __init__(self, xmax, ymax):
        self.xmax, self.ymax = xmax, ymax

    def px(self, k):
        return self.LEFT + (self.RIGHT - self.LEFT) * math.log10(k) / math.log10(self.xmax)

    def py(self, y):
        return self.BOTTOM - (self.BOTTOM - self.TOP) * y / self.ymax

    def axes(self, title):
        left, right, top, bottom = self.LEFT, self.RIGHT, self.TOP, self.BOTTOM
        out = [f'<rect x="{fmt(left)}" y="{fmt(top)}" width="{fmt(right - left)}" '
               f'height="{fmt(bottom - top)}" fill="none" stroke="currentColor" '
               f'stroke-width="{STROKE_FRAME}"/>']
        ticks = []
        for y in [v for v in (0, 9, 18, 27) if v < self.ymax] + [self.ymax]:
            ty = self.py(y)
            ticks.append(f"M{fmt(left - 7)},{fmt(ty)}H{fmt(left)}")
        for k in [v for v in (1, 10, 100, 1000) if v < self.xmax] + [self.xmax]:
            tx = self.px(k)
            ticks.append(f"M{fmt(tx)},{fmt(bottom)}V{fmt(bottom + 7)}")
        out.append(f'<path d="{"".join(ticks)}" fill="none" stroke="currentColor" '
                   f'stroke-width="{STROKE_FRAME}"/>')
        labels = [text(left - 12, self.py(y) + 5.5, str(y), FONT_TICK, "end")
                  for y in [v for v in (0, 9, 18, 27) if v < self.ymax] + [self.ymax]]
        labels += [text(self.px(k), bottom + 26, pt(k), FONT_TICK, "middle")
                   for k in [v for v in (1, 10, 100, 1000) if v < self.xmax] + [self.xmax]]
        labels.append(text((left + right) / 2, bottom + 52, X_AXIS_LABEL, FONT_AXIS, "middle"))
        labels.append(text(left, 26, title, FONT_TITLE, "start", "bold"))
        out.append('<g class="texto" font-family="inherit" fill="currentColor">')
        out.extend(labels)
        out.append("</g>")
        return out

    def legend(self, name, entries, col2_x, note):
        """entries: (rótulo, cor, traço); duas colunas, preenchidas por linha."""
        base_y = self.BOTTOM + 78
        out = ['<g class="legenda" font-family="inherit" fill="currentColor">']
        for i, (label, color, dash) in enumerate(entries):
            x = self.LEFT if i % 2 == 0 else col2_x
            y = base_y + (i // 2) * 28
            out.append(line_path(f"M{fmt(x)},{fmt(y)}H{fmt(x + 56)}", color, STROKE_CURVE, dash))
            out.append(text(x + 68, y + 5.5, label, FONT_LEGEND))
        rows = (len(entries) + 1) // 2
        out.append(text((self.LEFT + self.RIGHT) / 2, base_y + (rows - 1) * 28 + 32, note, FONT_NOTE, "middle"))
        out.append("</g>")
        return out


def render_curva_esforco(curve_rows, comparator_rows):
    series = [(name, pgf_curva.curve_series(curve_rows, name)) for name, *_ in pgf_curva.ORDERS]
    if len({points[-1][0] for _, points in series}) != 1:
        raise ValueError("As ordenações precisam do mesmo limite de k")
    xmax = series[0][1][-1][0]
    ymax = max(y for _, points in series for _, y in points)
    frame = Frame(xmax, ymax)
    body = frame.axes(TITLE)
    body.append('<g class="curvas">')
    for name, points in series:
        color, dash = CURVE_STYLE[name]
        body.append(line_path(step_path(points, frame.px, frame.py), color, STROKE_CURVE, dash))
    body.append("</g>")
    asr = {}
    for mode, short in (("vigilancia", "vig."), ("confirmacao", "conf.")):
        matches = [r for r in comparator_rows if r["ordenacao"] == "O3_p_cal_ml"
                   and r["modo"] == mode and r["carga"] == "K1_humana"]
        if len(matches) != 1:
            raise ValueError(f"{ARM} sem ponto único: {mode}")
        row = matches[0]
        asr[mode] = (short, int(row["P_pares_humano"]), int(row["P_par2_recuperados"]),
                     row["P_fp"], row["P_pares_llm"])
    body.append('<g class="asr" font-family="inherit">')
    for mode, (short, k, y, fp, llm) in asr.items():
        cx, cy = frame.px(k), frame.py(y)
        label = f"{ARM} {short}: ({k}, {y}), {fp} falsos positivos, +{llm} LLM"
        if mode == "vigilancia":  # texto à esquerda e acima do ponto
            tx, ty, anchor = cx - 30, cy - 26, "end"
            conn = f"M{fmt(cx)},{fmt(cy)}L{fmt(tx + 6)},{fmt(ty + 6)}"
        else:  # texto na região livre, abaixo das curvas e à direita do ponto
            tx, ty, anchor = 560.0, frame.BOTTOM - 40, "start"
            conn = f"M{fmt(cx)},{fmt(cy)}L{fmt(tx - 8)},{fmt(ty - 5)}"
        body.append(line_path(conn, VERMILION, 1.5))
        body.append(f'<circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="6.5" fill="{VERMILION}"/>')
        body.append(f'<text x="{fmt(tx)}" y="{fmt(ty)}" font-size="{FONT_LEGEND}" '
                    f'text-anchor="{anchor}" fill="currentColor">{escape(label)}</text>')
    body.append("</g>")
    entries = [(CURVE_LABEL[name], *CURVE_STYLE[name]) for name, *_ in pgf_curva.ORDERS]
    body.extend(frame.legend("curva_esforco", entries, 540.0, f"Eixo limitado a {pt(xmax)} revisões"))
    vig, conf = asr["vigilancia"], asr["confirmacao"]
    desc = (f"Curvas em degrau do número de pares verdadeiros recuperados abaixo do corte, de 0 a {ymax}, "
            f"conforme as revisões humanas, de 1 a {pt(xmax)} em escala logarítmica, para as ordenações "
            f"O1 (escore do OpenRecLink), O2 (nota final do CdR), O3 (probabilidade calibrada fora da "
            f"amostra) e O4 (probabilidade recalibrada de {ARM}), com revisor perfeito. Dois pontos "
            f"marcam o arranjo {ARM}: vigilância, {vig[2]} pares verdadeiros com {vig[1]} revisões, "
            f"{vig[3]} falsos positivos e {vig[4]} pares avaliados pelo LLM; confirmação, "
            f"{conf[2]} pares verdadeiros com {conf[1]} revisões, {conf[3]} falsos positivos e "
            f"{conf[4]} pares avaliados pelo LLM.")
    return svg_document("curva_esforco", 1000, 440, TITLE, desc, body)


# ---------------------------------------------------------------- curva das fusões

def load_filas(rows, xmax=XMAX_FUSOES):
    """Séries (k, pares verdadeiros acumulados) por fila, até xmax."""
    wanted = {name for name, _ in FILAS_FUSOES}
    series = {name: [] for name in wanted}
    for row in rows:
        name = row["fila"]
        if name not in wanted:
            continue
        k, value = int(row["k"]), float(row["par2_acumulados"])
        if value != int(value):
            raise ValueError(f"Valor não inteiro na fila {name}, k={k}")
        if k <= xmax:
            series[name].append((k, int(value)))
    for name in sorted(wanted):
        if not series[name]:
            raise ValueError(f"Fila ausente: {name}")
    return series


def first_k_reaching(rows, name, target):
    for row in rows:
        if row["fila"] == name and float(row["par2_acumulados"]) >= target:
            return int(row["k"])
    return None


def validate_filas(series, xmax):
    for name, points in series.items():
        points.sort()
        if [k for k, _ in points] != list(range(1, len(points) + 1)):
            raise ValueError(f"Orçamento k não contínuo ou duplicado: {name}")
        if len(points) < xmax:
            raise ValueError(f"Fila {name} termina antes de k = {xmax}")
        if any(b[1] < a[1] for a, b in zip(points, points[1:])):
            raise ValueError(f"Recuperação não monotônica: {name}")
        if points[0][1] < 0:
            raise ValueError(f"Valores inválidos: {name}")


def render_curva_fusoes(fila_rows, xmax=XMAX_FUSOES):
    """fila_rows: iterável de dicts com `fila`, `k`, `par2_acumulados`."""
    fila_rows = list(fila_rows)
    series = load_filas(fila_rows, xmax)
    validate_filas(series, xmax)
    ymax = max(y for points in series.values() for _, y in points)
    frame = Frame(xmax, ymax)
    body = frame.axes(TITLE)
    body.append('<g class="curvas">')
    for name, _ in FILAS_FUSOES:
        color, dash = FUSION_STYLE[name]
        body.append(line_path(step_path(series[name], frame.px, frame.py), color, STROKE_CURVE, dash))
    body.append("</g>")
    entries = [(label, *FUSION_STYLE[name]) for name, label in FILAS_FUSOES]
    note = f"Eixo limitado a {pt(xmax)} revisões"
    outside = [name for name, _ in FILAS_FUSOES if series[name][-1][1] < ymax]
    if outside:
        parts = []
        for name in outside:
            reach_k = first_k_reaching(fila_rows, name, ymax)
            if reach_k is not None:
                parts.append(f"{name} chega aos {ymax} pares verdadeiros em {pt(reach_k)} revisões")
        if parts:
            note += "; " + "; ".join(parts)
    body.extend(frame.legend("curva_fusoes", entries, 520.0, note))
    ref = 96  # ponto de leitura do texto da tese (revisões do arranjo de vigilância)
    desc = (f"Curvas em degrau do número de pares verdadeiros recuperados abaixo do corte, de 0 a {ymax}, "
            f"conforme as revisões humanas, de 1 a {pt(xmax)} em escala logarítmica, para as ordenações "
            f"O1 (escore do OpenRecLink), O3 (probabilidade calibrada fora da amostra), F3 (fusão de O3, "
            f"O1 e O3 com MACD) e F4 (fusão de O3 e O1), com revisor perfeito.")
    if ref <= xmax:
        at = {n: dict(series[n])[ref] for n in series}
        desc += (f" Com {ref} revisões, O1 recupera {at['O1']}, O3 {at['O3']}, F3 {at['F3']} "
                 f"e F4 {at['F4']}.")
    return svg_document("curva_fusoes", 1000, 440, TITLE, desc, body)


# ---------------------------------------------------------------- incerteza

PANEL_TOP, PANEL_HEIGHT = 44.0, 170.0
PANEL_LEFT, PANEL_RIGHT, PANEL_GAP = 66.0, 984.0, 40.0
CI_STROKE, TICK_HALF = 3.5, 8.0
MARKER_R = 7.0
ARROW_LEN, ARROW_HALF = 16.0, 8.0
ARROW_ROOM = 0.14  # fração do intervalo do eixo reservada do lado truncado


def render_incerteza(ordenacao_rows, fusao_rows):
    panels = pgf_inc.load_panels(ordenacao_rows, fusao_rows)
    count = len(panels)
    width = (PANEL_RIGHT - PANEL_LEFT - (count - 1) * PANEL_GAP) / count
    bottom = PANEL_TOP + PANEL_HEIGHT
    n_rows = len(pgf_inc.CONTRASTS)
    row_y = [PANEL_TOP + PANEL_HEIGHT * (i + 1) / (n_rows + 1) for i in range(n_rows)]
    graphics = []
    labels = []
    desc_parts = []
    for index, (goal, rows) in enumerate(panels):
        x0 = PANEL_LEFT + index * (width + PANEL_GAP)
        values = [data for _, _, data in rows]
        truncated = pgf_inc.truncated_limits(values)
        low, high = pgf_inc.axis_range(values, truncated)
        step = pgf_inc.nice_step(high - low)
        # folga para a seta: o limite truncado não pode encostar no marcador da mediana
        axis_low, axis_high = low, high  # os rótulos ficam dentro da faixa original
        room = ARROW_ROOM * (high - low)
        if any(j == 2 for _, j in truncated):
            high += room
        if any(j == 0 for _, j in truncated):
            low -= room

        def px(value, low=low, high=high, x0=x0):
            return x0 + width * (min(max(value, low), high) - low) / (high - low)

        labels.append(text(x0 + width / 2, PANEL_TOP - 14, f"Meta de {goal} pares verdadeiros",
                           FONT_AXIS, "middle", "bold"))
        graphics.append(f'<rect x="{fmt(x0)}" y="{fmt(PANEL_TOP)}" width="{fmt(width)}" '
                        f'height="{fmt(PANEL_HEIGHT)}" fill="none" stroke="currentColor" '
                        f'stroke-width="{STROKE_FRAME}"/>')
        ticks = []
        tick = math.ceil(axis_low / step)
        while tick * step <= axis_high:
            tx = px(tick * step)
            ticks.append(f"M{fmt(tx)},{fmt(bottom)}V{fmt(bottom + 7)}")
            labels.append(text(tx, bottom + 26, signed(tick * step), FONT_TICK, "middle"))
            tick += 1
        graphics.append(f'<path d="{"".join(ticks)}" fill="none" stroke="currentColor" '
                        f'stroke-width="{STROKE_FRAME}"/>')
        zero = px(0)
        graphics.append(f'<path d="M{fmt(zero)},{fmt(PANEL_TOP)}V{fmt(bottom)}" fill="none" '
                        f'stroke="currentColor" stroke-width="{STROKE_FRAME}" stroke-opacity="0.55" '
                        f'stroke-dasharray="6 5"/>')
        summary = []
        for i, (label, (median, lo, hi)) in enumerate((r[0], r[2]) for r in rows):
            y = row_y[i]
            color = CONTRAST_COLOR[label]
            graphics.append(f'<path d="M{fmt(px(lo))},{fmt(y)}H{fmt(px(hi))}" fill="none" '
                            f'stroke="{color}" stroke-width="{CI_STROKE}" stroke-linecap="butt"/>')
            for j, edge in ((0, lo), (2, hi)):
                ex = px(edge)
                if (i, j) in truncated:
                    sign = 1 if j == 2 else -1
                    back = ex - sign * ARROW_LEN
                    graphics.append(f'<path d="M{fmt(ex)},{fmt(y)}L{fmt(back)},{fmt(y - ARROW_HALF)}'
                                    f'L{fmt(back)},{fmt(y + ARROW_HALF)}Z" fill="{color}"/>')
                    anchor = "end" if j == 2 else "start"
                    labels.append(text(ex - sign * 6, y - 15, "até " + signed(edge, plus=True),
                                       FONT_NOTE, anchor))
                else:
                    graphics.append(f'<path d="M{fmt(ex)},{fmt(y - TICK_HALF)}V{fmt(y + TICK_HALF)}" '
                                    f'fill="none" stroke="{color}" stroke-width="{STROKE_CURVE}"/>')
            if pgf_inc.marker_filled(hi):
                graphics.append(f'<circle cx="{fmt(px(median))}" cy="{fmt(y)}" r="{MARKER_R}" '
                                f'fill="{color}" stroke="{color}" stroke-width="{STROKE_CURVE}" '
                                f'class="cheio"/>')
            else:
                graphics.append(f'<circle cx="{fmt(px(median))}" cy="{fmt(y)}" r="{MARKER_R}" '
                                f'fill="#ffffff" stroke="{color}" stroke-width="{STROKE_CURVE}" '
                                f'class="vazado"/>')
            if index == 0:
                labels.append(text(x0 - 10, y + 5.5, label, FONT_AXIS, "end"))
            summary.append(f"{label}, mediana {signed(median)}, IC 95% de {signed(lo)} a {signed(hi)}"
                           + (" (marcador cheio)" if pgf_inc.marker_filled(hi) else " (marcador vazado)"))
        desc_parts.append(f"Meta de {goal} pares verdadeiros: " + "; ".join(summary) + ".")
    height = 290
    labels.append(text((PANEL_LEFT + PANEL_RIGHT) / 2, bottom + 62, pgf_inc.AXIS_LABEL.replace("\\%", "%"),
                       FONT_AXIS, "middle"))
    style = (f'<style>.figura-incerteza_ordenacoes .vazado{{fill:var(--figura-fundo,#ffffff)}}</style>')
    title = "Diferença de revisões frente a O1, por meta de pares verdadeiros"
    desc = ("Três painéis, um por meta de pares verdadeiros recuperados abaixo do corte, com a mediana e o "
            "intervalo de confiança de 95% da diferença de revisões humanas de O3, F3 e F4 frente a O1; "
            "valores negativos indicam menos revisões que O1. Marcador cheio quando o intervalo inteiro "
            "fica abaixo de zero. " + " ".join(desc_parts))
    body = [style] + graphics
    body.append('<g class="texto" font-family="inherit" fill="currentColor">')
    body.extend(labels)
    body.append("</g>")
    return svg_document("incerteza_ordenacoes", 1000, height, title, desc, body)


# ---------------------------------------------------------------- arquivos e HTML

def stream_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def render_all(curva=CURVA_CSV, comparador=COMPARADOR_CSV, ordenacao=ORDENACAO_CSV,
               fusao=FUSAO_CSV, filas=FILAS_CSV):
    return {
        "curva_esforco": render_curva_esforco(pgf_curva.read_rows(curva), pgf_curva.read_rows(comparador)),
        "incerteza_ordenacoes": render_incerteza(pgf_inc.read_rows(ordenacao), pgf_inc.read_rows(fusao)),
        "curva_fusoes": render_curva_fusoes(list(stream_rows_filtered(filas))),
    }


def stream_rows_filtered(path):
    """Linhas das filas de interesse, lidas em fluxo (o CSV completo tem centenas de milhares)."""
    wanted = {name for name, _ in FILAS_FUSOES}
    for row in stream_rows(path):
        if row["fila"] in wanted:
            yield row


def generate(destino=DESTINO, **paths):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    written = []
    for name, content in render_all(**paths).items():
        target = destino / f"{name}.svg"
        target.write_bytes(content.encode("utf-8"))
        written.append(target)
    return written


def _block_pattern(name):
    return re.compile(rf"(?P<open>^[ \t]*<!-- FIGURA:{re.escape(name)} -->[ \t]*\r?\n)"
                      rf"(?P<body>.*?)"
                      rf"(?P<close>^[ \t]*<!-- /FIGURA:{re.escape(name)} -->)",
                      re.MULTILINE | re.DOTALL)


def _find_block(html, name):
    """Casamento do bloco da figura, None se não há marcadores; ValueError se estiver malformado."""
    opens = re.findall(rf"<!-- FIGURA:{re.escape(name)} -->", html)
    closes = re.findall(rf"<!-- /FIGURA:{re.escape(name)} -->", html)
    if not opens and not closes:
        return None
    match = _block_pattern(name).search(html)
    if len(opens) != 1 or len(closes) != 1 or match is None:
        raise ValueError(f"Marcadores da figura {name} malformados: abertura sem fechamento, "
                         "duplicados ou fora de linha própria")
    return match


def _read_html(html_path):
    return Path(html_path).read_bytes().decode("utf-8")


def injetar(html_path, svgs=None):
    """Substitui o conteúdo entre os marcadores de cada figura; devolve as figuras injetadas."""
    svgs = svgs if svgs is not None else render_all()
    html = _read_html(html_path)
    eol = "\r\n" if "\r\n" in html else "\n"
    injected = []
    for name in FIGURAS:
        match = _find_block(html, name)
        if match is None:
            continue
        body = svgs[name].replace("\n", eol)
        html = html[:match.start("body")] + body + html[match.end("body"):]
        injected.append(name)
    if injected:
        Path(html_path).write_bytes(html.encode("utf-8"))
    return injected


def verificar(html_path, svgs=None):
    """Figuras com marcadores cujo conteúdo difere do gerador."""
    svgs = svgs if svgs is not None else render_all()
    html = _read_html(html_path)
    stale = []
    for name in FIGURAS:
        match = _find_block(html, name)
        if match is not None and match.group("body").replace("\r\n", "\n") != svgs[name]:
            stale.append(name)
    return stale


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--injetar", type=Path, metavar="HTML", help="Injeta os SVG entre os marcadores")
    group.add_argument("--verificar", type=Path, metavar="HTML",
                       help="Código 1 se algum trecho diferir do gerador")
    parser.add_argument("--destino", type=Path, default=DESTINO)
    args = parser.parse_args(argv)
    if args.injetar:
        done = injetar(args.injetar)
        for name in FIGURAS:
            print(f"{name}: " + ("injetada" if name in done else "marcador ausente"))
        return 0
    if args.verificar:
        stale = verificar(args.verificar)
        for name in stale:
            print(f"dessincronizada: {name}")
        if not stale:
            print("todas as figuras com marcador estão em sincronia")
        return 1 if stale else 0
    for path in generate(args.destino):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
