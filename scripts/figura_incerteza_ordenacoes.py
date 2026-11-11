"""Gera PGF determinístico da incerteza das ordenações frente a O1, sem dados por par."""

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOALS = (9, 23, 36)
# (rótulo da linha, valor da coluna `comparacao`, arquivo de origem, cor)
CONTRASTS = (
    ("O3", "O3-O1", "ordenacao", "blue"),
    ("F3", "F3-O1", "fusao", "orange"),
    ("F4", "F4-O1", "fusao", "teal"),
)
AXIS_LABEL = "Diferença de revisões frente a O1 (mediana e IC 95\\%)"
# Regra de truncamento: um limite do IC é truncado quando seu módulo excede
# OUTLIER_FACTOR vezes o maior módulo entre os outros oito valores do painel.
OUTLIER_FACTOR = 3.0
PANEL_WIDTH, PANEL_HEIGHT, PANEL_GAP, LEFT_MARGIN = 3.5, 3.2, 0.9, 0.55
MAX_TICKS = 5
PAD_FRACTION = 0.06
MINUS = "$-$"


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_contrast(rows, comparison, goal):
    matches = [row for row in rows if row.get("comparacao") == comparison
               and int(float(row["meta_nominal"])) == goal]
    if not matches:
        raise ValueError(f"Contraste ausente: {comparison}, meta {goal}")
    if len(matches) > 1:
        raise ValueError(f"Contraste duplicado: {comparison}, meta {goal}")
    row = matches[0]
    median = float(row["diferenca_revisoes_mediana"])
    low = float(row["diferenca_revisoes_ic95_inf"])
    high = float(row["diferenca_revisoes_ic95_sup"])
    if not low <= median <= high:
        raise ValueError(f"IC inconsistente: {comparison}, meta {goal}")
    return median, low, high


def load_panels(ordenacao_rows, fusao_rows):
    sources = {"ordenacao": ordenacao_rows, "fusao": fusao_rows}
    return [(goal, [(label, color, find_contrast(sources[source], comparison, goal))
                    for label, comparison, source, color in CONTRASTS])
            for goal in GOALS]


def marker_filled(high):
    """Marcador cheio quando o IC inteiro fica abaixo de zero."""
    return high < 0


def truncated_limits(values):
    """Índices (linha, 0 inferior ou 2 superior) cujo limite foge da faixa dos demais."""
    flat = [(i, j, v) for i, (median, low, high) in enumerate(values)
            for j, v in enumerate((low, median, high))]
    truncated = set()
    for i, j, v in flat:
        if j == 1:
            continue
        reference = max(abs(w) for k, m, w in flat if (k, m) != (i, j))
        if abs(v) > OUTLIER_FACTOR * reference:
            truncated.add((i, j))
    return truncated


def nice_step(span):
    exponent = math.floor(math.log10(span / MAX_TICKS))
    for factor in (1, 2, 5, 10):
        step = factor * 10 ** exponent
        if span / step <= MAX_TICKS:
            return step
    raise ValueError("Passo de eixo não encontrado")


def axis_range(values, truncated):
    kept = [v for i, (median, low, high) in enumerate(values)
            for j, v in enumerate((low, median, high)) if (i, j) not in truncated]
    low, high = min(0.0, min(kept)), max(0.0, max(kept))
    pad = PAD_FRACTION * (high - low)
    if pad == 0:
        raise ValueError("Painel sem amplitude")
    return low - pad, high + pad


def thousands(value):
    return f"{abs(round(value)):,}".replace(",", ".")


def signed(value, plus=False):
    rounded = round(value)
    if rounded < 0:
        return MINUS + thousands(rounded)
    return ("+" if plus and rounded > 0 else "") + thousands(rounded)


def render(ordenacao_rows, fusao_rows):
    panels = load_panels(ordenacao_rows, fusao_rows)
    total_width = LEFT_MARGIN + len(panels) * PANEL_WIDTH + (len(panels) - 1) * PANEL_GAP
    row_y = [PANEL_HEIGHT * (len(CONTRASTS) - i) / (len(CONTRASTS) + 1)
             for i in range(len(CONTRASTS))]

    def text(x, y, value, anchor="center"):
        alignment = "" if anchor == "center" else anchor + ","
        return rf"\pgftext[{alignment}at={{\pgfpoint{{{x:.5f}cm}}{{{y:.5f}cm}}}}]{{\scriptsize {value}}}"

    lines = [r"% Gerado por scripts/figura_incerteza_ordenacoes.py; não editar.",
             r"\begin{pgfpicture}"]
    for index, (goal, rows) in enumerate(panels):
        x0 = LEFT_MARGIN + index * (PANEL_WIDTH + PANEL_GAP)
        values = [data for _, _, data in rows]
        truncated = truncated_limits(values)
        low, high = axis_range(values, truncated)
        step = nice_step(high - low)

        def px(value, low=low, high=high, x0=x0):
            clipped = min(max(value, low), high)
            return x0 + PANEL_WIDTH * (clipped - low) / (high - low)

        lines.extend([r"\color{black}", r"\pgfsetdash{}{0pt}", r"\pgfsetlinewidth{0.35pt}",
                      rf"\pgfpathrectangle{{\pgfpoint{{{x0:.5f}cm}}{{0cm}}}}"
                      rf"{{\pgfpoint{{{PANEL_WIDTH:.5f}cm}}{{{PANEL_HEIGHT:.5f}cm}}}}\pgfusepath{{stroke}}",
                      text(x0 + PANEL_WIDTH / 2, PANEL_HEIGHT + 0.30,
                           f"Meta de {goal} pares verdadeiros")])
        tick = math.ceil(low / step)
        while tick * step <= high:
            tx = px(tick * step)
            lines.extend([rf"\pgfpathmoveto{{\pgfpoint{{{tx:.5f}cm}}{{0cm}}}}",
                          rf"\pgfpathlineto{{\pgfpoint{{{tx:.5f}cm}}{{-0.08cm}}}}\pgfusepath{{stroke}}",
                          text(tx, -0.25, signed(tick * step))])
            tick += 1
        lines.append(rf"\pgftext[top,at={{\pgfpoint{{{x0 + PANEL_WIDTH / 2:.5f}cm}}{{-0.42cm}}}}]"
                     rf"{{\parbox{{{PANEL_WIDTH:.2f}cm}}{{\centering\scriptsize {AXIS_LABEL}}}}}")
        zero = px(0)
        lines.extend([r"\color{gray}", rf"\pgfpathmoveto{{\pgfpoint{{{zero:.5f}cm}}{{0cm}}}}",
                      rf"\pgfpathlineto{{\pgfpoint{{{zero:.5f}cm}}{{{PANEL_HEIGHT:.5f}cm}}}}\pgfusepath{{stroke}}"])
        for i, (label, color, (median, lo, hi)) in enumerate(rows):
            y = row_y[i]
            lines.extend([rf"\color{{{color}}}", r"\pgfsetlinewidth{1.2pt}",
                          rf"\pgfpathmoveto{{\pgfpoint{{{px(lo):.5f}cm}}{{{y:.5f}cm}}}}",
                          rf"\pgfpathlineto{{\pgfpoint{{{px(hi):.5f}cm}}{{{y:.5f}cm}}}}\pgfusepath{{stroke}}",
                          r"\pgfsetlinewidth{0.6pt}"])
            for j, edge in ((0, lo), (2, hi)):
                ex = px(edge)
                if (i, j) in truncated:
                    sign = 1 if j == 2 else -1
                    lines.extend([rf"\pgfpathmoveto{{\pgfpoint{{{ex:.5f}cm}}{{{y:.5f}cm}}}}",
                                  rf"\pgfpathlineto{{\pgfpoint{{{ex - sign * 0.14:.5f}cm}}{{{y + 0.07:.5f}cm}}}}",
                                  rf"\pgfpathlineto{{\pgfpoint{{{ex - sign * 0.14:.5f}cm}}{{{y - 0.07:.5f}cm}}}}",
                                  r"\pgfpathclose\pgfusepath{fill}"])
                    anchor = "right" if j == 2 else "left"
                    lines.extend([r"\color{black}",
                                  text(ex, y + 0.27, "até " + signed(edge, plus=True), anchor),
                                  rf"\color{{{color}}}"])
                else:
                    lines.extend([rf"\pgfpathmoveto{{\pgfpoint{{{ex:.5f}cm}}{{{y - 0.07:.5f}cm}}}}",
                                  rf"\pgfpathlineto{{\pgfpoint{{{ex:.5f}cm}}{{{y + 0.07:.5f}cm}}}}\pgfusepath{{stroke}}"])
            circle = rf"\pgfpathcircle{{\pgfpoint{{{px(median):.5f}cm}}{{{y:.5f}cm}}}}{{2pt}}"
            if marker_filled(hi):
                lines.extend([rf"\pgfsetfillcolor{{{color}}}", circle + r"\pgfusepath{fill}"])
            else:
                lines.extend([r"\pgfsetfillcolor{white}", circle + r"\pgfusepath{fill,stroke}"])
            lines.extend([r"\color{black}", text(x0 - 0.12, y, label, "right")])
    lines.extend([r"\color{black}", r"\end{pgfpicture}"])
    return "\n".join(lines) + "\n"


def generate(ordenacao, fusao, output):
    content = render(read_rows(ordenacao), read_rows(fusao))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content.encode("utf-8"))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordenacao", type=Path, default=ROOT / "data/ordenacao_bootstrap_o3_o1.csv")
    parser.add_argument("--fusao", type=Path, default=ROOT / "data/fusao_robustez_bootstrap.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "text/slides/figures/incerteza_ordenacoes.pgf",
                        help="Destino opcional; por padrão grava nos slides")
    args = parser.parse_args(argv)
    print(generate(args.ordenacao, args.fusao, args.output))


if __name__ == "__main__":
    main()
