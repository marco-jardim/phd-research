"""Gera PGF determinístico da curva exploratória de esforço, sem dados por par."""

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORDERS = (
    ("O1_SCORE", "O1: escore OpenRecLink", "gray"),
    ("O2_CdR", "O2: nota do CdR", "orange"),
    ("O3_p_cal_ml", "O3: probabilidade calibrada (OOF)", "blue"),
    ("O4_P_recalibrada", "O4: probabilidade recalibrada de {arm}", "teal"),
)
# Sigla do arranjo por destino: a tese usa ASR; os slides seguem com P até serem refeitos.
ARM_THESIS = "ASR"
ARM_SLIDES = "P"
LINE_STYLES = {
    "O1_SCORE": r"\pgfsetdash{}{0pt}",
    "O2_CdR": r"\pgfsetdash{{4pt}{2pt}}{0pt}",
    "O3_p_cal_ml": r"\pgfsetdash{{1pt}{1.7pt}}{0pt}",
    "O4_P_recalibrada": r"\pgfsetdash{{4pt}{1.5pt}{1pt}{1.5pt}}{0pt}",
}


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def curve_series(rows, order):
    series = sorted((int(row["k"]), int(row["par2_topk"]))
                    for row in rows if row["ordenacao"] == order)
    if not series or len({k for k, _ in series}) != len(series):
        raise ValueError(f"Curva ausente ou orçamento duplicado: {order}")
    if any(k < 1 or y < 0 for k, y in series):
        raise ValueError(f"Valores inválidos: {order}")
    if any(right[1] < left[1] for left, right in zip(series, series[1:])):
        raise ValueError(f"Recuperação não monotônica: {order}")
    if [k for k, _ in series] != list(range(1, series[-1][0] + 1)):
        raise ValueError(f"Orçamento k não contínuo: {order}")
    return series


def render(curve_rows, comparator_rows, arm=ARM_THESIS):
    series = [(name, label.format(arm=arm), color, curve_series(curve_rows, name))
              for name, label, color in ORDERS]
    xmax = max(points[-1][0] for _, _, _, points in series)
    if len({points[-1][0] for _, _, _, points in series}) != 1:
        raise ValueError("As ordenações precisam do mesmo limite de k")
    ymax = max(y for _, _, _, points in series for _, y in points)
    width, height = 11.4, 3.1

    def point(k, y):
        return rf"\pgfpoint{{{width * math.log10(k) / math.log10(xmax):.5f}cm}}{{{height * y / ymax:.5f}cm}}"

    def text(x, y, value, anchor="center"):
        alignment = "" if anchor == "center" else anchor + ","
        return rf"\pgftext[{alignment}at={{\pgfpoint{{{x:.5f}cm}}{{{y:.5f}cm}}}}]{{\scriptsize {value}}}"

    lines = [r"% Gerado por scripts/figura_curva_esforco.py; não editar.",
             r"\begin{pgfpicture}", r"\pgfsetlinewidth{0.35pt}",
             rf"\pgfpathrectangle{{\pgfpoint{{0cm}}{{0cm}}}}{{\pgfpoint{{{width}cm}}{{{height}cm}}}}\pgfusepath{{stroke}}"]
    for y in (0, 9, 18, 27, ymax):
        lines.append(text(-0.18, height*y/ymax, str(y), "right"))
    for k in (1, 10, 100, 1000, xmax):
        lines.append(text(width*math.log10(k)/math.log10(xmax), -0.23, f"{k:,}".replace(",", ".")))
    lines.append(text(width/2, -0.60, "Revisões humanas (escala logarítmica)"))
    lines.append(text(0, height+0.27, "PAR=2 recuperados, revisor perfeito", "left"))
    for name, label, color, points in series:
        lines.extend([rf"\color{{{color}}}", LINE_STYLES[name], r"\pgfsetlinewidth{0.9pt}",
                      rf"\pgfpathmoveto{{{point(*points[0])}}}"])
        last_y = points[0][1]
        for k, y in points[1:]:
            if y != last_y:
                lines.extend([rf"\pgfpathlineto{{{point(k, last_y)}}}", rf"\pgfpathlineto{{{point(k, y)}}}"])
                last_y = y
        lines.extend([rf"\pgfpathlineto{{{point(*points[-1])}}}", r"\pgfusepath{stroke}"])
    for mode, label, label_x, label_y in (("vigilancia", f"{arm} vig.", 0.35, height - 0.40),
                                         ("confirmacao", f"{arm} conf.", 6.30, 0.30)):
        matches = [r for r in comparator_rows if r["ordenacao"] == "O3_p_cal_ml"
                   and r["modo"] == mode and r["carga"] == "K1_humana"]
        if len(matches) != 1:
            raise ValueError(f"{arm} sem ponto único: {mode}")
        row = matches[0]
        k, y = int(row["P_pares_humano"]), int(row["P_par2_recuperados"])
        lines.extend([r"\color{red}", r"\pgfsetdash{}{0pt}", rf"\pgfpathcircle{{{point(k,y)}}}{{2pt}}\pgfusepath{{fill}}",
                      r"\pgfsetlinewidth{0.35pt}", rf"\pgfpathmoveto{{{point(k,y)}}}",
                      rf"\pgfpathlineto{{\pgfpoint{{{label_x:.5f}cm}}{{{label_y - 0.12:.5f}cm}}}}\pgfusepath{{stroke}}",
                      text(label_x, label_y,
                           f"{label}: ({k}, {y}), {row['P_fp']} FP, +{row['P_pares_llm']} LLM", "left")])
    for i, (name, label, color, _) in enumerate(series):
        label = label.replace("_", r"\_")
        x, y = (i % 2)*5.9, -1.0-(i // 2)*0.38
        lines.extend([rf"\color{{{color}}}", LINE_STYLES[name],
                      rf"\pgfpathmoveto{{\pgfpoint{{{x:.2f}cm}}{{{y:.2f}cm}}}}",
                      rf"\pgfpathlineto{{\pgfpoint{{{x+0.45:.2f}cm}}{{{y:.2f}cm}}}}\pgfusepath{{stroke}}",
                      text(x+0.55, y, label, "left")])
    lines.extend([r"\color{black}", r"\pgfsetdash{}{0pt}",
                  text(width/2, -1.8, f"Eixo x limitado a {xmax:,} revisões".replace(",", "."))])
    lines.extend([r"\color{black}", r"\end{pgfpicture}"])
    return "\n".join(lines) + "\n"


def generate(curve, comparator, output, arm=ARM_THESIS):
    content = render(read_rows(curve), read_rows(comparator), arm)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content.encode("utf-8"))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", type=Path, default=ROOT / "data/comparador_mesma_carga_curva.csv")
    parser.add_argument("--comparator", type=Path, default=ROOT / "data/comparador_mesma_carga.csv")
    parser.add_argument("--output", type=Path, help="Destino único opcional; por padrão grava tese e slides")
    parser.add_argument("--arm", default=ARM_THESIS,
                        help="Sigla do arranjo no destino único (padrão: ASR)")
    args = parser.parse_args()
    outputs = [(args.output, args.arm)] if args.output else [
        (ROOT / "text/latex/figures/curva_esforco.pgf", ARM_THESIS),
        (ROOT / "text/slides/figures/curva_esforco.pgf", ARM_SLIDES),
    ]
    for output, arm in outputs:
        print(generate(args.curve, args.comparator, output, arm))


if __name__ == "__main__":
    main()
