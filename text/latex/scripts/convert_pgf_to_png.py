"""Convert PGF figures to PNG via standalone LaTeX + pymupdf."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import fitz  # pymupdf

LATEX_DIR = Path(__file__).resolve().parent.parent  # text/latex/
FIG_DOC = LATEX_DIR / "fig_doc"
OUT_DIR = LATEX_DIR / "figures" / "pgf_png"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PGF_FILES = [
    "score_band_volume.pgf",
    "score_band_pct.pgf",
    "nb01_model_comparison.pgf",
    "ablation_best_category.pgf",
    "pareto_frontier_v2.pgf",
    "cv_boxplots_v2.pgf",
    "imbalance_sensitivity.pgf",
    "shap_summary.pgf",
    "feature_importance_heatmap.pgf",
    "epi_deteccao.pgf",
    "epi_revisoes.pgf",
    "epi_custo.pgf",
]

STANDALONE_TEMPLATE = r"""\documentclass[border=2pt]{{standalone}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{pgf}}
\usepackage{{amsmath,amssymb}}
\usepackage{{newtxtext,newtxmath}}
\begin{{document}}
\input{{{pgf_path}}}
\end{{document}}
"""

DPI = 300


def convert_one(pgf_name: str) -> bool:
    pgf_path = FIG_DOC / pgf_name
    if not pgf_path.exists():
        print(f"  SKIP (not found): {pgf_name}")
        return False

    stem = pgf_path.stem
    png_out = OUT_DIR / f"{stem}.png"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Write standalone .tex
        tex_content = STANDALONE_TEMPLATE.format(pgf_path=pgf_path.as_posix())
        tex_file = tmp / "fig.tex"
        tex_file.write_text(tex_content, encoding="utf-8")

        # Compile with pdflatex
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "fig.tex"],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=120,
        )
        pdf_file = tmp / "fig.pdf"
        if not pdf_file.exists():
            print(f"  FAIL (pdflatex): {pgf_name}")
            # Print last 20 lines of log for debugging
            log_file = tmp / "fig.log"
            if log_file.exists():
                lines = log_file.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                for line in lines[-20:]:
                    print(f"    {line}")
            return False

        # Convert PDF -> PNG at 300 DPI
        doc = fitz.open(str(pdf_file))
        page = doc[0]
        mat = fitz.Matrix(DPI / 72, DPI / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(png_out))
        doc.close()

    print(f"  OK: {pgf_name} -> {png_out.name} ({png_out.stat().st_size // 1024} KB)")
    return True


def main() -> None:
    print(f"Converting {len(PGF_FILES)} PGF figures to PNG at {DPI} DPI...")
    ok = 0
    fail = 0
    for pgf in PGF_FILES:
        success = convert_one(pgf)
        if success:
            ok += 1
        else:
            fail += 1
    print(f"\nDone: {ok} OK, {fail} failed")
    sys.exit(1 if fail > 0 else 0)


if __name__ == "__main__":
    main()
