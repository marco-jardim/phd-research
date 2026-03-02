"""
Build a pandoc-friendly .tex copy and convert to DOCX with all figures.

Replaces:
  - \input{fig_doc/xxx.pgf} → \includegraphics[width=\textwidth]{figures/pgf_png/xxx.png}
  - \input{fig_doc/xxx.pgf scaled} patterns (resizebox) handled
  - PDF figures already use \includegraphics, pandoc handles those

Usage:
    python scripts/build_docx.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

LATEX_DIR = Path(__file__).resolve().parent.parent
TESE_TEX = LATEX_DIR / "tese.tex"
PANDOC_TEX = LATEX_DIR / "tese_pandoc.tex"
OUTPUT_DOCX = LATEX_DIR / "tese.docx"
PGF_PNG_DIR = LATEX_DIR / "figures" / "pgf_png"


def find_pandoc() -> str:
    """Locate pandoc executable."""
    for candidate in [
        r"C:\Program Files\Pandoc\pandoc.exe",
        "pandoc",
    ]:
        p = Path(candidate)
        if p.exists():
            return str(p)
    # fallback to PATH
    return "pandoc"


def process_input_line(line: str) -> str:
    """Replace \\input{fig_doc/xxx.pgf} with \\includegraphics of the PNG."""
    # Pattern: \input{fig_doc/something.pgf}
    match = re.search(r"\\input\{fig_doc/([^}]+)\.pgf\}", line)
    if match:
        stem = match.group(1)
        png_path = PGF_PNG_DIR / f"{stem}.png"
        if png_path.exists():
            # Replace the entire \input with \includegraphics
            replacement = (
                f"\\includegraphics[width=\\textwidth]{{figures/pgf_png/{stem}.png}}"
            )
            # Also strip any surrounding \resizebox if present
            line = re.sub(
                r"\\resizebox\{[^}]*\}\{[^}]*\}\{\\input\{fig_doc/"
                + re.escape(stem)
                + r"\.pgf\}\}",
                replacement,
                line,
            )
            # If no resizebox, just replace the \input
            line = re.sub(
                r"\\input\{fig_doc/" + re.escape(stem) + r"\.pgf\}",
                replacement,
                line,
            )
    return line


def collect_inputs(main_tex: Path) -> str:
    """
    Read tese.tex, inline all \\input{} chapter files,
    and replace PGF inputs with PNG includegraphics.
    """
    content = main_tex.read_text(encoding="utf-8")

    # Process PGF inputs in every line
    lines = content.split("\n")
    processed = []
    for line in lines:
        processed.append(process_input_line(line))

    return "\n".join(processed)


def main() -> None:
    print(f"Reading {TESE_TEX}...")
    content = collect_inputs(TESE_TEX)

    # Write pandoc-friendly version
    PANDOC_TEX.write_text(content, encoding="utf-8")
    print(f"Wrote {PANDOC_TEX}")

    # Check PNGs exist
    png_count = len(list(PGF_PNG_DIR.glob("*.png")))
    print(f"Found {png_count} PNG figures in {PGF_PNG_DIR}")

    # Run pandoc
    pandoc = find_pandoc()
    cmd = [
        pandoc,
        str(PANDOC_TEX),
        "--from=latex",
        "--to=docx",
        f"--bibliography={LATEX_DIR / 'referencias.bib'}",
        "--citeproc",
        f"--resource-path={LATEX_DIR}",
        f"-o",
        str(OUTPUT_DOCX),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(LATEX_DIR))

    if result.returncode != 0:
        print(f"PANDOC STDERR:\n{result.stderr}")
        raise SystemExit(f"Pandoc failed with code {result.returncode}")

    if result.stderr:
        # Count warnings
        warnings = [l for l in result.stderr.strip().split("\n") if l.strip()]
        print(f"Pandoc completed with {len(warnings)} warnings")

    size_kb = OUTPUT_DOCX.stat().st_size / 1024
    print(f"Output: {OUTPUT_DOCX} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
