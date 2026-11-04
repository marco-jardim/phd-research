"""Compila os slides sem modificar o PDF oficial da tese.

Uso, a partir de qualquer diretório: python text/slides/build_slides.py
Requer pdflatex no PATH. A compilação usa apenas arquivos versionados.
"""

import os
from pathlib import Path
import shutil
import subprocess


def build() -> Path:
    directory = Path(__file__).resolve().parent
    executable = shutil.which("pdflatex")
    if executable is None:
        raise RuntimeError("pdflatex não encontrado no PATH")
    environment = os.environ.copy()
    dependencies = directory.parents[1] / ".texdeps"
    if dependencies.is_dir():
        environment["TEXINPUTS"] = (
            dependencies.as_posix() + "//" + os.pathsep
            + environment.get("TEXINPUTS", "")
        )
    for _ in range(2):
        subprocess.run(
            [executable, "-interaction=nonstopmode", "-halt-on-error", "defesa.tex"],
            cwd=directory,
            check=True,
            timeout=600,
            env=environment,
        )
    pdf = directory / "defesa.pdf"
    if not pdf.is_file():
        raise RuntimeError("A compilação não produziu defesa.pdf")
    print(f"PDF: {pdf}")
    return pdf


if __name__ == "__main__":
    build()
