"""Compila os slides sem modificar o PDF oficial da tese.

Uso, a partir de qualquer diretório: python text/slides/build_slides.py
Requer pdflatex no PATH. A compilação usa apenas arquivos versionados.

Sem --jobname, o produto é text/slides/defesa.pdf, que só muda em build
aprovado. Builds de teste usam um jobname alternativo, na forma
defesa_<tag>_<hash>, e gravam text/slides/defesa_<tag>_<hash>.pdf (ignorado
pelo git). O jobname aceita apenas [A-Za-z0-9_.-]; "--jobname defesa" equivale
ao padrão.
"""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess


JOBNAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


def build(jobname: str | None = None) -> Path:
    if jobname is not None and not JOBNAME_PATTERN.fullmatch(jobname):
        raise ValueError(
            f"jobname inválido: {jobname!r}; use apenas letras, dígitos, '_', '.' e '-'"
        )
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
    command = [executable, "-interaction=nonstopmode", "-halt-on-error"]
    if jobname is not None:
        command.append(f"-jobname={jobname}")
    command.append("defesa.tex")
    for _ in range(2):
        subprocess.run(
            command,
            cwd=directory,
            check=True,
            timeout=600,
            env=environment,
        )
    name = f"{jobname or 'defesa'}.pdf"
    pdf = directory / name
    if not pdf.is_file():
        raise RuntimeError(f"A compilação não produziu {name}")
    print(f"PDF: {pdf}")
    return pdf


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Compila os slides de defesa.")
    parser.add_argument(
        "--jobname",
        default=None,
        help="nome do produto de build de teste (defesa_<tag>_<hash>); "
        "sem ele, grava defesa.pdf",
    )
    arguments = parser.parse_args(argv)
    return build(arguments.jobname)


if __name__ == "__main__":
    main()
