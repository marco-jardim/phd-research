#!/usr/bin/env python3
"""
build_docx.py — Converte a tese LaTeX para DOCX via pandoc

Uso:
    python scripts/build_docx.py              # conversão completa
    python scripts/build_docx.py --outname meu_nome  # nome customizado
    python scripts/build_docx.py --verbose    # mostra saída completa do pandoc
    python scripts/build_docx.py --timestamp  # nome com timestamp

Requer: pandoc >= 3.0 instalado e disponível no PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "text" / "latex"
OUT_DIR = REPO_ROOT / "text" / "docx"
MAIN = "tese"

# Pandoc flags for LaTeX → DOCX
PANDOC_FLAGS: list[str] = [
    "--from=latex+raw_tex",
    "--to=docx",
    "--standalone",
    "--wrap=none",
    "--citeproc",  # process citations (natbib/bibtex via embedded bbl)
    "--toc",  # include table of contents
    "--toc-depth=3",
]


def find_pandoc() -> str | None:
    """Return pandoc executable path or None."""
    # Try common Windows installation paths first (shutil.which may return stale PATH entries)
    candidates = [
        Path(r"C:\Program Files\Pandoc\pandoc.exe"),
        Path(r"C:\ProgramData\chocolatey\bin\pandoc.exe"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Fall back to PATH lookup
    p = shutil.which("pandoc")
    if p and Path(p).exists():
        return p
    return None


def run(cmd: list[str], *, verbose: bool = False, label: str = "") -> int:
    """Run a command, return exit code."""
    if label:
        print(f"  {label}", flush=True)
    result = subprocess.run(
        cmd,
        capture_output=not verbose,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0 and not verbose:
        if result.stdout:
            print(result.stdout[-2000:])
        if result.stderr:
            print(result.stderr[-2000:], file=sys.stderr)
    elif result.stderr and verbose:
        # pandoc often writes warnings to stderr even on success
        print(result.stderr, file=sys.stderr)
    return result.returncode


def build(
    *,
    verbose: bool = False,
    outname: str | None = None,
    timestamp: bool = False,
) -> int:
    """Convert tese.tex → tese.docx using pandoc."""

    pandoc = find_pandoc()
    if pandoc is None:
        print(
            "[ERRO] 'pandoc' nao encontrado. Instale em https://pandoc.org/installing.html",
            file=sys.stderr,
        )
        return 1

    tex_file = SRC_DIR / f"{MAIN}.tex"
    bbl_file = SRC_DIR / f"{MAIN}.bbl"

    if not tex_file.exists():
        print(f"[ERRO] Arquivo fonte nao encontrado: {tex_file}", file=sys.stderr)
        return 1

    if not bbl_file.exists():
        print(
            f"[AVISO] {bbl_file} nao encontrado. "
            "Execute build_pdf.py primeiro para gerar o .bbl e ter referencias corretas.",
            file=sys.stderr,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    effective_outname = outname or MAIN
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        effective_outname = f"{effective_outname}_{ts}"
    effective_outname = effective_outname.removesuffix(".docx")

    dst = OUT_DIR / f"{effective_outname}.docx"

    cmd = [
        pandoc,
        *PANDOC_FLAGS,
        f"--output={dst}",
        "--resource-path",
        str(SRC_DIR),
        str(tex_file),
    ]

    print(f"[1/1] pandoc: {tex_file.name} → {dst.name}...", flush=True)
    rc = run(cmd, verbose=verbose)

    if rc != 0:
        print("[ERRO] pandoc falhou. Verifique as mensagens acima.", file=sys.stderr)
        return 1

    if not dst.exists():
        print("[ERRO] DOCX nao foi gerado.", file=sys.stderr)
        return 1

    size_kb = dst.stat().st_size // 1024
    print()
    print("[OK] DOCX gerado com sucesso:")
    print(f"     {dst}")
    print(f"     Tamanho: {size_kb} KB")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converte a tese LaTeX → DOCX via pandoc"
    )
    parser.add_argument(
        "--outname",
        default=None,
        help="Nome do DOCX em text/docx (sem extensao). Default: tese",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Mostra saida completa do pandoc"
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Adiciona timestamp ao nome do arquivo de saida.",
    )
    args = parser.parse_args()

    rc = build(
        verbose=args.verbose,
        outname=args.outname,
        timestamp=args.timestamp,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
