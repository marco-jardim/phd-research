#!/usr/bin/env python3
"""
build_pdf.py — Compila a tese LaTeX e gera PDF em text/pdf

Uso:
    python scripts/build_pdf.py              # compilação completa
    python scripts/build_pdf.py --quick      # apenas 1x pdflatex, sem bib/index
    python scripts/build_pdf.py --clean      # remove arquivos auxiliares
    python scripts/build_pdf.py --verbose    # mostra saída completa do LaTeX
    python scripts/build_pdf.py --timestamp  # jobname com timestamp (evita lock de PDF)
    python scripts/build_pdf.py --jobname tese_build --outname tese  # nomes customizados
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
OUT_DIR = REPO_ROOT / "text" / "pdf"
MAIN = "tese"

AUX_EXTENSIONS = [
    ".aux",
    ".log",
    ".toc",
    ".lof",
    ".lot",
    ".bbl",
    ".blg",
    ".out",
    ".nlo",
    ".nls",
    ".ilg",
    ".idx",
    ".ind",
    ".synctex.gz",
    ".nav",
    ".snm",
    ".vrb",
    ".fls",
    ".fdb_latexmk",
]


def run(cmd: list[str], *, cwd: Path, verbose: bool = False, label: str = "") -> int:
    """Run a command, return exit code. Optionally print output."""
    if label:
        print(f"  {label}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=not verbose,
        text=True,
        encoding="latin1",
        errors="replace",
    )
    if result.returncode != 0 and not verbose:
        # Show stderr/stdout on failure even in quiet mode
        if result.stdout:
            print(result.stdout[-2000:])
        if result.stderr:
            print(result.stderr[-2000:], file=sys.stderr)
    return result.returncode


def check_toolchain() -> bool:
    """Verify that pdflatex, bibtex, makeindex are available."""
    ok = True
    for tool in ("pdflatex", "bibtex", "makeindex"):
        if shutil.which(tool) is None:
            print(f"[ERRO] '{tool}' nao encontrado no PATH.", file=sys.stderr)
            ok = False
    return ok


def clean(src_dir: Path, main: str) -> None:
    """Remove build artifacts."""
    print("[CLEAN] Removendo arquivos auxiliares...")
    count = 0
    for ext in AUX_EXTENSIONS:
        for f in src_dir.glob(f"*{ext}"):
            f.unlink()
            count += 1
    print(f"[CLEAN] {count} arquivo(s) removido(s).")


def is_write_locked(path: Path) -> bool:
    """Best-effort check for Windows PDF lock.

    If a PDF viewer is holding an exclusive lock, pdflatex will fail with:
    "I can't write on file '<name>.pdf'".
    """
    if not path.exists():
        return False
    try:
        with path.open("ab"):
            return False
    except (PermissionError, OSError):
        return True


def build(
    *,
    quick: bool = False,
    verbose: bool = False,
    jobname: str | None = None,
    outname: str | None = None,
    timestamp: bool = False,
) -> int:
    """Full build chain: pdflatex → bibtex → makeindex → pdflatex × 2.

    Notes (Windows): if the output PDF is open, pdflatex may fail to overwrite it.
    Use --timestamp or --jobname to compile to a different PDF name.
    """

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not check_toolchain():
        return 1

    if not (SRC_DIR / f"{MAIN}.tex").exists():
        print(
            f"[ERRO] Arquivo fonte nao encontrado: {SRC_DIR / MAIN}.tex",
            file=sys.stderr,
        )
        return 1

    effective_jobname = jobname
    if timestamp:
        effective_jobname = f"{MAIN}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if outname is None:
            outname = effective_jobname

    if effective_jobname is None:
        # Default jobname is MAIN, but auto-fallback if MAIN.pdf seems locked.
        default_pdf = SRC_DIR / f"{MAIN}.pdf"
        if is_write_locked(default_pdf):
            effective_jobname = f"{MAIN}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if outname is None:
                outname = effective_jobname
            print(
                f"[AVISO] {default_pdf} parece estar aberto/bloqueado; "
                f"usando jobname '{effective_jobname}' para evitar erro de escrita."
            )
        else:
            effective_jobname = MAIN

    effective_outname = (outname or MAIN).removesuffix(".pdf")

    pdflatex_cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-jobname={effective_jobname}",
        f"{MAIN}.tex",
    ]

    total_steps = 2 if quick else 5
    step = 0

    # --- Pass 1 ---
    step += 1
    rc = run(
        pdflatex_cmd,
        cwd=SRC_DIR,
        verbose=verbose,
        label=f"[{step}/{total_steps}] pdflatex (1a passada)...",
    )
    if rc != 0:
        print("[ERRO] pdflatex falhou na 1a passada.")
        print(f"       Verifique: {SRC_DIR / f'{effective_jobname}.log'}")
        return 1

    if quick:
        step += 1
        print(f"  [{step}/{total_steps}] Copiando PDF (modo --quick)...")
    else:
        # --- BibTeX ---
        step += 1
        rc = run(
            ["bibtex", effective_jobname],
            cwd=SRC_DIR,
            verbose=verbose,
            label=f"[{step}/{total_steps}] bibtex...",
        )
        if rc != 0:
            print("[AVISO] bibtex retornou warnings (pode ser normal).")

        # --- makeindex (nomenclature) ---
        step += 1
        nlo = SRC_DIR / f"{effective_jobname}.nlo"
        if nlo.exists():
            rc = run(
                [
                    "makeindex",
                    f"{effective_jobname}.nlo",
                    "-s",
                    "nomencl.ist",
                    "-o",
                    f"{effective_jobname}.nls",
                ],
                cwd=SRC_DIR,
                verbose=verbose,
                label=f"[{step}/{total_steps}] makeindex (nomenclatura)...",
            )
            if rc != 0:
                print("[AVISO] makeindex retornou warnings.")
        else:
            print(f"  [{step}/{total_steps}] makeindex: sem .nlo, pulando.")

        # --- Pass 2 ---
        step += 1
        rc = run(
            pdflatex_cmd,
            cwd=SRC_DIR,
            verbose=verbose,
            label=f"[{step}/{total_steps}] pdflatex (2a passada)...",
        )
        if rc != 0:
            print("[ERRO] pdflatex falhou na 2a passada.")
            return 1

        # --- Pass 3 ---
        step += 1
        rc = run(
            pdflatex_cmd,
            cwd=SRC_DIR,
            verbose=verbose,
            label=f"[{step}/{total_steps}] pdflatex (3a passada)...",
        )
        if rc != 0:
            print("[ERRO] pdflatex falhou na 3a passada.")
            return 1

    # --- Copy PDF ---
    src_pdf = SRC_DIR / f"{effective_jobname}.pdf"
    dst_pdf = OUT_DIR / f"{effective_outname}.pdf"

    if not src_pdf.exists():
        print("[ERRO] PDF nao foi gerado.", file=sys.stderr)
        return 1

    try:
        shutil.copy2(str(src_pdf), str(dst_pdf))
    except PermissionError:
        # Destination likely open/locked. Save a side-by-side copy.
        fallback = OUT_DIR / f"{effective_jobname}.pdf"
        shutil.copy2(str(src_pdf), str(fallback))
        dst_pdf = fallback
        print(
            f"[AVISO] Nao foi possivel sobrescrever {OUT_DIR / f'{effective_outname}.pdf'} "
            f"(provavelmente aberto). PDF salvo em: {dst_pdf}"
        )
    size_kb = dst_pdf.stat().st_size // 1024
    print()
    print(f"[OK] PDF gerado com sucesso:")
    print(f"     {dst_pdf}")
    print(f"     Tamanho: {size_kb} KB")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compila a tese LaTeX → PDF")
    parser.add_argument(
        "--quick", action="store_true", help="Apenas 1x pdflatex, sem bibtex/makeindex"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Remove arquivos auxiliares de compilacao"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Mostra saida completa do LaTeX"
    )
    parser.add_argument(
        "--jobname",
        default=None,
        help=(
            "Jobname do pdflatex (prefixo de .aux/.log/.pdf). "
            "Use para evitar lock do PDF no Windows."
        ),
    )
    parser.add_argument(
        "--outname",
        default=None,
        help="Nome do PDF em text/pdf (sem extensao). Default: tese",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Gera jobname com timestamp (e, por padrao, salva o PDF com o mesmo nome).",
    )
    args = parser.parse_args()

    if args.clean:
        clean(SRC_DIR, MAIN)
        sys.exit(0)

    rc = build(
        quick=args.quick,
        verbose=args.verbose,
        jobname=args.jobname,
        outname=args.outname,
        timestamp=args.timestamp,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
