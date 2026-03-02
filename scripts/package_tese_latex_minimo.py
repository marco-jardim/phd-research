#!/usr/bin/env python3
"""
Gera um ZIP minimo com apenas os arquivos necessarios para compilar a tese.

Uso:
    python scripts/package_tese_latex_minimo.py
    python scripts/package_tese_latex_minimo.py --output tese_latex_minimo.zip
    python scripts/package_tese_latex_minimo.py --validate
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LATEX_DIR = REPO_ROOT / "text" / "latex"
MAIN_TEX = LATEX_DIR / "tese.tex"

TEXT_REF_EXTS = [".tex", ".pgf"]
BIB_EXTS = [".bib"]
ASSET_EXTS = [".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg", ".pgf"]

RE_INPUT_INCLUDE = re.compile(r"\\(input|include)\{([^}]+)\}")
RE_INCLUDEGRAPHICS = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
RE_INCLUDEPDF = re.compile(r"\\includepdf(?:\[[^\]]*\])?\{([^}]+)\}")
RE_BIB = re.compile(r"\\bibliography\{([^}]+)\}")
RE_USEPACKAGE = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}")
RE_DOCUMENTCLASS = re.compile(r"\\documentclass(?:\[[^\]]*\])?\{([^}]+)\}")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def resolve_candidate(ref: str, parent: Path, exts: list[str]) -> Path | None:
    ref = ref.strip()
    if not ref:
        return None
    if ref.startswith("./"):
        ref = ref[2:]

    direct = parent / ref
    if direct.exists() and direct.is_file():
        return direct.resolve()

    if Path(ref).suffix:
        return None

    for ext in exts:
        candidate = parent / f"{ref}{ext}"
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def is_placeholder_ref(ref: str) -> bool:
    return "<" in ref or ">" in ref


def include_local_style_files(
    tex_content: str, base_dir: Path, required: set[Path]
) -> None:
    # Inclui .cls local se existir
    m = RE_DOCUMENTCLASS.search(tex_content)
    if m:
        for cls_name in [x.strip() for x in m.group(1).split(",") if x.strip()]:
            cls_file = base_dir / f"{cls_name}.cls"
            if cls_file.exists():
                required.add(cls_file.resolve())

    # Inclui .sty locais citados em \usepackage{...}
    for m in RE_USEPACKAGE.finditer(tex_content):
        for pkg in [x.strip() for x in m.group(1).split(",") if x.strip()]:
            sty_file = base_dir / f"{pkg}.sty"
            if sty_file.exists():
                required.add(sty_file.resolve())


def discover_dependencies(main_tex: Path) -> tuple[list[Path], list[str]]:
    base_dir = main_tex.parent.resolve()
    visited: set[Path] = set()
    required: set[Path] = set()
    missing: set[str] = set()

    def scan_tex(path: Path) -> None:
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        required.add(path)

        content = read_text(path)
        include_local_style_files(content, base_dir, required)

        parent = path.parent

        for match in RE_INPUT_INCLUDE.finditer(content):
            ref = match.group(2)
            if is_placeholder_ref(ref):
                continue
            target = resolve_candidate(ref, parent, TEXT_REF_EXTS)
            if target and str(target).startswith(str(base_dir)):
                scan_tex(target)
            elif not target:
                missing.add(str((parent / ref).as_posix()))

        for match in RE_BIB.finditer(content):
            for ref in [x.strip() for x in match.group(1).split(",") if x.strip()]:
                if is_placeholder_ref(ref):
                    continue
                target = resolve_candidate(ref, parent, BIB_EXTS)
                if target and str(target).startswith(str(base_dir)):
                    required.add(target)
                else:
                    missing.add(str((parent / ref).as_posix()))

        for pattern in (RE_INCLUDEGRAPHICS, RE_INCLUDEPDF):
            for match in pattern.finditer(content):
                ref = match.group(1)
                if is_placeholder_ref(ref):
                    continue
                target = resolve_candidate(ref, parent, ASSET_EXTS)

                # fallback para \graphicspath{{fig_doc/}}
                if not target:
                    target = resolve_candidate(ref, base_dir / "fig_doc", ASSET_EXTS)

                if target and str(target).startswith(str(base_dir)):
                    required.add(target)
                    if target.suffix.lower() == ".pgf":
                        scan_tex(target)
                else:
                    missing.add(ref)

    scan_tex(main_tex)
    return sorted(required), sorted(missing)


def create_zip(files: list[Path], base_dir: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            zf.write(file_path, file_path.relative_to(base_dir).as_posix())


def run_cmd(cmd: list[str], cwd: Path) -> int:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="latin-1",
        errors="replace",
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout[-2000:])
        if result.stderr:
            print(result.stderr[-2000:], file=sys.stderr)
    return result.returncode


def validate_package(zip_path: Path) -> int:
    for tool in ("pdflatex", "bibtex"):
        if shutil.which(tool) is None:
            print(f"[ERRO] '{tool}' nao encontrado no PATH; validacao nao pode rodar.")
            return 2

    with tempfile.TemporaryDirectory(prefix="tese_zip_validate_") as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        steps = [
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "tese.tex"],
            ["bibtex", "tese"],
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "tese.tex"],
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "tese.tex"],
        ]

        for i, cmd in enumerate(steps, start=1):
            print(f"[VALIDACAO {i}/4] {' '.join(cmd)}")
            rc = run_cmd(cmd, cwd=tmp_dir)
            if rc != 0:
                print("[ERRO] Falha na validacao de compilacao.")
                return 1

        out_pdf = tmp_dir / "tese.pdf"
        if not out_pdf.exists():
            print("[ERRO] Validacao falhou: tese.pdf nao foi gerado.")
            return 1

        print(f"[OK] Validacao concluida: PDF gerado ({out_pdf.stat().st_size} bytes).")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Empacota os arquivos minimos necessarios para compilar text/latex/tese.tex"
    )
    parser.add_argument(
        "--main",
        default=str(MAIN_TEX),
        help="Caminho para o .tex principal (default: text/latex/tese.tex)",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "tese_latex_minimo.zip"),
        help="Arquivo ZIP de saida (default: tese_latex_minimo.zip na raiz do repo)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Extrai o ZIP em pasta temporaria e testa compilacao completa (pdflatex/bibtex)",
    )
    args = parser.parse_args()

    main_tex = Path(args.main).resolve()
    output_zip = Path(args.output).resolve()

    if not main_tex.exists():
        print(f"[ERRO] Arquivo principal nao encontrado: {main_tex}", file=sys.stderr)
        sys.exit(1)

    files, missing = discover_dependencies(main_tex)
    if not files:
        print("[ERRO] Nenhum arquivo encontrado para empacotar.", file=sys.stderr)
        sys.exit(1)

    create_zip(files, base_dir=main_tex.parent.resolve(), output_zip=output_zip)

    print(f"[OK] ZIP gerado: {output_zip}")
    print(f"[OK] Arquivos incluidos: {len(files)}")

    if missing:
        print("[AVISO] Referencias nao resolvidas (nao impediram o empacotamento):")
        for item in missing:
            print(f"  - {item}")

    if args.validate:
        rc = validate_package(output_zip)
        sys.exit(rc)


if __name__ == "__main__":
    main()
