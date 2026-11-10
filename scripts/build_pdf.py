#!/usr/bin/env python3
"""
build_pdf.py: compila a tese LaTeX e copia o PDF para text/latex e text/pdf.

A compilacao acontece num diretorio proprio e descartavel, text/latex/build/final/
(ignorado pelo git), limpo antes de cada build completo. Assim o resultado nao
depende de .aux/.bbl antigos deixados em text/latex por outras compilacoes.

Uso:
    python scripts/build_pdf.py              # compilacao completa e verificada
    python scripts/build_pdf.py --quick      # apenas 1x pdflatex, sem bib/index (PDF fica em build/final)
    python scripts/build_pdf.py --check      # so verifica o log do ultimo build, sem compilar
    python scripts/build_pdf.py --clean      # remove arquivos auxiliares e build/final
    python scripts/build_pdf.py --verbose    # mostra saida completa do LaTeX
    python scripts/build_pdf.py --timestamp  # jobname com timestamp (evita lock de PDF)
    python scripts/build_pdf.py --jobname tese_build --outname tese  # nomes customizados
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "text" / "latex"
OUT_DIR = REPO_ROOT / "text" / "pdf"
BUILD_SUBDIR = "build/final"  # relativo a SRC_DIR; text/latex/build/ e ignorado pelo git
MAIN = "tese"

MIN_PASSES = 3  # pdflatex, bibtex, pdflatex x2 (mesma sequencia minima do CI)
MAX_PASSES = 5

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

# Com -output-directory, o TeX Live procura cada arquivo lido primeiro no
# diretorio de saida e, se nao achar, faz a busca normal, que inclui o
# diretorio corrente (text/latex). Placeholders vazios no build limpo impedem
# que a 1a passada (ou uma passada apos bibtex falho) leia tese.aux/tese.bbl
# antigos de text/latex.
SEED_EXTENSIONS = (".aux", ".bbl", ".brf", ".out", ".toc", ".lof", ".lot", ".nls")

# Arquivos cujo conteudo muda enquanto as referencias nao estabilizam.
TRACKED_EXTENSIONS = (".aux", ".out", ".toc", ".lof", ".lot", ".brf")

# Mesmos valores do CI: log sem quebra de linha a cada 79 caracteres.
TEX_ENV = {"max_print_line": "10000", "error_line": "254", "half_error_line": "238"}
TEX_WRAP = 79  # max_print_line padrao do TeX, para logs gerados sem TEX_ENV

RERUN_RE = re.compile(
    r"Rerun to get|Label\(s\) may have changed|rerun LaTeX", re.IGNORECASE
)
CITATION_RE = re.compile(r"Citation `([^']*)'(?: on page \S+)? undefined")
REFERENCE_RE = re.compile(r"Reference `([^']*)'(?: on page \S+)? undefined")
MULTIPLY_RE = re.compile(r"Label `([^']*)' multiply defined")
ERROR_RE = re.compile(r"^(?:! .*|.*LaTeX Error.*|\S+\.tex:\d+: .*)$", re.MULTILINE)
PAGES_RE = re.compile(r"Output written on .*? \((\d+) pages?")
BLG_ERROR_RE = re.compile(
    r"^(?:\(There (?:was|were) \d+ error messages?\)"
    r"|I couldn't open"
    r"|I found no"
    r"|I'm skipping whatever remains"
    r"|Repeated entry"
    r"|Illegal,"
    r"|Sorry---you've exceeded"
    r"|Aborted at line).*$",
    re.MULTILINE,
)
BLG_WARNING_RE = re.compile(r"^Warning--", re.MULTILINE)


# ---------------------------------------------------------------------------
# Analise de logs (funcoes puras, testadas sem LaTeX)
# ---------------------------------------------------------------------------
def unwrap_log(text: str) -> str:
    """Junta as linhas que o TeX quebrou em TEX_WRAP caracteres."""
    lines: list[str] = []
    buffer = ""
    for line in text.splitlines():
        buffer += line
        if len(line) != TEX_WRAP:
            lines.append(buffer)
            buffer = ""
    if buffer:
        lines.append(buffer)
    return "\n".join(lines)


def needs_rerun(log_text: str) -> bool:
    """True se o log pede nova passada (rotulos, outlines, backref, longtable)."""
    return RERUN_RE.search(unwrap_log(log_text)) is not None


def is_stable(
    log_text: str, previous: dict[str, str], current: dict[str, str]
) -> bool:
    """Estavel = o log nao pede nova passada e os auxiliares nao mudaram."""
    return not needs_rerun(log_text) and previous == current


class LogReport(NamedTuple):
    undefined_citations: list[str]
    undefined_references: list[str]
    multiply_defined: list[str]
    errors: list[str]
    undefined_summary: bool
    multiply_summary: bool
    pages: int | None

    def problems(self) -> list[str]:
        found: list[str] = []
        if self.errors:
            found.append(f"{len(self.errors)} erro(s) do LaTeX: {self.errors[0]}")
        if self.undefined_citations:
            found.append(
                f"{len(self.undefined_citations)} citacao(oes) indefinida(s): "
                + _preview(self.undefined_citations)
            )
        if self.undefined_references:
            found.append(
                f"{len(self.undefined_references)} referencia(s) indefinida(s): "
                + _preview(self.undefined_references)
            )
        if self.undefined_summary and not (
            self.undefined_citations or self.undefined_references
        ):
            found.append("o LaTeX avisou referencias indefinidas")
        if self.multiply_defined:
            found.append(
                f"{len(self.multiply_defined)} rotulo(s) duplicado(s): "
                + _preview(self.multiply_defined)
            )
        elif self.multiply_summary:
            found.append("o LaTeX avisou rotulos duplicados")
        return found

    @property
    def ok(self) -> bool:
        return not self.problems()


def _preview(items: list[str], limit: int = 10) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f" (+{len(items) - limit})" if len(items) > limit else "")


def analyze_log(log_text: str) -> LogReport:
    flat = unwrap_log(log_text)
    pages = PAGES_RE.findall(flat)
    return LogReport(
        undefined_citations=sorted(set(CITATION_RE.findall(flat))),
        undefined_references=sorted(set(REFERENCE_RE.findall(flat))),
        multiply_defined=sorted(set(MULTIPLY_RE.findall(flat))),
        errors=[m.group(0).strip() for m in ERROR_RE.finditer(log_text)],
        undefined_summary="There were undefined references" in flat
        or "There were undefined citations" in flat,
        multiply_summary="There were multiply-defined labels" in flat,
        pages=int(pages[-1]) if pages else None,
    )


def bibtex_problems(
    returncode: int, blg_text: str | None, bbl_size: int | None
) -> list[str]:
    """Erros do bibtex: codigo >= 2, .blg ausente/com erro, .bbl ausente/vazio."""
    found: list[str] = []
    if returncode >= 2:
        found.append(f"bibtex saiu com codigo {returncode}")
    if blg_text is None:
        found.append("bibtex nao gerou o .blg (nao rodou sobre o .aux deste build)")
    else:
        found.extend(m.group(0).strip() for m in BLG_ERROR_RE.finditer(blg_text))
    if not bbl_size:
        found.append("bibtex nao gerou o .bbl (ausente ou vazio)")
    return found


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------
def build_dir() -> Path:
    return SRC_DIR / BUILD_SUBDIR


def tex_env(extra_inputs: Path | None = None) -> dict[str, str]:
    """Ambiente dos comandos TeX; extra_inputs entra na frente de BIBINPUTS/BSTINPUTS."""
    env = os.environ.copy()
    env.update(TEX_ENV)
    if extra_inputs is not None:
        # O separador final mantem a busca padrao do kpathsea (TEXMF).
        for var in ("BIBINPUTS", "BSTINPUTS"):
            env[var] = f"{extra_inputs}{os.pathsep}{os.environ.get(var, '')}"
    return env


def run(
    cmd: list[str],
    *,
    cwd: Path,
    verbose: bool = False,
    label: str = "",
    env: dict[str, str] | None = None,
) -> int:
    """Run a command, return exit code. Optionally print output."""
    if label:
        print(f"  {label}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
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


def clean(src_dir: Path, main: str) -> int:
    """Remove build artifacts."""
    print("[CLEAN] Removendo arquivos auxiliares...")
    count = 0
    for ext in AUX_EXTENSIONS:
        for f in src_dir.glob(f"*{ext}"):
            f.unlink()
            count += 1
    print(f"[CLEAN] {count} arquivo(s) removido(s).")
    work = src_dir / BUILD_SUBDIR
    if work.exists():
        try:
            shutil.rmtree(work)
        except PermissionError as exc:
            print(f"[ERRO] Nao foi possivel remover {work}: {exc}", file=sys.stderr)
            return 1
        print(f"[CLEAN] Diretorio de build removido: {work}")
    return 0


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


def read_text(path: Path) -> str:
    return path.read_text(encoding="latin1", errors="replace")


def fingerprint(work: Path, jobname: str) -> dict[str, str]:
    digests: dict[str, str] = {}
    for ext in TRACKED_EXTENSIONS:
        path = work / f"{jobname}{ext}"
        if path.exists():
            digests[ext] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def prepare_clean_build(work: Path, jobname: str) -> int:
    """Apaga os arquivos deste jobname em build/final e cria os placeholders."""
    work.mkdir(parents=True, exist_ok=True)
    for old in work.glob(f"{glob.escape(jobname)}.*"):
        try:
            old.unlink()
        except PermissionError:
            print(
                f"[ERRO] {old} esta bloqueado (aberto em outro programa?). "
                "Feche o arquivo ou use --jobname/--timestamp.",
                file=sys.stderr,
            )
            return 1
    for ext in SEED_EXTENSIONS:
        (work / f"{jobname}{ext}").write_bytes(b"")
    return 0


def print_report(report: LogReport, *, as_error: bool) -> None:
    if report.pages is not None:
        print(f"  [INFO] Paginas: {report.pages}")
    tag = "[ERRO]" if as_error else "[AVISO]"
    for problem in report.problems():
        print(f"  {tag} {problem}")


def copy_pdf(src_pdf: Path, dst_pdf: Path) -> Path:
    dst_pdf.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(str(src_pdf), str(dst_pdf))
    except PermissionError:
        # Destination likely open/locked. Save a side-by-side copy.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = dst_pdf.with_name(f"{dst_pdf.stem}_{stamp}.pdf")
        shutil.copy2(str(src_pdf), str(fallback))
        print(
            f"[AVISO] Nao foi possivel sobrescrever {dst_pdf} "
            f"(provavelmente aberto). PDF salvo em: {fallback}"
        )
        return fallback
    return dst_pdf


def resolve_names(
    jobname: str | None, outname: str | None, timestamp: bool, work: Path
) -> tuple[str, str]:
    effective_jobname = jobname
    if timestamp:
        effective_jobname = f"{MAIN}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if outname is None:
            outname = effective_jobname

    if effective_jobname is None:
        # Default jobname is MAIN, but auto-fallback if the build PDF seems locked.
        default_pdf = work / f"{MAIN}.pdf"
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

    return effective_jobname, (outname or MAIN).removesuffix(".pdf")


def build(
    *,
    quick: bool = False,
    verbose: bool = False,
    jobname: str | None = None,
    outname: str | None = None,
    timestamp: bool = False,
) -> int:
    """pdflatex, bibtex, makeindex e pdflatex ate estabilizar, em build/final.

    O build completo so copia o PDF se o bibtex rodar sem erro e o log final
    nao tiver citacao/referencia indefinida, erro do LaTeX nem rotulo duplicado.
    """
    if not check_toolchain():
        return 1

    if not (SRC_DIR / f"{MAIN}.tex").exists():
        print(
            f"[ERRO] Arquivo fonte nao encontrado: {SRC_DIR / MAIN}.tex",
            file=sys.stderr,
        )
        return 1

    work = build_dir()
    work.mkdir(parents=True, exist_ok=True)
    effective_jobname, effective_outname = resolve_names(
        jobname, outname, timestamp, work
    )
    log_path = work / f"{effective_jobname}.log"
    env = tex_env()

    pdflatex_cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-output-directory={BUILD_SUBDIR}",
        f"-jobname={effective_jobname}",
        f"{MAIN}.tex",
    ]

    if not quick and prepare_clean_build(work, effective_jobname) != 0:
        return 1

    total_steps = 2 if quick else 5
    step = 1
    rc = run(
        pdflatex_cmd,
        cwd=SRC_DIR,
        verbose=verbose,
        env=env,
        label=f"[{step}/{total_steps}] pdflatex (1a passada)...",
    )
    if rc != 0:
        print("[ERRO] pdflatex falhou na 1a passada.")
        print(f"       Verifique: {log_path}")
        return 1

    if quick:
        step += 1
        print(f"  [{step}/{total_steps}] Verificando log (modo --quick, sem bibtex)...")
        print_report(analyze_log(read_text(log_path)), as_error=False)
        src_pdf = work / f"{effective_jobname}.pdf"
        if not src_pdf.exists():
            print("[ERRO] PDF nao foi gerado.", file=sys.stderr)
            return 1
        print()
        print("[OK] PDF rapido (nao copiado para text/latex nem text/pdf):")
        print(f"     {src_pdf}")
        return 0

    # --- BibTeX (roda dentro de build/final; .bib e .bst vem de text/latex) ---
    step += 1
    rc = run(
        ["bibtex", effective_jobname],
        cwd=work,
        verbose=verbose,
        env=tex_env(SRC_DIR),
        label=f"[{step}/{total_steps}] bibtex...",
    )
    blg = work / f"{effective_jobname}.blg"
    bbl = work / f"{effective_jobname}.bbl"
    blg_text = read_text(blg) if blg.exists() else None
    problems = bibtex_problems(rc, blg_text, bbl.stat().st_size if bbl.exists() else None)
    if problems:
        print("[ERRO] bibtex falhou:")
        for problem in problems:
            print(f"       {problem}")
        print(f"       Verifique: {blg}")
        return 1
    warnings = len(BLG_WARNING_RE.findall(blg_text or ""))
    if warnings:
        print(f"  [INFO] bibtex: {warnings} aviso(s) em {blg}")

    # --- makeindex (nomenclature) ---
    step += 1
    nlo = work / f"{effective_jobname}.nlo"
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
            cwd=work,
            verbose=verbose,
            env=env,
            label=f"[{step}/{total_steps}] makeindex (nomenclatura)...",
        )
        if rc != 0:
            print("[AVISO] makeindex retornou warnings.")
    else:
        print(f"  [{step}/{total_steps}] makeindex: sem .nlo, pulando.")

    # --- pdflatex ate estabilizar ---
    step += 1
    print(f"  [{step}/{total_steps}] pdflatex ate estabilizar (max. {MAX_PASSES} passadas)...")
    previous = fingerprint(work, effective_jobname)
    pass_number = 1
    while True:
        pass_number += 1
        rc = run(
            pdflatex_cmd,
            cwd=SRC_DIR,
            verbose=verbose,
            env=env,
            label=f"  pdflatex ({pass_number}a passada)...",
        )
        if rc != 0:
            print(f"[ERRO] pdflatex falhou na {pass_number}a passada.")
            print(f"       Verifique: {log_path}")
            return 1
        current = fingerprint(work, effective_jobname)
        log_text = read_text(log_path)
        if pass_number >= MIN_PASSES and is_stable(log_text, previous, current):
            print(f"  pdflatex estabilizado em {pass_number} passadas.")
            break
        if pass_number >= MAX_PASSES:
            print(f"[ERRO] Referencias nao estabilizaram apos {MAX_PASSES} passadas.")
            print_report(analyze_log(log_text), as_error=True)
            print(f"       Verifique: {log_path}")
            return 1
        previous = current

    # --- Verificacao do log final ---
    step += 1
    print(f"  [{step}/{total_steps}] Verificando log final...")
    report = analyze_log(log_text)
    print_report(report, as_error=True)
    if not report.ok:
        print("[ERRO] Log final com problemas; PDF nao copiado.")
        print(f"       Verifique: {log_path}")
        return 1

    # --- Copy PDF ---
    src_pdf = work / f"{effective_jobname}.pdf"
    if not src_pdf.exists():
        print("[ERRO] PDF nao foi gerado.", file=sys.stderr)
        return 1

    copied = [
        copy_pdf(src_pdf, SRC_DIR / f"{effective_outname}.pdf"),
        copy_pdf(src_pdf, OUT_DIR / f"{effective_outname}.pdf"),
    ]
    size_kb = src_pdf.stat().st_size // 1024
    print()
    print("[OK] PDF gerado com sucesso:")
    for dst_pdf in copied:
        print(f"     {dst_pdf}")
    print(f"     Tamanho: {size_kb} KB")
    return 0


def check(*, jobname: str | None = None) -> int:
    """Verifica o log do ultimo build em build/final, sem compilar."""
    log_path = build_dir() / f"{jobname or MAIN}.log"
    if not log_path.exists():
        print(f"[ERRO] Log nao encontrado: {log_path}. Rode o build antes.")
        return 1
    print(f"[CHECK] {log_path}")
    report = analyze_log(read_text(log_path))
    print_report(report, as_error=True)
    if not report.ok:
        return 1
    print("[OK] Sem citacao/referencia indefinida, erro do LaTeX ou rotulo duplicado.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compila a tese LaTeX em PDF")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick", action="store_true", help="Apenas 1x pdflatex, sem bibtex/makeindex"
    )
    mode.add_argument(
        "--clean", action="store_true", help="Remove arquivos auxiliares de compilacao"
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help=(
            "Nao compila: verifica o log do ultimo build em build/final e sai com "
            "codigo 1 se houver citacao/referencia indefinida, LaTeX Error ou "
            "rotulo duplicado"
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Mostra saida completa do LaTeX"
    )
    parser.add_argument(
        "--jobname",
        default=None,
        help=(
            "Jobname do pdflatex (prefixo de .aux/.log/.pdf em build/final). "
            "Use para evitar lock do PDF no Windows."
        ),
    )
    parser.add_argument(
        "--outname",
        default=None,
        help="Nome do PDF copiado para text/latex e text/pdf (sem extensao). Default: tese",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Gera jobname com timestamp (e, por padrao, salva o PDF com o mesmo nome).",
    )
    args = parser.parse_args()

    if args.clean:
        sys.exit(clean(SRC_DIR, MAIN))

    if args.check:
        if args.timestamp:
            parser.error("--check nao combina com --timestamp (use --jobname)")
        sys.exit(check(jobname=args.jobname))

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
