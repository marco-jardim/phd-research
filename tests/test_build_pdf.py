"""Checks for scripts/build_pdf.py: clean build dir, effective bibtex, bounded stabilization."""

import importlib.util
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

CLEAN_BLG = (
    "This is BibTeX, Version 0.99d (TeX Live 2024)\n"
    "The top-level auxiliary file: check.aux\n"
    "The style file: abntex2-alf.bst\n"
    "Database file #1: abntex2-options.bib\n"
    "Database file #2: referencias.bib\n"
    "Warning--empty publisher in Fellegi1969theory\n"
    "You've used 129 entries,\n"
)
ERROR_BLG = (
    "This is BibTeX, Version 0.99d (TeX Live 2024)\n"
    "The top-level auxiliary file: check.aux\n"
    "The style file: abntex2-alf.bst\n"
    "Database file #1: abntex2-options.bib\n"
    "I couldn't open database file referencias.bib\n"
    "---line 1042 of file check.aux\n"
    "(There was 1 error message)\n"
)
RERUN_LABELS = "LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.\n"
RERUNFILECHECK_INFO = (
    "Package: rerunfilecheck 2022-07-10 v1.10 Rerun checks for auxiliary files (HO)\n"
)


@pytest.fixture
def build_module(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_pdf.py"
    spec = importlib.util.spec_from_file_location("build_pdf", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SRC_DIR = tmp_path / "latex"
    module.OUT_DIR = tmp_path / "pdf"
    module.SRC_DIR.mkdir()
    (module.SRC_DIR / "tese.tex").write_text("test", encoding="utf-8")
    return module


class FakeTeX:
    """Substitui run(): escreve .log/.pdf (pdflatex) e .blg/.bbl (bibtex) em build/final."""

    def __init__(
        self,
        module,
        *,
        stable_after=3,
        fail_on_pass=None,
        extra_log="",
        bibtex_writes=True,
        blg_text=CLEAN_BLG,
        bibtex_rc=1,
    ):
        self.module = module
        self.stable_after = stable_after
        self.fail_on_pass = fail_on_pass
        self.extra_log = extra_log
        self.bibtex_writes = bibtex_writes
        self.blg_text = blg_text
        self.bibtex_rc = bibtex_rc
        self.passes = 0
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        work = self.module.build_dir()
        if cmd[0] == "pdflatex":
            self.passes += 1
            job = next(arg.split("=", 1)[1] for arg in cmd if arg.startswith("-jobname="))
            log = RERUNFILECHECK_INFO
            if self.passes < self.stable_after:
                log += RERUN_LABELS
            log += self.extra_log
            log += f"Output written on build/final/{job}.pdf (3 pages, 1234 bytes).\n"
            (work / f"{job}.log").write_text(log, encoding="latin1")
            (work / f"{job}.pdf").write_bytes(b"test pdf")
            return 1 if self.passes == self.fail_on_pass else 0
        if cmd[0] == "bibtex":
            job = cmd[1]
            if self.bibtex_writes:
                (work / f"{job}.blg").write_text(self.blg_text, encoding="latin1")
                (work / f"{job}.bbl").write_text(
                    "\\begin{thebibliography}{1}\n\\end{thebibliography}\n",
                    encoding="latin1",
                )
            return self.bibtex_rc
        return 0


def run_build(module, fake, **kwargs):
    with patch.object(module, "check_toolchain", return_value=True), patch.object(
        module, "run", side_effect=fake
    ):
        return module.build(**kwargs)


# ---------------------------------------------------------------------------
# Estabilizacao (sem LaTeX)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "stable_after,expected_rc,expected_passes", [(3, 0, 3), (4, 0, 4), (5, 0, 5), (6, 1, 5)]
)
def test_build_waits_for_stable_references(
    build_module, stable_after, expected_rc, expected_passes
):
    fake = FakeTeX(build_module, stable_after=stable_after)
    assert run_build(build_module, fake, jobname="check", outname="check") == expected_rc
    assert fake.passes == expected_passes
    assert (build_module.OUT_DIR / "check.pdf").exists() == (expected_rc == 0)
    assert (build_module.SRC_DIR / "check.pdf").exists() == (expected_rc == 0)


def test_build_stops_on_extra_pass_failure(build_module):
    fake = FakeTeX(build_module, stable_after=99, fail_on_pass=4)
    assert run_build(build_module, fake, jobname="check", outname="check") == 1
    assert fake.passes == 4
    assert not (build_module.OUT_DIR / "check.pdf").exists()


@pytest.mark.parametrize(
    "text",
    [
        RERUN_LABELS,
        "(rerunfilecheck)                Rerun to get outlines right\n",
        "(rerunfilecheck)                Rerun to get bibliographical references right.\n",
        "Package longtable Warning: Table \\widths have changed. Rerun LaTeX.\n",
        "Package natbib Warning: Citation(s) may have changed.\n"
        "(natbib)                Rerun to get citations correct.\n",
    ],
)
def test_needs_rerun_detects_rerun_requests(build_module, text):
    assert build_module.needs_rerun(text)


@pytest.mark.parametrize(
    "text",
    [
        RERUNFILECHECK_INFO,
        "[221] (./tables/editorial/tab_cap8_reruns.tex\n",
        "LaTeX Warning: There were undefined references.\n",
        "Output written on tese.pdf (265 pages, 1264084 bytes).\n",
    ],
)
def test_needs_rerun_ignores_non_requests(build_module, text):
    assert not build_module.needs_rerun(text)


def test_needs_rerun_handles_tex_line_wrapping(build_module):
    message = "Package rerunfilecheck Warning: File `tese.brf' has changed. Please, Rerun to get it right."
    split = message.index("Rerun to get") + 3
    filler = "x" * (79 - split)
    wrapped_line = (filler + message)[:79]
    rest = (filler + message)[79:]
    assert len(wrapped_line) == 79 and "Rerun to get" not in wrapped_line
    assert build_module.needs_rerun(wrapped_line + "\n" + rest + "\n")


def test_is_stable_requires_unchanged_aux_files(build_module):
    same = {".aux": "a", ".out": "b"}
    assert build_module.is_stable("clean log\n", same, dict(same))
    assert not build_module.is_stable("clean log\n", same, {".aux": "c", ".out": "b"})
    assert not build_module.is_stable(RERUN_LABELS, same, dict(same))


def test_fingerprint_changes_when_aux_changes(build_module, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "job.aux").write_text("\\relax\n", encoding="latin1")
    first = build_module.fingerprint(work, "job")
    (work / "job.aux").write_text("\\relax\n\\bibcite{k}{1}\n", encoding="latin1")
    assert build_module.fingerprint(work, "job") != first


# ---------------------------------------------------------------------------
# Citacoes indefinidas e outros problemas do log (sem LaTeX)
# ---------------------------------------------------------------------------
def test_analyze_log_detects_undefined_citations_references_duplicates_and_errors(
    build_module,
):
    key = "K" * 38  # empurra "undefined" para a coluna 75, cortado na 79
    citation = f"LaTeX Warning: Citation `{key}' on page 1 undefined on input line 9."
    first, rest = citation[:79], citation[79:]
    assert "undefined" not in first and "undefined" not in rest
    log = (
        "LaTeX Warning: Citation `Fellegi1969theory' on page 1 undefined on input line 7.\n"
        f"{first}\n{rest}\n"
        "LaTeX Warning: Citation `Fellegi1969theory' on page 2 undefined on input line 20.\n"
        "LaTeX Warning: Reference `sec:dados' on page 3 undefined on input line 40.\n"
        "LaTeX Warning: Label `tab:resumo' multiply defined.\n"
        "./capitulo1.tex:12: LaTeX Error: Environment foo undefined.\n"
        "LaTeX Warning: There were undefined references.\n"
        "LaTeX Warning: There were multiply-defined labels.\n"
        "Output written on build/final/tese.pdf (263 pages, 1221264 bytes).\n"
    )
    report = build_module.analyze_log(log)
    assert report.undefined_citations == ["Fellegi1969theory", key]
    assert report.undefined_references == ["sec:dados"]
    assert report.multiply_defined == ["tab:resumo"]
    assert report.errors == ["./capitulo1.tex:12: LaTeX Error: Environment foo undefined."]
    assert report.pages == 263
    assert not report.ok
    assert len(report.problems()) == 4


def test_analyze_log_counts_many_undefined_citations(build_module):
    log = "".join(
        f"LaTeX Warning: Citation `Key{i}' on page {i} undefined on input line 1.\n"
        for i in range(411)
    )
    report = build_module.analyze_log(log)
    assert len(report.undefined_citations) == 411
    assert not report.ok


def test_analyze_log_summary_only_still_fails(build_module):
    report = build_module.analyze_log("LaTeX Warning: There were undefined references.\n")
    assert report.undefined_citations == [] and not report.ok


def test_analyze_log_clean(build_module):
    log = (
        RERUNFILECHECK_INFO
        + "(./tese.bbl (./tese.brf)\n"
        + "Output written on build/final/tese.pdf (265 pages, 1264084 bytes).\n"
    )
    report = build_module.analyze_log(log)
    assert report.ok
    assert report.problems() == []
    assert report.pages == 265


@pytest.mark.parametrize(
    "returncode,blg,bbl_size,expected",
    [
        (0, CLEAN_BLG, 100, 0),
        (1, CLEAN_BLG, 100, 0),  # 1 = apenas avisos (Warning--)
        (2, ERROR_BLG, 100, 3),  # codigo, "I couldn't open", resumo de erros
        (0, None, None, 2),  # bibtex sem efeito: sem .blg e sem .bbl
        (0, CLEAN_BLG, 0, 1),  # .bbl vazio (placeholder nao sobrescrito)
    ],
)
def test_bibtex_problems(build_module, returncode, blg, bbl_size, expected):
    assert len(build_module.bibtex_problems(returncode, blg, bbl_size)) == expected


# ---------------------------------------------------------------------------
# Fluxo do build com bibtex e log final (sem LaTeX)
# ---------------------------------------------------------------------------
def test_build_fails_when_bibtex_has_no_effect(build_module):
    fake = FakeTeX(build_module, bibtex_writes=False, bibtex_rc=0)
    assert run_build(build_module, fake, jobname="check", outname="check") == 1
    assert fake.passes == 1
    assert not (build_module.OUT_DIR / "check.pdf").exists()
    assert not (build_module.SRC_DIR / "check.pdf").exists()


def test_build_fails_when_blg_has_errors(build_module):
    fake = FakeTeX(build_module, blg_text=ERROR_BLG, bibtex_rc=2)
    assert run_build(build_module, fake, jobname="check", outname="check") == 1
    assert fake.passes == 1


def test_build_fails_on_undefined_citations_even_when_stable(build_module):
    extra = (
        "LaTeX Warning: Citation `Fellegi1969theory' on page 1 undefined on input line 9.\n"
        "LaTeX Warning: There were undefined references.\n"
    )
    fake = FakeTeX(build_module, extra_log=extra)
    assert run_build(build_module, fake, jobname="check", outname="check") == 1
    assert fake.passes == 3
    assert not (build_module.OUT_DIR / "check.pdf").exists()


def test_build_runs_bibtex_in_build_dir_with_bibinputs(build_module):
    fake = FakeTeX(build_module)
    assert run_build(build_module, fake, jobname="check", outname="check") == 0
    work = build_module.build_dir()
    (pdflatex_cmd, pdflatex_kwargs) = fake.calls[0]
    assert pdflatex_kwargs["cwd"] == build_module.SRC_DIR
    assert f"-output-directory={build_module.BUILD_SUBDIR}" in pdflatex_cmd
    assert pdflatex_kwargs["env"]["max_print_line"] == "10000"
    bibtex_cmd, bibtex_kwargs = next(c for c in fake.calls if c[0][0] == "bibtex")
    assert bibtex_cmd == ["bibtex", "check"]
    assert bibtex_kwargs["cwd"] == work
    for var in ("BIBINPUTS", "BSTINPUTS"):
        assert bibtex_kwargs["env"][var].startswith(f"{build_module.SRC_DIR}{os.pathsep}")


def test_full_build_starts_from_clean_build_dir(build_module):
    work = build_module.build_dir()
    work.mkdir(parents=True)
    (work / "check.aux").write_text("\\bibcite{Velho}{antigo}\n", encoding="latin1")
    (work / "check.blg").write_text(CLEAN_BLG, encoding="latin1")
    (work / "outro.aux").write_text("\\relax\n", encoding="latin1")
    assert build_module.prepare_clean_build(work, "check") == 0
    assert not (work / "check.blg").exists()
    for ext in build_module.SEED_EXTENSIONS:
        assert (work / f"check{ext}").read_bytes() == b""
    assert (work / "outro.aux").exists()


def test_quick_does_not_touch_final_pdfs(build_module):
    fake = FakeTeX(build_module)
    assert run_build(build_module, fake, quick=True, jobname="check", outname="check") == 0
    assert fake.passes == 1
    assert not any(cmd[0] == "bibtex" for cmd, _ in fake.calls)
    assert (build_module.build_dir() / "check.pdf").exists()
    assert not (build_module.OUT_DIR / "check.pdf").exists()
    assert not (build_module.SRC_DIR / "check.pdf").exists()


# ---------------------------------------------------------------------------
# --check (sem LaTeX)
# ---------------------------------------------------------------------------
def test_check_mode(build_module):
    assert build_module.check(jobname="check") == 1  # sem log
    work = build_module.build_dir()
    work.mkdir(parents=True)
    log = work / "check.log"
    log.write_text("Output written on build/final/check.pdf (3 pages, 1 bytes).\n", encoding="latin1")
    assert build_module.check(jobname="check") == 0
    log.write_text("LaTeX Warning: Label `fig:a' multiply defined.\n", encoding="latin1")
    assert build_module.check(jobname="check") == 1
    log.write_text("! LaTeX Error: File `foo.sty' not found.\n", encoding="latin1")
    assert build_module.check(jobname="check") == 1


# ---------------------------------------------------------------------------
# Integracao com LaTeX real (pulado quando pdflatex/bibtex nao estao no PATH)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("pdflatex", "bibtex", "makeindex")),
    reason="pdflatex/bibtex/makeindex nao instalados",
)
def test_real_build_resolves_citations_and_ignores_stale_files(build_module):
    src = build_module.SRC_DIR
    (src / "tese.tex").write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "Ver \\cite{fellegi} e a secao~\\ref{sec:a}.\n"
        "\\section{A}\\label{sec:a}\n"
        "\\bibliographystyle{plain}\n"
        "\\bibliography{refs}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    (src / "refs.bib").write_text(
        "@article{fellegi, author={Fellegi, Ivan P. and Sunter, Alan B.},\n"
        "  title={A theory for record linkage}, journal={JASA}, year={1969}}\n",
        encoding="utf-8",
    )
    # Restos antigos em text/latex: se forem lidos, o pdflatex para com erro.
    (src / "tese.aux").write_text("\\restoantigoquenaoexiste\n", encoding="latin1")
    (src / "tese.bbl").write_text("\\restoantigoquenaoexiste\n", encoding="latin1")

    assert build_module.build() == 0
    work = build_module.build_dir()
    report = build_module.analyze_log((work / "tese.log").read_text(encoding="latin1"))
    assert report.ok, report.problems()
    assert (work / "tese.blg").exists()
    assert "Fellegi" in (work / "tese.bbl").read_text(encoding="latin1")
    assert (src / "tese.pdf").exists()
    assert (build_module.OUT_DIR / "tese.pdf").exists()
    assert build_module.check() == 0
