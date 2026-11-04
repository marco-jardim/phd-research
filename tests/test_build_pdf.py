"""Checks for bounded LaTeX cross-reference stabilization."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


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


@pytest.mark.parametrize("stable_after,expected_rc,expected_passes", [(3, 0, 3), (4, 0, 4), (5, 0, 5), (6, 1, 5)])
def test_build_waits_for_stable_references(
    build_module, stable_after, expected_rc, expected_passes
):
    passes = 0

    def fake_run(cmd, **kwargs):
        nonlocal passes
        if cmd[0] == "pdflatex":
            passes += 1
            log = "Package: rerunfilecheck Rerun checks for auxiliary files\n"
            if passes < stable_after:
                log += "LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.\n"
            (build_module.SRC_DIR / "check.log").write_text(log, encoding="latin1")
            (build_module.SRC_DIR / "check.pdf").write_bytes(b"test pdf")
        return 0

    with patch.object(build_module, "check_toolchain", return_value=True), patch.object(
        build_module, "run", side_effect=fake_run
    ):
        assert build_module.build(jobname="check", outname="check") == expected_rc
    assert passes == expected_passes
    assert (build_module.OUT_DIR / "check.pdf").exists() == (expected_rc == 0)


def test_build_stops_on_extra_pass_failure(build_module):
    passes = 0

    def fake_run(cmd, **kwargs):
        nonlocal passes
        if cmd[0] == "pdflatex":
            passes += 1
            (build_module.SRC_DIR / "check.log").write_text(
                "Rerun to get cross-references right.", encoding="latin1"
            )
            return 1 if passes == 4 else 0
        return 0

    with patch.object(build_module, "check_toolchain", return_value=True), patch.object(
        build_module, "run", side_effect=fake_run
    ):
        assert build_module.build(jobname="check", outname="check") == 1
    assert passes == 4
    assert not (build_module.OUT_DIR / "check.pdf").exists()
