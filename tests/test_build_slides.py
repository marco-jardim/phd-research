"""Contrato de compilação e critérios editoriais dos slides de defesa."""

from pathlib import Path
import re
import runpy
import subprocess
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / "text" / "slides"


def load_builder():
    return runpy.run_path(str(SLIDES / "build_slides.py"))["build"]


def test_build_requires_pdflatex(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="pdflatex não encontrado"):
        load_builder()()


def test_build_runs_two_passes(monkeypatch, tmp_path):
    namespace = runpy.run_path(str(SLIDES / "build_slides.py"))
    build = namespace["build"]
    build.__globals__["__file__"] = str(tmp_path / "build_slides.py")
    (tmp_path / "defesa.pdf").write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr("shutil.which", lambda _: "pdflatex")
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)
    assert build() == tmp_path / "defesa.pdf"
    assert run.call_count == 2
    for call in run.call_args_list:
        assert call.kwargs["cwd"] == tmp_path
        assert call.kwargs["check"] is True
        assert call.kwargs["timeout"] == 600
        assert isinstance(call.kwargs["env"], dict)
        assert call.args[0][-1] == "defesa.tex"


def test_build_propagates_compilation_failure(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "pdflatex")
    run = Mock(side_effect=subprocess.CalledProcessError(1, "pdflatex"))
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        load_builder()()
    assert run.call_count == 1


def test_slide_structure_and_editorial_contract():
    source = (SLIDES / "defesa.tex").read_text(encoding="utf-8")
    main, backup = source.split(r"\appendix")
    assert main.count(r"\begin{frame}") == 30
    backup_frames = re.findall(r"\\begin\{frame\}(.*?)\\end\{frame\}", backup, re.S)
    assert backup_frames
    assert len(backup_frames) == backup.count(r"\begin{frame}")
    for frame in backup_frames:
        title = re.match(r"\{Backup:\s*([^{}]+)\}", frame)
        assert title and title.group(1).strip()
        assert re.search(r"\\fonte(?:\[[^\]]*\])?\{[^}]+\}", frame)
    frames = re.findall(r"\\begin\{frame\}(.*?)\\end\{frame\}", source, re.S)
    assert sum("Fellegi" in frame for frame in frames) == 1
    assert source.index("Limitações: referência") < source.index("Conclusões:")
    assert "211/247 = 85,4" in source
    assert "Fluxo Modal & XGBoost & 0,943" in source
    assert any(
        "base inteira (comparação)" in frame
        and "REFREC" in frame
        and "tab:cap8-cascata-v31" in frame
        for frame in backup_frames
    )
    for path in [SLIDES / "defesa.tex", SLIDES / "roteiro_defesa.md"]:
        text = path.read_text(encoding="utf-8")
        assert not re.search("zona (?:cinza|cinzenta)|[\u2013\u2014]", text, re.I)


def test_script_has_eleven_short_answers():
    source = (SLIDES / "roteiro_defesa.md").read_text(encoding="utf-8")
    questions = source.split("## Perguntas prováveis")[1].split(
        "## Figuras e critérios"
    )[0]
    answers = re.findall(r"### \d+\..*?\n\n(.*?)\n\nFonte:", questions, re.S)
    assert len(answers) == 11
    assert all(len(answer.split()) <= 70 for answer in answers)
