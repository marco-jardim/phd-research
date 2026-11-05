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
    assert len(backup_frames) == 16
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
        assert "[ATUALIZAR" not in text
        assert "CONFERIR" not in text


def test_script_has_twenty_two_short_answers():
    source = (SLIDES / "roteiro_defesa.md").read_text(encoding="utf-8")
    questions = source.split("## Perguntas prováveis")[1].split(
        "## Figuras e critérios"
    )[0]
    answers = re.findall(r"### \d+\..*?\n\n(.*?)\n\nFonte:", questions, re.S)
    assert len(answers) == 22
    assert [int(n) for n in re.findall(r"^### (\d+)\.", questions, re.M)] == list(
        range(1, 23)
    )
    assert all(len(answer.split()) <= 70 for answer in answers)


@pytest.mark.parametrize(
    ("frame_number", "required"),
    [
        (3, ("2.156", "2.522", "hipótese de revisão", "teto teórico")),
        (4, ("Hipótese condicionada à carga", "não aceitos pelo corte", "duas fronteiras")),
        (19, ("211+4", "211+8", "+19 e +1 vínculos ao humano")),
        (
            20,
            (
                "Vigilância & 4 & 10 & 19 & 17 & 13 & 9",
                "Confirmação & 8 & 8 & 1 & 11 & 27 & 17",
                "13 e 27",
                "36,1",
                "75,0",
                "1/6/6",
                "1/22/4",
            ),
        ),
        (21, ("4/25", "16,0", "6,4 a 34,7", "9,06", "impede interpretação prospectiva")),
        (
            22,
            (
                "foi dominado por P-Platt",
                "análise de sensibilidade",
                "não confirmatória",
                "dominância não sustentada",
                "7/20 e 4/20",
                "7/20 e 5/20",
                "Preservar o corte",
                "validação em dados novos",
            ),
        ),
        (
            24,
            (
                "Disparidade herdada do corte e do rótulo",
                "37,5",
                "25",
                "42,5 contra 45,0",
                "65 a 67",
            ),
        ),
        (
            26,
            (
                "Respostas históricas reutilizadas",
                "conhecido antes do registro",
                "recalibração, braços e avaliação",
                "desvios de protocolo",
                "não generalização",
            ),
        ),
        (28, ("hipótese é condicionada à carga", "2.156", "13", "27")),
        (
            46,
            (
                "não confirmatória",
                "não generalização",
                "7/20 & 4/20",
                "7/20 & 5/20",
                "0/20 & 0/20",
                "19/20",
                "0,43",
                "9 e 21",
                "0,65",
                "0,80",
            ),
        ),
    ],
)
def test_slide_s5_scientific_contract(frame_number, required):
    source = (SLIDES / "defesa.tex").read_text(encoding="utf-8")
    frames = re.findall(r"\\begin\{frame\}(.*?)\\end\{frame\}", source, re.S)
    frame = frames[frame_number - 1]
    for fragment in required:
        assert fragment in frame


def test_script_s5_questions_and_provenance():
    source = (SLIDES / "roteiro_defesa.md").read_text(encoding="utf-8")
    questions = source.split("## Perguntas prováveis")[1].split(
        "## Figuras e critérios"
    )[0]
    for topic in [
        "carga comparável",
        "VPP de 16%",
        "vínculos perdidos",
        "perder 75%",
        "taxas congeladas",
        "viés de incorporação",
        "somente com o Zenodo",
        "desvios de protocolo",
        "robustez confirma",
    ]:
        assert topic in questions
    assert "recalibração, braços e avaliação" in questions
    assert "arranjo_corte_metricas.csv:2,12" in questions
    assert "árvore compilada" in source
    assert re.search(r"SHA-256[^\n]*`[0-9a-f]{64}`", source)
