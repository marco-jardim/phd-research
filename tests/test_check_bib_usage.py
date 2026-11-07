"""Citation wrapper arguments must not count as missing bibliography entries."""

import subprocess
import sys
from pathlib import Path


def test_citation_wrapper_arguments_are_not_keys(tmp_path):
    latex = tmp_path / "text" / "latex"
    latex.mkdir(parents=True)
    (latex / "referencias.bib").write_text(
        "@article{Example2026, title={Example}, year={2026}}", encoding="utf-8"
    )
    (latex / "tese.tex").write_text(
        r"\providecommand{\citeyearpar}[1]{(\citeyear{#1})}"
        r"\cite{Example2026}\citeonline{Missing2026}",
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_bib_usage.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Total cited keys:   2" in result.stdout
    assert "Phantom (cited in .tex but NOT in .bib): 1" in result.stdout
    assert "  - Missing2026" in result.stdout
    assert "  - #1" not in result.stdout
