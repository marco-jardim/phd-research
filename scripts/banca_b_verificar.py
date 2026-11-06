"""Verifica os capitulos da fatia B com uma compilacao e jobname separado."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    latex = ROOT / "text/latex"
    targets = [f"estudo{n}_{suffix}.tex" for n, suffix in
               [(1, "comparador"), (2, "classificadores"), (3, "estrategias")]]
    tables = sorted((latex / "tables/editorial").glob("tab_nb0*.tex"))
    tables += [latex / "tables/editorial" / name for name in
               ["tab_pareto_frontier.tex", "tab_impacto_epidemiologico.tex", "tab_qw1_deep_learning.tex"]]
    scan = subprocess.run(
        ["rg", chr(0x2014) + "|" + chr(0x2013), *[str(latex / p) for p in targets],
         *[str(p) for p in tables]], capture_output=True, text=True, encoding="utf-8"
    )
    print(f"rg travessoes: exit={scan.returncode}, matches={scan.stdout.strip()!r}")
    if scan.returncode != 1:
        raise RuntimeError(scan.stderr or scan.stdout or "Falha no rg")
    build = ROOT / ".local/banca_b_build"
    build.mkdir(parents=True, exist_ok=True)
    preamble = re.split(r"(?m)^\\begin\{document\}", (latex / "tese.tex").read_text(encoding="utf-8"), maxsplit=1)[0]
    wrapper = build / "banca_b_check.tex"
    wrapper.write_text(preamble + "\n" + r"\begin{document}" + "\n" +
                       "\n".join(r"\input{" + name + "}" for name in targets) +
                       "\n" + r"\end{document}" + "\n", encoding="utf-8")
    command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
               "-jobname=banca_b_check", f"-output-directory={build}", str(wrapper)]
    result = subprocess.run(command, cwd=latex, capture_output=True)
    (build / "stdout.log").write_bytes(result.stdout + result.stderr)
    log = (build / "banca_b_check.log").read_text(encoding="utf-8", errors="replace")
    errors = [line for line in log.splitlines() if line.startswith("!") or "LaTeX Error:" in line]
    summary = {"jobname": "banca_b_check", "exit_code": result.returncode, "errors": errors,
               "pdf_exists": (build / "banca_b_check.pdf").exists(),
               "log": ".local/banca_b_build/banca_b_check.log",
               "scope": targets, "passes": 1, "rg_exit_code": scan.returncode}
    report = ROOT / "docs/reports/BANCA_B_verificacao.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if result.returncode:
        print(log[-7000:])
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
