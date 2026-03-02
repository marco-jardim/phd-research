"""Scan chapter .tex files for bare \\cite{} (not \\citeonline etc.)."""

import re
from pathlib import Path

tex_dir = Path(r"D:\git\phd-research\text\latex")
files = sorted(tex_dir.glob("capitulo*.tex")) + [
    tex_dir / "reserva_cap6.tex",
    tex_dir / "tese.tex",
]

for f in files:
    for i, line in enumerate(f.read_text("utf-8").splitlines(), 1):
        if line.strip().startswith("%"):
            continue
        for m in re.finditer(r"\\cite\{", line):
            pre = line[max(0, m.start() - 10) : m.start()]
            if any(k in pre for k in ("online", "author", "text", "year")):
                continue
            brace_end = line.find("}", m.start())
            if brace_end == -1:
                continue
            key = line[m.start() + 6 : brace_end]
            before = line[max(0, m.start() - 80) : m.start()].strip()
            after = line[brace_end + 1 : brace_end + 21].strip()
            print(f"{f.name}:{i}  {before} ||\\cite{{{key}}}|| {after}")
