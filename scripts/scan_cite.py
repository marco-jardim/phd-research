"""Scan for remaining bare \\cite where author is grammatical subject."""

import re
from pathlib import Path

d = Path("D:/git/phd-research/text/latex")
files = sorted(d.glob("capitulo*.tex")) + [d / "reserva_cap6.tex", d / "tese.tex"]

# Pattern 1: Capitalized word right before \cite{
pat1 = re.compile(r"([A-Z][a-záéíóúãõêâô]+)\s*\\cite\{")
# Pattern 2: colaboradores / et al. before \cite{
pat2 = re.compile(r"(colaboradores|et al\.)\s*\\cite\{")

hits = []
for f in files:
    lines = f.read_text("utf-8").splitlines()
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("%"):
            continue
        for pat in [pat1, pat2]:
            for m in pat.finditer(line):
                pre10 = line[max(0, m.start() - 10) : m.start()]
                if "online" in pre10:
                    continue
                bef = line[max(0, m.start() - 60) : m.end()].strip()
                hits.append(f"{f.name}:{i}  ...{bef}")

for h in sorted(set(hits)):
    print(h)
print(f"\nTotal: {len(set(hits))}")
