"""Check bib references against tex citations."""

import re
import glob
import os

tex_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(tex_dir):
    tex_dir = r"D:\git\phd-research\text\latex"

cite_pat = re.compile(r"\\cite\w*\{([^}]+)\}")

cite_keys = set()
for f in glob.glob(os.path.join(tex_dir, "*.tex")):
    bn = os.path.basename(f)
    if "tese_flat" in bn or "tese_pandoc" in bn or "reserva" in bn:
        continue
    with open(f, encoding="utf-8") as fh:
        content = fh.read()
    for m in cite_pat.finditer(content):
        for k in m.group(1).split(","):
            cite_keys.add(k.strip())

bib_path = os.path.join(tex_dir, "referencias.bib")
with open(bib_path, encoding="utf-8") as f:
    bib_content = f.read()

bib_keys = set(re.findall(r"@\w+\{(\w+)", bib_content))

missing = cite_keys - bib_keys
unused = bib_keys - cite_keys
print(f"Total cited keys: {len(cite_keys)}")
print(f"Total bib entries: {len(bib_keys)}")
print(f"\nMissing from bib: {sorted(missing) if missing else 'None'}")
print(f"\nUnused in bib: {sorted(unused) if unused else 'None'}")
