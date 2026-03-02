"""Check which .bib entries are cited in .tex files and vice-versa."""

import re
from pathlib import Path

bib_path = Path("text/latex/referencias.bib")
tex_dir = Path("text/latex")

# Parse bib keys
bib_text = bib_path.read_text(encoding="utf-8")
bib_keys = set(re.findall(r"@\w+\{([^,\s]+)", bib_text))

# Parse cited keys from all .tex
all_tex = "\n".join(p.read_text(encoding="utf-8") for p in tex_dir.glob("*.tex"))
cite_pattern = re.compile(r"\\cite\w*\{([^}]+)\}")
cited_keys: set[str] = set()
for match in cite_pattern.findall(all_tex):
    for key in match.split(","):
        cited_keys.add(key.strip())

orphan = sorted(bib_keys - cited_keys)
phantom = sorted(cited_keys - bib_keys)

print(f"Total .bib entries: {len(bib_keys)}")
print(f"Total cited keys:   {len(cited_keys)}")
print(f"\nOrphan (in .bib but NOT cited in any .tex): {len(orphan)}")
for k in orphan:
    print(f"  - {k}")
print(f"\nPhantom (cited in .tex but NOT in .bib): {len(phantom)}")
for k in phantom:
    print(f"  - {k}")
