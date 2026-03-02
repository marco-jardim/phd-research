import re
from datetime import date

BIB_FILE = "text/latex/referencias.bib"
OUT_FILE = "text/latex/referencias_abnt.bib"


def fix_bib():
    with open(BIB_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Simple regex-based parser/fixer since bibtexparser might not be installed
    # This is a heuristic approach.

    entries = content.split("@")
    new_entries = []

    for entry in entries:
        if not entry.strip():
            continue

        # Identify type
        etype_match = re.match(r"^(\w+)\s*{", entry)
        if not etype_match:
            new_entries.append(entry)
            continue

        etype = etype_match.group(1).lower()

        # Check address
        if etype in ["book", "mastersthesis", "phdthesis"]:
            if not re.search(r"\baddress\s*=", entry, re.IGNORECASE):
                # Try to infer or add S.l.
                if "Springer" in entry:
                    addr = "Berlin"
                elif "O'Reilly" in entry:
                    addr = "Sebastopol"
                elif "Wiley" in entry:
                    addr = "Hoboken"
                elif "UFRJ" in entry or "Rio de Janeiro" in entry:
                    addr = "Rio de Janeiro"
                else:
                    addr = "S.l."

                # Insert address field before the closing brace
                entry = entry.rstrip().rstrip("}") + f",\n  address = {{{addr}}}\n}}"

        # Check urlaccessdate for misc/url
        if "url" in entry and not re.search(
            r"\burlaccessdate\s*=", entry, re.IGNORECASE
        ):
            today_str = date.today().strftime("%d %b. %Y").lower()
            # Insert urlaccessdate
            entry = (
                entry.rstrip().rstrip("}") + f",\n  urlaccessdate = {{{today_str}}}\n}}"
            )

        new_entries.append(entry)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("@" + "@".join(new_entries))

    print(f"Fixed bib saved to {OUT_FILE}")


if __name__ == "__main__":
    fix_bib()
