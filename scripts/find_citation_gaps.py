#!/usr/bin/env python3
"""Find long LaTeX paragraphs without \cite commands.

This is a lightweight helper to support "citation density" review.

Heuristic:
- Split on blank lines (LaTeX paragraph breaks).
- Ignore blocks that are sectioning commands or environment plumbing.
- Report blocks that exceed a configurable minimum length and contain no \cite*.

Usage:
    python scripts/find_citation_gaps.py
    python scripts/find_citation_gaps.py --min-chars 300
    python scripts/find_citation_gaps.py --include-items
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TEX_DIR = Path("text/latex")


SECTION_RE = re.compile(r"^\\(section|subsection|subsubsection)\*?\{(.+?)\}\s*$")
CITE_RE = re.compile(
    r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]+)\}", re.MULTILINE
)


def collapse_ws(s: str, *, limit: int = 140) -> str:
    s2 = re.sub(r"\s+", " ", s).strip()
    if len(s2) <= limit:
        return s2
    return s2[: limit - 1] + "…"


def should_ignore_block(text: str, *, include_items: bool) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return True
    if stripped.startswith("%"):
        return True
    if stripped.startswith("\\begin{") or stripped.startswith("\\end{"):
        return True
    if stripped.startswith("\\caption") or stripped.startswith("\\label"):
        return True
    if stripped.startswith("\\includegraphics"):
        return True
    if stripped.startswith("\\centering"):
        return True
    if SECTION_RE.match(stripped):
        return True
    if stripped.startswith("\\item") and not include_items:
        return True
    return False


def iter_blocks(lines: list[str]):
    sec = ""
    sub = ""
    subsub = ""

    buf: list[str] = []
    buf_start = 1
    buf_sec = ""
    buf_sub = ""
    buf_subsub = ""

    def flush(end_line: int):
        nonlocal buf
        if not buf:
            return
        yield {
            "start": buf_start,
            "end": end_line,
            "section": buf_sec,
            "subsection": buf_sub,
            "subsubsection": buf_subsub,
            "text": "\n".join(buf).strip(),
        }
        buf = []

    for i, line in enumerate(lines, start=1):
        m = SECTION_RE.match(line.strip())
        if m:
            kind, title = m.group(1), m.group(2)
            if kind == "section":
                sec, sub, subsub = title, "", ""
            elif kind == "subsection":
                sub, subsub = title, ""
            else:
                subsub = title

        if not line.strip():
            yield from flush(i - 1)
            continue

        if not buf:
            buf_start = i
            buf_sec, buf_sub, buf_subsub = sec, sub, subsub
        buf.append(line)

    yield from flush(len(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List LaTeX blocks without citations (heuristic)."
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=220,
        help="Minimum non-whitespace chars in block to be reported.",
    )
    parser.add_argument(
        "--include-items",
        action="store_true",
        help="Also include blocks starting with \\item.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[
            "capitulo4.tex",
            "capitulo5.tex",
            "capitulo7.tex",
            "reserva_cap6.tex",
            "capitulo8.tex",
        ],
        help="Basenames under text/latex to scan.",
    )
    args = parser.parse_args()

    any_hit = False
    for name in args.files:
        path = TEX_DIR / name
        if not path.exists():
            print(f"[SKIP] Missing: {path}")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()

        hits = []
        for b in iter_blocks(lines):
            text = b["text"]
            if should_ignore_block(text, include_items=args.include_items):
                continue
            if CITE_RE.search(text):
                continue
            compact_len = len(re.sub(r"\s+", "", text))
            if compact_len < args.min_chars:
                continue
            hits.append((b["start"], compact_len, b))

        if not hits:
            continue
        any_hit = True
        print(f"\n== {name} ==")
        for start, clen, b in hits:
            where = " / ".join(
                [p for p in (b["section"], b["subsection"], b["subsubsection"]) if p]
            )
            if where:
                where = f"  ({where})"
            print(f"- L{start}  chars~{clen}{where}")
            print(f"  {collapse_ws(b['text'])}")

    if not any_hit:
        print("[OK] No blocks found (within current heuristic/thresholds).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
