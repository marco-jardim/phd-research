"""E0.1 — Parser de arquivo .bib → JSON normalizado.

Lê o .bib com bibtexparser, normaliza campos, classifica tipo de fonte,
e detecta entradas incompletas.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import bibtexparser

from .models import BibEntry, SourceType


def _normalize_doi(raw: str | None) -> str | None:
    """Extrai e normaliza DOI para formato canônico '10.xxxx/yyyy'."""
    if not raw:
        return None
    raw = raw.strip()
    # Remove URL prefix variants
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ):
        if raw.lower().startswith(prefix):
            raw = raw[len(prefix) :]
            break
    # Must start with 10.
    if raw.startswith("10."):
        return raw.strip()
    return None


def _normalize_title(raw: str | None) -> str | None:
    """Remove chaves LaTeX, normaliza espaços."""
    if not raw:
        return None
    t = raw.replace("{", "").replace("}", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _parse_authors(raw: str | None) -> list[str]:
    """Separa autores por 'and', normaliza."""
    if not raw:
        return []
    authors = re.split(r"\s+and\s+", raw, flags=re.IGNORECASE)
    return [a.strip().replace("{", "").replace("}", "") for a in authors if a.strip()]


def _classify_source(entry_type: str, fields: dict) -> SourceType:
    """Classifica tipo de fonte baseado no tipo BibTeX e campos."""
    et = entry_type.lower()
    journal = fields.get("journal", "")

    if et == "article" and journal:
        jl = journal.lower()
        if "arxiv" in jl or "preprint" in jl or "biorxiv" in jl or "medrxiv" in jl:
            return SourceType.PREPRINT
        return SourceType.PEER_REVIEWED_ARTICLE
    if et in ("inproceedings", "conference"):
        return SourceType.CONFERENCE_PAPER
    if et in ("book", "incollection", "inbook"):
        return SourceType.BOOK_OR_CHAPTER
    if et in ("phdthesis", "mastersthesis"):
        return SourceType.THESIS
    if et in ("techreport", "manual"):
        return SourceType.TECHREPORT
    if et == "misc":
        url = fields.get("url", "")
        if url and not fields.get("journal"):
            return SourceType.WEB_RESOURCE
        return SourceType.PREPRINT  # misc sem journal = provavelmente preprint
    return SourceType.WEB_RESOURCE


def _detect_incomplete(entry: BibEntry) -> list[str]:
    """Detecta campos ausentes que deveriam estar presentes."""
    issues: list[str] = []
    if not entry.title:
        issues.append("MISSING_TITLE")
    if not entry.authors:
        issues.append("MISSING_AUTHORS")
    if not entry.year:
        issues.append("MISSING_YEAR")
    if entry.source_type == SourceType.PEER_REVIEWED_ARTICLE:
        if not entry.doi:
            issues.append("MISSING_DOI")
        if not entry.journal:
            issues.append("MISSING_JOURNAL")
    if entry.source_type == SourceType.CONFERENCE_PAPER and not entry.doi:
        issues.append("MISSING_DOI")
    return issues


def parse_bib(bib_path: Path) -> list[BibEntry]:
    """Lê .bib e retorna lista de BibEntry normalizadas.

    Compatível com bibtexparser v1.x (API: load + BibTexParser).
    """
    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    with open(bib_path, encoding="utf-8") as f:
        bib_db = bibtexparser.load(f, parser=parser)

    entries: list[BibEntry] = []
    for item in bib_db.entries:
        # bibtexparser v1: each item is a plain dict
        entry_type = item.get("ENTRYTYPE", "misc")
        bib_key = item.get("ID", "unknown")

        doi = _normalize_doi(item.get("doi"))
        title = _normalize_title(item.get("title"))
        authors = _parse_authors(item.get("author"))
        year_raw = item.get("year", "")
        year = (
            int(re.sub(r"[^\d]", "", year_raw)[:4])
            if re.search(r"\d{4}", year_raw)
            else None
        )
        journal = _normalize_title(item.get("journal") or item.get("booktitle"))
        issn = item.get("issn")
        url = item.get("url")
        publisher = item.get("publisher")

        source_type = _classify_source(entry_type, item)

        entry = BibEntry(
            bib_key=bib_key,
            entry_type=entry_type,
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            journal=journal,
            issn=issn,
            url=url,
            publisher=publisher,
            source_type=source_type,
        )
        entry.completeness_issues = _detect_incomplete(entry)
        entries.append(entry)

    return entries


def save_normalized(entries: list[BibEntry], output_path: Path) -> None:
    """Salva entradas normalizadas como JSON."""
    data = [e.to_dict() for e in entries]
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_e0_parse(bib_path: Path, output_dir: Path) -> list[BibEntry]:
    """Executa E0.1 completo."""
    entries = parse_bib(bib_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_normalized(entries, output_dir / "refs_normalized.json")
    return entries
