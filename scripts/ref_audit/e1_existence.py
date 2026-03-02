"""E1 — Existence Layer: verify each BibEntry exists in external databases."""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from .api_clients import (
    crossref_get_by_doi,
    crossref_search_title,
    openalex_get_by_doi,
    openalex_search_title,
    semantic_scholar_get_by_doi,
)
from .models import BibEntry, L1Result, L1Score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metadata comparison helpers
# ---------------------------------------------------------------------------


_TEX_MACRO_RE = re.compile(r"\\[a-zA-Z]+\s*\{([^}]*)\}|\\([`'^~\"c])\{?([a-zA-Z])\}?")


def _normalize(text: str) -> str:
    """Lowercase, strip braces, collapse TeX macros, normalise unicode."""
    # Remove \textit{...}, \emph{...} etc. keeping inner text
    text = _TEX_MACRO_RE.sub(lambda m: m.group(1) or m.group(3) or "", text)
    text = text.replace("{", "").replace("}", "").replace("~", " ")
    # Normalise accented chars to their base form for comparison
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.lower().split())


def _title_match(
    bib_title: str | None, api_title: str | None, threshold: float = 85.0
) -> float:
    """Return fuzzy similarity score for titles.

    Uses the *max* of token_sort_ratio and token_set_ratio so that
    abbreviated API titles (e.g. missing subtitle) still score high.
    """
    if not bib_title or not api_title:
        return 0.0
    a = _normalize(bib_title)
    b = _normalize(api_title)
    return max(
        fuzz.token_sort_ratio(a, b),
        fuzz.token_set_ratio(a, b),
    )


def _year_match(bib_year: int | None, api_year: int | None) -> bool:
    if bib_year is None or api_year is None:
        return True  # can't verify, don't penalize
    return abs(bib_year - api_year) <= 1


def _extract_surname(author_str: str) -> str:
    """Extract surname from either 'Surname, Given' or 'Given Surname' format."""
    author_str = author_str.strip()
    if "," in author_str:
        # "Surname, Given …" → take text before comma
        return author_str.split(",")[0].strip().lower()
    # "Given [Middle] Surname" or compound "Given Diez Roux" → last token
    parts = author_str.split()
    if not parts:
        return ""
    return parts[-1].lower()


def _author_surname_match(
    bib_authors: list[str], api_authors: list[str], threshold: float = 90.0
) -> bool:
    """Check if at least one bib author surname matches an API author."""
    if not bib_authors or not api_authors:
        return True  # can't verify
    bib_surnames = {_extract_surname(a) for a in bib_authors if a.strip()}
    bib_surnames -= {"others", ""}
    if not bib_surnames:
        return True
    api_surnames: set[str] = set()
    for a in api_authors:
        parts = a.replace(",", " ").split()
        if parts:
            api_surnames.add(parts[0].lower())
            if len(parts) > 1:
                api_surnames.add(parts[-1].lower())
    for bs in bib_surnames:
        for aps in api_surnames:
            if fuzz.ratio(_normalize(bs), _normalize(aps)) >= threshold:
                return True
    return False


def _extract_cr_authors(cr_data: dict) -> list[str]:
    """Extract author names from CrossRef data."""
    authors_raw = cr_data.get("author", [])
    return [f"{a.get('family', '')}, {a.get('given', '')}" for a in authors_raw]


def _extract_cr_year(cr_data: dict) -> int | None:
    """Extract publication year from CrossRef data."""
    dp = (
        cr_data.get("published-print")
        or cr_data.get("published-online")
        or cr_data.get("created")
    )
    if dp:
        parts = dp.get("date-parts", [[None]])
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


def _extract_cr_title(cr_data: dict) -> str:
    titles = cr_data.get("title", [])
    return titles[0] if titles else ""


def _extract_cr_journal(cr_data: dict) -> str:
    names = cr_data.get("container-title", [])
    return names[0] if names else ""


# ---------------------------------------------------------------------------
# E1 main logic
# ---------------------------------------------------------------------------


def verify_entry(entry: BibEntry, cache_dir: Path) -> L1Result:
    """Run E1 verification for a single BibEntry."""
    cr_cache = cache_dir / "crossref"
    cr_cache.mkdir(parents=True, exist_ok=True)

    cr_data: dict | None = None
    cr_title = ""
    cr_authors: list[str] = []
    cr_year: int | None = None
    cr_journal = ""
    cr_type = ""
    citation_count: int | None = None
    resolved_doi = entry.doi
    title_similarity = 0.0
    year_ok = False
    author_ok = False

    # --- Step 1: Try resolving by DOI ---
    if entry.doi:
        cr_data = crossref_get_by_doi(entry.doi, cr_cache)
        if cr_data:
            cr_title = _extract_cr_title(cr_data)
            cr_authors = _extract_cr_authors(cr_data)
            cr_year = _extract_cr_year(cr_data)
            cr_journal = _extract_cr_journal(cr_data)
            cr_type = cr_data.get("type", "")
            citation_count = cr_data.get("is-referenced-by-count")

    # --- Step 2: No DOI or DOI failed -> search by title ---
    if cr_data is None and entry.title:
        time.sleep(0.3)  # politeness
        # Try OpenAlex first (faster, more permissive)
        oa_results = openalex_search_title(entry.title, cr_cache)
        if oa_results:
            for oa in oa_results:
                oa_title = oa.get("title", "")
                sim = _title_match(entry.title or "", oa_title)
                if sim >= 90:
                    # Found via OpenAlex, try to get DOI
                    oa_doi = oa.get("doi", "")
                    if oa_doi:
                        oa_doi = oa_doi.replace("https://doi.org/", "")
                        resolved_doi = oa_doi
                        # Now fetch CrossRef with discovered DOI
                        cr_data = crossref_get_by_doi(oa_doi, cr_cache)
                        if cr_data:
                            cr_title = _extract_cr_title(cr_data)
                            cr_authors = _extract_cr_authors(cr_data)
                            cr_year = _extract_cr_year(cr_data)
                            cr_journal = _extract_cr_journal(cr_data)
                            cr_type = cr_data.get("type", "")
                            citation_count = cr_data.get("is-referenced-by-count")
                        else:
                            # CrossRef may not cover some registrars (e.g., DataCite/arXiv).
                            # Fall back to OpenAlex metadata for existence verification.
                            cr_title = oa_title
                            cr_year = oa.get("publication_year")
                            citation_count = oa.get("cited_by_count")
                            authships = oa.get("authorships", [])
                            cr_authors = [
                                a.get("author", {}).get("display_name", "")
                                for a in authships
                            ]
                            loc = oa.get("primary_location") or {}
                            src = loc.get("source") or {}
                            cr_journal = src.get("display_name", "")
                            cr_data = oa  # mark as found
                    else:
                        # Use OpenAlex data directly
                        cr_title = oa_title
                        cr_year = oa.get("publication_year")
                        citation_count = oa.get("cited_by_count")
                        authships = oa.get("authorships", [])
                        cr_authors = [
                            a.get("author", {}).get("display_name", "")
                            for a in authships
                        ]
                        loc = oa.get("primary_location") or {}
                        src = loc.get("source") or {}
                        cr_journal = src.get("display_name", "")
                        cr_data = oa  # mark as found
                    break

        # Fallback: CrossRef title search
        if cr_data is None:
            time.sleep(0.3)
            cr_results = crossref_search_title(entry.title, cr_cache)
            for item in cr_results:
                item_title = _extract_cr_title(item)
                sim = _title_match(entry.title or "", item_title)
                if sim >= 90:
                    cr_data = item
                    cr_title = item_title
                    cr_authors = _extract_cr_authors(item)
                    cr_year = _extract_cr_year(item)
                    cr_journal = _extract_cr_journal(item)
                    cr_type = item.get("type", "")
                    citation_count = item.get("is-referenced-by-count")
                    resolved_doi = item.get("DOI", resolved_doi)
                    break

    # --- Step 3: Metadata verification ---
    if cr_data is not None:
        title_similarity = _title_match(entry.title or "", cr_title)
        year_ok = _year_match(entry.year, cr_year)
        author_ok = _author_surname_match(entry.authors, cr_authors)
    else:
        title_similarity = 0.0
        year_ok = False
        author_ok = False

    # --- Step 4: Score ---
    score = _compute_l1_score(
        entry=entry,
        found=(cr_data is not None),
        title_sim=title_similarity,
        year_ok=year_ok,
        author_ok=author_ok,
    )

    return L1Result(
        bib_key=entry.bib_key,
        score=score,
        resolved_doi=resolved_doi,
        cr_title=cr_title,
        cr_authors=cr_authors,
        cr_year=cr_year,
        cr_journal=cr_journal,
        cr_type=cr_type,
        citation_count=citation_count,
        title_similarity=title_similarity,
        year_match=year_ok,
        author_match=author_ok,
        is_retracted=False,  # E1.4 retraction check placeholder
    )


def _compute_l1_score(
    entry: BibEntry,
    found: bool,
    title_sim: float,
    year_ok: bool,
    author_ok: bool,
) -> L1Score:
    from .models import SourceType

    # Source types not expected to be in CrossRef/OpenAlex
    _NON_INDEXED_TYPES = {
        SourceType.WEB_RESOURCE,
        SourceType.THESIS,
        SourceType.TECHREPORT,
    }

    if not found:
        # For non-indexed source types without DOI, don't treat as ghost
        if entry.source_type in _NON_INDEXED_TYPES and not entry.doi:
            return L1Score.L1_WARN_NO_DOI
        if entry.doi:
            return L1Score.L1_FAIL_GHOST
        return L1Score.L1_WARN_NO_DOI

    if title_sim >= 85 and year_ok and author_ok:
        return L1Score.L1_PASS

    issues = []
    if title_sim < 85:
        issues.append("title")
    if not year_ok:
        issues.append("year")
    if not author_ok:
        issues.append("author")

    # For non-indexed types, be lenient — metadata may be incomplete
    if entry.source_type in _NON_INDEXED_TYPES and len(issues) >= 2:
        return L1Score.L1_WARN_META
    if len(issues) >= 2:
        return L1Score.L1_FAIL_GHOST  # too many mismatches
    return L1Score.L1_WARN_META


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_e1(
    entries: list[BibEntry],
    cache_dir: Path,
    output_dir: Path,
    delay: float = 0.5,
) -> list[L1Result]:
    """Run E1 for all entries. Saves results to l1_results.json."""
    results: list[L1Result] = []
    total = len(entries)

    for i, entry in enumerate(entries):
        logger.info("[E1 %d/%d] %s", i + 1, total, entry.bib_key)
        try:
            r = verify_entry(entry, cache_dir)
        except Exception as exc:
            logger.error("E1 failed for %s: %s", entry.bib_key, exc)
            r = L1Result(
                bib_key=entry.bib_key,
                score=L1Score.L1_WARN_NO_DOI,
                resolved_doi=entry.doi,
                cr_title="",
                cr_authors=[],
                cr_year=None,
                cr_journal="",
                cr_type="",
                citation_count=None,
                title_similarity=0.0,
                year_match=False,
                author_match=False,
                is_retracted=False,
            )
        results.append(r)
        if i < total - 1:
            time.sleep(delay)

    # Save
    out_path = output_dir / "l1_results.json"
    out_path.write_text(
        json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("E1 results saved to %s", out_path)

    # Summary
    from collections import Counter

    counts = Counter(r.score for r in results)
    for score, count in sorted(counts.items(), key=lambda x: x[0].value):
        logger.info("  %s: %d", score.value, count)

    return results
