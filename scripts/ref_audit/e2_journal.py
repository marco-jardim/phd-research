"""E2 — Journal quality assessment.

Evaluates journal quality using:
- CrossRef metadata (already cached from E1)
- Scimago SJR rankings (via CSV or API)
- Evidence level classification
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from rapidfuzz import fuzz

from .models import (
    BibEntry,
    EvidenceLevel,
    L1Result,
    L2Result,
    L2Score,
    SourceType,
)
from . import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scimago data (embedded top journals for health + CS + statistics)
# In production, download full CSV from scimagojr.com
# For now, we use a curated lookup + CrossRef metadata
# ---------------------------------------------------------------------------

# Curated Q1-Q2 journals relevant to this thesis domain
KNOWN_JOURNALS: dict[str, dict] = {
    # Health / Epidemiology / Public Health
    "the lancet": {"sjr_quartile": "Q1", "sjr_score": 10.0, "area": "Medicine"},
    "the new england journal of medicine": {
        "sjr_quartile": "Q1",
        "sjr_score": 15.0,
        "area": "Medicine",
    },
    "bmj": {"sjr_quartile": "Q1", "sjr_score": 3.0, "area": "Medicine"},
    "bulletin of the world health organization": {
        "sjr_quartile": "Q1",
        "sjr_score": 2.5,
        "area": "Public Health",
    },
    "international journal of epidemiology": {
        "sjr_quartile": "Q1",
        "sjr_score": 3.5,
        "area": "Epidemiology",
    },
    "american journal of epidemiology": {
        "sjr_quartile": "Q1",
        "sjr_score": 2.0,
        "area": "Epidemiology",
    },
    "journal of clinical epidemiology": {
        "sjr_quartile": "Q1",
        "sjr_score": 2.5,
        "area": "Epidemiology",
    },
    "cadernos de saúde pública": {
        "sjr_quartile": "Q2",
        "sjr_score": 0.6,
        "area": "Public Health",
    },
    "cadernos de saude publica": {
        "sjr_quartile": "Q2",
        "sjr_score": 0.6,
        "area": "Public Health",
    },
    "revista de saúde pública": {
        "sjr_quartile": "Q2",
        "sjr_score": 0.7,
        "area": "Public Health",
    },
    "revista de saude publica": {
        "sjr_quartile": "Q2",
        "sjr_score": 0.7,
        "area": "Public Health",
    },
    "ciência & saúde coletiva": {
        "sjr_quartile": "Q2",
        "sjr_score": 0.5,
        "area": "Public Health",
    },
    "ciencia & saude coletiva": {
        "sjr_quartile": "Q2",
        "sjr_score": 0.5,
        "area": "Public Health",
    },
    "ciencia e saude coletiva": {
        "sjr_quartile": "Q2",
        "sjr_score": 0.5,
        "area": "Public Health",
    },
    "epidemiologia e serviços de saúde": {
        "sjr_quartile": "Q3",
        "sjr_score": 0.4,
        "area": "Public Health",
    },
    "epidemiologia e servicos de saude": {
        "sjr_quartile": "Q3",
        "sjr_score": 0.4,
        "area": "Public Health",
    },
    "plos one": {"sjr_quartile": "Q1", "sjr_score": 0.9, "area": "Multidisciplinary"},
    "bmc public health": {
        "sjr_quartile": "Q1",
        "sjr_score": 1.1,
        "area": "Public Health",
    },
    "tropical medicine & international health": {
        "sjr_quartile": "Q1",
        "sjr_score": 1.0,
        "area": "Tropical Medicine",
    },
    "international journal of tuberculosis and lung disease": {
        "sjr_quartile": "Q2",
        "sjr_score": 0.8,
        "area": "Pulmonology",
    },
    "tuberculosis": {"sjr_quartile": "Q2", "sjr_score": 0.9, "area": "Pulmonology"},
    "journal of the american medical informatics association": {
        "sjr_quartile": "Q1",
        "sjr_score": 2.5,
        "area": "Health Informatics",
    },
    "international journal of medical informatics": {
        "sjr_quartile": "Q1",
        "sjr_score": 1.5,
        "area": "Health Informatics",
    },
    "frontiers in public health": {
        "sjr_quartile": "Q1",
        "sjr_score": 1.0,
        "area": "Public Health",
    },
    # Computer Science / ML / Data Science
    "journal of machine learning research": {
        "sjr_quartile": "Q1",
        "sjr_score": 3.0,
        "area": "CS/ML",
    },
    "information": {"sjr_quartile": "Q2", "sjr_score": 0.5, "area": "CS/IS"},
    "ieee transactions on knowledge and data engineering": {
        "sjr_quartile": "Q1",
        "sjr_score": 2.5,
        "area": "CS/Data",
    },
    "data mining and knowledge discovery": {
        "sjr_quartile": "Q1",
        "sjr_score": 2.0,
        "area": "CS/Data",
    },
    "artificial intelligence in medicine": {
        "sjr_quartile": "Q1",
        "sjr_score": 1.8,
        "area": "CS/Health",
    },
    # Statistics
    "statistical science": {
        "sjr_quartile": "Q1",
        "sjr_score": 2.5,
        "area": "Statistics",
    },
    "journal of the american statistical association": {
        "sjr_quartile": "Q1",
        "sjr_score": 3.5,
        "area": "Statistics",
    },
    "the annals of statistics": {
        "sjr_quartile": "Q1",
        "sjr_score": 3.0,
        "area": "Statistics",
    },
    # Blockchain / Security
    "ieee access": {"sjr_quartile": "Q1", "sjr_score": 0.9, "area": "Engineering"},
    "journal of medical internet research": {
        "sjr_quartile": "Q1",
        "sjr_score": 2.0,
        "area": "Health Informatics",
    },
    "blockchain in healthcare today": {
        "sjr_quartile": "Q3",
        "sjr_score": 0.3,
        "area": "Health Informatics",
    },
    "computers in biology and medicine": {
        "sjr_quartile": "Q1",
        "sjr_score": 1.5,
        "area": "CS/Health",
    },
    # Brazilian epidemiology / linkage
    "revista brasileira de epidemiologia": {
        "sjr_quartile": "Q3",
        "sjr_score": 0.4,
        "area": "Epidemiology",
    },
    "international journal of population data science": {
        "sjr_quartile": "Q2",
        "sjr_score": 0.6,
        "area": "Health Informatics",
    },
    "memórias do instituto oswaldo cruz": {
        "sjr_quartile": "Q2",
        "sjr_score": 0.7,
        "area": "Tropical Medicine",
    },
    "memorias do instituto oswaldo cruz": {
        "sjr_quartile": "Q2",
        "sjr_score": 0.7,
        "area": "Tropical Medicine",
    },
    "bmc infectious diseases": {
        "sjr_quartile": "Q1",
        "sjr_score": 1.2,
        "area": "Infectious Disease",
    },
    "journal of biomedical informatics": {
        "sjr_quartile": "Q1",
        "sjr_score": 1.8,
        "area": "Health Informatics",
    },
    "international journal of health geographics": {
        "sjr_quartile": "Q1",
        "sjr_score": 1.0,
        "area": "Health Informatics",
    },
    "statistics in medicine": {
        "sjr_quartile": "Q1",
        "sjr_score": 1.5,
        "area": "Statistics",
    },
    "saúde em debate": {
        "sjr_quartile": "Q3",
        "sjr_score": 0.3,
        "area": "Public Health",
    },
    "saude em debate": {
        "sjr_quartile": "Q3",
        "sjr_score": 0.3,
        "area": "Public Health",
    },
    "physis: revista de saúde coletiva": {
        "sjr_quartile": "Q3",
        "sjr_score": 0.3,
        "area": "Public Health",
    },
    "applied sciences": {
        "sjr_quartile": "Q2",
        "sjr_score": 0.5,
        "area": "Multidisciplinary",
    },
    "sensors": {"sjr_quartile": "Q1", "sjr_score": 0.8, "area": "Engineering"},
    "healthcare": {"sjr_quartile": "Q2", "sjr_score": 0.7, "area": "Health"},
}


def _normalize_journal(name: str) -> str:
    """Normalize journal name for matching."""
    import re

    name = name.lower().strip()
    # Remove braces, punctuation noise
    name = re.sub(r"[{}]", "", name)
    # Normalize common abbreviations
    name = name.replace("&amp;", "&")
    return name


def _match_journal(journal_name: str) -> dict | None:
    """Try to match journal name against known journals."""
    if not journal_name:
        return None

    norm = _normalize_journal(journal_name)

    # Exact match first
    if norm in KNOWN_JOURNALS:
        return KNOWN_JOURNALS[norm]

    # Fuzzy match
    best_score = 0.0
    best_match = None
    for known_name, data in KNOWN_JOURNALS.items():
        score = fuzz.token_sort_ratio(norm, known_name)
        if score > best_score:
            best_score = score
            best_match = data

    if best_score >= config.JOURNAL_SIMILARITY_THRESHOLD:
        return best_match

    return None


def _classify_evidence_level(
    entry: BibEntry,
    l1: L1Result,
    journal_info: dict | None,
) -> EvidenceLevel:
    """Classify evidence level A-E based on source type and journal quality."""
    # Level A: Q1-Q2 peer-reviewed OR seminal book with >1000 citations
    if journal_info:
        q = journal_info.get("sjr_quartile", "")
        if q in ("Q1", "Q2"):
            return EvidenceLevel.A

    if entry.source_type == SourceType.BOOK_OR_CHAPTER:
        if l1.citation_count and l1.citation_count > 1000:
            return EvidenceLevel.A
        if l1.citation_count and l1.citation_count > 200:
            return EvidenceLevel.B
        return EvidenceLevel.C

    # Level B: Q3-Q4 or top conference
    if journal_info:
        q = journal_info.get("sjr_quartile", "")
        if q in ("Q3", "Q4"):
            return EvidenceLevel.B

    if entry.source_type == SourceType.CONFERENCE_PAPER:
        # Assume decent conference if it has citations
        if l1.citation_count and l1.citation_count > 50:
            return EvidenceLevel.B
        return EvidenceLevel.C

    # Level C: thesis, techreport, WHO/government
    if entry.source_type in (
        SourceType.THESIS,
        SourceType.TECHREPORT,
    ):
        return EvidenceLevel.C

    # Level D: preprint, minor, web
    if entry.source_type in (SourceType.PREPRINT, SourceType.WEB_RESOURCE):
        return EvidenceLevel.D

    # Peer-reviewed but journal not found in our list
    if entry.source_type == SourceType.PEER_REVIEWED_ARTICLE:
        if l1.citation_count and l1.citation_count > 100:
            return EvidenceLevel.B
        return EvidenceLevel.C

    return EvidenceLevel.E


def _compute_l2_score(
    entry: BibEntry,
    journal_info: dict | None,
    evidence_level: EvidenceLevel,
) -> L2Score:
    """Compute L2 score based on journal quality and evidence level."""
    # No journal applicable (books, theses, reports)
    if entry.source_type not in (
        SourceType.PEER_REVIEWED_ARTICLE,
        SourceType.CONFERENCE_PAPER,
        SourceType.PREPRINT,
    ):
        return L2Score.L2_NA

    if evidence_level == EvidenceLevel.A:
        return L2Score.L2_PASS_HIGH

    if evidence_level == EvidenceLevel.B:
        return L2Score.L2_PASS

    if evidence_level in (EvidenceLevel.C, EvidenceLevel.D):
        if journal_info is None:
            return L2Score.L2_WARN_NOJOURNAL
        return L2Score.L2_WARN_LOW

    return L2Score.L2_WARN_NOJOURNAL


def assess_journal(
    entry: BibEntry,
    l1: L1Result,
) -> L2Result:
    """Assess journal quality for a single entry."""
    journal_name = l1.cr_journal or entry.journal or ""
    journal_info = _match_journal(journal_name)

    sjr_quartile = ""
    sjr_score = 0.0
    sjr_area = ""

    if journal_info:
        sjr_quartile = journal_info.get("sjr_quartile", "")
        sjr_score = journal_info.get("sjr_score", 0.0)
        sjr_area = journal_info.get("area", "")

    evidence_level = _classify_evidence_level(entry, l1, journal_info)
    score = _compute_l2_score(entry, journal_info, evidence_level)

    notes: list[str] = []
    if not journal_name:
        notes.append("No journal name found in bib or CrossRef")
    elif journal_info is None:
        notes.append(f"Journal '{journal_name}' not found in Scimago lookup")

    return L2Result(
        bib_key=entry.bib_key,
        score=score,
        journal_name=journal_name,
        sjr_quartile=sjr_quartile,
        sjr_score=sjr_score,
        sjr_hindex=0,
        sjr_area=sjr_area,
        evidence_level=evidence_level,
        is_predatory=False,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_e2(
    entries: list[BibEntry],
    l1_results: dict[str, L1Result],
    output_dir: Path,
) -> dict[str, L2Result]:
    """Run E2 journal assessment for all entries."""
    results: dict[str, L2Result] = {}

    for entry in entries:
        l1 = l1_results.get(entry.bib_key)
        if l1 is None:
            # Create minimal L1 for entries that weren't checked
            l1 = L1Result(bib_key=entry.bib_key)

        result = assess_journal(entry, l1)
        results[entry.bib_key] = result

    # Save
    out_path = output_dir / "l2_results.json"
    serializable = {k: v.to_dict() for k, v in results.items()}
    out_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Summary
    counter = Counter(r.score.value for r in results.values())
    logger.info("E2 complete: %s", dict(counter))
    for score_val, count in sorted(counter.items()):
        logger.info("  %s: %d", score_val, count)

    ev_counter = Counter(r.evidence_level.value for r in results.values())
    logger.info("Evidence levels: %s", dict(ev_counter))

    return results
