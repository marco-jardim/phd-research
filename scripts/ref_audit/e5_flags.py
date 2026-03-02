"""E5 — Audit flags: detect structural problems across the reference set."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

from .models import (
    AuditFlag,
    BibEntry,
    CitationContext,
    L1Result,
    L1Score,
    L2Result,
    L3Result,
    Severity,
)
from . import config

logger = logging.getLogger(__name__)


def compute_flags(
    entries: list[BibEntry],
    contexts: list[CitationContext],
    l1_results: dict[str, L1Result],
    l2_results: dict[str, L2Result],
    l3_results: dict[str, L3Result],
    orphan_cites: set[str],
    uncited_bib: set[str],
) -> list[AuditFlag]:
    """Compute all audit flags for the reference set."""
    flags: list[AuditFlag] = []

    # --- BROKEN_CITE (bloqueante) ---
    for key in orphan_cites:
        flags.append(
            AuditFlag(
                flag_type="BROKEN_CITE",
                severity=Severity.BLOQUEANTE,
                bib_key=key,
                detail=f"\\cite{{{key}}} found in .tex but no matching .bib entry.",
            )
        )

    # --- ORPHAN_BIB ---
    for key in uncited_bib:
        flags.append(
            AuditFlag(
                flag_type="ORPHAN_BIB",
                severity=Severity.BAIXA,
                bib_key=key,
                detail=f"Entry '{key}' in .bib but never cited in any .tex file.",
            )
        )

    # --- STALE_REF ---
    for entry in entries:
        if entry.bib_key in uncited_bib:
            continue
        l1 = l1_results.get(entry.bib_key)
        year = entry.year
        cit_count = l1.citation_count if l1 else None
        if year and year < config.STALE_YEAR:
            if cit_count is not None and cit_count < config.STALE_MIN_CITATIONS:
                flags.append(
                    AuditFlag(
                        flag_type="STALE_REF",
                        severity=Severity.MEDIA,
                        bib_key=entry.bib_key,
                        detail=f"Published {year}, only {cit_count} citations. "
                        f"Threshold: <{config.STALE_YEAR} AND <{config.STALE_MIN_CITATIONS} cit.",
                    )
                )

    # --- MISSING_DOI (only for source types where DOI is expected) ---
    from .models import SourceType

    _DOI_EXPECTED_TYPES = {
        SourceType.PEER_REVIEWED_ARTICLE,
        SourceType.CONFERENCE_PAPER,
        SourceType.PREPRINT,
    }
    cited_keys = {c.bib_key for c in contexts}
    for entry in entries:
        if entry.bib_key not in cited_keys:
            continue
        if entry.source_type not in _DOI_EXPECTED_TYPES:
            continue
        if not entry.doi:
            l1 = l1_results.get(entry.bib_key)
            resolved = l1.resolved_doi if l1 else None
            if not resolved:
                flags.append(
                    AuditFlag(
                        flag_type="MISSING_DOI",
                        severity=Severity.MEDIA,
                        bib_key=entry.bib_key,
                        detail="No DOI in .bib and none resolved via APIs.",
                    )
                )

    # --- CLUSTER_IMBALANCE (>30% same author surname) ---
    cited_entries = [e for e in entries if e.bib_key in cited_keys]
    if cited_entries:
        author_counts: Counter[str] = Counter()
        for entry in cited_entries:
            for author in entry.authors:
                surname = author.split(",")[0].strip().split()[-1].lower()
                if len(surname) > 2:
                    author_counts[surname] += 1
        total_cited = len(cited_entries)
        for surname, count in author_counts.most_common(5):
            ratio = count / total_cited
            if ratio > config.CLUSTER_IMBALANCE_THRESHOLD:
                flags.append(
                    AuditFlag(
                        flag_type="CLUSTER_IMBALANCE",
                        severity=Severity.MEDIA,
                        bib_key=None,
                        detail=f"Author '{surname}' appears in {count}/{total_cited} "
                        f"({ratio:.0%}) cited refs. Threshold: {config.CLUSTER_IMBALANCE_THRESHOLD:.0%}.",
                    )
                )

    # --- LOW_DIVERSITY (>50% same journal) ---
    journal_counts: Counter[str] = Counter()
    for entry in cited_entries:
        j = entry.journal.strip().lower() if entry.journal else ""
        if j:
            journal_counts[j] += 1
    if journal_counts:
        top_j, top_count = journal_counts.most_common(1)[0]
        ratio = top_count / len(cited_entries)
        if ratio > config.LOW_DIVERSITY_THRESHOLD:
            flags.append(
                AuditFlag(
                    flag_type="LOW_DIVERSITY",
                    severity=Severity.MEDIA,
                    bib_key=None,
                    detail=f"Journal '{top_j}' accounts for {top_count}/{len(cited_entries)} "
                    f"({ratio:.0%}) cited refs. Threshold: {config.LOW_DIVERSITY_THRESHOLD:.0%}.",
                )
            )

    # --- SELF_CITE_HIGH (>20% advisor group) ---
    # Heuristic: count refs where advisor surname appears
    advisor_surnames = {"pinheiro", "sobrino"}  # from tese.tex orientadora
    self_count = 0
    for entry in cited_entries:
        for author in entry.authors:
            if any(s in author.lower() for s in advisor_surnames):
                self_count += 1
                break
    if cited_entries:
        ratio = self_count / len(cited_entries)
        if ratio > config.SELF_CITE_THRESHOLD:
            flags.append(
                AuditFlag(
                    flag_type="SELF_CITE_HIGH",
                    severity=Severity.BAIXA,
                    bib_key=None,
                    detail=f"Advisor-group refs: {self_count}/{len(cited_entries)} "
                    f"({ratio:.0%}). Threshold: {config.SELF_CITE_THRESHOLD:.0%}.",
                )
            )

    # --- PREPRINT_NO_PUBLISHED ---
    from .models import SourceType

    for entry in cited_entries:
        if entry.source_type == SourceType.PREPRINT:
            flags.append(
                AuditFlag(
                    flag_type="PREPRINT_NO_PUBLISHED",
                    severity=Severity.ALTA,
                    bib_key=entry.bib_key,
                    detail="Preprint cited without indication of published version.",
                )
            )

    logger.info("E5: %d flags generated.", len(flags))
    return flags
