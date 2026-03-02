"""E4 — Consolidated report: markdown + CSV output."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .models import (
    AuditFlag,
    BibEntry,
    CitationContext,
    CompositeResult,
    CompositeScore,
    L1Result,
    L1Score,
    L2Result,
    L2Score,
    L3Result,
    L3Score,
    Severity,
)
from .verified_refs import load_verified, is_verified_ok

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Composite score computation
# ---------------------------------------------------------------------------


def _compute_composite(
    l1: Optional[L1Result],
    l2: Optional[L2Result],
    l3: Optional[L3Result],
    entry_flags: list[AuditFlag],
) -> CompositeScore:
    """Derive composite score from L1/L2/L3 + flags."""
    # CRITICAL: retracted or predatory
    if l1 and l1.is_retracted:
        return CompositeScore.REF_CRITICAL
    if l2 and l2.is_predatory:
        return CompositeScore.REF_CRITICAL
    if any(f.severity == Severity.BLOQUEANTE for f in entry_flags):
        return CompositeScore.REF_CRITICAL

    # PROBLEM: any FAIL
    has_fail = False
    if l1 and l1.score in (L1Score.L1_FAIL_GHOST, L1Score.L1_FAIL_RETRACT):
        has_fail = True
    if l2 and l2.score == L2Score.L2_FAIL_PREDATORY:
        has_fail = True
    if l3 and l3.score == L3Score.L3_FAIL_UNSUPPORTED:
        has_fail = True
    if has_fail:
        return CompositeScore.REF_PROBLEM

    # REVIEW: any WARN
    has_warn = False
    if l1 and l1.score in (
        L1Score.L1_WARN_META,
        L1Score.L1_WARN_NO_DOI,
        L1Score.L1_WARN_PARTIAL,
    ):
        has_warn = True
    if l2 and l2.score in (L2Score.L2_WARN_LOW, L2Score.L2_WARN_NOJOURNAL):
        has_warn = True
    if l3 and l3.score in (L3Score.L3_WARN_PARTIAL, L3Score.L3_WARN_INCONCLUSIVE):
        has_warn = True
    if any(f.severity in (Severity.ALTA, Severity.MEDIA) for f in entry_flags):
        has_warn = True
    if has_warn:
        return CompositeScore.REF_REVIEW

    # SOLID vs OK
    l1_pass = l1 and l1.score == L1Score.L1_PASS
    l2_high = l2 and l2.score in (L2Score.L2_PASS_HIGH, L2Score.L2_PASS)
    l3_pass = l3 and l3.score == L3Score.L3_PASS
    if l1_pass and l2_high and l3_pass:
        return CompositeScore.REF_SOLID

    return CompositeScore.REF_OK


# ---------------------------------------------------------------------------
# Build composite results
# ---------------------------------------------------------------------------


def build_composite_results(
    entries: list[BibEntry],
    contexts: list[CitationContext],
    l1_results: dict[str, L1Result],
    l2_results: dict[str, L2Result],
    l3_results: dict[str, L3Result],
    flags: list[AuditFlag],
) -> dict[str, CompositeResult]:
    """Assemble CompositeResult per bib_key."""
    ctx_by_key: dict[str, list[CitationContext]] = {}
    for c in contexts:
        ctx_by_key.setdefault(c.bib_key, []).append(c)

    flags_by_key: dict[str, list[AuditFlag]] = {}
    for f in flags:
        key = f.bib_key or "__global__"
        flags_by_key.setdefault(key, []).append(f)

    results: dict[str, CompositeResult] = {}
    for entry in entries:
        k = entry.bib_key
        l1 = l1_results.get(k)
        l2 = l2_results.get(k)
        l3 = l3_results.get(k)
        ef = flags_by_key.get(k, [])
        composite = _compute_composite(l1, l2, l3, ef)
        results[k] = CompositeResult(
            bib_key=k,
            entry=entry,
            l1=l1,
            l2=l2,
            l3=l3,
            composite=composite,
            flags=ef,
            contexts=ctx_by_key.get(k, []),
        )

    # Add global flags
    for f in flags_by_key.get("__global__", []):
        results.setdefault(
            "__global_flags__",
            CompositeResult(
                bib_key="__global__",
                composite=CompositeScore.REF_REVIEW,
                flags=[],
            ),
        ).flags.append(f)

    return results


def apply_verified_overrides(
    composites: dict[str, CompositeResult],
    verified_db: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Aplica overrides de verificação manual ao composite score.

    Entries com status VERIFIED_OK ou VERIFIED_ACCEPT que tinham
    REF_REVIEW ou REF_PROBLEM são promovidas a REF_OK.

    Returns:
        Dict mapping bib_key -> verified entry info (only overridden ones).
    """
    if verified_db is None:
        verified_db = load_verified()

    overridden: dict[str, dict[str, Any]] = {}
    for key, cr in composites.items():
        if key == "__global__":
            continue
        if not is_verified_ok(verified_db, key):
            continue
        ventry = verified_db[key]
        if cr.composite in (CompositeScore.REF_REVIEW, CompositeScore.REF_PROBLEM):
            overridden[key] = {
                "original_composite": cr.composite.value,
                **ventry,
            }
            cr.composite = CompositeScore.REF_OK

    if overridden:
        logger.info(
            "Verificações manuais: %d entradas promovidas a REF_OK",
            len(overridden),
        )
    return overridden


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def generate_markdown_report(
    composites: dict[str, CompositeResult],
    output_dir: Path,
    overridden: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Generate RELATORIO_AUDITORIA_REFERENCIAS.md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "RELATORIO_AUDITORIA_REFERENCIAS.md"

    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"# Relatório de Auditoria de Referências\n")
    lines.append(f"**Gerado em:** {now}\n")

    # Summary counts
    from collections import Counter

    score_counts = Counter(
        c.composite for c in composites.values() if c.bib_key != "__global__"
    )
    total = sum(score_counts.values())
    lines.append("## Resumo\n")
    lines.append(f"| Score | Qtd | % |")
    lines.append(f"|-------|-----|---|")
    for sc in CompositeScore:
        cnt = score_counts.get(sc, 0)
        pct = f"{cnt / total:.0%}" if total else "0%"
        emoji = {
            "REF_SOLID": "OK",
            "REF_OK": "OK",
            "REF_REVIEW": "!",
            "REF_PROBLEM": "X",
            "REF_CRITICAL": "XX",
        }.get(sc.value, "?")
        lines.append(f"| {emoji} {sc.value} | {cnt} | {pct} |")
    lines.append(f"| **TOTAL** | **{total}** | |")
    lines.append("")

    # Global flags
    global_flags = composites.get("__global_flags__")
    if global_flags and global_flags.flags:
        lines.append("## Flags Globais\n")
        for f in global_flags.flags:
            lines.append(f"- **{f.flag_type}** [{f.severity.value}]: {f.detail}")
        lines.append("")

    # Detail by severity
    for section_score in [
        CompositeScore.REF_CRITICAL,
        CompositeScore.REF_PROBLEM,
        CompositeScore.REF_REVIEW,
    ]:
        entries_in_section = [
            c
            for c in composites.values()
            if c.composite == section_score and c.bib_key != "__global__"
        ]
        if not entries_in_section:
            continue
        lines.append(
            f"## {section_score.value} ({len(entries_in_section)} referências)\n"
        )
        for cr in sorted(entries_in_section, key=lambda x: x.bib_key):
            lines.append(f"### `{cr.bib_key}`\n")
            if cr.entry:
                title = cr.entry.title or "(sem título)"
                year = cr.entry.year or "?"
                lines.append(f"**Título:** {title}  ")
                lines.append(f"**Ano:** {year}  ")
                lines.append(f"**DOI:** {cr.entry.doi or 'N/A'}  ")
            if cr.l1:
                l1_score_str = (
                    cr.l1.score.value
                    if hasattr(cr.l1.score, "value")
                    else str(cr.l1.score)
                )
                lines.append(
                    f"**L1 (Existência):** {l1_score_str} "
                    f"(title_sim={cr.l1.title_similarity:.0f}%, "
                    f"year={cr.l1.year_match}, author={cr.l1.author_match})"
                )
            if cr.l2:
                l2_score_str = (
                    cr.l2.score.value
                    if hasattr(cr.l2.score, "value")
                    else str(cr.l2.score)
                )
                evidence_val = (
                    cr.l2.evidence_level.value
                    if hasattr(cr.l2.evidence_level, "value")
                    else str(cr.l2.evidence_level)
                )
                lines.append(
                    f"**L2 (Journal):** {l2_score_str} "
                    f"(journal={cr.l2.journal_name}, quartile={cr.l2.sjr_quartile}, "
                    f"evidence={evidence_val})"
                )
            if cr.l3:
                l3_score_str = (
                    cr.l3.score.value
                    if hasattr(cr.l3.score, "value")
                    else str(cr.l3.score)
                )
                lines.append(
                    f"**L3 (Relevância):** {l3_score_str} "
                    f"(abstract={'sim' if cr.l3.abstract_available else 'não'})"
                )
                for ev in cr.l3.evaluations:
                    claim_type_val = (
                        ev.claim_type.value
                        if hasattr(ev.claim_type, "value")
                        else str(ev.claim_type)
                    )
                    ev_score_val = (
                        ev.score.value if hasattr(ev.score, "value") else str(ev.score)
                    )
                    lines.append(
                        f"  - {claim_type_val}: {ev_score_val} "
                        f"[{ev.confidence}] — {ev.justification}"
                    )
            if cr.flags:
                lines.append("**Flags:**")
                for f in cr.flags:
                    lines.append(f"  - {f.flag_type} [{f.severity.value}]: {f.detail}")
            lines.append("")

    # Verified overrides section
    if overridden:
        lines.append(f"## Verificadas manualmente ({len(overridden)} referências)\n")
        lines.append(
            "Entradas abaixo foram revisadas manualmente e promovidas a "
            "REF_OK. O score original (antes do override) é indicado.\n"
        )
        lines.append("| Chave | Score original | Status | Motivo |")
        lines.append("|-------|---------------|--------|--------|")
        for key in sorted(overridden):
            v = overridden[key]
            orig = v.get("original_composite", "?")
            status = v.get("status", "?")
            reason = v.get("reason", "")
            lines.append(f"| `{key}` | {orig} | {status} | {reason} |")
        lines.append("")

    # REF_SOLID and REF_OK — compact listing
    ok_entries = [
        c
        for c in composites.values()
        if c.composite in (CompositeScore.REF_SOLID, CompositeScore.REF_OK)
        and c.bib_key != "__global__"
    ]
    if ok_entries:
        lines.append(f"## REF_SOLID + REF_OK ({len(ok_entries)} referências)\n")
        lines.append("| Chave | Score | L1 | L2 | L3 |")
        lines.append("|-------|-------|----|----|----|")
        for cr in sorted(ok_entries, key=lambda x: x.bib_key):
            l1s = _l1_score_display(cr)
            l2s = _l2_score_display(cr)
            l3s = _l3_score_display(cr)
            composite_val = (
                cr.composite.value
                if hasattr(cr.composite, "value")
                else str(cr.composite)
            )
            lines.append(
                f"| `{cr.bib_key}` | {composite_val} | {l1s} | {l2s} | {l3s} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown report saved to %s", out_path)
    return out_path


def _l1_score_display(cr: CompositeResult) -> str:
    """Get display string for L1 score, handling both enum and string."""
    if not cr.l1:
        return "N/A"
    # Handle both enum objects and string values from JSON
    if hasattr(cr.l1.score, "value"):
        return cr.l1.score.value
    return str(cr.l1.score)


def _l2_score_display(cr: CompositeResult) -> str:
    """Get display string for L2 score, handling both enum and string."""
    if not cr.l2:
        return "N/A"
    if hasattr(cr.l2.score, "value"):
        return cr.l2.score.value
    return str(cr.l2.score)


def _l3_score_display(cr: CompositeResult) -> str:
    """Get display string for L3 score, handling both enum and string."""
    if not cr.l3:
        return "N/A"
    if hasattr(cr.l3.score, "value"):
        return cr.l3.score.value
    return str(cr.l3.score)


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------


def generate_csv_report(
    composites: dict[str, CompositeResult],
    output_dir: Path,
) -> Path:
    """Generate ref_audit_report.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ref_audit_report.csv"

    headers = [
        "bib_key",
        "title",
        "year",
        "doi",
        "source_type",
        "journal",
        "composite_score",
        "l1_score",
        "l1_title_sim",
        "l1_year_match",
        "l1_author_match",
        "l1_citation_count",
        "l1_retracted",
        "l2_score",
        "l2_journal",
        "l2_quartile",
        "l2_evidence",
        "l2_predatory",
        "l3_score",
        "l3_abstract_available",
        "l3_evaluations_count",
        "flags_count",
        "flag_types",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for cr in composites.values():
            if cr.bib_key == "__global__":
                continue
            row = {
                "bib_key": cr.bib_key,
                "title": cr.entry.title if cr.entry else "",
                "year": cr.entry.year if cr.entry else "",
                "doi": cr.entry.doi if cr.entry else "",
                "source_type": cr.entry.source_type.value if cr.entry else "",
                "journal": cr.entry.journal if cr.entry else "",
                "composite_score": cr.composite.value,
                "l1_score": (
                    cr.l1.score.value
                    if hasattr(cr.l1.score, "value")
                    else str(cr.l1.score)
                )
                if cr.l1
                else "",
                "l1_title_sim": f"{cr.l1.title_similarity:.0f}" if cr.l1 else "",
                "l1_year_match": cr.l1.year_match if cr.l1 else "",
                "l1_author_match": cr.l1.author_match if cr.l1 else "",
                "l1_citation_count": cr.l1.citation_count if cr.l1 else "",
                "l1_retracted": getattr(
                    cr.l1, "is_retracted", getattr(cr.l1, "retracted", "")
                )
                if cr.l1
                else "",
                "l2_score": (
                    cr.l2.score.value
                    if hasattr(cr.l2.score, "value")
                    else str(cr.l2.score)
                )
                if cr.l2
                else "",
                "l2_journal": cr.l2.journal_name if cr.l2 else "",
                "l2_quartile": cr.l2.sjr_quartile if cr.l2 else "",
                "l2_evidence": (
                    cr.l2.evidence_level.value
                    if hasattr(cr.l2.evidence_level, "value")
                    else str(cr.l2.evidence_level)
                )
                if cr.l2
                else "",
                "l2_predatory": getattr(cr.l2, "predatory_warning", "")
                if cr.l2
                else "",
                "l3_score": (
                    cr.l3.score.value
                    if hasattr(cr.l3.score, "value")
                    else str(cr.l3.score)
                )
                if cr.l3
                else "",
                "l3_abstract_available": cr.l3.abstract_available if cr.l3 else "",
                "l3_evaluations_count": len(cr.l3.evaluations) if cr.l3 else 0,
                "flags_count": len(cr.flags),
                "flag_types": "; ".join(f.flag_type for f in cr.flags),
            }
            writer.writerow(row)

    logger.info("CSV report saved to %s", out_path)
    return out_path
