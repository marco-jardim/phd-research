"""Modelos de dados para o pipeline de auditoria de referências.

Usa dataclasses padrão com helper to_dict() para serialização JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SourceType(str, Enum):
    PEER_REVIEWED_ARTICLE = "peer_reviewed_article"
    CONFERENCE_PAPER = "conference_paper"
    BOOK_OR_CHAPTER = "book_or_chapter"
    THESIS = "thesis"
    TECHREPORT = "techreport"
    PREPRINT = "preprint"
    WEB_RESOURCE = "web_resource"
    UNKNOWN = "unknown"


class L1Score(str, Enum):
    L1_PASS = "L1_PASS"
    L1_WARN_META = "L1_WARN_META"
    L1_WARN_NO_DOI = "L1_WARN_NO_DOI"
    L1_WARN_PARTIAL = "L1_WARN_PARTIAL"
    L1_FAIL_GHOST = "L1_FAIL_GHOST"
    L1_FAIL_RETRACT = "L1_FAIL_RETRACT"


class L2Score(str, Enum):
    L2_PASS_HIGH = "L2_PASS_HIGH"
    L2_PASS = "L2_PASS"
    L2_WARN_LOW = "L2_WARN_LOW"
    L2_WARN_NOJOURNAL = "L2_WARN_NOJOURNAL"
    L2_FAIL_PREDATORY = "L2_FAIL_PREDATORY"
    L2_NA = "L2_NA"


class L3Score(str, Enum):
    L3_PASS = "L3_PASS"
    L3_WARN_PARTIAL = "L3_WARN_PARTIAL"
    L3_WARN_INCONCLUSIVE = "L3_WARN_INCONCLUSIVE"
    L3_FAIL_UNSUPPORTED = "L3_FAIL_UNSUPPORTED"
    L3_NA = "L3_NA"


class ClaimType(str, Enum):
    DEFINE = "DEFINE"
    SUPPORT_FACT = "SUPPORT_FACT"
    SUPPORT_METHOD = "SUPPORT_METHOD"
    ATTRIBUTE = "ATTRIBUTE"
    CONTRAST = "CONTRAST"
    EXTEND = "EXTEND"
    BACKGROUND = "BACKGROUND"
    DATA_SOURCE = "DATA_SOURCE"
    TOOL = "TOOL"
    UNKNOWN = "UNKNOWN"


class CompositeScore(str, Enum):
    REF_SOLID = "REF_SOLID"
    REF_OK = "REF_OK"
    REF_REVIEW = "REF_REVIEW"
    REF_PROBLEM = "REF_PROBLEM"
    REF_CRITICAL = "REF_CRITICAL"


class Severity(str, Enum):
    BLOQUEANTE = "BLOQUEANTE"
    ALTA = "ALTA"
    MEDIA = "MÉDIA"
    BAIXA = "BAIXA"


class EvidenceLevel(str, Enum):
    A = "A"  # Q1-Q2 or seminal book >1000cit
    B = "B"  # Q3-Q4 or top conference
    C = "C"  # thesis/techreport/WHO
    D = "D"  # preprint/minor conf/web
    E = "E"  # unclassifiable


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _dc_to_dict(obj: object) -> dict:
    """Converte dataclass para dict serializável, tratando enums."""
    d = asdict(obj)  # type: ignore[arg-type]

    def _convert(v: object) -> object:
        if isinstance(v, Enum):
            return v.value
        if isinstance(v, dict):
            return {kk: _convert(vv) for kk, vv in v.items()}
        if isinstance(v, list):
            return [_convert(item) for item in v]
        return v

    return {k: _convert(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BibEntry:
    """Entrada normalizada do .bib."""

    bib_key: str
    entry_type: str  # article, book, inproceedings, etc.
    title: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None  # normalized: 10.xxxx/yyyy
    journal: Optional[str] = None
    issn: Optional[str] = None
    url: Optional[str] = None
    publisher: Optional[str] = None
    source_type: SourceType = SourceType.UNKNOWN
    completeness_issues: list[str] = field(default_factory=list)

    @property
    def has_doi(self) -> bool:
        return self.doi is not None and len(self.doi) > 3

    def to_dict(self) -> dict:
        return _dc_to_dict(self)


@dataclass
class CitationContext:
    """Contexto de uma citação no .tex."""

    bib_key: str
    tex_file: str
    line_number: int
    sentence: str
    paragraph: str
    section: Optional[str] = None
    claim_type: ClaimType = ClaimType.UNKNOWN

    def to_dict(self) -> dict:
        return _dc_to_dict(self)


@dataclass
class L1Result:
    """Resultado da camada de existência."""

    bib_key: str
    score: L1Score
    resolved_doi: Optional[str] = None
    cr_title: str = ""
    cr_authors: list[str] = field(default_factory=list)
    cr_year: Optional[int] = None
    cr_journal: str = ""
    cr_type: str = ""
    citation_count: Optional[int] = None
    title_similarity: float = 0.0
    year_match: bool = False
    author_match: bool = False
    is_retracted: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return _dc_to_dict(self)


@dataclass
class L2Result:
    """Resultado da camada de journal."""

    bib_key: str = ""
    score: L2Score = L2Score.L2_NA
    journal_name: str = ""
    sjr_score: float = 0.0
    sjr_quartile: str = ""  # Q1, Q2, Q3, Q4
    sjr_hindex: int = 0
    sjr_area: str = ""
    qualis_score: str = ""  # A1, A2, B1, ..., C
    evidence_level: EvidenceLevel = EvidenceLevel.E
    is_predatory: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return _dc_to_dict(self)


@dataclass
class L3Evaluation:
    """Avaliação individual de uma citação vs abstract."""

    score: L3Score = L3Score.L3_WARN_INCONCLUSIVE
    confidence: str = "LOW"  # HIGH, MEDIUM, LOW
    justification: str = ""
    claim_type: ClaimType = ClaimType.UNKNOWN

    def to_dict(self) -> dict:
        return _dc_to_dict(self)


@dataclass
class L3Result:
    """Resultado da camada de relevância."""

    bib_key: str = ""
    score: L3Score = L3Score.L3_NA
    abstract_available: bool = False
    evaluations: list[L3Evaluation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return _dc_to_dict(self)


@dataclass
class AuditFlag:
    """Flag de auditoria."""

    flag_type: str  # ORPHAN_BIB, BROKEN_CITE, STALE_REF, etc.
    severity: Severity
    bib_key: Optional[str] = None
    detail: str = ""

    def to_dict(self) -> dict:
        return _dc_to_dict(self)


@dataclass
class CompositeResult:
    """Resultado consolidado por referência."""

    bib_key: str
    entry: Optional[BibEntry] = None
    l1: Optional[L1Result] = None
    l2: Optional[L2Result] = None
    l3: Optional[L3Result] = None
    composite: CompositeScore = CompositeScore.REF_REVIEW
    flags: list[AuditFlag] = field(default_factory=list)
    contexts: list[CitationContext] = field(default_factory=list)
    # Via B (Consensus) — optional enrichment
    consensus_enrichment: Optional[dict] = None
    via_b_resolved: bool = False
    # Fulltext layer — optional re-evaluation
    fulltext_available: bool = False
    fulltext_source: str = ""
    fulltext_l3_score: Optional[str] = None
    fulltext_upgraded: bool = False

    def to_dict(self) -> dict:
        return _dc_to_dict(self)
