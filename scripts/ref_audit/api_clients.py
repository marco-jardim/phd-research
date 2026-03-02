"""API clients for CrossRef, OpenAlex, Semantic Scholar, and PubMed."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from . import config

logger = logging.getLogger(__name__)

_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=30.0, follow_redirects=True)
    return _client


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_key(prefix: str, identifier: str) -> str:
    h = hashlib.md5(identifier.encode()).hexdigest()
    return f"{prefix}_{h}"


def _read_cache(cache_dir: Path, key: str) -> Any | None:
    path = cache_dir / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _write_cache(cache_dir: Path, key: str, data: Any) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CrossRef
# ---------------------------------------------------------------------------


def _crossref_headers() -> dict[str, str]:
    h: dict[str, str] = {"Accept": "application/json"}
    if config.CROSSREF_MAILTO:
        h["User-Agent"] = f"PhDResearchAudit/1.0 (mailto:{config.CROSSREF_MAILTO})"
    return h


def crossref_get_by_doi(doi: str, cache_dir: Path) -> dict | None:
    """Fetch CrossRef metadata for a DOI. Returns the 'message' dict or None."""
    ck = _cache_key("cr_doi", doi)
    cached = _read_cache(cache_dir, ck)
    if cached is not None:
        return cached

    url = f"https://api.crossref.org/works/{doi}"
    try:
        resp = _get_client().get(url, headers=_crossref_headers())
        if resp.status_code == 200:
            data = resp.json().get("message", {})
            _write_cache(cache_dir, ck, data)
            return data
        logger.warning("CrossRef DOI %s returned %d", doi, resp.status_code)
    except Exception as exc:
        logger.error("CrossRef DOI error for %s: %s", doi, exc)
    return None


def crossref_search_title(title: str, cache_dir: Path, rows: int = 5) -> list[dict]:
    """Search CrossRef by title. Returns list of work items."""
    ck = _cache_key("cr_title", title.lower()[:100])
    cached = _read_cache(cache_dir, ck)
    if cached is not None:
        return cached if isinstance(cached, list) else []

    url = "https://api.crossref.org/works"
    params = {"query.bibliographic": title, "rows": rows}
    try:
        resp = _get_client().get(url, params=params, headers=_crossref_headers())
        if resp.status_code == 200:
            items = resp.json().get("message", {}).get("items", [])
            _write_cache(cache_dir, ck, items)
            return items
    except Exception as exc:
        logger.error("CrossRef search error: %s", exc)
    return []


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------


def _openalex_headers() -> dict[str, str]:
    h: dict[str, str] = {"Accept": "application/json"}
    if config.CROSSREF_MAILTO:
        h["User-Agent"] = f"PhDResearchAudit/1.0 (mailto:{config.CROSSREF_MAILTO})"
    return h


def openalex_get_by_doi(doi: str, cache_dir: Path) -> dict | None:
    """Fetch OpenAlex work by DOI."""
    ck = _cache_key("oa_doi", doi)
    cached = _read_cache(cache_dir, ck)
    if cached is not None:
        return cached

    url = f"https://api.openalex.org/works/doi:{doi}"
    params = {}
    if config.CROSSREF_MAILTO:
        params["mailto"] = config.CROSSREF_MAILTO
    try:
        resp = _get_client().get(url, params=params, headers=_openalex_headers())
        if resp.status_code == 200:
            data = resp.json()
            _write_cache(cache_dir, ck, data)
            return data
    except Exception as exc:
        logger.error("OpenAlex DOI error for %s: %s", doi, exc)
    return None


def openalex_search_title(title: str, cache_dir: Path, per_page: int = 5) -> list[dict]:
    """Search OpenAlex by title."""
    ck = _cache_key("oa_title", title.lower()[:100])
    cached = _read_cache(cache_dir, ck)
    if cached is not None:
        return cached if isinstance(cached, list) else []

    url = "https://api.openalex.org/works"
    params: dict[str, Any] = {
        "search": title,
        "per_page": per_page,
    }
    if config.CROSSREF_MAILTO:
        params["mailto"] = config.CROSSREF_MAILTO
    try:
        resp = _get_client().get(url, params=params, headers=_openalex_headers())
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            _write_cache(cache_dir, ck, results)
            return results
    except Exception as exc:
        logger.error("OpenAlex search error: %s", exc)
    return []


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------


def semantic_scholar_get_by_doi(doi: str, cache_dir: Path) -> dict | None:
    """Fetch Semantic Scholar paper by DOI."""
    ck = _cache_key("s2_doi", doi)
    cached = _read_cache(cache_dir, ck)
    if cached is not None:
        return cached

    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    params = {
        "fields": "title,abstract,year,authors,citationCount,tldr,journal,externalIds"
    }
    headers: dict[str, str] = {}
    if config.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = config.SEMANTIC_SCHOLAR_API_KEY
    try:
        resp = _get_client().get(url, params=params, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            _write_cache(cache_dir, ck, data)
            return data
        if resp.status_code == 429:
            logger.warning("Semantic Scholar rate limited, sleeping 60s")
            time.sleep(60)
    except Exception as exc:
        logger.error("Semantic Scholar DOI error for %s: %s", doi, exc)
    return None


def semantic_scholar_search_title(title: str, cache_dir: Path) -> list[dict]:
    """Search Semantic Scholar by title."""
    ck = _cache_key("s2_title", title.lower()[:100])
    cached = _read_cache(cache_dir, ck)
    if cached is not None:
        return cached if isinstance(cached, list) else []

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title[:200],
        "limit": 5,
        "fields": "title,abstract,year,authors,citationCount,tldr,journal,externalIds",
    }
    headers: dict[str, str] = {}
    if config.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = config.SEMANTIC_SCHOLAR_API_KEY
    try:
        resp = _get_client().get(url, params=params, headers=headers)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            _write_cache(cache_dir, ck, data)
            return data
        if resp.status_code == 429:
            logger.warning("Semantic Scholar rate limited, sleeping 60s")
            time.sleep(60)
    except Exception as exc:
        logger.error("Semantic Scholar search error: %s", exc)
    return []


# ---------------------------------------------------------------------------
# PubMed (via Entrez E-utilities)
# ---------------------------------------------------------------------------


def pubmed_search(title: str, cache_dir: Path, max_results: int = 3) -> list[str]:
    """Search PubMed by title, return list of PMIDs."""
    ck = _cache_key("pm_search", title.lower()[:100])
    cached = _read_cache(cache_dir, ck)
    if cached is not None:
        return cached if isinstance(cached, list) else []

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params: dict[str, Any] = {
        "db": "pubmed",
        "term": f"{title}[Title]",
        "retmax": max_results,
        "retmode": "json",
    }
    if config.NCBI_API_KEY:
        params["api_key"] = config.NCBI_API_KEY
    try:
        resp = _get_client().get(url, params=params)
        if resp.status_code == 200:
            ids = resp.json().get("esearchresult", {}).get("idlist", [])
            _write_cache(cache_dir, ck, ids)
            return ids
    except Exception as exc:
        logger.error("PubMed search error: %s", exc)
    return []


def pubmed_fetch_abstract(pmid: str, cache_dir: Path) -> dict | None:
    """Fetch abstract and metadata for a PubMed ID."""
    ck = _cache_key("pm_abstract", pmid)
    cached = _read_cache(cache_dir, ck)
    if cached is not None:
        return cached

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params: dict[str, Any] = {
        "db": "pubmed",
        "id": pmid,
        "rettype": "abstract",
        "retmode": "xml",
    }
    if config.NCBI_API_KEY:
        params["api_key"] = config.NCBI_API_KEY
    try:
        resp = _get_client().get(url, params=params)
        if resp.status_code == 200:
            # Parse XML minimally to get abstract text
            text = resp.text
            abstract = _extract_xml_abstract(text)
            data = {"pmid": pmid, "abstract": abstract, "raw_xml_length": len(text)}
            _write_cache(cache_dir, ck, data)
            return data
    except Exception as exc:
        logger.error("PubMed fetch error for %s: %s", pmid, exc)
    return None


def _extract_xml_abstract(xml_text: str) -> str:
    """Extract abstract text from PubMed XML (simple regex-based)."""
    import re

    # Find all AbstractText elements
    matches = re.findall(
        r"<AbstractText[^>]*>(.*?)</AbstractText>", xml_text, re.DOTALL
    )
    if matches:
        # Clean HTML tags
        abstract = " ".join(matches)
        abstract = re.sub(r"<[^>]+>", "", abstract)
        return abstract.strip()
    return ""
