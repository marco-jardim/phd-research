"""E3 — Relevance layer: abstract retrieval + claim-support evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

# Suppress noisy httpx/httpcore debug logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from . import config
from .api_clients import (
    pubmed_fetch_abstract,
    pubmed_search,
    semantic_scholar_get_by_doi,
    semantic_scholar_search_title,
)
from .models import (
    BibEntry,
    CitationContext,
    ClaimType,
    L1Result,
    L3Evaluation,
    L3Result,
    L3Score,
)

logger = logging.getLogger(__name__)


def _openalex_inverted_index_to_text(index: object) -> str:
    """Best-effort reconstruction for OpenAlex `abstract_inverted_index`."""
    if not isinstance(index, dict):
        return ""

    pos_to_word: dict[int, str] = {}
    for word, positions in index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            return ""
        for p in positions:
            if not isinstance(p, int):
                return ""
            pos_to_word[p] = word

    if not pos_to_word:
        return ""

    max_pos = max(pos_to_word)
    words: list[str | None] = [None] * (max_pos + 1)
    for p, w in pos_to_word.items():
        if 0 <= p < len(words):
            words[p] = w

    return " ".join(w for w in words if isinstance(w, str) and w)


def _extract_abstract_text(value: object) -> str:
    """Normalize cached/remote abstract payloads to plain text."""
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""

    # Common structured shapes
    for k in ("abstract", "text", "paperAbstract"):
        v = value.get(k)
        if isinstance(v, str) and v.strip():
            return v

    # OpenAlex full work payload
    inv = value.get("abstract_inverted_index")
    text = _openalex_inverted_index_to_text(inv)
    if text.strip():
        return text

    # OpenAlex inverted index directly
    text = _openalex_inverted_index_to_text(value)
    if text.strip():
        return text

    return ""


# ---------------------------------------------------------------------------
# E3.1  Abstract retrieval
# ---------------------------------------------------------------------------


def _get_abstract(
    entry: BibEntry,
    l1: Optional[L1Result],
    cache_dir: Path,
) -> Optional[str]:
    """Try PubMed -> Semantic Scholar -> OpenAlex to get abstract."""
    cache_file = cache_dir / "abstracts" / f"{entry.bib_key}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        cached = data.get("abstract")
        if isinstance(cached, str) and cached.strip():
            return cached
        cached_text = _extract_abstract_text(cached)
        if cached_text.strip():
            return cached_text

    abstract: Optional[str] = None

    # --- PubMed (best for health/biomed) ---
    doi = (l1.resolved_doi if l1 else None) or entry.doi
    if doi:
        pmids = pubmed_search(doi, cache_dir)
        if pmids:
            pm_result = pubmed_fetch_abstract(pmids[0], cache_dir)
            if isinstance(pm_result, dict):
                abstract = (
                    pm_result.get("abstract") or pm_result.get("text") or str(pm_result)
                )
            elif isinstance(pm_result, str):
                abstract = pm_result

    if not abstract and entry.title:
        pmids = pubmed_search(entry.title[:200], cache_dir)
        if pmids:
            pm_result = pubmed_fetch_abstract(pmids[0], cache_dir)
            if isinstance(pm_result, dict):
                abstract = (
                    pm_result.get("abstract") or pm_result.get("text") or str(pm_result)
                )
            elif isinstance(pm_result, str):
                abstract = pm_result

    # --- Semantic Scholar (has TLDR + abstract) ---
    if not abstract:
        s2_data: Optional[dict] = None
        if doi:
            s2_data = semantic_scholar_get_by_doi(doi, cache_dir)
        if not s2_data and entry.title:
            s2_results = semantic_scholar_search_title(entry.title, cache_dir)
            if s2_results and isinstance(s2_results, list) and len(s2_results) > 0:
                s2_data = s2_results[0]
            elif isinstance(s2_results, dict):
                s2_data = s2_results
        if s2_data and isinstance(s2_data, dict):
            abs_val = s2_data.get("abstract")
            if isinstance(abs_val, str) and abs_val:
                abstract = abs_val
            tldr = s2_data.get("tldr")
            if tldr and isinstance(tldr, dict):
                tldr_text = tldr.get("text", "")
                if tldr_text and not abstract:
                    abstract = tldr_text

    # Cache result
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(
            {"bib_key": entry.bib_key, "abstract": abstract or ""}, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    return abstract if abstract else None


# ---------------------------------------------------------------------------
# E3.2  Claim-type classification (heuristic)
# ---------------------------------------------------------------------------

_CLAIM_PATTERNS: list[tuple[ClaimType, re.Pattern]] = [
    (
        ClaimType.DEFINE,
        re.compile(r"(defin[ei]|conceitu|segundo|de acordo com|conforme)", re.I),
    ),
    (
        ClaimType.SUPPORT_METHOD,
        re.compile(
            r"(m[eé]todo|t[eé]cnica|algoritmo|abordagem|estrat[eé]gia|proposto por)",
            re.I,
        ),
    ),
    (
        ClaimType.SUPPORT_FACT,
        re.compile(
            r"(demonstr|evidenci|observ|estim|revel|identific|aument|reduz)", re.I
        ),
    ),
    (
        ClaimType.ATTRIBUTE,
        re.compile(r"(propôs|desenvolve[ur]|introduzi|criou|elabor)", re.I),
    ),
    (
        ClaimType.DATA_SOURCE,
        re.compile(
            r"(base de dados|sistema de informa|SIM|Sinan|SIH|SITETB|GAL)", re.I
        ),
    ),
    (
        ClaimType.TOOL,
        re.compile(r"(software|ferramenta|pacote|biblioteca|OpenRecLink|scikit)", re.I),
    ),
    (
        ClaimType.CONTRAST,
        re.compile(
            r"(entretanto|contudo|por[eé]m|diferentemente|ao contr[aá]rio)", re.I
        ),
    ),
    (
        ClaimType.EXTEND,
        re.compile(r"(ampli|expan|esten|complementa|al[eé]m d[eo])", re.I),
    ),
]


def classify_claim(sentence: str) -> ClaimType:
    """Heuristic claim-type from citation sentence."""
    for ctype, pattern in _CLAIM_PATTERNS:
        if pattern.search(sentence):
            return ctype
    return ClaimType.BACKGROUND


# ---------------------------------------------------------------------------
# E3.3  LLM evaluation (optional — falls back to heuristic)
# ---------------------------------------------------------------------------


def _build_llm_prompt(
    abstract: str,
    sentence: str,
    paragraph: str,
    claim_type: ClaimType,
) -> str:
    """Build the LLM evaluation prompt. Shared by real-time and batch modes."""
    abs_text = str(abstract)[:2000] if abstract else ""
    sent_text = str(sentence)[:500] if sentence else ""
    para_text = str(paragraph)[:1000] if paragraph else ""

    return f"""You are a scientific citation auditor. Evaluate whether the ABSTRACT of a cited paper supports the CLAIM made in the citing thesis.

CLAIM TYPE: {claim_type.value}
THESIS SENTENCE (containing the citation): {sent_text}
THESIS PARAGRAPH (broader context): {para_text}
CITED PAPER ABSTRACT: {abs_text}

Rules:
- For BACKGROUND/DATA_SOURCE/TOOL claims: be lenient — topical relevance is sufficient.
- For SUPPORT_FACT/SUPPORT_METHOD claims: be strict — the abstract must contain evidence for the specific claim.
- For DEFINE/ATTRIBUTE claims: the abstract should mention the concept/contribution attributed.

Respond in JSON only:
{{"verdict": "SUPPORTS" | "PARTIALLY_SUPPORTS" | "DOES_NOT_SUPPORT" | "INCONCLUSIVE",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "justification": "one sentence explaining your verdict"}}"""


_VERDICT_MAP = {
    "SUPPORTS": L3Score.L3_PASS,
    "PARTIALLY_SUPPORTS": L3Score.L3_WARN_PARTIAL,
    "DOES_NOT_SUPPORT": L3Score.L3_FAIL_UNSUPPORTED,
    "INCONCLUSIVE": L3Score.L3_WARN_INCONCLUSIVE,
}


def _parse_llm_response(content: str, claim_type: ClaimType) -> L3Evaluation:
    """Parse LLM JSON response into L3Evaluation."""
    result = json.loads(content)
    return L3Evaluation(
        score=_VERDICT_MAP.get(result.get("verdict", ""), L3Score.L3_WARN_INCONCLUSIVE),
        confidence=result.get("confidence", "LOW"),
        justification=result.get("justification", ""),
        claim_type=claim_type,
    )


def _llm_evaluate(
    abstract: str,
    sentence: str,
    paragraph: str,
    claim_type: ClaimType,
) -> L3Evaluation:
    """Use LLM to evaluate if abstract supports the citation claim.

    Falls back to heuristic if no LLM configured.
    """
    # Try LLM if configured
    if config.LLM_API_KEY:
        return _llm_evaluate_openai(abstract, sentence, paragraph, claim_type)

    # Heuristic fallback: keyword overlap
    return _heuristic_evaluate(abstract, sentence, claim_type)


def _llm_evaluate_openai(
    abstract: str,
    sentence: str,
    paragraph: str,
    claim_type: ClaimType,
) -> L3Evaluation:
    """Call OpenAI-compatible API for evaluation."""
    try:
        import httpx

        prompt = _build_llm_prompt(abstract, sentence, paragraph, claim_type)

        base_url = config.LLM_BASE_URL.rstrip("/")
        url = f"{base_url}/chat/completions"

        resp = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {config.LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 2048,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "response_format": {"type": "json_object"},
                # Fireworks specific
                "top_k": 40,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return _parse_llm_response(content, claim_type)
    except Exception as exc:
        logger.warning("LLM evaluation failed for claim: %s", exc)
        return _heuristic_evaluate(abstract, sentence, claim_type)


def _heuristic_evaluate(
    abstract: object,
    sentence: str,
    claim_type: ClaimType,
) -> L3Evaluation:
    """Keyword-overlap heuristic when LLM is unavailable."""
    if not isinstance(abstract, str):
        abstract = _extract_abstract_text(abstract) or ""

    # Extract meaningful words from sentence (>4 chars, not stopwords)
    stop = {
        "para",
        "como",
        "pela",
        "pelo",
        "pode",
        "sido",
        "mais",
        "sobre",
        "este",
        "esta",
        "estes",
        "estas",
        "entre",
        "outros",
        "outras",
        "desde",
        "ainda",
        "sendo",
        "foram",
        "have",
        "been",
        "with",
        "that",
        "from",
        "their",
        "which",
        "these",
        "those",
        "into",
    }
    sent_words = {
        w.lower() for w in re.findall(r"\b\w{4,}\b", sentence) if w.lower() not in stop
    }
    abs_words = {
        w.lower() for w in re.findall(r"\b\w{4,}\b", abstract) if w.lower() not in stop
    }

    if not sent_words:
        return L3Evaluation(
            score=L3Score.L3_WARN_INCONCLUSIVE,
            confidence="LOW",
            justification="No meaningful keywords in citation sentence.",
            claim_type=claim_type,
        )

    overlap = len(sent_words & abs_words) / len(sent_words)

    # Lenient threshold for background/tool/data_source
    lenient = claim_type in (
        ClaimType.BACKGROUND,
        ClaimType.TOOL,
        ClaimType.DATA_SOURCE,
    )
    threshold_pass = 0.25 if lenient else 0.35
    threshold_partial = 0.15 if lenient else 0.20

    if overlap >= threshold_pass:
        score = L3Score.L3_PASS
    elif overlap >= threshold_partial:
        score = L3Score.L3_WARN_PARTIAL
    else:
        score = L3Score.L3_WARN_INCONCLUSIVE  # heuristic can't confidently FAIL

    return L3Evaluation(
        score=score,
        confidence="LOW",
        justification=f"Heuristic: {overlap:.0%} keyword overlap ({len(sent_words & abs_words)}/{len(sent_words)} words).",
        claim_type=claim_type,
    )


# ---------------------------------------------------------------------------
# E3.4  Main: evaluate all references (async concurrent)
# ---------------------------------------------------------------------------


def evaluate_reference(
    entry: BibEntry,
    l1: Optional[L1Result],
    contexts: list[CitationContext],
    cache_dir: Path,
) -> L3Result:
    """Full L3 evaluation for one reference."""
    abstract = _get_abstract(entry, l1, cache_dir)

    if not abstract:
        return L3Result(
            bib_key=entry.bib_key,
            score=L3Score.L3_NA,
            abstract_available=False,
            evaluations=[],
            notes=["No abstract found via PubMed or Semantic Scholar."],
        )

    evaluations: list[L3Evaluation] = []
    for ctx in contexts:
        claim_type = classify_claim(ctx.sentence)
        ev = _llm_evaluate(abstract, ctx.sentence, ctx.paragraph, claim_type)
        evaluations.append(ev)

    if not evaluations:
        return L3Result(
            bib_key=entry.bib_key,
            score=L3Score.L3_NA,
            abstract_available=True,
            evaluations=[],
            notes=["Abstract found but no citation contexts to evaluate."],
        )

    # Worst-case aggregation
    score_priority = [
        L3Score.L3_FAIL_UNSUPPORTED,
        L3Score.L3_WARN_INCONCLUSIVE,
        L3Score.L3_WARN_PARTIAL,
        L3Score.L3_PASS,
    ]
    worst = L3Score.L3_PASS
    for ev in evaluations:
        if score_priority.index(ev.score) < score_priority.index(worst):
            worst = ev.score

    return L3Result(
        bib_key=entry.bib_key,
        score=worst,
        abstract_available=True,
        evaluations=evaluations,
        notes=[],
    )


# ---------------------------------------------------------------------------
# Async LLM evaluation using Fireworks SDK
# ---------------------------------------------------------------------------


async def _async_llm_evaluate(
    client,  # AsyncFireworks
    semaphore: asyncio.Semaphore,
    custom_id: str,
    prompt: str,
    claim_type: ClaimType,
) -> tuple[str, L3Evaluation]:
    """Single async LLM evaluation with semaphore control."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            ev = _parse_llm_response(content, claim_type)
            return custom_id, ev
        except Exception as exc:
            logger.warning("Async LLM eval failed for %s: %s", custom_id, exc)
            return custom_id, L3Evaluation(
                score=L3Score.L3_WARN_INCONCLUSIVE,
                confidence="LOW",
                justification=f"LLM call failed: {exc}",
                claim_type=claim_type,
            )


async def _async_run_e3_core(
    entries_with_data: list[tuple[str, str, list[tuple[int, str, str, ClaimType]]]],
    max_concurrent: int = 50,
) -> dict[str, list[tuple[int, L3Evaluation]]]:
    """Fire all LLM requests concurrently using AsyncFireworks.

    Args:
        entries_with_data: list of (bib_key, abstract, [(ctx_idx, sentence, paragraph, claim_type)])
        max_concurrent: semaphore limit for concurrent requests

    Returns:
        dict of bib_key → [(ctx_idx, L3Evaluation)]
    """
    from fireworks import AsyncFireworks

    client = AsyncFireworks(
        api_key=config.LLM_API_KEY,
        max_retries=3,
    )
    semaphore = asyncio.Semaphore(max_concurrent)

    # Build all tasks
    tasks = []
    task_meta: dict[str, tuple[str, int]] = {}  # custom_id → (bib_key, ctx_idx)

    for bib_key, abstract, ctx_data in entries_with_data:
        for ctx_idx, sentence, paragraph, claim_type in ctx_data:
            custom_id = f"{bib_key}__ctx{ctx_idx}"
            prompt = _build_llm_prompt(abstract, sentence, paragraph, claim_type)
            task_meta[custom_id] = (bib_key, ctx_idx)
            tasks.append(
                _async_llm_evaluate(client, semaphore, custom_id, prompt, claim_type)
            )

    logger.info(
        "E3-ASYNC: Firing %d LLM requests concurrently (max_concurrent=%d)",
        len(tasks),
        max_concurrent,
    )
    start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start
    logger.info(
        "E3-ASYNC: All %d requests completed in %.1fs (%.1f req/s)",
        len(results),
        elapsed,
        len(results) / elapsed if elapsed > 0 else 0,
    )

    # Group by bib_key
    evals_by_key: dict[str, list[tuple[int, L3Evaluation]]] = {}
    for custom_id, ev in results:
        bib_key, ctx_idx = task_meta[custom_id]
        evals_by_key.setdefault(bib_key, []).append((ctx_idx, ev))

    return evals_by_key


def run_e3(
    entries: list[BibEntry],
    l1_results: dict[str, L1Result],
    all_contexts: list[CitationContext],
    output_dir: Path,
    max_concurrent: int = 50,
) -> dict[str, L3Result]:
    """Run L3 for all entries using async concurrent LLM calls.

    Uses AsyncFireworks SDK for optimal connection pooling and throughput.
    Falls back to heuristic if no LLM_API_KEY configured.
    """
    # Group contexts by bib_key
    ctx_by_key: dict[str, list[CitationContext]] = {}
    for c in all_contexts:
        ctx_by_key.setdefault(c.bib_key, []).append(c)

    results: dict[str, L3Result] = {}

    # Phase 1: Collect abstracts (sync — cached I/O)
    entries_with_data: list[tuple[str, str, list[tuple[int, str, str, ClaimType]]]] = []
    for entry in entries:
        ctxs = ctx_by_key.get(entry.bib_key, [])
        if not ctxs:
            results[entry.bib_key] = L3Result(
                bib_key=entry.bib_key,
                score=L3Score.L3_NA,
                abstract_available=False,
                evaluations=[],
                notes=["Not cited in any .tex file."],
            )
            continue

        abstract = _get_abstract(entry, l1_results.get(entry.bib_key), config.CACHE_DIR)
        if not abstract:
            results[entry.bib_key] = L3Result(
                bib_key=entry.bib_key,
                score=L3Score.L3_NA,
                abstract_available=False,
                evaluations=[],
                notes=["No abstract found via PubMed or Semantic Scholar."],
            )
            continue

        ctx_data = []
        for i, ctx in enumerate(ctxs):
            claim_type = classify_claim(ctx.sentence)
            ctx_data.append((i, ctx.sentence, ctx.paragraph, claim_type))
        entries_with_data.append((entry.bib_key, abstract, ctx_data))

    # Phase 2: Async LLM evaluation
    if entries_with_data and config.LLM_API_KEY:
        total_requests = sum(len(cd) for _, _, cd in entries_with_data)
        logger.info(
            "E3: %d LLM requests for %d refs (async concurrent)",
            total_requests,
            len(entries_with_data),
        )
        evals_by_key = asyncio.run(
            _async_run_e3_core(entries_with_data, max_concurrent)
        )
    elif entries_with_data:
        # No LLM key — heuristic fallback for all
        logger.warning("E3: No LLM_API_KEY — using heuristic for all entries.")
        evals_by_key = {}
        for bib_key, abstract, ctx_data in entries_with_data:
            evals = []
            for ctx_idx, sentence, _para, claim_type in ctx_data:
                ev = _heuristic_evaluate(abstract, sentence, claim_type)
                evals.append((ctx_idx, ev))
            evals_by_key[bib_key] = evals
    else:
        evals_by_key = {}

    # Phase 3: Aggregate per entry
    score_priority = [
        L3Score.L3_FAIL_UNSUPPORTED,
        L3Score.L3_WARN_INCONCLUSIVE,
        L3Score.L3_WARN_PARTIAL,
        L3Score.L3_PASS,
    ]
    for bib_key, evals_with_idx in evals_by_key.items():
        evaluations = [ev for _, ev in sorted(evals_with_idx, key=lambda x: x[0])]
        worst = L3Score.L3_PASS
        for ev in evaluations:
            if score_priority.index(ev.score) < score_priority.index(worst):
                worst = ev.score
        results[bib_key] = L3Result(
            bib_key=bib_key,
            score=worst,
            abstract_available=True,
            evaluations=evaluations,
            notes=[],
        )
        logger.info(
            "  E3 done: %s → %s (%d evals)",
            bib_key,
            worst.name,
            len(evaluations),
        )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "l3_results.json"
    serializable = {k: v.to_dict() for k, v in results.items()}
    out_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("E3 results saved to %s", out_path)
    return results


# ---------------------------------------------------------------------------
# E3.5  Batch mode (Fireworks AI batch inference — 50% cheaper)
# ---------------------------------------------------------------------------


def run_e3_batch(
    entries: list[BibEntry],
    l1_results: dict[str, L1Result],
    all_contexts: list[CitationContext],
    output_dir: Path,
) -> dict[str, L3Result]:
    """Run L3 for all entries using Fireworks batch inference.

    Same logic as run_e3() but collects all LLM requests into a single
    batch job for 50% cost savings.
    """
    from .batch_inference import BatchRequest, submit_batch

    # Group contexts by bib_key
    ctx_by_key: dict[str, list[CitationContext]] = {}
    for c in all_contexts:
        ctx_by_key.setdefault(c.bib_key, []).append(c)

    # Phase 1: Collect abstracts and build batch requests
    logger.info("E3-BATCH: Collecting abstracts and building requests...")
    batch_requests: list[BatchRequest] = []
    # Track mapping: custom_id → (bib_key, ctx_index, claim_type)
    request_meta: dict[str, tuple[str, int, ClaimType]] = {}
    # Track which entries have abstracts
    abstracts_found: dict[str, str] = {}
    entries_without_abstract: set[str] = set()

    for entry in entries:
        ctxs = ctx_by_key.get(entry.bib_key, [])
        if not ctxs:
            continue

        abstract = _get_abstract(entry, l1_results.get(entry.bib_key), config.CACHE_DIR)
        if not abstract:
            entries_without_abstract.add(entry.bib_key)
            continue

        abstracts_found[entry.bib_key] = abstract
        for i, ctx in enumerate(ctxs):
            claim_type = classify_claim(ctx.sentence)
            prompt = _build_llm_prompt(
                abstract, ctx.sentence, ctx.paragraph, claim_type
            )
            custom_id = f"{entry.bib_key}__ctx{i}"
            batch_requests.append(
                BatchRequest(
                    custom_id=custom_id,
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            request_meta[custom_id] = (entry.bib_key, i, claim_type)

    logger.info(
        "E3-BATCH: %d requests for %d refs (%d refs without abstract)",
        len(batch_requests),
        len(abstracts_found),
        len(entries_without_abstract),
    )

    # Phase 2: Submit batch and wait for results
    results_map: dict[str, str] = {}
    if batch_requests:
        if not config.LLM_API_KEY:
            logger.warning(
                "E3-BATCH: No LLM_API_KEY — falling back to heuristic for all."
            )
        else:
            results_map = submit_batch(batch_requests)

    # Phase 3: Parse results into L3Results
    results: dict[str, L3Result] = {}

    # Entries with no contexts
    for entry in entries:
        if entry.bib_key not in ctx_by_key or not ctx_by_key[entry.bib_key]:
            results[entry.bib_key] = L3Result(
                bib_key=entry.bib_key,
                score=L3Score.L3_NA,
                abstract_available=False,
                evaluations=[],
                notes=["Not cited in any .tex file."],
            )

    # Entries without abstracts
    for bib_key in entries_without_abstract:
        results[bib_key] = L3Result(
            bib_key=bib_key,
            score=L3Score.L3_NA,
            abstract_available=False,
            evaluations=[],
            notes=["No abstract found via PubMed or Semantic Scholar."],
        )

    # Entries with batch results
    evals_by_key: dict[str, list[L3Evaluation]] = {}
    for custom_id, (bib_key, ctx_idx, claim_type) in request_meta.items():
        content = results_map.get(custom_id)
        if content:
            try:
                ev = _parse_llm_response(content, claim_type)
            except Exception as exc:
                logger.warning(
                    "Failed to parse batch result for %s: %s", custom_id, exc
                )
                ctxs = ctx_by_key[bib_key]
                abstract = abstracts_found[bib_key]
                ev = _heuristic_evaluate(abstract, ctxs[ctx_idx].sentence, claim_type)
        else:
            # No batch result — use heuristic fallback
            ctxs = ctx_by_key[bib_key]
            abstract = abstracts_found.get(bib_key, "")
            ev = _heuristic_evaluate(abstract, ctxs[ctx_idx].sentence, claim_type)
        evals_by_key.setdefault(bib_key, []).append(ev)

    # Aggregate evaluations per entry
    score_priority = [
        L3Score.L3_FAIL_UNSUPPORTED,
        L3Score.L3_WARN_INCONCLUSIVE,
        L3Score.L3_WARN_PARTIAL,
        L3Score.L3_PASS,
    ]
    for bib_key, evaluations in evals_by_key.items():
        worst = L3Score.L3_PASS
        for ev in evaluations:
            if score_priority.index(ev.score) < score_priority.index(worst):
                worst = ev.score
        results[bib_key] = L3Result(
            bib_key=bib_key,
            score=worst,
            abstract_available=True,
            evaluations=evaluations,
            notes=[],
        )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "l3_results.json"
    serializable = {k: v.to_dict() for k, v in results.items()}
    out_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("E3-BATCH results saved to %s", out_path)
    return results
