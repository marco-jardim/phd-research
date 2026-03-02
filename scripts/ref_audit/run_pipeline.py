"""
E6 — CLI unificada para o pipeline de auditoria de referências.

Uso:
    python -m scripts.ref_audit.run_pipeline [opções]

Opções:
    --layer L1|L2|L3|ALL     Executar até a camada especificada (default: ALL)
    --key <bib_key>          Auditar apenas uma referência específica
    --report-only            Gerar relatório a partir de resultados existentes
    --fulltext               Ativar camada full-text (requer módulo privado)
    --fulltext-only          Executar apenas full-text sobre L3 existente
    --verbose                Logging detalhado
    --no-cache               Ignorar cache de APIs
    --clean                  Limpar resultados anteriores
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from . import config
from .e0_parse_bib import parse_bib, run_e0_parse
from .e0_extract_contexts import run_e0_contexts
from .e1_existence import verify_entry
from .e2_journal import assess_journal, run_e2
from .e3_relevance import evaluate_reference, run_e3, run_e3_batch
from .e4_report import (
    apply_verified_overrides,
    build_composite_results,
    generate_markdown_report,
    generate_csv_report,
)
from .e5_flags import compute_flags
from .models import L1Score, L1Result, L2Result, L3Result

# Private fulltext module — graceful degradation
_HAS_FULLTEXT = False
FulltextFetcher = None  # type: ignore[assignment]
run_fulltext_l3 = None  # type: ignore[assignment]
try:
    from .private import is_available as _fulltext_available

    if _fulltext_available():
        from .private import FulltextFetcher as _FF, run_fulltext_l3 as _RFL3

        FulltextFetcher = _FF  # type: ignore[assignment]
        run_fulltext_l3 = _RFL3  # type: ignore[assignment]
        _HAS_FULLTEXT = True
    else:
        _HAS_FULLTEXT = False
except ImportError:
    _HAS_FULLTEXT = False


logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _clean_outputs() -> None:
    """Remove resultados anteriores e cache."""
    for d in [config.OUTPUT_DIR, config.CACHE_DIR]:
        if d.exists():
            shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            logger.info("Limpou: %s", d)


def _load_json(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _compute_fingerprint(entry: Any) -> str:
    """Compute a hash fingerprint for an entry based on its key identifying fields."""
    # Use DOI if available (most stable identifier)
    if hasattr(entry, "doi") and entry.doi:
        return hashlib.sha256(entry.doi.encode()).hexdigest()[:16]
    
    # Otherwise use title + first author + year
    content = "|".join([
        (getattr(entry, "title", "") or "").lower().strip(),
        (getattr(entry, "authors", [])[0] if getattr(entry, "authors", []) else "").lower().strip(),
        str(getattr(entry, "year", "") or ""),
    ])
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _load_cached_results(output_dir: Path) -> tuple[dict, dict, dict]:
    """Load previous L1, L2, L3 results from JSON if they exist."""
    l1_path = output_dir / "l1_results.json"
    l2_path = output_dir / "l2_results.json"
    l3_path = output_dir / "l3_results.json"
    
    l1_cache = {}
    l2_cache = {}
    l3_cache = {}
    
    if l1_path.exists():
        with open(l1_path, encoding="utf-8") as f:
            data = json.load(f)
            for k, v in data.items():
                l1_cache[k] = L1Result(**v)
    
    if l2_path.exists():
        with open(l2_path, encoding="utf-8") as f:
            data = json.load(f)
            for k, v in data.items():
                l2_cache[k] = L2Result(**v)
    
    if l3_path.exists():
        with open(l3_path, encoding="utf-8") as f:
            data = json.load(f)
            for k, v in data.items():
                l3_cache[k] = L3Result(**v)
    
    return l1_cache, l2_cache, l3_cache


def run_full_pipeline(
    layer: str = "ALL",
    single_key: str | None = None,
    report_only: bool = False,
    no_cache: bool = False,
    fulltext: bool = False,
    fulltext_only: bool = False,
    batch: bool = False,
) -> Path:
    """Executa o pipeline completo ou parcial com cache de resultados."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load previous results for cache lookup
    l1_cache, l2_cache, l3_cache = {}, {}, {}
    if not no_cache:
        l1_cache, l2_cache, l3_cache = _load_cached_results(config.OUTPUT_DIR)
        logger.info("Cache carregado: %d L1, %d L2, %d L3", len(l1_cache), len(l2_cache), len(l3_cache))

    # ── E0: Parse .bib e extrair contextos ──
    logger.info("═══ E0: Parsing .bib e extraindo contextos ═══")
    entries = run_e0_parse(config.BIB_FILE, config.OUTPUT_DIR)
    bib_keys = {e.bib_key for e in entries}
    contexts, orphan_cites, uncited_bib = run_e0_contexts(
        config.TEX_DIR, bib_keys, config.OUTPUT_DIR
    )
    
    # Build lookup for contexts
    contexts_by_key: dict[str, list] = {}
    for c in contexts:
        contexts_by_key.setdefault(c.bib_key, []).append(c)
    
    # Filter to single key if specified
    if single_key:
        entries = [e for e in entries if e.bib_key == single_key]
        contexts = [c for c in contexts if c.bib_key == single_key]
        # Clear caches for single-key mode to force reprocessing
        l1_cache, l2_cache, l3_cache = {}, {}, {}
    
    # Determine which entries need processing vs can use cache
    entries_to_process = []
    l1_results = {}
    l2_results = {}
    l3_results = {}
    
    for entry in entries:
        if no_cache or entry.bib_key not in l1_cache:
            entries_to_process.append(entry)
        else:
            # Cache hit - use cached results
            l1_results[entry.bib_key] = l1_cache[entry.bib_key]
            if entry.bib_key in l2_cache:
                l2_results[entry.bib_key] = l2_cache[entry.bib_key]
            if entry.bib_key in l3_cache:
                l3_results[entry.bib_key] = l3_cache[entry.bib_key]
    
    logger.info(
        "E0 concluído: %d entries, %d contexts, %d orphans, %d uncited, %d cache hits",
        len(entries), len(contexts), len(orphan_cites), len(uncited_bib),
        len(entries) - len(entries_to_process)
    )
    
    if report_only and not entries_to_process:
        # All results cached, just regenerate reports
        return _generate_reports(
            entries, contexts, l1_results, l2_results, l3_results, orphan_cites, uncited_bib
        )

    # ── E1: Existência (only for cache misses) ──
    if entries_to_process and layer in ("L1", "ALL"):
        logger.info("═══ E1: Verificação de existência (%d entradas) ═══", len(entries_to_process))
        for i, entry in enumerate(entries_to_process, 1):
            logger.info(
                "  [%d/%d] %s (doi=%s)", i, len(entries_to_process), entry.bib_key, entry.doi or "N/A"
            )
            l1_results[entry.bib_key] = verify_entry(entry, config.CACHE_DIR)
            time.sleep(1.0 / config.CROSSREF_RPS)
        
        # Save L1 results (merge with cache)
        l1_path = config.OUTPUT_DIR / "l1_results.json"
        with open(l1_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: v.to_dict() for k, v in l1_results.items()},
                f, indent=2, ensure_ascii=False
            )
        logger.info("E1 concluído: %s", {s.value: sum(1 for v in l1_results.values() if v.score == s) for s in L1Score})
    elif layer == "L1":
        return _generate_reports(entries, contexts, l1_results, {}, {}, orphan_cites, uncited_bib)

    # ── E2: Journal (only for entries that need it) ──
    if layer in ("L2", "ALL"):
        logger.info("═══ E2: Avaliação de journals ═══")
        entries_for_e2 = [e for e in entries_to_process if e.bib_key not in l2_cache]
        if entries_for_e2:
            new_l2 = run_e2(entries_for_e2, l1_results, config.OUTPUT_DIR)
            l2_results.update(new_l2)
            logger.info("E2 concluído: %d journals avaliados (novos)", len(new_l2))
        else:
            logger.info("E2: Todos os journals em cache (%d)", len(l2_results))
        
        if layer == "L2":
            return _generate_reports(entries, contexts, l1_results, l2_results, {}, orphan_cites, uncited_bib)

    # ── E3: Relevância (only for entries that need it) ──
    if layer in ("L3", "ALL"):
        logger.info("═══ E3: Avaliação de relevância ═══")
        entries_for_e3 = [e for e in entries_to_process if e.bib_key not in l3_cache]
        
        if entries_for_e3:
            if batch:
                logger.info("E3: Usando modo BATCH (Fireworks AI — 50%% mais barato)")
                new_l3 = run_e3_batch(entries_for_e3, l1_results, contexts, config.OUTPUT_DIR)
            else:
                new_l3 = run_e3(entries_for_e3, l1_results, contexts, config.OUTPUT_DIR)
            l3_results.update(new_l3)
            logger.info("E3 concluído: %d refs avaliadas (novas)", len(new_l3))
        else:
            logger.info("E3: Todas as refs em cache (%d)", len(l3_results))

    # ── Fulltext (private layer) ──
    fulltext_l3_results = {}
    if (fulltext or fulltext_only) and _HAS_FULLTEXT:
        logger.info("═══ FULLTEXT: Busca e re-avaliação com texto integral ═══")
        config.FULLTEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        assert FulltextFetcher is not None  # guarded by _HAS_FULLTEXT
        fetcher = FulltextFetcher(
            unpaywall_email=config.UNPAYWALL_EMAIL,
            core_api_key=config.CORE_API_KEY,
            enable_annas=config.ENABLE_ANNAS_ARCHIVE,
            cache_dir=config.FULLTEXT_CACHE_DIR,
        )
        # --- FULLTEXT fetching (parallel) ---
        logger.info("FULLTEXT: Buscando texto integral para %d refs", len(target_entries))
        ft_entries = [
            {
                "bib_key": entry.bib_key,
                "doi": entry.doi or "",
                "title": entry.title or "",
                "author": entry.authors[0] if entry.authors else "",
                "pmid": getattr(entry, "pmid", "") or "",
            }
            for entry in target_entries
        ]
        all_ft = fetcher.fetch_many(ft_entries, max_workers=4)
        fulltext_results = {k: v for k, v in all_ft.items() if v.available}

        # Re-evaluate L3 with full text
        if fulltext_results:
            logger.info(
                "FULLTEXT: Re-avaliando L3 com texto integral (%d refs)",
                len(fulltext_results),
            )
            # Build dicts expected by run_fulltext_l3
            ctx_by_key: dict[str, list] = {}
            for ctx in contexts:
                ctx_by_key.setdefault(ctx.bib_key, []).append(ctx)
            entry_by_key = {e.bib_key: e for e in entries}
            assert run_fulltext_l3 is not None  # guarded by _HAS_FULLTEXT
            fulltext_l3_results = run_fulltext_l3(
                fulltext_results=fulltext_results,
                l3_results=l3_results,
                contexts=ctx_by_key,
                entries=entry_by_key,
                api_key=config.LLM_API_KEY,
                model=config.LLM_MODEL,
                base_url=config.LLM_BASE_URL,
                only_problems=True,
            )
            # Merge upgrades into l3_results
            upgraded = 0
            for key, ft_result in fulltext_l3_results.items():
                if ft_result.upgraded and key in l3_results:
                    from .models import L3Score

                    l3_results[key].score = L3Score(ft_result.fulltext_l3_score)
                    l3_results[key].notes.append(
                        f"[FULLTEXT_UPGRADE: {ft_result.original_l3_score}"
                        f"→{ft_result.fulltext_l3_score} via {ft_result.fulltext_source}]"
                    )
                    upgraded += 1
            logger.info(
                "FULLTEXT concluído: %d textos obtidos, %d upgrades de L3",
                len(fulltext_results),
                upgraded,
            )
        else:
            logger.info("FULLTEXT: Nenhum texto integral obtido.")
    elif (fulltext or fulltext_only) and not _HAS_FULLTEXT:
        logger.warning(
            "FULLTEXT solicitado mas módulo privado não disponível. "
            "Verifique scripts/ref_audit/private/"
        )

    # ── E4+E5: Relatório ──
    return _generate_reports(
        entries, contexts, l1_results, l2_results, l3_results, orphan_cites, uncited_bib
    )


def _generate_reports(
    entries, contexts, l1_results, l2_results, l3_results, orphan_cites, uncited_bib
) -> Path:
    """E4+E5: Computa flags, constrói composites, gera relatórios."""
    logger.info("═══ E4+E5: Gerando relatório consolidado ═══")

    # Build entries/contexts lookup
    entries_map = {e.bib_key: e for e in entries}
    contexts_map: dict[str, list] = {}
    for c in contexts:
        contexts_map.setdefault(c.bib_key, []).append(c)

    # E5: Flags
    flags = compute_flags(
        entries, contexts, l1_results, l2_results, l3_results, orphan_cites, uncited_bib
    )
    logger.info("E5: %d flags geradas", len(flags))

    # E4: Composite
    composites = build_composite_results(
        entries, contexts, l1_results, l2_results, l3_results, flags
    )

    # Apply manual verification overrides
    overridden = apply_verified_overrides(composites)
    if overridden:
        logger.info(
            "Verificações manuais aplicadas: %d overrides", len(overridden)
        )

    # Generate outputs
    md_path = generate_markdown_report(composites, config.OUTPUT_DIR, overridden=overridden)
    csv_path = generate_csv_report(composites, config.OUTPUT_DIR)
    logger.info("Relatório MD: %s", md_path)
    logger.info("Relatório CSV: %s", csv_path)
    return md_path


def _report_only() -> Path:
    """Gera relatório a partir de JSONs existentes."""
    logger.info("Modo --report-only: carregando resultados existentes...")
    # This would load from existing JSONs — simplified for now
    raise NotImplementedError(
        "--report-only ainda não implementado. Execute o pipeline completo."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de auditoria de referências bibliográficas",
        prog="python -m scripts.ref_audit.run_pipeline",
    )
    parser.add_argument(
        "--layer",
        choices=["L1", "L2", "L3", "ALL"],
        default="ALL",
        help="Executar até a camada especificada (default: ALL)",
    )
    parser.add_argument("--key", type=str, help="Auditar apenas uma referência")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Gerar relatório de resultados existentes",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Logging detalhado"
    )
    parser.add_argument("--no-cache", action="store_true", help="Ignorar cache")
    parser.add_argument(
        "--clean", action="store_true", help="Limpar resultados anteriores"
    )
    parser.add_argument(
        "--fulltext",
        action="store_true",
        help="Ativar camada full-text (requer módulo privado)",
    )
    parser.add_argument(
        "--fulltext-only",
        action="store_true",
        help="Executar apenas camada full-text (pula L1-L3)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Usar Fireworks batch inference para L3 (50%% mais barato)",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.clean:
        _clean_outputs()

    report_path = run_full_pipeline(
        layer=args.layer,
        single_key=args.key,
        report_only=args.report_only,
        no_cache=args.no_cache,
        fulltext=args.fulltext,
        fulltext_only=args.fulltext_only,
        batch=args.batch,
    )
    print(f"\n✓ Auditoria concluída. Relatório: {report_path}")


if __name__ == "__main__":
    main()
