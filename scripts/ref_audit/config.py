"""Configuração do pipeline de auditoria."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Carrega .env do diretório raiz do projeto
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BIB_FILE = _PROJECT_ROOT / "text" / "latex" / "referencias.bib"
TEX_DIR = _PROJECT_ROOT / "text" / "latex"
CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"
CONSENSUS_DIR = Path(__file__).resolve().parent / "data" / "consensus_results"
FULLTEXT_CACHE_DIR = Path(__file__).resolve().parent / "data" / "fulltext"
OUTPUT_DIR = _PROJECT_ROOT / "docs" / "support_text"

# ---------------------------------------------------------------------------
# APIs
# ---------------------------------------------------------------------------
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_ACCOUNT_ID = os.getenv("FIREWORKS_ACCOUNT_ID", "")
LLM_API_KEY = os.getenv("LLM_API_KEY") or FIREWORKS_API_KEY or OPENAI_API_KEY

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

default_base = "https://api.openai.com/v1"
if LLM_PROVIDER == "fireworks":
    default_base = "https://api.fireworks.ai/inference/v1"
    if not LLM_MODEL or LLM_MODEL == "gpt-4o":
        LLM_MODEL = (
            "accounts/fireworks/models/kimi-k2p5"  # Correct slug provided by user
        )

LLM_BASE_URL = os.getenv("LLM_BASE_URL", default_base)

# Fulltext layer (private)
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "")
CORE_API_KEY = os.getenv("CORE_API_KEY", "")
ENABLE_ANNAS_ARCHIVE = os.getenv("ENABLE_ANNAS_ARCHIVE", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Limiares
# ---------------------------------------------------------------------------
TITLE_SIMILARITY_THRESHOLD = 85.0
AUTHOR_SIMILARITY_THRESHOLD = 90.0
JOURNAL_SIMILARITY_THRESHOLD = 80.0
YEAR_TOLERANCE = 1
OPENALEX_TITLE_THRESHOLD = 90.0

# Rate limits (requests per second)
CROSSREF_RPS = 10.0  # conservative
SEMANTIC_SCHOLAR_RPS = 1.5  # 100 per 5 min
PUBMED_RPS = 3.0
OPENALEX_RPS = 10.0

# Stale threshold
STALE_YEAR = 2005
STALE_MIN_CITATIONS = 200

# Self-cite threshold
SELF_CITE_THRESHOLD = 0.20
CLUSTER_IMBALANCE_THRESHOLD = 0.30
LOW_DIVERSITY_THRESHOLD = 0.50
