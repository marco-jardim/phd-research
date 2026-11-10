"""Utilitários comuns da atestação bibliográfica (sem rede e sem API)."""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_ADJ_MODEL = "claude-opus-5-5"
DEFAULT_QUERY_MODEL = "claude-haiku-5-5"
DEFAULT_OUT = ".local/biblio"
DEFAULT_BIB = "text/latex/referencias.bib"
ESCOPO_TESE = "tese"
TESE_LABEL = "Tese completa"


# ---------------------------------------------------------------------------
# Caminhos e arquivos
# ---------------------------------------------------------------------------
def resolve_path(value: str | Path) -> Path:
    """Caminho absoluto; caminhos relativos são relativos à raiz do repositório."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def slug(text: str) -> str:
    """Slug ASCII em minúsculas (``Introdução`` vira ``introducao``)."""
    plain = strip_accents(text).casefold()
    plain = re.sub(r"[^a-z0-9]+", "-", plain).strip("-")
    return plain or "capitulo"


def key_filename(key: str) -> str:
    """Nome de arquivo seguro (Windows) para uma chave do .bib."""
    return re.sub(r"[^\w.\-]", "_", key)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def replace_with_retry(
    src: Path, dst: Path, *, retries: int = 8, sleep: Callable[[float], None] = time.sleep
) -> None:
    """``os.replace`` com backoff: no Windows o destino pode estar bloqueado."""
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            sleep(0.1 * 2**attempt)


def write_text_atomic(
    path: Path, text: str, *, sleep: Callable[[float], None] = time.sleep
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    replace_with_retry(tmp, path, sleep=sleep)


def write_json_atomic(path: Path, obj, *, sleep: Callable[[float], None] = time.sleep) -> None:
    write_text_atomic(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n", sleep=sleep)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    write_text_atomic(path, "\n".join(lines) + ("\n" if lines else ""))


# ---------------------------------------------------------------------------
# Normalização de texto
# ---------------------------------------------------------------------------
_QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u00b4": "'",
        "\u0060": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\u00ab": '"',
        "\u00bb": '"',
    }
)
_DASH_MAP = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\ufe58": "-",
        "\ufe63": "-",
        "\uff0d": "-",
    }
)
_LINEBREAK_HYPHEN = re.compile(r"(?<=[^\W\d_])-[ \t]*\r?\n\s*(?=[^\W\d_])")
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Forma canônica para comparar trechos: NFKC, aspas e travessões unificados,
    sem hífen suave nem hifenização de fim de linha, em minúsculas, espaços colapsados."""
    out = unicodedata.normalize("NFKC", text)
    out = out.replace("\u00ad", "")
    out = out.translate(_QUOTE_MAP).translate(_DASH_MAP)
    out = _LINEBREAK_HYPHEN.sub("", out)
    out = out.casefold()
    return _WS.sub(" ", out).strip()


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


_LATEX_ACCENT = re.compile(r"\\[`'^\"~=.]\s*\{?\\?([A-Za-z])\}?")
_LATEX_ACCENT_ALPHA = re.compile(r"\\[uvHcdbkr](?:\s*\{([A-Za-z])\}| ([A-Za-z])(?![A-Za-z]))")
_LATEX_CMD = re.compile(r"\\[A-Za-z]+\s*")


def clean_latex(text: str) -> str:
    """Remove acentos e comandos LaTeX comuns de campos do .bib."""
    out = _LATEX_ACCENT.sub(r"\1", text)
    out = _LATEX_ACCENT_ALPHA.sub(lambda m: m.group(1) or m.group(2), out)
    out = out.replace("\\&", "&").replace("\\_", "_").replace("\\%", "%")
    out = _LATEX_CMD.sub("", out)
    out = out.replace("{", "").replace("}", "").replace("~", " ")
    return _WS.sub(" ", out).strip()


def normalize_title(text: str) -> str:
    """Título comparável: sem LaTeX, sem acentos, minúsculo, só letras e dígitos."""
    plain = strip_accents(clean_latex(text)).casefold()
    plain = re.sub(r"[^a-z0-9]+", " ", plain)
    return plain.strip()


def title_similarity(a: str, b: str) -> float:
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def normalize_doi(raw: str | None) -> str:
    if not raw:
        return ""
    doi = clean_latex(raw) if "{" in raw else raw.strip()
    doi = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    return doi.strip().lower()


def truncate_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]) + "…"


# ---------------------------------------------------------------------------
# Chave da API (nunca impressa, registrada ou gravada)
# ---------------------------------------------------------------------------
def load_env_var(name: str, env_file: Path | None = None) -> str:
    """Valor de ``name`` em ``os.environ`` ou no ``.env``; vazio se ausente."""
    if os.environ.get(name):
        return os.environ[name]
    path = env_file if env_file is not None else ROOT / ".env"
    if not path.is_file():
        return ""
    try:
        from dotenv import dotenv_values
    except ImportError:
        return _parse_env_file(path).get(name, "")
    return str(dotenv_values(path).get(name) or "")


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        name, sep, value = line.partition("=")
        if sep:
            values[name.strip()] = value.strip().strip('"').strip("'")
    return values
