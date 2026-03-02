"""E0.2 — Extrator de contextos de citação dos .tex.

Busca \\cite{key} em cada arquivo .tex, extrai frase, parágrafo e seção
mais próxima. Detecta cites órfãs e entradas .bib não citadas.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import CitationContext


def _find_cites(text: str) -> list[tuple[int, list[str]]]:
    """Encontra todas as \\cite*{key1,key2} variantes e retorna (posição, [keys])."""
    results = []
    # Match \cite, \citeonline, \citet, \citep, \citeauthor, etc.
    # with optional [...]  arguments (up to 2)
    for m in re.finditer(
        r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]+)\}", text
    ):
        keys = [k.strip() for k in m.group(1).split(",")]
        results.append((m.start(), keys))
    return results


def _get_line_number(text: str, pos: int) -> int:
    """Retorna número da linha (1-based) para posição no texto."""
    return text[:pos].count("\n") + 1


def _get_sentence(text: str, pos: int) -> str:
    """Extrai frase ao redor da posição (entre pontos finais)."""
    # Procura o ponto anterior
    start = text.rfind(".", 0, pos)
    start = start + 1 if start != -1 else 0
    # Procura o ponto posterior
    end = text.find(".", pos)
    end = end + 1 if end != -1 else len(text)
    sentence = text[start:end].strip()
    # Remove quebras de linha
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence[:500]  # limita tamanho


def _get_paragraph(text: str, pos: int) -> str:
    """Extrai parágrafo (entre linhas em branco)."""
    # Procura linha em branco anterior
    start = text.rfind("\n\n", 0, pos)
    start = start + 2 if start != -1 else 0
    # Procura linha em branco posterior
    end = text.find("\n\n", pos)
    end = end if end != -1 else len(text)
    para = text[start:end].strip()
    para = re.sub(r"\s+", " ", para)
    return para[:1500]  # limita tamanho


def _get_section(text: str, pos: int) -> str | None:
    """Encontra \\section ou \\subsection mais próxima antes da posição."""
    pattern = r"\\(?:section|subsection|subsubsection)\{([^}]+)\}"
    last_match = None
    for m in re.finditer(pattern, text[:pos]):
        last_match = m
    if last_match:
        return last_match.group(1).strip()
    return None


def extract_contexts(
    tex_dir: Path,
    bib_keys: set[str],
) -> tuple[list[CitationContext], list[str], list[str]]:
    """Extrai contextos de citação de todos os .tex.

    Returns:
        (contexts, orphan_cites, uncited_bib_keys)
    """
    # Scan all .tex files, excluding build artifacts and style files
    _exclude = {"setspace.sty"}
    tex_files = sorted(
        p for p in tex_dir.glob("*.tex")
        if p.name not in _exclude
    )
    all_contexts: list[CitationContext] = []
    cited_keys: set[str] = set()

    for tex_file in tex_files:
        text = tex_file.read_text(encoding="utf-8")
        cites = _find_cites(text)

        for pos, keys in cites:
            line_no = _get_line_number(text, pos)
            sentence = _get_sentence(text, pos)
            paragraph = _get_paragraph(text, pos)
            section = _get_section(text, pos)

            for key in keys:
                cited_keys.add(key)
                ctx = CitationContext(
                    bib_key=key,
                    tex_file=tex_file.name,
                    line_number=line_no,
                    sentence=sentence,
                    paragraph=paragraph,
                    section=section,
                )
                all_contexts.append(ctx)

    # Cites a chaves que não existem no .bib
    orphan_cites = sorted(cited_keys - bib_keys)
    # Entradas .bib que nunca são citadas
    uncited_bib = sorted(bib_keys - cited_keys)

    return all_contexts, orphan_cites, uncited_bib


def save_contexts(
    contexts: list[CitationContext],
    orphan_cites: list[str],
    uncited_bib: list[str],
    output_dir: Path,
) -> None:
    """Salva contextos e diagnósticos como JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "contexts": [c.to_dict() for c in contexts],
        "orphan_cites": orphan_cites,
        "uncited_bib_entries": uncited_bib,
        "stats": {
            "total_contexts": len(contexts),
            "unique_cited_keys": len({c.bib_key for c in contexts}),
            "orphan_cite_count": len(orphan_cites),
            "uncited_bib_count": len(uncited_bib),
        },
    }
    (output_dir / "citation_contexts.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_e0_contexts(
    tex_dir: Path,
    bib_keys: set[str],
    output_dir: Path,
) -> tuple[list[CitationContext], list[str], list[str]]:
    """Executa E0.2 completo."""
    contexts, orphan_cites, uncited_bib = extract_contexts(tex_dir, bib_keys)
    save_contexts(contexts, orphan_cites, uncited_bib, output_dir)
    return contexts, orphan_cites, uncited_bib
