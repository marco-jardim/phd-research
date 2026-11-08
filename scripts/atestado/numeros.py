"""Conferência mecânica de números (sem LLM): sinaliza números da sentença ausentes da evidência.

Os números da sentença estão em formato brasileiro (``39,3``; ``84.368``; ``21,4%``). A evidência pode
trazer o mesmo valor em formato brasileiro ou inglês (``39.3``; ``84,368``), sem separador de milhar ou
com espaço. A busca é por valor exato; arredondamento (92 para 92,4) aparece como não encontrado.
Isto é um sinal para o revisor, nunca um veredito.

Inteiros de um dígito sem decimal nem porcentagem são ignorados (ruído: "um", "2 grupos").
"""

from __future__ import annotations

import re

from .common import normalize_text

_NUM = re.compile(
    r"(?<![^\W\d_])(?<![^\W\d_]-)(?<!\d)(?<!\d[.,])"
    r"(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?)(\s?%)?"
    r"(?!\d)(?![.,]\d)(?![^\W\d_])"
)


def extract_numbers(sentence: str) -> list[tuple[str, str, str]]:
    """Números da sentença como ``(texto, parte_inteira, parte_decimal)``, sem repetições."""
    found: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for match in _NUM.finditer(sentence):
        token, percent = match.group(1), match.group(2)
        integer, _, decimals = token.partition(",")
        integer = integer.replace(".", "")
        if not decimals and not percent and len(integer) < 2:
            continue
        text = token + ("%" if percent else "")
        if text in seen:
            continue
        seen.add(text)
        found.append((text, integer, decimals))
    return found


def _group(digits: str, sep: str) -> str:
    if len(digits) <= 3:
        return digits
    head = len(digits) % 3 or 3
    parts = [digits[:head]] + [digits[i : i + 3] for i in range(head, len(digits), 3)]
    return sep.join(parts)


def surface_forms(integer: str, decimals: str) -> set[str]:
    """Grafias plausíveis do mesmo valor em português, inglês, sem agrupamento ou com espaço."""
    forms: set[str] = set()
    for group_sep in ("", ".", ",", " "):
        head = _group(integer, group_sep) if group_sep else integer
        if not decimals:
            forms.add(head)
            continue
        for dec_sep in (",", "."):
            if group_sep == dec_sep:
                continue
            forms.add(f"{head}{dec_sep}{decimals}")
    return forms


def prepare_evidence(evidence_text: str) -> str:
    return normalize_text(evidence_text)


def _present(form: str, evidence_norm: str) -> bool:
    pattern = r"(?<!\d)(?<!\d[.,])" + re.escape(form) + r"(?!\d)(?![.,]\d)"
    return re.search(pattern, evidence_norm) is not None


def numeros_nao_encontrados(sentence: str, evidence_norm: str) -> list[str]:
    """Números da sentença sem nenhuma grafia equivalente na evidência (já normalizada)."""
    if not evidence_norm:
        return []
    missing: list[str] = []
    for text, integer, decimals in extract_numbers(sentence):
        if not any(_present(form, evidence_norm) for form in surface_forms(integer, decimals)):
            missing.append(text)
    return missing
