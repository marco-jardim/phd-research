"""Verificação mecânica dos trechos, regras da segunda passada e veredito final."""

from __future__ import annotations

import re

from .common import normalize_text

VERDICTS = ("SUSTENTA", "SUSTENTA_PARCIALMENTE", "NAO_SUSTENTA", "CONTRADIZ", "NAO_VERIFICAVEL")
PROBLEMS = (
    "afirma_mais_que_a_fonte",
    "atribuicao_errada",
    "numero_divergente",
    "fonte_secundaria",
    "fora_de_escopo",
    "outro",
)
NEEDS_QUOTE = {"SUSTENTA", "SUSTENTA_PARCIALMENTE", "CONTRADIZ"}
SECOND_PASS_VERDICTS = {"NAO_SUSTENTA", "CONTRADIZ"}
MAX_QUOTE_WORDS = 60
MIN_QUOTE_WORDS = 6
MIN_QUOTE_CHARS = 40

DIVERGENTE = "DIVERGENTE"


def _loose(text: str) -> str:
    """Forma tolerante: sem hífens e sem espaços (hifenização ambígua de PDFs)."""
    return re.sub(r"[-\s]+", "", text)


def verify_quote(quote: str, evidence_text: str) -> bool:
    """O trecho (normalizado) é substring do texto de evidência efetivamente enviado?"""
    nq = normalize_text(quote)
    if not nq:
        return False
    ne = normalize_text(evidence_text)
    if nq in ne:
        return True
    lq = _loose(nq)
    return bool(lq) and lq in _loose(ne)


def quote_long_enough(quote: str) -> bool:
    """Trecho com pelo menos ``MIN_QUOTE_WORDS`` palavras ou ``MIN_QUOTE_CHARS`` caracteres.

    Trechos mais curtos (uma sigla, um número) aparecem em qualquer fonte e não sustentam um veredito.
    """
    text = re.sub(r"\s+", " ", quote).strip()
    return len(text.split()) >= MIN_QUOTE_WORDS or len(text) >= MIN_QUOTE_CHARS


def make_assessment(raw: dict | None, evidence_text: str) -> dict:
    """Avaliação de uma passada com verificação mecânica do trecho.

    ``raw`` é o objeto devolvido pelo modelo para o contexto (None se o modelo não respondeu).
    """
    flags: list[str] = []
    if raw is None:
        return {
            "veredito_modelo": None, "veredito": "NAO_VERIFICAVEL", "trecho_fonte": "",
            "trecho_verificado": None, "justificativa": "", "problema": None,
            "nivel_modelo": None, "flags": ["sem_resposta_modelo"],
        }
    verdict_raw = str(raw.get("veredito") or "").strip().upper()
    if verdict_raw in VERDICTS:
        verdict = verdict_raw
    else:
        verdict = "NAO_VERIFICAVEL"
        flags.append("veredito_invalido")
    quote = re.sub(r"\s+", " ", str(raw.get("trecho_fonte") or "")).strip()
    if len(quote.split()) > MAX_QUOTE_WORDS:
        flags.append("trecho_longo")
    verified: bool | None
    if quote:
        verified = verify_quote(quote, evidence_text)
        if not verified:
            flags.append("trecho_nao_encontrado")
        elif verdict in NEEDS_QUOTE and not quote_long_enough(quote):
            verified = False
            flags.append("trecho_curto")
    elif verdict_raw in NEEDS_QUOTE:
        verified = False
        flags.append("trecho_ausente")
    else:
        verified = None
    effective = verdict
    if verified is False:
        effective = "NAO_VERIFICAVEL"
    problem = raw.get("problema")
    if problem is not None and problem not in PROBLEMS:
        problem = "outro"
    return {
        "veredito_modelo": verdict_raw or None,
        "veredito": effective,
        "trecho_fonte": quote,
        "trecho_verificado": verified,
        "justificativa": str(raw.get("justificativa") or "").strip(),
        "problema": problem,
        "nivel_modelo": raw.get("nivel_evidencia"),
        "flags": flags,
    }


def needs_second_pass(assessment: dict) -> bool:
    """Veredito crítico, parcial com problema, trecho não verificado ou resposta inválida."""
    verdict = assessment.get("veredito_modelo")
    if verdict in SECOND_PASS_VERDICTS:
        return True
    if verdict == "SUSTENTA_PARCIALMENTE" and assessment.get("problema"):
        return True
    if assessment.get("trecho_verificado") is False:
        return True
    return bool({"veredito_invalido", "sem_resposta_modelo"} & set(assessment.get("flags", [])))


def combine_passes(first: dict, second: dict | None) -> tuple[str, list[str]]:
    """Veredito final: concordância entre as passadas, senão DIVERGENTE (revisão humana)."""
    flags = list(first.get("flags", []))
    if not needs_second_pass(first):
        return first["veredito"], flags
    if second is None:
        flags.append("segunda_passada_pendente")
        return first["veredito"], flags
    flags += [f"p2:{flag}" for flag in second.get("flags", [])]
    if first["veredito"] == second["veredito"]:
        return first["veredito"], flags
    return DIVERGENTE, flags
