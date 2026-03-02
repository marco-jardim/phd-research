"""
PoC: 3-model LLM clerical review comparison
=============================================
Sprint 3A task L — compare Qwen3-235B, Kimi-K2.5, DeepSeek-R1-0528
on clerical review of grey-zone candidate pairs.

Usage:
    python scripts/poc_llm_3model_test.py [--n-pairs 5]
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

FIREWORKS_API_KEY = os.environ["FIREWORKS_API_KEY"]
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

MODELS = {
    "qwen3_235b": "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
    "kimi_k2.5": "accounts/fireworks/models/kimi-k2p5",
    "deepseek_r1": "accounts/fireworks/models/deepseek-r1-0528",
}

# Feature columns used to build dossiers (linkage subscores only — no PII)
NOME_COLS = [
    "NOME prim frag igual",
    "NOME ult frag igual",
    "NOME qtd frag iguais",
    "NOME qtd frag raros",
    "NOME qtd frag comuns",
    "NOME qtd frag muito parec",
    "NOME qtd frag abrev",
]
NOMEMAE_COLS = [
    "NOMEMAE prim frag igual",
    "NOMEMAE ult frag igual",
    "NOMEMAE qtd frag iguais",
    "NOMEMAE qtd frag raros",
    "NOMEMAE qtd frag comuns",
    "NOMEMAE qtd frag muito parec",
    "NOMEMAE qtd frag abrev",
]
DTNASC_COLS = [
    "DTNASC dt iguais",
    "DTNASC dt ap 1digi",
    "DTNASC dt inv dia",
    "DTNASC dt inv mes",
    "DTNASC dt inv ano",
]
CODMUN_COLS = [
    "CODMUNRES uf igual",
    "CODMUNRES uf prox",
    "CODMUNRES local igual",
    "CODMUNRES local prox",
]
ENDERECO_COLS = [
    "ENDERECO via igual",
    "ENDERECO via prox",
    "ENDERECO numero igual",
    "ENDERECO compl prox",
    "ENDERECO texto prox",
    "ENDERECO tokens jacc",
]
ALL_FEATURE_COLS = NOME_COLS + NOMEMAE_COLS + DTNASC_COLS + CODMUN_COLS + ENDERECO_COLS


# ── System prompt (from gzcmd_v3_llm_prompt.md) ──────────────────────
SYSTEM_PROMPT = """\
Você é um revisor clerical especialista em record linkage probabilístico.
Sua tarefa: decidir se um par candidato refere-se à mesma pessoa (MATCH),
a pessoas diferentes (NONMATCH) ou se a evidência é genuinamente inconclusiva (UNSURE).

Restrições obrigatórias:
1. Use apenas a informação contida no dossiê JSON recebido.
2. Não invente informações ausentes — dados faltantes são neutros, não negativos.
3. Não gere PII (nomes, datas completas, endereços). Se aparecerem no dossiê, não repita.
4. Retorne somente JSON válido, sem texto adicional fora do JSON.
5. Seja decisivo: UNSURE existe para incerteza genuína, não como refúgio seguro.
   Um par com evidências positivas fortes e nenhuma contradição deve ser MATCH,
   mesmo com dados faltantes em campos secundários.

Inferência de reason_codes a partir de features:
- nota_final >= 9.0 → HIGH_SCORE_ANCHOR; <= 3.0 → LOW_SCORE_ANCHOR
- média(PNOME, UNOME) >= 0.80 → NAME_STRONG; <= 0.40 → NAME_WEAK
- DTNASC >= 0.90 → DOB_STRONG; <= 0.50 → DOB_WEAK
- NMAE >= 0.75 → MOTHER_STRONG; ausente/zero → MOTHER_MISSING
- ENDRES ou MUNRES >= 0.70 → MUNICIPALITY_STRONG; baixo → ADDRESS_WEAK
- p_cal >= 0.70 → MODEL_HIGH_P; <= 0.40 → MODEL_AMBIGUOUS
- Features fortes conflitantes → CONFLICTING_SIGNALS
- > 50% campos ausentes → INSUFFICIENT_EVIDENCE

Política de decisão calibrada por banda:

NEAR_HIGH (nota 7–8.99) — prior: alta probabilidade de match.
- Se DOB_STRONG e (NAME_STRONG ou MOTHER_STRONG) → MATCH (confiança >= 0.75)
- Se DOB_STRONG e NAME_STRONG mesmo com MOTHER_MISSING → MATCH (confiança >= 0.70)
- Só retorne UNSURE se houver contradição ativa (DOB_WEAK ou NAME_WEAK).
- Dados faltantes NÃO são contradição.

GREY_MID (nota 5–6.99) — prior: incerto.
- Se DOB_STRONG e NAME_STRONG e (MOTHER_STRONG ou MUNICIPALITY_STRONG) → MATCH
- Se NAME_WEAK e DOB_WEAK → NONMATCH
- Se sinais mistos sem contradição direta → UNSURE

GREY_LOW (nota 3.01–4.99) — prior: baixa probabilidade de match.
- Só MATCH se todas as evidências primárias forem STRONG (nome + DOB + mãe).
- Se NAME_WEAK ou DOB_WEAK → NONMATCH (confiança >= 0.65)
- Caso contrário → UNSURE

FORA DA GREY ZONE:
- nota >= 9 sem inconsistência forte → MATCH (confiança >= 0.90)
- nota <= 3 → NONMATCH (confiança >= 0.90)

Regra do modelo calibrado:
- p_cal >= 0.70 e sem contradição forte → reforça MATCH
- p_cal <= 0.30 → reforça NONMATCH
- 0.40 <= p_cal <= 0.60 com evidências mistas → favorece UNSURE

Schema de saída:
{
  "pair_id": "string",
  "decision": "MATCH | NONMATCH | UNSURE",
  "confidence": 0.0-1.0,
  "reason_codes": ["lista de 1-5 códigos"],
  "evidence_summary": {
    "supports_match": [...],
    "supports_nonmatch": [...],
    "tie_breakers": [...]
  },
  "quality_flags": {
    "pii_leak_detected": false,
    "insufficient_evidence": false,
    "inconsistent_input": false
  }
}

Reason codes válidos: HIGH_SCORE_ANCHOR, LOW_SCORE_ANCHOR, NAME_STRONG, NAME_WEAK,
DOB_STRONG, DOB_WEAK, MOTHER_STRONG, MOTHER_MISSING, ADDRESS_WEAK,
MUNICIPALITY_STRONG, MODEL_HIGH_P, MODEL_AMBIGUOUS, CONFLICTING_SIGNALS,
INSUFFICIENT_EVIDENCE"""


# ── Helpers ───────────────────────────────────────────────────────────


def _make_pair_id(row: pd.Series) -> str:
    """Generate anonymous pair ID from PAR + PASSO."""
    raw = f"{row.get('PAR', 'x')}_{row.get('PASSO', 'x')}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_dossier(row: pd.Series) -> dict:
    """Build a safe dossier JSON from a DataFrame row (linkage features only)."""
    pair_id = _make_pair_id(row)

    # Group features by domain
    def _extract(cols: list[str]) -> dict:
        return {
            c: float(row[c]) if pd.notna(row[c]) else None
            for c in cols
            if c in row.index
        }

    nota = float(row["nota final"])

    return {
        "pair_id": pair_id,
        "nota_final": nota,
        "features": {
            "nome": _extract(NOME_COLS),
            "nome_mae": _extract(NOMEMAE_COLS),
            "data_nascimento": _extract(DTNASC_COLS),
            "municipio_residencia": _extract(CODMUN_COLS),
            "endereco": _extract(ENDERECO_COLS),
        },
        "model_outputs": {
            "band": _classify_band(nota),
            "p_cal": None,  # Not yet available in PoC
        },
    }


def _classify_band(nota: float) -> str:
    if nota >= 9:
        return "high"
    elif nota >= 7:
        return "near_high"
    elif nota >= 5:
        return "grey_mid"
    elif nota >= 3:
        return "grey_low"
    else:
        return "low"


def sample_grey_zone_pairs(csv_path: str | Path, n: int = 5) -> pd.DataFrame:
    """Sample n pairs from the grey zone (nota 4-8) with mix of true/false matches."""
    df = pd.read_csv(csv_path, sep=";", decimal=",")
    df.columns = [c.split(",")[0] for c in df.columns]

    grey = df[(df["nota final"] >= 4) & (df["nota final"] <= 8)].copy()

    # Stratified sample: try to get a spread across the grey zone
    bins = [(4, 5), (5, 6), (6, 7), (7, 8)]
    samples = []
    per_bin = max(1, n // len(bins))
    for lo, hi in bins:
        sub = grey[(grey["nota final"] >= lo) & (grey["nota final"] < hi)]
        if len(sub) > 0:
            samples.append(sub.sample(n=min(per_bin, len(sub)), random_state=42))

    result = pd.concat(samples).head(n)
    log.info(
        "Sampled %d grey-zone pairs (nota range: %.2f - %.2f)",
        len(result),
        result["nota final"].min(),
        result["nota final"].max(),
    )
    return result


# ── Fireworks API ────────────────────────────────────────────────────


def call_fireworks(model_id: str, dossier: dict, timeout: int = 120) -> dict:
    """Call Fireworks chat completion API and return parsed result."""
    user_msg = (
        "Avalie o dossiê abaixo e retorne sua decisão como JSON.\n\n"
        f"```json\n{json.dumps(dossier, ensure_ascii=False, indent=2)}\n```"
    )

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 800,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }

    t0 = time.time()
    resp = requests.post(
        FIREWORKS_BASE_URL, json=payload, headers=headers, timeout=timeout
    )
    elapsed = time.time() - t0

    if resp.status_code != 200:
        return {
            "error": True,
            "status_code": resp.status_code,
            "body": resp.text[:500],
            "elapsed_s": round(elapsed, 2),
        }

    data = resp.json()
    choice = data["choices"][0]
    raw_content = choice["message"]["content"]
    usage = data.get("usage", {})

    # Parse JSON from response
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown fences
        import re

        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_content, re.DOTALL)
        if m:
            parsed = json.loads(m.group(1))
        else:
            parsed = {"raw": raw_content, "parse_error": True}

    return {
        "error": False,
        "parsed": parsed,
        "elapsed_s": round(elapsed, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
    }


# ── Main ──────────────────────────────────────────────────────────────


def main(n_pairs: int = 5) -> None:
    csv_path = Path(__file__).resolve().parents[1] / "data" / "COMPARADORSEMIDENT.csv"
    if not csv_path.exists():
        log.error("Dataset not found: %s", csv_path)
        sys.exit(1)

    # Sample pairs
    pairs_df = sample_grey_zone_pairs(csv_path, n=n_pairs)

    # Build dossiers
    dossiers = []
    for _, row in pairs_df.iterrows():
        dossiers.append(build_dossier(row))

    log.info("Built %d dossiers", len(dossiers))

    # Test each model
    all_results: dict[str, list[dict]] = {}

    for model_name, model_id in MODELS.items():
        log.info("=" * 60)
        log.info("Testing model: %s (%s)", model_name, model_id)
        log.info("=" * 60)

        model_results = []
        for i, dossier in enumerate(dossiers):
            log.info(
                "  Pair %d/%d (nota=%.2f, band=%s) ...",
                i + 1,
                len(dossiers),
                dossier["nota_final"],
                dossier["model_outputs"]["band"],
            )

            result = call_fireworks(model_id, dossier)

            if result["error"]:
                log.warning(
                    "  ERROR: status=%s body=%s",
                    result["status_code"],
                    result["body"][:200],
                )
            else:
                parsed = result["parsed"]
                decision = parsed.get("decision", "???")
                confidence = parsed.get("confidence", "???")
                reasons = parsed.get("reason_codes", [])
                log.info(
                    "  → decision=%s  confidence=%s  reasons=%s  (%.1fs, %d tokens)",
                    decision,
                    confidence,
                    reasons,
                    result["elapsed_s"],
                    result.get("completion_tokens", 0) or 0,
                )

            model_results.append(
                {
                    "pair_id": dossier["pair_id"],
                    "nota_final": dossier["nota_final"],
                    "band": dossier["model_outputs"]["band"],
                    **result,
                }
            )

            # Small delay between calls to be polite
            time.sleep(0.5)

        all_results[model_name] = model_results

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    # Header
    print(f"\n{'Pair ID':<18} {'Nota':>5} {'Band':<10}", end="")
    for m in MODELS:
        print(f" | {m:<20}", end="")
    print()
    print("-" * (38 + 23 * len(MODELS)))

    for i in range(len(dossiers)):
        d = dossiers[i]
        print(
            f"{d['pair_id']:<18} {d['nota_final']:5.2f} {d['model_outputs']['band']:<10}",
            end="",
        )
        for m in MODELS:
            r = all_results[m][i]
            if r["error"]:
                cell = f"ERR({r['status_code']})"
            else:
                p = r["parsed"]
                dec = p.get("decision", "???")
                conf = p.get("confidence", 0)
                cell = f"{dec}({conf:.2f})"
            print(f" | {cell:<20}", end="")
        print()

    # Latency & cost summary
    print(f"\n{'Model':<20} {'Avg latency':>12} {'Total tokens':>14} {'Errors':>8}")
    print("-" * 58)
    for m in MODELS:
        results = all_results[m]
        latencies = [r["elapsed_s"] for r in results if not r["error"]]
        tokens = sum(
            r.get("completion_tokens", 0) or 0 for r in results if not r["error"]
        )
        errors = sum(1 for r in results if r["error"])
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        print(f"{m:<20} {avg_lat:>10.2f}s {tokens:>14d} {errors:>8d}")

    # Agreement matrix
    print("\nAgreement between models:")
    model_names = list(MODELS.keys())
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i + 1 :]:
            agree = 0
            total = 0
            for j in range(len(dossiers)):
                r1 = all_results[m1][j]
                r2 = all_results[m2][j]
                if not r1["error"] and not r2["error"]:
                    total += 1
                    if r1["parsed"].get("decision") == r2["parsed"].get("decision"):
                        agree += 1
            pct = (agree / total * 100) if total > 0 else 0
            print(f"  {m1} vs {m2}: {agree}/{total} ({pct:.0f}%)")

    # Save raw results
    out_path = (
        Path(__file__).resolve().parents[1] / "data" / "poc_llm_3model_results.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=5)
    args = parser.parse_args()
    main(n_pairs=args.n_pairs)
