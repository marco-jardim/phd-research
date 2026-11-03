"""Exp2 com Claude Sonnet 5.5: reutiliza run_exp2_sem_model_outputs sem alterá-lo.

Diferenças em relação ao runner do Haiku (somente metadados de custo):
- tabela de preços do Sonnet 5.5 (USD/M: entrada 2, saída 10, escrita cache 5m 2,50, leitura cache 0,10);
- cost_note correspondente.
Parâmetros de chamada idênticos ao Haiku: sem temperature, max_tokens 2500, sem extended thinking.
A chave é lida de os.environ ou do .env, nunca impressa nem gravada.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

MODEL = "claude-sonnet-5-5"
COST_NOTE = ("Estimate only: standard/global USD/M for Claude Sonnet 5.5 input 2, output 10, "
             "cache read 0.10, cache write (5m) 2.50; not invoice. "
             "Source: platform.claude.com/docs/en/about-claude/pricing, 2026-11-03.")


def load_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == "ANTHROPIC_API_KEY":
                os.environ["ANTHROPIC_API_KEY"] = value.strip().strip('"').strip("'")


def sonnet_cost_estimate(stats):
    return round((stats["input_tokens"] * 2.0 + stats["output_tokens"] * 10.0
                  + stats["cache_read_input_tokens"] * 0.10
                  + stats["cache_creation_input_tokens"] * 2.50) / 1_000_000, 8)


def main():
    load_key()
    import run_exp2_sem_model_outputs as base
    abl = base.abl
    abl.haiku_cost_estimate = sonnet_cost_estimate
    inner_save = abl.save

    def save(path, value):
        if isinstance(value, dict) and "cost_note" in value and value.get("model") == MODEL:
            value["cost_note"] = COST_NOTE
        return inner_save(path, value)

    abl.save = save
    if "--modelo" not in sys.argv:
        sys.argv += ["--modelo", MODEL]
    base.main()


if __name__ == "__main__":
    main()
