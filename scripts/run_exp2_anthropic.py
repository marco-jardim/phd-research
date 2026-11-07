"""Exp 2 (revisão assistida por LLM) chamando a Messages API da Anthropic diretamente.

Reaproveita integralmente scripts/run_exp2_llm_sync.py (pipeline v3, dossiês,
protocolo duplo agente + árbitro, checkpoints, métricas). Apenas o transporte muda:
cada chamada é um ``messages.create`` independente (sem histórico entre pares),
com o prompt de sistema do pacote e o dossiê como mensagem do usuário.

Parâmetros de geração: max_tokens=2500 (o Haiku 5.5 escreve 1.000–1.300 tokens de JSON
por decisão), sem raciocínio estendido. Claude Haiku 5.5 não aceita `temperature` (a API
devolve 400 "deprecated for this model"), ao contrário das execuções com Kimi K2.5 e
DeepSeek V4.1 Flash, feitas a T=0. Sondagem (02/11/2026, mesmo dossiê, 4 chamadas):
decisão e confiança idênticas nas quatro. `_extract_json` tolera cercas ```json.

Uso:
    python scripts/run_exp2_anthropic.py --mode vigilancia [--model claude-haiku-5-5]
        [--workers 6] [--limit 0] [--out-dir data/sprint3b/anthropic] [--resume <json>]
Depois:
    python scripts/reconstruir_gabarito_exp2.py --json data/sprint3b/anthropic/exp2_llm_sync_<modo>.json
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
load_dotenv(ROOT / ".env")

import gzcmd.llm_review as llm_review  # noqa: E402
import run_exp2_llm_sync as exp2  # noqa: E402
from gzcmd.llm_review import LLMCallError, LLMReviewer, _extract_json  # noqa: E402

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_OUT = ROOT / "data" / "sprint3b" / "anthropic"


class AnthropicReviewer(LLMReviewer):
    """LLMReviewer cujo transporte é a Messages API da Anthropic."""

    _client: anthropic.Anthropic | None = None

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key, max_retries=0, timeout=self.timeout_s)
        return self._client

    def _call_api(self, messages: list[dict[str, str]], *, model: str | None = None):
        model_id = model or self.model
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.monotonic()
            try:
                # Claude Haiku 5.5 rejeita `temperature` ("deprecated for this model"); a amostragem
                # fica no padrão da API. A repetibilidade é medida por sondagem (ver docstring).
                resp = self._get_client().messages.create(
                    model=model_id,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    max_tokens=self.max_tokens,
                )
                latency = time.monotonic() - t0
                text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
                if resp.stop_reason == "max_tokens":
                    raise ValueError(f"resposta truncada em {self.max_tokens} tokens")
                parsed = _extract_json(text)
                return parsed, latency, int(resp.usage.input_tokens), int(resp.usage.output_tokens)
            except anthropic.RateLimitError as exc:
                last_error = exc
                delay = self.retry_delay_s * (2 ** (attempt - 1)) * 2
                llm_review.logger.warning("Rate limited (429), attempt %d/%d, sleeping %.1fs", attempt, self.max_retries, delay)
                time.sleep(delay)
            except (anthropic.APIStatusError, anthropic.APIConnectionError, ValueError, KeyError) as exc:
                last_error = exc
                llm_review.logger.warning("Anthropic call failed (attempt %d/%d): %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_s * (2 ** (attempt - 1)))
        raise LLMCallError(f"All {self.max_retries} attempts failed: {last_error}", model=model_id, pair_id="",
                           attempts=self.max_retries)


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY não encontrada no ambiente/.env")
    mode, model, out_dir, argv, i = "vigilancia", DEFAULT_MODEL, DEFAULT_OUT, [], 0
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
        if a == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]; i += 2; continue
        if a == "--out-dir" and i + 1 < len(sys.argv):
            out_dir = ROOT / sys.argv[i + 1]; i += 2; continue
        argv.append(a); i += 1
    sys.argv = argv

    llm_review.LLMReviewer.from_env = classmethod(
        lambda cls, **ov: AnthropicReviewer(api_key=api_key, base_url="https://api.anthropic.com", model=model,
                                            fallback_model=model, temperature=0.0, max_tokens=2500,
                                            timeout_s=120.0, max_retries=5, retry_delay_s=3.0, **ov)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    exp2.OUT_DIR = out_dir
    exp2.log.info("Provedor: Anthropic Messages API | modelo: %s | modo: %s | saída: %s", model, mode, out_dir)
    exp2.main()


if __name__ == "__main__":
    main()
