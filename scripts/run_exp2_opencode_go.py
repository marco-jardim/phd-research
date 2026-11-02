"""Exp 2 (revisão assistida por LLM) via OpenCode Go, com DeepSeek V4.1 Flash.

Reaproveita integralmente scripts/run_exp2_llm_sync.py (pipeline v3, dossiês,
protocolo duplo agente + árbitro, checkpoints). Diferenças:
  - provedor OpenCode Go (``https://opencode.ai/zen/go/v1``), chave em
    OPENCODE_GO_API_KEY (.env), com o header obrigatório ``x-opencode-session``
    estável por execução, conforme https://opencode.ai/docs/go/;
  - modelo ``deepseek-v4.1-flash`` (sem modelo de fallback distinto);
  - saída em data/sprint3b/opencode_go/, preservando os artefatos da execução
    original com Kimi K2.5 (data/sprint3b/exp2_llm_sync_*.json).

Uso:
    python scripts/run_exp2_opencode_go.py --mode vigilancia [--workers 10] [--limit 0]
    python scripts/run_exp2_opencode_go.py --mode confirmacao
Depois:
    python scripts/reconstruir_gabarito_exp2.py --json data/sprint3b/opencode_go/exp2_llm_sync_<modo>.json
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

load_dotenv(ROOT / ".env")

import gzcmd.llm_review as llm_review  # noqa: E402
import run_exp2_llm_sync as exp2  # noqa: E402

BASE_URL = "https://opencode.ai/zen/go/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"
OUT_DIR = ROOT / "data" / "sprint3b" / "opencode_go"
# Decisão determinística com saída JSON: sem amostragem (T=0, top_k=1), sem penalidades
# (distorceriam os reason_codes repetitivos) e margem de tokens para não truncar o JSON,
# o que viraria UNSURE indevido. Mesma temperatura da execução original com Kimi K2.5.
GEN_PARAMS = dict(
    temperature=0.0, top_p=1.0, top_k=1,
    presence_penalty=0.0, frequency_penalty=0.0,
    max_tokens=800, timeout_s=120.0, max_retries=4,
)
# Sondagem (02/11/2026, mesmo dossiê, 9 chamadas): com raciocínio ligado o modelo gastou
# 650–1.576 tokens pensando, truncou o JSON e deu decisões diferentes a T=0; com
# raciocínio desligado, ~190 tokens de saída e decisão idêntica em todas as chamadas.


def _session_id(mode: str) -> str:
    return hashlib.sha256(f"phd-research-exp2-{mode}".encode()).hexdigest()[:32]


# Campos extras por modelo. DeepSeek V4.1 Flash: raciocínio desligado (com ele ligado o modelo
# consome o orçamento no raciocínio, devolve `content` vazio e decide de forma instável a T=0).
# Kimi K3: não aceita desligar o raciocínio; `reasoning_effort: low` foi a única configuração com
# decisão idêntica em três chamadas do mesmo dossiê (sondagem 02/11/2026). Kimi K2.5/K2.6 não
# estão disponíveis no OpenCode Go (unavailable / deprecated); K3.1 Pro e K2.7 exigem T>0.
# Atenção: quando a quota do modelo se esgota, o gateway responde 400 "Invalid request
# parameters" (não 429), em ~1 s, para qualquer dossiê; o run de confirmação com Kimi K3 em
# 02/11/2026 parou em 343/435 por isso. Retomar com --resume após renovação da quota; o
# checkpoint guarda os pares com erro em `errors` e o script os reprocessa se essa lista for
# esvaziada antes do resume.
EXTRA_BY_MODEL: dict[str, dict] = {
    "deepseek-v4.1-flash": {"thinking": {"type": "disabled"}, "reasoning_effort": "none"},
    "kimi-k3": {"reasoning_effort": "low"},
}
MAX_TOKENS_BY_MODEL: dict[str, int] = {"deepseek-v4.1-flash": 800, "kimi-k3": 2500}


class _RequestsWithSession:
    """Envolve ``requests`` acrescentando o header x-opencode-session e campos extras no corpo."""

    def __init__(self, inner, session_id: str, extra_payload: dict | None = None) -> None:
        self._inner = inner
        self._session_id = session_id
        self._extra = extra_payload or {}

    def post(self, url, headers=None, json=None, **kwargs):
        headers = dict(headers or {})
        headers["x-opencode-session"] = self._session_id
        if isinstance(json, dict) and self._extra:
            json = {**json, **self._extra}
        return self._inner.post(url, headers=headers, json=json, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def main() -> None:
    api_key = os.environ.get("OPENCODE_GO_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENCODE_GO_API_KEY não encontrada no ambiente/.env")

    # --model e --out-dir são consumidos aqui e removidos antes de delegar ao argparse do exp2.
    mode, model, out_dir = "vigilancia", MODEL, OUT_DIR
    argv = []
    i = 0
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

    extra = EXTRA_BY_MODEL.get(model, {})
    gen = {**GEN_PARAMS, "max_tokens": MAX_TOKENS_BY_MODEL.get(model, GEN_PARAMS["max_tokens"])}
    llm_review.requests = _RequestsWithSession(llm_review.requests, _session_id(f"{model}-{mode}"), extra)
    llm_review.LLMReviewer.from_env = classmethod(
        lambda cls, **ov: cls(api_key=api_key, base_url=BASE_URL, model=model, fallback_model=model,
                              **{**gen, **ov})
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    exp2.OUT_DIR = out_dir
    exp2.log.info("Provedor: OpenCode Go | modelo: %s | saída: %s", model, out_dir)
    exp2.main()


if __name__ == "__main__":
    main()
