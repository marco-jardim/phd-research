"""Exp 2 (revisão assistida por LLM) via servidor OpenCode v2, com qualquer modelo
configurado no OpenCode (padrão: anthropic/claude-haiku-5-5, pela autenticação do usuário).

Reaproveita integralmente scripts/run_exp2_llm_sync.py (pipeline v3, dossiês,
protocolo duplo agente + árbitro, checkpoints). Em vez de chat completions, cada
chamada ao modelo abre uma sessão descartável no servidor (`POST /session`), envia o
prompt de sistema e o dossiê (`POST /session/{id}/message` com `system`, `model` e
`tools` desligadas), lê o texto da resposta e apaga a sessão. Sem contexto entre pares.

Limitações: a API de sessões não expõe temperatura nem response_format; o determinismo
depende do modelo e do prompt. Tokens vêm de `info.tokens` quando disponíveis.

Pré-requisito: servidor dedicado com config mínima (sem plugins que reescrevam o modelo,
como o model-router), por exemplo:
    $env:OPENCODE_CONFIG = "<caminho>/opencode-exp2.json"   # define o agente "exp2-haiku"
    opencode serve --port 4097
    $env:OPENCODE_SERVER_URL = "http://127.0.0.1:4097"
    $env:OPENCODE_SERVER_PASSWORD = "<senha impressa pelo serve>"
Uso:
    python scripts/run_exp2_opencode_server.py --mode vigilancia [--model anthropic/claude-haiku-5-5]
        [--workers 4] [--limit 0] [--out-dir data/sprint3b/opencode_server]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
load_dotenv(ROOT / ".env")

import gzcmd.llm_review as llm_review  # noqa: E402
import run_exp2_llm_sync as exp2  # noqa: E402
from gzcmd.llm_review import LLMCallError, LLMReviewer, _extract_json  # noqa: E402

# OpenCode v2: API sob /api; `opencode serve` imprime "server password <senha>" no stdout
# (HTTP basic, usuário "opencode") quando OPENCODE_SERVER_PASSWORD não é definida.
SERVER = os.environ.get("OPENCODE_SERVER_URL", "http://127.0.0.1:4096").rstrip("/") + "/api"
DEFAULT_MODEL = "anthropic/claude-haiku-5-5"
DEFAULT_OUT = ROOT / "data" / "sprint3b" / "opencode_server"


def _auth() -> tuple[str, str] | None:
    pw = os.environ.get("OPENCODE_SERVER_PASSWORD")
    return (os.environ.get("OPENCODE_SERVER_USERNAME", "opencode"), pw) if pw else None


class OpenCodeServerReviewer(LLMReviewer):
    """LLMReviewer cujo transporte é o servidor OpenCode (sessão descartável por chamada)."""

    agent: str = "exp2-haiku"  # agente definido em opencode-exp2.json (modelo fixo, sem ferramentas)

    def _call_api(self, messages: list[dict[str, str]], *, model: str | None = None):
        model_id = model or self.model
        provider_id, _, model_name = model_id.partition("/")
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.monotonic()
            sid = None
            try:
                r = requests.post(f"{SERVER}/session", json={"title": f"exp2-{model_name}"}, auth=_auth(), timeout=30)
                r.raise_for_status()
                sid = r.json()["data"]["id"]
                # OpenCode v2: POST /session/{id}/prompt é assíncrono; a resposta chega em /message.
                body: dict[str, Any] = {"text": user, "agent": self.agent, "system": system,
                                        "model": {"providerID": provider_id, "modelID": model_name}}
                r = requests.post(f"{SERVER}/session/{sid}/prompt", json=body, auth=_auth(), timeout=60)
                r.raise_for_status()
                msg = None
                while time.monotonic() - t0 < self.timeout_s:
                    time.sleep(0.5)
                    ms = requests.get(f"{SERVER}/session/{sid}/message", auth=_auth(), timeout=30).json()["data"]
                    done = [m for m in ms if m.get("type") == "assistant" and (m.get("time") or {}).get("completed")]
                    if done:
                        msg = done[-1]
                        break
                latency = time.monotonic() - t0
                if msg is None:
                    raise ValueError(f"timeout after {latency:.0f}s waiting for assistant message")
                got = f"{msg['model']['providerID']}/{msg['model']['id']}"
                if got != model_id:
                    raise ValueError(f"server answered with {got}, expected {model_id}")
                text = "".join(c.get("text", "") for c in msg.get("content", []) if c.get("type") == "text")
                tok = msg.get("tokens") or {}
                parsed = _extract_json(text)
                return parsed, latency, int(tok.get("input", 0)), int(tok.get("output", 0))
            except (requests.RequestException, ValueError, KeyError) as exc:
                last_error = exc
                llm_review.logger.warning("OpenCode server call failed (attempt %d/%d): %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_s * (2 ** (attempt - 1)))
            finally:
                if sid:
                    try:
                        requests.delete(f"{SERVER}/session/{sid}", auth=_auth(), timeout=30)
                    except requests.RequestException:
                        pass
        raise LLMCallError(f"All {self.max_retries} attempts failed: {last_error}", model=model_id, pair_id="",
                           attempts=self.max_retries)


def main() -> None:
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

    h = requests.get(f"{SERVER}/provider", auth=_auth(), timeout=30)
    h.raise_for_status()
    providers = [p["id"] for p in h.json()["data"]]
    if model.partition("/")[0] not in providers:
        raise SystemExit(f"provedor {model.partition('/')[0]} não está ativo no servidor: {providers}")
    exp2.log.info("OpenCode server %s | provedores: %s", SERVER, providers)

    llm_review.LLMReviewer.from_env = classmethod(
        lambda cls, **ov: OpenCodeServerReviewer(api_key="opencode-server", base_url=SERVER, model=model,
                                                 fallback_model=model, temperature=0.0, max_tokens=800,
                                                 timeout_s=180.0, max_retries=4, **ov)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    exp2.OUT_DIR = out_dir
    exp2.log.info("Provedor: OpenCode server | modelo: %s | modo: %s | saída: %s", model, mode, out_dir)
    exp2.main()


if __name__ == "__main__":
    main()
