"""Cliente do OpenCode Go (endpoint compatível com OpenAI) com a interface ``messages.create`` do cliente Anthropic.

Assim ``execucao.executar_sync`` (pool de threads, retry com backoff, checkpoint, ``usage``) roda igual para os
dois provedores. Só o modo sync existe: não há ``messages.batches``.

A requisição segue o que os runs do repositório fizeram com o DeepSeek V4.1 Flash
(``scripts/run_exp2_opencode_go.py``): ``POST https://opencode.ai/zen/go/v1/chat/completions``, chave em
``Authorization: Bearer``, header ``x-opencode-session`` estável por execução, ``temperature`` 0,0, saída em
objeto JSON e raciocínio desligado (com ele ligado o modelo gasta o orçamento pensando e devolve conteúdo
vazio). Os blocos de sistema e de usuário do formato Anthropic viram mensagens de texto simples; o
``cache_control`` não tem equivalente e é descartado. A chave só entra no header; nenhuma exceção, log ou
objeto devolvido a contém.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

CHAVE_ENV_OPENCODE = "OPENCODE_GO_API_KEY"
URL_OPENCODE = "https://opencode.ai/zen/go/v1/chat/completions"
MODELO_OPENCODE_PADRAO = "deepseek-v4.1-flash"
TIMEOUT_S = 300.0
# Campos extras por modelo, como em run_exp2_opencode_go.EXTRA_BY_MODEL.
EXTRA_POR_MODELO: dict[str, dict] = {
    "deepseek-v4.1-flash": {"thinking": {"type": "disabled"}, "reasoning_effort": "none"},
}


class ErroOpencode(Exception):
    """Resposta HTTP de erro do gateway; guarda o status (e o corpo, só para detectar falta de crédito)."""

    def __init__(self, status: int, corpo: str = "") -> None:
        super().__init__(f"OpenCode Go status {status}")
        self.status_code = status
        self.body = corpo[:500]


class APIConnectionError(Exception):
    """Falha de rede ou tempo esgotado (reconhecida pelo nome da classe em ``av.is_api_exception``)."""


def _texto_dos_blocos(conteudo) -> str:
    if isinstance(conteudo, str):
        return conteudo
    return "\n\n".join(str(bloco.get("text", "")) for bloco in conteudo)


def session_id(modelo: str) -> str:
    return hashlib.sha256(f"phd-research-cruzamento-{modelo}".encode()).hexdigest()[:32]


def montar_payload(params: dict) -> dict:
    """Corpo ``chat/completions`` a partir dos parâmetros no formato Anthropic de ``execucao.montar_params``."""
    mensagens = []
    if params.get("system"):
        mensagens.append({"role": "system", "content": _texto_dos_blocos(params["system"])})
    for mensagem in params["messages"]:
        mensagens.append({"role": mensagem["role"], "content": _texto_dos_blocos(mensagem["content"])})
    return {
        "model": params["model"],
        "messages": mensagens,
        "max_tokens": params["max_tokens"],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        **EXTRA_POR_MODELO.get(params["model"], {}),
    }


def _mensagem(corpo: dict) -> SimpleNamespace:
    """Resposta no formato que ``execucao.nova_entrada`` lê (``content``, ``usage``, ``stop_reason``)."""
    escolha = (corpo.get("choices") or [{}])[0]
    texto = str((escolha.get("message") or {}).get("content") or "")
    uso = corpo.get("usage") or {}
    detalhes = uso.get("prompt_tokens_details") or {}
    final = escolha.get("finish_reason")
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=texto)],
        usage=SimpleNamespace(
            input_tokens=int(uso.get("prompt_tokens") or 0),
            output_tokens=int(uso.get("completion_tokens") or 0),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=int(detalhes.get("cached_tokens") or 0),
        ),
        stop_reason="max_tokens" if final == "length" else final,
    )


class _Mensagens:
    def __init__(self, cliente: ClienteOpencode) -> None:
        self._cliente = cliente

    def create(self, **params):
        return self._cliente.completar(params)


class ClienteOpencode:
    """``cliente.messages.create(**params)`` sobre ``chat/completions``; ``http`` é o módulo ``requests`` (ou um dublê)."""

    def __init__(self, api_key: str, http=None, url: str = URL_OPENCODE, timeout_s: float = TIMEOUT_S) -> None:
        if http is None:
            import requests

            http = requests
        self._api_key = api_key
        self._http = http
        self._url = url
        self._timeout_s = timeout_s
        self.messages = _Mensagens(self)

    def completar(self, params: dict) -> SimpleNamespace:
        payload = montar_payload(params)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "x-opencode-session": session_id(payload["model"]),
        }
        try:
            resposta = self._http.post(self._url, headers=headers, json=payload, timeout=self._timeout_s)
        except OSError as exc:  # requests.RequestException herda de OSError (conexão, tempo esgotado)
            raise APIConnectionError(type(exc).__name__) from None
        if resposta.status_code >= 400:
            raise ErroOpencode(resposta.status_code, getattr(resposta, "text", "") or "")
        try:
            corpo = resposta.json()
        except ValueError:
            raise APIConnectionError("corpo_nao_json") from None
        return _mensagem(corpo)


def criar_cliente(api_key: str) -> ClienteOpencode:
    return ClienteOpencode(api_key)
