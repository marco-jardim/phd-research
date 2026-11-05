"""Anthropic-only transport with explicit, byte-preserving prompt breakpoints.

No dotenv loading and no client construction until a request is made.
"""
from __future__ import annotations

import time

import anthropic

import gzcmd.llm_review as llm_review
from gzcmd.llm_review import LLMCallError, LLMReviewer, _extract_json


def caching_metadata(warmup: bool = True) -> dict:
    return {"enabled": True, "breakpoints": ["system[0]", "messages[0].content[0]"],
            "warmup": warmup}


class AnthropicReviewer(LLMReviewer):
    """Keep historical builders/parser/protocol, changing only the wire format."""

    _client: anthropic.Anthropic | None = None

    def _get_client(self):
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key, max_retries=0,
                                                timeout=self.timeout_s)
        return self._client

    def cache_blocks(self, messages):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        system_prompt, user_template = self._get_prompts()
        if not system.startswith(system_prompt) or not user.startswith(user_template):
            raise ValueError("Messages do not match the selected prompt")

        def blocks(text, prefix):
            return [{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": text[len(prefix):]}]

        # Separators stay at the start of the variable suffix. Never strip text.
        # The arbiter's dossier and opinions remain in their original order.
        return blocks(system, system_prompt), blocks(user, user_template)

    def _call_api(self, messages, *, model=None):
        model_id = model or self.model
        system, user = self.cache_blocks(messages)
        for attempt in range(1, self.max_retries + 1):
            started = time.monotonic()
            try:
                response = self._get_client().messages.create(
                    model=model_id, system=system,
                    messages=[{"role": "user", "content": user}], max_tokens=self.max_tokens)
                text = "".join(getattr(b, "text", "") for b in response.content
                               if getattr(b, "type", "") == "text")
                if response.stop_reason == "max_tokens":
                    raise ValueError("Truncated response")
                parsed = self._parse_response(text)
                return (parsed, time.monotonic() - started,
                        int(response.usage.input_tokens), int(response.usage.output_tokens))
            except (anthropic.APIStatusError, anthropic.APIConnectionError, ValueError, KeyError):
                # Do not log provider text, dossiers or credentials.
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_s * (2 ** (attempt - 1)))
        raise LLMCallError("Anthropic attempts exhausted", model=model_id, pair_id="",
                           attempts=self.max_retries)

    def _parse_response(self, text):
        return _extract_json(text)
