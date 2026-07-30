"""Anthropic provider.

Notes that matter for current Claude models, and that this module encodes so
callers do not have to remember them:

* ``temperature`` / ``top_p`` / ``top_k`` are rejected with a 400 on Claude
  Opus 5, Sonnet 5 and the Opus 4.7/4.8 family. They are never sent.
* Thinking is configured as ``{"type": "adaptive"}``; the old
  ``budget_tokens`` form also 400s. Depth is controlled with ``effort``.
* ``stop_reason`` must be checked before reading content — a refusal returns
  HTTP 200 with an empty or partial content list.
"""

from __future__ import annotations

import logging
from typing import Any

from mlango.agents.providers.base import Completion, Provider, ToolCall, Usage
from mlango.core.exceptions import ProviderError

logger = logging.getLogger("mlango.agents.anthropic")

#: Effort levels the current models accept.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


class AnthropicProvider(Provider):
    name = "anthropic"
    requires = ("anthropic",)

    def __init__(self, **options: Any):
        super().__init__(**options)
        self._client: Any = None

    # -- client --------------------------------------------------------------

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            # No api_key argument: the SDK resolves ANTHROPIC_API_KEY, then
            # ANTHROPIC_AUTH_TOKEN, then a configured CLI profile. Passing a
            # key explicitly would defeat profile-based auth.
            self._client = anthropic.Anthropic(**self.options)
        return self._client

    # -- completion ----------------------------------------------------------

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        thinking: str | None = "adaptive",
        effort: str | None = None,
        stop_sequences: list[str] | None = None,
        **extra: Any,
    ) -> Completion:
        import anthropic

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools
        if stop_sequences:
            payload["stop_sequences"] = stop_sequences
        if thinking:
            payload["thinking"] = {"type": thinking}
        if effort:
            if effort not in EFFORT_LEVELS:
                raise ProviderError(
                    f"Unknown effort {effort!r}; use one of {', '.join(EFFORT_LEVELS)}."
                )
            payload["output_config"] = {"effort": effort}

        # Sampling parameters are removed on current models and return a 400.
        for rejected in ("temperature", "top_p", "top_k"):
            if extra.pop(rejected, None) is not None:
                logger.warning(
                    "Dropping %r: current Claude models reject sampling parameters. "
                    "Steer behaviour with the system prompt or `effort` instead.",
                    rejected,
                )
        payload.update(extra)

        try:
            response = self.client.messages.create(**payload)
        except anthropic.RateLimitError as exc:
            raise ProviderError(f"Anthropic rate limit reached: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"Could not reach the Anthropic API: {exc}") from exc

        return self._to_completion(response)

    # -- translation ---------------------------------------------------------

    @staticmethod
    def _to_completion(response: Any) -> Completion:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content or []:
            kind = getattr(block, "type", None)
            if kind == "text":
                text_parts.append(block.text)
            elif kind == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
                )

        raw_usage = getattr(response, "usage", None)
        usage = Usage(
            input_tokens=getattr(raw_usage, "input_tokens", 0) or 0,
            output_tokens=getattr(raw_usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(raw_usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(raw_usage, "cache_creation_input_tokens", 0) or 0,
        )

        refusal = None
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            refusal = {
                "category": getattr(details, "category", None),
                "explanation": getattr(details, "explanation", ""),
            }

        return Completion(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or Completion.END_TURN,
            usage=usage,
            raw_content=response.content,
            model=getattr(response, "model", ""),
            refusal=refusal,
        )

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "offline": False, "sdk": "anthropic"}
