"""A deterministic offline provider.

Every framework needs a backend that runs with no credentials, no network and
no cost — Django has SQLite and the locmem email backend for exactly this. The
echo provider makes agent code testable in CI: it follows simple scripted rules
so a test can assert on tool dispatch, tracing and memory without an LLM.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from mlango.agents.providers.base import Completion, Provider, ToolCall, Usage


class EchoProvider(Provider):
    """Replies deterministically, and calls tools when asked to in plain text.

    Rules, applied to the latest user message:

    * ``use <tool> {json}`` — emit a tool call with those arguments.
    * ``use <tool>`` — emit a tool call with no arguments.
    * anything else — echo the message back, prefixed with ``echo:``.

    After a tool result comes back, it summarises the results and stops, which
    is enough to exercise a full multi-step loop.
    """

    name = "echo"
    offline = True

    #: Extra rules a test can register: ``(pattern, handler)``.
    rules: list[tuple[re.Pattern[str], Callable[[re.Match[str]], Completion]]] = []

    TOOL_RE = re.compile(r"^\s*use\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\{.*\})?\s*$", re.DOTALL)

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
        last_user = _last_text(messages, role="user")
        usage = Usage(input_tokens=_rough_tokens(messages, system), output_tokens=0)

        # A turn whose last user message carries tool results: summarise and stop.
        if _has_tool_results(messages):
            summary = _summarise_tool_results(messages)
            usage.output_tokens = _rough_tokens_text(summary)
            return Completion(
                text=summary, stop_reason=Completion.END_TURN, usage=usage, model=model
            )

        for pattern, handler in self.rules:
            match = pattern.search(last_user)
            if match:
                return handler(match)

        match = self.TOOL_RE.match(last_user)
        if match and tools:
            name = match.group(1)
            known = {t.get("name") for t in tools}
            if name in known:
                try:
                    arguments = json.loads(match.group(2)) if match.group(2) else {}
                except json.JSONDecodeError:
                    arguments = {}
                call = ToolCall(id=f"echo_{name}_{len(messages)}", name=name, arguments=arguments)
                return Completion(
                    text="",
                    tool_calls=[call],
                    stop_reason=Completion.TOOL_USE,
                    usage=usage,
                    model=model,
                )

        text = f"echo: {last_user}" if last_user else "echo: (no input)"
        usage.output_tokens = _rough_tokens_text(text)
        return Completion(text=text, stop_reason=Completion.END_TURN, usage=usage, model=model)

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "offline": True, "deterministic": True}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _last_text(messages: list[dict[str, Any]], *, role: str) -> str:
    for message in reversed(messages):
        if message.get("role") != role:
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            if parts:
                return "".join(parts)
    return ""


def _has_tool_results(messages: list[dict[str, Any]]) -> bool:
    if not messages:
        return False
    content = messages[-1].get("content")
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def _summarise_tool_results(messages: list[dict[str, Any]]) -> str:
    blocks = [
        b
        for b in messages[-1].get("content", [])
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    rendered = []
    for block in blocks:
        content = block.get("content")
        if isinstance(content, list):
            content = " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
        rendered.append(str(content))
    return "tool results: " + "; ".join(rendered)


def _rough_tokens_text(text: str) -> int:
    """A crude ~4-characters-per-token estimate. Only for offline accounting."""
    return max(1, len(text) // 4)


def _rough_tokens(messages: list[dict[str, Any]], system: str) -> int:
    total = _rough_tokens_text(system)
    for message in messages:
        content = message.get("content")
        total += _rough_tokens_text(
            content if isinstance(content, str) else json.dumps(content, default=str)
        )
    return total
