"""The ``Agent`` declarative class.

    class SupportAgent(Agent):
        \"\"\"Answers product questions from the docs.\"\"\"

        class Meta:
            model = "claude-opus-5"
            system = "You are a support engineer. Cite the docs you used."
            tools = [search_docs, create_ticket]
            memory = BufferMemory(k=20)

    SupportAgent().run("How do I rotate an API key?")

The tool-use loop, tracing, memory and usage accounting are the framework's.
An agent declares *what* it is; the loop that makes it work is written once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from mlango.agents.memory import Memory, NullMemory
from mlango.agents.providers.base import Completion, Provider, Usage, get_provider
from mlango.agents.tools import Tool, Toolbox, ToolResult
from mlango.agents.tracing import Tracer
from mlango.core.base import Declarative
from mlango.core.exceptions import ProviderError, RunError
from mlango.core.signals import agent_finished, agent_started, agent_step, tool_called
from mlango.metastore.models import RunStatus, Span

logger = logging.getLogger("mlango.agents")


@dataclass
class AgentRun:
    """The outcome of one agent invocation."""

    output: str = ""
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = Completion.END_TURN
    trace_uuid: str = ""
    session_id: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def tools_used(self) -> list[str]:
        return [r.name for r in self.tool_results]

    def describe(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "steps": self.steps,
            "usage": self.usage.describe(),
            "stop_reason": self.stop_reason,
            "trace": self.trace_uuid,
            "tools_used": self.tools_used,
            "error": self.error,
        }

    def __str__(self) -> str:
        return self.output


class Agent(Declarative):
    """A declared LLM agent."""

    _kind = "agent"
    _meta_options = (
        "model",
        "system",
        "tools",
        "provider",
        "max_steps",
        "max_tokens",
        "thinking",
        "effort",
        "memory",
        "stop_sequences",
        "tracing",
        "input_field",
    )

    class Meta:
        abstract = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._toolbox: Toolbox | None = None

    # -- configuration -------------------------------------------------------

    @classmethod
    def get_provider(cls) -> Provider:
        return get_provider(cls._meta.extras.get("provider"))

    @classmethod
    def get_model(cls) -> str:
        from mlango.conf import settings

        return str(cls._meta.extras.get("model") or settings.DEFAULT_AGENT_MODEL)

    def get_system(self) -> str:
        """The system prompt. Override to build it from the agent's fields."""
        return str(type(self)._meta.extras.get("system", "") or "")

    def get_tools(self) -> Toolbox:
        if self._toolbox is None:
            declared = type(self)._meta.extras.get("tools") or []
            self._toolbox = Toolbox([_as_tool(item) for item in declared])
        return self._toolbox

    @classmethod
    def get_memory(cls) -> Memory:
        memory = cls._meta.extras.get("memory")
        if memory is None:
            return NullMemory()
        return memory() if isinstance(memory, type) else memory

    @classmethod
    def get_max_steps(cls) -> int:
        from mlango.conf import settings

        return int(cls._meta.extras.get("max_steps") or settings.AGENT_MAX_STEPS)

    @classmethod
    def tracing_enabled(cls) -> bool:
        from mlango.conf import settings

        declared = cls._meta.extras.get("tracing")
        return bool(settings.TRACING if declared is None else declared)

    # -- running -------------------------------------------------------------

    @classmethod
    def chat(cls, message: str, **kwargs: Any) -> AgentRun:
        """One-liner for the common case: ``SupportAgent.chat("hi")``."""
        return cls().run(message, **kwargs)

    def run(
        self,
        message: str,
        *,
        session_id: str = "",
        memory: Memory | None = None,
        history: list[dict[str, Any]] | None = None,
        max_steps: int | None = None,
        run_id: int | None = None,
        **provider_kwargs: Any,
    ) -> AgentRun:
        """Run the tool-use loop until the model stops asking for tools."""
        opts = type(self)._meta
        provider = self.get_provider()
        toolbox = self.get_tools()
        store = memory if memory is not None else self.get_memory()
        limit = max_steps or self.get_max_steps()

        prior = list(history) if history is not None else store.load(session_id)
        user_turn = {"role": "user", "content": message}
        messages: list[dict[str, Any]] = [*prior, user_turn]

        tracer = Tracer(
            opts.label,
            session_id=session_id,
            run_id=run_id,
            enabled=self.tracing_enabled(),
            meta={"model": self.get_model(), "provider": provider.name},
        ).start(message)

        result = AgentRun(session_id=session_id, trace_uuid=tracer.uuid, messages=messages)
        agent_started.send(sender=type(self), agent=self, trace=tracer)

        try:
            self._loop(provider, toolbox, messages, tracer, result, limit, provider_kwargs)
        except ProviderError as exc:
            result.error = str(exc)
            tracer.fail(exc, steps=result.steps)
            agent_finished.send(sender=type(self), agent=self, trace=tracer)
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as RunError below
            result.error = f"{type(exc).__name__}: {exc}"
            tracer.fail(exc, steps=result.steps)
            agent_finished.send(sender=type(self), agent=self, trace=tracer)
            raise RunError(f"{opts.label} failed: {exc}") from exc

        tracer.finish(
            result.output,
            steps=result.steps,
            usage=result.usage,
            status=RunStatus.FAILED if result.error else RunStatus.FINISHED,
            error=result.error,
        )
        result.trace_uuid = tracer.uuid

        if session_id and not result.error:
            store.append(session_id, [user_turn, {"role": "assistant", "content": result.output}])

        agent_finished.send(sender=type(self), agent=self, trace=tracer)
        return result

    # -- the loop ------------------------------------------------------------

    def _loop(
        self,
        provider: Provider,
        toolbox: Toolbox,
        messages: list[dict[str, Any]],
        tracer: Tracer,
        result: AgentRun,
        limit: int,
        provider_kwargs: dict[str, Any],
    ) -> None:
        opts = type(self)._meta
        system = self.get_system()
        schemas = toolbox.schemas() or None

        for step in range(1, limit + 1):
            result.steps = step
            with tracer.span(
                f"llm:{self.get_model()}", Span.LLM, {"messages": len(messages)}
            ) as span:
                completion = provider.complete(
                    model=self.get_model(),
                    messages=messages,
                    system=system,
                    tools=schemas,
                    max_tokens=int(opts.extras.get("max_tokens", 4096)),
                    thinking=opts.extras.get("thinking", "adaptive"),
                    effort=opts.extras.get("effort"),
                    stop_sequences=opts.extras.get("stop_sequences"),
                    **provider_kwargs,
                )
                span["output"] = {"text": completion.text[:2000], "stop": completion.stop_reason}
                span["usage"] = completion.usage.describe()

            result.usage = result.usage.add(completion.usage)
            result.stop_reason = completion.stop_reason
            agent_step.send(sender=type(self), agent=self, trace=tracer, step=step)

            if completion.stop_reason == Completion.REFUSAL:
                detail = completion.refusal or {}
                result.error = (
                    "The model declined this request"
                    + (f" ({detail.get('category')})" if detail.get("category") else "")
                    + "."
                )
                result.output = completion.text
                return

            messages.append(_assistant_turn(completion))

            if completion.stop_reason == Completion.PAUSE_TURN:
                # A server-side tool hit its iteration cap. Re-send as-is; the
                # server resumes where it left off — do not inject a nudge.
                continue

            if not completion.wants_tools:
                result.output = completion.text
                return

            results = self._dispatch(toolbox, completion, tracer)
            result.tool_results.extend(results)
            # Every tool_result for one assistant turn goes back in a single
            # user message; splitting them teaches the model to stop batching.
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": r.tool_call_id,
                            "content": r.as_text(),
                            **({"is_error": True} if r.is_error else {}),
                        }
                        for r in results
                    ],
                }
            )

        result.error = (
            f"{opts.label} stopped after {limit} steps without finishing. "
            f"Raise max_steps, or narrow the task."
        )
        result.output = _last_text(messages)

    def _dispatch(
        self, toolbox: Toolbox, completion: Completion, tracer: Tracer
    ) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call in completion.tool_calls:
            found = toolbox.get(call.name)
            if found is None:
                results.append(
                    ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        content=(
                            f"No tool named {call.name!r} is available. "
                            f"Available tools: {', '.join(toolbox.names()) or '(none)'}."
                        ),
                        is_error=True,
                    )
                )
                continue

            tool_called.send(sender=type(self), agent=self, tool=found, arguments=call.arguments)
            with tracer.span(f"tool:{call.name}", Span.TOOL, call.arguments) as span:
                outcome = found.run(call_id=call.id, **call.arguments)
                span["output"] = {"content": outcome.as_text()[:4000], "is_error": outcome.is_error}
            results.append(outcome)
        return results

    # -- serving -------------------------------------------------------------

    @classmethod
    def as_endpoint(cls, **agent_kwargs: Any) -> Any:
        """A chat endpoint for ``routes.py``, à la Django's ``as_view()``."""
        from mlango.serve.endpoints import agent_endpoint

        return agent_endpoint(cls, **agent_kwargs)

    # -- introspection -------------------------------------------------------

    @classmethod
    def summary(cls) -> dict[str, Any]:
        instance = cls()
        return {
            "label": cls._meta.label,
            "model": cls.get_model(),
            "provider": cls._meta.extras.get("provider"),
            "tools": instance.get_tools().names(),
            "max_steps": cls.get_max_steps(),
            "memory": cls.get_memory().describe(),
            "system": instance.get_system(),
            "config": {f.name: f.get_default() for f in cls._meta.fields},
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _as_tool(item: Any) -> Tool:
    from mlango.agents.tools import tool as make_tool

    if isinstance(item, Tool):
        return item
    if callable(item):
        return make_tool(item)
    raise TypeError(f"{item!r} is not a tool: pass a @tool function or a Tool instance.")


def _assistant_turn(completion: Completion) -> dict[str, Any]:
    """The assistant turn to echo back, preserving provider-native blocks."""
    if completion.raw_content is not None:
        return {"role": "assistant", "content": completion.raw_content}

    content: list[dict[str, Any]] = []
    if completion.text:
        content.append({"type": "text", "text": completion.text})
    for call in completion.tool_calls:
        content.append(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        )
    return {"role": "assistant", "content": content or [{"type": "text", "text": ""}]}


def _last_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                getattr(b, "text", None) or (b.get("text") if isinstance(b, dict) else None)
                for b in content
            ]
            joined = "".join(p for p in parts if p)
            if joined:
                return joined
    return ""


__all__ = ["Agent", "AgentRun"]
