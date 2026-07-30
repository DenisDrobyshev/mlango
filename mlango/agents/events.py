"""Agent events.

``Agent.run()`` returns only when the loop is finished, which is the wrong shape
for a user interface: a multi-step agent can take a minute, and a blank screen
for a minute reads as broken. ``Agent.stream()`` yields these events as they
happen instead.

The event stream is a **superset** of what ``run()`` reports — the final
:class:`Finished` event carries the same :class:`~mlango.agents.agent.AgentRun`,
so nothing is lost by streaming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentEvent:
    """Base class for everything the loop emits."""

    #: Which step of the loop this belongs to, starting at 1.
    step: int = 0

    @property
    def kind(self) -> str:
        """A stable, lowercase name for wire formats and UI switches."""
        name = type(self).__name__
        out = []
        for index, char in enumerate(name):
            if char.isupper() and index:
                out.append("_")
            out.append(char.lower())
        return "".join(out)

    def describe(self) -> dict[str, Any]:
        """A JSON-safe payload — what an SSE endpoint sends."""
        from dataclasses import asdict

        from mlango.core.serialization import jsonable

        return {"event": self.kind, **jsonable(asdict(self))}


@dataclass
class Started(AgentEvent):
    """The loop is about to make its first model call."""

    agent: str = ""
    trace: str = ""
    session_id: str = ""
    message: str = ""


@dataclass
class Thinking(AgentEvent):
    """A model call is in flight.

    Emitted before the request so a UI can show progress during the wait, which
    is the part of an agent turn that actually takes time.
    """

    model: str = ""


@dataclass
class TextChunk(AgentEvent):
    """A piece of assistant text.

    Providers that stream token-by-token emit many of these; providers that do
    not emit exactly one per model call, so a consumer never needs to care which
    kind it is talking to.
    """

    text: str = ""


@dataclass
class ToolCalled(AgentEvent):
    """The model asked for a tool, and it is about to run."""

    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str = ""


@dataclass
class ToolFinished(AgentEvent):
    """A tool returned, or failed."""

    name: str = ""
    content: str = ""
    is_error: bool = False
    duration_s: float = 0.0
    tool_call_id: str = ""


@dataclass
class StepFinished(AgentEvent):
    """One pass of the loop completed, with its token usage."""

    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Finished(AgentEvent):
    """The loop is done. Carries the same result ``run()`` would have returned."""

    output: str = ""
    trace: str = ""
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    error: str = ""
    #: The full result object. Excluded from ``describe()`` because it is not
    #: JSON, and the fields above already carry everything a client needs.
    result: Any = None

    @property
    def ok(self) -> bool:
        return not self.error

    def describe(self) -> dict[str, Any]:
        from mlango.core.serialization import jsonable

        return {
            "event": self.kind,
            "step": self.step,
            "output": self.output,
            "trace": self.trace,
            "tools_used": list(self.tools_used),
            "usage": jsonable(self.usage),
            "error": self.error,
        }


@dataclass
class Failed(AgentEvent):
    """The loop raised. The exception is re-raised after this event is yielded."""

    error: str = ""
    exception_type: str = ""


__all__ = [
    "AgentEvent",
    "Started",
    "Thinking",
    "TextChunk",
    "ToolCalled",
    "ToolFinished",
    "StepFinished",
    "Finished",
    "Failed",
]
