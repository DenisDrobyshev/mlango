"""Tools, the agent loop, memory and tracing."""

from __future__ import annotations

from typing import Literal

import pytest

from mlango.agents import Agent, BufferMemory, NullMemory, WindowMemory, tool
from mlango.agents.tools import Tool, ToolError, Toolbox
from mlango.core.exceptions import ValidationError


@tool
def echo_tool(text: str, times: int = 1) -> str:
    """Repeat the text.

    Args:
        text: What to repeat.
        times: How many times to repeat it.
    """
    return " ".join([text] * times)


@tool
def strict_tool(ticket: str, status: Literal["open", "closed"]) -> str:
    """Change a ticket's status.

    Args:
        ticket: Ticket id, which must start with T-.
        status: The new status.
    """
    if not ticket.startswith("T-"):
        raise ToolError("Ticket ids start with T-.")
    return f"{ticket} -> {status}"


class TestToolSchema:
    def test_name_and_description_come_from_the_function(self):
        schema = echo_tool.to_schema()
        assert schema["name"] == "echo_tool"
        assert schema["description"] == "Repeat the text."

    def test_types_map_to_json_schema(self):
        properties = echo_tool.to_schema()["input_schema"]["properties"]
        assert properties["text"]["type"] == "string"
        assert properties["times"]["type"] == "integer"

    def test_argument_docs_become_descriptions(self):
        properties = echo_tool.to_schema()["input_schema"]["properties"]
        assert properties["text"]["description"] == "What to repeat."

    def test_wrapped_docs_are_joined(self):
        @tool
        def wrapped(value: str) -> str:
            """Do a thing.

            Args:
                value: A description that continues
                    onto a second line.
            """
            return value

        description = wrapped.to_schema()["input_schema"]["properties"]["value"]["description"]
        assert description == "A description that continues onto a second line."

    def test_defaults_make_arguments_optional(self):
        schema = echo_tool.to_schema()["input_schema"]
        assert schema["required"] == ["text"]
        assert schema["properties"]["times"]["default"] == 1

    def test_literal_becomes_an_enum(self):
        status = strict_tool.to_schema()["input_schema"]["properties"]["status"]
        assert status["type"] == "string"
        assert status["enum"] == ["open", "closed"]

    def test_list_annotation_becomes_an_array(self):
        @tool
        def listy(items: list[int]) -> int:
            """Sum the items.

            Args:
                items: Numbers to add.
            """
            return sum(items)

        schema = listy.to_schema()["input_schema"]["properties"]["items"]
        assert schema == {"type": "array", "items": {"type": "integer"}, "description": "Numbers to add."}

    def test_optional_annotation_unwraps(self):
        @tool
        def maybe(value: str | None = None) -> str:
            """Maybe do a thing.

            Args:
                value: Optional value.
            """
            return value or ""

        assert maybe.to_schema()["input_schema"]["properties"]["value"]["type"] == "string"

    def test_strict_mode_requires_everything(self):
        @tool(strict=True)
        def exact(a: str, b: int = 2) -> str:
            """Do it exactly.

            Args:
                a: First.
                b: Second.
            """
            return a

        schema = exact.to_schema()
        assert schema["strict"] is True
        assert schema["input_schema"]["required"] == ["a", "b"]
        assert schema["input_schema"]["additionalProperties"] is False

    def test_the_original_function_is_still_callable(self):
        assert echo_tool("hi", 2) == "hi hi"


class TestToolExecution:
    def test_success(self):
        result = echo_tool.run(call_id="1", text="hi", times=2)
        assert result.content == "hi hi"
        assert result.is_error is False

    def test_tool_error_becomes_an_error_result(self):
        result = strict_tool.run(call_id="1", ticket="bad", status="open")
        assert result.is_error is True
        assert "start with T-" in result.content

    def test_unexpected_exception_is_reported_not_raised(self):
        @tool
        def explodes() -> str:
            """Always fails."""
            raise ZeroDivisionError("boom")

        result = explodes.run(call_id="1")
        assert result.is_error is True
        assert "ZeroDivisionError: boom" in result.content

    def test_missing_argument_is_reported(self):
        result = echo_tool.run(call_id="1")
        assert result.is_error is True
        assert "missing argument" in result.content

    def test_unknown_argument_is_reported(self):
        result = echo_tool.run(call_id="1", text="hi", nope=1)
        assert result.is_error is True
        assert "unknown argument" in result.content

    def test_result_serialises_non_strings(self):
        @tool
        def structured() -> dict:
            """Return a mapping."""
            return {"a": 1}

        assert structured.run(call_id="1").as_text() == '{"a": 1}'


class TestToolbox:
    def test_plain_functions_are_wrapped(self):
        def plain(x: str) -> str:
            """Plain function."""
            return x

        box = Toolbox([plain])
        assert isinstance(box.get("plain"), Tool)

    def test_duplicate_names_are_rejected(self):
        clash = Tool(
            name=echo_tool.name,
            description="A different tool with the same name.",
            parameters={"type": "object", "properties": {}},
            fn=lambda: "",
        )
        with pytest.raises(ValidationError, match="both named"):
            Toolbox([echo_tool, clash])

    def test_membership_and_length(self):
        box = Toolbox([echo_tool, strict_tool])
        assert len(box) == 2
        assert "echo_tool" in box


class AgentUnderTest(Agent):
    """An agent for the test suite."""

    class Meta:
        system = "You are under test."
        tools = [echo_tool, strict_tool]
        max_steps = 5


class TestAgentLoop:
    def test_plain_reply(self, project):
        result = AgentUnderTest().run("hello")
        assert result.output == "echo: hello"
        assert result.steps == 1
        assert result.ok

    def test_tool_call_round_trip(self, project):
        result = AgentUnderTest().run('use echo_tool {"text": "hi", "times": 2}')
        assert result.steps == 2
        assert result.tools_used == ["echo_tool"]
        assert "hi hi" in result.output

    def test_tool_failure_is_recovered_from(self, project):
        result = AgentUnderTest().run('use strict_tool {"ticket": "bad", "status": "open"}')
        assert result.ok
        assert "start with T-" in result.output

    def test_usage_accumulates_across_steps(self, project):
        result = AgentUnderTest().run('use echo_tool {"text": "hi"}')
        assert result.usage.total_tokens > 0
        assert result.usage.input_tokens > 0

    def test_unknown_tool_is_reported_to_the_model(self, project):
        from mlango.agents.providers.base import Completion, ToolCall

        agent = AgentUnderTest()
        completion = Completion(
            tool_calls=[ToolCall(id="1", name="nope", arguments={})],
            stop_reason=Completion.TOOL_USE,
        )

        class FakeTracer:
            def span(self, *args, **kwargs):
                from contextlib import nullcontext

                return nullcontext({})

        results = agent._dispatch(agent.get_tools(), completion, FakeTracer())
        assert results[0].is_error
        assert "No tool named" in results[0].content

    def test_step_limit_is_enforced(self, project):
        class Looping(Agent):
            """Always asks for the same tool again."""

            class Meta:
                tools = [echo_tool]
                max_steps = 2

        from mlango.agents.providers.base import Completion, ToolCall, Usage
        from mlango.agents.providers.echo import EchoProvider

        class AlwaysTool(EchoProvider):
            def complete(self, **kwargs):
                return Completion(
                    tool_calls=[ToolCall(id="x", name="echo_tool", arguments={"text": "again"})],
                    stop_reason=Completion.TOOL_USE,
                    usage=Usage(input_tokens=1, output_tokens=1),
                )

        agent = Looping()
        agent.get_provider = lambda: AlwaysTool()  # type: ignore[method-assign]
        result = agent.run("go")
        assert result.steps == 2
        assert "without finishing" in result.error


class TestMemory:
    def test_null_memory_forgets_everything(self):
        memory = NullMemory()
        memory.append("s", [{"role": "user", "content": "hi"}])
        assert memory.load("s") == []

    def test_buffer_memory_keeps_the_last_k(self):
        memory = BufferMemory(k=2)
        memory.append("s", [{"role": "user", "content": str(i)} for i in range(5)])
        assert [m["content"] for m in memory.load("s")] == ["3", "4"]

    def test_buffer_memory_is_per_session(self):
        memory = BufferMemory(k=5)
        memory.append("a", [{"role": "user", "content": "a"}])
        memory.append("b", [{"role": "user", "content": "b"}])
        assert len(memory.load("a")) == 1

    def test_window_memory_keeps_the_anchor(self):
        memory = WindowMemory(k=2, keep_first=1)
        memory.append("s", [{"role": "user", "content": str(i)} for i in range(5)])
        contents = [m["content"] for m in memory.load("s")]
        assert contents[0] == "0"
        assert contents[-1] == "4"

    def test_agent_uses_memory_across_calls(self, project):
        class Remembering(Agent):
            """Keeps a buffer."""

            class Meta:
                memory = BufferMemory(k=10)

        agent = Remembering()
        agent.run("first", session_id="s1")
        result = agent.run("second", session_id="s1")
        # The prior turn is replayed into the request.
        assert len(result.messages) > 2

    def test_clear_forgets_a_session(self):
        memory = BufferMemory(k=5)
        memory.append("s", [{"role": "user", "content": "x"}])
        memory.clear("s")
        assert memory.load("s") == []


class TestTracing:
    def test_a_trace_is_written(self, project):
        from mlango.agents.tracing import get_trace

        result = AgentUnderTest().run("hello")
        trace = get_trace(result.trace_uuid)
        assert trace is not None
        assert trace.status == "finished"
        assert trace.input == "hello"

    def test_spans_record_the_sequence(self, project):
        from mlango.agents.tracing import get_trace

        result = AgentUnderTest().run('use echo_tool {"text": "hi"}')
        trace = get_trace(result.trace_uuid)
        assert [s.kind for s in trace.spans] == ["llm", "tool", "llm"]

    def test_spans_are_ordered(self, project):
        from mlango.agents.tracing import get_trace

        result = AgentUnderTest().run('use echo_tool {"text": "hi"}')
        orderings = [s.ordering for s in get_trace(result.trace_uuid).spans]
        assert orderings == sorted(orderings)

    def test_tracing_can_be_switched_off(self, project):
        class Untraced(Agent):
            """No tracing."""

            class Meta:
                tracing = False

        result = Untraced().run("hello")
        assert result.trace_uuid == ""

    def test_recent_traces_filters_by_agent(self, project):
        from mlango.agents.tracing import recent_traces

        AgentUnderTest().run("hello")
        traces = recent_traces(agent=AgentUnderTest._meta.label)
        assert len(traces) == 1


class TestProviders:
    def test_echo_provider_is_offline(self, project):
        from mlango.agents.providers import get_provider

        provider = get_provider("echo")
        assert provider.offline is True

    def test_unknown_provider_lists_alternatives(self, project):
        from mlango.agents.providers import get_provider
        from mlango.core.exceptions import ImproperlyConfigured

        with pytest.raises(ImproperlyConfigured, match="Available:"):
            get_provider("nope")

    def test_anthropic_provider_drops_sampling_parameters(self, monkeypatch, project):
        """Current Claude models 400 on temperature/top_p/top_k."""
        pytest.importorskip("anthropic")
        from mlango.agents.providers.anthropic import AnthropicProvider

        captured: dict = {}

        class FakeMessages:
            def create(self, **payload):
                captured.update(payload)
                raise RuntimeError("stop here")

        class FakeClient:
            messages = FakeMessages()

        provider = AnthropicProvider()
        provider._client = FakeClient()

        with pytest.raises(Exception):
            provider.complete(
                model="claude-opus-5",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.7,
                top_p=0.9,
            )

        assert "temperature" not in captured
        assert "top_p" not in captured
        assert captured["thinking"] == {"type": "adaptive"}

    def test_refusal_is_surfaced(self, project):
        from mlango.agents.providers.base import Completion, Usage
        from mlango.agents.providers.echo import EchoProvider

        class Refusing(EchoProvider):
            def complete(self, **kwargs):
                return Completion(
                    stop_reason=Completion.REFUSAL,
                    refusal={"category": "cyber", "explanation": "nope"},
                    usage=Usage(),
                )

        agent = AgentUnderTest()
        agent.get_provider = lambda: Refusing()  # type: ignore[method-assign]
        result = agent.run("something")
        assert "declined" in result.error
        assert "cyber" in result.error
