"""Agent.stream() and the Server-Sent Events endpoint."""

from __future__ import annotations

import json

import pytest

from mlango.agents import (
    Agent,
    Failed,
    Finished,
    Started,
    StepFinished,
    TextChunk,
    Thinking,
    ToolCalled,
    ToolFinished,
    tool,
)

pytestmark = pytest.mark.usefixtures("isolated_registry")


@tool
def shout(text: str) -> str:
    """Uppercase the text.

    Args:
        text: What to shout.
    """
    return text.upper()


@tool
def explodes() -> str:
    """Always fails."""
    raise ValueError("boom")


class Streamer(Agent):
    """Streams for the tests."""

    class Meta:
        tools = [shout, explodes]
        max_steps = 5


class TestEventShape:
    def test_kind_is_snake_case(self):
        assert Started().kind == "started"
        assert TextChunk().kind == "text_chunk"
        assert ToolFinished().kind == "tool_finished"

    def test_describe_is_json_serialisable(self):
        payload = ToolCalled(step=1, name="shout", arguments={"text": "hi"}).describe()
        json.dumps(payload)
        assert payload["event"] == "tool_called"
        assert payload["arguments"] == {"text": "hi"}

    def test_finished_describe_omits_the_result_object(self):
        finished = Finished(output="hi", result=object())
        payload = finished.describe()
        json.dumps(payload)
        assert "result" not in payload


class TestStreaming:
    def test_a_plain_turn_emits_the_expected_sequence(self, project):
        kinds = [event.kind for event in Streamer().stream("hello")]
        assert kinds == ["started", "thinking", "text_chunk", "step_finished", "finished"]

    def test_a_tool_turn_emits_call_and_result(self, project):
        events = list(Streamer().stream('use shout {"text": "hi"}'))
        kinds = [e.kind for e in events]

        assert "tool_called" in kinds
        assert "tool_finished" in kinds
        assert kinds.index("tool_called") < kinds.index("tool_finished")

        called = next(e for e in events if isinstance(e, ToolCalled))
        assert called.name == "shout"
        assert called.arguments == {"text": "hi"}

        done = next(e for e in events if isinstance(e, ToolFinished))
        assert done.content == "HI"
        assert done.is_error is False

    def test_a_failing_tool_is_reported_not_raised(self, project):
        events = list(Streamer().stream("use explodes {}"))
        done = next(e for e in events if isinstance(e, ToolFinished))
        assert done.is_error is True
        assert "ValueError: boom" in done.content
        assert next(e for e in events if isinstance(e, Finished)).ok

    def test_thinking_precedes_every_model_call(self, project):
        events = list(Streamer().stream('use shout {"text": "hi"}'))
        thinking = [e for e in events if isinstance(e, Thinking)]
        assert len(thinking) == 2  # one before each of the two model calls
        assert thinking[0].model

    def test_step_numbers_increase(self, project):
        events = list(Streamer().stream('use shout {"text": "hi"}'))
        steps = [e.step for e in events if e.step]
        assert steps == sorted(steps)

    def test_usage_is_reported_per_step(self, project):
        events = list(Streamer().stream("hello"))
        step = next(e for e in events if isinstance(e, StepFinished))
        assert step.input_tokens > 0
        assert step.stop_reason == "end_turn"

    def test_the_final_event_carries_the_same_result_as_run(self, project):
        streamed = next(
            e for e in Streamer().stream('use shout {"text": "hi"}') if isinstance(e, Finished)
        )
        direct = Streamer().run('use shout {"text": "hi"}')

        # One loop, two views of it: the two must agree.
        assert streamed.output == direct.output
        assert streamed.tools_used == direct.tools_used
        assert streamed.usage["total_tokens"] == direct.usage.total_tokens
        assert streamed.result.steps == direct.steps

    def test_the_trace_is_closed_by_streaming_too(self, project):
        from mlango.agents.tracing import get_trace

        finished = next(e for e in Streamer().stream("hello") if isinstance(e, Finished))
        trace = get_trace(finished.trace)
        assert trace is not None
        assert trace.status == "finished"
        assert trace.output == finished.output

    def test_memory_is_updated_by_streaming_too(self, project):
        from mlango.agents import BufferMemory

        memory = BufferMemory(k=10)
        list(Streamer().stream("first", session_id="s1", memory=memory))
        assert len(memory.load("s1")) == 2

    def test_a_refusal_surfaces_as_an_error_on_the_final_event(self, project):
        from mlango.agents.providers.base import Completion, Usage
        from mlango.agents.providers.echo import EchoProvider

        class Refusing(EchoProvider):
            def complete(self, **kwargs):
                return Completion(
                    stop_reason=Completion.REFUSAL,
                    refusal={"category": "cyber", "explanation": "no"},
                    usage=Usage(),
                )

        agent = Streamer()
        agent.get_provider = lambda: Refusing()  # type: ignore[method-assign]
        finished = next(e for e in agent.stream("something") if isinstance(e, Finished))
        assert not finished.ok
        assert "declined" in finished.error

    def test_a_provider_failure_yields_failed_then_raises(self, project):
        from mlango.agents.providers.echo import EchoProvider
        from mlango.core.exceptions import ProviderError

        class Broken(EchoProvider):
            def complete(self, **kwargs):
                raise ProviderError("upstream is down")

        agent = Streamer()
        agent.get_provider = lambda: Broken()  # type: ignore[method-assign]

        seen = []
        with pytest.raises(ProviderError, match="upstream is down"):
            for event in agent.stream("hello"):
                seen.append(event)

        assert isinstance(seen[-1], Failed)
        assert seen[-1].exception_type == "ProviderError"

    def test_the_generator_is_lazy(self, project):
        stream = Streamer().stream("hello")
        # Nothing has run yet; the trace only opens on the first next().
        first = next(stream)
        assert isinstance(first, Started)
        stream.close()


class TestStreamEndpoint:
    def test_serves_server_sent_events(self, project):
        from fastapi.testclient import TestClient

        from mlango.serve import path
        from mlango.serve.api import create_app

        app = create_app(
            include_admin=False,
            routes=[path("stream/", Streamer.as_stream_endpoint())],
        )

        with TestClient(app) as client:
            response = client.post("/api/stream/", json={"message": "hello"})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = [
            line[len("event: ") :]
            for line in response.text.splitlines()
            if line.startswith("event: ")
        ]
        assert events[0] == "started"
        assert events[-1] == "finished"

    def test_every_data_line_is_valid_json(self, project):
        from fastapi.testclient import TestClient

        from mlango.serve import path
        from mlango.serve.api import create_app

        app = create_app(
            include_admin=False,
            routes=[path("stream/", Streamer.as_stream_endpoint())],
        )

        with TestClient(app) as client:
            response = client.post("/api/stream/", json={"message": 'use shout {"text": "hi"}'})

        payloads = [
            json.loads(line[len("data: ") :])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert payloads
        assert {p["event"] for p in payloads} >= {"started", "tool_called", "finished"}

    def test_the_endpoint_advertises_itself_as_streaming(self, project):
        endpoint = Streamer.as_stream_endpoint()
        assert endpoint.meta["streaming"] is True
        assert "text_chunk" in endpoint.description
