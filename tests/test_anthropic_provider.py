"""The Anthropic provider, against a stand-in for the SDK.

The provider does one job: build a request payload and translate one response
into a ``Completion``. That is testable without the real package — and has to
be, because ``anthropic`` is not in any extra CI installs, so the single test
that guarded this module was skipped everywhere and it was the least covered
file in the project.

What a fake buys and what it does not: it pins our half of the contract — which
parameters we send, which we refuse to send, how each block type and stop reason
is read. It cannot notice the SDK changing shape underneath us. The response
shapes below follow the documented API, and the notes in the provider's own
docstring are what they are checking.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from typing import Any

import pytest

from mlango.agents.providers.base import Completion
from mlango.core.exceptions import ProviderError

# --------------------------------------------------------------------------- #
# A stand-in for the SDK
# --------------------------------------------------------------------------- #


class Block:
    """One content block, as the SDK exposes it: a `.type` plus its own fields."""

    def __init__(self, type: str, **fields: Any):
        self.type = type
        for key, value in fields.items():
            setattr(self, key, value)


class Response:
    def __init__(
        self,
        content: list[Block] | None = None,
        *,
        stop_reason: str = "end_turn",
        stop_details: Any = None,
        model: str = "claude-opus-5",
        usage: Any = None,
    ):
        self.content = content if content is not None else []
        self.stop_reason = stop_reason
        self.stop_details = stop_details
        self.model = model
        self.usage = usage


def usage(**fields: int) -> Any:
    defaults = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    return types.SimpleNamespace(**{**defaults, **fields})


@pytest.fixture
def sdk(monkeypatch):
    """Install a fake ``anthropic`` module and hand back a handle to it.

    ``__spec__`` is set because ``Provider.check_available`` goes through
    ``importlib.util.find_spec``, which raises on a module without one.
    """
    module = types.ModuleType("anthropic")
    module.__spec__ = importlib.machinery.ModuleSpec("anthropic", loader=None)

    class APIError(Exception):
        pass

    class APIStatusError(APIError):
        def __init__(self, message: str = "", status_code: int = 400):
            super().__init__(message)
            self.message = message
            self.status_code = status_code

    class RateLimitError(APIStatusError):
        """Subclasses APIStatusError, as in the real SDK — so catch order matters."""

    class APIConnectionError(APIError):
        pass

    calls: list[dict[str, Any]] = []
    scripted: list[Any] = []

    class Messages:
        def create(self, **payload: Any) -> Any:
            calls.append(payload)
            if not scripted:
                return Response([Block("text", text="ok")], usage=usage())
            outcome = scripted.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    class Anthropic:
        def __init__(self, **options: Any):
            self.options = options
            self.messages = Messages()

    module.Anthropic = Anthropic
    module.APIError = APIError
    module.APIStatusError = APIStatusError
    module.RateLimitError = RateLimitError
    module.APIConnectionError = APIConnectionError

    monkeypatch.setitem(sys.modules, "anthropic", module)
    module.calls = calls
    module.scripted = scripted
    return module


@pytest.fixture
def provider(sdk):
    from mlango.agents.providers.anthropic import AnthropicProvider

    return AnthropicProvider()


def complete(provider, **overrides: Any) -> Completion:
    request = {"model": "claude-opus-5", "messages": [{"role": "user", "content": "hi"}]}
    request.update(overrides)
    return provider.complete(**request)


# --------------------------------------------------------------------------- #
# The request
# --------------------------------------------------------------------------- #


class TestPayload:
    def test_the_minimum_is_model_messages_and_max_tokens(self, provider, sdk):
        complete(provider)

        payload = sdk.calls[0]
        assert payload["model"] == "claude-opus-5"
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        assert payload["max_tokens"] == 4096

    def test_thinking_is_adaptive_by_default(self, provider, sdk):
        """budget_tokens is a 400 on current models; the type form is what works."""
        complete(provider)
        assert sdk.calls[0]["thinking"] == {"type": "adaptive"}

    def test_thinking_can_be_turned_off(self, provider, sdk):
        complete(provider, thinking=None)
        assert "thinking" not in sdk.calls[0]

    @pytest.mark.parametrize("rejected", ["temperature", "top_p", "top_k"])
    def test_sampling_parameters_are_never_sent(self, provider, sdk, caplog, rejected):
        """Current Claude models reject these with a 400."""
        complete(provider, **{rejected: 0.5})

        assert rejected not in sdk.calls[0]
        assert rejected in caplog.text

    def test_all_three_are_dropped_at_once(self, provider, sdk):
        complete(provider, temperature=0.7, top_p=0.9, top_k=40)

        payload = sdk.calls[0]
        assert not {"temperature", "top_p", "top_k"} & set(payload)

    def test_a_sampling_parameter_of_zero_is_still_dropped(self, provider, sdk):
        """`temperature=0` is the value people reach for most, and it also 400s."""
        complete(provider, temperature=0)
        assert "temperature" not in sdk.calls[0]

    def test_optional_parts_are_omitted_when_empty(self, provider, sdk):
        complete(provider, system="", tools=[], stop_sequences=[])

        payload = sdk.calls[0]
        for key in ("system", "tools", "stop_sequences"):
            assert key not in payload

    def test_optional_parts_are_passed_through_when_given(self, provider, sdk):
        tools = [{"name": "search", "input_schema": {"type": "object"}}]
        complete(provider, system="be brief", tools=tools, stop_sequences=["STOP"])

        payload = sdk.calls[0]
        assert payload["system"] == "be brief"
        assert payload["tools"] == tools
        assert payload["stop_sequences"] == ["STOP"]

    def test_effort_becomes_output_config(self, provider, sdk):
        complete(provider, effort="high")
        assert sdk.calls[0]["output_config"] == {"effort": "high"}

    @pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh", "max"])
    def test_every_documented_effort_level(self, provider, sdk, level):
        complete(provider, effort=level)
        assert sdk.calls[0]["output_config"] == {"effort": level}

    def test_an_unknown_effort_lists_the_valid_ones(self, provider, sdk):
        with pytest.raises(ProviderError, match="low, medium, high, xhigh, max"):
            complete(provider, effort="turbo")

        assert sdk.calls == []  # rejected before the request went out

    def test_unrecognised_keywords_are_forwarded(self, provider, sdk):
        """New API parameters must not need a provider change to reach the API."""
        complete(provider, metadata={"user_id": "u1"}, service_tier="standard_only")

        payload = sdk.calls[0]
        assert payload["metadata"] == {"user_id": "u1"}
        assert payload["service_tier"] == "standard_only"


class TestClient:
    def test_the_client_is_built_once_and_reused(self, provider, sdk):
        first = provider.client
        assert provider.client is first

    def test_no_api_key_is_passed(self, provider, sdk):
        """The SDK resolves the key itself; passing one defeats profile auth."""
        assert provider.client.options == {}

    def test_constructor_options_reach_the_client(self, sdk):
        from mlango.agents.providers.anthropic import AnthropicProvider

        built = AnthropicProvider(base_url="https://proxy.internal", max_retries=5)
        assert built.client.options == {
            "base_url": "https://proxy.internal",
            "max_retries": 5,
        }

    def test_it_reports_itself_as_online(self, provider, sdk):
        described = provider.describe()
        assert described == {"provider": "anthropic", "offline": False, "sdk": "anthropic"}

    def test_it_resolves_through_the_registry(self, project, sdk):
        from mlango.agents.providers import clear_provider_cache, get_provider

        clear_provider_cache()
        try:
            assert get_provider("anthropic").name == "anthropic"
        finally:
            clear_provider_cache()


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class TestErrors:
    def test_a_rate_limit_says_so(self, provider, sdk):
        sdk.scripted.append(sdk.RateLimitError("slow down", status_code=429))

        with pytest.raises(ProviderError, match="rate limit"):
            complete(provider)

    def test_a_rate_limit_is_not_swallowed_by_the_status_handler(self, provider, sdk):
        """RateLimitError subclasses APIStatusError, so the order of except matters."""
        sdk.scripted.append(sdk.RateLimitError("slow down", status_code=429))

        with pytest.raises(ProviderError) as caught:
            complete(provider)
        assert "429" not in str(caught.value)

    def test_a_status_error_carries_the_code(self, provider, sdk):
        sdk.scripted.append(sdk.APIStatusError("bad request", status_code=400))

        with pytest.raises(ProviderError, match="error 400: bad request"):
            complete(provider)

    def test_a_connection_error_says_it_could_not_reach_the_api(self, provider, sdk):
        sdk.scripted.append(sdk.APIConnectionError("dns failed"))

        with pytest.raises(ProviderError, match="Could not reach"):
            complete(provider)

    def test_an_unexpected_error_is_not_disguised_as_a_provider_error(self, provider, sdk):
        """A bug in our own code must not come back looking like an API failure."""
        sdk.scripted.append(TypeError("that is our fault"))

        with pytest.raises(TypeError, match="our fault"):
            complete(provider)


# --------------------------------------------------------------------------- #
# The response
# --------------------------------------------------------------------------- #


class TestTranslation:
    def test_text_blocks_are_joined(self, provider, sdk):
        sdk.scripted.append(
            Response([Block("text", text="Hello "), Block("text", text="there.")], usage=usage())
        )
        assert complete(provider).text == "Hello there."

    def test_a_tool_call_is_normalised(self, provider, sdk):
        sdk.scripted.append(
            Response(
                [Block("tool_use", id="tu_1", name="search", input={"q": "weather"})],
                stop_reason="tool_use",
                usage=usage(),
            )
        )
        result = complete(provider)

        assert result.wants_tools
        call = result.tool_calls[0]
        assert (call.id, call.name, call.arguments) == ("tu_1", "search", {"q": "weather"})
        assert result.stop_reason == Completion.TOOL_USE

    def test_a_tool_call_with_no_input(self, provider, sdk):
        sdk.scripted.append(
            Response([Block("tool_use", id="tu_1", name="now", input=None)], usage=usage())
        )
        assert complete(provider).tool_calls[0].arguments == {}

    def test_text_and_tool_calls_together(self, provider, sdk):
        sdk.scripted.append(
            Response(
                [
                    Block("text", text="Looking that up."),
                    Block("tool_use", id="tu_1", name="search", input={"q": "x"}),
                ],
                stop_reason="tool_use",
                usage=usage(),
            )
        )
        result = complete(provider)

        assert result.text == "Looking that up."
        assert len(result.tool_calls) == 1

    def test_thinking_blocks_stay_out_of_the_text(self, provider, sdk):
        """Reasoning is not the answer, but it must survive in raw_content.

        The assistant turn is echoed back verbatim on the next request, and
        dropping a thinking block there invalidates the turn.
        """
        blocks = [
            Block("thinking", thinking="working it out", signature="sig"),
            Block("text", text="42"),
        ]
        sdk.scripted.append(Response(blocks, usage=usage()))
        result = complete(provider)

        assert result.text == "42"
        assert result.raw_content is blocks

    def test_an_unknown_block_type_is_ignored_not_fatal(self, provider, sdk):
        """A block type added server-side must not break an existing client."""
        sdk.scripted.append(
            Response(
                [Block("something_new", payload={"a": 1}), Block("text", text="still fine")],
                usage=usage(),
            )
        )
        assert complete(provider).text == "still fine"

    def test_empty_content(self, provider, sdk):
        sdk.scripted.append(Response([], usage=usage()))
        result = complete(provider)

        assert result.text == ""
        assert result.tool_calls == []

    def test_content_of_none(self, provider, sdk):
        sdk.scripted.append(Response(None, usage=usage()))
        assert complete(provider).text == ""

    def test_the_model_that_answered_is_recorded(self, provider, sdk):
        sdk.scripted.append(Response([], model="claude-sonnet-5", usage=usage()))
        assert complete(provider).model == "claude-sonnet-5"


class TestUsage:
    def test_tokens_are_carried_over(self, provider, sdk):
        sdk.scripted.append(Response([], usage=usage(input_tokens=120, output_tokens=45)))
        result = complete(provider)

        assert result.usage.input_tokens == 120
        assert result.usage.output_tokens == 45
        assert result.usage.total_tokens == 165

    def test_cache_tokens_use_the_sdk_s_names(self, provider, sdk):
        """cache_creation_input_tokens / cache_read_input_tokens, not our names."""
        sdk.scripted.append(
            Response(
                [],
                usage=usage(cache_read_input_tokens=900, cache_creation_input_tokens=300),
            )
        )
        result = complete(provider)

        assert result.usage.cache_read_tokens == 900
        assert result.usage.cache_write_tokens == 300

    def test_a_response_without_usage(self, provider, sdk):
        sdk.scripted.append(Response([], usage=None))
        assert complete(provider).usage.total_tokens == 0

    def test_none_token_counts_become_zero(self, provider, sdk):
        sdk.scripted.append(Response([], usage=usage(input_tokens=None, output_tokens=None)))
        assert complete(provider).usage.total_tokens == 0


class TestStopReasons:
    def test_end_turn(self, provider, sdk):
        sdk.scripted.append(Response([Block("text", text="done")], usage=usage()))
        assert complete(provider).stop_reason == Completion.END_TURN

    def test_max_tokens(self, provider, sdk):
        sdk.scripted.append(Response([], stop_reason="max_tokens", usage=usage()))
        assert complete(provider).stop_reason == Completion.MAX_TOKENS

    def test_pause_turn(self, provider, sdk):
        sdk.scripted.append(Response([], stop_reason="pause_turn", usage=usage()))
        assert complete(provider).stop_reason == Completion.PAUSE_TURN

    def test_a_missing_stop_reason_defaults_to_end_turn(self, provider, sdk):
        sdk.scripted.append(Response([], stop_reason=None, usage=usage()))
        assert complete(provider).stop_reason == Completion.END_TURN

    def test_a_refusal_is_reported_with_its_category(self, provider, sdk):
        """A refusal is HTTP 200 with empty content — reading content first hides it."""
        sdk.scripted.append(
            Response(
                [],
                stop_reason="refusal",
                stop_details=types.SimpleNamespace(
                    category="cyber", explanation="declined to help with that"
                ),
                usage=usage(),
            )
        )
        result = complete(provider)

        assert result.stop_reason == Completion.REFUSAL
        assert result.refusal == {
            "category": "cyber",
            "explanation": "declined to help with that",
        }

    def test_a_refusal_without_details(self, provider, sdk):
        sdk.scripted.append(Response([], stop_reason="refusal", usage=usage()))
        result = complete(provider)

        assert result.refusal == {"category": None, "explanation": ""}

    def test_refusal_is_absent_on_a_normal_turn(self, provider, sdk):
        sdk.scripted.append(Response([Block("text", text="hi")], usage=usage()))
        assert complete(provider).refusal is None


class TestThroughAnAgent:
    def test_an_agent_runs_against_the_provider(self, project, sdk):
        """The loop and the provider have to agree, not just each be right alone."""
        from mlango.agents import Agent, tool
        from mlango.agents.providers import clear_provider_cache
        from mlango.conf import settings
        from mlango.core.registry import apps

        @tool
        def shout(text: str) -> str:
            """Upper-case the text.

            Args:
                text: What to shout.
            """
            return text.upper()

        class Loud(Agent):
            """Calls one tool, then answers."""

            class Meta:
                provider = "anthropic"
                tools = [shout]

        sdk.scripted.extend(
            [
                Response(
                    [Block("tool_use", id="tu_1", name="shout", input={"text": "hello"})],
                    stop_reason="tool_use",
                    usage=usage(input_tokens=10, output_tokens=5),
                ),
                Response(
                    [Block("text", text="I said HELLO.")],
                    usage=usage(input_tokens=20, output_tokens=6),
                ),
            ]
        )

        settings.DEFAULT_AGENT_MODEL = "claude-opus-5"
        clear_provider_cache()
        try:
            result = Loud().run("say hello")
        finally:
            clear_provider_cache()
            apps.unregister("agent", Loud._meta.label)

        assert result.output == "I said HELLO."
        assert result.tools_used == ["shout"]
        assert result.steps == 2
        # Usage accumulates across both turns.
        assert result.usage.input_tokens == 30
        assert result.usage.output_tokens == 11

        # The tool result is echoed back in one user message, as the API requires.
        second_request = sdk.calls[1]
        tool_results = [
            block
            for message in second_request["messages"]
            if message["role"] == "user"
            for block in (message["content"] if isinstance(message["content"], list) else [])
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_use_id"] == "tu_1"

    def test_a_refusal_reaches_the_agent_result(self, project, sdk):
        from mlango.agents import Agent
        from mlango.agents.providers import clear_provider_cache
        from mlango.core.registry import apps

        class Careful(Agent):
            """Gets refused."""

            class Meta:
                provider = "anthropic"

        sdk.scripted.append(
            Response(
                [],
                stop_reason="refusal",
                stop_details=types.SimpleNamespace(category="cyber", explanation="no"),
                usage=usage(),
            )
        )

        clear_provider_cache()
        try:
            result = Careful().run("do something questionable")
        finally:
            clear_provider_cache()
            apps.unregister("agent", Careful._meta.label)

        assert "declined" in result.error
        assert "cyber" in result.error
