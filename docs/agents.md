# Agents

An `Agent` declares a model, a system prompt, tools and memory. The framework
owns the tool-use loop, tool dispatch, usage accounting and tracing.

```python
from mlango.agents import Agent, BufferMemory, tool


@tool
def search_docs(query: str, limit: int = 5) -> list[str]:
    """Search the product documentation.

    Args:
        query: What to search for.
        limit: Maximum number of results.
    """
    return retrieve(query, limit)


class Support(Agent):
    """Answers product questions from the docs."""

    class Meta:
        model = "claude-opus-5"
        system = "You are a support engineer. Cite the docs you used."
        tools = [search_docs]
        memory = BufferMemory(k=20)
        max_steps = 8
        effort = "high"
```

```python
result = Support().run("How do I rotate an API key?")
result.output
result.tools_used
result.usage.total_tokens
result.trace_uuid
```

```bash
python manage.py agent support.Support                    # interactive
python manage.py agent support.Support "how do I..."      # one shot
python manage.py agent support.Support "..." --show-steps
```

## `Meta` options

| Option | Default | Purpose |
|---|---|---|
| `model` | `DEFAULT_AGENT_MODEL` | Model id |
| `system` | `""` | System prompt |
| `tools` | `[]` | `@tool` functions or `Tool` instances |
| `provider` | `DEFAULT_PROVIDER` | A key from the `PROVIDERS` setting |
| `memory` | `NullMemory()` | Conversation memory backend |
| `max_steps` | `AGENT_MAX_STEPS` | Hard stop on the tool-use loop |
| `max_tokens` | `4096` | Output cap per model call |
| `thinking` | `"adaptive"` | Thinking mode; `None` omits the parameter |
| `effort` | unset | `low`, `medium`, `high`, `xhigh`, `max` |
| `tracing` | `TRACING` | Record spans for this agent |

Build the prompt from the agent's own fields by overriding `get_system()`:

```python
class Support(Agent):
    tone = fields.ChoiceField(["formal", "friendly"], default="friendly")

    def get_system(self) -> str:
        return f"You are a support engineer. Write in a {self.tone} tone."
```

## Tools

The `@tool` decorator reads your type hints and docstring and produces the JSON
Schema the model needs, so a tool is described in exactly one place.

```python
@tool
def set_status(ticket: str, status: Literal["open", "closed"]) -> str:
    """Change a ticket's status.

    Args:
        ticket: Ticket id, which must start with T-.
        status: The new status.
    """
    if not ticket.startswith("T-"):
        raise ToolError("Ticket ids start with T-.")
    return f"{ticket} -> {status}"
```

| Annotation | Schema |
|---|---|
| `str`, `int`, `float`, `bool` | `string`, `integer`, `number`, `boolean` |
| `Literal["a", "b"]` | `string` with an `enum` |
| `list[int]` | `array` of `integer` |
| `dict[str, int]` | `object` |
| `X \| None` | `X`, and not required |
| An `Enum` subclass | `string` with an `enum` of its values |
| A parameter with a default | not required, `default` recorded |

Google-style `Args:` entries become per-property descriptions, including ones
wrapped over several lines.

### Errors are for the model, not for you

A tool that raises does not take the agent down. The exception becomes an error
result the model can read and recover from:

```python
raise ToolError("Ticket ids start with T-.")   # a message you wrote
raise ValueError("boom")                       # becomes "ValueError: boom"
```

That is deliberate: an unhandled exception mid-loop loses the whole trace, and
the model is usually able to correct itself when told what went wrong.

### Strict mode

```python
@tool(strict=True)
def book(destination: str, passengers: int) -> str:
    ...
```

Guarantees the arguments validate exactly against the schema.

## Memory

| Backend | Keeps | Survives a restart |
|---|---|---|
| `NullMemory()` | nothing — every run starts fresh (default) | — |
| `BufferMemory(k=20)` | the last `k` messages, in process | no |
| `WindowMemory(k=20, keep_first=1)` | the anchor turn plus the last `k` | no |
| `MetastoreMemory(max_turns=20)` | rebuilt from recorded traces | yes |

`MetastoreMemory` stores nothing extra: since every call is traced anyway, the
conversation is reconstructed from those rows. One source of truth, so the
admin's trace view and the agent's memory cannot disagree.

Memory is keyed by `session_id`:

```python
agent.run("My name is Ada", session_id="user-42")
agent.run("What is my name?", session_id="user-42")
```

## Providers

| Provider | Notes |
|---|---|
| `anthropic` | Claude. Needs `pip install "mlango[anthropic]"` and credentials |
| `echo` | Deterministic, offline. No credentials, no cost |

The `echo` provider is why the framework's own test suite runs in CI for free,
and why a fresh project works before you have an API key. It follows simple
scripted rules — `use <tool> {json}` triggers a tool call — which is enough to
exercise a full multi-step loop.

Switch in settings:

```python
DEFAULT_PROVIDER = "anthropic"
DEFAULT_AGENT_MODEL = "claude-opus-5"
```

!!! note "Sampling parameters"
    Current Claude models reject `temperature`, `top_p` and `top_k`. The provider
    drops them with a warning rather than letting the request fail, and depth is
    controlled with `effort` instead.

Adding a provider is one class:

```python
from mlango.agents.providers import Completion, Provider, ToolCall, Usage


class VLLMProvider(Provider):
    name = "vllm"
    requires = ("openai",)

    def complete(self, *, model, messages, system="", tools=None,
                 max_tokens=4096, thinking=None, effort=None, **kw) -> Completion:
        response = call_your_server(...)
        return Completion(
            text=response.text,
            tool_calls=[ToolCall(id=c.id, name=c.name, arguments=c.args)
                        for c in response.calls],
            stop_reason=Completion.TOOL_USE if response.calls else Completion.END_TURN,
            usage=Usage(input_tokens=response.prompt_tokens,
                        output_tokens=response.completion_tokens),
        )
```

```python
PROVIDERS = {"vllm": "myproject.providers.VLLMProvider"}
```

A provider does exactly one thing: turn a request into one response. The loop,
memory and tracing stay the framework's, so swapping providers never changes
agent behaviour.

## Tracing

Every model call and tool call becomes an ordered span under one trace, so "why
did it say that?" is answerable afterwards from the admin.

```bash
python manage.py traces list
python manage.py traces list --agent support.Support
python manage.py traces show a1b2c3d4 -v 2
```

```python
from mlango.agents import get_trace, recent_traces

trace = get_trace("a1b2c3d4")
[(s.kind, s.name, s.duration_s) for s in trace.spans]
```

Tracing is best effort: a metastore hiccup degrades observability, it never
breaks the agent. Switch it off per agent with `Meta.tracing = False`, or
globally with `TRACING = False`.

## The loop

Per step, until the model stops asking for tools or `max_steps` is reached:

1. Call the provider with the messages, system prompt and tool schemas
2. Record an `llm` span with the token usage
3. On `refusal`, stop and report it
4. On `pause_turn`, re-send unchanged so a server-side tool can resume
5. Execute every requested tool, each in its own `tool` span
6. Append **all** tool results in a single user message
7. Repeat

Returning results in one message matters: splitting them teaches the model to
stop batching its calls.

## Signals

```python
from mlango.core.signals import agent_finished, agent_started, agent_step, tool_called


@receiver(tool_called)
def audit(sender, agent, tool, arguments, **kwargs):
    log.info("%s called %s with %s", sender._meta.label, tool.name, arguments)
```

## Serving

```python
from mlango.serve import path

urlpatterns = [
    path("chat/", Support.as_endpoint()),
]
```

```bash
curl -X POST http://127.0.0.1:8000/api/chat/ \
  -H 'Content-Type: application/json' \
  -d '{"message": "How do I rotate an API key?", "session_id": "user-42"}'
```

## Security

A tool runs with your process's privileges. One that shells out, writes files or
calls an internal API gives the model that reach. Validate inputs, and gate
anything destructive behind an explicit confirmation on your side rather than
trusting the model to be careful.
