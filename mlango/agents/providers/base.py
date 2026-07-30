"""LLM provider contract.

A provider does exactly one thing: turn a request into a single model
response. The agent loop, tool dispatch, memory and tracing all live in the
framework, so swapping providers never changes agent behaviour — the same
separation that lets a Django app move from SQLite to Postgres untouched.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from mlango.core.exceptions import ImproperlyConfigured, ProviderError
from mlango.core.module_loading import import_string

_cache: dict[str, Provider] = {}


@dataclass
class ToolCall:
    """A tool the model wants executed."""

    id: str
    name: str
    arguments: dict[str, Any]

    def describe(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )

    def describe(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class Completion:
    """One normalised model response."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: Provider-native reason, normalised to one of the constants below.
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    #: The assistant turn exactly as the provider expects it echoed back.
    raw_content: Any = None
    model: str = ""
    #: Populated when ``stop_reason`` is ``refusal``.
    refusal: dict[str, Any] | None = None

    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    PAUSE_TURN = "pause_turn"
    REFUSAL = "refusal"

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class Provider(abc.ABC):
    """Base class for every LLM backend."""

    name: str = ""
    requires: tuple[str, ...] = ()
    #: Providers that never reach the network can run in tests and CI.
    offline: bool = False

    def __init__(self, **options: Any):
        self.options = options

    @classmethod
    def check_available(cls) -> None:
        import importlib.util

        missing = [m for m in cls.requires if importlib.util.find_spec(m) is None]
        if missing:
            raise ProviderError(
                f"The {cls.name!r} provider needs {', '.join(missing)}. "
                f"Install it with: pip install mlango[{cls.name}]"
            )

    @abc.abstractmethod
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
        """Produce a single completion."""

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "offline": self.offline}

    def __repr__(self) -> str:
        return f"<Provider {self.name!r}>"


def get_provider(name: str | None = None) -> Provider:
    """Build (and cache) the provider registered under ``name``."""
    from mlango.conf import settings

    name = name or settings.DEFAULT_PROVIDER
    if name in _cache:
        return _cache[name]

    try:
        path = settings.PROVIDERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(settings.PROVIDERS)) or "(none)"
        raise ImproperlyConfigured(
            f"No provider registered as {name!r}. Available: {known}. "
            f"Add one to the PROVIDERS setting."
        ) from exc

    provider_class = import_string(str(path))
    if not (isinstance(provider_class, type) and issubclass(provider_class, Provider)):
        raise ImproperlyConfigured(f"{path} is not a Provider subclass.")
    provider_class.check_available()
    instance = provider_class()
    instance.name = instance.name or name
    _cache[name] = instance
    return instance


def available_providers() -> dict[str, bool]:
    from mlango.conf import settings

    out: dict[str, bool] = {}
    for name, path in settings.PROVIDERS.items():
        try:
            provider_class = import_string(str(path))
            provider_class.check_available()
            out[name] = True
        except Exception:
            out[name] = False
    return out


def clear_provider_cache() -> None:
    _cache.clear()


__all__ = [
    "Provider",
    "Completion",
    "ToolCall",
    "Usage",
    "get_provider",
    "available_providers",
    "clear_provider_cache",
]
