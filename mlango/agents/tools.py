"""Tools.

A tool is an ordinary Python function. The ``@tool`` decorator reads its type
hints and docstring and produces the JSON Schema the model needs, so the
description of a tool lives in exactly one place — the function itself. This is
the same instinct behind Django's ``ModelForm``: derive the boring part from
the declaration you already wrote.

    @tool
    def search_docs(query: str, limit: int = 5) -> list[str]:
        \"\"\"Search the product documentation.

        Args:
            query: What to search for.
            limit: Maximum number of results.
        \"\"\"
"""

from __future__ import annotations

import enum
import inspect
import re
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin

from mlango.core.exceptions import ValidationError

_PRIMITIVES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    type(None): "null",
}

ARGS_HEADER = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:\s*$", re.IGNORECASE)
SECTION_HEADER = re.compile(r"^\s*(Returns|Raises|Yields|Examples?|Notes?)\s*:\s*$", re.IGNORECASE)
ARG_LINE = re.compile(r"^\s{0,8}(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


class ToolError(Exception):
    """Raised inside a tool to return an error to the model instead of crashing."""


@dataclass
class ToolResult:
    """What a tool produced, ready to be handed back to the model."""

    tool_call_id: str
    name: str
    content: Any
    is_error: bool = False
    duration_s: float = 0.0

    def as_text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        import json

        try:
            return json.dumps(self.content, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(self.content)


@dataclass
class Tool:
    """A callable exposed to the model."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    #: Guarantees the model's arguments validate against the schema exactly.
    strict: bool = False
    #: Human-facing grouping used by the admin.
    tags: list[str] = field(default_factory=list)

    # -- schema --------------------------------------------------------------

    def to_schema(self) -> dict[str, Any]:
        """The tool definition as the Messages API expects it."""
        schema: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }
        if self.strict:
            schema["strict"] = True
        return schema

    # -- execution -----------------------------------------------------------

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        required = set(self.parameters.get("required", []))
        properties = self.parameters.get("properties", {})
        missing = sorted(required - set(arguments))
        if missing:
            raise ValidationError(f"{self.name} is missing argument(s): {', '.join(missing)}")
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise ValidationError(f"{self.name} got unknown argument(s): {', '.join(unknown)}")
        return arguments

    def run(self, call_id: str = "", **arguments: Any) -> ToolResult:
        """Execute the tool, turning any failure into an error result.

        A tool that raises must not take the agent down — the model is usually
        able to recover if it is told what went wrong, and an unhandled
        exception mid-loop loses the whole trace.
        """
        import time

        started = time.perf_counter()
        try:
            self.validate(arguments)
            value = self.fn(**arguments)
            return ToolResult(
                tool_call_id=call_id,
                name=self.name,
                content=value,
                duration_s=time.perf_counter() - started,
            )
        except (ToolError, ValidationError) as exc:
            return ToolResult(
                tool_call_id=call_id,
                name=self.name,
                content=str(exc),
                is_error=True,
                duration_s=time.perf_counter() - started,
            )
        except Exception as exc:  # noqa: BLE001 - reported to the model verbatim
            return ToolResult(
                tool_call_id=call_id,
                name=self.name,
                content=f"{type(exc).__name__}: {exc}",
                is_error=True,
                duration_s=time.perf_counter() - started,
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Calling the decorated object still calls the original function."""
        return self.fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<Tool {self.name}({', '.join(self.parameters.get('properties', {}))})>"


# --------------------------------------------------------------------------- #
# Decorator
# --------------------------------------------------------------------------- #


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool = False,
    tags: list[str] | None = None,
) -> Any:
    """Turn a function into a :class:`Tool`. Usable bare or with arguments."""

    def wrap(func: Callable[..., Any]) -> Tool:
        summary, arg_docs = _parse_docstring(func.__doc__ or "")
        built = Tool(
            name=name or func.__name__,
            description=description or summary or f"Call {func.__name__}.",
            parameters=build_schema(func, arg_docs, strict=strict),
            fn=func,
            strict=strict,
            tags=list(tags or []),
        )
        # Keep introspection working: help(), functools.wraps semantics, tests.
        built.__doc__ = func.__doc__
        return built

    return wrap(fn) if fn is not None else wrap


def build_schema(
    func: Callable[..., Any], arg_docs: dict[str, str] | None = None, *, strict: bool = False
) -> dict[str, Any]:
    """JSON Schema for a function's parameters, derived from its signature."""
    arg_docs = arg_docs or {}
    signature = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except Exception:  # pragma: no cover - exotic annotations
        hints = getattr(func, "__annotations__", {})

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, parameter in signature.parameters.items():
        if param_name in {"self", "cls"}:
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue

        annotation = hints.get(param_name, parameter.annotation)
        entry = _schema_for(annotation)
        if param_name in arg_docs:
            entry["description"] = arg_docs[param_name]
        if parameter.default is not inspect.Parameter.empty:
            if _json_safe(parameter.default):
                entry["default"] = parameter.default
        else:
            required.append(param_name)
        properties[param_name] = entry

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    # Strict mode needs every property listed as required and no extras, so
    # optional parameters become nullable instead of absent.
    if strict:
        schema["required"] = list(properties)
        schema["additionalProperties"] = False
    elif required:
        schema["required"] = required
    return schema


def _schema_for(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is typing.Literal:
        values = list(args)
        kinds = {_PRIMITIVES.get(type(v), "string") for v in values}
        return {"type": kinds.pop() if len(kinds) == 1 else "string", "enum": values}

    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        non_null = [a for a in args if a is not type(None)]
        if len(non_null) == 1:
            return _schema_for(non_null[0])
        return {"anyOf": [_schema_for(a) for a in non_null]}

    if origin in (list, set, frozenset, tuple):
        item = args[0] if args and args[0] is not Ellipsis else str
        return {"type": "array", "items": _schema_for(item)}

    if origin is dict:
        value_type = args[1] if len(args) == 2 else Any
        return {"type": "object", "additionalProperties": _schema_for(value_type)}

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return {"type": "string", "enum": [member.value for member in annotation]}

    if isinstance(annotation, type) and issubclass(annotation, bool):
        return {"type": "boolean"}

    # Pydantic models and dataclasses describe themselves well enough.
    model_schema = getattr(annotation, "model_json_schema", None)
    if callable(model_schema):
        try:
            return model_schema()
        except Exception:  # pragma: no cover - defensive
            pass

    return {"type": "string"}


def _json_safe(value: Any) -> bool:
    import json

    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into a summary and per-argument text."""
    if not doc:
        return "", {}

    lines = inspect.cleandoc(doc).splitlines()
    summary_lines: list[str] = []
    arg_docs: dict[str, str] = {}
    in_args = False
    summary_done = False
    current: str | None = None

    for line in lines:
        if ARGS_HEADER.match(line):
            in_args, summary_done, current = True, True, None
            continue
        if SECTION_HEADER.match(line):
            in_args, summary_done, current = False, True, None
            continue

        if in_args:
            match = ARG_LINE.match(line)
            if match:
                current = match.group(1).lstrip("*")
                arg_docs[current] = match.group(2).strip()
            elif current and line.strip():
                arg_docs[current] = f"{arg_docs[current]} {line.strip()}".strip()
            continue

        # The summary is the first paragraph; a blank line ends it, but parsing
        # continues so that a later Args: section is still picked up.
        if summary_done:
            continue
        if line.strip():
            summary_lines.append(line)
        elif summary_lines:
            summary_done = True

    return " ".join(part.strip() for part in summary_lines if part.strip()), arg_docs


# --------------------------------------------------------------------------- #
# Collections
# --------------------------------------------------------------------------- #


class Toolbox:
    """A named set of tools, resolvable by name during the agent loop."""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for item in tools or []:
            self.add(item)

    def add(self, item: Tool | Callable[..., Any]) -> Tool:
        built = item if isinstance(item, Tool) else tool(item)
        if built.name in self._tools and self._tools[built.name] is not built:
            raise ValidationError(f"Two tools are both named {built.name!r}.")
        self._tools[built.name] = built
        return built

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"<Toolbox {', '.join(self._tools) or '(empty)'}>"


__all__ = ["tool", "Tool", "ToolResult", "ToolError", "Toolbox", "build_schema"]
