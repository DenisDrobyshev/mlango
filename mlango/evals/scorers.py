"""Reusable scorers.

A scorer takes ``(output, expected)`` and returns a float in ``[0, 1]`` or a
bool. They compose: an eval usually declares several and the runner records
each one separately so a regression can be traced to the criterion it broke.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

Scorer = Callable[[Any, Any], float | bool]


def exact_match(output: Any, expected: Any) -> bool:
    return str(output).strip() == str(expected).strip()


def iexact_match(output: Any, expected: Any) -> bool:
    return str(output).strip().casefold() == str(expected).strip().casefold()


def contains(output: Any, expected: Any) -> bool:
    return str(expected).casefold() in str(output).casefold()


def contains_all(*needles: str) -> Scorer:
    def score(output: Any, expected: Any = None) -> float:
        text = str(output).casefold()
        hits = sum(1 for n in needles if n.casefold() in text)
        return hits / len(needles) if needles else 0.0

    return score


def not_contains(*needles: str) -> Scorer:
    def score(output: Any, expected: Any = None) -> bool:
        text = str(output).casefold()
        return not any(n.casefold() in text for n in needles)

    return score


def regex_match(pattern: str, *, flags: int = 0) -> Scorer:
    compiled = re.compile(pattern, flags)

    def score(output: Any, expected: Any = None) -> bool:
        return compiled.search(str(output)) is not None

    return score


def json_equals(output: Any, expected: Any) -> bool:
    """Compare two JSON payloads structurally, ignoring key order."""
    return _as_json(output) == _as_json(expected)


def json_subset(output: Any, expected: Any) -> float:
    """Fraction of the expected keys present with matching values."""
    got, want = _as_json(output), _as_json(expected)
    if not isinstance(want, dict):
        return float(got == want)
    if not isinstance(got, dict) or not want:
        return 0.0
    hits = sum(1 for key, value in want.items() if got.get(key) == value)
    return hits / len(want)


def numeric_close(tolerance: float = 1e-6) -> Scorer:
    def score(output: Any, expected: Any) -> bool:
        try:
            return abs(float(output) - float(expected)) <= tolerance
        except (TypeError, ValueError):
            return False

    return score


def length_between(minimum: int = 0, maximum: int | None = None) -> Scorer:
    def score(output: Any, expected: Any = None) -> bool:
        size = len(str(output))
        return size >= minimum and (maximum is None or size <= maximum)

    return score


def token_f1(output: Any, expected: Any) -> float:
    """Word-overlap F1 — a cheap stand-in for semantic similarity."""
    got = _tokens(output)
    want = _tokens(expected)
    if not got or not want:
        return float(got == want)
    overlap = len(got & want)
    if not overlap:
        return 0.0
    precision = overlap / len(got)
    recall = overlap / len(want)
    return 2 * precision * recall / (precision + recall)


def used_tool(name: str) -> Callable[[Any], bool]:
    """Check that an agent reached for a particular tool.

    Takes the whole :class:`~mlango.agents.agent.AgentRun` rather than its text,
    so an eval can assert on behaviour and not only on the final answer.
    """

    def score(run: Any, expected: Any = None) -> bool:
        return name in getattr(run, "tools_used", [])

    # Tells the eval runner to hand over the AgentRun, not its text.
    score.wants_run = True  # type: ignore[attr-defined]
    return score


def llm_judge(
    agent: Any,
    *,
    rubric: str,
    passing: float = 0.7,
) -> Scorer:
    """Score with another agent.

    The judge is asked for a bare number so parsing stays trivial; anything
    unparseable scores 0 rather than raising, because one confused judgement
    should not abort an eval sweep.
    """

    def score(output: Any, expected: Any = None) -> float:
        prompt = (
            f"{rubric}\n\n"
            f"--- Candidate answer ---\n{output}\n"
            + (f"\n--- Reference answer ---\n{expected}\n" if expected is not None else "")
            + "\nReply with only a number between 0 and 1."
        )
        result = agent.run(prompt) if hasattr(agent, "run") else agent(prompt)
        text = getattr(result, "output", result)
        match = re.search(r"\d*\.?\d+", str(text))
        if not match:
            return 0.0
        try:
            return max(0.0, min(1.0, float(match.group())))
        except ValueError:
            return 0.0

    score.passing = passing  # type: ignore[attr-defined]
    return score


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _as_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return value


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"\w+", str(value).casefold()))


REGISTRY: dict[str, Scorer] = {
    "exact_match": exact_match,
    "iexact_match": iexact_match,
    "contains": contains,
    "json_equals": json_equals,
    "json_subset": json_subset,
    "token_f1": token_f1,
}

__all__ = [
    "Scorer",
    "exact_match",
    "iexact_match",
    "contains",
    "contains_all",
    "not_contains",
    "regex_match",
    "json_equals",
    "json_subset",
    "numeric_close",
    "length_between",
    "token_f1",
    "used_tool",
    "llm_judge",
    "REGISTRY",
]
