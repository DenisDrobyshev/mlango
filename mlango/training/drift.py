"""Comparing what a model was trained on against what it is being asked.

A model does not break loudly when the world moves. It keeps returning
confident answers about data it has never seen, and the metrics that would show
it — accuracy against labels — do not exist in production, because the labels
are the thing you are waiting for. What *is* available on day one is the input,
and the input distribution moving is the earliest honest signal there is.

So training records a summary of the data it fitted on, serving records what it
was asked, and this module compares the two. The measure is the population
stability index, which is standard in credit risk for exactly this job and has
the property that matters here: one number per column, comparable across
columns of different types, with published thresholds nobody has to invent.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

#: Deciles. Enough resolution to see a shifted mode, few enough that a few
#: hundred production rows still put something in most buckets.
BINS = 10

#: Beyond this a "category" is really an identifier, and counting its values
#: measures nothing. Text columns are profiled by length instead.
MAX_CATEGORIES = 50

#: PSI is a sum of ratios, so an empty bucket sends it to infinity. Standard
#: practice is to floor every proportion; the value only matters in that it is
#: small relative to a real proportion.
EPSILON = 1e-4

#: The conventional reading of a PSI, from the credit-scoring literature.
STABLE = 0.1
SIGNIFICANT = 0.25


def verdict(score: float) -> str:
    """``stable``, ``moderate`` or ``significant`` for a PSI."""
    if score < STABLE:
        return "stable"
    if score < SIGNIFICANT:
        return "moderate"
    return "significant"


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #


def profile(records: Iterable[Any], columns: list[str]) -> dict[str, Any]:
    """Summarise ``columns`` across ``records``, small enough to store on a row.

    Records are the same shape the trainer sees, so a single-feature model can
    hand over bare values and they are filed under that feature's name — the
    alternative is a baseline whose keys do not match what serving logs.
    """
    gathered = _gather(records, columns)
    return {name: _profile_column(values) for name, values in gathered.items() if values}


def _gather(records: Iterable[Any], columns: list[str]) -> dict[str, list[Any]]:
    """Column-wise values, from dicts or from the bare values of a lone column."""
    gathered: dict[str, list[Any]] = {name: [] for name in columns}
    for record in records:
        if isinstance(record, dict):
            for name in columns:
                gathered[name].append(record.get(name))
        elif len(columns) == 1:
            gathered[columns[0]].append(record)
    return gathered


def _profile_column(values: list[Any], kind: str | None = None) -> dict[str, Any]:
    """Summarise one column, optionally forced into a kind.

    Observed data is always profiled as whatever the baseline decided it was.
    Inferring twice looks harmless and is not: a text column whose production
    traffic happens to repeat itself would be read as categorical, and the
    comparison would silently return nothing at exactly the moment the input
    collapsed to a handful of values — the case worth catching.
    """
    present = [value for value in values if value is not None]
    missing = len(values) - len(present)
    common = {"count": len(values), "missing": missing}

    if not present:
        return {"kind": "empty", **common}

    if kind is None or kind == "numeric":
        numbers = _as_numbers(present)
        if numbers is not None:
            return {**_numeric_profile(numbers), **common}
        if kind == "numeric":
            return {"kind": "incomparable", **common}

    texts = [str(value) for value in present]
    distinct = set(texts)
    if kind == "text" or (kind is None and len(distinct) > MAX_CATEGORIES):
        # An identifier or free text. Its values will never repeat, so their
        # frequencies say nothing; length is a crude proxy, but it does move
        # when the input changes shape, and it is honest about being crude.
        return {
            **_numeric_profile([float(len(text)) for text in texts]),
            "kind": "text",
            **common,
            "distinct": len(distinct),
        }

    counts: dict[str, int] = {}
    for text in texts:
        counts[text] = counts.get(text, 0) + 1
    return {
        "kind": "categorical",
        **common,
        "values": dict(sorted(counts.items(), key=lambda pair: -pair[1])),
    }


def _numeric_profile(numbers: list[float]) -> dict[str, Any]:
    ordered = sorted(numbers)
    edges = _quantile_edges(ordered, BINS)
    return {
        "kind": "numeric",
        "mean": _mean(ordered),
        "std": _std(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "edges": edges,
        "counts": _bucket(ordered, edges),
    }


def _quantile_edges(ordered: list[float], bins: int) -> list[float]:
    """Interior bucket edges at equal *counts*, not equal width.

    Equal-width buckets on a skewed column put every row in one of them, and a
    PSI computed from that cannot detect anything. Quantiles give each baseline
    bucket the same share, which is the shape PSI assumes.
    """
    if len(ordered) < 2 or ordered[0] == ordered[-1]:
        return []
    edges: list[float] = []
    for index in range(1, bins):
        position = index / bins * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        value = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
        # Ties collapse a bucket to zero width; keeping only strict increases
        # means a column with a dominant value gets fewer buckets, not broken ones.
        if not edges or value > edges[-1]:
            edges.append(value)
    return edges


def _bucket(numbers: Iterable[float], edges: list[float]) -> list[int]:
    import bisect

    counts = [0] * (len(edges) + 1)
    for number in numbers:
        counts[bisect.bisect_right(edges, number)] += 1
    return counts


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def compare(baseline: dict[str, Any], records: Iterable[Any]) -> dict[str, Any]:
    """PSI per column, for the columns the baseline knows about.

    Columns the observed data does not carry are skipped rather than reported
    as maximally drifted: a payload that omits a feature is a bug in the
    caller, and calling it drift would bury the real signal.
    """
    gathered = _gather(records, list(baseline))
    out: dict[str, Any] = {}
    for name, reference in baseline.items():
        values = gathered.get(name)
        if not values:
            continue
        current = _profile_column(values, kind=reference.get("kind"))
        score = _psi(reference, current)
        if score is None:
            continue
        out[name] = {
            "psi": score,
            "verdict": verdict(score),
            "kind": reference.get("kind", "unknown"),
            "count": current.get("count", 0),
        }
    return out


def _psi(reference: dict[str, Any], current: dict[str, Any]) -> float | None:
    kind = reference.get("kind")
    if kind in {"numeric", "text"}:
        edges = reference.get("edges") or []
        expected = reference.get("counts") or []
        if not expected:
            return None
        if current.get("kind") not in {"numeric", "text"}:
            return None
        actual = _rebucket(current, edges)
        if actual is None:
            return None
        return _divergence(expected, actual)

    if kind == "categorical":
        expected_counts = reference.get("values") or {}
        actual_counts = dict(current.get("values") or {})
        if not expected_counts:
            return None
        if current.get("kind") != "categorical":
            # A column that was categorical at training and is now free text
            # has not drifted, it has changed meaning. Say nothing rather than
            # produce a number that implies the comparison was valid.
            return None
        keys = list(expected_counts) + [k for k in actual_counts if k not in expected_counts]
        return _divergence(
            [expected_counts.get(key, 0) for key in keys],
            [actual_counts.get(key, 0) for key in keys],
        )

    return None


def _rebucket(current: dict[str, Any], edges: list[float]) -> list[int] | None:
    """Recount the observed column into the baseline's own buckets.

    The observed profile has quantile edges of its own, and comparing two
    different binnings measures the binning rather than the data.
    """
    observed_edges = current.get("edges") or []
    observed_counts = current.get("counts") or []
    if not observed_counts:
        return None
    if observed_edges == edges:
        return list(observed_counts)

    # Reconstruct representative values: a bucket's rows are somewhere inside
    # it, and its midpoint is the only defensible guess without keeping every
    # value. Sufficient for a stability index, which reads shape, not moments.
    values: list[float] = []
    low = current.get("min", 0.0)
    high = current.get("max", low)
    bounds = [low, *observed_edges, high]
    for index, count in enumerate(observed_counts):
        if not count:
            continue
        left = bounds[index] if index < len(bounds) else high
        right = bounds[index + 1] if index + 1 < len(bounds) else high
        values.extend([(left + right) / 2.0] * count)
    return _bucket(values, edges)


def _divergence(expected: list[int], actual: list[int]) -> float:
    expected_total = sum(expected) or 1
    actual_total = sum(actual) or 1
    score = 0.0
    for e, a in zip(expected, actual, strict=True):
        p = max(e / expected_total, EPSILON)
        q = max(a / actual_total, EPSILON)
        score += (q - p) * math.log(q / p)
    return round(score, 6)


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #


def _as_numbers(values: list[Any]) -> list[float] | None:
    """Every value as a float, or None if any of them is not a number.

    Booleans are excluded deliberately: ``True`` is a category with two values,
    and treating it as 1.0 would file it under quantile buckets that cannot
    exist.
    """
    numbers: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        numbers.append(float(value))
    return numbers


def _mean(numbers: list[float]) -> float:
    return round(sum(numbers) / len(numbers), 6)


def _std(numbers: list[float]) -> float:
    if len(numbers) < 2:
        return 0.0
    average = sum(numbers) / len(numbers)
    variance = sum((number - average) ** 2 for number in numbers) / (len(numbers) - 1)
    return round(math.sqrt(variance), 6)


__all__ = ["profile", "compare", "verdict", "BINS", "STABLE", "SIGNIFICANT"]
