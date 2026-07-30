"""The lazy dataset QuerySet.

``Reviews.objects.filter(label="pos").shuffle(seed=0).batch(32)`` builds a
pipeline description; nothing is read until you iterate. That laziness is what
lets the same expression run over ten rows in a test and ten million in
production, and what lets the pipeline itself be recorded alongside a run so
the exact data view is reproducible.

Lookups follow Django's ``field__op`` spelling, which keeps the API predictable
for anyone who has written a Django query.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from mlango.core.exceptions import FieldError, ValidationError
from mlango.core.hashing import fingerprint, full_digest


class Record(dict):
    """A record that answers to both ``record["text"]`` and ``record.text``."""

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"Record has no field {name!r}. Present: {', '.join(self) or '(empty)'}."
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


# --------------------------------------------------------------------------- #
# Lookups
# --------------------------------------------------------------------------- #

LOOKUPS: dict[str, Callable[[Any, Any], bool]] = {
    "exact": lambda value, target: value == target,
    "iexact": lambda value, target: str(value).lower() == str(target).lower(),
    "ne": lambda value, target: value != target,
    "gt": lambda value, target: value is not None and value > target,
    "gte": lambda value, target: value is not None and value >= target,
    "lt": lambda value, target: value is not None and value < target,
    "lte": lambda value, target: value is not None and value <= target,
    "in": lambda value, target: value in target,
    "contains": lambda value, target: target in value if value is not None else False,
    "icontains": lambda value, target: (
        str(target).lower() in str(value).lower() if value is not None else False
    ),
    "startswith": lambda value, target: str(value).startswith(str(target)),
    "endswith": lambda value, target: str(value).endswith(str(target)),
    "isnull": lambda value, target: (value is None) is bool(target),
    "len": lambda value, target: value is not None and len(value) == target,
    "len_gt": lambda value, target: value is not None and len(value) > target,
    "len_lt": lambda value, target: value is not None and len(value) < target,
    "regex": lambda value, target: __import__("re").search(target, str(value)) is not None,
}


def _split_lookup(key: str) -> tuple[str, str]:
    if "__" not in key:
        return key, "exact"
    field, _, suffix = key.rpartition("__")
    if suffix in LOOKUPS:
        return field, suffix
    return key, "exact"


def _matches(record: dict[str, Any], conditions: dict[str, Any]) -> bool:
    for key, target in conditions.items():
        field, lookup = _split_lookup(key)
        try:
            value = record[field]
        except KeyError:
            if lookup == "isnull":
                value = None
            else:
                return False
        if not LOOKUPS[lookup](value, target):
            return False
    return True


# --------------------------------------------------------------------------- #
# QuerySet
# --------------------------------------------------------------------------- #


class DataQuerySet:
    """An immutable, lazily-evaluated view over a dataset."""

    def __init__(
        self, dataset: type, source: Any = None, pipeline: list[dict[str, Any]] | None = None
    ):
        self.dataset = dataset
        self._source = source
        self._pipeline: list[dict[str, Any]] = list(pipeline or [])
        self._cache: list[Record] | None = None

    # -- chaining ------------------------------------------------------------

    def _chain(self, step: dict[str, Any]) -> DataQuerySet:
        return type(self)(self.dataset, self._source, [*self._pipeline, step])

    def filter(self, **conditions: Any) -> DataQuerySet:
        self._check_fields(conditions)
        return self._chain({"op": "filter", "conditions": conditions})

    def exclude(self, **conditions: Any) -> DataQuerySet:
        self._check_fields(conditions)
        return self._chain({"op": "exclude", "conditions": conditions})

    def where(self, predicate: Callable[[Record], bool], *, label: str = "") -> DataQuerySet:
        """Filter with an arbitrary predicate when lookups are not enough."""
        return self._chain({"op": "where", "predicate": predicate, "label": label})

    def map(self, fn: Callable[[Record], dict[str, Any]], *, label: str = "") -> DataQuerySet:
        """Replace each record with ``fn(record)``."""
        return self._chain({"op": "map", "fn": fn, "label": label})

    def annotate(self, **producers: Callable[[Record], Any]) -> DataQuerySet:
        """Add computed fields, leaving existing ones intact."""
        return self._chain({"op": "annotate", "producers": producers})

    def only(self, *names: str) -> DataQuerySet:
        self._check_names(names)
        return self._chain({"op": "only", "names": list(names)})

    def defer(self, *names: str) -> DataQuerySet:
        self._check_names(names)
        return self._chain({"op": "defer", "names": list(names)})

    def rename(self, **mapping: str) -> DataQuerySet:
        """``rename(body="text")`` moves ``body`` onto ``text``."""
        return self._chain({"op": "rename", "mapping": mapping})

    def order_by(self, *names: str) -> DataQuerySet:
        """Sort by fields; prefix with ``-`` to reverse. Materialises the data."""
        return self._chain({"op": "order_by", "names": list(names)})

    def shuffle(self, seed: int | None = None) -> DataQuerySet:
        return self._chain({"op": "shuffle", "seed": seed})

    def skip(self, n: int) -> DataQuerySet:
        return self._chain({"op": "skip", "n": n})

    def take(self, n: int) -> DataQuerySet:
        return self._chain({"op": "take", "n": n})

    def distinct(self, *names: str) -> DataQuerySet:
        return self._chain({"op": "distinct", "names": list(names)})

    def validate(self) -> DataQuerySet:
        """Run every record through the declared fields as it streams past."""
        return self._chain({"op": "validate"})

    def clean(self) -> DataQuerySet:
        """Like :meth:`validate`, but also coerces values to their field types."""
        return self._chain({"op": "clean"})

    def repeat(self, times: int) -> DataQuerySet:
        return self._chain({"op": "repeat", "times": times})

    # -- splitting -----------------------------------------------------------

    def split(self, **ratios: float) -> dict[str, DataQuerySet]:
        """Deterministically partition into named subsets.

        ``Reviews.objects.split(train=0.8, val=0.1, test=0.1)`` assigns each
        record by hashing its key, not by position, so adding rows to the source
        never reshuffles the existing assignment — the property that makes a
        held-out test set trustworthy over time.
        """
        total = sum(ratios.values())
        if not ratios:
            raise ValueError("split() needs at least one named ratio.")
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}.")

        bounds: dict[str, tuple[float, float]] = {}
        cursor = 0.0
        for name in ratios:
            bounds[name] = (cursor, cursor + ratios[name])
            cursor += ratios[name]

        salt = str(getattr(self.dataset._meta, "extras", {}).get("split_salt", ""))
        key_field = getattr(self.dataset._meta, "extras", {}).get("primary_key")
        return {
            name: self._chain(
                {
                    "op": "split",
                    "name": name,
                    "bounds": bound,
                    "key_field": key_field,
                    "salt": salt,
                }
            )
            for name, bound in bounds.items()
        }

    def batch(self, size: int, *, drop_last: bool = False) -> Iterator[list[Record]]:
        """Yield lists of records. Terminal: returns an iterator, not a QuerySet."""
        if size <= 0:
            raise ValueError("Batch size must be positive.")
        chunk: list[Record] = []
        for record in self:
            chunk.append(record)
            if len(chunk) == size:
                yield chunk
                chunk = []
        if chunk and not drop_last:
            yield chunk

    # -- evaluation ----------------------------------------------------------

    def __iter__(self) -> Iterator[Record]:
        if self._cache is not None:
            yield from self._cache
            return
        yield from self._execute()

    def _execute(self) -> Iterator[Record]:
        stream: Iterable[Any] = iter(self._resolved_source())
        stream = (Record(record) for record in stream)
        for step in self._pipeline:
            stream = self._apply(step, stream)
        yield from stream

    def _apply(self, step: dict[str, Any], stream: Iterable[Record]) -> Iterator[Record]:
        op = step["op"]

        if op == "filter":
            conditions = step["conditions"]
            return (r for r in stream if _matches(r, conditions))
        if op == "exclude":
            conditions = step["conditions"]
            return (r for r in stream if not _matches(r, conditions))
        if op == "where":
            predicate = step["predicate"]
            return (r for r in stream if predicate(r))
        if op == "map":
            fn = step["fn"]
            return (Record(fn(r)) for r in stream)
        if op == "annotate":
            producers = step["producers"]
            return (Record({**r, **{k: fn(r) for k, fn in producers.items()}}) for r in stream)
        if op == "only":
            names = step["names"]
            return (Record({k: v for k, v in r.items() if k in names}) for r in stream)
        if op == "defer":
            names = set(step["names"])
            return (Record({k: v for k, v in r.items() if k not in names}) for r in stream)
        if op == "rename":
            mapping = step["mapping"]
            return (_rename(r, mapping) for r in stream)
        if op == "skip":
            return _skip(stream, step["n"])
        if op == "take":
            return _take(stream, step["n"])
        if op == "repeat":
            return _repeat(stream, step["times"])
        if op == "distinct":
            return _distinct(stream, step["names"])
        if op == "validate":
            return self._validated(stream, coerce=False)
        if op == "clean":
            return self._validated(stream, coerce=True)
        if op == "split":
            return _split_stream(stream, step)
        if op == "shuffle":
            return _shuffle(stream, step["seed"])
        if op == "order_by":
            return _order_by(stream, step["names"])

        raise FieldError(f"Unknown pipeline operation {op!r}.")

    def _validated(self, stream: Iterable[Record], *, coerce: bool) -> Iterator[Record]:
        fields = self.dataset._meta.fields
        for index, record in enumerate(stream):
            cleaned: dict[str, Any] = dict(record)
            errors: dict[str, list[str]] = {}
            for field in fields:
                name = field.name or ""
                raw = record.get(name, field.get_default())
                try:
                    value = field.clean(raw)
                except ValidationError as exc:
                    for key, messages in exc.errors.items():
                        errors.setdefault(key if key != "__all__" else name, []).extend(messages)
                    continue
                if coerce:
                    cleaned[name] = value
            if errors:
                raise ValidationError(
                    {f"row {index}": [f"{k}: {'; '.join(v)}" for k, v in errors.items()]}
                )
            yield Record(cleaned)

    # -- terminal operations -------------------------------------------------

    def all(self) -> list[Record]:
        return list(self)

    def cache(self) -> DataQuerySet:
        """Evaluate once and hold the result in memory for repeated passes."""
        clone = type(self)(self.dataset, self._source, self._pipeline)
        clone._cache = list(self._execute())
        return clone

    def count(self) -> int:
        if self._cache is not None:
            return len(self._cache)
        return sum(1 for _ in self)

    def first(self) -> Record | None:
        for record in self:
            return record
        return None

    def get(self, **conditions: Any) -> Record:
        from mlango.core.exceptions import DoesNotExist, MultipleObjectsReturned

        matches = list(self.filter(**conditions).take(2))
        if not matches:
            raise DoesNotExist(f"{self.dataset._meta.label} has no record matching {conditions!r}.")
        if len(matches) > 1:
            raise MultipleObjectsReturned(
                f"{self.dataset._meta.label} has several records matching {conditions!r}."
            )
        return matches[0]

    def exists(self) -> bool:
        return self.first() is not None

    def values(self, *names: str) -> list[dict[str, Any]]:
        if not names:
            return [dict(r) for r in self]
        return [{n: r.get(n) for n in names} for r in self]

    def values_list(self, *names: str, flat: bool = False) -> list[Any]:
        if flat:
            if len(names) != 1:
                raise ValueError("flat=True requires exactly one field name.")
            name = names[0]
            return [r.get(name) for r in self]
        return [tuple(r.get(n) for n in names) for r in self]

    def columns(self, *names: str) -> dict[str, list[Any]]:
        """Column-oriented view — what most trainers actually want."""
        names = names or tuple(self.dataset._meta.field_names)
        out: dict[str, list[Any]] = {name: [] for name in names}
        for record in self:
            for name in names:
                out[name].append(record.get(name))
        return out

    def xy(
        self, *, target: str | None = None, features: list[str] | None = None
    ) -> tuple[list[Any], list[Any]]:
        """``(inputs, targets)`` for a target field and a set of feature fields.

        With a single feature the inputs are that field's raw values, so a text
        pipeline receives a list of strings; with several they become dicts.
        """
        opts = self.dataset._meta
        if target is None:
            targets = opts.target_fields
            if len(targets) != 1:
                raise FieldError(
                    f"{opts.label} declares {len(targets)} target fields; pass target= to pick one."
                )
            target = targets[0].name or ""
        input_names = list(features) if features else [f.name for f in opts.input_fields]
        xs: list[Any] = []
        ys: list[Any] = []
        for record in self:
            if len(input_names) == 1:
                xs.append(record.get(input_names[0]))
            else:
                xs.append({n: record.get(n) for n in input_names})
            ys.append(record.get(target))
        return xs, ys

    def to_pandas(self):  # pragma: no cover - optional dependency
        import pandas as pd

        return pd.DataFrame(list(self))

    def describe_pipeline(self) -> list[dict[str, Any]]:
        """A JSON-safe description of the pipeline, stored alongside runs."""
        out = []
        for step in self._pipeline:
            entry: dict[str, Any] = {"op": step["op"]}
            for key, value in step.items():
                if key == "op":
                    continue
                if callable(value):
                    entry[key] = (
                        f"{getattr(value, '__module__', '?')}.{getattr(value, '__qualname__', 'fn')}"
                    )
                elif isinstance(value, dict) and any(callable(v) for v in value.values()):
                    entry[key] = sorted(value)
                else:
                    entry[key] = value
            out.append(entry)
        return out

    def fingerprint(self) -> str:
        """Identity of this *view*: source plus pipeline, not the rows."""
        return fingerprint(
            {
                "dataset": self.dataset._meta.label,
                "schema": self.dataset._meta.fingerprint(),
                "source": self._describe_source(),
                "pipeline": self.describe_pipeline(),
            }
        )

    def content_hash(self) -> str:
        """Identity of the *rows*, computed by streaming them once."""
        import hashlib

        from mlango.core.hashing import canonical_json

        digest = hashlib.sha256()
        for record in self:
            digest.update(canonical_json(dict(record)).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    # -- helpers -------------------------------------------------------------

    def _resolved_source(self) -> Any:
        source = self._source if self._source is not None else self.dataset.get_source()
        if source is None:
            raise FieldError(
                f"{self.dataset._meta.label} has no source. Set `source` in its Meta, or "
                f"override the classmethod `records()`."
            )
        return source

    def _describe_source(self) -> dict[str, Any]:
        source = self._resolved_source()
        describe = getattr(source, "describe", None)
        return describe() if callable(describe) else {"type": type(source).__name__}

    def _check_fields(self, conditions: dict[str, Any]) -> None:
        self._check_names([_split_lookup(key)[0] for key in conditions])

    def _check_names(self, names: Iterable[str]) -> None:
        opts = self.dataset._meta
        # Annotations and maps legitimately introduce fields that are not
        # declared, so only reject names that no step could have produced.
        produced = set(opts.field_names)
        for step in self._pipeline:
            if step["op"] == "annotate":
                produced |= set(step["producers"])
            elif step["op"] == "rename":
                produced |= set(step["mapping"])
            elif step["op"] in {"map", "where"}:
                return  # Arbitrary code ran; we can no longer verify names.
        unknown = [n for n in names if n not in produced]
        if unknown:
            raise FieldError(
                f"{opts.label} has no field(s) {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(produced)) or '(none)'}."
            )

    def __repr__(self) -> str:
        steps = " -> ".join(step["op"] for step in self._pipeline) or "all"
        return f"<DataQuerySet {self.dataset._meta.label}: {steps}>"

    def __getitem__(self, item: int | slice) -> Any:
        if isinstance(item, int):
            if item < 0:
                return self.all()[item]
            for index, record in enumerate(self):
                if index == item:
                    return record
            raise IndexError(f"Index {item} is out of range.")
        start = item.start or 0
        if item.step not in (None, 1):
            raise ValueError("Step slicing is not supported; use shuffle() or map().")
        query = self.skip(start) if start else self
        if item.stop is not None:
            query = query.take(item.stop - start)
        return query


# --------------------------------------------------------------------------- #
# Stream helpers
# --------------------------------------------------------------------------- #


def _rename(record: Record, mapping: dict[str, str]) -> Record:
    out = Record(record)
    for old, new in mapping.items():
        if old in out:
            out[new] = out.pop(old)
    return out


def _skip(stream: Iterable[Record], n: int) -> Iterator[Record]:
    for index, record in enumerate(stream):
        if index >= n:
            yield record


def _take(stream: Iterable[Record], n: int) -> Iterator[Record]:
    if n <= 0:
        return
    for index, record in enumerate(stream):
        yield record
        if index + 1 >= n:
            return


def _repeat(stream: Iterable[Record], times: int) -> Iterator[Record]:
    buffered = list(stream)
    for _ in range(times):
        yield from (Record(r) for r in buffered)


def _distinct(stream: Iterable[Record], names: list[str]) -> Iterator[Record]:
    seen: set[str] = set()
    for record in stream:
        key = full_digest({n: record.get(n) for n in names} if names else dict(record))
        if key in seen:
            continue
        seen.add(key)
        yield record


def _shuffle(stream: Iterable[Record], seed: int | None) -> Iterator[Record]:
    # A full shuffle has to see every record, so this is a materialising step.
    buffered = list(stream)
    random.Random(seed).shuffle(buffered)
    yield from buffered


def _order_by(stream: Iterable[Record], names: list[str]) -> Iterator[Record]:
    buffered = list(stream)
    for name in reversed(names):
        reverse = name.startswith("-")
        key = name[1:] if reverse else name
        buffered.sort(key=lambda r, k=key: (r.get(k) is None, r.get(k)), reverse=reverse)
    yield from buffered


def _split_stream(stream: Iterable[Record], step: dict[str, Any]) -> Iterator[Record]:
    low, high = step["bounds"]
    key_field = step["key_field"]
    salt = step["salt"]
    for record in stream:
        key = record.get(key_field) if key_field else None
        # Hash the content, never the position: inserting rows must not move
        # existing ones between splits, or a held-out set stops being held out.
        material = f"{salt}|{key}" if key is not None else f"{salt}|{full_digest(dict(record))}"
        bucket = int(full_digest(material)[:8], 16) / 0xFFFFFFFF
        if low <= bucket < high or (high >= 1.0 - 1e-9 and bucket >= low):
            yield record


__all__ = ["DataQuerySet", "Record", "LOOKUPS"]
