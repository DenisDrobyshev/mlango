"""Where a dataset's raw records come from.

A source only has to yield mappings; validation, filtering and splitting are the
QuerySet's job. Keeping that boundary sharp is what lets the same pipeline run
over a JSONL file today and a warehouse table tomorrow.
"""

from __future__ import annotations

import abc
import csv
import json
import os
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from mlango.core.exceptions import ImproperlyConfigured


class Source(abc.ABC):
    """Base class for every data source."""

    @abc.abstractmethod
    def __iter__(self) -> Iterator[dict[str, Any]]:
        ...

    def count(self) -> int | None:
        """Row count when it is cheap to know, else ``None``."""
        return None

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__}

    def __repr__(self) -> str:
        detail = ", ".join(f"{k}={v!r}" for k, v in self.describe().items() if k != "type")
        return f"<{type(self).__name__} {detail}>"


class InMemorySource(Source):
    """A list of dicts — fixtures, tests, and small hand-curated sets."""

    def __init__(self, records: Iterable[dict[str, Any]]):
        self.records = list(records)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from (dict(record) for record in self.records)

    def count(self) -> int:
        return len(self.records)

    def describe(self) -> dict[str, Any]:
        return {"type": "InMemorySource", "rows": len(self.records)}


class PythonSource(Source):
    """A callable returning an iterable of records — generated or scraped data."""

    def __init__(self, factory: Callable[[], Iterable[dict[str, Any]]], *, count: int | None = None):
        if not callable(factory):
            raise ImproperlyConfigured("PythonSource expects a callable returning records.")
        self.factory = factory
        self._count = count

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self.factory()

    def count(self) -> int | None:
        return self._count

    def describe(self) -> dict[str, Any]:
        return {
            "type": "PythonSource",
            "factory": f"{getattr(self.factory, '__module__', '?')}."
            f"{getattr(self.factory, '__qualname__', repr(self.factory))}",
        }


class _FileSource(Source):
    def __init__(self, path: str, *, encoding: str = "utf-8"):
        self.path = path
        self.encoding = encoding

    def resolve(self) -> str:
        if os.path.isabs(self.path):
            return self.path
        from mlango.conf import settings

        return os.path.join(str(settings.BASE_DIR), self.path)

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "path": self.path}


class JSONLSource(_FileSource):
    """One JSON object per line — the default interchange format."""

    def __iter__(self) -> Iterator[dict[str, Any]]:
        resolved = self.resolve()
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Dataset file not found: {resolved}")
        with open(resolved, encoding=self.encoding) as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{resolved}:{line_number} is not valid JSON: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{resolved}:{line_number} is not a JSON object.")
                yield record

    def count(self) -> int | None:
        resolved = self.resolve()
        if not os.path.exists(resolved):
            return None
        with open(resolved, encoding=self.encoding) as fh:
            return sum(1 for line in fh if line.strip())


class JSONSource(_FileSource):
    """A single JSON array, or an object whose ``key`` holds the array."""

    def __init__(self, path: str, *, key: str | None = None, encoding: str = "utf-8"):
        super().__init__(path, encoding=encoding)
        self.key = key

    def __iter__(self) -> Iterator[dict[str, Any]]:
        with open(self.resolve(), encoding=self.encoding) as fh:
            payload = json.load(fh)
        if self.key is not None:
            payload = payload[self.key]
        if not isinstance(payload, list):
            raise ValueError(f"{self.resolve()} does not contain a list of records.")
        yield from (dict(record) for record in payload)

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        if self.key:
            info["key"] = self.key
        return info


class CSVSource(_FileSource):
    """Delimited text. Values arrive as strings; fields coerce them."""

    def __init__(self, path: str, *, delimiter: str = ",", encoding: str = "utf-8"):
        super().__init__(path, encoding=encoding)
        self.delimiter = delimiter

    def __iter__(self) -> Iterator[dict[str, Any]]:
        with open(self.resolve(), newline="", encoding=self.encoding) as fh:
            for row in csv.DictReader(fh, delimiter=self.delimiter):
                yield dict(row)

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["delimiter"] = self.delimiter
        return info


class DirectorySource(Source):
    """The classic ``root/<class>/<file>`` layout for image and audio sets.

    Yields ``{"path": ..., "label": <parent directory name>, "filename": ...}``.
    """

    def __init__(
        self,
        root: str,
        *,
        extensions: Iterable[str] = ("png", "jpg", "jpeg", "webp"),
        path_field: str = "path",
        label_field: str = "label",
    ):
        self.root = root
        self.extensions = {e.lower().lstrip(".") for e in extensions}
        self.path_field = path_field
        self.label_field = label_field

    def resolve(self) -> str:
        if os.path.isabs(self.root):
            return self.root
        from mlango.conf import settings

        return os.path.join(str(settings.BASE_DIR), self.root)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        root = self.resolve()
        for dirpath, _dirnames, filenames in os.walk(root):
            label = os.path.basename(dirpath)
            for filename in sorted(filenames):
                if filename.rsplit(".", 1)[-1].lower() not in self.extensions:
                    continue
                full = os.path.join(dirpath, filename)
                yield {
                    self.path_field: full,
                    self.label_field: label,
                    "filename": filename,
                }

    def describe(self) -> dict[str, Any]:
        return {
            "type": "DirectorySource",
            "root": self.root,
            "extensions": sorted(self.extensions),
        }


class ChainSource(Source):
    """Concatenate several sources — train shards, multiple dumps."""

    def __init__(self, *sources: Source):
        self.sources = list(sources)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for source in self.sources:
            yield from source

    def count(self) -> int | None:
        counts = [s.count() for s in self.sources]
        return sum(counts) if all(c is not None for c in counts) else None  # type: ignore[arg-type]

    def describe(self) -> dict[str, Any]:
        return {"type": "ChainSource", "sources": [s.describe() for s in self.sources]}


__all__ = [
    "Source",
    "InMemorySource",
    "PythonSource",
    "JSONLSource",
    "JSONSource",
    "CSVSource",
    "DirectorySource",
    "ChainSource",
]
