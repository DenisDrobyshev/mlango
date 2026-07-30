"""Sources for the formats real projects actually keep data in.

Each one imports its dependency lazily and reports a fixable message when it is
missing, so the core install stays small and a project only pays for what it
uses.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from mlango.core.exceptions import ImproperlyConfigured
from mlango.data.sources import Source


def _require(module: str, extra: str) -> Any:
    """Import a lazy dependency, or explain exactly how to install it."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImproperlyConfigured(
            f"This source needs {module}. Install it with: pip install 'mlango[{extra}]'"
        ) from exc


class ParquetSource(Source):
    """Columnar data via pyarrow, read in row-group batches.

    Streams rather than loading the file, so a Parquet file larger than memory
    still works with a lazy queryset.
    """

    def __init__(
        self,
        path: str,
        *,
        columns: list[str] | None = None,
        batch_size: int = 8192,
    ):
        self.path = path
        self.columns = list(columns) if columns else None
        self.batch_size = batch_size

    def resolve(self) -> str:
        if os.path.isabs(self.path):
            return self.path
        from mlango.conf import settings

        return os.path.join(str(settings.BASE_DIR), self.path)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        pq = _require("pyarrow.parquet", "parquet")

        resolved = self.resolve()
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Parquet file not found: {resolved}")

        parquet_file = pq.ParquetFile(resolved)
        for batch in parquet_file.iter_batches(batch_size=self.batch_size, columns=self.columns):
            yield from batch.to_pylist()

    def count(self) -> int | None:
        pq = _require("pyarrow.parquet", "parquet")

        resolved = self.resolve()
        if not os.path.exists(resolved):
            return None
        # Parquet stores the row count in its footer, so this is cheap.
        return int(pq.ParquetFile(resolved).metadata.num_rows)

    def describe(self) -> dict[str, Any]:
        return {
            "type": "ParquetSource",
            "path": self.path,
            "columns": self.columns,
        }


class SQLSource(Source):
    """Rows from any database SQLAlchemy can reach.

    ``url`` defaults to the project's metastore, which makes it easy to build a
    dataset from data the framework already recorded — evaluation results, for
    instance, becoming the training set for a reranker.
    """

    def __init__(
        self,
        query: str,
        *,
        url: str | None = None,
        params: dict[str, Any] | None = None,
        stream_size: int = 1000,
    ):
        self.query = query
        self.url = url
        self.params = dict(params or {})
        self.stream_size = stream_size

    def _engine(self) -> Any:
        from sqlalchemy import create_engine

        if self.url:
            return create_engine(self.url)
        from mlango.metastore.session import get_engine

        return get_engine()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        from sqlalchemy import text

        engine = self._engine()
        with engine.connect() as connection:
            result = connection.execution_options(stream_results=True).execute(
                text(self.query), self.params
            )
            while True:
                rows = result.fetchmany(self.stream_size)
                if not rows:
                    break
                for row in rows:
                    yield dict(row._mapping)

    def count(self) -> int | None:
        from sqlalchemy import text

        # Wrapping an arbitrary query in a COUNT is not always valid SQL, so a
        # failure here means "unknown" rather than an error.
        try:
            engine = self._engine()
            with engine.connect() as connection:
                wrapped = f"SELECT count(*) FROM ({self.query}) AS mlango_count"
                return int(connection.execute(text(wrapped), self.params).scalar_one())
        except Exception:
            return None

    def describe(self) -> dict[str, Any]:
        # The URL may carry a password, so it is never recorded.
        return {
            "type": "SQLSource",
            "query": self.query,
            "url": "metastore" if not self.url else "external",
        }


class HuggingFaceSource(Source):
    """A split of a dataset from the Hugging Face hub.

    ``streaming=True`` avoids downloading the whole thing, which matters for the
    large public corpora.
    """

    def __init__(
        self,
        path: str,
        *,
        split: str = "train",
        name: str | None = None,
        streaming: bool = False,
        rename: dict[str, str] | None = None,
        **load_kwargs: Any,
    ):
        self.path = path
        self.split = split
        self.name = name
        self.streaming = streaming
        self.rename = dict(rename or {})
        self.load_kwargs = load_kwargs

    def _load(self) -> Any:
        datasets = _require("datasets", "huggingface")

        return datasets.load_dataset(
            self.path,
            name=self.name,
            split=self.split,
            streaming=self.streaming,
            **self.load_kwargs,
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in self._load():
            record = dict(row)
            for source_key, target_key in self.rename.items():
                if source_key in record:
                    record[target_key] = record.pop(source_key)
            yield record

    def count(self) -> int | None:
        if self.streaming:
            return None
        try:
            return int(self._load().num_rows)
        except Exception:
            return None

    def describe(self) -> dict[str, Any]:
        return {
            "type": "HuggingFaceSource",
            "path": self.path,
            "split": self.split,
            "name": self.name,
            "streaming": self.streaming,
        }


class DatasetVersionSource(Source):
    """A materialised snapshot of another dataset.

    Lets one dataset build on another's frozen version, so a derived dataset is
    pinned to an exact upstream state rather than whatever the source says today.
    """

    def __init__(self, label: str, version: int | None = None):
        self.label = label
        self.version = version

    def _resolve_path(self) -> str:
        from sqlalchemy import select

        from mlango.metastore.models import DatasetVersion
        from mlango.metastore.session import session_scope
        from mlango.storage import default_storage

        with session_scope() as session:
            statement = select(DatasetVersion).where(DatasetVersion.label == self.label)
            statement = (
                statement.where(DatasetVersion.version == self.version)
                if self.version is not None
                else statement.order_by(DatasetVersion.version.desc())
            )
            row = session.execute(statement).scalars().first()

        if row is None:
            detail = f" v{self.version}" if self.version is not None else ""
            raise LookupError(
                f"{self.label} has no materialised version{detail}. "
                f"Run: manage.py dataset materialize {self.label}"
            )
        return default_storage().path(row.path or "")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        from mlango.data.sources import JSONLSource

        yield from JSONLSource(self._resolve_path())

    def count(self) -> int | None:
        from mlango.data.sources import JSONLSource

        try:
            return JSONLSource(self._resolve_path()).count()
        except LookupError:
            return None

    def describe(self) -> dict[str, Any]:
        return {
            "type": "DatasetVersionSource",
            "label": self.label,
            "version": self.version,
        }


__all__ = [
    "ParquetSource",
    "SQLSource",
    "HuggingFaceSource",
    "DatasetVersionSource",
]
