"""The ``Dataset`` declarative class and its manager.

    class Reviews(Dataset):
        text = TextField()
        label = LabelField(["neg", "pos"])

        class Meta:
            source = JSONLSource("data/reviews.jsonl")
            primary_key = "id"

``Reviews.objects`` then behaves like a Django manager, and
``Reviews.materialize()`` freezes the current view into a content-addressed
version that runs can point at.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from typing import Any

from mlango.core.base import Declarative
from mlango.core.exceptions import ImproperlyConfigured
from mlango.core.hashing import canonical_json
from mlango.core.typing import DatasetClass
from mlango.data.query import DataQuerySet, Record
from mlango.data.sources import InMemorySource, PythonSource, Source

#: Methods forwarded from the manager straight to a fresh QuerySet.
_QUERYSET_METHODS = (
    "filter",
    "exclude",
    "where",
    "map",
    "annotate",
    "only",
    "defer",
    "rename",
    "order_by",
    "shuffle",
    "skip",
    "take",
    "distinct",
    "validate",
    "clean",
    "repeat",
    "split",
    "batch",
    "all",
    "cache",
    "count",
    "first",
    "get",
    "exists",
    "values",
    "values_list",
    "columns",
    "xy",
    "to_pandas",
    "fingerprint",
    "content_hash",
    "describe_pipeline",
)


class Manager:
    """``Dataset.objects`` — hands out fresh QuerySets."""

    def __init__(self, dataset: DatasetClass):
        self.dataset = dataset

    def get_queryset(self) -> DataQuerySet:
        return DataQuerySet(self.dataset)

    def __iter__(self) -> Iterator[Record]:
        return iter(self.get_queryset())

    def __getitem__(self, item: Any) -> Any:
        return self.get_queryset()[item]

    def __getattr__(self, name: str) -> Any:
        if name in _QUERYSET_METHODS:
            return getattr(self.get_queryset(), name)
        raise AttributeError(
            f"{self.dataset.__name__}.objects has no attribute {name!r}. "
            f"QuerySet methods available: {', '.join(_QUERYSET_METHODS)}."
        )

    def __repr__(self) -> str:
        return f"<Manager for {self.dataset._meta.label}>"


class Dataset(Declarative):
    """A declared, versionable collection of records."""

    _kind = "dataset"
    _meta_options = (
        "source",
        "primary_key",
        "split_salt",
        "default_splits",
        "license",
        "homepage",
    )

    objects: Manager

    class Meta:
        abstract = True

    # -- class wiring --------------------------------------------------------

    @classmethod
    def _prepare(cls) -> None:
        if not cls._meta.abstract:
            cls.objects = Manager(cls)

    # -- source --------------------------------------------------------------

    @classmethod
    def records(cls) -> Any:
        """Override to produce records in code instead of declaring a source."""
        raise NotImplementedError

    @classmethod
    def get_source(cls) -> Source | None:
        """Resolve ``Meta.source``, falling back to an overridden ``records()``."""
        source = cls._meta.extras.get("source")
        if source is not None:
            if isinstance(source, Source):
                return source
            if callable(source):
                return PythonSource(source)
            if isinstance(source, (list, tuple)):
                return InMemorySource(source)
            raise ImproperlyConfigured(
                f"{cls._meta.label}.Meta.source must be a Source, a callable or a list; "
                f"got {type(source).__name__}."
            )
        if cls.records.__func__ is not Dataset.records.__func__:  # type: ignore[attr-defined]
            return PythonSource(cls.records)
        return None

    # -- versioning ----------------------------------------------------------

    @classmethod
    def storage_prefix(cls, version: int | None = None) -> str:
        label = cls._meta.label.replace(".", "/")
        return f"datasets/{label}" if version is None else f"datasets/{label}/v{version}"

    @classmethod
    def materialize(
        cls,
        queryset: DataQuerySet | None = None,
        *,
        notes: str = "",
        force: bool = False,
    ) -> Any:
        """Freeze the current view into a numbered, content-addressed version.

        Re-materialising identical data returns the existing version instead of
        piling up duplicates, so putting ``materialize()`` in a nightly job is
        cheap when nothing changed.
        """
        from mlango.core.signals import dataset_materialized

        query = queryset if queryset is not None else cls.objects.get_queryset()
        opts = cls._meta

        # Staged outside storage, because the version number — and so the name
        # this ends up under — is not known until the content hash has decided
        # whether a new version is being created at all.
        staging_dir = tempfile.mkdtemp(prefix="mlango-materialize-")
        staging_path = os.path.join(staging_dir, "data.jsonl")
        digest = hashlib.sha256()
        row_count = 0

        try:
            with open(staging_path, "w", encoding="utf-8", newline="\n") as fh:
                for record in query:
                    payload = canonical_json(dict(record))
                    digest.update(payload.encode("utf-8"))
                    digest.update(b"\n")
                    fh.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
                    row_count += 1

            content_hash = digest.hexdigest()
            schema_fingerprint = opts.fingerprint()
            version = cls._register_materialized(
                query,
                staging_path,
                content_hash=content_hash,
                schema_fingerprint=schema_fingerprint,
                row_count=row_count,
                notes=notes,
                force=force,
            )
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

        dataset_materialized.send(sender=cls, dataset=cls, version=version)
        return version

    @classmethod
    def _register_materialized(
        cls,
        query: Any,
        staging_path: str,
        *,
        content_hash: str,
        schema_fingerprint: str,
        row_count: int,
        notes: str,
        force: bool,
    ) -> Any:
        from sqlalchemy import func, select

        from mlango.metastore.models import DatasetVersion
        from mlango.metastore.session import session_scope
        from mlango.storage import default_storage

        storage = default_storage()
        opts = cls._meta

        with session_scope() as session:
            if not force:
                # first(), not scalar_one_or_none(): force=True deliberately
                # creates duplicates, so several rows can share a content hash.
                # Return the earliest — the original snapshot of this content.
                existing = (
                    session.execute(
                        select(DatasetVersion)
                        .where(
                            DatasetVersion.label == opts.label,
                            DatasetVersion.content_hash == content_hash,
                            DatasetVersion.fingerprint == schema_fingerprint,
                        )
                        .order_by(DatasetVersion.version)
                    )
                    .scalars()
                    .first()
                )
                if existing is not None:
                    return existing

            highest = session.execute(
                select(func.max(DatasetVersion.version)).where(DatasetVersion.label == opts.label)
            ).scalar()
            version_number = (highest or 0) + 1

            final_name = f"{cls.storage_prefix(version_number)}/data.jsonl"
            with storage.writable(final_name) as target:
                # Copied rather than moved: the staging directory is a temp dir
                # and the storage root need not be on the same volume.
                shutil.copyfile(staging_path, target.path)

            version = DatasetVersion(
                label=opts.label,
                version=version_number,
                fingerprint=schema_fingerprint,
                content_hash=content_hash,
                row_count=row_count,
                schema=opts.schema(),
                pipeline=query.describe_pipeline(),
                path=final_name,
                notes=notes,
            )
            session.add(version)

        return version

    @classmethod
    def versions(cls) -> list[Any]:
        from sqlalchemy import select

        from mlango.metastore.models import DatasetVersion
        from mlango.metastore.session import session_scope

        with session_scope() as session:
            return list(
                session.execute(
                    select(DatasetVersion)
                    .where(DatasetVersion.label == cls._meta.label)
                    .order_by(DatasetVersion.version.desc())
                ).scalars()
            )

    @classmethod
    def latest_version(cls) -> Any | None:
        versions = cls.versions()
        return versions[0] if versions else None

    @classmethod
    def load_version(cls, version: int | None = None) -> DataQuerySet:
        """A QuerySet reading a frozen snapshot rather than the live source."""
        from sqlalchemy import select

        from mlango.data.sources import JSONLSource
        from mlango.metastore.models import DatasetVersion
        from mlango.metastore.session import session_scope
        from mlango.storage import default_storage

        with session_scope() as session:
            statement = select(DatasetVersion).where(DatasetVersion.label == cls._meta.label)
            statement = (
                statement.where(DatasetVersion.version == version)
                if version is not None
                else statement.order_by(DatasetVersion.version.desc())
            )
            row = session.execute(statement).scalars().first()

        if row is None:
            raise LookupError(
                f"{cls._meta.label} has no materialised version"
                + (f" {version}." if version is not None else ". Run materialize() first.")
            )
        return DataQuerySet(cls, JSONLSource(default_storage().fetch(row.path or "")))

    # -- convenience ---------------------------------------------------------

    @classmethod
    def peek(cls, n: int = 5) -> list[Record]:
        return list(cls.objects.take(n))

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return cls._meta.schema()

    @classmethod
    def label_field(cls):
        targets = cls._meta.target_fields
        return targets[0] if targets else None

    @classmethod
    def summary(cls) -> dict[str, Any]:
        source = cls.get_source()
        return {
            "label": cls._meta.label,
            "fields": cls._meta.field_names,
            "targets": [f.name for f in cls._meta.target_fields],
            "source": source.describe() if source is not None else None,
            "rows": source.count() if source is not None else None,
        }


__all__ = ["Dataset", "Manager", "DataQuerySet", "Record"]
