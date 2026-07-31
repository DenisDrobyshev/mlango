"""Artifact storage.

Checkpoints, materialised datasets and run outputs all go through one narrow
interface so that moving a project from a laptop to S3 is a settings change,
not a rewrite.

The pair that makes that true is :meth:`Storage.writable` and
:meth:`Storage.readable`. Every serialisation library worth using — joblib,
torch, transformers — wants a filesystem path, so a backend that is not a
filesystem has to stage one. Doing that inside the storage layer is what keeps
it out of every trainer.
"""

from __future__ import annotations

import abc
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO, Any


class Written:
    """A local path to write to, and the name it will be stored under.

    Returned by :meth:`Storage.writable`. ``path`` is somewhere a library can
    write; ``name`` is what to record in the metastore. They are the same
    string for local storage and deliberately different for anything remote.
    """

    def __init__(self, path: str, name: str):
        self.path = path
        self.name = name

    def __fspath__(self) -> str:
        return self.path

    def __str__(self) -> str:
        return self.path


class Storage(abc.ABC):
    """The contract every storage backend implements."""

    def __init__(self, **options: Any):
        self.options = options

    @abc.abstractmethod
    def path(self, name: str) -> str:
        """Absolute location for ``name``, creating parent directories."""

    @abc.abstractmethod
    def open(self, name: str, mode: str = "rb") -> IO[Any]: ...

    @abc.abstractmethod
    def save_bytes(self, name: str, data: bytes) -> str: ...

    @abc.abstractmethod
    def read_bytes(self, name: str) -> bytes: ...

    @abc.abstractmethod
    def exists(self, name: str) -> bool: ...

    @abc.abstractmethod
    def delete(self, name: str) -> None: ...

    @abc.abstractmethod
    def size(self, name: str) -> int: ...

    @abc.abstractmethod
    def listdir(self, prefix: str = "") -> list[str]: ...

    # -- staging -------------------------------------------------------------

    @contextmanager
    def writable(self, name: str, *, directory: bool = False) -> Iterator[Written]:
        """A local path to write ``name`` to, published when the block exits.

        A backend that *is* a filesystem hands back the real path and has
        nothing to publish, which is why the default implementation is this
        short. Everything else stages locally and uploads on the way out —
        and only on success, so a crashed run does not leave half an artifact
        in the bucket for the next load to find.
        """
        yield Written(self.makedirs(name) if directory else self.path(name), name)

    @contextmanager
    def readable(self, name: str) -> Iterator[str]:
        """A local path holding ``name``, valid for the length of the block."""
        yield self.locate(name)

    def locate(self, name: str) -> str:
        """Resolve a stored name to a local path, if this backend has one.

        Absolute paths pass through. Versions registered before artifacts were
        recorded by name carry one, and refusing to load them would break
        every project that upgrades.
        """
        if os.path.isabs(name):
            return name
        return self.path(name)

    def fetch(self, name: str) -> str:
        """A local path for ``name`` that outlives the call.

        For anything lazily read — a materialised dataset iterated long after
        it was resolved — a with-block is the wrong shape, so a remote backend
        caches a copy instead. A filesystem backend has nothing to do.
        """
        return self.locate(name)

    def makedirs(self, name: str) -> str:
        """A local directory to write ``name`` into."""
        return self.path(name)

    # -- convenience ---------------------------------------------------------

    def save_text(self, name: str, text: str, encoding: str = "utf-8") -> str:
        return self.save_bytes(name, text.encode(encoding))

    def read_text(self, name: str, encoding: str = "utf-8") -> str:
        return self.read_bytes(name).decode(encoding)

    def url(self, name: str) -> str:
        return self.path(name)


_default: Storage | None = None


def default_storage() -> Storage:
    """The project's configured storage backend, built once and cached."""
    global _default
    if _default is None:
        from mlango.conf import settings
        from mlango.core.module_loading import import_string

        config = dict(settings.STORAGE)
        backend = config.pop("BACKEND")
        _default = import_string(str(backend))(**{k.lower(): v for k, v in config.items()})
    return _default


def reset_default_storage() -> None:
    """Forget the cached backend — used when settings change under tests."""
    global _default
    _default = None
