"""Artifact storage.

Checkpoints, materialised datasets and run outputs all go through one narrow
interface so that moving a project from a laptop to S3 is a settings change,
not a rewrite.
"""

from __future__ import annotations

import abc
from typing import IO, Any


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
