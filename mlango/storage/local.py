"""Filesystem-backed artifact storage — the zero-configuration default."""

from __future__ import annotations

import os
import shutil
from typing import IO, Any

from mlango.storage.base import Storage


class LocalStorage(Storage):
    def __init__(self, root: str = "artifacts", **options: Any):
        super().__init__(root=root, **options)
        from mlango.conf import settings

        self.root = root if os.path.isabs(root) else os.path.join(str(settings.BASE_DIR), root)

    # -- helpers -------------------------------------------------------------

    def _abs(self, name: str, *, create_parent: bool = False) -> str:
        target = os.path.normpath(os.path.join(self.root, name.replace("\\", "/")))
        root = os.path.normpath(self.root)
        if not (target == root or target.startswith(root + os.sep)):
            raise ValueError(f"Refusing to access {name!r}: it escapes the storage root.")
        if create_parent:
            os.makedirs(os.path.dirname(target) or root, exist_ok=True)
        return target

    # -- Storage API ---------------------------------------------------------

    def path(self, name: str) -> str:
        return self._abs(name, create_parent=True)

    def open(self, name: str, mode: str = "rb") -> IO[Any]:
        writing = any(flag in mode for flag in ("w", "a", "x", "+"))
        return open(self._abs(name, create_parent=writing), mode)

    def save_bytes(self, name: str, data: bytes) -> str:
        target = self._abs(name, create_parent=True)
        # Write to a sibling temp file then replace, so a crash mid-write never
        # leaves a half-written checkpoint that looks valid.
        tmp = target + ".part"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, target)
        return target

    def read_bytes(self, name: str) -> bytes:
        with open(self._abs(name), "rb") as fh:
            return fh.read()

    def exists(self, name: str) -> bool:
        return os.path.exists(self._abs(name))

    def delete(self, name: str) -> None:
        target = self._abs(name)
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
        elif os.path.exists(target):
            os.remove(target)

    def size(self, name: str) -> int:
        return os.path.getsize(self._abs(name))

    def listdir(self, prefix: str = "") -> list[str]:
        base = self._abs(prefix) if prefix else self.root
        if not os.path.isdir(base):
            return []
        out: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(base):
            for filename in filenames:
                full = os.path.join(dirpath, filename)
                out.append(os.path.relpath(full, self.root).replace("\\", "/"))
        return sorted(out)

    def makedirs(self, name: str) -> str:
        target = self._abs(name)
        os.makedirs(target, exist_ok=True)
        return target

    def __repr__(self) -> str:
        return f"<LocalStorage root={self.root!r}>"
