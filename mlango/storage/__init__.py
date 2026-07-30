"""Pluggable artifact storage."""

from mlango.storage.base import Storage, default_storage, reset_default_storage
from mlango.storage.local import LocalStorage

__all__ = ["Storage", "LocalStorage", "default_storage", "reset_default_storage"]
