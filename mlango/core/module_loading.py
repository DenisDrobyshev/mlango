"""Dotted-path import helpers."""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from typing import Any

from mlango.core.exceptions import ImproperlyConfigured


def import_string(dotted_path: str) -> Any:
    """Import ``package.module.Attribute`` and return the attribute itself."""
    try:
        module_path, _, class_name = dotted_path.rpartition(".")
        if not module_path:
            raise ValueError(f"{dotted_path} is not a dotted module path")
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ImproperlyConfigured(f"Could not import {dotted_path!r}: {exc}") from exc


def module_has_submodule(package: Any, module_name: str) -> bool:
    """True when ``package.module_name`` exists without importing it."""
    package_name = getattr(package, "__name__", None)
    package_path = getattr(package, "__path__", None)
    if package_name is None or package_path is None:
        return False
    full_name = f"{package_name}.{module_name}"
    try:
        return importlib.util.find_spec(full_name) is not None
    except (ModuleNotFoundError, ImportError, ValueError):
        return False


def import_submodule(package: Any, module_name: str) -> Any | None:
    """Import ``package.module_name`` when present, else return ``None``.

    An ImportError raised *inside* the submodule is never swallowed — that is
    almost always a real bug in user code, and hiding it makes app loading
    impossible to debug.
    """
    if not module_has_submodule(package, module_name):
        return None
    return importlib.import_module(f"{package.__name__}.{module_name}")


def iter_submodules(package: Any) -> list[str]:
    path = getattr(package, "__path__", None)
    if not path:
        return []
    return [name for _, name, _ in pkgutil.iter_modules(path)]
