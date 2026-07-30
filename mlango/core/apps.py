"""Application configuration.

An mlango app is an importable package that groups datasets, models, agents and
evals belonging to one problem domain — ``sentiment``, ``recsys``, ``support``.
Apps are the unit of reuse and the unit of migration, exactly as in Django.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from mlango.core.exceptions import ImproperlyConfigured
from mlango.core.module_loading import import_string, module_has_submodule


class AppConfig:
    """Metadata about one installed app.

    Subclass it in ``<app>/apps.py`` to set a verbose name or to run wiring
    code in :meth:`ready` — signal receivers, backend registration, and so on.
    """

    #: Full dotted Python path, e.g. ``"sentiment"`` or ``"project.sentiment"``.
    name: str = ""
    #: Short unique label used in object labels and migration directories.
    label: str = ""
    #: Human-readable name for the admin.
    verbose_name: str = ""
    #: Set by the registry.
    module: Any = None
    path: str = ""

    def __init__(self, app_name: str, app_module: Any):
        self.name = app_name
        self.module = app_module
        if not self.label:
            self.label = app_name.rpartition(".")[2]
        if not self.verbose_name:
            self.verbose_name = self.label.replace("_", " ").title()
        self.path = self._resolve_path(app_module)
        self._imported: dict[str, Any] = {}

    # -- construction --------------------------------------------------------

    @staticmethod
    def _resolve_path(module: Any) -> str:
        paths = list(getattr(module, "__path__", []))
        if paths:
            return paths[0]
        filename = getattr(module, "__file__", None)
        return os.path.dirname(os.path.abspath(filename)) if filename else ""

    @classmethod
    def create(cls, entry: str) -> AppConfig:
        """Build an AppConfig from an ``INSTALLED_APPS`` entry.

        The entry may point at a package (``"sentiment"``), in which case an
        ``apps.py`` with a single AppConfig subclass is honoured, or directly
        at an AppConfig subclass (``"sentiment.apps.SentimentConfig"``).
        """
        try:
            module = importlib.import_module(entry)
        except ImportError:
            module = None

        if module is not None and not module_has_submodule(module, "apps"):
            return cls(entry, module)

        if module is not None:
            candidates = _app_configs_in(f"{entry}.apps")
            if len(candidates) > 1:
                explicit = [c for c in candidates if getattr(c, "default", True)]
                candidates = explicit or candidates
            if len(candidates) == 1:
                config_class = candidates[0]
                app_name = getattr(config_class, "name", None) or entry
                return config_class(app_name, importlib.import_module(app_name))
            return cls(entry, module)

        # Not a package: must be a dotted path to an AppConfig subclass.
        try:
            config_class = import_string(entry)
        except ImproperlyConfigured as exc:
            raise ImproperlyConfigured(
                f"Could not import INSTALLED_APPS entry {entry!r}. It must name either an "
                f"importable package or an AppConfig subclass."
            ) from exc
        if not (isinstance(config_class, type) and issubclass(config_class, AppConfig)):
            raise ImproperlyConfigured(f"{entry!r} is not an AppConfig subclass.")
        app_name = getattr(config_class, "name", None)
        if not app_name:
            raise ImproperlyConfigured(f"{entry!r} must declare a `name` attribute.")
        return config_class(app_name, importlib.import_module(app_name))

    # -- lifecycle -----------------------------------------------------------

    def import_declarations(self, module_names: tuple[str, ...]) -> None:
        """Import the app's declarative modules so classes register themselves."""
        for module_name in module_names:
            if not module_has_submodule(self.module, module_name):
                continue
            self._imported[module_name] = importlib.import_module(f"{self.name}.{module_name}")

    def ready(self) -> None:
        """Hook for wiring performed once every app is loaded."""

    # -- introspection -------------------------------------------------------

    @property
    def migrations_dir(self) -> str:
        return os.path.join(self.path, "migrations")

    def objects(self, kind: str | None = None) -> list[type]:
        from mlango.core.registry import apps

        return [
            obj
            for obj in apps.get_registered(kind)
            if getattr(obj, "_meta", None) and obj._meta.app_label == self.label
        ]

    def __repr__(self) -> str:
        return f"<AppConfig: {self.label}>"


def _app_configs_in(module_path: str) -> list[type[AppConfig]]:
    module = importlib.import_module(module_path)
    return [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type)
        and issubclass(obj, AppConfig)
        and obj is not AppConfig
        and obj.__module__ == module_path
    ]
