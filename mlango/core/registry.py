"""The application registry.

``mlango.apps`` is the single source of truth for "what exists in this
project": which apps are installed and which datasets, models, agents and evals
they declare. The admin, the CLI and the migration autodetector all read from
here rather than scanning the filesystem.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

from mlango.core.apps import AppConfig
from mlango.core.exceptions import AppRegistryNotReady, ImproperlyConfigured
from mlango.core.typing import (
    AgentClass,
    DatasetClass,
    DeclarativeClass,
    EvalClass,
    ModelClass,
)

#: The four families of declarative object, in the order the admin shows them.
KINDS = ("dataset", "model", "agent", "eval")


class Registry:
    def __init__(self) -> None:
        self.app_configs: dict[str, AppConfig] = {}
        self._objects: dict[str, dict[str, DeclarativeClass]] = defaultdict(dict)
        self._pending: list[tuple[str, DeclarativeClass]] = []
        self.ready = False
        self.loading = False
        self._lock = threading.RLock()

    # -- population ----------------------------------------------------------

    def populate(self, installed_apps: list[str] | tuple[str, ...]) -> None:
        """Import every installed app and its declarative modules. Idempotent."""
        if self.ready:
            return
        with self._lock:
            if self.ready:
                return
            if self.loading:
                raise ImproperlyConfigured(
                    "populate() is re-entrant here — an app module imported at "
                    "load time triggered another populate(). Move that import "
                    "into AppConfig.ready()."
                )
            self.loading = True
            try:
                from mlango.conf import settings

                for entry in installed_apps:
                    config = AppConfig.create(entry)
                    if config.label in self.app_configs:
                        raise ImproperlyConfigured(
                            f"Duplicate app label {config.label!r} in INSTALLED_APPS. "
                            f"Give one of them a distinct `label` on its AppConfig."
                        )
                    self.app_configs[config.label] = config

                module_names = tuple(settings.APP_MODULES)
                for config in self.app_configs.values():
                    config.import_declarations(module_names)

                self._flush_pending()
                self.ready = True

                for config in self.app_configs.values():
                    config.ready()
            finally:
                self.loading = False

        from mlango.core.signals import apps_ready

        apps_ready.send(sender=self)

    def check_ready(self) -> None:
        if not self.ready:
            raise AppRegistryNotReady(
                "The app registry has not been populated. Call mlango.setup() first "
                "(manage.py and the CLI do this for you)."
            )

    def clear(self) -> None:
        """Reset the registry — used by tests and by ``startproject``."""
        with self._lock:
            self.app_configs.clear()
            self._objects.clear()
            self._pending.clear()
            self.ready = False
            self.loading = False

    # -- registration --------------------------------------------------------

    def register(self, kind: str, cls: DeclarativeClass) -> None:
        label = cls._meta.label
        if not self.ready and not self.loading:
            # Declared before setup() ran (a test module, a script). Hold it and
            # attach it once apps are known, so nothing is silently lost.
            self._pending.append((kind, cls))
        existing = self._objects[kind].get(label)
        if existing is not None and existing is not cls:
            raise ImproperlyConfigured(
                f"Two {kind}s are both labelled {label!r}: {existing.__module__} and "
                f"{cls.__module__}. Labels must be unique within an app."
            )
        self._objects[kind][label] = cls

    def unregister(self, kind: str, target: DeclarativeClass | str) -> bool:
        """Remove a registration.

        Labels are unique on purpose — two datasets called ``Reviews`` in one
        app is a bug, not a convenience. That makes this the escape hatch for
        tests and notebooks that redeclare a class: unregister the old one
        first, or scope the declaration so it happens once.
        """
        label = target if isinstance(target, str) else target._meta.label
        with self._lock:
            removed = self._objects.get(kind, {}).pop(label, None) is not None
            self._pending = [
                (k, c) for k, c in self._pending if not (k == kind and c._meta.label == label)
            ]
        return removed

    def _flush_pending(self) -> None:
        for kind, cls in self._pending:
            self._objects[kind][cls._meta.label] = cls
        self._pending.clear()

    # -- lookups -------------------------------------------------------------

    def get_app_configs(self) -> list[AppConfig]:
        return list(self.app_configs.values())

    def get_app_config(self, label: str) -> AppConfig:
        try:
            return self.app_configs[label]
        except KeyError as exc:
            known = ", ".join(sorted(self.app_configs)) or "(none installed)"
            raise LookupError(f"No installed app with label {label!r}. Known: {known}.") from exc

    def get_containing_app_config(self, module_name: str) -> AppConfig | None:
        """Find the app a module belongs to, longest prefix wins."""
        best: AppConfig | None = None
        for config in self.app_configs.values():
            name = config.name
            if module_name == name or module_name.startswith(name + "."):
                if best is None or len(name) > len(best.name):
                    best = config
        return best

    def get_registered(self, kind: str | None = None) -> list[DeclarativeClass]:
        if kind is None:
            out: list[DeclarativeClass] = []
            for k in KINDS:
                out.extend(self._objects.get(k, {}).values())
            return out
        return list(self._objects.get(kind, {}).values())

    def get(self, kind: str, label: str) -> Any:
        """Look up by ``"app.Object"`` or, when unambiguous, by ``"Object"``."""
        registry = self._objects.get(kind, {})
        if label in registry:
            return registry[label]
        lowered = label.lower()
        matches = [
            cls
            for key, cls in registry.items()
            if key.lower() == lowered or key.rpartition(".")[2].lower() == lowered
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            names = ", ".join(sorted(c._meta.label for c in matches))
            raise LookupError(f"{label!r} is ambiguous; it matches {names}.")
        known = ", ".join(sorted(registry)) or "(none)"
        raise LookupError(f"No {kind} named {label!r}. Registered {kind}s: {known}.")

    # The per-kind getters return the concrete family, so callers get
    # Dataset.objects and Model.load() without a cast at every call site.

    def get_dataset(self, label: str) -> DatasetClass:
        return self.get("dataset", label)

    def get_model(self, label: str) -> ModelClass:
        return self.get("model", label)

    def get_agent(self, label: str) -> AgentClass:
        return self.get("agent", label)

    def get_eval(self, label: str) -> EvalClass:
        return self.get("eval", label)

    def find(self, label: str) -> tuple[str, Any]:
        """Look a label up across every kind — used by the CLI and admin."""
        for kind in KINDS:
            try:
                return kind, self.get(kind, label)
            except LookupError:
                continue
        raise LookupError(f"Nothing registered under {label!r}.")

    def summary(self) -> dict[str, Any]:
        return {
            "apps": sorted(self.app_configs),
            "counts": {kind: len(self._objects.get(kind, {})) for kind in KINDS},
            "objects": {kind: sorted(self._objects.get(kind, {})) for kind in KINDS},
        }

    def __repr__(self) -> str:
        state = "ready" if self.ready else "not populated"
        return f"<Registry [{state}] apps={len(self.app_configs)}>"


apps = Registry()

__all__ = ["apps", "Registry", "KINDS"]
