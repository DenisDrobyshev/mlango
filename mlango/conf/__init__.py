"""Settings access.

``from mlango.conf import settings`` gives a lazy object that resolves the
project's settings module on first attribute access, exactly like Django. The
module is named by the ``MLANGO_SETTINGS_MODULE`` environment variable or by an
explicit :meth:`Settings.configure` call.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from mlango.core.exceptions import ImproperlyConfigured

from . import global_settings

ENVIRONMENT_VARIABLE = "MLANGO_SETTINGS_MODULE"

_empty = object()


class SettingsHolder:
    """The resolved settings: defaults overlaid with the project's module."""

    def __init__(self, settings_module: str | None = None, **overrides: Any):
        self._explicit: set[str] = set()
        self.SETTINGS_MODULE = settings_module

        for setting in dir(global_settings):
            if setting.isupper():
                setattr(self, setting, getattr(global_settings, setting))

        if settings_module:
            mod = importlib.import_module(settings_module)
            for setting in dir(mod):
                if not setting.isupper():
                    continue
                setattr(self, setting, getattr(mod, setting))
                self._explicit.add(setting)

        for setting, value in overrides.items():
            if not setting.isupper():
                raise ImproperlyConfigured(
                    f"Setting {setting!r} must be uppercase to be recognised."
                )
            setattr(self, setting, value)
            self._explicit.add(setting)

        self._normalise()

    def _normalise(self) -> None:
        # Nested dict settings are merged with their defaults so a project can
        # override just METASTORE["URL"] without restating the whole mapping.
        for name in ("METASTORE", "STORAGE"):
            default = dict(getattr(global_settings, name))
            default.update(getattr(self, name, {}) or {})
            setattr(self, name, default)

        for name in ("TRAINERS", "PROVIDERS"):
            merged = dict(getattr(global_settings, name))
            merged.update(getattr(self, name, {}) or {})
            setattr(self, name, merged)

        if not getattr(self, "BASE_DIR", None):
            self.BASE_DIR = os.getcwd()

    def is_overridden(self, setting: str) -> bool:
        return setting in self._explicit

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in dir(self) if k.isupper()}


class LazySettings:
    """Defers loading the settings module until something is actually read."""

    def __init__(self) -> None:
        self._wrapped: SettingsHolder | object = _empty

    # -- setup ---------------------------------------------------------------

    def _setup(self, name: str | None = None) -> None:
        settings_module = os.environ.get(ENVIRONMENT_VARIABLE)
        if not settings_module:
            hint = f"accessing settings.{name}" if name else "using settings"
            raise ImproperlyConfigured(
                f"Settings are not configured; you cannot start {hint}. "
                f"Define the {ENVIRONMENT_VARIABLE} environment variable or call "
                f"settings.configure() before touching settings."
            )
        self._wrapped = SettingsHolder(settings_module)

    def configure(self, settings_module: str | None = None, **overrides: Any) -> None:
        """Configure settings by hand — handy in notebooks and tests."""
        self._wrapped = SettingsHolder(settings_module, **overrides)

    def configure_from_module(self, settings_module: str) -> None:
        os.environ[ENVIRONMENT_VARIABLE] = settings_module
        self._wrapped = SettingsHolder(settings_module)

    def reset(self) -> None:
        self._wrapped = _empty

    @property
    def configured(self) -> bool:
        return self._wrapped is not _empty

    # -- attribute proxying --------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if self._wrapped is _empty:
            self._setup(name)
        try:
            return getattr(self._wrapped, name)
        except AttributeError as exc:
            raise AttributeError(
                f"{name!r} is not a known mlango setting. Settings must be uppercase "
                f"and declared in mlango.conf.global_settings."
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_wrapped":
            object.__setattr__(self, name, value)
            return
        if self._wrapped is _empty:
            self._setup(name)
        setattr(self._wrapped, name, value)

    def __dir__(self):  # pragma: no cover - interactive convenience
        if self._wrapped is _empty:
            return list(super().__dir__())
        return dir(self._wrapped)

    def __repr__(self) -> str:
        if self._wrapped is _empty:
            return "<LazySettings [unconfigured]>"
        return f"<LazySettings {getattr(self._wrapped, 'SETTINGS_MODULE', None)!r}>"


settings = LazySettings()

__all__ = ["settings", "SettingsHolder", "LazySettings", "ENVIRONMENT_VARIABLE"]
