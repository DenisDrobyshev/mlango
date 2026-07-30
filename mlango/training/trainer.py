"""Trainer backends.

A trainer knows how to turn a declared :class:`~mlango.training.model.Model`
plus data into a fitted object, how to get predictions out of it, and how to
save and load it. Everything else — run tracking, metrics, versioning,
callbacks — is the framework's job and is written once.

That split is why adding a backend is a small file rather than a fork.
"""

from __future__ import annotations

import abc
from typing import Any

from mlango.core.exceptions import BackendNotAvailable, ImproperlyConfigured
from mlango.core.module_loading import import_string

_cache: dict[str, Trainer] = {}


class Trainer(abc.ABC):
    """The contract every backend implements."""

    #: Key used in ``settings.TRAINERS`` and ``Model.Meta.trainer``.
    name: str = ""
    #: Importable modules the backend needs, checked before use.
    requires: tuple[str, ...] = ()
    #: File extension used when saving.
    extension: str = "bin"

    def __init__(self, **options: Any):
        self.options = options

    # -- availability --------------------------------------------------------

    @classmethod
    def check_available(cls) -> None:
        import importlib.util

        missing = [m for m in cls.requires if importlib.util.find_spec(m) is None]
        if missing:
            raise BackendNotAvailable(
                f"The {cls.name!r} trainer needs {', '.join(missing)}. "
                f"Install it with: pip install mlango[{cls.name}]"
            )

    # -- required behaviour --------------------------------------------------

    @abc.abstractmethod
    def fit(
        self,
        model: Any,
        train: Any,
        validation: Any | None,
        run: Any,
        callbacks: Any,
        **kwargs: Any,
    ) -> Any:
        """Train and return the fitted object."""

    @abc.abstractmethod
    def predict(self, model: Any, fitted: Any, inputs: list[Any]) -> list[Any]:
        """Predictions for a batch of raw inputs."""

    @abc.abstractmethod
    def save(self, model: Any, fitted: Any, name: str) -> str:
        """Persist ``fitted`` under ``name`` in storage; return the full path."""

    @abc.abstractmethod
    def load(self, model: Any, path: str) -> Any:
        """Restore a fitted object saved by :meth:`save`."""

    # -- optional behaviour --------------------------------------------------

    def predict_proba(self, model: Any, fitted: Any, inputs: list[Any]) -> list[Any] | None:
        """Class probabilities when the backend can produce them."""
        return None

    def describe(self, model: Any, fitted: Any) -> dict[str, Any]:
        """Backend-specific detail shown in the admin."""
        return {"backend": self.name}

    def resolve_device(self) -> str:
        from mlango.conf import settings

        device = str(settings.DEVICE)
        if device != "auto":
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def __repr__(self) -> str:
        return f"<Trainer {self.name!r}>"


def get_trainer(name: str) -> Trainer:
    """Build (and cache) the trainer registered under ``name``."""
    if name in _cache:
        return _cache[name]

    from mlango.conf import settings

    try:
        path = settings.TRAINERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(settings.TRAINERS)) or "(none)"
        raise ImproperlyConfigured(
            f"No trainer registered as {name!r}. Available: {known}. "
            f"Add one to the TRAINERS setting."
        ) from exc

    trainer_class = import_string(str(path))
    if not (isinstance(trainer_class, type) and issubclass(trainer_class, Trainer)):
        raise ImproperlyConfigured(f"{path} is not a Trainer subclass.")
    trainer_class.check_available()
    instance = trainer_class()
    instance.name = instance.name or name
    _cache[name] = instance
    return instance


def available_trainers() -> dict[str, bool]:
    """Every registered trainer and whether its dependencies are installed."""
    from mlango.conf import settings

    out: dict[str, bool] = {}
    for name, path in settings.TRAINERS.items():
        try:
            trainer_class = import_string(str(path))
            trainer_class.check_available()
            out[name] = True
        except Exception:
            out[name] = False
    return out


def clear_trainer_cache() -> None:
    _cache.clear()


__all__ = ["Trainer", "get_trainer", "available_trainers", "clear_trainer_cache"]
