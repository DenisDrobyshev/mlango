"""A minimal synchronous signal dispatcher.

Same contract as Django's: receivers are ``fn(sender, **kwargs)``, connecting
twice with the same ``dispatch_uid`` is a no-op, and a receiver that raises does
not take the whole run down — it is logged and the remaining receivers still
fire. Instrumentation must never be the reason a training job dies.
"""

from __future__ import annotations

import logging
import threading
import weakref
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("mlango.signals")


class Signal:
    def __init__(self, name: str, providing_args: tuple[str, ...] = ()):
        self.name = name
        self.providing_args = providing_args
        self._receivers: list[tuple[Any, Any, bool]] = []
        self._lock = threading.Lock()

    # -- wiring --------------------------------------------------------------

    def connect(
        self,
        receiver: Callable[..., Any],
        sender: Any = None,
        *,
        weak: bool = True,
        dispatch_uid: str | None = None,
    ) -> Callable[..., Any]:
        key = dispatch_uid or (id(receiver), id(sender))
        ref: Any = receiver
        is_weak = False
        if weak:
            try:
                if hasattr(receiver, "__self__") and hasattr(receiver, "__func__"):
                    ref = weakref.WeakMethod(receiver)
                else:
                    ref = weakref.ref(receiver)
                is_weak = True
            except TypeError:
                ref, is_weak = receiver, False

        with self._lock:
            if any(existing_key == key for existing_key, _, _ in self._receivers):
                return receiver
            self._receivers.append((key, (ref, sender), is_weak))
        return receiver

    def disconnect(
        self, receiver: Callable[..., Any] | None = None, sender: Any = None, *, dispatch_uid=None
    ) -> bool:
        key = dispatch_uid or (id(receiver), id(sender))
        with self._lock:
            for index, (existing_key, _, _) in enumerate(self._receivers):
                if existing_key == key:
                    del self._receivers[index]
                    return True
        return False

    def has_listeners(self, sender: Any = None) -> bool:
        return bool(self._live_receivers(sender))

    # -- dispatch ------------------------------------------------------------

    def _live_receivers(self, sender: Any) -> list[Callable[..., Any]]:
        live: list[Callable[..., Any]] = []
        dead: list[Any] = []
        for key, (ref, wanted_sender), is_weak in list(self._receivers):
            if wanted_sender is not None and wanted_sender is not sender:
                continue
            fn = ref() if is_weak else ref
            if fn is None:
                dead.append(key)
                continue
            live.append(fn)
        if dead:
            with self._lock:
                self._receivers = [r for r in self._receivers if r[0] not in dead]
        return live

    def send(self, sender: Any, **kwargs: Any) -> list[tuple[Callable[..., Any], Any]]:
        """Call every receiver, swallowing and logging individual failures."""
        results = []
        for fn in self._live_receivers(sender):
            try:
                results.append((fn, fn(sender=sender, signal=self, **kwargs)))
            except Exception:
                logger.exception("Receiver %r for signal %r failed", fn, self.name)
                results.append((fn, None))
        return results

    def send_strict(self, sender: Any, **kwargs: Any) -> list[tuple[Callable[..., Any], Any]]:
        """Like :meth:`send`, but the first failing receiver propagates."""
        return [
            (fn, fn(sender=sender, signal=self, **kwargs)) for fn in self._live_receivers(sender)
        ]

    def __repr__(self) -> str:
        return f"<Signal {self.name!r} ({len(self._receivers)} receivers)>"


def receiver(signal: Signal | list[Signal], sender: Any = None, **kwargs: Any):
    """Decorator form: ``@receiver(run_finished)``."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        for sig in signal if isinstance(signal, (list, tuple)) else [signal]:
            sig.connect(fn, sender=sender, weak=False, **kwargs)
        return fn

    return decorator


# --------------------------------------------------------------------------- #
# Framework signals
# --------------------------------------------------------------------------- #

apps_ready = Signal("apps_ready")

run_started = Signal("run_started", ("run",))
run_finished = Signal("run_finished", ("run", "status"))
run_failed = Signal("run_failed", ("run", "exception"))

epoch_started = Signal("epoch_started", ("run", "epoch"))
epoch_finished = Signal("epoch_finished", ("run", "epoch", "metrics"))
metric_logged = Signal("metric_logged", ("run", "key", "value", "step"))

dataset_materialized = Signal("dataset_materialized", ("dataset", "version"))
model_registered = Signal("model_registered", ("model", "version"))

agent_started = Signal("agent_started", ("agent", "trace"))
agent_step = Signal("agent_step", ("agent", "trace", "step"))
agent_finished = Signal("agent_finished", ("agent", "trace"))
tool_called = Signal("tool_called", ("agent", "tool", "arguments"))

pre_predict = Signal("pre_predict", ("model", "inputs"))
post_predict = Signal("post_predict", ("model", "inputs", "outputs"))

__all__ = [
    "Signal",
    "receiver",
    "apps_ready",
    "run_started",
    "run_finished",
    "run_failed",
    "epoch_started",
    "epoch_finished",
    "metric_logged",
    "dataset_materialized",
    "model_registered",
    "agent_started",
    "agent_step",
    "agent_finished",
    "tool_called",
    "pre_predict",
    "post_predict",
]
