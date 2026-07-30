"""Training callbacks — the middleware of the training loop.

Same idea as Django middleware: a stack of small objects with hooks, configured
in settings or per model, that observe and can influence the request — here the
run — without any of them knowing about the others.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

logger = logging.getLogger("mlango.callbacks")


class Callback:
    """Override the hooks you care about; the rest are no-ops."""

    def on_train_begin(self, run: Any, model: Any, **kwargs: Any) -> None: ...

    def on_train_end(self, run: Any, model: Any, **kwargs: Any) -> None: ...

    def on_epoch_begin(self, run: Any, epoch: int, **kwargs: Any) -> None: ...

    def on_epoch_end(
        self, run: Any, epoch: int, metrics: dict[str, float], **kwargs: Any
    ) -> None: ...

    def on_batch_end(
        self, run: Any, step: int, metrics: dict[str, float], **kwargs: Any
    ) -> None: ...

    def on_evaluate_end(self, run: Any, metrics: dict[str, Any], **kwargs: Any) -> None: ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


class CallbackList:
    """Fans hooks out to every callback, isolating failures."""

    def __init__(self, callbacks: list[Callback] | None = None):
        self.callbacks = list(callbacks or [])

    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def __iter__(self):
        return iter(self.callbacks)

    def __len__(self) -> int:
        return len(self.callbacks)

    def emit(self, hook: str, *args: Any, **kwargs: Any) -> None:
        for callback in self.callbacks:
            method = getattr(callback, hook, None)
            if method is None:
                continue
            try:
                method(*args, **kwargs)
            except Exception:
                # A misbehaving progress bar must not kill a six-hour job.
                logger.exception("Callback %r failed in %s", callback, hook)


class ProgressBar(Callback):
    """Plain-text progress on stdout — readable in CI logs, unlike a TUI bar."""

    def __init__(self, every: int = 1):
        self.every = every
        self._epoch_started = 0.0

    def on_train_begin(self, run, model, **kwargs):
        print(f"[{run.short_id}] training {model._meta.label}")

    def on_epoch_begin(self, run, epoch, **kwargs):
        self._epoch_started = time.perf_counter()

    def on_epoch_end(self, run, epoch, metrics, **kwargs):
        if epoch % self.every:
            return
        elapsed = time.perf_counter() - self._epoch_started
        body = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float)))
        print(f"[{run.short_id}] epoch {epoch:>3}  {body}  ({elapsed:.1f}s)")

    def on_train_end(self, run, model, **kwargs):
        print(f"[{run.short_id}] done")


class EarlyStopping(Callback):
    """Stop when the monitored metric stops improving.

    Sets ``run.should_stop``; backends check that flag between epochs, so the
    decision lives here rather than being reimplemented in each backend.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        *,
        patience: int = 3,
        mode: str = "min",
        min_delta: float = 0.0,
    ):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = math.inf if mode == "min" else -math.inf
        self.best_epoch = 0
        self.waited = 0

    def _improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def on_epoch_end(self, run, epoch, metrics, **kwargs):
        value = metrics.get(self.monitor)
        if value is None:
            return
        if self._improved(float(value)):
            self.best = float(value)
            self.best_epoch = epoch
            self.waited = 0
            return
        self.waited += 1
        if self.waited >= self.patience:
            run.should_stop = True
            logger.info(
                "Early stopping at epoch %s: %s has not improved on %.4f for %s epochs.",
                epoch,
                self.monitor,
                self.best,
                self.patience,
            )
            run.add_tag("early-stopped")


class Checkpoint(Callback):
    """Persist the best model seen so far, as an artifact of the run."""

    def __init__(self, monitor: str = "val_loss", *, mode: str = "min", every: int | None = None):
        self.monitor = monitor
        self.mode = mode
        self.every = every
        self.best = math.inf if mode == "min" else -math.inf
        self.best_path: str | None = None

    def _improved(self, value: float) -> bool:
        return value < self.best if self.mode == "min" else value > self.best

    def on_epoch_end(self, run, epoch, metrics, **kwargs):
        trainer = kwargs.get("trainer")
        fitted = kwargs.get("fitted")
        model = kwargs.get("model")
        if trainer is None or fitted is None or model is None:
            return

        due_periodically = self.every is not None and epoch % self.every == 0
        value = metrics.get(self.monitor)
        due_by_metric = value is not None and self._improved(float(value))
        if not (due_periodically or due_by_metric):
            return
        if value is not None and due_by_metric:
            self.best = float(value)

        name = f"checkpoint-epoch{epoch}"
        path = trainer.save(model, fitted, f"runs/{run.uuid}/{name}")
        self.best_path = path
        run.log_artifact(name, path, kind="checkpoint", meta={"epoch": epoch, self.monitor: value})


class MetricThreshold(Callback):
    """Fail the run when a metric crosses a line — a guardrail for CI."""

    def __init__(self, metric: str, *, minimum: float | None = None, maximum: float | None = None):
        self.metric = metric
        self.minimum = minimum
        self.maximum = maximum

    def on_evaluate_end(self, run, metrics, **kwargs):
        value = metrics.get(self.metric)
        if value is None:
            return
        if self.minimum is not None and value < self.minimum:
            raise AssertionError(f"{self.metric}={value:.4f} is below the required {self.minimum}.")
        if self.maximum is not None and value > self.maximum:
            raise AssertionError(f"{self.metric}={value:.4f} is above the allowed {self.maximum}.")


def build_callbacks(extra: list[Callback] | None = None) -> CallbackList:
    """Settings defaults plus whatever the caller passed."""
    from mlango.conf import settings
    from mlango.core.module_loading import import_string

    callbacks: list[Callback] = []
    for path in settings.DEFAULT_CALLBACKS:
        callbacks.append(import_string(path)())
    callbacks.extend(extra or [])
    return CallbackList(callbacks)


__all__ = [
    "Callback",
    "CallbackList",
    "ProgressBar",
    "EarlyStopping",
    "Checkpoint",
    "MetricThreshold",
    "build_callbacks",
]
