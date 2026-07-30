"""Training callbacks — the extension point users are most likely to subclass."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from mlango.core.exceptions import ImproperlyConfigured
from mlango.training.callbacks import (
    Callback,
    CallbackList,
    Checkpoint,
    EarlyStopping,
    MetricThreshold,
    ProgressBar,
    build_callbacks,
)


class FakeRun:
    """Just the surface callbacks are allowed to touch."""

    def __init__(self):
        self.uuid = "0123456789abcdef"
        self.short_id = "01234567"
        self.should_stop = False
        self.tags: list[str] = []
        self.artifacts: list[dict] = []

    def add_tag(self, tag: str) -> None:
        self.tags.append(tag)

    def log_artifact(self, name, path, *, kind="", meta=None):
        self.artifacts.append({"name": name, "path": path, "kind": kind, "meta": meta or {}})


@pytest.fixture
def run():
    return FakeRun()


class TestCallbackList:
    def test_hooks_reach_every_callback_in_order(self, run):
        seen: list[str] = []

        class Recorder(Callback):
            def __init__(self, tag):
                self.tag = tag

            def on_train_begin(self, run, model, **kwargs):
                seen.append(self.tag)

        CallbackList([Recorder("a"), Recorder("b")]).emit("on_train_begin", run, None)
        assert seen == ["a", "b"]

    def test_one_failing_callback_does_not_stop_the_others(self, run, caplog):
        """A misbehaving progress bar must not kill a six-hour job."""
        seen: list[str] = []

        class Broken(Callback):
            def on_train_begin(self, run, model, **kwargs):
                raise RuntimeError("bad bar")

        class Fine(Callback):
            def on_train_begin(self, run, model, **kwargs):
                seen.append("ran")

        CallbackList([Broken(), Fine()]).emit("on_train_begin", run, None)

        assert seen == ["ran"]
        assert "bad bar" in caplog.text

    def test_an_unknown_hook_is_ignored(self, run):
        CallbackList([Callback()]).emit("on_something_else", run)

    def test_a_callback_missing_the_hook_is_skipped(self, run):
        CallbackList([SimpleNamespace()]).emit("on_train_begin", run, None)

    def test_it_behaves_like_a_list(self):
        callbacks = CallbackList()
        assert len(callbacks) == 0

        callbacks.append(Callback())
        assert len(callbacks) == 1
        assert [type(c) for c in callbacks] == [Callback]

    def test_the_base_class_hooks_are_all_no_ops(self, run):
        callback = Callback()
        callback.on_train_begin(run, None)
        callback.on_train_end(run, None)
        callback.on_epoch_begin(run, 0)
        callback.on_epoch_end(run, 0, {})
        callback.on_batch_end(run, 0, {})
        callback.on_evaluate_end(run, {})
        assert repr(callback) == "<Callback>"


class TestProgressBar:
    def test_it_prints_the_run_and_the_metrics(self, run, capsys):
        bar = ProgressBar()
        bar.on_train_begin(run, SimpleNamespace(_meta=SimpleNamespace(label="app.Model")))
        bar.on_epoch_begin(run, 0)
        bar.on_epoch_end(run, 0, {"loss": 0.5, "accuracy": 0.9})
        bar.on_train_end(run, None)

        out = capsys.readouterr().out
        assert "training app.Model" in out
        assert "loss=0.5000" in out
        assert "accuracy=0.9000" in out
        assert "done" in out

    def test_non_numeric_metrics_are_left_out(self, run, capsys):
        bar = ProgressBar()
        bar.on_epoch_begin(run, 0)
        bar.on_epoch_end(run, 0, {"loss": 0.5, "report": {"a": 1}})
        assert "report" not in capsys.readouterr().out

    def test_every_thins_the_output(self, run, capsys):
        bar = ProgressBar(every=3)
        bar.on_epoch_begin(run, 0)
        for epoch in range(6):
            bar.on_epoch_end(run, epoch, {"loss": 0.1})

        printed = [line for line in capsys.readouterr().out.splitlines() if "epoch" in line]
        assert len(printed) == 2  # epochs 0 and 3


class TestEarlyStopping:
    def test_a_bad_mode_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="mode must be"):
            EarlyStopping(mode="lowest")

    def test_it_stops_after_patience_is_exhausted(self, run):
        stopper = EarlyStopping("val_loss", patience=2)

        stopper.on_epoch_end(run, 0, {"val_loss": 1.0})
        assert run.should_stop is False

        stopper.on_epoch_end(run, 1, {"val_loss": 1.1})
        assert run.should_stop is False

        stopper.on_epoch_end(run, 2, {"val_loss": 1.2})
        assert run.should_stop is True
        assert "early-stopped" in run.tags

    def test_improvement_resets_the_counter(self, run):
        stopper = EarlyStopping("val_loss", patience=2)
        stopper.on_epoch_end(run, 0, {"val_loss": 1.0})
        stopper.on_epoch_end(run, 1, {"val_loss": 1.5})
        stopper.on_epoch_end(run, 2, {"val_loss": 0.5})

        assert stopper.waited == 0
        assert stopper.best == 0.5
        assert stopper.best_epoch == 2
        assert run.should_stop is False

    def test_max_mode_wants_the_metric_to_rise(self, run):
        stopper = EarlyStopping("accuracy", patience=1, mode="max")
        stopper.on_epoch_end(run, 0, {"accuracy": 0.8})
        assert stopper.best == 0.8

        stopper.on_epoch_end(run, 1, {"accuracy": 0.7})
        assert run.should_stop is True

    def test_min_delta_ignores_noise(self, run):
        """A change smaller than min_delta is not an improvement."""
        stopper = EarlyStopping("val_loss", patience=1, min_delta=0.1)
        stopper.on_epoch_end(run, 0, {"val_loss": 1.0})
        stopper.on_epoch_end(run, 1, {"val_loss": 0.95})

        assert stopper.best == 1.0
        assert run.should_stop is True

    def test_a_missing_metric_is_not_a_failure(self, run):
        """A backend that reports no validation loss must not be punished."""
        stopper = EarlyStopping("val_loss", patience=1)
        stopper.on_epoch_end(run, 0, {"accuracy": 0.9})
        stopper.on_epoch_end(run, 1, {"accuracy": 0.9})

        assert run.should_stop is False
        assert stopper.best == math.inf


class TestCheckpoint:
    @pytest.fixture
    def trainer(self, tmp_path):
        saved: list[str] = []

        class FakeTrainer:
            def save(self, model, fitted, name):
                saved.append(name)
                return str(tmp_path / f"{name.replace('/', '_')}.bin")

        return FakeTrainer(), saved

    def test_nothing_happens_without_a_trainer(self, run):
        Checkpoint().on_epoch_end(run, 0, {"val_loss": 0.5})
        assert run.artifacts == []

    def test_an_improvement_is_saved_as_an_artifact(self, run, trainer):
        fake, saved = trainer
        checkpoint = Checkpoint("val_loss")
        checkpoint.on_epoch_end(
            run, 1, {"val_loss": 0.5}, trainer=fake, fitted=object(), model=object()
        )

        assert saved == [f"runs/{run.uuid}/checkpoint-epoch1"]
        assert run.artifacts[0]["name"] == "checkpoint-epoch1"
        assert run.artifacts[0]["kind"] == "checkpoint"
        assert run.artifacts[0]["meta"] == {"epoch": 1, "val_loss": 0.5}
        assert checkpoint.best == 0.5
        assert checkpoint.best_path

    def test_a_worse_epoch_is_not_saved(self, run, trainer):
        fake, _saved = trainer
        checkpoint = Checkpoint("val_loss")
        checkpoint.on_epoch_end(run, 1, {"val_loss": 0.5}, trainer=fake, fitted=1, model=1)
        checkpoint.on_epoch_end(run, 2, {"val_loss": 0.9}, trainer=fake, fitted=1, model=1)

        assert len(run.artifacts) == 1
        assert checkpoint.best == 0.5

    def test_max_mode_keeps_the_highest(self, run, trainer):
        fake, _saved = trainer
        checkpoint = Checkpoint("accuracy", mode="max")
        checkpoint.on_epoch_end(run, 1, {"accuracy": 0.7}, trainer=fake, fitted=1, model=1)
        checkpoint.on_epoch_end(run, 2, {"accuracy": 0.9}, trainer=fake, fitted=1, model=1)

        assert checkpoint.best == 0.9
        assert len(run.artifacts) == 2

    def test_every_saves_on_a_schedule_regardless_of_the_metric(self, run, trainer):
        fake, _saved = trainer
        checkpoint = Checkpoint("val_loss", every=2)
        for epoch in range(1, 5):
            checkpoint.on_epoch_end(
                run, epoch, {"val_loss": 10.0 + epoch}, trainer=fake, fitted=1, model=1
            )

        # Epoch 1 is the first metric ever seen, so it improves on infinity;
        # 2 and 4 are then saved by the schedule while the metric only worsens.
        assert [a["meta"]["epoch"] for a in run.artifacts] == [1, 2, 4]

    def test_a_missing_metric_with_no_schedule_saves_nothing(self, run, trainer):
        fake, _saved = trainer
        Checkpoint("val_loss").on_epoch_end(run, 1, {}, trainer=fake, fitted=1, model=1)
        assert run.artifacts == []


class TestMetricThreshold:
    def test_a_metric_below_the_floor_fails_the_run(self, run):
        with pytest.raises(AssertionError, match="below the required 0.8"):
            MetricThreshold("accuracy", minimum=0.8).on_evaluate_end(run, {"accuracy": 0.5})

    def test_a_metric_above_the_ceiling_fails_the_run(self, run):
        with pytest.raises(AssertionError, match="above the allowed 0.2"):
            MetricThreshold("loss", maximum=0.2).on_evaluate_end(run, {"loss": 0.9})

    def test_a_metric_inside_the_band_passes(self, run):
        MetricThreshold("accuracy", minimum=0.5, maximum=1.0).on_evaluate_end(
            run, {"accuracy": 0.9}
        )

    def test_a_metric_that_was_not_reported_is_not_a_failure(self, run):
        MetricThreshold("accuracy", minimum=0.9).on_evaluate_end(run, {"f1": 0.5})

    def test_the_boundary_is_inclusive(self, run):
        MetricThreshold("accuracy", minimum=0.8).on_evaluate_end(run, {"accuracy": 0.8})


class TestBuildCallbacks:
    def test_the_default_stack_comes_from_settings(self, project):
        from mlango.conf import settings

        settings.DEFAULT_CALLBACKS = ["mlango.training.callbacks.ProgressBar"]
        built = build_callbacks()

        assert len(built) == 1
        assert isinstance(list(built)[0], ProgressBar)

    def test_extras_are_appended_after_the_defaults(self, project):
        from mlango.conf import settings

        settings.DEFAULT_CALLBACKS = ["mlango.training.callbacks.ProgressBar"]
        extra = EarlyStopping()
        built = build_callbacks([extra])

        assert [type(c) for c in built] == [ProgressBar, EarlyStopping]

    def test_an_empty_default_stack_is_allowed(self, project):
        """Metric recording is the framework's job, so this must stay safe."""
        from mlango.conf import settings

        settings.DEFAULT_CALLBACKS = []
        assert len(build_callbacks()) == 0

    def test_a_typo_in_the_stack_is_reported(self, project):
        from mlango.conf import settings

        settings.DEFAULT_CALLBACKS = ["mlango.training.callbacks.Nope"]
        with pytest.raises(ImproperlyConfigured):
            build_callbacks()
