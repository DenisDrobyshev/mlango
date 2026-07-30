"""scikit-learn backend.

Fits in one shot, so the "epoch" loop is a single pass. That still goes through
the same callback and metric machinery as a torch run, which is the point: the
admin, the CLI and the run comparison view do not care which backend produced a
number.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mlango.core.exceptions import RunError
from mlango.core.signals import epoch_finished, epoch_started
from mlango.training import metrics as metric_lib
from mlango.training.trainer import Trainer


def _as_matrix(inputs: list[Any], features: list[str] | None = None) -> Any:
    """Normalise records into something an estimator accepts.

    This mirrors ``DataQuerySet.xy()`` exactly, and it has to: training feeds the
    estimator through xy() while a served request arrives as a dict, so any
    disagreement between the two shapes means the deployed model scores
    different columns than the fitted one. One input field is passed through
    untouched, so a text pipeline still receives a list of strings; several
    become a dense 2-D array in the declared field order.
    """
    if not inputs:
        return []
    first = inputs[0]
    if not isinstance(first, dict):
        return list(inputs)

    # Prefer the declared order over sorted(): sorting a request's own keys makes
    # the column order depend on the payload, and a missing or extra key would
    # silently shift every column.
    columns = list(features) if features else sorted(first)
    if len(columns) == 1:
        return [record.get(columns[0]) for record in inputs]
    return np.asarray([[record.get(column) for column in columns] for record in inputs])


def _features_of(model) -> list[str] | None:
    """The model's declared features, or None if the declaration cannot say.

    Inference must not fail on introspection: a loaded model whose dataset is no
    longer importable can still predict from the payload's own keys.
    """
    from mlango.core.exceptions import MlangoError

    try:
        return model.get_features()
    except (MlangoError, AttributeError, LookupError):
        return None


class SklearnTrainer(Trainer):
    name = "sklearn"
    requires = ("sklearn", "joblib")
    extension = "joblib"

    # -- training ------------------------------------------------------------

    def fit(
        self,
        model,
        train,
        validation,
        run,
        callbacks,
        *,
        target: str = "",
        features: list[str] | None = None,
        **kwargs: Any,
    ):
        estimator = model.build()
        if not hasattr(estimator, "fit"):
            raise RunError(
                f"{type(model)._meta.label}.build() returned {type(estimator).__name__}, which has "
                f"no fit() method. The sklearn trainer expects an estimator or a Pipeline."
            )

        features = features or model.get_features()
        x_train, y_train = train.xy(target=target or None, features=features)
        if not x_train:
            raise RunError("The training split is empty; check the dataset source and filters.")

        epoch_started.send(sender=type(model), run=run, epoch=0)
        callbacks.emit("on_epoch_begin", run, 0, model=model)

        estimator.fit(_as_matrix(x_train, features), y_train)

        epoch_metrics = self._epoch_metrics(model, estimator, train, validation, target, features)
        # Recording metrics is the framework's job, not a callback's: a user who
        # customises DEFAULT_CALLBACKS must not silently lose metric history.
        run.log_metrics(epoch_metrics, epoch=0, step=0)
        callbacks.emit(
            "on_epoch_end",
            run,
            0,
            epoch_metrics,
            model=model,
            trainer=self,
            fitted=estimator,
        )
        epoch_finished.send(sender=type(model), run=run, epoch=0, metrics=epoch_metrics)
        return estimator

    def _epoch_metrics(
        self, model, estimator, train, validation, target: str, features: list[str]
    ) -> dict[str, float]:
        task = model.get_task()
        out: dict[str, float] = {}

        x_train, y_train = train.xy(target=target or None, features=features)
        train_report = metric_lib.report_for_task(
            task, y_train, estimator.predict(_as_matrix(x_train, features))
        )
        out.update({f"train_{k}": v for k, v in metric_lib.flatten_report(train_report).items()})

        if validation is not None:
            x_val, y_val = validation.xy(target=target or None, features=features)
            if x_val:
                val_report = metric_lib.report_for_task(
                    task, y_val, estimator.predict(_as_matrix(x_val, features))
                )
                flattened = metric_lib.flatten_report(val_report)
                out.update({f"val_{k}": v for k, v in flattened.items()})
                # Give EarlyStopping something to monitor for regression too.
                if "val_mse" in out:
                    out["val_loss"] = out["val_mse"]
                out.update(
                    {k: v for k, v in flattened.items() if k in {"accuracy", "f1_macro", "r2"}}
                )
        return out

    # -- inference -----------------------------------------------------------

    def predict(self, model, fitted, inputs: list[Any]) -> list[Any]:
        predictions = fitted.predict(_as_matrix(inputs, _features_of(model)))
        return [_python(value) for value in predictions]

    def predict_proba(self, model, fitted, inputs: list[Any]):
        if not hasattr(fitted, "predict_proba"):
            return None
        classes = [_python(c) for c in getattr(fitted, "classes_", [])]
        probabilities = fitted.predict_proba(_as_matrix(inputs, _features_of(model)))
        return [
            dict(zip(classes, (float(p) for p in row), strict=True))
            if classes
            else [float(p) for p in row]
            for row in probabilities
        ]

    # -- persistence ---------------------------------------------------------

    def save(self, model, fitted, name: str) -> str:
        import joblib

        from mlango.storage import default_storage

        path = default_storage().path(f"{name}.{self.extension}")
        joblib.dump(fitted, path)
        return path

    def load(self, model, path: str):
        import joblib

        return joblib.load(path)

    # -- introspection -------------------------------------------------------

    def describe(self, model, fitted) -> dict[str, Any]:
        info: dict[str, Any] = {"backend": self.name, "estimator": type(fitted).__name__}
        steps = getattr(fitted, "steps", None)
        if steps:
            info["pipeline"] = [name for name, _ in steps]
        classes = getattr(fitted, "classes_", None)
        if classes is not None:
            info["n_classes"] = int(len(classes))
        return info


def _python(value: Any) -> Any:
    """Convert numpy scalars so JSON columns and API responses stay clean."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            return value
    return value
