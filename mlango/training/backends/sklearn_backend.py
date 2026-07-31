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

#: A text pipeline has tens of thousands of features and nobody reads past the
#: first page. Storing the whole vector would put a megabyte of noise in every
#: version row for a list that is only ever skimmed.
TOP_FEATURES = 40


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


def _declared_classes(model) -> list[Any] | None:
    """The target's declared classes, in the order they were declared.

    ``LabelField.classes`` says it "doubles as the class ordering used when
    encoding to indices, so training and serving agree on what index 0 means".
    This is what makes that true.
    """
    try:
        dataset_class = model.get_dataset()
        field = dataset_class._meta.get_field(model.get_target(dataset_class))
    except Exception:  # noqa: BLE001 - an incomplete declaration must still train
        return None
    classes = getattr(field, "classes", None)
    return list(classes) if classes else None


def _encode(values: list[Any], classes: list[Any]) -> list[Any]:
    """Class values to indices, leaving anything unrecognised alone.

    scikit-learn's own estimators accept string labels, but XGBoost, LightGBM
    and CatBoost reject them with a message about "invalid classes inferred
    from unique values of y" that mentions neither mlango nor what to do about
    it. Encoding here makes every sklearn-compatible estimator work with a
    declared LabelField, which is the whole point of declaring one.
    """
    lookup = {value: index for index, value in enumerate(classes)}
    return [lookup.get(value, value) for value in values]


def _decode(values: Any, classes: list[Any]) -> list[Any]:
    """Indices back to the declared class values."""
    out = []
    for value in values:
        index = _python(value)
        if isinstance(index, bool) or not isinstance(index, int):
            out.append(index)
        elif 0 <= index < len(classes):
            out.append(classes[index])
        else:
            out.append(index)
    return out


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

        classes = _declared_classes(model)
        fit_targets = _encode(y_train, classes) if classes else y_train
        estimator.fit(_as_matrix(x_train, features), fit_targets)

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
        classes = _declared_classes(model)

        def scored(query) -> Any:
            """Predictions in the same vocabulary as the truth they are scored against."""
            x, y = query.xy(target=target or None, features=features)
            if not x:
                return None, None
            raw = estimator.predict(_as_matrix(x, features))
            return y, (_decode(raw, classes) if classes else [_python(v) for v in raw])

        y_train, train_predictions = scored(train)
        train_report = metric_lib.report_for_task(task, y_train, train_predictions)
        out.update({f"train_{k}": v for k, v in metric_lib.flatten_report(train_report).items()})

        if validation is not None:
            y_val, val_predictions = scored(validation)
            if y_val is not None:
                val_report = metric_lib.report_for_task(task, y_val, val_predictions)
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
        declared = _declared_classes(model)
        predictions = fitted.predict(_as_matrix(inputs, _features_of(model)))
        if declared:
            return _decode(predictions, declared)
        return [_python(value) for value in predictions]

    def predict_proba(self, model, fitted, inputs: list[Any]):
        if not hasattr(fitted, "predict_proba"):
            return None
        declared = _declared_classes(model)
        fitted_classes = [_python(c) for c in getattr(fitted, "classes_", [])]
        # The columns are in the estimator's own class order, which is the index
        # order when the target was encoded. Name them from the declaration.
        classes = _decode(fitted_classes, declared) if declared else fitted_classes
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

        with default_storage().writable(f"{name}.{self.extension}") as target:
            joblib.dump(fitted, target.path)
            return target.name

    def load(self, model, path: str):
        import joblib

        from mlango.storage import default_storage

        with default_storage().readable(path) as local:
            return joblib.load(local)

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

    def importances(self, model, fitted) -> dict[str, float] | None:
        """Whatever the fitted estimator exposes, paired with real column names.

        Trees publish ``feature_importances_`` and linear models publish
        ``coef_``; both are anonymous vectors. The names come from the last
        transformer that can produce them — which is what turns a
        ``TfidfVectorizer`` pipeline from 40,000 numbered slots into the words
        the model actually keyed on — and from the declared fields otherwise.
        """
        estimator, names = _explainable(fitted)
        weights = getattr(estimator, "feature_importances_", None)
        if weights is None:
            coefficients = getattr(estimator, "coef_", None)
            if coefficients is None:
                return None
            matrix = np.asarray(coefficients, dtype=float)
            if matrix.ndim > 1 and matrix.shape[0] > 1:
                # One row per class. There is no signed answer to give — a
                # feature that argues for one class argues against another — so
                # report magnitude, which answers "does this matter at all".
                weights = np.abs(matrix).mean(axis=0)
            else:
                # Binary or regression: one row, and its sign is the direction
                # of the effect. Keeping it is the difference between "this word
                # matters" and "this word means negative".
                weights = matrix.ravel()

        values = np.asarray(weights, dtype=float).ravel()
        if not values.size:
            return None
        if names is None or len(names) != values.size:
            declared = _features_of(model) or []
            names = (
                list(declared)
                if len(declared) == values.size
                else [f"feature_{i}" for i in range(values.size)]
            )

        ranked = sorted(zip(names, values, strict=True), key=lambda pair: -abs(pair[1]))
        return {str(name): float(value) for name, value in ranked[:TOP_FEATURES]}


def _explainable(fitted: Any) -> tuple[Any, list[str] | None]:
    """The estimator that holds the weights, and names for its columns."""
    steps = getattr(fitted, "steps", None)
    if not steps:
        return fitted, None

    estimator = steps[-1][1]
    for _, step in reversed(steps[:-1]):
        get_names = getattr(step, "get_feature_names_out", None)
        if not callable(get_names):
            continue
        try:
            return estimator, [str(name) for name in get_names()]
        except Exception:  # noqa: BLE001 - unfitted or nameless, try the next one
            continue
    return estimator, None


def _python(value: Any) -> Any:
    """Convert numpy scalars so JSON columns and API responses stay clean."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            return value
    return value
