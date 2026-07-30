"""Metric functions with no dependency beyond numpy.

Keeping these self-contained means ``mlango`` reports the same accuracy whether
or not scikit-learn is installed, and a torch-only project never pulls in a
second ML stack just to compute an F1.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _as_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(list(values))


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    true, pred = _as_array(y_true), _as_array(y_pred)
    if true.size == 0:
        return 0.0
    return float((true == pred).mean())


def confusion(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, dict[str, int]]:
    labels = sorted({*map(str, y_true), *map(str, y_pred)})
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for actual, predicted in zip(y_true, y_pred):
        matrix[str(actual)][str(predicted)] += 1
    return matrix


def per_class(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, dict[str, float]]:
    """Precision, recall, F1 and support for every class."""
    labels = sorted({*map(str, y_true), *map(str, y_pred)})
    true = [str(v) for v in y_true]
    pred = [str(v) for v in y_pred]
    out: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(true, pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(true, pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(true, pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(sum(1 for t in true if t == label)),
        }
    return out


def _average(scores: dict[str, dict[str, float]], key: str, average: str) -> float:
    if not scores:
        return 0.0
    if average == "macro":
        return float(np.mean([v[key] for v in scores.values()]))
    if average == "weighted":
        total = sum(v["support"] for v in scores.values())
        if not total:
            return 0.0
        return float(sum(v[key] * v["support"] for v in scores.values()) / total)
    raise ValueError(f"Unknown average {average!r}; use 'macro' or 'weighted'.")


def precision(y_true, y_pred, *, average: str = "macro") -> float:
    return _average(per_class(y_true, y_pred), "precision", average)


def recall(y_true, y_pred, *, average: str = "macro") -> float:
    return _average(per_class(y_true, y_pred), "recall", average)


def f1(y_true, y_pred, *, average: str = "macro") -> float:
    return _average(per_class(y_true, y_pred), "f1", average)


def classification_report(y_true, y_pred) -> dict[str, Any]:
    scores = per_class(y_true, y_pred)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "f1_macro": _average(scores, "f1", "macro"),
        "f1_weighted": _average(scores, "f1", "weighted"),
        "precision_macro": _average(scores, "precision", "macro"),
        "recall_macro": _average(scores, "recall", "macro"),
        "per_class": scores,
        "confusion": confusion(y_true, y_pred),
        "support": len(list(y_true)),
    }


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    true, pred = _as_array(y_true).astype(float), _as_array(y_pred).astype(float)
    return float(np.mean((true - pred) ** 2)) if true.size else 0.0


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    true, pred = _as_array(y_true).astype(float), _as_array(y_pred).astype(float)
    return float(np.mean(np.abs(true - pred))) if true.size else 0.0


def r2(y_true, y_pred) -> float:
    true, pred = _as_array(y_true).astype(float), _as_array(y_pred).astype(float)
    if true.size == 0:
        return 0.0
    residual = float(np.sum((true - pred) ** 2))
    total = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - residual / total if total else 0.0


def regression_report(y_true, y_pred) -> dict[str, Any]:
    return {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "support": len(list(y_true)),
    }


#: Looked up by name from ``Model.Meta.metrics``.
REGISTRY = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
}


def report_for_task(task: str, y_true, y_pred) -> dict[str, Any]:
    """The conventional metric bundle for a task type."""
    if task == "regression":
        return regression_report(y_true, y_pred)
    return classification_report(y_true, y_pred)


def flatten_report(report: dict[str, Any]) -> dict[str, float]:
    """Scalar-only view of a report, suitable for the metrics table."""
    return {k: float(v) for k, v in report.items() if isinstance(v, (int, float))}


__all__ = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion",
    "per_class",
    "classification_report",
    "mse",
    "rmse",
    "mae",
    "r2",
    "regression_report",
    "report_for_task",
    "flatten_report",
    "REGISTRY",
]
