"""``manage.py train`` — train a declared model."""

from __future__ import annotations

import json
from typing import Any

from mlango.core.typing import ModelClass
from mlango.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Train a declared model and register the resulting version."

    def add_arguments(self, parser) -> None:
        parser.add_argument("model", help="Model label, e.g. sentiment.Classifier.")
        parser.add_argument(
            "-p",
            "--param",
            action="append",
            default=[],
            metavar="NAME=VALUE",
            help="Override a hyperparameter. Repeatable.",
        )
        parser.add_argument("--dataset", help="Train on this dataset instead of Meta.dataset.")
        parser.add_argument("--name", default="", help="Name for the run.")
        parser.add_argument("--notes", default="", help="Free-text notes for the run.")
        parser.add_argument("--tag", action="append", default=[], help="Tag the run. Repeatable.")
        parser.add_argument("--seed", type=int, help="Override the seed for this run.")
        parser.add_argument(
            "--materialize",
            action="store_true",
            help="Freeze the training view into a dataset version first.",
        )
        parser.add_argument(
            "--no-register",
            action="store_true",
            help="Train without registering a model version.",
        )

    def handle(self, **options: Any) -> None:
        from mlango.core.registry import apps

        model_class = apps.get_model(options["model"])
        overrides = _parse_params(model_class, options["param"])

        dataset = None
        if options.get("dataset"):
            dataset = apps.get_dataset(options["dataset"])

        model = model_class(**overrides)
        self.write(self.style.bold(f"Training {model_class._meta.label}"))
        for key, value in model.params.items():
            self.write(self.style.dim(f"  {key} = {value}"))

        run = model.train(
            dataset=dataset,
            name=options["name"],
            notes=options["notes"],
            tags=options["tag"] or None,
            seed=options.get("seed"),
            materialize=options["materialize"],
            register=not options["no_register"],
        )

        record = run.refresh()
        self.write("")
        self.write(self.style.bold("Result"))
        self.write(f"  run    {run.uuid}")
        self.write(f"  status {record.status if record else 'unknown'}")

        if record and record.summary:
            self.write("")
            self.table(
                ["metric", "value"],
                [
                    [key, f"{value:.4f}" if isinstance(value, float) else value]
                    for key, value in record.summary.items()
                ],
            )

        if model._version is not None:
            self.write("")
            self.ok(f"Registered {model._version.ref}")


def _parse_params(model_class: ModelClass, entries: list[str]) -> dict[str, Any]:
    """Turn ``name=value`` strings into cleaned hyperparameters."""
    out: dict[str, Any] = {}
    for entry in entries:
        if "=" not in entry:
            raise CommandError(f"--param expects NAME=VALUE, got {entry!r}.")
        name, _, raw = entry.partition("=")
        name = name.strip()
        if not model_class._meta.has_field(name):
            available = ", ".join(model_class._meta.field_names) or "(none)"
            raise CommandError(
                f"{model_class._meta.label} has no hyperparameter {name!r}. Available: {available}."
            )
        field = model_class._meta.get_field(name)
        try:
            out[name] = field.clean(_coerce(raw.strip()))
        except Exception as exc:
            raise CommandError(f"--param {name}: {exc}") from exc
    return out


def _coerce(raw: str) -> Any:
    """Best-effort literal parsing so ``--param C=2.0`` and ``--param x=[1,2]`` both work."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw
