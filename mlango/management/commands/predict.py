"""``manage.py predict`` — score data with a registered model version.

Training from the terminal was already possible; scoring was not, unless you
started a server. This closes that gap: point a trained model at its own dataset,
a file, or a value on the command line.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any

from mlango.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Predict with a trained model: from its dataset, a file, or a literal input."

    def add_arguments(self, parser) -> None:
        parser.add_argument("model", help="Model label, e.g. reviews.Sentiment.")
        parser.add_argument(
            "input",
            nargs="*",
            help="Values to score. Omit to read from --file or the model's own dataset.",
        )
        parser.add_argument("--file", help="Score every record in this file (csv/jsonl/json).")
        parser.add_argument(
            "--dataset",
            action="store_true",
            help="Score the model's declared dataset instead of a file.",
        )
        parser.add_argument(
            "--filter",
            action="append",
            default=[],
            metavar="FIELD=VALUE",
            help="Narrow the dataset, e.g. --filter label=pos. Repeatable.",
        )
        parser.add_argument("-n", "--limit", type=int, help="Stop after this many records.")
        parser.add_argument("--version", type=int, help="Model version to load (default: latest).")
        parser.add_argument("--stage", help="Load the version at this stage, e.g. production.")
        parser.add_argument("--proba", action="store_true", help="Include class probabilities.")
        parser.add_argument(
            "--format",
            choices=["table", "jsonl", "csv"],
            default="table",
            help="Output format (default table).",
        )
        parser.add_argument("--output", help="Write to this file instead of stdout.")

    def handle(self, **options: Any) -> None:
        from mlango.core.registry import apps

        model_class = apps.get_model(options["model"])
        model = self._load(model_class, options)

        inputs, context = self._collect(model, model_class, options)
        predictions = model.predict(inputs)
        probabilities = model.predict_proba(inputs) if options["proba"] else None
        if options["proba"] and probabilities is None:
            self.warn(f"{model_class._meta.label} cannot produce probabilities; skipping --proba.")

        rows = self._rows(inputs, predictions, probabilities, context)
        self._emit(rows, options)

    # -- loading -------------------------------------------------------------

    def _load(self, model_class: Any, options: dict[str, Any]) -> Any:
        from mlango.core.exceptions import MlangoError

        try:
            model = model_class.load(version=options.get("version"), stage=options.get("stage"))
        except LookupError as exc:
            raise CommandError(
                f"{exc} Train it first: manage.py train {model_class._meta.label}"
            ) from exc
        except MlangoError as exc:
            raise CommandError(str(exc)) from exc

        version = getattr(model, "_version", None)
        if version is not None and self.verbosity >= 1:
            from mlango.metastore.models import Stage

            # Only mention the stage when it says something; every version starts
            # at "none", and printing that on every call is noise.
            current = getattr(version, "stage", "") or ""
            stage = f" stage={current}" if current and current != Stage.NONE else ""
            self.write(
                self.style.dim(f"Loaded {model_class._meta.label}@v{version.version}{stage}")
            )
        return model

    # -- input ---------------------------------------------------------------

    def _collect(
        self, model: Any, model_class: Any, options: dict[str, Any]
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """The inputs to score, plus the record each came from for the output."""
        if options["input"]:
            if options["file"] or options["dataset"]:
                raise CommandError("Pass either literal values, --file or --dataset — not both.")
            return list(options["input"]), [{} for _ in options["input"]]

        if options["file"]:
            if options["dataset"]:
                raise CommandError("Pass either --file or --dataset, not both.")
            records = self._from_file(options["file"], options.get("limit"))
        elif options["dataset"]:
            records = self._from_dataset(model_class, options)
        else:
            raise CommandError(
                "Nothing to score. Pass values, --file, or --dataset to use the model's own data."
            )

        features = model_class.get_features()
        self._check_features(model_class, features, records[0])
        inputs = [_as_input(record, features) for record in records]
        return inputs, records

    def _check_features(
        self, model_class: Any, features: list[str], sample: dict[str, Any]
    ) -> None:
        """Refuse data the model cannot read, before the backend sees it.

        Handing a missing feature through as None fails deep inside the trainer —
        `'NoneType' object has no attribute 'lower'` from a vectoriser tells you
        nothing about the column that was actually absent.
        """
        missing = [name for name in features if name not in sample]
        if not missing:
            return

        present = ", ".join(sorted(sample)) or "(no columns)"
        raise CommandError(
            f"{model_class._meta.label} needs {', '.join(missing)}, which the data does not "
            f"have. Columns found: {present}.\n"
            f"Rename them to match, or map them first with a queryset: "
            f"Dataset.objects.rename({missing[0]}=<source column>)."
        )

    def _from_file(self, path: str, limit: int | None) -> list[dict[str, Any]]:
        from mlango.conf import settings
        from mlango.core.exceptions import ImproperlyConfigured
        from mlango.data.inspect import source_for

        resolved = path if os.path.isabs(path) else os.path.join(str(settings.BASE_DIR), path)
        if not os.path.exists(resolved):
            raise CommandError(f"No such file: {resolved}")

        try:
            source, _expr = source_for(path)
        except ImproperlyConfigured as exc:
            raise CommandError(str(exc)) from exc

        records = []
        for record in source:
            records.append(record)
            if limit and len(records) >= limit:
                break
        if not records:
            raise CommandError(f"{resolved} contained no records.")
        return records

    def _from_dataset(self, model_class: Any, options: dict[str, Any]) -> list[dict[str, Any]]:
        from mlango.core.exceptions import FieldError

        dataset = model_class.get_dataset()
        query = dataset.objects.get_queryset()

        for raw in options["filter"]:
            if "=" not in raw:
                raise CommandError(f"--filter expects FIELD=VALUE, got {raw!r}.")
            key, _, value = raw.partition("=")
            try:
                query = query.filter(**{key.strip(): _coerce(value.strip())})
            except FieldError as exc:
                raise CommandError(str(exc)) from exc

        if options.get("limit"):
            query = query.take(options["limit"])

        records = [dict(record) for record in query]
        if not records:
            # Naming the filters matters: the usual cause is a value that does
            # not occur, and the generic "nothing to score" hides that entirely.
            applied = ", ".join(options["filter"]) or "(none)"
            raise CommandError(
                f"No records in {dataset._meta.label} matched. Filters applied: {applied}. "
                f"Check the values with: manage.py dataset head {dataset._meta.label}"
            )
        return records

    # -- output --------------------------------------------------------------

    def _rows(
        self,
        inputs: list[Any],
        predictions: list[Any],
        probabilities: list[Any] | None,
        context: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows = []
        for index, (value, prediction) in enumerate(zip(inputs, predictions, strict=True)):
            row: dict[str, Any] = {"input": value, "prediction": prediction}
            if probabilities is not None:
                row["probabilities"] = probabilities[index]
            record = context[index] if index < len(context) else {}
            # Carry the key through so a scored file can be joined back to its source.
            for key in ("id", "uuid", "pk"):
                if key in record:
                    row = {key: record[key], **row}
                    break
            rows.append(row)
        return rows

    def _emit(self, rows: list[dict[str, Any]], options: dict[str, Any]) -> None:
        handle = sys.stdout
        opened = None
        if options["output"]:
            from mlango.conf import settings

            path = options["output"]
            resolved = path if os.path.isabs(path) else os.path.join(str(settings.BASE_DIR), path)
            opened = open(resolved, "w", encoding="utf-8", newline="")
            handle = opened

        try:
            if options["format"] == "jsonl":
                for row in rows:
                    handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
            elif options["format"] == "csv":
                columns = list(rows[0])
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: _flat(v) for k, v in row.items()})
            elif opened is not None:
                # A table is for reading on a terminal; a file wants data.
                for row in rows:
                    handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
            else:
                self._table(rows)
        finally:
            if opened is not None:
                opened.close()

        if opened is not None:
            self.ok(f"Wrote {len(rows)} prediction(s) to {options['output']}")
        elif options["format"] == "table":
            self.write("")
            self.write(self.style.dim(f"{len(rows)} prediction(s)."))

    def _table(self, rows: list[dict[str, Any]]) -> None:
        columns = list(rows[0])
        self.table(columns, [[_flat(row.get(c)) for c in columns] for row in rows])


def _as_input(record: dict[str, Any], features: list[str]) -> Any:
    """Shape one record the way the trainer expects it.

    One feature is passed as a raw value and several as a dict — matching
    QuerySet.xy(), so a prediction here means the same thing as one made during
    training.
    """
    if len(features) == 1:
        return record.get(features[0])
    return {name: record.get(name) for name in features}


def _coerce(value: str) -> Any:
    """Turn a command-line filter value into something comparable."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def _flat(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(_jsonable(value), ensure_ascii=False)
    text = "" if value is None else str(value)
    return text if len(text) <= 60 else text[:59] + "…"


def _jsonable(value: Any) -> Any:
    from mlango.core.serialization import jsonable

    return jsonable(value)
