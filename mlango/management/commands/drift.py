"""``manage.py drift`` — has the input moved away from what the model was fitted on?

Accuracy in production needs labels, and labels arrive late or never. The input
distribution is available on the first request, and it moving is the earliest
honest warning that a model is being asked questions it was not trained for.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any

from mlango.management.base import BaseCommand, CommandError

_WINDOW = re.compile(r"^(\d+)([hdw])$")
_UNITS = {"h": "hours", "d": "days", "w": "weeks"}


class Command(BaseCommand):
    help = "Compare logged predictions against the data a model version was trained on."

    def add_arguments(self, parser) -> None:
        parser.add_argument("model", help="Model label, e.g. reviews.Sentiment.")
        parser.add_argument("--version", type=int, help="Baseline version (default: latest).")
        parser.add_argument("--stage", help="Use the version at this stage, e.g. production.")
        parser.add_argument(
            "--since", default="7d", help="Window of logged predictions: 24h, 7d, 4w. Default 7d."
        )
        parser.add_argument(
            "--against",
            help="Compare against a dataset label instead of the prediction log.",
        )
        parser.add_argument("-n", "--limit", type=int, default=10_000, help="Cap the rows read.")
        parser.add_argument("--json", action="store_true", help="Emit the scores as JSON.")
        parser.add_argument(
            "--fail-on",
            choices=["moderate", "significant"],
            help="Exit non-zero when any column reaches this verdict. For CI.",
        )

    def handle(self, **options: Any) -> None:
        from mlango.core.registry import apps
        from mlango.training import drift

        model_class = apps.get_model(options["model"])
        version = self._version(model_class, options)

        baseline = version.baseline or {}
        if not baseline:
            raise CommandError(
                f"{version.ref} has no training profile, so there is nothing to compare "
                f"against. Versions trained before drift detection existed do not have "
                f"one; retrain, or point --version at a newer one."
            )

        records, outputs, source, window = self._observed(model_class, version, options)
        if not records:
            raise CommandError(
                f"No {source} to compare. "
                + (
                    "Turn the log on with PREDICTION_LOG = {'ENABLED': True} and let it "
                    "collect some traffic."
                    if source == "logged predictions"
                    else "The dataset returned no rows."
                )
            )

        features = model_class.get_features()
        target = model_class.get_target()

        scores = drift.compare({k: v for k, v in baseline.items() if k in features}, records)
        # The prediction distribution is the other half of the question, and the
        # only one available when a model takes free text: labels that used to
        # split evenly and now come back 90% one class say something is wrong
        # long before anyone has ground truth to prove it.
        if outputs and target in baseline:
            predicted = drift.compare({target: baseline[target]}, outputs)
            if target in predicted:
                scores[f"{target} (predicted)"] = predicted[target]

        if not scores:
            raise CommandError(
                f"None of the baseline's columns ({', '.join(baseline)}) appear in the "
                f"{source}. The comparison would be between two different things."
            )

        if options["json"]:
            self.write(json.dumps(scores, ensure_ascii=False, indent=2))
        else:
            self._report(version, scores, len(records), source, window)

        self._maybe_fail(scores, options.get("fail_on"))

    # -- inputs --------------------------------------------------------------

    def _version(self, model_class: Any, options: dict[str, Any]) -> Any:
        from sqlalchemy import select

        from mlango.metastore.models import ModelVersion
        from mlango.metastore.session import session_scope

        label = model_class._meta.label
        statement = select(ModelVersion).where(ModelVersion.label == label)
        if options.get("version"):
            statement = statement.where(ModelVersion.version == options["version"])
        if options.get("stage"):
            statement = statement.where(ModelVersion.stage == options["stage"])

        with session_scope() as session:
            found = session.execute(
                statement.order_by(ModelVersion.version.desc()).limit(1)
            ).scalar_one_or_none()

        if found is None:
            raise CommandError(
                f"No registered version of {label} matched. Train it first: manage.py train {label}"
            )
        return found

    def _observed(
        self, model_class: Any, version: Any, options: dict[str, Any]
    ) -> tuple[list[Any], list[Any], str, str]:
        """The rows to measure, their predictions, and where they came from."""
        if options.get("against"):
            from mlango.core.registry import apps

            dataset = apps.get_dataset(options["against"])
            rows = list(dataset.objects.get_queryset().take(options["limit"]))
            # A dataset carries its own labels; comparing those to the training
            # labels answers a different question than "what is the model
            # saying now", so no predictions are claimed here.
            return rows, [], f"rows of {dataset._meta.label}", ""

        window = _parse_window(options["since"])
        inputs, outputs = self._from_log(model_class, window, options["limit"])
        return inputs, outputs, "logged predictions", options["since"]

    def _from_log(
        self, model_class: Any, window: dt.timedelta, limit: int
    ) -> tuple[list[Any], list[Any]]:
        from sqlalchemy import select

        from mlango.metastore.models import Prediction, utcnow
        from mlango.metastore.session import session_scope

        statement = (
            select(Prediction.inputs, Prediction.output)
            .where(
                Prediction.label == model_class._meta.label,
                Prediction.created_at >= utcnow() - window,
            )
            .order_by(Prediction.created_at.desc())
            .limit(limit)
        )
        with session_scope() as session:
            rows = list(session.execute(statement))
        return [row[0] for row in rows], [row[1] for row in rows if row[1] is not None]

    # -- output --------------------------------------------------------------

    def _report(
        self, version: Any, scores: dict[str, Any], rows: int, source: str, window: str
    ) -> None:
        from mlango.training import drift

        against = f" over the last {window}" if window else ""
        self.write(self.style.bold(f"{version.ref} vs {rows} {source}{against}"))
        self.write("")
        self.table(
            ["Column", "Kind", "PSI", "Verdict"],
            [
                [name, entry["kind"], f"{entry['psi']:.4f}", entry["verdict"]]
                for name, entry in sorted(scores.items(), key=lambda pair: -pair[1]["psi"])
            ],
        )
        self.write("")
        self.write(
            self.style.dim(
                f"PSI below {drift.STABLE} is stable, "
                f"{drift.STABLE}–{drift.SIGNIFICANT} moderate, "
                f"above {drift.SIGNIFICANT} significant."
            )
        )

        worst = max(scores.values(), key=lambda entry: entry["psi"])
        if worst["verdict"] == "significant":
            self.warn("A column has moved significantly. Retraining is likely overdue.")
        elif worst["verdict"] == "stable":
            self.ok("Nothing has moved.")

    def _maybe_fail(self, scores: dict[str, Any], threshold: str | None) -> None:
        if not threshold:
            return
        levels = {"stable": 0, "moderate": 1, "significant": 2}
        breached = [
            name for name, entry in scores.items() if levels[entry["verdict"]] >= levels[threshold]
        ]
        if breached:
            raise CommandError(f"Drift at or above {threshold}: {', '.join(sorted(breached))}")


def _parse_window(value: str) -> dt.timedelta:
    match = _WINDOW.match(value.strip().lower())
    if not match:
        raise CommandError(f"--since expects a window like 24h, 7d or 4w, got {value!r}.")
    amount, unit = match.groups()
    return dt.timedelta(**{_UNITS[unit]: int(amount)})
