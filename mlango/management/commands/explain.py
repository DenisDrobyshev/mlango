"""``manage.py explain`` — what a trained version weighted, without loading it.

The weights are recorded on the version row at registration, so this answers
"why did it say that" from the metastore alone. A version trained before the
column existed, or by a backend that has since learned to explain itself, can
be filled in with ``--recompute``.
"""

from __future__ import annotations

import json
from typing import Any

from mlango.management.base import BaseCommand, CommandError

#: Width of the bar column. Narrow enough to leave room for a long feature name
#: and its weight on an 80-column terminal.
BAR_WIDTH = 32


class Command(BaseCommand):
    help = "Show which features a trained model version relied on."

    def add_arguments(self, parser) -> None:
        parser.add_argument("model", help="Model label, e.g. reviews.Sentiment.")
        parser.add_argument("--version", type=int, help="Version to explain (default: latest).")
        parser.add_argument("--stage", help="Explain the version at this stage, e.g. production.")
        parser.add_argument(
            "-n", "--top", type=int, default=20, help="How many features to show (default 20)."
        )
        parser.add_argument("--json", action="store_true", help="Emit JSON instead of a chart.")
        parser.add_argument(
            "--recompute",
            action="store_true",
            help="Load the artifact and re-derive the weights, then store them.",
        )

    def handle(self, **options: Any) -> None:
        from mlango.core.registry import apps

        model_class = apps.get_model(options["model"])
        version = self._version(model_class, options)

        weights = dict(version.importances or {})
        if options["recompute"] or not weights:
            weights = self._recompute(model_class, version, options)

        if not weights:
            trainer = model_class._meta.extras.get("trainer", "")
            raise CommandError(
                f"{model_class._meta.label}@v{version.version} has no feature weights. "
                f"The {trainer!r} backend does not report any — that is expected for "
                f"neural networks, where no vector of numbers names a column."
            )

        ranked = sorted(weights.items(), key=lambda pair: -abs(pair[1]))[: options["top"]]
        if options["json"]:
            self.write(json.dumps(dict(ranked), ensure_ascii=False, indent=2))
            return
        self._chart(model_class, version, ranked, len(weights))

    # -- selection -----------------------------------------------------------

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
            asked = ""
            if options.get("version"):
                asked = f" at v{options['version']}"
            elif options.get("stage"):
                asked = f" at stage {options['stage']!r}"
            raise CommandError(
                f"No registered version of {label}{asked}. Train it first: manage.py train {label}"
            )
        return found

    def _recompute(
        self, model_class: Any, version: Any, options: dict[str, Any]
    ) -> dict[str, float]:
        """Derive the weights from the stored artifact and write them back.

        Loading is the expensive path, which is why it is not the default. It
        exists because a version trained before this column was added is
        otherwise permanently unexplainable, and refusing to look at an
        artifact that is sitting right there would be an odd thing for a
        framework to insist on.
        """
        from sqlalchemy import select

        from mlango.core.exceptions import MlangoError
        from mlango.metastore.models import ModelVersion
        from mlango.metastore.session import session_scope

        try:
            model = model_class.load(version=version.version)
        except (LookupError, MlangoError) as exc:
            raise CommandError(f"Could not load {version.ref}: {exc}") from exc

        weights = model_class.get_trainer().importances(model, model.fitted) or {}
        if weights:
            with session_scope() as session:
                row = session.execute(
                    select(ModelVersion).where(ModelVersion.id == version.id)
                ).scalar_one()
                row.importances = weights
            if self.verbosity >= 1:
                self.write(self.style.dim(f"Stored {len(weights)} weights on {version.ref}."))
        return {str(k): float(v) for k, v in weights.items()}

    # -- output --------------------------------------------------------------

    def _chart(
        self, model_class: Any, version: Any, ranked: list[tuple[str, float]], total: int
    ) -> None:
        self.write(self.style.bold(f"{model_class._meta.label}@v{version.version}"))
        shown = f"top {len(ranked)} of {total}" if len(ranked) < total else f"{total} features"
        self.write(self.style.dim(f"{shown}, largest weight first"))
        self.write("")

        width = max(len(name) for name, _ in ranked)
        largest = max(abs(value) for _, value in ranked)
        for name, value in ranked:
            filled = round(abs(value) / largest * BAR_WIDTH) if largest else 0
            bar = "█" * filled + "·" * (BAR_WIDTH - filled)
            # The sign matters for a linear model — a strongly negative
            # coefficient is evidence *against* the class, and a bar drawn from
            # the magnitude alone would show it as agreement.
            mark = "-" if value < 0 else " "
            self.write(f"{name:<{width}}  {bar} {mark}{abs(value):.4f}")
