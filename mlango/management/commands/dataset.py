"""``manage.py dataset`` — inspect, validate and materialise datasets."""

from __future__ import annotations

import json
from typing import Any

from mlango.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Inspect, validate and materialise declared datasets."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "action",
            choices=["list", "show", "head", "validate", "materialize", "versions"],
            help="What to do.",
        )
        parser.add_argument("dataset", nargs="?", help="Dataset label (not needed for 'list').")
        parser.add_argument("-n", "--rows", type=int, default=10, help="Rows for 'head'.")
        parser.add_argument("--notes", default="", help="Notes for 'materialize'.")
        parser.add_argument(
            "--force", action="store_true", help="Materialise even if the content is unchanged."
        )

    def handle(self, **options: Any) -> None:
        from mlango.core.registry import apps

        action = options["action"]
        if action == "list":
            self._list(apps)
            return

        if not options.get("dataset"):
            raise CommandError(f"'{action}' needs a dataset label.")
        dataset = apps.get_dataset(options["dataset"])
        getattr(self, f"_{action}")(dataset, options)

    # -- actions -------------------------------------------------------------

    def _list(self, apps: Any) -> None:
        rows = []
        for dataset in apps.get_registered("dataset"):
            source = dataset.get_source()
            count = source.count() if source is not None else None
            rows.append(
                [
                    dataset._meta.label,
                    len(dataset._meta.fields),
                    source.describe().get("type") if source else "-",
                    count if count is not None else "?",
                ]
            )
        self.table(["dataset", "fields", "source", "rows"], rows)

    def _show(self, dataset: type, options: dict[str, Any]) -> None:
        self.write(self.style.bold(dataset._meta.label))
        if dataset._meta.description:
            self.write(self.style.dim(f"  {dataset._meta.description}"))
        self.write("")
        self.table(
            ["field", "type", "required", "detail"],
            [
                [
                    f.name,
                    type(f).__name__,
                    "yes" if f.required else "no",
                    ", ".join(
                        f"{k}={v}"
                        for k, v in f.describe().items()
                        if k not in {"name", "class", "kind", "required", "null"}
                    ),
                ]
                for f in dataset._meta.fields
            ],
        )
        self.write("")
        self.write(json.dumps(dataset.summary(), indent=2, default=str))

    def _head(self, dataset: type, options: dict[str, Any]) -> None:
        rows = list(dataset.objects.take(options["rows"]))
        if not rows:
            self.write("(no rows)")
            return
        columns = list(rows[0].keys())
        self.table(columns, [[str(row.get(c))[:40] for c in columns] for row in rows])

    def _validate(self, dataset: type, options: dict[str, Any]) -> None:
        from mlango.core.exceptions import ValidationError

        checked = 0
        try:
            for _record in dataset.objects.validate():
                checked += 1
        except ValidationError as exc:
            self.write(self.style.error(f"Validation failed after {checked} row(s):"))
            for key, messages in exc.errors.items():
                for message in messages:
                    self.write(f"  {key}: {message}")
            raise CommandError("The dataset does not match its declaration.") from exc
        self.ok(f"{checked} row(s) validated against {dataset._meta.label}.")

    def _materialize(self, dataset: type, options: dict[str, Any]) -> None:
        version = dataset.materialize(notes=options["notes"], force=options["force"])
        self.ok(f"{version.ref} - {version.row_count} row(s)")
        self.write(self.style.dim(f"  content {(version.content_hash or '')[:16]}"))
        self.write(self.style.dim(f"  path    {version.path}"))

    def _versions(self, dataset: type, options: dict[str, Any]) -> None:
        self.table(
            ["version", "rows", "schema", "content", "notes", "created"],
            [
                [
                    f"v{v.version}",
                    v.row_count,
                    v.fingerprint,
                    (v.content_hash or "")[:12],
                    v.notes[:30],
                    v.created_at.strftime("%Y-%m-%d %H:%M"),
                ]
                for v in dataset.versions()
            ],
        )
