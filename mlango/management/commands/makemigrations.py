"""``manage.py makemigrations`` — write migrations for changed declarations."""

from __future__ import annotations

import os
from typing import Any

from mlango.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Write migration files for changes to declared datasets, models, agents and evals."

    def add_arguments(self, parser) -> None:
        parser.add_argument("app_label", nargs="?", help="Limit to a single app.")
        parser.add_argument("-n", "--name", help="Use this name instead of a generated one.")
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be written without touching the filesystem.",
        )
        parser.add_argument(
            "--empty",
            action="store_true",
            help="Write an empty migration to fill in by hand (for data migrations).",
        )

    def handle(self, **options: Any) -> None:
        from mlango.core.registry import apps
        from mlango.migrations import (
            MigrationAutodetector,
            MigrationLoader,
            MigrationWriter,
            ProjectState,
            build_filename,
            next_migration_number,
        )

        app_labels = [options["app_label"]] if options.get("app_label") else None
        if app_labels:
            apps.get_app_config(app_labels[0])  # raises a clear LookupError if unknown

        loader = MigrationLoader()
        from_state = loader.project_state(app_labels)
        to_state = ProjectState.from_registry(app_labels)

        if options["empty"]:
            if not app_labels:
                raise CommandError("--empty needs an app label: makemigrations myapp --empty")
            changes: dict[str, list[Any]] = {app_labels[0]: []}
        else:
            changes = MigrationAutodetector(from_state, to_state).changes(app_labels)

        if not changes:
            self.write("No changes detected.")
            return

        for app_label, operations in changes.items():
            config = apps.get_app_config(app_label)
            directory = config.migrations_dir
            number = next_migration_number(directory)
            initial = number == 1
            suggested = options.get("name") or MigrationAutodetector.suggest_name(
                operations, initial=initial
            )
            filename = build_filename(number, suggested)

            dependencies = []
            previous = loader.latest_name(app_label)
            if previous:
                dependencies.append((app_label, previous))

            writer = MigrationWriter(app_label, filename, operations, dependencies, initial=initial)

            self.write(self.style.bold(f"{app_label}/migrations/{filename}.py"))
            for operation in operations:
                self.write(f"  - {operation.describe()}")
            if not operations:
                self.write(self.style.dim("  (empty — add operations by hand)"))

            if options["dry_run"]:
                self.write(self.style.dim(writer.as_string()), level=2)
                continue

            path = writer.write(directory)
            self.ok(f"  written {os.path.relpath(path, str(config.path))}")

        if options["dry_run"]:
            self.warn("Dry run: nothing was written.")
        else:
            self.ok("Run 'manage.py migrate' to apply them.")
