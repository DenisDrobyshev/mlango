"""``manage.py migrate`` — create metastore tables and apply pending migrations."""

from __future__ import annotations

from typing import Any

from mlango.management.base import BaseCommand


class Command(BaseCommand):
    help = "Create the metastore tables and apply pending migrations."

    def add_arguments(self, parser) -> None:
        parser.add_argument("app_label", nargs="?", help="Limit to a single app.")
        parser.add_argument(
            "--fake",
            action="store_true",
            help="Record migrations as applied without running their operations.",
        )
        parser.add_argument(
            "--plan",
            action="store_true",
            help="Show what would be applied, then stop.",
        )

    def handle(self, **options: Any) -> None:
        from mlango.metastore.session import create_all, metastore_url
        from mlango.migrations import MigrationExecutor

        app_labels = [options["app_label"]] if options.get("app_label") else None
        executor = MigrationExecutor()

        if options["plan"]:
            plan = executor.migration_plan(app_labels)
            if not plan:
                self.ok("No migrations to apply.")
                return
            self.write(self.style.bold(f"Plan ({len(plan)} migration(s)):"))
            for migration in plan:
                self.write(f"  {migration}")
                for line in migration.plan():
                    self.write(self.style.dim(f"      {line}"))
            return

        self.write(f"Metastore: {metastore_url()}")
        create_all()
        self.write("  metastore tables ready")

        def progress(event: str, migration: Any) -> None:
            if event == "apply_start":
                self.write(f"  applying {migration}...", level=1)
            elif event == "apply_success":
                self.ok(f"  applied  {migration}")

        applied = executor.migrate(app_labels, fake=options["fake"], progress=progress)
        if not applied:
            self.write("No migrations to apply.")
        else:
            verb = "faked" if options["fake"] else "applied"
            self.ok(f"{len(applied)} migration(s) {verb}.")
