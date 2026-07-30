"""``manage.py showmigrations`` — which migrations exist and which have run."""

from __future__ import annotations

from typing import Any

from mlango.management.base import BaseCommand


class Command(BaseCommand):
    help = "List each app's migrations and whether they have been applied."

    def add_arguments(self, parser) -> None:
        parser.add_argument("app_label", nargs="?", help="Limit to a single app.")

    def handle(self, **options: Any) -> None:
        from mlango.migrations import MigrationExecutor

        status = MigrationExecutor().status()
        wanted = options.get("app_label")

        if not status:
            self.write("No apps have a migrations package yet.")
            return

        for app_label, migrations in status.items():
            if wanted and app_label != wanted:
                continue
            self.write(self.style.bold(app_label))
            if not migrations:
                self.write(self.style.dim("  (no migrations)"))
                continue
            for entry in migrations:
                mark = self.style.success("[X]") if entry["applied"] else "[ ]"
                self.write(f"  {mark} {entry['name']}")
                for line in entry["operations"]:
                    self.write(self.style.dim(f"        {line}"), level=2)
