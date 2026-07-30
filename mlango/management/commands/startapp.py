"""``manage.py startapp`` — scaffold a new app inside a project."""

from __future__ import annotations

import os
import re
from typing import Any

from mlango.management.base import BaseCommand, CommandError

NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


class Command(BaseCommand):
    help = "Create a new app: datasets, models, agents, evals, admin and migrations."

    #: The app does not exist yet, so there is nothing to import.
    requires_apps = False

    def add_arguments(self, parser) -> None:
        parser.add_argument("name", help="App name, lowercase with underscores.")
        parser.add_argument(
            "directory", nargs="?", help="Where to create it. Defaults to ./<name>."
        )

    def handle(self, **options: Any) -> None:
        from mlango.conf import settings
        from mlango.template import render_app

        name = options["name"]
        if not NAME_RE.match(name):
            raise CommandError(
                f"{name!r} is not a valid app name. Use lowercase letters, digits and "
                f"underscores, starting with a letter or underscore."
            )
        if name in {"mlango", "test", "admin"}:
            raise CommandError(f"{name!r} would shadow an existing module. Pick another name.")

        target = options.get("directory") or os.path.join(str(settings.BASE_DIR), name)
        if os.path.exists(target) and os.listdir(target):
            raise CommandError(f"{target} already exists and is not empty.")

        created = render_app(name, target)

        self.ok(f"Created app {name!r} in {target}")
        for path in created:
            self.write(self.style.dim(f"  {os.path.relpath(path, target)}"))

        self.write("")
        self.write("Next steps:")
        self.write(f"  1. Add {name!r} to INSTALLED_APPS in your settings module.")
        self.write(f"  2. Declare a dataset in {name}/datasets.py.")
        self.write("  3. Run: python manage.py makemigrations && python manage.py migrate")
