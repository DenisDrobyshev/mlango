"""``mlango startproject`` — scaffold a new, already-working project."""

from __future__ import annotations

import os
import re
from typing import Any

from mlango.management.base import BaseCommand, CommandError

NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
RESERVED = {"mlango", "test", "tests", "admin", "django", "manage", "settings"}


class Command(BaseCommand):
    help = "Create a new project, ready to migrate, train and serve."

    #: Nothing exists yet, so there are no settings and no apps to load.
    requires_settings = False
    requires_apps = False

    def add_arguments(self, parser) -> None:
        parser.add_argument("name", help="Project name, lowercase with underscores.")
        parser.add_argument(
            "directory", nargs="?", help="Where to create it. Defaults to ./<name>."
        )
        parser.add_argument(
            "--bare",
            action="store_true",
            help="Skip the demo app and create an empty project.",
        )

    def handle(self, **options: Any) -> None:
        from mlango.template import render_project

        name = options["name"]
        if not NAME_RE.match(name):
            raise CommandError(
                f"{name!r} is not a valid project name. Use lowercase letters, digits "
                f"and underscores, starting with a letter or underscore."
            )
        if name in RESERVED:
            raise CommandError(f"{name!r} would shadow an existing module. Pick another name.")

        target = os.path.abspath(options.get("directory") or name)
        if os.path.exists(target) and os.listdir(target):
            raise CommandError(f"{target} already exists and is not empty.")

        created = render_project(name, target, demo=not options["bare"])

        self.ok(f"Created project {name!r} in {target}")
        for path in created:
            self.write(self.style.dim(f"  {os.path.relpath(path, target)}"))

        self.write("")
        self.write(self.style.bold("Next steps"))
        directory = os.path.basename(target)
        self.write(f"  cd {directory}")
        self.write("  pip install -r requirements.txt")
        self.write("  python manage.py migrate")
        if not options["bare"]:
            self.write("  python manage.py train demo.Sentiment")
        self.write("  python manage.py runserver")
        self.write("")
        if options["bare"]:
            self.write("Then add an app: python manage.py startapp myapp")
        else:
            self.write("Then open http://127.0.0.1:8000/admin/ — it will already have data in it.")
            self.write("")
            # The scaffold writes tests, so it has to say how to run them. They
            # need pytest, which is not in requirements.txt because that file
            # builds the production image.
            self.write(self.style.dim("The project ships tests. To run them:"))
            self.write(self.style.dim("  pip install -r requirements-dev.txt"))
            self.write(self.style.dim("  python manage.py test"))
