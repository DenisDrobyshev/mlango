"""``mlango startplugin`` — scaffold a package that extends mlango.

Every extension point is a small contract, and writing the class was never the
hard part. Packaging it was: the entry-point stanza that turns "add this dotted
path to your settings" into "pip install it" is four lines nobody remembers, and
getting them wrong produces a package that installs cleanly and is never found.
"""

from __future__ import annotations

import os
import re
from typing import Any

from mlango.management.base import BaseCommand, CommandError

NAME_RE = re.compile(r"^[a-z][a-z0-9]*([-_][a-z0-9]+)*$")


class Command(BaseCommand):
    help = "Create a distributable package: a trainer, provider, storage backend or source."

    #: Scaffolding a package has nothing to do with the current project, and
    #: requiring one would mean you could not write an extension without first
    #: inventing a project to write it in.
    requires_apps = False
    requires_settings = False

    def add_arguments(self, parser) -> None:
        from mlango.plugin_template import KINDS

        parser.add_argument("name", help="Distribution name, e.g. mlango-lightgbm.")
        parser.add_argument(
            "directory", nargs="?", help="Where to create it. Defaults to ./<name>."
        )
        parser.add_argument(
            "--kind",
            choices=KINDS,
            default="trainer",
            help="What this package adds (default: trainer).",
        )
        parser.add_argument("--author", default="", help="Name for the LICENSE.")

    def handle(self, **options: Any) -> None:
        from mlango.plugin_template import (
            GROUPS,
            class_name,
            entry_name,
            module_name,
            render_plugin,
        )

        name = options["name"]
        kind = options["kind"]

        if not NAME_RE.match(name):
            raise CommandError(
                f"{name!r} is not a usable distribution name. Use lowercase letters, "
                f"digits, hyphens and underscores, starting with a letter — "
                f"for example mlango-lightgbm."
            )

        target = options.get("directory") or os.path.join(os.getcwd(), name)
        if os.path.exists(target) and os.listdir(target):
            raise CommandError(f"{target} already exists and is not empty.")

        if not name.startswith("mlango-"):
            # A convention, not a rule: the package works either way, and a
            # user searching PyPI for what extends mlango finds only the half
            # that followed it.
            self.warn(
                f"By convention an mlango extension is named mlango-<what it adds>, "
                f"so mlango-{name} rather than {name}. Continuing with {name!r}."
            )

        created = render_plugin(name, target, kind=kind, author=options["author"])
        entry = entry_name(name)
        module = module_name(name)

        self.ok(f"Created {kind} package {name!r} in {target}")
        for path in created:
            self.write(self.style.dim(f"  {os.path.relpath(path, target)}"))

        self.write("")
        self.write("Next steps:")
        self.write(f"  1. Implement {class_name(entry, kind)} in src/{module}/{kind}.py")
        self.write('  2. pip install -e ".[dev]" && pytest')
        if kind in GROUPS:
            self.write(
                f"  3. Any project that installs it can then say {kind} = {entry!r} "
                f"— the {GROUPS[kind]} entry point is already declared."
            )
        else:
            self.write(f"  3. Point settings at {module}.{kind}.{class_name(entry, kind)}")
