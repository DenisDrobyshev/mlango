"""``manage.py inspectdata`` — write a Dataset declaration from a data file.

Django's ``inspectdb`` for data files. Point it at a CSV, JSONL, JSON or Parquet
file and it prints a declaration you can paste into ``datasets.py``.
"""

from __future__ import annotations

import os
from typing import Any

from mlango.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Generate a Dataset declaration by sampling a data file."

    # Reading a file needs settings (for BASE_DIR) but not a loaded registry:
    # this command exists to be run *before* anything is declared.
    requires_apps = False

    def add_arguments(self, parser) -> None:
        parser.add_argument("path", help="Data file: .csv, .tsv, .jsonl, .json or .parquet.")
        parser.add_argument(
            "--name",
            help="Class name. Defaults to a CamelCase form of the filename.",
        )
        parser.add_argument("--app", help="App the declaration is destined for, used in --write.")
        parser.add_argument(
            "-n",
            "--sample",
            type=int,
            default=1000,
            help="Rows to read before deciding on types (default 1000).",
        )
        parser.add_argument(
            "--write",
            action="store_true",
            help="Write to <app>/datasets.py instead of printing. Requires --app.",
        )
        parser.add_argument(
            "--force", action="store_true", help="Overwrite an existing file when writing."
        )

    def handle(self, **options: Any) -> None:
        from mlango.conf import settings
        from mlango.data.inspect import profile_source, render_declaration, source_for

        path = options["path"]
        resolved = path if os.path.isabs(path) else os.path.join(str(settings.BASE_DIR), path)
        if not os.path.exists(resolved):
            raise CommandError(f"No such file: {resolved}")

        if options["sample"] < 1:
            raise CommandError("--sample must be at least 1.")

        source, source_expr = source_for(path)
        name = options["name"] or _class_name(path)
        if not name.isidentifier():
            raise CommandError(f"{name!r} is not a valid class name; pass --name.")

        result = profile_source(source, sample=options["sample"])
        if not result.columns:
            raise CommandError(f"{resolved} yielded no records, so there is nothing to declare.")

        declaration = render_declaration(
            result, name=name, source_expr=source_expr, app=options.get("app")
        )

        if options["write"]:
            self._write(declaration, options)
        else:
            print(declaration)

        self._report(result, name)

    # -- output --------------------------------------------------------------

    def _write(self, declaration: str, options: dict[str, Any]) -> None:
        from mlango.conf import settings

        app = options.get("app")
        if not app:
            raise CommandError("--write needs --app to say where the file goes.")

        directory = os.path.join(str(settings.BASE_DIR), app)
        if not os.path.isdir(directory):
            raise CommandError(f"No such app directory: {directory}. Run startapp {app} first.")

        target = os.path.join(directory, "datasets.py")
        if os.path.exists(target) and not options["force"] and _declares_something(target):
            raise CommandError(
                f"{target} already declares something. Review the output first, then "
                f"pass --force to overwrite it."
            )

        with open(target, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(declaration)
        self.ok(f"Wrote {target}")

    def _report(self, result: Any, name: str) -> None:
        if result.rows_total is None or result.rows_total <= result.rows_sampled:
            scale = f"Read {result.rows_sampled} rows."
        else:
            scale = f"Sampled {result.rows_sampled} of {result.rows_total} rows."

        self.write("")
        self.write(self.style.dim(scale))

        rows = [
            [c.name, c.field_class, c.distinct, c.nulls, _preview(c.samples)]
            for c in result.columns
        ]
        self.table(["column", "field", "distinct", "empty", "sample"], rows)

        if result.primary_key:
            self.write(f"  primary_key  {result.primary_key}")
        if result.target:
            self.write(f"  target       {result.target}")

        for warning in result.warnings:
            self.warn(f"  {warning}")

        self.write("")
        self.write(
            self.style.dim(
                "The types are inferred from a sample. Read the declaration before you rely on it."
            )
        )


def _declares_something(path: str) -> bool:
    """True if the file defines a class, rather than only describing one.

    A scaffolded ``datasets.py`` is a docstring and a commented-out example, so
    overwriting it costs nothing — but searching the text for "class" finds the
    comment too, and would refuse to write into a brand-new app.
    """
    import ast

    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except (OSError, SyntaxError):
        # Unreadable or half-edited: treat it as precious and make the user look.
        return True
    return any(isinstance(node, (ast.ClassDef, ast.FunctionDef)) for node in tree.body)


def _class_name(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = [part for part in stem.replace("-", "_").replace(".", "_").split("_") if part]
    return "".join(part[:1].upper() + part[1:] for part in parts) or "Records"


def _preview(samples: list[Any]) -> str:
    if not samples:
        return "-"
    text = ", ".join(str(s) for s in samples)
    return text if len(text) <= 34 else text[:33] + "…"
