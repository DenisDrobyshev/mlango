"""The management command base class.

    class Command(BaseCommand):
        help = "Train a declared model."

        def add_arguments(self, parser):
            parser.add_argument("model")

        def handle(self, **options):
            ...

Same contract as Django's: argparse for the interface, ``handle`` for the work,
a ``CommandError`` for anything the user should see as a message rather than a
traceback.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from mlango.core.exceptions import MlangoError


class CommandError(MlangoError):
    """A user-facing failure: printed as a message, exits non-zero."""

    def __init__(self, message: str, *, returncode: int = 1):
        super().__init__(message)
        self.returncode = returncode


class Style:
    """ANSI colours, disabled automatically when output is redirected."""

    def __init__(self, enabled: bool | None = None):
        self.enabled = sys.stdout.isatty() if enabled is None else enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def success(self, text: str) -> str:
        return self._wrap("32", text)

    def warn(self, text: str) -> str:
        return self._wrap("33", text)

    def error(self, text: str) -> str:
        return self._wrap("31", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)


class BaseCommand:
    """Base class for every ``manage.py`` subcommand."""

    #: One-line description shown by ``manage.py help``.
    help: str = ""
    #: Set False for commands that must run before apps can be imported.
    requires_apps: bool = True
    #: Set False for commands that must run without a settings module.
    requires_settings: bool = True

    def __init__(self, name: str = ""):
        self.name = name or type(self).__module__.rpartition(".")[2]
        self.style = Style()
        self.verbosity = 1

    # -- interface -----------------------------------------------------------

    def create_parser(self, prog: str) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog=f"{prog} {self.name}",
            description=self.help or None,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument("--settings", help="Dotted path to the settings module for this run.")
        parser.add_argument(
            "-v",
            "--verbosity",
            type=int,
            default=1,
            choices=[0, 1, 2, 3],
            help="0 = quiet, 1 = normal, 2 = verbose, 3 = very verbose.",
        )
        parser.add_argument(
            "--traceback",
            action="store_true",
            help="Show the full traceback on error instead of a message.",
        )
        self.add_arguments(parser)
        return parser

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Declare the command's own arguments."""

    # -- execution -----------------------------------------------------------

    def run_from_argv(self, argv: list[str], prog: str = "manage.py") -> int:
        parser = self.create_parser(prog)
        options = vars(parser.parse_args(argv))
        traceback = options.pop("traceback", False)

        try:
            self.execute(**options)
        except CommandError as exc:
            if traceback:
                raise
            self.stderr(self.style.error(f"error: {exc}"))
            return exc.returncode
        except MlangoError as exc:
            if traceback:
                raise
            self.stderr(self.style.error(f"{type(exc).__name__}: {exc}"))
            return 1
        except KeyboardInterrupt:
            self.stderr(self.style.warn("\ninterrupted"))
            return 130
        return 0

    def execute(self, **options: Any) -> Any:
        self.verbosity = int(options.get("verbosity", 1))
        settings_module = options.pop("settings", None)

        if self.requires_settings:
            self._configure(settings_module)
        if self.requires_apps:
            import mlango

            mlango.setup()

        return self.handle(**options)

    def _configure(self, settings_module: str | None) -> None:
        import logging
        import os

        from mlango.conf import ENVIRONMENT_VARIABLE, settings

        if settings_module:
            os.environ[ENVIRONMENT_VARIABLE] = settings_module
        if not os.environ.get(ENVIRONMENT_VARIABLE):
            raise CommandError(
                f"Settings are not configured. Run this through manage.py, or set "
                f"{ENVIRONMENT_VARIABLE}, or pass --settings=myproject.settings."
            )
        logging.basicConfig(
            level=getattr(logging, str(settings.LOG_LEVEL).upper(), logging.INFO),
            format=settings.LOG_FORMAT,
        )

    def handle(self, **options: Any) -> Any:
        raise NotImplementedError("Subclasses of BaseCommand must implement handle().")

    # -- output --------------------------------------------------------------

    def write(self, message: str = "", *, level: int = 1) -> None:
        if self.verbosity >= level:
            print(message)

    def stderr(self, message: str) -> None:
        print(message, file=sys.stderr)

    def ok(self, message: str) -> None:
        self.write(self.style.success(message))

    def warn(self, message: str) -> None:
        self.write(self.style.warn(message))

    def table(self, headers: list[str], rows: list[list[Any]]) -> None:
        """Print an aligned table — the CLI's answer to the admin's list view."""
        if not rows:
            self.write(self.style.dim("(nothing to show)"))
            return
        # Normalise to the header count: a ragged row is a display glitch, not a
        # reason for the whole command to fail.
        width = len(headers)
        cells = [[str(c) for c in row][:width] + [""] * max(0, width - len(row)) for row in rows]
        widths = [
            max(len(headers[index]), max((len(row[index]) for row in cells), default=0))
            for index in range(width)
        ]
        self.write(
            self.style.bold("  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)))
        )
        self.write(self.style.dim("  ".join("-" * w for w in widths)))
        for row in cells:
            self.write("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))


class LabelCommand(BaseCommand):
    """A command that takes one or more object labels."""

    label_name = "label"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(self.label_name, nargs="+", help="One or more object labels.")

    def handle(self, **options: Any) -> Any:
        for label in options.pop(self.label_name):
            self.handle_label(label, **options)

    def handle_label(self, label: str, **options: Any) -> Any:
        raise NotImplementedError


__all__ = ["BaseCommand", "LabelCommand", "CommandError", "Style"]
