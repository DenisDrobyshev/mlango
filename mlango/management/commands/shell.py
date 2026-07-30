"""``manage.py shell`` — a Python shell with the project already loaded."""

from __future__ import annotations

import code
from typing import Any

from mlango.management.base import BaseCommand


class Command(BaseCommand):
    help = "Start a Python shell with every declared object already imported."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "-c", "--command", help="Run this code instead of starting an interactive shell."
        )
        parser.add_argument(
            "--plain",
            action="store_true",
            help="Force the plain shell even if IPython is installed.",
        )

    def handle(self, **options: Any) -> None:
        namespace = self._namespace()

        if options.get("command"):
            exec(options["command"], namespace)  # noqa: S102 - the point of the command
            return

        banner = self._banner()

        if not options["plain"]:
            try:
                from IPython import start_ipython

                print(banner)
                start_ipython(argv=[], user_ns=namespace, display_banner=False)
                return
            except ImportError:
                pass

        code.interact(banner=banner, local=namespace, exitmsg="")

    # -- namespace -----------------------------------------------------------

    def _namespace(self) -> dict[str, Any]:
        import mlango
        from mlango.agents.tracing import get_trace, recent_traces
        from mlango.conf import settings
        from mlango.core.registry import KINDS, apps
        from mlango.training.run import get_run, recent_runs

        namespace: dict[str, Any] = {
            "mlango": mlango,
            "settings": settings,
            "apps": apps,
            "recent_runs": recent_runs,
            "get_run": get_run,
            "recent_traces": recent_traces,
            "get_trace": get_trace,
        }
        for kind in KINDS:
            for obj in apps.get_registered(kind):
                namespace[obj.__name__] = obj
        return namespace

    def _banner(self) -> str:
        import sys

        import mlango
        from mlango.core.registry import KINDS, apps

        lines = [f"Python {sys.version.split()[0]} — mlango {mlango.get_version()} shell", ""]
        for kind in KINDS:
            objects = apps.get_registered(kind)
            if objects:
                lines.append(f"  {kind}s: " + ", ".join(o.__name__ for o in objects))
        lines.append("")
        lines.append("  helpers: apps, settings, recent_runs(), get_run(), recent_traces()")
        return "\n".join(lines)
