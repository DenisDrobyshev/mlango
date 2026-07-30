"""``manage.py runserver`` — serve the admin and the inference API."""

from __future__ import annotations

from typing import Any

from mlango.management.base import BaseCommand


class Command(BaseCommand):
    help = "Run the development server: admin plus inference API."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "addrport",
            nargs="?",
            help="Optional [host:]port, e.g. 8000 or 0.0.0.0:8000.",
        )
        parser.add_argument("--host", help="Bind address.")
        parser.add_argument("--port", type=int, help="Port to listen on.")
        parser.add_argument(
            "--reload", action="store_true", help="Restart the server when files change."
        )
        parser.add_argument("--no-admin", action="store_true", help="Serve the API only.")

    def handle(self, **options: Any) -> None:
        from mlango.conf import settings
        from mlango.core.registry import apps
        from mlango.serve.api import run
        from mlango.serve.routing import load_routes

        host, port = _resolve_address(options, settings)

        counts = apps.summary()["counts"]
        self.write(self.style.bold(f"mlango development server"))
        self.write(f"  settings   {settings.SETTINGS_MODULE}")
        self.write(
            "  declared   " + ", ".join(f"{kind}s: {n}" for kind, n in counts.items())
        )

        routes = load_routes()
        if routes:
            for route in routes:
                self.write(f"  endpoint   POST /api{route.path} → {route.endpoint.label}")
        else:
            self.write(self.style.dim("  endpoint   (none — add urlpatterns to ROOT_ROUTECONF)"))

        admin = settings.ADMIN_ENABLED and not options["no_admin"]
        self.write("")
        if admin:
            self.ok(f"  Admin      http://{host}:{port}{settings.ADMIN_URL}/")
        self.ok(f"  API docs   http://{host}:{port}/api/docs")
        self.write(self.style.dim("  Quit with CTRL-C.\n"))

        if options["no_admin"]:
            settings.ADMIN_ENABLED = False

        run(host=host, port=port, reload=options["reload"])


def _resolve_address(options: dict[str, Any], settings: Any) -> tuple[str, int]:
    host = options.get("host") or settings.SERVE_HOST
    port = options.get("port") or settings.SERVE_PORT

    addrport = options.get("addrport")
    if addrport:
        if ":" in addrport:
            raw_host, _, raw_port = addrport.rpartition(":")
            host = raw_host or host
            port = int(raw_port)
        else:
            port = int(addrport)
    return str(host), int(port)
