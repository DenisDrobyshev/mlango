"""``manage.py check`` — validate the project without running anything."""

from __future__ import annotations

from typing import Any

from mlango.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Check the project for configuration problems."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--fail-level",
            choices=["warning", "error"],
            default="error",
            help="Exit non-zero at this severity or above.",
        )

    def handle(self, **options: Any) -> None:
        from mlango.agents.providers import available_providers
        from mlango.conf import settings
        from mlango.core.registry import apps
        from mlango.metastore.session import metastore_ready, metastore_url
        from mlango.migrations import MigrationExecutor
        from mlango.training.trainer import available_trainers

        errors: list[str] = []
        warnings: list[str] = []

        summary = apps.summary()
        self.write(self.style.bold("Project"))
        self.write(f"  settings   {settings.SETTINGS_MODULE}")
        self.write(f"  base dir   {settings.BASE_DIR}")
        self.write(f"  apps       {', '.join(summary['apps']) or '(none)'}")
        self.write(
            "  declared   " + ", ".join(f"{kind}s: {n}" for kind, n in summary["counts"].items())
        )

        if not settings.INSTALLED_APPS:
            warnings.append("INSTALLED_APPS is empty, so nothing is declared.")
        if not settings.SECRET_KEY:
            warnings.append("SECRET_KEY is empty; set it before deploying.")
        if settings.DEBUG:
            warnings.append("DEBUG is on; turn it off in production.")

        self.write("")
        self.write(self.style.bold("Metastore"))
        self.write(f"  url        {metastore_url()}")
        ready = metastore_ready()
        self.write(f"  tables     {'present' if ready else 'not created yet'}")
        if not ready:
            warnings.append("Metastore tables do not exist yet. Run: manage.py migrate")

        self.write("")
        self.write(self.style.bold("Backends"))
        for name, ok in available_trainers().items():
            self.write(f"  trainer    {name}: {'available' if ok else 'missing dependency'}")
        for name, ok in available_providers().items():
            self.write(f"  provider   {name}: {'available' if ok else 'missing dependency'}")

        # Dotted-path settings only fail when something first uses them, which
        # in a sweep means the same import error repeated once per trial.
        # Resolving them here turns that into one clear message up front.
        self.write("")
        self.write(self.style.bold("Wiring"))
        errors.extend(self._resolve_paths("DEFAULT_CALLBACKS", settings.DEFAULT_CALLBACKS))
        errors.extend(self._resolve_paths("SERVE_MIDDLEWARE", settings.SERVE_MIDDLEWARE))
        errors.extend(self._resolve_paths("STORAGE.BACKEND", [settings.STORAGE.get("BACKEND")]))
        if settings.ROOT_ROUTECONF:
            try:
                from mlango.serve.routing import load_routes

                routes = load_routes()
                self.write(f"  routes     {len(routes)} endpoint(s)")
            except Exception as exc:
                errors.append(f"ROOT_ROUTECONF could not be loaded: {exc}")
        else:
            self.write(self.style.dim("  routes     (ROOT_ROUTECONF is not set)"))

        self.write("")
        self.write(self.style.bold("Migrations"))
        try:
            plan = MigrationExecutor().migration_plan()
            if plan:
                warnings.append(f"{len(plan)} migration(s) are not applied. Run: manage.py migrate")
                for migration in plan:
                    self.write(f"  pending    {migration}")
            else:
                self.write("  up to date")
        except Exception as exc:
            errors.append(f"Could not read migrations: {exc}")

        self.write("")
        self.write(self.style.bold("Admin"))
        try:
            from mlango.admin import site
            from mlango.admin.auth import auth_configured

            site.autodiscover()
            errors.extend(site.check())
            self.write(f"  registered {len(site)} object(s)")
            self.write(f"  auth       {'enabled' if auth_configured() else 'off'}")
            if not auth_configured() and not settings.DEBUG:
                warnings.append(
                    "The admin is unauthenticated and DEBUG is off. Set ADMIN_PASSWORD, "
                    "or put the admin behind your identity provider."
                )
        except Exception as exc:
            errors.append(f"Could not build the admin site: {exc}")

        self.write("")
        for message in warnings:
            self.warn(f"WARNING: {message}")
        for message in errors:
            self.write(self.style.error(f"ERROR: {message}"))

        if errors:
            raise CommandError(f"{len(errors)} error(s) found.")
        if warnings and options["fail_level"] == "warning":
            raise CommandError(f"{len(warnings)} warning(s) found.")
        self.ok(f"Check complete — {len(warnings)} warning(s), no errors.")

    def _resolve_paths(self, setting: str, paths: list[Any]) -> list[str]:
        """Import each dotted path, reporting the ones that do not resolve."""
        from mlango.core.module_loading import import_string

        problems: list[str] = []
        entries = [p for p in paths if p]
        if not entries:
            self.write(self.style.dim(f"  {setting.split('.')[0].lower():<10} (empty)"))
            return problems

        for dotted in entries:
            try:
                import_string(str(dotted))
                self.write(f"  {'ok':<10} {setting}: {dotted}")
            except Exception as exc:
                problems.append(f"{setting} names {dotted!r}, which cannot be imported: {exc}")
        return problems
