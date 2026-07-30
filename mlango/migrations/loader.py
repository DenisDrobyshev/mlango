"""Discovering migration modules on disk and ordering them."""

from __future__ import annotations

import importlib
import os
from typing import Any

from mlango.core.exceptions import MigrationError
from mlango.core.module_loading import module_has_submodule
from mlango.migrations.migration import Migration
from mlango.migrations.state import ProjectState
from mlango.migrations.writer import MIGRATION_RE


class MigrationLoader:
    """Loads ``<app>/migrations/NNNN_name.py`` for every installed app."""

    def __init__(self, *, ignore_missing: bool = True):
        self.ignore_missing = ignore_missing
        self.disk_migrations: dict[tuple[str, str], Migration] = {}
        self.by_app: dict[str, list[Migration]] = {}
        self.load()

    # -- loading -------------------------------------------------------------

    def load(self) -> None:
        from mlango.core.registry import apps

        self.disk_migrations.clear()
        self.by_app.clear()

        for config in apps.get_app_configs():
            names = self._migration_names(config)
            migrations: list[Migration] = []
            for name in names:
                migration = self._import_migration(config.name, config.label, name)
                self.disk_migrations[(config.label, name)] = migration
                migrations.append(migration)
            self.by_app[config.label] = migrations

    def _migration_names(self, config: Any) -> list[str]:
        directory = config.migrations_dir
        if not os.path.isdir(directory):
            return []
        names = [
            filename[:-3] for filename in os.listdir(directory) if MIGRATION_RE.match(filename)
        ]
        return sorted(names)

    def _import_migration(self, app_name: str, app_label: str, name: str) -> Migration:
        module_path = f"{app_name}.migrations.{name}"
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise MigrationError(f"Could not import migration {module_path}: {exc}") from exc
        migration_class = getattr(module, "Migration", None)
        if migration_class is None or not issubclass(migration_class, Migration):
            raise MigrationError(
                f"{module_path} does not define a Migration class subclassing "
                f"mlango.migrations.Migration."
            )
        return migration_class(name, app_label)

    # -- queries -------------------------------------------------------------

    def migrations_for(self, app_label: str) -> list[Migration]:
        return list(self.by_app.get(app_label, []))

    def latest_name(self, app_label: str) -> str | None:
        migrations = self.migrations_for(app_label)
        return migrations[-1].name if migrations else None

    def has_migrations(self, app_label: str) -> bool:
        return bool(self.by_app.get(app_label))

    def graph(self) -> list[Migration]:
        """All migrations in a dependency-respecting order.

        Within an app, the numeric prefix already gives a total order; across
        apps, declared ``dependencies`` are honoured. A cycle is a hard error
        rather than something to sort around.
        """
        ordered: list[Migration] = []
        seen: set[tuple[str, str]] = set()
        visiting: set[tuple[str, str]] = set()

        def visit(migration: Migration) -> None:
            key = migration.key
            if key in seen:
                return
            if key in visiting:
                raise MigrationError(f"Circular migration dependency at {migration}.")
            visiting.add(key)

            # Implicit dependency on the previous migration of the same app.
            siblings = self.migrations_for(migration.app_label)
            index = siblings.index(migration)
            if index > 0:
                visit(siblings[index - 1])

            for dep in migration.dependencies:
                dependency = self.disk_migrations.get(tuple(dep))  # type: ignore[arg-type]
                if dependency is None:
                    if self.ignore_missing:
                        continue
                    raise MigrationError(
                        f"{migration} depends on {dep[0]}.{dep[1]}, which does not exist."
                    )
                visit(dependency)

            visiting.discard(key)
            seen.add(key)
            ordered.append(migration)

        for app_label in sorted(self.by_app):
            for migration in self.migrations_for(app_label):
                visit(migration)
        return ordered

    def project_state(self, app_labels: list[str] | None = None) -> ProjectState:
        """Replay every migration to rebuild the state the files describe."""
        state = ProjectState()
        for migration in self.graph():
            if app_labels is not None and migration.app_label not in app_labels:
                continue
            migration.mutate_state(state)
        return state

    @staticmethod
    def app_has_migrations_package(config: Any) -> bool:
        return module_has_submodule(config.module, "migrations")
