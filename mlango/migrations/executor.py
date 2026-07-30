"""Applying migrations and recording that they ran."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import select

from mlango.metastore.models import MigrationRecord
from mlango.metastore.session import create_all, session_scope
from mlango.migrations.loader import MigrationLoader
from mlango.migrations.migration import Migration
from mlango.migrations.state import ProjectState


class MigrationExecutor:
    def __init__(self, loader: MigrationLoader | None = None):
        self.loader = loader or MigrationLoader()

    # -- state ---------------------------------------------------------------

    def applied_migrations(self) -> set[tuple[str, str]]:
        create_all()
        with session_scope() as session:
            rows = session.execute(select(MigrationRecord.app, MigrationRecord.name)).all()
        return {(app, name) for app, name in rows}

    def migration_plan(self, app_labels: list[str] | None = None) -> list[Migration]:
        applied = self.applied_migrations()
        plan = []
        for migration in self.loader.graph():
            if app_labels is not None and migration.app_label not in app_labels:
                continue
            if migration.key not in applied:
                plan.append(migration)
        return plan

    # -- execution -----------------------------------------------------------

    def migrate(
        self,
        app_labels: list[str] | None = None,
        *,
        fake: bool = False,
        progress: Callable[[str, Migration], None] | None = None,
    ) -> list[Migration]:
        """Create metastore tables, then apply every pending migration."""
        create_all()
        applied = self.applied_migrations()
        plan = self.migration_plan(app_labels)

        # State up to the first unapplied migration, so data migrations see the
        # schema as it was when they were written.
        state = ProjectState()
        for migration in self.loader.graph():
            if migration.key in applied:
                migration.mutate_state(state)

        done: list[Migration] = []
        for migration in plan:
            if progress:
                progress("apply_start", migration)
            if fake:
                migration.mutate_state(state)
            else:
                with session_scope() as session:
                    migration.apply(state, {"session": session, "executor": self})
            self._record(migration)
            done.append(migration)
            if progress:
                progress("apply_success", migration)
        return done

    def unapply(self, app_label: str, name: str) -> bool:
        """Forget that a migration ran. Does not undo its side effects."""
        with session_scope() as session:
            record = session.execute(
                select(MigrationRecord).where(
                    MigrationRecord.app == app_label, MigrationRecord.name == name
                )
            ).scalar_one_or_none()
            if record is None:
                return False
            session.delete(record)
        return True

    def _record(self, migration: Migration) -> None:
        with session_scope() as session:
            exists = session.execute(
                select(MigrationRecord).where(
                    MigrationRecord.app == migration.app_label,
                    MigrationRecord.name == migration.name,
                )
            ).scalar_one_or_none()
            if exists is None:
                session.add(MigrationRecord(app=migration.app_label, name=migration.name))

    # -- reporting -----------------------------------------------------------

    def status(self) -> dict[str, list[dict[str, Any]]]:
        applied = self.applied_migrations()
        out: dict[str, list[dict[str, Any]]] = {}
        for app_label, migrations in sorted(self.loader.by_app.items()):
            out[app_label] = [
                {"name": m.name, "applied": m.key in applied, "operations": m.plan()}
                for m in migrations
            ]
        return out
