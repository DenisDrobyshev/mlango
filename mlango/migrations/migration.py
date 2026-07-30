"""The Migration class that generated files subclass."""

from __future__ import annotations

from typing import Any

from mlango.migrations.operations import Operation
from mlango.migrations.state import ProjectState


class Migration:
    """A named, ordered set of operations belonging to one app."""

    #: ``[(app_label, migration_name), ...]`` that must be applied first.
    dependencies: list[tuple[str, str]] = []
    operations: list[Operation] = []
    #: True for the first migration of an app.
    initial: bool = False
    #: Data migrations that must not run inside the schema pass.
    atomic: bool = True

    def __init__(self, name: str, app_label: str):
        self.name = name
        self.app_label = app_label
        self.dependencies = list(self.dependencies)
        self.operations = list(self.operations)

    @property
    def key(self) -> tuple[str, str]:
        return (self.app_label, self.name)

    def mutate_state(self, state: ProjectState) -> ProjectState:
        """Advance ``state`` through this migration without side effects."""
        for operation in self.operations:
            operation.state_forwards(self.app_label, state)
        return state

    def apply(self, state: ProjectState, context: dict[str, Any]) -> ProjectState:
        for operation in self.operations:
            operation.state_forwards(self.app_label, state)
            operation.apply(self.app_label, state, context)
        return state

    def plan(self) -> list[str]:
        return [op.describe() for op in self.operations]

    def __repr__(self) -> str:
        return f"<Migration {self.app_label}.{self.name}>"

    def __str__(self) -> str:
        return f"{self.app_label}.{self.name}"
