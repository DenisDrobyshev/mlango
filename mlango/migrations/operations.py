"""Migration operations.

Two things happen when a migration is applied:

* ``state_forwards`` updates the in-memory project state, which is how the
  autodetector knows what the code looked like at each point in history;
* ``apply`` does real work — for schema operations that means recording the
  change, and for :class:`RunPython` it means running the user's callable, which
  is where dataset backfills and re-encodings live.

The metastore's own tables never change shape here; what evolves is the
*declared schema* of datasets, models and agents.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mlango.core.fields import Field
from mlango.migrations.state import ObjectState, ProjectState


class Operation:
    """Base class for everything a migration can do."""

    #: Shown by ``manage.py migrate --plan``.
    reversible = False

    def state_forwards(self, app_label: str, state: ProjectState) -> None:
        raise NotImplementedError

    def apply(self, app_label: str, state: ProjectState, context: dict[str, Any]) -> None:
        """Side effects beyond state. Schema operations have none."""

    def describe(self) -> str:
        raise NotImplementedError

    def deconstruct(self) -> tuple[str, list[Any], dict[str, Any]]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.describe()}>"


class CreateObject(Operation):
    """Declare a new dataset, model, agent or eval."""

    def __init__(
        self,
        name: str,
        kind: str,
        fields: list[tuple[str, Field]] | None = None,
        options: dict[str, Any] | None = None,
    ):
        self.name = name
        self.kind = kind
        self.fields = list(fields or [])
        self.options = dict(options or {})

    def state_forwards(self, app_label: str, state: ProjectState) -> None:
        state.add(
            ObjectState(
                app_label=app_label,
                name=self.name,
                kind=self.kind,
                fields=list(self.fields),
                options=dict(self.options),
            )
        )

    def describe(self) -> str:
        return f"Create {self.kind} {self.name} ({len(self.fields)} fields)"

    def deconstruct(self):
        kwargs: dict[str, Any] = {"name": self.name, "kind": self.kind, "fields": self.fields}
        if self.options:
            kwargs["options"] = self.options
        return type(self).__name__, [], kwargs


class DeleteObject(Operation):
    def __init__(self, name: str):
        self.name = name

    def state_forwards(self, app_label: str, state: ProjectState) -> None:
        state.remove(app_label, self.name)

    def describe(self) -> str:
        return f"Delete {self.name}"

    def deconstruct(self):
        return type(self).__name__, [], {"name": self.name}


class RenameObject(Operation):
    reversible = True

    def __init__(self, old_name: str, new_name: str):
        self.old_name = old_name
        self.new_name = new_name

    def state_forwards(self, app_label: str, state: ProjectState) -> None:
        obj = state.get(app_label, self.old_name)
        if obj is None:
            return
        state.remove(app_label, self.old_name)
        renamed = obj.clone()
        renamed.name = self.new_name
        state.add(renamed)

    def describe(self) -> str:
        return f"Rename {self.old_name} to {self.new_name}"

    def deconstruct(self):
        return type(self).__name__, [], {"old_name": self.old_name, "new_name": self.new_name}


class AddField(Operation):
    def __init__(self, object_name: str, name: str, field: Field):
        self.object_name = object_name
        self.name = name
        self.field = field

    def state_forwards(self, app_label: str, state: ProjectState) -> None:
        obj = state.get(app_label, self.object_name)
        if obj is None:
            return
        obj.fields = [(n, f) for n, f in obj.fields if n != self.name]
        obj.fields.append((self.name, self.field))

    def describe(self) -> str:
        return f"Add field {self.name} to {self.object_name}"

    def deconstruct(self):
        return (
            type(self).__name__,
            [],
            {"object_name": self.object_name, "name": self.name, "field": self.field},
        )


class RemoveField(Operation):
    def __init__(self, object_name: str, name: str):
        self.object_name = object_name
        self.name = name

    def state_forwards(self, app_label: str, state: ProjectState) -> None:
        obj = state.get(app_label, self.object_name)
        if obj is None:
            return
        obj.fields = [(n, f) for n, f in obj.fields if n != self.name]

    def describe(self) -> str:
        return f"Remove field {self.name} from {self.object_name}"

    def deconstruct(self):
        return type(self).__name__, [], {"object_name": self.object_name, "name": self.name}


class AlterField(Operation):
    def __init__(self, object_name: str, name: str, field: Field):
        self.object_name = object_name
        self.name = name
        self.field = field

    def state_forwards(self, app_label: str, state: ProjectState) -> None:
        obj = state.get(app_label, self.object_name)
        if obj is None:
            return
        obj.fields = [(n, self.field if n == self.name else f) for n, f in obj.fields]

    def describe(self) -> str:
        return f"Alter field {self.name} on {self.object_name}"

    def deconstruct(self):
        return (
            type(self).__name__,
            [],
            {"object_name": self.object_name, "name": self.name, "field": self.field},
        )


class AlterOptions(Operation):
    def __init__(self, object_name: str, options: dict[str, Any]):
        self.object_name = object_name
        self.options = dict(options)

    def state_forwards(self, app_label: str, state: ProjectState) -> None:
        obj = state.get(app_label, self.object_name)
        if obj is not None:
            obj.options = dict(self.options)

    def describe(self) -> str:
        return f"Alter options on {self.object_name}"

    def deconstruct(self):
        return type(self).__name__, [], {"object_name": self.object_name, "options": self.options}


class RunPython(Operation):
    """Run arbitrary code — dataset backfills, re-encodings, cache warming.

    The callable receives ``(state, context)``. ``context`` carries a metastore
    session under ``"session"`` and the app label under ``"app_label"``, so a
    data migration can read and write versions without opening its own
    connection.
    """

    def __init__(
        self,
        code: Callable[..., Any],
        reverse_code: Callable[..., Any] | None = None,
        *,
        description: str = "",
    ):
        self.code = code
        self.reverse_code = reverse_code
        self.reversible = reverse_code is not None
        self._description = description

    def state_forwards(self, app_label: str, state: ProjectState) -> None:
        """Data migrations do not change the declared schema."""

    def apply(self, app_label: str, state: ProjectState, context: dict[str, Any]) -> None:
        self.code(state, {**context, "app_label": app_label})

    def describe(self) -> str:
        return self._description or f"Run python: {getattr(self.code, '__name__', 'code')}"

    def deconstruct(self):
        return type(self).__name__, [self.code], {}


#: Everything the migration writer knows how to render back into source.
OPERATION_TYPES = {
    cls.__name__: cls
    for cls in (
        CreateObject,
        DeleteObject,
        RenameObject,
        AddField,
        RemoveField,
        AlterField,
        AlterOptions,
        RunPython,
    )
}

__all__ = ["Operation", *OPERATION_TYPES, "OPERATION_TYPES"]
