"""Diffing two project states into a list of operations.

Deliberately conservative: it never guesses that a removed field and an added
field are "really" a rename, because guessing wrong on an ML schema silently
corrupts the meaning of stored data. Renames are expressed by hand with
:class:`~mlango.migrations.operations.RenameObject`.
"""

from __future__ import annotations

from typing import Any

from mlango.migrations import operations as ops
from mlango.migrations.state import ProjectState, _deconstruct_payload


class MigrationAutodetector:
    def __init__(self, from_state: ProjectState, to_state: ProjectState):
        self.from_state = from_state
        self.to_state = to_state

    # -- entry point ---------------------------------------------------------

    def changes(self, app_labels: list[str] | None = None) -> dict[str, list[ops.Operation]]:
        """Operations to turn ``from_state`` into ``to_state``, grouped by app."""
        labels = set(app_labels) if app_labels else self._all_app_labels()
        result: dict[str, list[ops.Operation]] = {}
        for app_label in sorted(labels):
            operations = self._changes_for_app(app_label)
            if operations:
                result[app_label] = operations
        return result

    def _all_app_labels(self) -> set[str]:
        return {key[0] for key in self.from_state.objects} | {
            key[0] for key in self.to_state.objects
        }

    # -- per-app diff --------------------------------------------------------

    def _changes_for_app(self, app_label: str) -> list[ops.Operation]:
        old = self.from_state.for_app(app_label)
        new = self.to_state.for_app(app_label)

        created = [key for key in new if key not in old]
        deleted = [key for key in old if key not in new]
        common = [key for key in new if key in old]

        operations: list[ops.Operation] = []

        for key in sorted(created, key=lambda k: k[1]):
            obj = new[key]
            operations.append(
                ops.CreateObject(
                    name=obj.name,
                    kind=obj.kind,
                    fields=list(obj.fields),
                    options=dict(obj.options),
                )
            )

        for key in sorted(common, key=lambda k: k[1]):
            operations.extend(self._changes_for_object(old[key], new[key]))

        # Deletions last: an AddField on a surviving object should not be
        # ordered behind the removal of an unrelated one.
        for key in sorted(deleted, key=lambda k: k[1]):
            operations.append(ops.DeleteObject(name=old[key].name))

        return operations

    def _changes_for_object(self, old_obj, new_obj) -> list[ops.Operation]:
        operations: list[ops.Operation] = []
        old_fields = dict(old_obj.fields)
        new_fields = dict(new_obj.fields)

        for name in new_obj.field_names():
            if name not in old_fields:
                operations.append(ops.AddField(new_obj.name, name, new_fields[name]))
            elif _field_changed(old_fields[name], new_fields[name]):
                operations.append(ops.AlterField(new_obj.name, name, new_fields[name]))

        for name in old_obj.field_names():
            if name not in new_fields:
                operations.append(ops.RemoveField(new_obj.name, name))

        if _normalise(old_obj.options) != _normalise(new_obj.options):
            operations.append(ops.AlterOptions(new_obj.name, dict(new_obj.options)))

        if old_obj.kind != new_obj.kind:
            # A dataset that became a model is a new object, not an edit.
            operations = [
                ops.DeleteObject(name=old_obj.name),
                ops.CreateObject(
                    name=new_obj.name,
                    kind=new_obj.kind,
                    fields=list(new_obj.fields),
                    options=dict(new_obj.options),
                ),
            ]
        return operations

    # -- naming --------------------------------------------------------------

    @staticmethod
    def suggest_name(operations: list[ops.Operation], *, initial: bool = False) -> str:
        if initial:
            return "initial"
        if len(operations) == 1:
            operation = operations[0]
            if isinstance(operation, ops.CreateObject):
                return f"create_{operation.name.lower()}"
            if isinstance(operation, ops.DeleteObject):
                return f"delete_{operation.name.lower()}"
            if isinstance(operation, ops.AddField):
                return f"{operation.object_name.lower()}_{operation.name}"
            if isinstance(operation, ops.RemoveField):
                return f"remove_{operation.object_name.lower()}_{operation.name}"
            if isinstance(operation, ops.AlterField):
                return f"alter_{operation.object_name.lower()}_{operation.name}"
            if isinstance(operation, ops.AlterOptions):
                return f"alter_{operation.object_name.lower()}_options"
        if operations and all(isinstance(op, ops.CreateObject) for op in operations):
            return "_".join(op.name.lower() for op in operations[:3])  # type: ignore[attr-defined]

        from datetime import datetime

        return f"auto_{datetime.now().strftime('%Y%m%d_%H%M')}"


def _field_changed(old_field, new_field) -> bool:
    return _deconstruct_payload(old_field) != _deconstruct_payload(new_field)


def _normalise(options: dict[str, Any]) -> dict[str, Any]:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(options.items())}
