"""The declared state of a project, and how to snapshot it.

A migration is a diff between two of these. ``ProjectState.from_registry()``
reads what the code says *right now*; replaying an app's migration files
rebuilds what the code said when they were written. Comparing the two is how
``makemigrations`` knows what changed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field as dc_field
from typing import Any

from mlango.core.fields import Field
from mlango.core.hashing import fingerprint


@dataclass
class ObjectState:
    """One declared dataset, model, agent or eval, frozen in time."""

    app_label: str
    name: str
    kind: str
    fields: list[tuple[str, Field]] = dc_field(default_factory=list)
    options: dict[str, Any] = dc_field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.app_label}.{self.name}"

    @property
    def key(self) -> tuple[str, str]:
        return (self.app_label, self.name)

    def field_names(self) -> list[str]:
        return [name for name, _ in self.fields]

    def get_field(self, name: str) -> Field | None:
        for field_name, field_obj in self.fields:
            if field_name == name:
                return field_obj
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "fields": [
                {"name": name, **_deconstruct_payload(field_obj)} for name, field_obj in self.fields
            ],
            "options": {k: v for k, v in sorted(self.options.items())},
        }

    def fingerprint(self) -> str:
        return fingerprint(self.describe())

    def clone(self) -> ObjectState:
        return ObjectState(
            app_label=self.app_label,
            name=self.name,
            kind=self.kind,
            fields=[(name, copy.deepcopy(f)) for name, f in self.fields],
            options=copy.deepcopy(self.options),
        )


def _deconstruct_payload(field_obj: Field) -> dict[str, Any]:
    _name, path, args, kwargs = field_obj.deconstruct()
    return {"path": path, "args": list(args), "kwargs": kwargs}


class ProjectState:
    def __init__(self, objects: dict[tuple[str, str], ObjectState] | None = None):
        self.objects: dict[tuple[str, str], ObjectState] = objects or {}

    # -- construction --------------------------------------------------------

    @classmethod
    def from_registry(cls, app_labels: list[str] | None = None) -> ProjectState:
        from mlango.core.registry import KINDS, apps

        state = cls()
        for kind in KINDS:
            for obj in apps.get_registered(kind):
                opts = obj._meta
                if app_labels is not None and opts.app_label not in app_labels:
                    continue
                state.add(
                    ObjectState(
                        app_label=opts.app_label or "",
                        name=opts.object_name,
                        kind=kind,
                        fields=[(f.name or "", f) for f in opts.fields],
                        options=_serialisable_options(opts),
                    )
                )
        return state

    def add(self, obj: ObjectState) -> None:
        self.objects[obj.key] = obj

    def remove(self, app_label: str, name: str) -> None:
        self.objects.pop((app_label, name), None)

    def get(self, app_label: str, name: str) -> ObjectState | None:
        return self.objects.get((app_label, name))

    def for_app(self, app_label: str) -> dict[tuple[str, str], ObjectState]:
        return {k: v for k, v in self.objects.items() if k[0] == app_label}

    def clone(self) -> ProjectState:
        return ProjectState({key: obj.clone() for key, obj in self.objects.items()})

    def describe(self) -> dict[str, Any]:
        return {obj.label: obj.describe() for obj in sorted(self.objects.values(), key=lambda o: o.label)}

    def __len__(self) -> int:
        return len(self.objects)

    def __repr__(self) -> str:
        return f"<ProjectState {len(self.objects)} objects>"


def _serialisable_options(opts: Any) -> dict[str, Any]:
    """Keep only Meta options that survive a round trip through a file.

    A ``source`` pointing at a live Python generator cannot be written into a
    migration, and pretending otherwise would produce migrations that crash on
    import. Those options are simply not part of migration state.
    """
    out: dict[str, Any] = {}
    for key, value in opts.extras.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            out[key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, (str, int, float, bool, type(None))) for v in value
        ):
            out[key] = list(value)
        elif isinstance(value, dict) and all(isinstance(k, str) for k in value):
            try:
                import json

                json.dumps(value)
            except (TypeError, ValueError):
                continue
            out[key] = value
    if opts.description:
        out.setdefault("description", opts.description)
    return out
