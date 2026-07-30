"""The admin site registry.

Unlike Django, everything declared is visible by default. A project's first
question is always "what datasets and runs do I have?", and answering it should
not require writing an ``admin.py`` first. Registering explicitly is still how
you customise columns, filters and search.
"""

from __future__ import annotations

from typing import Any

from mlango.admin.options import DEFAULTS, ObjectAdmin
from mlango.core.exceptions import ImproperlyConfigured


class AdminSite:
    def __init__(self, name: str = "admin"):
        self.name = name
        self._registry: dict[str, ObjectAdmin] = {}
        self._explicit: set[str] = set()

    # -- registration --------------------------------------------------------

    def register(
        self, target: type | None = None, admin_class: type[ObjectAdmin] | None = None
    ) -> Any:
        """Register an object, optionally with custom options.

        Usable directly (``admin.site.register(Reviews, ReviewsAdmin)``) or as
        a decorator (``@admin.register(Reviews)``).
        """
        if target is None:
            raise ImproperlyConfigured("register() needs the object to register.")

        if admin_class is None:

            def decorator(options: type[ObjectAdmin]) -> type[ObjectAdmin]:
                self._attach(target, options, explicit=True)
                return options

            return decorator

        self._attach(target, admin_class, explicit=True)
        return admin_class

    def _attach(self, target: type, admin_class: type[ObjectAdmin], *, explicit: bool) -> None:
        meta = getattr(target, "_meta", None)
        if meta is None:
            raise ImproperlyConfigured(f"{target!r} is not an mlango declarative class.")
        label = meta.label
        if explicit and label in self._explicit:
            raise ImproperlyConfigured(f"{label} is already registered with the admin.")
        self._registry[label] = admin_class(target, self)
        if explicit:
            self._explicit.add(label)

    def unregister(self, target: type) -> None:
        label = target._meta.label
        self._registry.pop(label, None)
        self._explicit.discard(label)

    def autodiscover(self) -> None:
        """Give every declared object a default admin unless one was declared."""
        from mlango.core.registry import KINDS, apps

        for kind in KINDS:
            for target in apps.get_registered(kind):
                if target._meta.label in self._explicit:
                    continue
                self._attach(target, DEFAULTS.get(kind, ObjectAdmin), explicit=False)

    # -- lookups -------------------------------------------------------------

    def get(self, label: str) -> ObjectAdmin:
        try:
            return self._registry[label]
        except KeyError as exc:
            known = ", ".join(sorted(self._registry)) or "(nothing registered)"
            raise LookupError(f"{label!r} is not in the admin. Registered: {known}.") from exc

    def all(self, kind: str | None = None) -> list[ObjectAdmin]:
        entries = sorted(self._registry.values(), key=lambda a: a.label)
        return [a for a in entries if kind is None or a.kind == kind]

    def app_list(self) -> list[dict[str, Any]]:
        """Objects grouped by app, then by kind — the sidebar's data."""
        from mlango.core.registry import KINDS, apps

        grouped: dict[str, dict[str, list[ObjectAdmin]]] = {}
        for entry in self.all():
            app_label = entry.opts.app_label or "(unscoped)"
            grouped.setdefault(app_label, {}).setdefault(entry.kind, []).append(entry)

        out = []
        for app_label in sorted(grouped):
            config = apps.app_configs.get(app_label)
            out.append(
                {
                    "label": app_label,
                    "name": config.verbose_name if config else app_label.replace("_", " ").title(),
                    "kinds": [
                        {"kind": kind, "objects": grouped[app_label][kind]}
                        for kind in KINDS
                        if kind in grouped[app_label]
                    ],
                }
            )
        return out

    def check(self) -> list[str]:
        problems: list[str] = []
        for entry in self.all():
            problems.extend(entry.check())
        return problems

    def __contains__(self, label: object) -> bool:
        return label in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"<AdminSite {self.name!r}: {len(self._registry)} objects>"


#: The default site, the one ``manage.py runserver`` mounts.
site = AdminSite()


def register(target: type, admin_class: type[ObjectAdmin] | None = None) -> Any:
    """Module-level shortcut for ``site.register``."""
    return site.register(target, admin_class)


def autodiscover() -> None:
    site.autodiscover()


__all__ = ["AdminSite", "site", "register", "autodiscover", "ObjectAdmin"]
