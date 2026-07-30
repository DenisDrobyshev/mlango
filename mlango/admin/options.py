"""Per-object admin configuration.

    @admin.register(Reviews)
    class ReviewsAdmin(ObjectAdmin):
        list_display = ("id", "text", "label")
        list_filter = ("label",)
        search_fields = ("text",)

Same shape as Django's ``ModelAdmin``, named ``ObjectAdmin`` because "model"
already means something specific here.
"""

from __future__ import annotations

from typing import Any

from mlango.core.exceptions import FieldError
from mlango.core.typing import DeclarativeClass


class ObjectAdmin:
    """Presentation options for one declared dataset, model, agent or eval."""

    #: Columns shown in the list view. Empty means "the first few fields".
    list_display: tuple[str, ...] = ()
    #: Fields offered as dropdown filters.
    list_filter: tuple[str, ...] = ()
    #: Fields searched by the search box (case-insensitive substring).
    search_fields: tuple[str, ...] = ()
    #: Default ordering, ``-field`` for descending.
    ordering: tuple[str, ...] = ()
    #: Rows per page in the data preview.
    list_per_page: int = 25
    #: How many rows the preview may scan before giving up on a huge source.
    preview_limit: int = 5000
    #: Free text shown at the top of the object's page.
    help_text: str = ""
    #: Field whose date the preview can be drilled into, Django's
    #: ``date_hierarchy``. Must name a DateField or DateTimeField.
    date_hierarchy: str = ""
    #: Names of ``action_*`` methods offered above the preview, in this order.
    #: Empty means every ``action_*`` this class defines.
    actions: tuple[str, ...] = ()

    def __init__(self, target: DeclarativeClass, site: Any):
        self.target = target
        self.site = site

    # -- actions -------------------------------------------------------------

    def get_actions(self) -> dict[str, str]:
        """``{name: description}`` for the actions this admin offers.

        An action is a method named ``action_<name>`` taking the selected
        records. Its first docstring line is the label, so the thing a user
        reads and the thing a developer reads cannot drift apart::

            class ReviewsAdmin(ObjectAdmin):
                def action_export(self, records):
                    "Export the selected rows as JSONL"
                    ...
        """
        names = self.actions or tuple(
            sorted(n[len("action_") :] for n in dir(type(self)) if n.startswith("action_"))
        )
        found: dict[str, str] = {}
        for name in names:
            method = getattr(self, f"action_{name}", None)
            if not callable(method):
                continue
            summary = (method.__doc__ or "").strip().splitlines()
            found[name] = summary[0].strip() if summary else name.replace("_", " ").capitalize()
        return found

    def run_action(self, name: str, records: list[Any]) -> str:
        """Run one action and return what to tell the user."""
        from mlango.core.exceptions import FieldError

        if name not in self.get_actions():
            available = ", ".join(self.get_actions()) or "(none)"
            raise FieldError(f"{self.label} has no admin action {name!r}. Available: {available}.")
        outcome = getattr(self, f"action_{name}")(records)
        if outcome:
            return str(outcome)
        return f"{self.get_actions()[name]}: {len(records)} row(s)."

    # -- naming --------------------------------------------------------------

    @property
    def opts(self) -> Any:
        return self.target._meta

    @property
    def label(self) -> str:
        return self.opts.label

    @property
    def kind(self) -> str:
        return self.opts.kind

    @property
    def title(self) -> str:
        return self.opts.verbose_name_plural.title()

    # -- columns -------------------------------------------------------------

    def get_list_display(self) -> list[str]:
        if self.list_display:
            return list(self.list_display)
        names = self.opts.field_names
        return names[:6]

    def get_list_filter(self) -> list[str]:
        if self.list_filter:
            return list(self.list_filter)
        # Fields with a bounded set of values make useful default filters.
        return [
            f.name or ""
            for f in self.opts.fields
            if getattr(f, "classes", None) or f.choices or f.kind == "boolean"
        ][:4]

    def get_search_fields(self) -> list[str]:
        if self.search_fields:
            return list(self.search_fields)
        return [f.name or "" for f in self.opts.fields if f.kind in {"text", "char"}][:3]

    def check(self) -> list[str]:
        """Report options that name fields the object does not declare."""
        problems: list[str] = []
        known = set(self.opts.field_names)
        for option in ("list_display", "list_filter", "search_fields"):
            for name in getattr(self, option):
                if name.lstrip("-") not in known:
                    problems.append(
                        f"{type(self).__name__}.{option} names {name!r}, which "
                        f"{self.label} does not declare."
                    )
        return problems

    # -- data ----------------------------------------------------------------

    def get_queryset(self) -> Any:
        """The rows shown in the preview. Only meaningful for datasets."""
        manager = getattr(self.target, "objects", None)
        if manager is None:
            raise FieldError(f"{self.label} has no objects manager to preview.")
        query = manager.get_queryset()
        if self.ordering:
            query = query.order_by(*self.ordering)
        return query

    def filtered(
        self, *, search: str = "", filters: dict[str, str] | None = None, period: str = ""
    ) -> Any:
        query = self.get_queryset()
        for name, value in (filters or {}).items():
            if value:
                query = query.filter(**{name: value})
        if period and self.date_hierarchy:
            query = query.where(
                lambda record, field=self.date_hierarchy, prefix=period: str(
                    record.get(field) or ""
                ).startswith(prefix),
                label=f"{self.date_hierarchy}:{period}",
            )
        if search:
            terms = self.get_search_fields()
            if terms:
                needle = search.casefold()
                query = query.where(
                    lambda record, terms=terms, needle=needle: any(
                        needle in str(record.get(t, "")).casefold() for t in terms
                    ),
                    label=f"search:{search}",
                )
        return query

    def date_periods(self, *, limit: int = 24) -> list[str]:
        """Distinct ``YYYY-MM`` prefixes for the date drill-down, newest first."""
        if not self.date_hierarchy:
            return []
        try:
            seen: set[str] = set()
            for record in self.get_queryset().take(self.preview_limit):
                stamp = str(record.get(self.date_hierarchy) or "")
                if len(stamp) >= 7:
                    seen.add(stamp[:7])
            return sorted(seen, reverse=True)[:limit]
        except Exception:
            # A drill-down is a convenience; a source that cannot be scanned
            # must not take the page down with it.
            return []

    def filter_values(self, name: str, *, limit: int = 25) -> list[Any]:
        """Distinct values for a filter dropdown, from the declaration if possible."""
        field = self.opts.fields_map.get(name)
        declared = getattr(field, "classes", None) or (field.choices if field else None)
        if declared:
            return list(declared)[:limit]
        try:
            seen: list[Any] = []
            for record in self.get_queryset().take(self.preview_limit):
                value = record.get(name)
                if value not in seen:
                    seen.append(value)
                if len(seen) >= limit:
                    break
            return seen
        except Exception:
            return []

    # -- rendering -----------------------------------------------------------

    def render(self, record: Any, column: str) -> str:
        """Text for one cell. Override for custom formatting."""
        formatter = getattr(self, f"render_{column}", None)
        if callable(formatter):
            return str(formatter(record))
        value = record.get(column)
        if value is None:
            return "—"
        text = str(value)
        return text if len(text) <= 160 else text[:157] + "…"

    def summary(self) -> dict[str, Any]:
        """Headline facts shown on the object page."""
        summarise = getattr(self.target, "summary", None)
        if callable(summarise):
            try:
                return summarise()
            except Exception as exc:
                return {"label": self.label, "error": str(exc)}
        return {"label": self.label}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} for {self.label}>"


class DatasetAdmin(ObjectAdmin):
    """Default options for datasets."""


class ModelAdmin(ObjectAdmin):
    """Default options for models."""


class AgentAdmin(ObjectAdmin):
    """Default options for agents."""


class EvalAdmin(ObjectAdmin):
    """Default options for evals."""


DEFAULTS: dict[str, type[ObjectAdmin]] = {
    "dataset": DatasetAdmin,
    "model": ModelAdmin,
    "agent": AgentAdmin,
    "eval": EvalAdmin,
}

__all__ = ["ObjectAdmin", "DatasetAdmin", "ModelAdmin", "AgentAdmin", "EvalAdmin", "DEFAULTS"]
