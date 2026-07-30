"""Infer a ``Dataset`` declaration from a data file.

Django has ``inspectdb`` for the same reason this exists: the first thing anyone
does with a new framework is point it at data they already have, and hand-writing
a field for every column is exactly the friction that stops them.

    python manage.py inspectdata data/reviews.csv

prints a declaration ready to paste into ``datasets.py``. It is a starting point,
not an oracle — the output is meant to be read and edited, which is why every
guess it makes carries a comment saying so.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from mlango.core.exceptions import ImproperlyConfigured
from mlango.data.sources import CSVSource, JSONLSource, JSONSource, Source

#: Column names that conventionally hold the thing you want to predict. Checked
#: in order, so `label` wins over `category` when a file has both.
TARGET_NAMES = (
    "label",
    "labels",
    "target",
    "y",
    "class",
    "outcome",
    "sentiment",
    "category",
    "rating",
)

#: Above this many distinct values, a string column is free text rather than a
#: category. Deliberately low: a wrong LabelField is more annoying than a wrong
#: CharField, because it also changes what the model tries to predict.
MAX_CLASSES = 24

#: A column is only categorical if its values actually repeat.
MAX_DISTINCT_RATIO = 0.5

#: Longest value a column may hold and still be treated as bounded. Above this
#: the column becomes a TextField. The threshold is low on purpose: a CharField
#: whose max_length is too small rejects valid data later, while a TextField
#: never rejects anything — so when in doubt, do not impose a limit.
SHORT_VALUE_LENGTH = 32

_TRUE = {"true", "t", "yes", "y", "1", "on"}
_FALSE = {"false", "f", "no", "n", "0", "off"}
_INT_RE = re.compile(r"^[+-]?\d+$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class ColumnProfile:
    """What one column looks like, and the field chosen for it."""

    name: str
    field_class: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    count: int = 0
    nulls: int = 0
    distinct: int = 0
    samples: list[Any] = field(default_factory=list)
    #: Set when the choice was a judgement call worth a comment in the output.
    note: str = ""

    @property
    def all_null(self) -> bool:
        return self.count > 0 and self.nulls == self.count

    @property
    def unique(self) -> bool:
        return self.count > 0 and self.distinct == self.count and self.nulls == 0


@dataclass
class DataProfile:
    """The inferred shape of a file."""

    columns: list[ColumnProfile] = field(default_factory=list)
    rows_sampled: int = 0
    rows_total: int | None = None
    primary_key: str | None = None
    target: str | None = None
    source_repr: str = ""
    warnings: list[str] = field(default_factory=list)

    def get(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)


# --------------------------------------------------------------------------- #
# Source detection
# --------------------------------------------------------------------------- #

_BY_SUFFIX = {
    ".jsonl": "JSONLSource",
    ".ndjson": "JSONLSource",
    ".json": "JSONSource",
    ".csv": "CSVSource",
    ".tsv": "CSVSource",
    ".parquet": "ParquetSource",
    ".pq": "ParquetSource",
}


def source_for(path: str) -> tuple[Source, str]:
    """A source for ``path``, plus the expression that reproduces it.

    Returns the expression too so the generated declaration says
    ``CSVSource("data/reviews.csv", delimiter="\\t")`` rather than making the
    reader work out which arguments were used.
    """
    suffix = os.path.splitext(path)[1].lower()
    name = _BY_SUFFIX.get(suffix)
    if name is None:
        known = ", ".join(sorted(_BY_SUFFIX))
        raise ImproperlyConfigured(
            f"Cannot tell how to read {os.path.basename(path)!r}. Recognised extensions: {known}."
        )

    literal = json.dumps(path)
    if name == "CSVSource":
        if suffix == ".tsv":
            return CSVSource(path, delimiter="\t"), f'CSVSource({literal}, delimiter="\\t")'
        return CSVSource(path), f"CSVSource({literal})"
    if name == "JSONLSource":
        return JSONLSource(path), f"JSONLSource({literal})"
    if name == "JSONSource":
        return JSONSource(path), f"JSONSource({literal})"

    from mlango.data.extra_sources import ParquetSource

    return ParquetSource(path), f"ParquetSource({literal})"


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #


def profile_source(source: Source, *, sample: int = 1000) -> DataProfile:
    """Read up to ``sample`` records and decide on a field for each column."""
    rows: list[dict[str, Any]] = []
    for record in source:
        rows.append(record)
        if len(rows) >= sample:
            break

    result = DataProfile(rows_sampled=len(rows))
    try:
        result.rows_total = source.count()
    except Exception:  # noqa: BLE001 - a count is a nicety, not a requirement
        result.rows_total = None

    if not rows:
        result.warnings.append("The file contained no records, so there is nothing to infer.")
        return result

    # Column order follows first appearance, which for CSV is the header order.
    names: list[str] = []
    for record in rows:
        for key in record:
            if key not in names:
                names.append(key)

    for name in names:
        values = [record.get(name) for record in rows]
        result.columns.append(_profile_column(name, values))

    _rename_unusable(result)
    result.primary_key = _pick_primary_key(result)
    result.target = _pick_target(result)
    _apply_target(result)
    return result


def _profile_column(name: str, values: list[Any]) -> ColumnProfile:
    present = [v for v in values if v is not None and v != ""]
    nulls = len(values) - len(present)

    distinct: set[Any] = set()
    for value in present:
        try:
            distinct.add(value)
        except TypeError:
            # dicts and lists are unhashable; their exact cardinality does not
            # matter, only that they are not categorical.
            distinct.add(repr(value))

    column = ColumnProfile(
        name=name,
        field_class="CharField",
        count=len(values),
        nulls=nulls,
        distinct=len(distinct),
        samples=present[:3],
    )

    if not present:
        column.field_class = "CharField"
        column.note = "every sampled value was empty, so the type is a guess"
        column.kwargs = {"null": True, "required": False}
        return column

    kind, kwargs = _infer(name, present, distinct)
    column.field_class = kind
    column.kwargs = kwargs
    if nulls:
        column.kwargs["null"] = True
        column.kwargs["required"] = False
    return column


def _infer(name: str, present: list[Any], distinct: set[Any]) -> tuple[str, dict[str, Any]]:
    """Pick a field class and its arguments for one column's values."""
    if _all_bool(present):
        return "BooleanField", {}

    if _all(present, _is_json_container):
        return "JSONField", {}

    if _all(present, _is_int):
        integers = [int(v) for v in present]
        return "IntegerField", {"min_value": min(integers), "max_value": max(integers)}

    if _all(present, _is_float):
        floats = [float(v) for v in present]
        return "FloatField", {"min_value": _round(min(floats)), "max_value": _round(max(floats))}

    if _all(present, _is_datetime):
        return "DateTimeField", {}

    longest = max(len(str(v)) for v in present)
    categorical = (
        len(distinct) <= MAX_CLASSES and len(distinct) <= len(present) * MAX_DISTINCT_RATIO
    )

    if categorical:
        # Not a LabelField yet: exactly one column becomes the target, chosen
        # later. Declaring several would leave Model.get_target() unable to
        # decide, which is a worse first experience than a plain CharField.
        return "CharField", {
            "max_length": _length_cap(longest),
            "choices": sorted(distinct, key=str),
        }

    if longest > SHORT_VALUE_LENGTH:
        return "TextField", {}
    return "CharField", {"max_length": _length_cap(longest)}


def _rename_unusable(result: DataProfile) -> None:
    """Flag columns whose names cannot be Python attributes.

    A CSV header of ``Review Text`` or ``class`` cannot be an attribute name, and
    silently mangling it would make the declaration disagree with the file.
    """
    import keyword

    for column in result.columns:
        if not _IDENTIFIER_RE.match(column.name) or keyword.iskeyword(column.name):
            suggestion = re.sub(r"\W+", "_", column.name).strip("_").lower() or "column"
            if keyword.iskeyword(suggestion):
                suggestion += "_"
            result.warnings.append(
                f"Column {column.name!r} is not a valid Python name. Declare it as "
                f"{suggestion!r} and add .rename({suggestion}={column.name!r}) to the "
                f"queryset, or rename it in the file."
            )
            column.note = f"cannot be an attribute name; shown as {suggestion!r}"
            column.name = suggestion


def _pick_primary_key(result: DataProfile) -> str | None:
    """A column that identifies a row, for stable splits and dedup."""
    candidates = [c for c in result.columns if c.unique]
    for name in ("id", "uuid", "pk"):
        match = next((c for c in candidates if c.name.lower() == name), None)
        if match is not None:
            return match.name
    return next((c.name for c in candidates if c.name.lower().endswith("_id")), None)


def _pick_target(result: DataProfile) -> str | None:
    """The column most likely to be what you want to predict."""
    # Trailing underscore stripped because we add it ourselves: a column named
    # `class` becomes `class_`, and it is still conventionally the target.
    by_name = {c.name.lower().rstrip("_"): c for c in result.columns}
    for name in TARGET_NAMES:
        column = by_name.get(name)
        if column is not None and column.name != result.primary_key:
            return column.name

    # No conventional name: fall back to the last categorical column, which is
    # where a label usually sits in a CSV exported for training.
    categorical = [
        c for c in result.columns if c.kwargs.get("choices") and c.name != result.primary_key
    ]
    return categorical[-1].name if categorical else None


def _apply_target(result: DataProfile) -> None:
    """Turn the chosen target column into a target field."""
    if result.target is None:
        result.warnings.append(
            "No obvious target column. Add a LabelField or TargetField for whatever "
            "you want to predict, or set Meta.target on the model."
        )
        return

    column = result.get(result.target)
    if column is None:
        return

    named_conventionally = column.name.lower() in TARGET_NAMES
    if column.kwargs.get("choices"):
        classes = column.kwargs.pop("choices")
        column.kwargs.pop("max_length", None)
        column.field_class = "LabelField"
        column.kwargs = {"classes": classes, **column.kwargs}
        if not named_conventionally:
            column.note = "guessed as the target because it is the last categorical column"
    elif column.field_class in {"IntegerField", "FloatField"}:
        column.field_class = "TargetField"
        column.kwargs.pop("min_value", None)
        column.kwargs.pop("max_value", None)
        column.note = "a continuous target; use LabelField instead if it is categorical"
    else:
        # A free-text or datetime column named `label` is not a usable target.
        result.warnings.append(
            f"Column {column.name!r} looks like a target by name but its values are "
            f"{column.field_class}, so it was left as-is."
        )
        result.target = None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_declaration(
    result: DataProfile,
    *,
    name: str,
    source_expr: str,
    app: str | None = None,
) -> str:
    """The Python for a ``Dataset`` matching ``result``."""
    field_names = sorted({c.field_class for c in result.columns})
    matched = re.match(r"(\w+)", source_expr)
    source_class = matched.group(1) if matched else "JSONLSource"

    if result.rows_total is None:
        scale = f"{len(result.columns)} columns, inferred from {result.rows_sampled} rows"
    elif result.rows_total > result.rows_sampled:
        scale = (
            f"{result.rows_total} rows, {len(result.columns)} columns "
            f"(types inferred from the first {result.rows_sampled})"
        )
    else:
        scale = f"{result.rows_total} rows, {len(result.columns)} columns"

    lines = [
        f'"""Datasets{f" for the {app} app" if app else ""}.',
        "",
        "Generated by `manage.py inspectdata` from a sample of the file. Read it",
        "before you keep it: the types are inferred, not declared.",
        '"""',
        "",
        # Wrapped to the project's own line length, so the generated file passes
        # the same ruff configuration as everything else.
        *_render_import("from mlango.core.fields import", field_names),
        f"from mlango.data import Dataset, {source_class}",
        "",
        "",
        f"class {name}(Dataset):",
        f'    """{scale}."""',
        "",
    ]

    for column in result.columns:
        rendered = f"    {column.name} = {column.field_class}({_render_kwargs(column.kwargs)})"
        if column.note:
            lines.append(f"    # {column.note}")
        lines.append(rendered)

    lines += ["", "    class Meta:", f"        source = {source_expr}"]
    if result.primary_key:
        lines.append(f'        primary_key = "{result.primary_key}"')
    else:
        lines.append("        # No unique column found. Splits fall back to hashing the whole")
        lines.append("        # record, which is stable but slower.")

    if result.warnings:
        lines.append("")
        lines.append("")
        lines.append("# Worth checking:")
        for warning in result.warnings:
            for chunk in _wrap(warning, 76):
                lines.append(f"#   {chunk}")

    return "\n".join(lines) + "\n"


def _render_import(prefix: str, names: list[str], limit: int = 96) -> list[str]:
    """One line if it fits, a parenthesised block if it does not."""
    single = f"{prefix} {', '.join(names)}"
    if len(single) <= limit:
        return [single]
    return [f"{prefix} (", *(f"    {n}," for n in names), ")"]


def _render_kwargs(kwargs: dict[str, Any]) -> str:
    parts = []
    for key, value in kwargs.items():
        if key == "classes":
            parts.append(_literal(value))  # positional on LabelField
        else:
            parts.append(f"{key}={_literal(value)}")
    return ", ".join(parts)


def _literal(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, list):
        return "[" + ", ".join(_literal(v) for v in value) + "]"
    return repr(value)


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width) or [text]


# --------------------------------------------------------------------------- #
# Predicates
# --------------------------------------------------------------------------- #


def _all(values: list[Any], predicate: Any) -> bool:
    return all(predicate(v) for v in values)


def _all_bool(values: list[Any]) -> bool:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
            continue
        return False
    # A column of 0s and 1s is more useful as an integer than as a boolean:
    # counts and ids look like this too, and IntegerField keeps arithmetic.
    return any(isinstance(v, bool) or str(v).strip().lower() not in {"0", "1"} for v in values)


def _is_int(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, str) and bool(_INT_RE.match(value.strip()))


def _is_float(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if not isinstance(value, str):
        return False
    try:
        float(value.strip())
    except ValueError:
        return False
    return True


def _is_datetime(value: Any) -> bool:
    if isinstance(value, (_dt.datetime, _dt.date)):
        return True
    if not isinstance(value, str):
        return False
    try:
        _dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _is_json_container(value: Any) -> bool:
    if isinstance(value, (dict, list)):
        return True
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or text[0] not in "{[":
        return False
    try:
        return isinstance(json.loads(text), (dict, list))
    except json.JSONDecodeError:
        return False


def _round(value: float) -> float:
    return round(value, 6)


def _length_cap(observed: int) -> int:
    """A max_length with headroom, so a slightly longer value later still fits."""
    for cap in (16, 32, 64, 128, 255, 512, 1024):
        if observed <= cap:
            return cap
    return observed * 2


__all__ = [
    "ColumnProfile",
    "DataProfile",
    "profile_source",
    "render_declaration",
    "source_for",
]
