"""The declarative machinery shared by datasets, models, agents and evals.

Writing::

    class Reviews(Dataset):
        text = TextField()
        label = LabelField(["neg", "pos"])

        class Meta:
            source = JSONLSource("data/reviews.jsonl")

should feel exactly like writing a Django model, and for the same reasons: the
class body is a *declaration*, the metaclass turns it into metadata, and the
rest of the framework reads that metadata instead of asking the user to repeat
themselves in a config file.
"""

from __future__ import annotations

from typing import Any

from mlango.core.exceptions import ValidationError
from mlango.core.fields import NOT_PROVIDED, Field
from mlango.core.options import Options


class FieldDescriptor:
    """Class access returns the :class:`Field`; instance access returns the value."""

    __slots__ = ("field", "name")

    def __init__(self, field: Field):
        self.field = field
        self.name = field.name or ""

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self.field
        try:
            return instance._values[self.name]
        except KeyError:
            value = self.field.get_default()
            instance._values[self.name] = value
            return value

    def __set__(self, instance: Any, value: Any) -> None:
        # Store raw. Coercion and validation happen in ``full_clean`` so that a
        # half-built object is never rejected mid-construction.
        instance._values[self.name] = value

    def __delete__(self, instance: Any) -> None:
        instance._values.pop(self.name, None)


class DeclarativeMeta(type):
    """Collects fields, builds ``_meta`` and registers the class."""

    #: Overridden by each family's base class.
    _kind: str = "object"
    #: Meta options this family understands, on top of the common ones.
    _meta_options: tuple[str, ...] = ()

    def __new__(mcls, name: str, bases: tuple[type, ...], attrs: dict[str, Any], **kwargs: Any):
        parents = [b for b in bases if isinstance(b, DeclarativeMeta)]
        if not parents:
            # ``Declarative`` itself — nothing to introspect yet.
            return super().__new__(mcls, name, bases, attrs)

        module = attrs.get("__module__", "")
        meta_class = attrs.pop("Meta", None)

        local_fields: list[Field] = []
        body: dict[str, Any] = {}
        for key, value in attrs.items():
            if isinstance(value, Field):
                value.name = value.name or key
                local_fields.append(value)
            else:
                body[key] = value

        cls = super().__new__(mcls, name, bases, body, **kwargs)

        kind = body.get("_kind") or next(
            (getattr(b, "_kind") for b in cls.__mro__[1:] if getattr(b, "_kind", None)), "object"
        )
        allowed = tuple(
            dict.fromkeys(
                sum(
                    (tuple(getattr(b, "_meta_options", ())) for b in reversed(cls.__mro__)),
                    (),
                )
            )
        )

        opts = Options(meta_class, name, module, allowed)
        opts.kind = kind

        # Inherited fields first (base declaration order), then local ones.
        merged: dict[str, Field] = {}
        for base in reversed(cls.__mro__[1:]):
            base_meta = getattr(base, "_meta", None)
            if base_meta is None:
                continue
            for field in base_meta.fields:
                if field.name:
                    merged[field.name] = field
        for field in sorted(local_fields, key=lambda f: f.creation_counter):
            if field.name:
                merged[field.name] = field

        opts.local_fields = sorted(local_fields, key=lambda f: f.creation_counter)
        opts.contribute_fields(list(merged.values()))
        opts.bind(cls)

        if opts.app_label is None and not opts.abstract:
            opts.app_label = _resolve_app_label(module)

        cls._meta = opts  # type: ignore[attr-defined]

        for field in opts.fields:
            field.__set_name__(cls, field.name or "")
            setattr(cls, field.name or "", FieldDescriptor(field))

        prepare = getattr(cls, "_prepare", None)
        if callable(prepare):
            prepare()

        if not opts.abstract:
            from mlango.core.registry import apps

            apps.register(kind, cls)

        return cls

    # Nice repr in tracebacks and the shell.
    def __repr__(cls) -> str:
        meta = getattr(cls, "_meta", None)
        return f"<{cls.__name__}>" if meta is None else f"<{meta.kind}: {meta.label}>"


def _resolve_app_label(module: str) -> str:
    from mlango.core.registry import apps

    config = apps.get_containing_app_config(module)
    if config is not None:
        return config.label
    # Declared outside any installed app (a script, a notebook, a test): fall
    # back to the top-level package so the object still has a stable label.
    return module.split(".")[0] or "__main__"


class Declarative(metaclass=DeclarativeMeta):
    """Common behaviour for every declared object.

    Subclasses are instantiated with keyword arguments matching their fields;
    unknown keywords are an error, because silently ignoring a typo'd
    hyperparameter is how experiments become unreproducible.
    """

    _kind = "object"
    _meta_options: tuple[str, ...] = ()
    _meta: Options

    def __init__(self, **kwargs: Any):
        object.__setattr__(self, "_values", {})
        opts = type(self)._meta
        unknown = sorted(set(kwargs) - set(opts.field_names))
        if unknown:
            raise TypeError(
                f"{opts.label}() got unexpected keyword argument(s): {', '.join(unknown)}. "
                f"Declared fields: {', '.join(opts.field_names) or '(none)'}."
            )
        for field in opts.fields:
            name = field.name or ""
            if name in kwargs:
                self._values[name] = kwargs[name]
            elif field.has_default():
                self._values[name] = field.get_default()

    # -- values --------------------------------------------------------------

    def to_dict(self, *, include_missing: bool = False) -> dict[str, Any]:
        opts = type(self)._meta
        out: dict[str, Any] = {}
        for field in opts.fields:
            name = field.name or ""
            if name in self._values:
                out[name] = self._values[name]
            elif include_missing:
                out[name] = field.get_default()
        return out

    def full_clean(self) -> dict[str, Any]:
        """Coerce and validate every field, raising a combined ValidationError."""
        opts = type(self)._meta
        cleaned: dict[str, Any] = {}
        errors: dict[str, list[str]] = {}
        for field in opts.fields:
            name = field.name or ""
            raw = self._values.get(name, NOT_PROVIDED)
            if isinstance(raw, type(NOT_PROVIDED)) or raw is NOT_PROVIDED:
                raw = field.get_default()
            try:
                cleaned[name] = field.clean(raw)
            except ValidationError as exc:
                for key, messages in exc.errors.items():
                    errors.setdefault(key if key != "__all__" else name, []).extend(messages)
        if errors:
            raise ValidationError(errors)
        self._values.update(cleaned)
        return cleaned

    @classmethod
    def clean_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate an external mapping against the declaration.

        Used by the inference API and by dataset loading, so a record from a
        request body and a record from a JSONL file get the exact same checks.
        """
        instance = cls(**{k: v for k, v in payload.items() if cls._meta.has_field(k)})
        unknown = sorted(set(payload) - set(cls._meta.field_names))
        if unknown:
            raise ValidationError(
                {"__all__": [f"Unknown field(s): {', '.join(unknown)}"]}
            )
        return instance.full_clean()

    # -- identity ------------------------------------------------------------

    @classmethod
    def label(cls) -> str:
        return cls._meta.label

    @classmethod
    def fingerprint(cls) -> str:
        return cls._meta.fingerprint()

    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash((type(self).__name__, tuple(sorted(map(str, self.to_dict().items())))))

    def __repr__(self) -> str:
        values = ", ".join(f"{k}={v!r}" for k, v in list(self.to_dict().items())[:4])
        return f"{type(self).__name__}({values})"
