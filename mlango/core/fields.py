"""The field system.

One field system serves four jobs, which is what keeps the framework small:

* the **schema of a dataset record** (``text = TextField()``),
* the **hyperparameters of a model** (``learning_rate = FloatField(default=1e-3)``),
* the **configuration of an agent** (``temperature = FloatField(default=0.7)``),
* the **input contract of an inference endpoint** (validation on the way in).

Fields know how to validate a value, coerce it to Python, serialise themselves
for a schema fingerprint, and deconstruct themselves into ``(path, args,
kwargs)`` so the migration writer can render them back into source code.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Iterable, Sequence
from typing import Any

from mlango.core.exceptions import FieldError, ValidationError


class NotProvided:
    """Sentinel distinguishing "no default" from "default is None"."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<not provided>"

    def __bool__(self) -> bool:
        return False


NOT_PROVIDED = NotProvided()


class Field:
    """Base class for every declarative field."""

    #: Incremented on every instantiation so declaration order survives the
    #: unordered class namespace, the same trick Django uses.
    creation_counter = 0

    #: Reported in schemas and the admin.
    kind = "field"
    python_type: type | tuple[type, ...] = object
    #: Fields that describe a prediction target rather than an input.
    is_target = False

    def __init__(
        self,
        *,
        default: Any = NOT_PROVIDED,
        null: bool = False,
        required: bool | None = None,
        choices: Sequence[Any] | None = None,
        help_text: str = "",
        verbose_name: str | None = None,
        validators: Iterable[Any] = (),
        editable: bool = True,
        tunable: bool = False,
    ):
        self.default = default
        self.null = null
        # A field with a default is optional unless the caller insists.
        self.required = (default is NOT_PROVIDED) if required is None else required
        self.choices = list(choices) if choices is not None else None
        self.help_text = help_text
        self._verbose_name = verbose_name
        self.validators = list(validators)
        self.editable = editable
        #: Marks a hyperparameter a sweep is allowed to vary.
        self.tunable = tunable

        self.name: str | None = None
        self.owner: type | None = None

        self.creation_counter = Field.creation_counter
        Field.creation_counter += 1

    # -- descriptor protocol -------------------------------------------------

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.owner = owner

    @property
    def verbose_name(self) -> str:
        if self._verbose_name:
            return self._verbose_name
        return (self.name or "").replace("_", " ").strip().capitalize()

    @property
    def attname(self) -> str:
        if self.name is None:  # pragma: no cover - guarded by the metaclass
            raise FieldError("Field is not bound to a class yet.")
        return self.name

    # -- values --------------------------------------------------------------

    def has_default(self) -> bool:
        return not isinstance(self.default, NotProvided)

    def get_default(self) -> Any:
        if not self.has_default():
            return None
        if callable(self.default):
            return self.default()
        return self.default

    def to_python(self, value: Any) -> Any:
        """Coerce an external value into the field's Python type."""
        return value

    def clean(self, value: Any) -> Any:
        """Coerce then validate, returning the cleaned value."""
        value = self.to_python(value)
        self.validate(value)
        return value

    def validate(self, value: Any) -> None:
        if value is None:
            if self.null or not self.required:
                return
            raise ValidationError("This field cannot be null.", field=self.name)
        if self.choices is not None and value not in self.choices:
            raise ValidationError(
                f"{value!r} is not one of the available choices: {self.choices!r}.",
                field=self.name,
            )
        for validator in self.validators:
            validator(value)

    # -- introspection -------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """A JSON-safe description used for schemas and fingerprints."""
        info: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "class": type(self).__name__,
            "null": self.null,
            "required": self.required,
        }
        if self.has_default():
            default = self.get_default()
            info["default"] = default if _json_safe(default) else repr(default)
        if self.choices is not None:
            info["choices"] = list(self.choices)
        if self.help_text:
            info["help_text"] = self.help_text
        if self.is_target:
            info["target"] = True
        info.update(self.extra_describe())
        return info

    def extra_describe(self) -> dict[str, Any]:
        """Hook for subclasses to add their own descriptors."""
        return {}

    def deconstruct(self) -> tuple[str, str, tuple, dict[str, Any]]:
        """Return ``(name, path, args, kwargs)`` for the migration writer."""
        path = f"{type(self).__module__}.{type(self).__qualname__}"
        kwargs: dict[str, Any] = {}
        if self.has_default() and not callable(self.default):
            kwargs["default"] = self.default
        if self.null:
            kwargs["null"] = True
        if self.choices is not None:
            kwargs["choices"] = list(self.choices)
        if self.help_text:
            kwargs["help_text"] = self.help_text
        if self.tunable:
            kwargs["tunable"] = True
        kwargs.update(self.extra_deconstruct())
        return self.name or "", path, (), kwargs

    def extra_deconstruct(self) -> dict[str, Any]:
        return {}

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.name or '(unbound)'}>"


def _json_safe(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


# --------------------------------------------------------------------------- #
# Scalars
# --------------------------------------------------------------------------- #


class BooleanField(Field):
    kind = "boolean"
    python_type = bool

    _TRUE = {"true", "1", "yes", "y", "on", "t"}
    _FALSE = {"false", "0", "no", "n", "off", "f"}

    def to_python(self, value: Any) -> Any:
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in self._TRUE:
                return True
            if lowered in self._FALSE:
                return False
        raise ValidationError(f"{value!r} is not a valid boolean.", field=self.name)


class _NumericField(Field):
    def __init__(self, *, min_value: Any = None, max_value: Any = None, **kwargs: Any):
        self.min_value = min_value
        self.max_value = max_value
        super().__init__(**kwargs)

    def validate(self, value: Any) -> None:
        super().validate(value)
        if value is None:
            return
        if self.min_value is not None and value < self.min_value:
            raise ValidationError(
                f"Value {value} is below the minimum of {self.min_value}.", field=self.name
            )
        if self.max_value is not None and value > self.max_value:
            raise ValidationError(
                f"Value {value} is above the maximum of {self.max_value}.", field=self.name
            )

    def extra_describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {}
        if self.min_value is not None:
            info["min_value"] = self.min_value
        if self.max_value is not None:
            info["max_value"] = self.max_value
        return info

    def extra_deconstruct(self) -> dict[str, Any]:
        return self.extra_describe()


class IntegerField(_NumericField):
    kind = "integer"
    python_type = int

    def to_python(self, value: Any) -> Any:
        if value is None or isinstance(value, int) and not isinstance(value, bool):
            return value
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{value!r} is not a valid integer.", field=self.name) from exc


class FloatField(_NumericField):
    kind = "float"
    python_type = float

    def to_python(self, value: Any) -> Any:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{value!r} is not a valid float.", field=self.name) from exc


class TextField(Field):
    kind = "text"
    python_type = str

    def __init__(self, *, max_length: int | None = None, min_length: int = 0, **kwargs: Any):
        self.max_length = max_length
        self.min_length = min_length
        super().__init__(**kwargs)

    def to_python(self, value: Any) -> Any:
        if value is None or isinstance(value, str):
            return value
        return str(value)

    def validate(self, value: Any) -> None:
        super().validate(value)
        if value is None:
            return
        if self.max_length is not None and len(value) > self.max_length:
            raise ValidationError(
                f"Text is {len(value)} characters, longer than the {self.max_length} allowed.",
                field=self.name,
            )
        if len(value) < self.min_length:
            raise ValidationError(
                f"Text is {len(value)} characters, shorter than the {self.min_length} required.",
                field=self.name,
            )

    def extra_describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {}
        if self.max_length is not None:
            info["max_length"] = self.max_length
        if self.min_length:
            info["min_length"] = self.min_length
        return info

    def extra_deconstruct(self) -> dict[str, Any]:
        return self.extra_describe()


class CharField(TextField):
    """A TextField that insists on a length limit."""

    kind = "char"

    def __init__(self, *, max_length: int = 255, **kwargs: Any):
        super().__init__(max_length=max_length, **kwargs)


class ChoiceField(Field):
    kind = "choice"

    def __init__(self, choices: Sequence[Any], **kwargs: Any):
        super().__init__(choices=choices, **kwargs)

    def deconstruct(self):
        name, path, _args, kwargs = super().deconstruct()
        choices = kwargs.pop("choices", list(self.choices or []))
        return name, path, (choices,), kwargs


class DateTimeField(Field):
    kind = "datetime"
    python_type = _dt.datetime

    def to_python(self, value: Any) -> Any:
        if value is None or isinstance(value, _dt.datetime):
            return value
        if isinstance(value, str):
            try:
                return _dt.datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValidationError(
                    f"{value!r} is not an ISO-8601 datetime.", field=self.name
                ) from exc
        raise ValidationError(f"{value!r} is not a valid datetime.", field=self.name)


class JSONField(Field):
    kind = "json"
    python_type = (dict, list)

    def to_python(self, value: Any) -> Any:
        if value is None or isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"Invalid JSON: {exc}", field=self.name) from exc
        raise ValidationError(f"{value!r} is not valid JSON data.", field=self.name)


# --------------------------------------------------------------------------- #
# ML-specific fields
# --------------------------------------------------------------------------- #


class LabelField(Field):
    """A categorical prediction target.

    ``classes`` doubles as the class ordering used when encoding to indices,
    so training and serving agree on what index 0 means.
    """

    kind = "label"
    is_target = True

    def __init__(self, classes: Sequence[Any] | None = None, *, multi: bool = False, **kwargs: Any):
        self.classes = list(classes) if classes is not None else None
        self.multi = multi
        super().__init__(**kwargs)

    def to_python(self, value: Any) -> Any:
        if value is None:
            return None
        if self.multi and not isinstance(value, (list, tuple, set)):
            return [value]
        return value

    def validate(self, value: Any) -> None:
        super().validate(value)
        if value is None or self.classes is None:
            return
        values = value if self.multi else [value]
        unknown = [v for v in values if v not in self.classes]
        if unknown:
            raise ValidationError(
                f"Unknown label(s) {unknown!r}; expected one of {self.classes!r}.",
                field=self.name,
            )

    def index_of(self, value: Any) -> int:
        if self.classes is None:
            raise FieldError(f"{self.name}: classes must be declared to encode labels.")
        return self.classes.index(value)

    def extra_describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {"multi": self.multi}
        if self.classes is not None:
            info["classes"] = list(self.classes)
            info["n_classes"] = len(self.classes)
        return info

    def deconstruct(self):
        name, path, _args, kwargs = super().deconstruct()
        kwargs.pop("choices", None)
        if self.multi:
            kwargs["multi"] = True
        args = (list(self.classes),) if self.classes is not None else ()
        return name, path, args, kwargs


class TargetField(FloatField):
    """A continuous prediction target (regression)."""

    kind = "target"
    is_target = True


class TensorField(Field):
    """A dense numeric array of a declared shape.

    ``shape`` may contain ``None`` for a free dimension, e.g. ``(None, 128)``
    for a variable-length sequence of 128-dim vectors.
    """

    kind = "tensor"

    def __init__(
        self,
        *,
        shape: Sequence[int | None] | None = None,
        dtype: str = "float32",
        **kwargs: Any,
    ):
        self.shape = tuple(shape) if shape is not None else None
        self.dtype = dtype
        super().__init__(**kwargs)

    def to_python(self, value: Any) -> Any:
        if value is None:
            return None
        import numpy as np

        arr = value if isinstance(value, np.ndarray) else np.asarray(value, dtype=self.dtype)
        return arr

    def validate(self, value: Any) -> None:
        super().validate(value)
        if value is None or self.shape is None:
            return
        actual = tuple(getattr(value, "shape", ()))
        if len(actual) != len(self.shape):
            raise ValidationError(
                f"Expected a {len(self.shape)}-D tensor, got shape {actual}.", field=self.name
            )
        for expected, got in zip(self.shape, actual, strict=True):
            if expected is not None and expected != got:
                raise ValidationError(
                    f"Expected shape {self.shape}, got {actual}.", field=self.name
                )

    def extra_describe(self) -> dict[str, Any]:
        return {"shape": list(self.shape) if self.shape else None, "dtype": self.dtype}

    def extra_deconstruct(self) -> dict[str, Any]:
        info: dict[str, Any] = {"dtype": self.dtype}
        if self.shape is not None:
            info["shape"] = list(self.shape)
        return info


class EmbeddingField(TensorField):
    """A fixed-width vector, typically produced by an encoder."""

    kind = "embedding"

    def __init__(self, dim: int, **kwargs: Any):
        self.dim = dim
        kwargs.setdefault("dtype", "float32")
        super().__init__(shape=(dim,), **kwargs)

    def extra_describe(self) -> dict[str, Any]:
        info = super().extra_describe()
        info["dim"] = self.dim
        return info

    def deconstruct(self):
        name, path, _args, kwargs = super().deconstruct()
        kwargs.pop("shape", None)
        return name, path, (self.dim,), kwargs


class FileField(Field):
    """A path to a file in the configured artifact storage."""

    kind = "file"
    python_type = str

    def __init__(self, *, extensions: Sequence[str] | None = None, **kwargs: Any):
        self.extensions = [e.lower().lstrip(".") for e in extensions] if extensions else None
        super().__init__(**kwargs)

    def validate(self, value: Any) -> None:
        super().validate(value)
        if value is None or self.extensions is None:
            return
        suffix = str(value).rsplit(".", 1)[-1].lower()
        if suffix not in self.extensions:
            raise ValidationError(
                f"{value!r} does not have one of the allowed extensions {self.extensions!r}.",
                field=self.name,
            )

    def extra_describe(self) -> dict[str, Any]:
        return {"extensions": self.extensions} if self.extensions else {}

    def extra_deconstruct(self) -> dict[str, Any]:
        return self.extra_describe()


class ImageField(FileField):
    kind = "image"

    def __init__(self, *, size: tuple[int, int] | None = None, mode: str = "RGB", **kwargs: Any):
        self.size = tuple(size) if size else None
        self.mode = mode
        kwargs.setdefault("extensions", ["png", "jpg", "jpeg", "webp", "bmp"])
        super().__init__(**kwargs)

    def extra_describe(self) -> dict[str, Any]:
        info = super().extra_describe()
        info.update({"size": list(self.size) if self.size else None, "mode": self.mode})
        return info

    def extra_deconstruct(self) -> dict[str, Any]:
        info: dict[str, Any] = {"mode": self.mode}
        if self.size:
            info["size"] = list(self.size)
        return info


class AudioField(FileField):
    kind = "audio"

    def __init__(self, *, sample_rate: int | None = None, **kwargs: Any):
        self.sample_rate = sample_rate
        kwargs.setdefault("extensions", ["wav", "mp3", "flac", "ogg"])
        super().__init__(**kwargs)

    def extra_describe(self) -> dict[str, Any]:
        info = super().extra_describe()
        info["sample_rate"] = self.sample_rate
        return info

    def extra_deconstruct(self) -> dict[str, Any]:
        return {"sample_rate": self.sample_rate} if self.sample_rate else {}


__all__ = [
    "NOT_PROVIDED",
    "NotProvided",
    "Field",
    "BooleanField",
    "IntegerField",
    "FloatField",
    "TextField",
    "CharField",
    "ChoiceField",
    "DateTimeField",
    "JSONField",
    "LabelField",
    "TargetField",
    "TensorField",
    "EmbeddingField",
    "FileField",
    "ImageField",
    "AudioField",
]
