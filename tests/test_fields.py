"""Field validation, coercion and deconstruction."""

from __future__ import annotations

import pytest

from mlango.core import fields
from mlango.core.exceptions import ValidationError


class TestCoercion:
    def test_integer_accepts_numeric_strings(self):
        assert fields.IntegerField().clean("42") == 42

    def test_float_accepts_integers(self):
        assert fields.FloatField().clean(3) == 3.0

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("true", True), ("no", False), ("1", True), ("off", False), (1, True)],
    )
    def test_boolean_accepts_common_spellings(self, raw, expected):
        assert fields.BooleanField().clean(raw) is expected

    def test_boolean_rejects_nonsense(self):
        with pytest.raises(ValidationError):
            fields.BooleanField().clean("maybe")

    def test_json_parses_strings(self):
        assert fields.JSONField().clean('{"a": 1}') == {"a": 1}

    def test_datetime_parses_iso(self):
        value = fields.DateTimeField().clean("2026-07-29T12:30:00")
        assert (value.year, value.month, value.hour) == (2026, 7, 12)


class TestValidation:
    def test_required_field_rejects_none(self):
        with pytest.raises(ValidationError):
            fields.TextField().clean(None)

    def test_null_field_accepts_none(self):
        assert fields.TextField(null=True).clean(None) is None

    def test_default_makes_field_optional(self):
        field = fields.IntegerField(default=7)
        assert field.required is False
        assert field.get_default() == 7

    def test_numeric_bounds(self):
        field = fields.IntegerField(min_value=1, max_value=5)
        assert field.clean(3) == 3
        with pytest.raises(ValidationError, match="below the minimum"):
            field.clean(0)
        with pytest.raises(ValidationError, match="above the maximum"):
            field.clean(6)

    def test_text_length_bounds(self):
        with pytest.raises(ValidationError, match="longer than"):
            fields.TextField(max_length=3).clean("abcd")
        with pytest.raises(ValidationError, match="shorter than"):
            fields.TextField(min_length=3).clean("ab")

    def test_choices_are_enforced(self):
        with pytest.raises(ValidationError, match="not one of"):
            fields.ChoiceField(["a", "b"]).clean("c")

    def test_callable_default_is_called_each_time(self):
        counter = iter(range(10))
        field = fields.IntegerField(default=lambda: next(counter))
        assert field.get_default() == 0
        assert field.get_default() == 1


class TestLabelField:
    def test_unknown_label_is_rejected(self):
        field = fields.LabelField(["neg", "pos"])
        field.name = "label"
        with pytest.raises(ValidationError, match="Unknown label"):
            field.clean("neutral")

    def test_index_of_gives_stable_encoding(self):
        field = fields.LabelField(["neg", "pos"])
        assert field.index_of("neg") == 0
        assert field.index_of("pos") == 1

    def test_multi_wraps_a_single_value(self):
        field = fields.LabelField(["a", "b"], multi=True)
        assert field.clean("a") == ["a"]

    def test_is_marked_as_a_target(self):
        assert fields.LabelField(["a"]).is_target is True


class TestTensorFields:
    def test_shape_is_enforced(self):
        field = fields.TensorField(shape=(2, 3))
        field.name = "x"
        assert field.clean([[1, 2, 3], [4, 5, 6]]).shape == (2, 3)
        with pytest.raises(ValidationError, match="Expected shape"):
            field.clean([[1, 2], [3, 4]])

    def test_free_dimension_accepts_any_length(self):
        field = fields.TensorField(shape=(None, 2))
        assert field.clean([[1, 2], [3, 4], [5, 6]]).shape == (3, 2)

    def test_embedding_reports_its_dimension(self):
        field = fields.EmbeddingField(8)
        assert field.describe()["dim"] == 8
        assert field.clean(list(range(8))).shape == (8,)


class TestFileFields:
    def test_extension_is_checked(self):
        field = fields.ImageField()
        field.name = "photo"
        assert field.clean("cat.png") == "cat.png"
        with pytest.raises(ValidationError, match="extensions"):
            field.clean("cat.txt")


class TestDeconstruct:
    def test_round_trips_through_kwargs(self):
        field = fields.IntegerField(default=5, min_value=0, tunable=True)
        _name, path, args, kwargs = field.deconstruct()
        assert path == "mlango.core.fields.IntegerField"
        assert args == ()
        rebuilt = fields.IntegerField(*args, **kwargs)
        assert rebuilt.get_default() == 5
        assert rebuilt.min_value == 0
        assert rebuilt.tunable is True

    def test_label_field_puts_classes_in_args(self):
        _name, _path, args, kwargs = fields.LabelField(["a", "b"]).deconstruct()
        assert args == (["a", "b"],)
        assert "choices" not in kwargs

    def test_declaration_order_is_preserved(self):
        first = fields.TextField()
        second = fields.TextField()
        assert first.creation_counter < second.creation_counter
