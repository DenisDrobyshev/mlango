"""The metaclass, ``_meta`` and the app registry."""

from __future__ import annotations

import pytest

from mlango.core import fields
from mlango.core.base import Declarative
from mlango.core.exceptions import FieldError, ImproperlyConfigured, ValidationError


class Thing(Declarative):
    """A thing worth declaring."""

    _kind = "dataset"
    _meta_options = ("flavour",)

    name = fields.TextField(max_length=10)
    size = fields.IntegerField(default=3, min_value=0)
    label = fields.LabelField(["a", "b"])

    class Meta:
        flavour = "vanilla"
        verbose_name = "widget"


class TestMeta:
    def test_fields_are_collected_in_declaration_order(self):
        assert Thing._meta.field_names == ["name", "size", "label"]

    def test_fields_leave_the_class_namespace(self):
        # Class access returns the Field itself, not a value.
        assert isinstance(Thing.name, fields.TextField)

    def test_target_and_input_fields_are_separated(self):
        assert [f.name for f in Thing._meta.target_fields] == ["label"]
        assert [f.name for f in Thing._meta.input_fields] == ["name", "size"]

    def test_kind_is_recorded(self):
        assert Thing._meta.kind == "dataset"

    def test_meta_extras_are_captured(self):
        assert Thing._meta.extras["flavour"] == "vanilla"

    def test_verbose_name_is_honoured(self):
        assert Thing._meta.verbose_name == "widget"
        assert Thing._meta.verbose_name_plural == "widgets"

    def test_description_comes_from_the_docstring(self):
        assert Thing._meta.description == "A thing worth declaring."

    def test_get_field_reports_what_is_available(self):
        with pytest.raises(FieldError, match="Available: name, size, label"):
            Thing._meta.get_field("nope")

    def test_unknown_meta_option_is_rejected(self):
        with pytest.raises(ImproperlyConfigured, match="unknown option"):

            class Bad(Declarative):
                _kind = "dataset"

                class Meta:
                    definitely_not_an_option = 1


class TestPluralisation:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Review", "reviews"),
            # Already plural: must not become "reviewses".
            ("Reviews", "reviews"),
            ("Box", "boxes"),
            ("Category", "categories"),
            ("Survey", "surveys"),
        ],
    )
    def test_plural_forms(self, name, expected):
        built = type(name, (Declarative,), {"_kind": "dataset", "__module__": __name__})
        assert built._meta.verbose_name_plural == expected


class TestInstances:
    def test_defaults_are_applied(self):
        assert Thing(name="x", label="a").size == 3

    def test_values_round_trip(self):
        thing = Thing(name="x", size=9, label="b")
        assert thing.to_dict() == {"name": "x", "size": 9, "label": "b"}

    def test_unknown_keyword_is_an_error(self):
        with pytest.raises(TypeError, match="unexpected keyword"):
            Thing(name="x", label="a", nope=1)

    def test_full_clean_reports_every_field_at_once(self):
        with pytest.raises(ValidationError) as info:
            Thing(name="x" * 20, label="zzz").full_clean()
        assert set(info.value.errors) == {"name", "label"}

    def test_full_clean_coerces(self):
        cleaned = Thing(name="x", size="7", label="a").full_clean()
        assert cleaned["size"] == 7

    def test_clean_payload_rejects_unknown_keys(self):
        with pytest.raises(ValidationError, match="Unknown field"):
            Thing.clean_payload({"name": "x", "label": "a", "extra": 1})


class TestFingerprint:
    def test_is_stable_across_calls(self):
        assert Thing.fingerprint() == Thing.fingerprint()

    def test_changes_when_a_field_changes(self):
        class Before(Declarative):
            _kind = "dataset"
            text = fields.TextField()

        class After(Declarative):
            _kind = "dataset"
            text = fields.TextField(max_length=100)

        assert Before.fingerprint() != After.fingerprint()


class TestInheritance:
    def test_abstract_bases_contribute_fields(self):
        class Base(Declarative):
            _kind = "dataset"
            created_at = fields.DateTimeField(null=True)

            class Meta:
                abstract = True

        class Child(Base):
            name = fields.TextField()

        assert Child._meta.field_names == ["created_at", "name"]

    def test_abstract_classes_are_not_registered(self):
        from mlango.core.registry import apps

        labels = [c._meta.label for c in apps.get_registered("dataset")]
        assert not any(label.endswith(".Base") for label in labels)


class TestRegistry:
    def test_declared_classes_are_findable(self):
        from mlango.core.registry import apps

        assert apps.get_dataset("Thing") is Thing

    def test_lookup_error_lists_alternatives(self):
        from mlango.core.registry import apps

        with pytest.raises(LookupError, match="Registered datasets"):
            apps.get_dataset("NotDeclaredAnywhere")

    def test_find_searches_every_kind(self):
        from mlango.core.registry import apps

        kind, found = apps.find("Thing")
        assert (kind, found) == ("dataset", Thing)
