"""Schema inference: what `manage.py inspectdata` decides, and why.

Every guess this module makes ends up in a file someone keeps, so the rules are
pinned here rather than left to a reviewer's eye.
"""

from __future__ import annotations

import json

import pytest

from mlango.core.exceptions import ImproperlyConfigured
from mlango.data.inspect import (
    profile_source,
    render_declaration,
    source_for,
)
from mlango.data.sources import InMemorySource


def profile(rows, **kwargs):
    return profile_source(InMemorySource(rows), **kwargs)


def field_for(rows, column="value"):
    return profile(rows).get(column)


# --------------------------------------------------------------------------- #
# Type inference
# --------------------------------------------------------------------------- #


class TestNumbers:
    def test_integers_carry_their_range(self):
        column = field_for([{"value": 1}, {"value": 7}, {"value": 3}])
        assert column.field_class == "IntegerField"
        assert column.kwargs == {"min_value": 1, "max_value": 7}

    def test_integers_written_as_strings_still_count(self):
        """CSV hands everything over as text."""
        column = field_for([{"value": "1"}, {"value": "-4"}])
        assert column.field_class == "IntegerField"
        assert column.kwargs["min_value"] == -4

    def test_floats(self):
        column = field_for([{"value": "1.5"}, {"value": "2.25"}])
        assert column.field_class == "FloatField"
        assert column.kwargs == {"min_value": 1.5, "max_value": 2.25}

    def test_one_float_makes_the_whole_column_a_float(self):
        column = field_for([{"value": "1"}, {"value": "2.5"}, {"value": "3"}])
        assert column.field_class == "FloatField"

    def test_a_float_range_is_rounded_so_the_output_stays_readable(self):
        column = field_for([{"value": 1 / 3}, {"value": 2 / 3}])
        assert column.kwargs["min_value"] == 0.333333

    def test_zeros_and_ones_stay_integers(self):
        """Counts and flags look alike; IntegerField keeps arithmetic working."""
        column = field_for([{"value": 0}, {"value": 1}, {"value": 1}])
        assert column.field_class == "IntegerField"


class TestBooleans:
    @pytest.mark.parametrize(
        "values",
        [
            [True, False],
            ["true", "false"],
            ["yes", "no"],
            ["Y", "N"],
            ["t", "f"],
        ],
    )
    def test_boolean_spellings(self, values):
        column = field_for([{"value": v} for v in values])
        assert column.field_class == "BooleanField"

    def test_a_real_bool_survives_alongside_ones(self):
        column = field_for([{"value": True}, {"value": False}, {"value": True}])
        assert column.field_class == "BooleanField"


class TestStrings:
    def test_a_low_cardinality_column_records_its_choices(self):
        # A second column so the target fallback does not claim this one.
        rows = [
            {"value": v, "label": "a" if i % 2 else "b"}
            for i, v in enumerate(["GB", "US", "GB", "DE", "US", "GB"])
        ]
        column = field_for(rows)
        assert column.field_class == "CharField"
        assert column.kwargs["choices"] == ["DE", "GB", "US"]

    def test_prose_becomes_a_text_field(self):
        rows = [{"value": f"a fairly long sentence about something, number {i}"} for i in range(10)]
        assert field_for(rows).field_class == "TextField"

    def test_a_value_too_long_to_bound_is_not_given_a_limit(self):
        """A CharField whose max_length is too small rejects valid data later."""
        rows = [
            {"value": f"a value of {i} that runs past the short-value threshold"} for i in range(6)
        ]
        column = field_for(rows)
        assert column.field_class == "TextField"
        assert column.kwargs == {}

    def test_short_unique_strings_become_char_fields(self):
        rows = [{"value": f"sku-{i:04d}"} for i in range(50)]
        column = field_for(rows)
        assert column.field_class == "CharField"
        assert column.kwargs["max_length"] == 16
        assert "choices" not in column.kwargs

    def test_values_that_never_repeat_are_not_categories(self):
        """Distinct-per-row means an identifier, however few rows there are."""
        rows = [{"value": f"v{i}"} for i in range(6)]
        assert "choices" not in field_for(rows).kwargs

    def test_max_length_leaves_headroom(self):
        column = field_for([{"value": "x" * 20}, {"value": "y" * 5}])
        assert column.kwargs["max_length"] == 32


class TestOtherTypes:
    def test_iso_datetimes(self):
        rows = [{"value": "2026-07-30T12:00:00"}, {"value": "2026-07-31T09:30:00"}]
        assert field_for(rows).field_class == "DateTimeField"

    def test_a_trailing_z_is_accepted(self):
        rows = [{"value": "2026-07-30T12:00:00Z"}, {"value": "2026-07-31T09:30:00Z"}]
        assert field_for(rows).field_class == "DateTimeField"

    def test_json_objects_and_arrays(self):
        rows = [{"value": {"a": 1}}, {"value": {"b": 2}}]
        assert field_for(rows).field_class == "JSONField"

    def test_json_encoded_as_a_string(self):
        rows = [{"value": json.dumps(["a", "b"])}, {"value": json.dumps(["c"])}]
        assert field_for(rows).field_class == "JSONField"

    def test_a_bare_string_is_not_json(self):
        assert field_for([{"value": "hello"}, {"value": "there"}]).field_class == "CharField"


class TestNulls:
    def test_missing_values_relax_the_field(self):
        column = field_for([{"value": 1}, {"value": None}, {"value": 3}])
        assert column.kwargs["null"] is True
        assert column.kwargs["required"] is False
        assert column.nulls == 1

    def test_an_empty_string_counts_as_missing(self):
        column = field_for([{"value": "a"}, {"value": ""}])
        assert column.nulls == 1

    def test_a_column_that_is_always_empty_says_so(self):
        column = field_for([{"value": None}, {"value": ""}])
        assert column.all_null
        assert "guess" in column.note

    def test_a_column_absent_from_some_records(self):
        column = field_for([{"value": 1}, {}, {"value": 2}])
        assert column.nulls == 1


# --------------------------------------------------------------------------- #
# Keys and targets
# --------------------------------------------------------------------------- #


class TestPrimaryKey:
    def test_a_unique_id_column_is_the_key(self):
        rows = [{"id": i, "text": "x"} for i in range(5)]
        assert profile(rows).primary_key == "id"

    def test_a_repeated_id_is_not(self):
        rows = [{"id": 1, "text": "x"}, {"id": 1, "text": "y"}]
        assert profile(rows).primary_key is None

    def test_a_suffixed_id_is_accepted(self):
        rows = [{"order_id": f"o{i}", "total": i} for i in range(5)]
        assert profile(rows).primary_key == "order_id"

    def test_uuid_counts_too(self):
        rows = [{"uuid": f"u{i}", "text": "x"} for i in range(4)]
        assert profile(rows).primary_key == "uuid"

    def test_no_key_is_reported_rather_than_invented(self):
        rows = [{"text": "a"}, {"text": "b"}]
        result = profile(rows)
        assert result.primary_key is None
        assert "No unique column" in render_declaration(
            result, name="Rows", source_expr='CSVSource("x.csv")'
        )


class TestTarget:
    def test_a_column_named_label_wins(self):
        rows = [{"country": "GB", "label": "pos"} for _ in range(4)]
        rows += [{"country": "US", "label": "neg"} for _ in range(4)]
        result = profile(rows)

        assert result.target == "label"
        assert result.get("label").field_class == "LabelField"
        assert result.get("label").kwargs["classes"] == ["neg", "pos"]

    def test_only_one_column_becomes_a_target(self):
        """Two LabelFields would leave Model.get_target() unable to decide."""
        rows = [{"country": "GB", "tier": "a", "label": "pos"} for _ in range(4)]
        rows += [{"country": "US", "tier": "b", "label": "neg"} for _ in range(4)]
        result = profile(rows)

        targets = [c.name for c in result.columns if c.field_class in {"LabelField", "TargetField"}]
        assert targets == ["label"]
        assert result.get("country").kwargs["choices"] == ["GB", "US"]

    def test_the_conventional_names_are_recognised(self):
        for name in ("target", "y", "sentiment", "outcome"):
            rows = [{"text": "long enough to be prose here now", name: "a"} for _ in range(3)]
            rows += [{"text": "another sentence of some length!", name: "b"} for _ in range(3)]
            assert profile(rows).target == name, name

    def test_a_target_named_class_survives_being_renamed(self):
        """`class` is a keyword, so the column is renamed before the target is picked."""
        rows = [{"text": "long enough to be prose here now", "class": "a"} for _ in range(3)]
        rows += [{"text": "another sentence of some length!", "class": "b"} for _ in range(3)]
        result = profile(rows)

        assert result.target == "class_"
        assert result.get("class_").field_class == "LabelField"

    def test_a_numeric_target_becomes_a_target_field(self):
        rows = [{"feature": i, "target": i * 1.5} for i in range(10)]
        result = profile(rows)

        assert result.get("target").field_class == "TargetField"
        # A range on a target would reject a legitimate out-of-sample value.
        assert "min_value" not in result.get("target").kwargs

    def test_the_primary_key_is_never_the_target(self):
        rows = [{"id": i, "label": "a" if i % 2 else "b"} for i in range(6)]
        result = profile(rows)
        assert result.primary_key == "id"
        assert result.target == "label"

    def test_the_last_categorical_column_is_the_fallback(self):
        rows = [{"colour": "red", "size": "s"} for _ in range(3)]
        rows += [{"colour": "blue", "size": "m"} for _ in range(3)]
        result = profile(rows)

        assert result.target == "size"
        assert "guessed" in result.get("size").note

    def test_free_text_named_label_is_not_forced_into_a_target(self):
        rows = [{"label": f"a unique sentence of some length number {i}"} for i in range(10)]
        result = profile(rows)

        assert result.target is None
        assert any("left as-is" in w for w in result.warnings)

    def test_no_candidate_at_all_is_reported(self):
        rows = [{"text": f"unique prose number {i} written out at length"} for i in range(10)]
        result = profile(rows)

        assert result.target is None
        assert any("No obvious target" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# Awkward column names
# --------------------------------------------------------------------------- #


class TestColumnNames:
    def test_a_name_with_spaces_is_flagged_and_suggested(self):
        result = profile([{"Review Text": "hello there friend"} for _ in range(3)])

        assert result.columns[0].name == "review_text"
        assert any("not a valid Python name" in w for w in result.warnings)
        assert "rename" in result.warnings[0]

    def test_a_python_keyword_gets_a_suffix(self):
        result = profile([{"class": "a"}, {"class": "b"}, {"class": "a"}])
        assert result.columns[0].name == "class_"

    def test_a_name_of_only_punctuation_still_yields_something_usable(self):
        result = profile([{"???": "a"}, {"???": "b"}])
        assert result.columns[0].name.isidentifier()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


class TestRendering:
    @pytest.fixture
    def rendered(self):
        rows = [
            {
                "id": i,
                "text": f"a review of reasonable length, number {i}",
                "stars": (i % 5) + 1,
                "label": "pos" if i % 2 else "neg",
            }
            for i in range(20)
        ]
        return render_declaration(
            profile(rows), name="Reviews", source_expr='CSVSource("data/reviews.csv")'
        )

    def test_the_output_is_valid_python(self, rendered):
        import ast

        ast.parse(rendered)

    def test_it_declares_the_class_and_its_fields(self, rendered):
        assert "class Reviews(Dataset):" in rendered
        assert "id = IntegerField(min_value=0, max_value=19)" in rendered
        assert "text = TextField()" in rendered
        assert 'label = LabelField(["neg", "pos"])' in rendered

    def test_it_wires_up_meta(self, rendered):
        assert 'source = CSVSource("data/reviews.csv")' in rendered
        assert 'primary_key = "id"' in rendered

    def test_the_imports_cover_what_it_used(self, rendered):
        assert "from mlango.data import Dataset, CSVSource" in rendered
        for name in ("IntegerField", "TextField", "LabelField"):
            assert name in rendered.split("class Reviews")[0]

    def test_it_says_where_it_came_from(self, rendered):
        assert "inspectdata" in rendered
        assert "inferred, not declared" in rendered

    def test_a_long_import_list_is_wrapped(self):
        """The generated file has to pass the project's own line-length rule."""
        rows = [
            {
                "a": 1,
                "b": 1.5,
                "c": "true",
                "d": {"x": 1},
                "e": "2026-07-30T00:00:00",
                "f": "long prose that goes on for a while here",
                "label": "x" if _ % 2 else "y",
            }
            for _ in range(6)
        ]
        rendered = render_declaration(
            profile(rows), name="Wide", source_expr='JSONLSource("w.jsonl")'
        )

        assert "from mlango.core.fields import (" in rendered
        assert all(len(line) <= 100 for line in rendered.splitlines())

    def test_warnings_are_carried_into_the_file(self):
        rows = [{"Odd Name": "a"}, {"Odd Name": "b"}]
        rendered = render_declaration(profile(rows), name="Rows", source_expr='CSVSource("x.csv")')
        assert "# Worth checking:" in rendered

    def test_the_app_name_reaches_the_docstring(self):
        rows = [{"id": 1, "label": "a"}, {"id": 2, "label": "b"}]
        rendered = render_declaration(
            profile(rows), name="Rows", source_expr='CSVSource("x.csv")', app="reviews"
        )
        assert "for the reviews app" in rendered

    def test_a_sampled_file_says_so(self):
        rows = [{"id": i, "label": "a" if i % 2 else "b"} for i in range(50)]
        result = profile_source(InMemorySource(rows), sample=10)
        rendered = render_declaration(result, name="Rows", source_expr='CSVSource("x.csv")')
        assert result.rows_sampled == 10
        assert "inferred from the first 10" in rendered


# --------------------------------------------------------------------------- #
# Source detection
# --------------------------------------------------------------------------- #


class TestSourceDetection:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("a.csv", 'CSVSource("a.csv")'),
            ("a.jsonl", 'JSONLSource("a.jsonl")'),
            ("a.ndjson", 'JSONLSource("a.ndjson")'),
            ("a.json", 'JSONSource("a.json")'),
        ],
    )
    def test_the_expression_reproduces_the_source(self, path, expected):
        _source, expression = source_for(path)
        assert expression == expected

    def test_a_tsv_carries_its_delimiter(self):
        source, expression = source_for("a.tsv")
        assert source.delimiter == "\t"
        assert 'delimiter="\\t"' in expression

    def test_the_extension_is_matched_case_insensitively(self):
        _source, expression = source_for("A.CSV")
        assert expression.startswith("CSVSource")

    def test_an_unknown_extension_lists_the_known_ones(self):
        with pytest.raises(ImproperlyConfigured, match="Recognised extensions"):
            source_for("data.xlsx")


class TestSampling:
    def test_sampling_stops_early(self):
        rows = [{"id": i} for i in range(1000)]
        assert profile_source(InMemorySource(rows), sample=25).rows_sampled == 25

    def test_the_total_is_reported_when_the_source_knows_it(self):
        rows = [{"id": i} for i in range(30)]
        result = profile_source(InMemorySource(rows), sample=10)
        assert result.rows_total == 30

    def test_an_empty_file_is_not_a_crash(self):
        result = profile([])
        assert result.columns == []
        assert any("no records" in w for w in result.warnings)

    def test_column_order_follows_the_file(self):
        rows = [{"z": 1, "a": 2}, {"a": 3, "z": 4}]
        assert [c.name for c in profile(rows).columns] == ["z", "a"]

    def test_a_column_only_in_a_later_record_is_still_found(self):
        rows = [{"a": 1}, {"a": 2, "b": 3}]
        assert [c.name for c in profile(rows).columns] == ["a", "b"]
