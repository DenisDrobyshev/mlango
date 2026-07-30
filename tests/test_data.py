"""Datasets, the lazy queryset and versioned materialisation."""

from __future__ import annotations

import pytest

from mlango.core import fields
from mlango.core.exceptions import DoesNotExist, FieldError, ValidationError
from mlango.data import CSVSource, Dataset, InMemorySource, JSONLSource


class TestLookups:
    def test_exact_and_negation(self, reviews):
        assert reviews.objects.filter(label="pos").count() == 50
        assert reviews.objects.exclude(label="pos").count() == 50

    def test_comparisons(self, reviews):
        assert reviews.objects.filter(stars__gte=4).count() == 40
        assert reviews.objects.filter(stars__lt=2).count() == 20

    def test_membership(self, reviews):
        assert reviews.objects.filter(stars__in=[1, 2]).count() == 40

    def test_substring_is_case_insensitive(self, reviews):
        assert reviews.objects.filter(text__icontains="GREAT").count() == 50

    def test_unknown_field_is_rejected_early(self, reviews):
        with pytest.raises(FieldError, match="has no field"):
            reviews.objects.filter(nope=1)

    def test_chaining_narrows(self, reviews):
        assert reviews.objects.filter(label="pos").filter(stars=1).count() == 10


class TestRecords:
    def test_supports_both_access_styles(self, reviews):
        record = reviews.objects.first()
        assert record.text == record["text"]

    def test_attribute_error_names_available_fields(self, reviews):
        record = reviews.objects.first()
        with pytest.raises(AttributeError, match="Present:"):
            _ = record.nope


class TestPipeline:
    def test_map_replaces_records(self, reviews):
        record = reviews.objects.map(lambda r: {"n": len(r.text)}).first()
        assert set(record) == {"n"}

    def test_annotate_adds_a_field(self, reviews):
        record = reviews.objects.annotate(n=lambda r: len(r.text)).first()
        assert record.n == len(record.text)

    def test_only_and_defer(self, reviews):
        assert set(reviews.objects.only("id", "text").first()) == {"id", "text"}
        assert "text" not in reviews.objects.defer("text").first()

    def test_order_by_supports_descending(self, reviews):
        assert reviews.objects.order_by("-id").first().id == 99

    def test_take_and_skip(self, reviews):
        assert [r.id for r in reviews.objects.skip(98).take(2)] == [98, 99]

    def test_slicing(self, reviews):
        assert [r.id for r in reviews.objects[10:13]] == [10, 11, 12]

    def test_indexing(self, reviews):
        assert reviews.objects[5].id == 5

    def test_shuffle_is_seeded(self, reviews):
        first = [r.id for r in reviews.objects.shuffle(seed=1)]
        second = [r.id for r in reviews.objects.shuffle(seed=1)]
        assert first == second
        assert first != [r.id for r in reviews.objects]

    def test_distinct(self, reviews):
        assert reviews.objects.distinct("label").count() == 2

    def test_lazy_until_iterated(self, reviews):
        calls = []

        def produce():
            calls.append(1)
            yield {"id": 1, "text": "a", "label": "pos", "stars": 1}

        class Lazy(Dataset):
            id = fields.IntegerField()
            text = fields.TextField()
            label = fields.LabelField(["neg", "pos"])
            stars = fields.IntegerField()

            class Meta:
                # A class body is not a closure, so the factory has to be
                # referenced by a name that is not shadowed by the option.
                source = staticmethod(produce)

        query = Lazy.objects.filter(label="pos").take(1)
        assert calls == []
        query.all()
        assert calls == [1]

    def test_cache_evaluates_once(self, reviews):
        cached = reviews.objects.filter(label="pos").cache()
        assert cached.count() == cached.count() == 50


class TestTerminals:
    def test_get_requires_exactly_one(self, reviews):
        assert reviews.objects.get(id=7).id == 7
        with pytest.raises(DoesNotExist):
            reviews.objects.get(id=10_000)

    def test_values_list_flat(self, reviews):
        assert reviews.objects.take(3).values_list("id", flat=True) == [0, 1, 2]

    def test_columns_is_column_oriented(self, reviews):
        columns = reviews.objects.take(2).columns("id", "label")
        assert columns == {"id": [0, 1], "label": ["pos", "neg"]}

    def test_xy_uses_declared_features(self, reviews):
        inputs, targets = reviews.objects.take(2).xy(features=["text"])
        assert inputs == ["great movie 0", "terrible movie 1"]
        assert targets == ["pos", "neg"]

    def test_batch_yields_lists(self, reviews):
        assert [len(b) for b in reviews.objects.take(25).batch(10)] == [10, 10, 5]

    def test_batch_can_drop_the_remainder(self, reviews):
        assert [len(b) for b in reviews.objects.take(25).batch(10, drop_last=True)] == [10, 10]


class TestSplits:
    def test_partitions_cover_everything_once(self, reviews):
        parts = reviews.objects.split(train=0.8, val=0.1, test=0.1)
        sizes = {name: query.count() for name, query in parts.items()}
        assert sum(sizes.values()) == 100
        ids = [set(query.values_list("id", flat=True)) for query in parts.values()]
        assert set.intersection(*ids) == set()

    def test_ratios_must_sum_to_one(self, reviews):
        with pytest.raises(ValueError, match="sum to 1.0"):
            reviews.objects.split(train=0.5, val=0.2)

    def test_assignment_is_stable_when_rows_are_added(self):
        rows = [{"id": i, "text": f"t{i}", "label": "pos"} for i in range(50)]

        class Growing(Dataset):
            id = fields.IntegerField()
            text = fields.TextField()
            label = fields.LabelField(["pos"])

            class Meta:
                source = InMemorySource(rows)
                primary_key = "id"

        before = set(
            Growing.objects.split(train=0.8, test=0.2)["train"].values_list("id", flat=True)
        )
        rows.extend({"id": i, "text": f"t{i}", "label": "pos"} for i in range(50, 80))
        after = set(
            Growing.objects.split(train=0.8, test=0.2)["train"].values_list("id", flat=True)
        )

        # This is the property that keeps a held-out set trustworthy over time.
        assert before <= after


class TestValidation:
    def test_validate_raises_on_a_bad_row(self):
        class Broken(Dataset):
            id = fields.IntegerField()
            label = fields.LabelField(["a", "b"])

            class Meta:
                source = InMemorySource([{"id": 1, "label": "a"}, {"id": 2, "label": "zzz"}])

        with pytest.raises(ValidationError, match="row 1"):
            Broken.objects.validate().all()

    def test_clean_coerces_values(self):
        class Loose(Dataset):
            id = fields.IntegerField()
            stars = fields.IntegerField()

            class Meta:
                source = InMemorySource([{"id": "1", "stars": "5"}])

        record = Loose.objects.clean().first()
        assert record.id == 1 and record.stars == 5


class TestSources:
    def test_jsonl_round_trip(self, project):
        path = project / "rows.jsonl"
        path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
        source = JSONLSource(str(path))
        assert [r["a"] for r in source] == [1, 2]
        assert source.count() == 2

    def test_jsonl_reports_the_offending_line(self, project):
        path = project / "bad.jsonl"
        path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="bad.jsonl:2"):
            list(JSONLSource(str(path)))

    def test_csv_reads_headers(self, project):
        path = project / "rows.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        assert list(CSVSource(str(path))) == [{"a": "1", "b": "2"}]

    def test_missing_file_is_reported_clearly(self, project):
        with pytest.raises(FileNotFoundError):
            list(JSONLSource(str(project / "nope.jsonl")))

    @pytest.mark.parametrize(
        ("name", "body", "source_class"),
        [
            ("bom.jsonl", '{"a": 1}\n{"a": 2}\n', JSONLSource),
            ("bom.csv", "a\n1\n2\n", CSVSource),
        ],
    )
    def test_a_byte_order_mark_does_not_break_the_first_record(
        self, project, name, body, source_class
    ):
        """Excel, Notepad and PowerShell all write a BOM by default.

        Without utf-8-sig the first record failed to parse, and the message
        talked about byte 0xEF rather than about the file having a BOM.
        """
        path = project / name
        path.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))

        rows = list(source_class(str(path)))
        assert len(rows) == 2
        assert list(rows[0]) == ["a"], rows[0]

    def test_a_file_without_a_bom_is_unaffected(self, project):
        path = project / "plain.jsonl"
        path.write_text('{"a": 1}\n', encoding="utf-8")
        assert list(JSONLSource(str(path))) == [{"a": 1}]

    def test_a_json_array_with_a_bom(self, project):
        from mlango.data import JSONSource

        path = project / "bom.json"
        path.write_bytes(b"\xef\xbb\xbf" + b'[{"a": 1}]')
        assert list(JSONSource(str(path))) == [{"a": 1}]


class TestMaterialize:
    def test_writes_a_version(self, project, reviews):
        version = reviews.materialize(reviews.objects.filter(label="pos"), notes="pos only")
        assert version.version == 1
        assert version.row_count == 50
        assert version.notes == "pos only"
        assert version.content_hash

    def test_identical_content_is_deduplicated(self, project, reviews):
        first = reviews.materialize(reviews.objects.filter(label="pos"))
        second = reviews.materialize(reviews.objects.filter(label="pos"))
        assert first.id == second.id

    def test_force_creates_a_duplicate_on_purpose(self, project, reviews):
        first = reviews.materialize(reviews.objects.filter(label="pos"))
        forced = reviews.materialize(reviews.objects.filter(label="pos"), force=True)
        assert forced.id != first.id
        assert forced.version == first.version + 1

    def test_deduplication_survives_a_forced_duplicate(self, project, reviews):
        """force=True must not poison later calls.

        Two versions can legitimately share a content hash once force has been
        used, so the dedup lookup has to tolerate several matches rather than
        insisting on exactly one.
        """
        original = reviews.materialize(reviews.objects.filter(label="pos"))
        reviews.materialize(reviews.objects.filter(label="pos"), force=True)

        again = reviews.materialize(reviews.objects.filter(label="pos"))
        assert again.id == original.id  # the earliest snapshot of this content

    def test_different_content_makes_a_new_version(self, project, reviews):
        reviews.materialize(reviews.objects.filter(label="pos"))
        second = reviews.materialize(reviews.objects.get_queryset())
        assert second.version == 2
        assert second.row_count == 100

    def test_load_version_reads_the_snapshot(self, project, reviews):
        reviews.materialize(reviews.objects.filter(label="pos"))
        assert reviews.load_version(1).count() == 50

    def test_load_version_without_one_explains_itself(self, project, reviews):
        with pytest.raises(LookupError, match="materialize"):
            reviews.load_version()

    def test_pipeline_is_recorded(self, project, reviews):
        version = reviews.materialize(reviews.objects.filter(label="pos").shuffle(seed=3))
        operations = [step["op"] for step in version.pipeline]
        assert operations == ["filter", "shuffle"]


class TestFingerprints:
    def test_view_fingerprint_reflects_the_pipeline(self, reviews):
        plain = reviews.objects.get_queryset().fingerprint()
        filtered = reviews.objects.filter(label="pos").fingerprint()
        assert plain != filtered

    def test_content_hash_reflects_the_rows(self, reviews):
        assert (
            reviews.objects.filter(label="pos").content_hash()
            != reviews.objects.filter(label="neg").content_hash()
        )
