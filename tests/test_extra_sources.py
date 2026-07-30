"""Parquet, SQL, Hugging Face and dataset-version sources."""

from __future__ import annotations

import pytest

from mlango.core import fields
from mlango.core.exceptions import ImproperlyConfigured
from mlango.data import (
    Dataset,
    DatasetVersionSource,
    HuggingFaceSource,
    ParquetSource,
    SQLSource,
)


@pytest.fixture
def parquet_file(project):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    table = pa.table(
        {
            "id": list(range(50)),
            "text": [f"row {i}" for i in range(50)],
            "label": ["pos" if i % 2 == 0 else "neg" for i in range(50)],
        }
    )
    path = project / "rows.parquet"
    pq.write_table(table, path)
    return path


class TestParquetSource:
    def test_reads_every_row(self, parquet_file):
        source = ParquetSource(str(parquet_file))
        rows = list(source)
        assert len(rows) == 50
        assert rows[0] == {"id": 0, "text": "row 0", "label": "pos"}

    def test_count_comes_from_the_footer(self, parquet_file):
        assert ParquetSource(str(parquet_file)).count() == 50

    def test_column_projection(self, parquet_file):
        rows = list(ParquetSource(str(parquet_file), columns=["id", "label"]))
        assert set(rows[0]) == {"id", "label"}

    def test_streams_in_batches(self, parquet_file):
        # A small batch size must not change the result, only the memory profile.
        rows = list(ParquetSource(str(parquet_file), batch_size=7))
        assert len(rows) == 50

    def test_missing_file_is_reported(self, project):
        with pytest.raises(FileNotFoundError):
            list(ParquetSource(str(project / "nope.parquet")))

    def test_count_of_a_missing_file_is_unknown(self, project):
        assert ParquetSource(str(project / "nope.parquet")).count() is None

    def test_powers_a_dataset(self, parquet_file):
        class ParquetRows(Dataset):
            id = fields.IntegerField()
            text = fields.TextField()
            label = fields.LabelField(["neg", "pos"])

            class Meta:
                source = ParquetSource(str(parquet_file))
                primary_key = "id"

        assert ParquetRows.objects.filter(label="pos").count() == 25
        assert ParquetRows.objects.first().text == "row 0"

    def test_describe_is_json_safe(self, parquet_file):
        import json

        json.dumps(ParquetSource(str(parquet_file)).describe())


class TestSQLSource:
    def test_reads_from_the_metastore(self, project, reviews):
        # Materialise, then read the recorded versions back out with SQL.
        reviews.materialize(reviews.objects.filter(label="pos"))

        source = SQLSource("SELECT label, version, row_count FROM mlango_dataset_versions")
        rows = list(source)
        assert len(rows) == 1
        assert rows[0]["row_count"] == 50

    def test_count_wraps_the_query(self, project, reviews):
        reviews.materialize(reviews.objects.get_queryset())
        assert SQLSource("SELECT * FROM mlango_dataset_versions").count() == 1

    def test_count_of_invalid_sql_is_unknown(self, project):
        assert SQLSource("NOT VALID SQL").count() is None

    def test_parameters_are_bound(self, project, reviews):
        reviews.materialize(reviews.objects.get_queryset())
        source = SQLSource(
            "SELECT * FROM mlango_dataset_versions WHERE label = :label",
            params={"label": reviews._meta.label},
        )
        assert len(list(source)) == 1

    def test_describe_never_leaks_a_password(self):
        described = SQLSource("SELECT 1", url="postgresql://user:secret@host/db").describe()
        assert "secret" not in str(described)
        assert described["url"] == "external"


class TestDatasetVersionSource:
    def test_reads_a_frozen_snapshot(self, project, reviews):
        reviews.materialize(reviews.objects.filter(label="pos"))

        source = DatasetVersionSource(reviews._meta.label, version=1)
        assert len(list(source)) == 50
        assert source.count() == 50

    def test_latest_version_by_default(self, project, reviews):
        reviews.materialize(reviews.objects.filter(label="pos"))
        reviews.materialize(reviews.objects.get_queryset())

        assert len(list(DatasetVersionSource(reviews._meta.label))) == 100

    def test_missing_version_explains_the_fix(self, project, reviews):
        with pytest.raises(LookupError, match="materialize"):
            list(DatasetVersionSource(reviews._meta.label))

    def test_a_derived_dataset_can_pin_an_upstream_version(self, project, reviews):
        reviews.materialize(reviews.objects.filter(label="pos"))

        class DerivedReviews(Dataset):
            """Built on a pinned upstream snapshot."""

            id = fields.IntegerField()
            text = fields.TextField()
            label = fields.LabelField(["neg", "pos"])
            stars = fields.IntegerField()

            class Meta:
                source = DatasetVersionSource(reviews._meta.label, version=1)
                primary_key = "id"

        assert DerivedReviews.objects.count() == 50
        assert DerivedReviews.objects.exclude(label="pos").count() == 0


class TestHuggingFaceSource:
    def test_describe_without_loading(self):
        source = HuggingFaceSource("imdb", split="train", streaming=True)
        assert source.describe() == {
            "type": "HuggingFaceSource",
            "path": "imdb",
            "split": "train",
            "name": None,
            "streaming": True,
        }

    def test_streaming_count_is_unknown(self):
        assert HuggingFaceSource("imdb", streaming=True).count() is None

    def test_missing_dependency_explains_the_extra(self, monkeypatch):
        import importlib

        real_import = importlib.import_module

        def fail_for_datasets(name, *args, **kwargs):
            if name == "datasets":
                raise ImportError("no module named datasets")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", fail_for_datasets)

        with pytest.raises(ImproperlyConfigured, match=r"mlango\[huggingface\]"):
            list(HuggingFaceSource("imdb"))
