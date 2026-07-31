"""The S3 backend, against an in-memory bucket.

boto3 is not a dependency of the test suite and a network round trip is not a
unit test, so these drive a fake client with the same surface the backend uses.
That is enough to pin what actually matters here: that a name written on one
machine is readable on another, and that nothing is published until the write
has finished.
"""

from __future__ import annotations

import os

import pytest

from mlango.core.exceptions import ImproperlyConfigured
from mlango.storage.s3 import S3Storage, _split


class FakeS3:
    """Just enough of the boto3 S3 client, backed by a dict."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.puts = 0

    def put_object(self, Bucket, Key, Body):  # noqa: N803 - boto3's signature
        self.objects[Key] = Body
        self.puts += 1

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": _Body(self.objects[Key])}

    def head_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[Key])}

    def delete_objects(self, Bucket, Delete):  # noqa: N803
        for entry in Delete["Objects"]:
            self.objects.pop(entry["Key"], None)

    def get_paginator(self, _name):
        return _Paginator(self.objects)


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class _Paginator:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects

    def paginate(self, Bucket, Prefix=""):  # noqa: N803
        contents = [{"Key": key} for key in sorted(self.objects) if key.startswith(Prefix)]
        yield {"Contents": contents}


@pytest.fixture
def s3(project):
    storage = S3Storage(root="s3://bucket/mlango")
    storage._client = FakeS3()
    return storage


class TestConfiguration:
    @pytest.mark.parametrize(
        ("root", "expected"),
        [
            ("s3://bucket/prefix", ("bucket", "prefix")),
            ("s3://bucket", ("bucket", "")),
            ("bucket/deep/prefix", ("bucket", "deep/prefix")),
            ("s3://bucket/prefix/", ("bucket", "prefix")),
        ],
    )
    def test_the_root_is_split_into_bucket_and_prefix(self, root, expected):
        assert _split(root) == expected

    def test_a_missing_bucket_is_refused_with_the_setting_to_write(self):
        with pytest.raises(ImproperlyConfigured, match="ROOT"):
            S3Storage()

    def test_keys_are_prefixed(self, s3):
        assert s3.key("models/demo.joblib") == "mlango/models/demo.joblib"

    def test_escaping_the_prefix_is_refused(self, s3):
        with pytest.raises(ValueError, match="escapes"):
            s3.key("../../etc/passwd")

    def test_path_is_a_url_not_a_filesystem_path(self, s3):
        assert s3.path("a/b.bin") == "s3://bucket/mlango/a/b.bin"

    def test_repr_names_the_bucket(self, s3):
        assert "bucket" in repr(s3)


class TestReadWrite:
    def test_bytes_round_trip(self, s3):
        assert s3.save_bytes("a.bin", b"payload") == "a.bin"
        assert s3.read_bytes("a.bin") == b"payload"

    def test_text_round_trip(self, s3):
        s3.save_text("notes.txt", "héllo")
        assert s3.read_text("notes.txt") == "héllo"

    def test_save_returns_the_name_not_a_url(self, s3):
        """What goes into the metastore has to be resolvable from anywhere."""
        assert s3.save_bytes("models/x.joblib", b"") == "models/x.joblib"

    def test_open_for_reading_gives_a_stream(self, s3):
        s3.save_bytes("a.bin", b"payload")
        assert s3.open("a.bin").read() == b"payload"
        assert s3.open("a.bin", "r").read() == "payload"

    def test_open_for_writing_says_what_to_use_instead(self, s3):
        with pytest.raises(ImproperlyConfigured, match="writable"):
            s3.open("a.bin", "wb")

    def test_exists_and_size(self, s3):
        s3.save_bytes("a.bin", b"1234")
        assert s3.exists("a.bin")
        assert s3.size("a.bin") == 4
        assert not s3.exists("nope.bin")

    def test_listing_strips_the_prefix(self, s3):
        s3.save_bytes("models/a.bin", b"")
        s3.save_bytes("models/deep/b.bin", b"")
        s3.save_bytes("other.bin", b"")
        assert s3.listdir("models") == ["models/a.bin", "models/deep/b.bin"]
        assert s3.listdir() == ["models/a.bin", "models/deep/b.bin", "other.bin"]

    def test_delete_removes_a_whole_prefix(self, s3):
        s3.save_bytes("bundle/a.bin", b"")
        s3.save_bytes("bundle/b.bin", b"")
        s3.delete("bundle")
        assert s3.listdir("bundle") == []


class TestStaging:
    def test_a_file_is_staged_locally_then_uploaded(self, s3):
        with s3.writable("models/demo.joblib") as target:
            assert os.path.isabs(target.path)
            assert not s3.exists("models/demo.joblib"), "nothing published mid-write"
            with open(target.path, "wb") as fh:
                fh.write(b"weights")
            name = target.name

        assert name == "models/demo.joblib"
        assert s3.read_bytes("models/demo.joblib") == b"weights"

    def test_a_failed_write_publishes_nothing(self, s3):
        """A crashed run must not leave half an artifact for the next load."""
        with pytest.raises(RuntimeError):
            with s3.writable("models/demo.joblib") as target:
                with open(target.path, "wb") as fh:
                    fh.write(b"half")
                raise RuntimeError("training blew up")

        assert not s3.exists("models/demo.joblib")

    def test_the_staging_directory_is_cleaned_up(self, s3):
        with s3.writable("models/demo.joblib") as target:
            with open(target.path, "wb") as fh:
                fh.write(b"x")
            staged = target.path
        assert not os.path.exists(staged)

    def test_a_directory_artifact_round_trips(self, s3):
        """The Hugging Face layout is a directory, not a file."""
        with s3.writable("models/bert/model", directory=True) as target:
            for name in ("config.json", "model.safetensors"):
                with open(os.path.join(target.path, name), "wb") as fh:
                    fh.write(name.encode())
            os.makedirs(os.path.join(target.path, "nested"))
            with open(os.path.join(target.path, "nested", "extra.txt"), "wb") as fh:
                fh.write(b"deep")

        assert s3.listdir("models/bert/model") == [
            "models/bert/model/config.json",
            "models/bert/model/model.safetensors",
            "models/bert/model/nested/extra.txt",
        ]

        with s3.readable("models/bert/model") as local:
            assert sorted(os.listdir(local)) == ["config.json", "model.safetensors", "nested"]
            with open(os.path.join(local, "config.json"), "rb") as fh:
                assert fh.read() == b"config.json"
            with open(os.path.join(local, "nested", "extra.txt"), "rb") as fh:
                assert fh.read() == b"deep"

    def test_readable_downloads_and_then_cleans_up(self, s3):
        s3.save_bytes("models/demo.joblib", b"weights")
        with s3.readable("models/demo.joblib") as local:
            with open(local, "rb") as fh:
                assert fh.read() == b"weights"
            downloaded = local
        assert not os.path.exists(downloaded)

    def test_an_absolute_local_path_is_read_in_place(self, s3, project):
        """A version registered before this project moved to S3."""
        legacy = str(project / "old.bin")
        with open(legacy, "wb") as fh:
            fh.write(b"local")
        with s3.readable(legacy) as local:
            assert local == legacy

    def test_locate_says_what_to_use_instead(self, s3):
        with pytest.raises(ImproperlyConfigured, match="readable"):
            s3.locate("models/demo.joblib")

    def test_fetch_caches_so_a_lazy_read_still_works(self, s3):
        """A materialised dataset is iterated long after its path is resolved."""
        s3.save_bytes("datasets/x/v1/data.jsonl", b'{"a": 1}\n')

        first = s3.fetch("datasets/x/v1/data.jsonl")
        with open(first, encoding="utf-8") as fh:
            assert fh.read().strip() == '{"a": 1}'

        s3.client.objects.clear()
        assert s3.fetch("datasets/x/v1/data.jsonl") == first, "the second call is served from cache"


class TestARemoteProject:
    """The story this backend exists for: train here, load there."""

    @pytest.fixture
    def remote(self, project, s3):
        """Point the project's default storage at the fake bucket."""
        from mlango.storage import base, reset_default_storage

        base._default = s3
        yield s3
        reset_default_storage()

    @pytest.fixture
    def sentiment(self, reviews, isolated_registry):
        pytest.importorskip("sklearn")
        from mlango.core import fields
        from mlango.training import Model

        class RemoteSentiment(Model):
            C = fields.FloatField(default=1.0)

            class Meta:
                dataset = reviews
                trainer = "sklearn"
                task = "classification"
                features = ["text"]

            def build(self):
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline

                return make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=500))

        return RemoteSentiment

    def test_a_version_records_a_name_that_is_not_a_local_path(self, remote, sentiment):
        version = sentiment.fit()._version
        assert not os.path.isabs(version.path)
        assert version.path.startswith("models/")
        assert remote.exists(version.path), "the artifact really is in the bucket"

    def test_the_artifact_loads_back_from_the_bucket(self, remote, sentiment):
        sentiment.fit()

        # Nothing of the fitted model survives in this process: load() has to go
        # to storage for it, which is what a second machine would do.
        loaded = sentiment.load()
        assert loaded.predict("great movie") in {"pos", "neg"}

    def test_a_materialised_dataset_round_trips(self, remote, reviews):
        version = reviews.materialize(reviews.objects.take(10))
        assert remote.exists(version.path)
        assert len(list(reviews.load_version(version.version))) == 10

    def test_a_run_artifact_is_recorded_by_name(self, remote, sentiment):
        from mlango.training.run import get_run

        run = sentiment().train()
        artifacts = get_run(run.uuid).artifacts
        assert artifacts, "training registers the model artifact"
        assert all(not os.path.isabs(artifact.path) for artifact in artifacts)
