"""Artifact storage: the local backend, path safety and backend resolution."""

from __future__ import annotations

import os

import pytest

from mlango.storage import default_storage, reset_default_storage
from mlango.storage.local import LocalStorage


@pytest.fixture
def storage(project):
    return LocalStorage(root=str(project / "store"))


class TestPaths:
    def test_the_root_is_resolved_against_base_dir(self, project):
        assert LocalStorage(root="artifacts").root == os.path.join(str(project), "artifacts")

    def test_an_absolute_root_is_left_alone(self, project, tmp_path):
        elsewhere = str(tmp_path / "elsewhere")
        assert LocalStorage(root=elsewhere).root == elsewhere

    def test_path_creates_the_parent_directory(self, storage):
        target = storage.path("deep/nested/file.txt")
        assert os.path.isdir(os.path.dirname(target))
        assert not os.path.exists(target)

    def test_backslashes_are_accepted(self, storage):
        """A Windows-style name must land in the same place as a POSIX one."""
        assert storage.path("a\\b.txt") == storage.path("a/b.txt")

    @pytest.mark.parametrize(
        "name",
        ["../escape.txt", "a/../../escape.txt", "/../escape.txt", "a/b/../../../escape.txt"],
    )
    def test_escaping_the_root_is_refused(self, storage, name):
        """Storage names come from labels and sometimes from requests."""
        with pytest.raises(ValueError, match="escapes the storage root"):
            storage.path(name)

    def test_traversal_that_stays_inside_is_allowed(self, storage):
        assert storage.path("a/../b.txt") == storage.path("b.txt")

    def test_the_root_itself_is_allowed(self, storage):
        assert storage.path("") == os.path.normpath(storage.root)

    def test_a_sibling_directory_with_a_shared_prefix_is_refused(self, project):
        """`store2` must not pass merely because it starts with `store`."""
        storage = LocalStorage(root=str(project / "store"))
        with pytest.raises(ValueError, match="escapes"):
            storage.path("../store2/file.txt")


class TestReadWrite:
    def test_bytes_round_trip(self, storage):
        storage.save_bytes("a/b.bin", b"\x00\x01")
        assert storage.read_bytes("a/b.bin") == b"\x00\x01"

    def test_text_round_trip(self, storage):
        storage.save_text("notes.txt", "привет")
        assert storage.read_text("notes.txt") == "привет"

    def test_a_write_is_atomic(self, storage):
        """A crash mid-write must not leave a half-written checkpoint."""
        storage.save_bytes("model.bin", b"first")
        assert storage.listdir() == ["model.bin"]  # no stray .part file

        storage.save_bytes("model.bin", b"second")
        assert storage.read_bytes("model.bin") == b"second"

    def test_open_for_writing_creates_parents(self, storage):
        with storage.open("logs/run.txt", "w") as fh:
            fh.write("line")
        assert storage.read_text("logs/run.txt") == "line"

    def test_open_for_reading_does_not(self, storage):
        with pytest.raises(FileNotFoundError):
            storage.open("missing/file.txt", "rb")

    def test_append_mode_counts_as_writing(self, storage):
        with storage.open("logs/append.txt", "a") as fh:
            fh.write("one")
        with storage.open("logs/append.txt", "a") as fh:
            fh.write("two")
        assert storage.read_text("logs/append.txt") == "onetwo"

    def test_reading_something_that_is_not_there(self, storage):
        with pytest.raises(FileNotFoundError):
            storage.read_bytes("nope.bin")


class TestMetadataAndCleanup:
    def test_exists(self, storage):
        assert storage.exists("a.txt") is False
        storage.save_text("a.txt", "x")
        assert storage.exists("a.txt") is True

    def test_size(self, storage):
        storage.save_bytes("a.bin", b"12345")
        assert storage.size("a.bin") == 5

    def test_delete_a_file(self, storage):
        storage.save_text("a.txt", "x")
        storage.delete("a.txt")
        assert not storage.exists("a.txt")

    def test_delete_a_directory(self, storage):
        storage.save_text("tree/one.txt", "x")
        storage.save_text("tree/deep/two.txt", "y")
        storage.delete("tree")
        assert storage.listdir() == []

    def test_deleting_what_is_not_there_is_quiet(self, storage):
        storage.delete("never-existed.txt")

    def test_makedirs_is_idempotent(self, storage):
        first = storage.makedirs("checkpoints")
        assert storage.makedirs("checkpoints") == first
        assert os.path.isdir(first)

    def test_url_falls_back_to_the_path(self, storage):
        assert storage.url("a.txt") == storage.path("a.txt")

    def test_repr_shows_the_root(self, storage):
        # repr() escapes Windows separators, so compare on the tail only.
        assert repr(storage).startswith("<LocalStorage root=")
        assert "store" in repr(storage)


class TestListing:
    def test_listing_is_recursive_sorted_and_posix(self, storage):
        storage.save_text("b.txt", "x")
        storage.save_text("a/deep/c.txt", "x")
        storage.save_text("a/b.txt", "x")
        assert storage.listdir() == ["a/b.txt", "a/deep/c.txt", "b.txt"]

    def test_listing_a_prefix(self, storage):
        storage.save_text("models/one.bin", "x")
        storage.save_text("data/two.bin", "x")
        assert storage.listdir("models") == ["models/one.bin"]

    def test_listing_a_missing_prefix_is_empty_not_an_error(self, storage):
        assert storage.listdir("nope") == []

    def test_listing_an_empty_store(self, storage):
        assert storage.listdir() == []


class TestBackendResolution:
    def test_the_default_comes_from_settings(self, project):
        assert isinstance(default_storage(), LocalStorage)

    def test_the_backend_is_cached(self, project):
        assert default_storage() is default_storage()

    def test_reset_rebuilds_it(self, project):
        first = default_storage()
        reset_default_storage()
        assert default_storage() is not first

    def test_settings_keys_are_lowercased_into_kwargs(self, project):
        from mlango.conf import settings

        reset_default_storage()
        settings.STORAGE = {
            "BACKEND": "mlango.storage.local.LocalStorage",
            "ROOT": "custom-root",
        }
        assert default_storage().root.endswith("custom-root")

    def test_a_typo_in_the_backend_is_reported(self, project):
        from mlango.conf import settings
        from mlango.core.exceptions import ImproperlyConfigured

        reset_default_storage()
        settings.STORAGE = {"BACKEND": "mlango.storage.local.Nope", "ROOT": "artifacts"}
        with pytest.raises(ImproperlyConfigured):
            default_storage()

    def test_a_custom_backend_only_needs_the_abstract_methods(self, project):
        """The contract is small on purpose — this is the whole surface."""
        from mlango.storage.base import Storage

        class MemoryStorage(Storage):
            def __init__(self, **options):
                super().__init__(**options)
                self.blobs: dict[str, bytes] = {}

            def path(self, name):
                return f"memory://{name}"

            def open(self, name, mode="rb"):
                raise NotImplementedError

            def save_bytes(self, name, data):
                self.blobs[name] = data
                return self.path(name)

            def read_bytes(self, name):
                return self.blobs[name]

            def exists(self, name):
                return name in self.blobs

            def delete(self, name):
                self.blobs.pop(name, None)

            def size(self, name):
                return len(self.blobs[name])

            def listdir(self, prefix=""):
                return sorted(k for k in self.blobs if k.startswith(prefix))

        store = MemoryStorage()
        store.save_text("a.txt", "hello")
        assert store.read_text("a.txt") == "hello"
        assert store.url("a.txt") == "memory://a.txt"
        assert store.listdir() == ["a.txt"]
