"""State diffing, the autodetector and the migration writer."""

from __future__ import annotations

import pytest

from mlango.core import fields
from mlango.migrations import (
    AddField,
    AlterField,
    AlterOptions,
    CreateObject,
    DeleteObject,
    MigrationAutodetector,
    MigrationWriter,
    ObjectState,
    ProjectState,
    RemoveField,
    RenameObject,
    build_filename,
    next_migration_number,
)


def state_with(*objects: ObjectState) -> ProjectState:
    state = ProjectState()
    for obj in objects:
        state.add(obj)
    return state


def reviews_state(**overrides) -> ObjectState:
    declared = {
        "app_label": "reviews",
        "name": "Reviews",
        "kind": "dataset",
        "fields": [("id", fields.IntegerField()), ("text", fields.TextField())],
        "options": {},
    }
    declared.update(overrides)
    return ObjectState(**declared)


class TestAutodetector:
    def test_new_object_becomes_create(self):
        changes = MigrationAutodetector(ProjectState(), state_with(reviews_state())).changes()
        assert [type(op) for op in changes["reviews"]] == [CreateObject]
        assert changes["reviews"][0].name == "Reviews"

    def test_removed_object_becomes_delete(self):
        changes = MigrationAutodetector(state_with(reviews_state()), ProjectState()).changes()
        assert [type(op) for op in changes["reviews"]] == [DeleteObject]

    def test_added_field(self):
        after = reviews_state(
            fields=[
                ("id", fields.IntegerField()),
                ("text", fields.TextField()),
                ("lang", fields.CharField(max_length=2)),
            ]
        )
        changes = MigrationAutodetector(state_with(reviews_state()), state_with(after)).changes()
        operation = changes["reviews"][0]
        assert isinstance(operation, AddField)
        assert operation.name == "lang"

    def test_removed_field(self):
        after = reviews_state(fields=[("id", fields.IntegerField())])
        changes = MigrationAutodetector(state_with(reviews_state()), state_with(after)).changes()
        assert isinstance(changes["reviews"][0], RemoveField)

    def test_altered_field(self):
        after = reviews_state(
            fields=[("id", fields.IntegerField()), ("text", fields.TextField(max_length=500))]
        )
        changes = MigrationAutodetector(state_with(reviews_state()), state_with(after)).changes()
        assert isinstance(changes["reviews"][0], AlterField)

    def test_altered_options(self):
        after = reviews_state(options={"primary_key": "id"})
        changes = MigrationAutodetector(state_with(reviews_state()), state_with(after)).changes()
        assert isinstance(changes["reviews"][0], AlterOptions)

    def test_no_changes_produces_nothing(self):
        assert MigrationAutodetector(state_with(reviews_state()), state_with(reviews_state())).changes() == {}

    def test_a_rename_is_not_guessed(self):
        """Guessing renames would silently change what stored data means."""
        after = reviews_state(
            fields=[("id", fields.IntegerField()), ("body", fields.TextField())]
        )
        changes = MigrationAutodetector(state_with(reviews_state()), state_with(after)).changes()
        kinds = {type(op) for op in changes["reviews"]}
        assert kinds == {AddField, RemoveField}
        assert RenameObject not in kinds

    def test_changing_kind_recreates(self):
        after = reviews_state(kind="model")
        changes = MigrationAutodetector(state_with(reviews_state()), state_with(after)).changes()
        assert [type(op) for op in changes["reviews"]] == [DeleteObject, CreateObject]

    def test_deletions_come_last(self):
        before = state_with(reviews_state(), reviews_state(name="Old"))
        after = state_with(
            reviews_state(
                fields=[
                    ("id", fields.IntegerField()),
                    ("text", fields.TextField()),
                    ("extra", fields.TextField()),
                ]
            )
        )
        operations = MigrationAutodetector(before, after).changes()["reviews"]
        assert isinstance(operations[-1], DeleteObject)

    def test_apps_are_diffed_independently(self):
        before = state_with(reviews_state())
        after = state_with(reviews_state(), reviews_state(app_label="other", name="Other"))
        changes = MigrationAutodetector(before, after).changes()
        assert set(changes) == {"other"}


class TestNaming:
    def test_initial_is_named_initial(self):
        assert MigrationAutodetector.suggest_name([], initial=True) == "initial"

    def test_single_create_is_descriptive(self):
        name = MigrationAutodetector.suggest_name(
            [CreateObject(name="Reviews", kind="dataset")]
        )
        assert name == "create_reviews"

    def test_single_add_field_is_descriptive(self):
        name = MigrationAutodetector.suggest_name(
            [AddField("Reviews", "lang", fields.CharField())]
        )
        assert name == "reviews_lang"

    def test_mixed_operations_fall_back_to_a_timestamp(self):
        name = MigrationAutodetector.suggest_name(
            [
                AddField("Reviews", "lang", fields.CharField()),
                RemoveField("Reviews", "text"),
            ]
        )
        assert name.startswith("auto_")


class TestStateReplay:
    def test_operations_advance_the_state(self):
        state = ProjectState()
        CreateObject(
            name="Reviews", kind="dataset", fields=[("id", fields.IntegerField())]
        ).state_forwards("reviews", state)
        assert state.get("reviews", "Reviews") is not None

        AddField("Reviews", "text", fields.TextField()).state_forwards("reviews", state)
        assert state.get("reviews", "Reviews").field_names() == ["id", "text"]

        RemoveField("Reviews", "id").state_forwards("reviews", state)
        assert state.get("reviews", "Reviews").field_names() == ["text"]

        DeleteObject(name="Reviews").state_forwards("reviews", state)
        assert state.get("reviews", "Reviews") is None

    def test_rename_moves_the_object(self):
        state = state_with(reviews_state())
        RenameObject("Reviews", "Feedback").state_forwards("reviews", state)
        assert state.get("reviews", "Reviews") is None
        assert state.get("reviews", "Feedback") is not None

    def test_clone_is_independent(self):
        state = state_with(reviews_state())
        clone = state.clone()
        AddField("Reviews", "extra", fields.TextField()).state_forwards("reviews", clone)
        assert state.get("reviews", "Reviews").field_names() == ["id", "text"]


class TestWriter:
    def test_renders_valid_python(self):
        import ast

        writer = MigrationWriter(
            "reviews",
            "0001_initial",
            [
                CreateObject(
                    name="Reviews",
                    kind="dataset",
                    fields=[
                        ("id", fields.IntegerField()),
                        ("label", fields.LabelField(["neg", "pos"])),
                    ],
                    options={"primary_key": "id", "splits": {"train": 0.8, "val": 0.2}},
                )
            ],
            [],
            initial=True,
        )
        source = writer.as_string()
        ast.parse(source)
        assert "initial = True" in source
        assert "from mlango.core import fields" in source
        assert "fields.LabelField(['neg', 'pos'])" in source

    def test_nested_dicts_keep_their_indentation(self):
        writer = MigrationWriter(
            "reviews",
            "0002_options",
            [AlterOptions("Reviews", {"splits": {"train": 0.8}})],
            [],
        )
        source = writer.as_string()
        assert "                    'train': 0.8," in source

    def test_dependencies_are_rendered(self):
        writer = MigrationWriter(
            "reviews", "0002_next", [DeleteObject(name="Old")], [("reviews", "0001_initial")]
        )
        assert "dependencies = [('reviews', '0001_initial')]" in writer.as_string()

    def test_written_file_can_be_imported(self, tmp_path):
        import importlib.util

        writer = MigrationWriter(
            "reviews",
            "0001_initial",
            [CreateObject(name="Reviews", kind="dataset", fields=[("id", fields.IntegerField())])],
            [],
            initial=True,
        )
        path = writer.write(str(tmp_path))

        spec = importlib.util.spec_from_file_location("m0001", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        migration = module.Migration("0001_initial", "reviews")
        assert migration.initial is True
        assert len(migration.operations) == 1

    def test_run_python_is_never_generated(self):
        from mlango.core.exceptions import MigrationError
        from mlango.migrations import RunPython

        writer = MigrationWriter("reviews", "0002", [RunPython(lambda state, ctx: None)], [])
        with pytest.raises(MigrationError, match="written by hand"):
            writer.as_string()


class TestFilenames:
    def test_numbering_starts_at_one(self, tmp_path):
        assert next_migration_number(str(tmp_path)) == 1

    def test_numbering_continues(self, tmp_path):
        (tmp_path / "0001_initial.py").write_text("")
        (tmp_path / "0007_later.py").write_text("")
        (tmp_path / "notes.txt").write_text("")
        assert next_migration_number(str(tmp_path)) == 8

    def test_filenames_are_zero_padded(self):
        assert build_filename(3, "add_lang") == "0003_add_lang"
