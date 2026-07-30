"""Meta-option inheritance, and the ready-made model bases built on it."""

from __future__ import annotations

import pytest

from mlango.core import fields
from mlango.core.base import Declarative
from mlango.core.exceptions import ImproperlyConfigured
from mlango.data import Dataset, InMemorySource
from mlango.training import (
    TabularClassifier,
    TabularRegressor,
    TextClassifier,
    TextRegressor,
)

ROWS = [
    {
        "id": index,
        "text": ("good " if index % 2 else "bad ") + str(index),
        "label": "pos" if index % 2 else "neg",
        "score": float(index % 5),
        "x1": float(index),
        "x2": float(index * 2),
    }
    for index in range(40)
]


class Samples(Dataset):
    """Rows for the preset tests."""

    id = fields.IntegerField()
    text = fields.TextField()
    label = fields.LabelField(["neg", "pos"])
    score = fields.TargetField()
    x1 = fields.FloatField()
    x2 = fields.FloatField()

    class Meta:
        source = InMemorySource(ROWS)
        primary_key = "id"


#: Every test here declares classes locally, so each needs the registry rolled
#: back afterwards — labels are unique by design.
pytestmark = pytest.mark.usefixtures("isolated_registry")


class TestMetaInheritance:
    """A subclass writing its own `class Meta` must not lose the parent's options.

    Python class bodies are independent, so without this a reusable base class
    would be impossible to write — which is what presets are.
    """

    def test_options_are_inherited(self):
        class Base(Declarative):
            _kind = "dataset"
            _meta_options = ("flavour", "colour")

            class Meta:
                abstract = True
                flavour = "vanilla"
                colour = "white"

        class Child(Base):
            class Meta:
                colour = "brown"

        assert Child._meta.extras["flavour"] == "vanilla"  # inherited
        assert Child._meta.extras["colour"] == "brown"  # overridden

    def test_abstract_is_never_inherited(self):
        class Base(Declarative):
            _kind = "dataset"

            class Meta:
                abstract = True

        class Child(Base):
            pass

        assert Base._meta.abstract is True
        assert Child._meta.abstract is False

    def test_a_child_without_its_own_meta_still_inherits(self):
        class Base(Declarative):
            _kind = "dataset"
            _meta_options = ("flavour",)

            class Meta:
                abstract = True
                flavour = "vanilla"

        class Child(Base):
            pass

        assert Child._meta.extras["flavour"] == "vanilla"

    def test_inheritance_walks_the_whole_chain(self):
        class A(Declarative):
            _kind = "dataset"
            _meta_options = ("one", "two", "three")

            class Meta:
                abstract = True
                one = 1

        class B(A):
            class Meta:
                abstract = True
                two = 2

        class C(B):
            class Meta:
                three = 3

        assert C._meta.extras == {"one": 1, "two": 2, "three": 3}

    def test_the_nearest_ancestor_wins(self):
        class A(Declarative):
            _kind = "dataset"
            _meta_options = ("flavour",)

            class Meta:
                abstract = True
                flavour = "vanilla"

        class B(A):
            class Meta:
                abstract = True
                flavour = "chocolate"

        class C(B):
            pass

        assert C._meta.extras["flavour"] == "chocolate"


class TestTextClassifier:
    def test_a_three_line_declaration_is_complete(self):
        class Sentiment(TextClassifier):
            """Sentiment over the samples."""

            class Meta:
                dataset = Samples
                features = ["text"]
                target = "label"

        assert Sentiment._meta.extras["trainer"] == "transformers"
        assert Sentiment.get_task() == "classification"
        assert Sentiment.monitor() == ("val_accuracy", "max")
        assert Sentiment.get_features() == ["text"]

    def test_hyperparameters_are_inherited_and_overridable(self):
        class Sentiment(TextClassifier):
            learning_rate = fields.FloatField(default=5e-5, tunable=True)

            class Meta:
                dataset = Samples
                features = ["text"]
                target = "label"

        instance = Sentiment()
        assert instance.learning_rate == 5e-5  # overridden
        assert instance.base_model == "distilbert-base-uncased"  # inherited
        assert instance.epochs == 3

    def test_tunable_fields_give_a_default_sweep_space(self):
        class Sentiment(TextClassifier):
            class Meta:
                dataset = Samples
                features = ["text"]
                target = "label"

        space = Sentiment.default_space()
        assert set(space) == {"learning_rate", "epochs", "batch_size"}
        assert 2e-5 in space["learning_rate"]

    def test_presets_are_abstract_and_unregistered(self):
        from mlango.core.registry import apps

        registered = {m._meta.object_name for m in apps.get_registered("model")}
        assert "TextClassifier" not in registered
        assert "TransformerModel" not in registered

    def test_build_requires_declared_classes(self):
        class Broken(TextClassifier):
            """The target has no declared classes."""

            class Meta:
                dataset = Samples
                features = ["text"]
                target = "score"  # a TargetField, no classes
                task = "classification"

        with pytest.raises(ImproperlyConfigured, match="must declare its classes"):
            Broken().build()

    def test_regression_preset_flips_task_and_monitor(self):
        class Rating(TextRegressor):
            class Meta:
                dataset = Samples
                features = ["text"]
                target = "score"

        assert Rating.get_task() == "regression"
        assert Rating.monitor() == ("val_loss", "min")


class TestTabularPresets:
    def test_classifier_builds_a_network_from_the_declaration(self):
        pytest.importorskip("torch")
        import torch.nn as nn

        class Tab(TabularClassifier):
            class Meta:
                dataset = Samples
                features = ["x1", "x2"]
                target = "label"

        module = Tab(depth=2, hidden_size=8).build()
        assert isinstance(module, nn.Sequential)

        linears = [layer for layer in module if isinstance(layer, nn.Linear)]
        assert linears[0].in_features == 2  # two features
        assert linears[-1].out_features == 2  # two classes

    def test_depth_controls_the_layer_count(self):
        pytest.importorskip("torch")
        import torch.nn as nn

        class Tab(TabularClassifier):
            class Meta:
                dataset = Samples
                features = ["x1", "x2"]
                target = "label"

        shallow = [x for x in Tab(depth=1).build() if isinstance(x, nn.Linear)]
        deep = [x for x in Tab(depth=3).build() if isinstance(x, nn.Linear)]
        assert len(deep) > len(shallow)

    def test_regressor_has_a_single_output(self):
        pytest.importorskip("torch")
        import torch.nn as nn

        class Tab(TabularRegressor):
            class Meta:
                dataset = Samples
                features = ["x1", "x2"]
                target = "score"

        linears = [x for x in Tab().build() if isinstance(x, nn.Linear)]
        assert linears[-1].out_features == 1

    def test_a_tabular_preset_trains_end_to_end(self, project):
        pytest.importorskip("torch")

        class Tab(TabularClassifier):
            """Trained for the preset test."""

            class Meta:
                dataset = Samples
                features = ["x1", "x2"]
                target = "label"

        model = Tab(epochs=2, hidden_size=8, depth=1)
        run = model.train()

        record = run.refresh()
        assert record.status == "finished"
        assert record.params["_trainer"] == "torch"
        assert model.predict({"x1": 1.0, "x2": 2.0}) in {"neg", "pos"}


class TestTransformersTrainer:
    """Logic that can be checked without downloading pretrained weights."""

    def test_the_backend_is_registered(self, project):
        from mlango.conf import settings

        assert "transformers" in settings.TRAINERS

    def test_a_missing_dependency_names_the_extra(self, project):
        pytest.importorskip("torch")
        if __import__("importlib.util", fromlist=["util"]).find_spec("transformers"):
            pytest.skip("transformers is installed, so there is no missing dependency to report")

        from mlango.core.exceptions import (
            BackendNotAvailable,
            ProviderError,  # noqa: F401
        )
        from mlango.training.trainer import get_trainer

        with pytest.raises(BackendNotAvailable, match=r"mlango\[transformers\]"):
            get_trainer("transformers")

    def test_a_single_text_field_is_used_directly(self):
        from mlango.data.query import Record
        from mlango.training.backends.transformers_backend import TransformersTrainer

        trainer = TransformersTrainer()
        record = Record({"text": "hello", "other": "ignored"})
        assert trainer._text_of(record, ["text"]) == "hello"

    def test_two_text_fields_become_a_pair(self):
        from mlango.data.query import Record
        from mlango.training.backends.transformers_backend import TransformersTrainer

        trainer = TransformersTrainer()
        record = Record({"question": "q", "answer": "a"})
        assert trainer._text_of(record, ["question", "answer"]) == ("q", "a")

    def test_three_text_fields_are_refused_with_advice(self):
        from mlango.core.exceptions import RunError
        from mlango.data.query import Record
        from mlango.training.backends.transformers_backend import TransformersTrainer

        trainer = TransformersTrainer()
        with pytest.raises(RunError, match="encode_batch"):
            trainer._text_of(Record({"a": "1", "b": "2", "c": "3"}), ["a", "b", "c"])

    def test_raw_strings_become_records_with_a_placeholder_label(self):
        from mlango.training.backends.transformers_backend import _as_records

        records = _as_records(["hello", "world"], ["text"], "label", Samples)
        assert records[0]["text"] == "hello"
        # A placeholder is needed so tokenisation has a label column to fill.
        assert records[0]["label"] == "neg"

    def test_dicts_pass_through(self):
        from mlango.training.backends.transformers_backend import _as_records

        records = _as_records([{"text": "hi"}], ["text"], "label", Samples)
        assert records[0]["text"] == "hi"
