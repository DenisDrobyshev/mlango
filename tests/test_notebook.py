"""Using mlango from a notebook, with no project on disk.

Data scientists work in Jupyter, and everything else in this project assumes a
``manage.py``. These are the behaviours that decide whether the framework is
usable there at all.
"""

from __future__ import annotations

import pytest

from mlango.core import fields
from mlango.core.exceptions import ImproperlyConfigured
from mlango.data import Dataset, InMemorySource

pytestmark = pytest.mark.usefixtures("isolated_registry")

ROWS = [
    {"id": i, "text": ("good " if i % 2 else "bad ") + str(i), "label": "pos" if i % 2 else "neg"}
    for i in range(60)
]


def declare_reviews():
    """Exactly what re-running a notebook cell does: run the class body again."""

    class Reviews(Dataset):
        id = fields.IntegerField()
        text = fields.TextField()
        label = fields.LabelField(["neg", "pos"])

        class Meta:
            source = InMemorySource(ROWS)
            primary_key = "id"

    return Reviews


class TestRerunningACell:
    def test_declaring_the_same_class_twice_replaces_it(self, project):
        """Re-running a cell is the commonest thing anyone does in Jupyter.

        Refusing made the framework unusable there, and the message named the
        same module twice, which reads as nonsense.
        """
        first = declare_reviews()
        second = declare_reviews()

        assert first is not second
        assert second.objects.count() == 60

    def test_the_registry_holds_the_newest_declaration(self, project):
        from mlango.core.registry import apps

        declare_reviews()
        second = declare_reviews()

        assert apps.get_dataset(second._meta.label) is second

    def test_a_redeclaration_can_change_the_fields(self, project):
        """Editing a cell and re-running it is how a notebook is written."""
        from mlango.core.registry import apps

        class Rows(Dataset):
            id = fields.IntegerField()

            class Meta:
                source = InMemorySource([{"id": 1}])

        class Rows(Dataset):  # noqa: F811 - the point of the test
            id = fields.IntegerField()
            extra = fields.TextField()

            class Meta:
                source = InMemorySource([{"id": 1, "extra": "x"}])

        assert apps.get_dataset(Rows._meta.label)._meta.field_names == ["id", "extra"]

    def test_two_modules_claiming_one_label_is_still_an_error(self, project):
        """The check exists for a real collision, and that has not gone away.

        Registration reads only the label and the module, so a stand-in is
        enough — and clearer than contriving two modules that really collide.
        """
        from types import SimpleNamespace

        from mlango.core.registry import apps

        first = declare_reviews()

        impostor = SimpleNamespace(
            __module__="some.other.app",
            _meta=SimpleNamespace(label=first._meta.label),
        )

        with pytest.raises(ImproperlyConfigured, match="Labels must be unique"):
            apps.register("dataset", impostor)


class TestNotebookHelper:
    def test_it_configures_everything_needed(self, tmp_path, monkeypatch):
        import mlango
        from mlango.conf import settings

        monkeypatch.chdir(tmp_path)
        settings.reset()

        mlango.notebook()

        assert settings.configured
        assert str(settings.BASE_DIR) == str(tmp_path)
        assert settings.DEFAULT_PROVIDER == "echo"

    def test_calling_it_twice_is_harmless(self, tmp_path, monkeypatch):
        """Re-running the first cell must not reset a session's settings."""
        import mlango
        from mlango.conf import settings

        monkeypatch.chdir(tmp_path)
        settings.reset()

        mlango.notebook()
        settings.SEED = 4242
        mlango.notebook()

        assert settings.SEED == 4242

    def test_overrides_are_accepted(self, tmp_path, monkeypatch):
        import mlango
        from mlango.conf import settings

        monkeypatch.chdir(tmp_path)
        settings.reset()

        mlango.notebook(SEED=7, DEFAULT_PROVIDER="echo")
        assert settings.SEED == 7

    def test_an_explicit_base_dir(self, tmp_path, monkeypatch):
        import mlango
        from mlango.conf import settings

        monkeypatch.chdir(tmp_path)
        settings.reset()

        elsewhere = tmp_path / "work"
        elsewhere.mkdir()
        mlango.notebook(base_dir=str(elsewhere))
        assert str(settings.BASE_DIR) == str(elsewhere)


class TestTrainingWithNoProject:
    @pytest.fixture
    def declared(self, project, sklearn_or_skip):
        from mlango.training import Model

        reviews = declare_reviews()

        class Sentiment(Model):
            C = fields.FloatField(default=1.0, tunable=True)

            class Meta:
                dataset = reviews
                trainer = "sklearn"
                task = "classification"
                features = ["text"]

            def build(self):
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline

                return make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=300))

        return reviews, Sentiment

    def test_the_metastore_creates_itself(self, declared):
        """No migrate step: a notebook user should not have to know the word."""
        from mlango.metastore.session import metastore_ready

        _reviews, sentiment = declared
        sentiment().train()

        assert metastore_ready()

    def test_a_run_is_recorded(self, declared):
        from mlango.training import recent_runs

        _reviews, sentiment = declared
        run = sentiment().train()

        record = run.refresh()
        assert record.status == "finished"
        assert record.summary["accuracy"] > 0.5
        assert [r.kind for r in recent_runs(limit=1)] == ["train"]

    def test_querying_works_before_anything_is_trained(self, declared):
        reviews, _sentiment = declared
        assert reviews.objects.filter(label="pos").count() == 30

    def test_predicting_after_training(self, declared):
        _reviews, sentiment = declared
        model = sentiment()
        model.train()
        assert model.predict("good and warm") in {"pos", "neg"}

    def test_a_second_training_run_is_a_second_version(self, declared):
        _reviews, sentiment = declared
        sentiment().train()
        sentiment(C=2.0).train()

        assert [v.version for v in sentiment.versions()][:2] == [2, 1]
