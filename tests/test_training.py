"""Models, trainers, runs and the model registry."""

from __future__ import annotations

import pytest

from mlango.core import fields
from mlango.core.exceptions import ImproperlyConfigured
from mlango.metastore.models import Stage
from mlango.training import Model, metrics
from mlango.training.callbacks import Callback, CallbackList, EarlyStopping


@pytest.fixture(scope="module")
def sentiment(reviews, sklearn_or_skip):
    """Declared once for the module: labels are unique, so redeclaring would clash."""

    class Sentiment(Model):
        """TF-IDF into logistic regression."""

        max_features = fields.IntegerField(default=500, tunable=True)
        C = fields.FloatField(default=1.0, min_value=0.0, tunable=True)

        class Meta:
            dataset = reviews
            trainer = "sklearn"
            task = "classification"
            features = ["text"]

        def build(self):
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline

            return make_pipeline(
                TfidfVectorizer(max_features=self.max_features),
                LogisticRegression(C=self.C, max_iter=1000),
            )

    return Sentiment


class TestDeclaration:
    def test_hyperparameters_are_fields(self, sentiment):
        assert sentiment._meta.field_names == ["max_features", "C"]

    def test_tunable_fields_are_marked(self, sentiment):
        assert [f.name for f in sentiment._meta.tunable_fields] == ["max_features", "C"]

    def test_params_are_validated(self, sentiment):
        with_override = sentiment(C=2.5)
        assert with_override.params == {"max_features": 500, "C": 2.5}

    def test_out_of_range_param_is_rejected(self, sentiment):
        from mlango.core.exceptions import ValidationError

        with pytest.raises(ValidationError):
            sentiment(C=-1).full_clean()

    def test_primary_key_is_excluded_from_features(self, reviews, sklearn_or_skip):
        class Implicit(Model):
            class Meta:
                dataset = reviews
                trainer = "sklearn"

            def build(self):
                return None

        # `id` is the primary key and `label` the target, so neither is a feature.
        assert Implicit.get_features() == ["text", "stars"]

    def test_missing_trainer_is_reported(self, reviews):
        class NoTrainer(Model):
            class Meta:
                dataset = reviews

            def build(self):
                return None

        with pytest.raises(ImproperlyConfigured, match="Meta.trainer"):
            NoTrainer.get_trainer()

    def test_missing_build_is_reported(self, reviews):
        class NoBuild(Model):
            class Meta:
                dataset = reviews
                trainer = "sklearn"

        with pytest.raises(NotImplementedError, match="must implement build"):
            NoBuild().build()


class TestTraining:
    def test_end_to_end(self, project, sentiment):
        model = sentiment(C=2.0)
        run = model.train(tags=["unit"])

        record = run.refresh()
        assert record.status == "finished"
        assert record.kind == "train"
        assert record.summary["accuracy"] == pytest.approx(1.0)
        assert record.tags == ["unit"]

    def test_run_captures_reproducibility_metadata(self, project, sentiment):
        run = sentiment().train()
        record = run.refresh()
        assert record.seed == 0
        assert record.python_version
        assert record.params["_data_fingerprint"]
        assert record.params["_features"] == ["text"]

    def test_predict_after_training(self, project, sentiment):
        model = sentiment()
        model.train()
        assert model.predict("great movie 1") == "pos"
        assert model.predict(["terrible movie 3"]) == ["neg"]

    def test_predict_proba_returns_class_map(self, project, sentiment):
        model = sentiment()
        model.train()
        proba = model.predict_proba("great movie 1")
        assert set(proba) == {"neg", "pos"}
        assert sum(proba.values()) == pytest.approx(1.0)

    def test_predicting_before_training_explains_itself(self, project, sentiment):
        from mlango.core.exceptions import RunError

        with pytest.raises(RunError, match="no fitted weights"):
            sentiment().predict("anything")

    def test_metrics_are_recorded(self, project, sentiment):
        from mlango.training import metric_keys

        run = sentiment().train()
        keys = metric_keys(run.run_id)
        assert "accuracy" in keys
        assert "val_accuracy" in keys


class TestVersions:
    def test_training_registers_a_version(self, project, sentiment):
        model = sentiment()
        model.train()
        assert model._version.version == 1
        assert model._version.stage == Stage.NONE

    def test_versions_increment(self, project, sentiment):
        sentiment().train()
        second = sentiment(C=3.0)
        second.train()
        assert second._version.version == 2

    def test_load_restores_hyperparameters(self, project, sentiment):
        sentiment(C=2.5).train()
        loaded = sentiment.load()
        assert loaded.C == 2.5
        assert loaded.predict("great movie 1") == "pos"

    def test_promote_demotes_the_incumbent(self, project, sentiment):
        sentiment().train()
        sentiment(C=2.0).train()
        sentiment.promote(1, Stage.PRODUCTION)
        sentiment.promote(2, Stage.PRODUCTION)

        stages = {v.version: v.stage for v in sentiment.versions()}
        assert stages[2] == Stage.PRODUCTION
        assert stages[1] == Stage.ARCHIVED

    def test_production_loads_the_promoted_version(self, project, sentiment):
        sentiment().train()
        sentiment.promote(1, Stage.PRODUCTION)
        assert sentiment.production()._version.version == 1

    def test_unknown_stage_is_rejected(self, project, sentiment):
        sentiment().train()
        with pytest.raises(ValueError, match="Unknown stage"):
            sentiment.promote(1, "sideways")

    def test_no_register_skips_the_registry(self, project, sentiment):
        model = sentiment()
        model.train(register=False)
        assert model._version is None
        assert sentiment.versions() == []


class TestFailures:
    def test_a_failing_build_marks_the_run_failed(self, project, reviews):
        from mlango.core.exceptions import RunError

        class Broken(Model):
            class Meta:
                dataset = reviews
                trainer = "sklearn"
                features = ["text"]

            def build(self):
                raise ZeroDivisionError("boom")

        with pytest.raises(RunError, match="boom"):
            Broken().train()

        from mlango.training import recent_runs

        latest = recent_runs(limit=1)[0]
        assert latest.status == "failed"
        assert "ZeroDivisionError" in latest.error


class TestCallbacks:
    def test_hooks_fire_in_order(self, project, sentiment):
        seen = []

        class Recorder(Callback):
            def on_train_begin(self, run, model, **kwargs):
                seen.append("begin")

            def on_epoch_end(self, run, epoch, metrics, **kwargs):
                seen.append("epoch")

            def on_train_end(self, run, model, **kwargs):
                seen.append("end")

        sentiment().train(callbacks=[Recorder()])
        assert seen == ["begin", "epoch", "end"]

    def test_a_failing_callback_does_not_break_the_run(self, project, sentiment):
        class Exploding(Callback):
            def on_epoch_end(self, run, epoch, metrics, **kwargs):
                raise RuntimeError("instrumentation should never kill a run")

        run = sentiment().train(callbacks=[Exploding()])
        assert run.refresh().status == "finished"

    def test_early_stopping_sets_the_stop_flag(self):
        class FakeRun:
            should_stop = False

            def add_tag(self, tag):
                pass

        stopper = EarlyStopping(monitor="val_loss", patience=2, mode="min")
        run = FakeRun()
        stopper.on_epoch_end(run, 1, {"val_loss": 1.0})
        stopper.on_epoch_end(run, 2, {"val_loss": 2.0})
        assert run.should_stop is False
        stopper.on_epoch_end(run, 3, {"val_loss": 2.0})
        assert run.should_stop is True

    def test_callback_list_isolates_failures(self):
        class Bad(Callback):
            def on_train_begin(self, run, model, **kwargs):
                raise RuntimeError

        class Good(Callback):
            called = False

            def on_train_begin(self, run, model, **kwargs):
                type(self).called = True

        CallbackList([Bad(), Good()]).emit("on_train_begin", None, None)
        assert Good.called is True


class TestMetrics:
    def test_accuracy(self):
        assert metrics.accuracy(["a", "b"], ["a", "b"]) == 1.0
        assert metrics.accuracy(["a", "b"], ["a", "a"]) == 0.5

    def test_f1_is_between_zero_and_one(self):
        value = metrics.f1(["a", "b", "a"], ["a", "b", "b"])
        assert 0.0 <= value <= 1.0

    def test_classification_report_shape(self):
        report = metrics.classification_report(["a", "b"], ["a", "b"])
        assert set(report) >= {"accuracy", "f1_macro", "per_class", "confusion", "support"}

    def test_regression_metrics(self):
        assert metrics.mse([1.0, 2.0], [1.0, 2.0]) == 0.0
        assert metrics.r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_flatten_keeps_only_scalars(self):
        flat = metrics.flatten_report(metrics.classification_report(["a"], ["a"]))
        assert "per_class" not in flat
        assert flat["accuracy"] == 1.0

    def test_empty_input_does_not_crash(self):
        assert metrics.accuracy([], []) == 0.0
        assert metrics.mse([], []) == 0.0


class TestImportances:
    def test_a_text_pipeline_names_words_not_column_numbers(self, project, sentiment):
        """The whole point: a vectoriser's columns get their real names back."""
        model = sentiment.fit()
        weights = model._version.importances

        assert weights, "the sklearn backend should have reported weights"
        assert any(word in weights for word in ("great", "terrible", "movie"))
        assert not any(name.startswith("feature_") for name in weights)

    def test_the_sign_of_a_binary_coefficient_survives(self, project, sentiment):
        """Direction is the difference between 'matters' and 'means negative'."""
        weights = sentiment.fit()._version.importances
        assert min(weights.values()) < 0 < max(weights.values())

    def test_the_stored_list_is_capped(self, project, sentiment):
        from mlango.training.backends.sklearn_backend import TOP_FEATURES

        weights = sentiment.fit(max_features=500)._version.importances
        assert 0 < len(weights) <= TOP_FEATURES

    def test_declared_columns_name_themselves(self, project, reviews, isolated_registry):
        """No vectoriser in the pipeline, so the names come from the fields."""
        pytest.importorskip("sklearn")

        class Stars(Model):
            class Meta:
                dataset = reviews
                trainer = "sklearn"
                task = "regression"
                features = ["id", "stars"]
                target = "stars"

            def build(self):
                from sklearn.linear_model import LinearRegression

                return LinearRegression()

        assert set(Stars.fit()._version.importances) == {"id", "stars"}

    def test_a_backend_that_cannot_explain_says_so(self):
        from mlango.training.trainer import Trainer

        assert Trainer.importances(object(), None, None) is None

    def test_a_raising_backend_does_not_lose_the_run(self, project, sentiment, monkeypatch):
        """An explanation is a nice-to-have; the trained model is not."""
        from mlango.training.backends.sklearn_backend import SklearnTrainer

        def boom(self, model, fitted):
            raise RuntimeError("estimator has no idea")

        monkeypatch.setattr(SklearnTrainer, "importances", boom)

        model = sentiment.fit()
        assert model._version.version >= 1
        assert model._version.importances is None


class TestBaseline:
    def test_training_records_what_it_was_fitted_on(self, project, sentiment):
        baseline = sentiment.fit()._version.baseline
        assert set(baseline) == {"text", "label"}
        assert baseline["label"]["kind"] == "categorical"
        assert baseline["text"]["kind"] == "text"

    def test_the_baseline_covers_the_training_split_only(self, project, sentiment):
        """A profile of all 100 rows would describe data the model never saw."""
        baseline = sentiment.fit(splits={"train": 0.5, "val": 0.5})._version.baseline
        assert baseline["label"]["count"] < 100

    def test_a_failing_profile_does_not_lose_the_run(self, project, sentiment, monkeypatch):
        from mlango.training import drift

        monkeypatch.setattr(
            drift, "profile", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("nope"))
        )
        model = sentiment.fit()
        assert model._version.version >= 1
        assert model._version.baseline is None


class TestPredictionLog:
    @pytest.fixture
    def logging_on(self, project):
        from mlango.conf import settings

        settings.PREDICTION_LOG = {"ENABLED": True, "SAMPLE": 1.0, "MAX_ROWS": 0}
        yield
        settings.PREDICTION_LOG = {"ENABLED": False, "SAMPLE": 1.0, "MAX_ROWS": 100_000}

    def _logged(self):
        import sqlalchemy as sa

        from mlango.metastore.models import Prediction
        from mlango.metastore.session import session_scope

        with session_scope() as session:
            return list(session.execute(sa.select(Prediction)).scalars())

    def test_nothing_is_recorded_by_default(self, project, sentiment):
        sentiment.fit().predict(["great movie", "awful film"])
        assert self._logged() == []

    def test_a_batch_records_one_row_per_input(self, logging_on, sentiment):
        model = sentiment.fit()
        model.predict(["great movie", "awful film"])

        rows = self._logged()
        assert len(rows) == 2
        assert {row.inputs for row in rows} == {"great movie", "awful film"}
        assert all(row.label == sentiment._meta.label for row in rows)
        assert all(row.version == model._version.version for row in rows)
        assert all(row.output in {"pos", "neg"} for row in rows)

    def test_latency_is_per_row_not_per_batch(self, logging_on, sentiment):
        model = sentiment.fit()
        model.predict(["a movie"] * 10)
        latencies = [row.latency_ms for row in self._logged()]
        assert all(value is not None and value >= 0 for value in latencies)
        assert len(set(latencies)) == 1, "one call, so every row gets the same share"

    def test_sampling_keeps_a_fraction(self, logging_on, sentiment):
        from mlango.conf import settings

        settings.PREDICTION_LOG = {"ENABLED": True, "SAMPLE": 0.0, "MAX_ROWS": 0}
        sentiment.fit().predict(["great movie"] * 50)
        assert self._logged() == []

    def test_max_rows_trims_the_oldest(self, logging_on, sentiment):
        from mlango.conf import settings

        settings.PREDICTION_LOG = {"ENABLED": True, "SAMPLE": 1.0, "MAX_ROWS": 5}
        model = sentiment.fit()
        for index in range(12):
            model.predict(f"movie number {index}")

        rows = self._logged()
        assert len(rows) == 5
        assert "movie number 11" in {row.inputs for row in rows}

    def test_a_broken_metastore_does_not_break_a_prediction(self, logging_on, sentiment):
        """Observability must never be able to take an endpoint down."""
        from mlango.metastore import session as session_module

        model = sentiment.fit()
        original = session_module.session_scope

        def refuse(*args, **kwargs):
            raise RuntimeError("database is gone")

        session_module.session_scope = refuse
        try:
            assert model.predict("great movie") in {"pos", "neg"}
        finally:
            session_module.session_scope = original

    def test_logged_inputs_can_be_compared_against_the_baseline(self, logging_on, sentiment):
        """The whole point: what was logged lines up with what was profiled."""
        from mlango.training import drift

        model = sentiment.fit()
        model.predict(["ok"] * 40)

        baseline = {"text": model._version.baseline["text"]}
        scores = drift.compare(baseline, [row.inputs for row in self._logged()])
        assert scores["text"]["verdict"] == "significant"
