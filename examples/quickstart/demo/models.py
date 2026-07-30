"""Models for the demo app."""

from mlango.core import fields
from mlango.training import Model

from demo.datasets import Reviews


class Sentiment(Model):
    """TF-IDF features into logistic regression."""

    # Hyperparameters are fields: validated, defaulted, recorded on every run
    # and sweepable. `tunable` marks the ones a sweep may vary.
    max_features = fields.IntegerField(default=5000, min_value=1, tunable=True)
    C = fields.FloatField(default=1.0, min_value=0.0, tunable=True)

    class Meta:
        dataset = Reviews
        trainer = "sklearn"
        task = "classification"
        # Which dataset fields feed the model. Without this, `id` would be
        # treated as a feature — a classic way to leak the answer.
        features = ["text"]
        splits = {"train": 0.8, "val": 0.2}

    def build(self):
        """Return the estimator. The trainer handles everything around it."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        return make_pipeline(
            TfidfVectorizer(max_features=self.max_features),
            LogisticRegression(C=self.C, max_iter=1000),
        )
