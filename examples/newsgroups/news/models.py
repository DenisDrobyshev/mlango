"""A linear text classifier over the newsgroup posts.

TF-IDF into a linear model is the standard baseline for this benchmark. It is
here because it is the honest comparison point: anything more elaborate should
have to beat it, and on this dataset most things do not by much.
"""

from mlango.core import fields
from mlango.training import Model

from news.datasets import Posts


class Topic(Model):
    """TF-IDF over the post body, into logistic regression."""

    max_features = fields.IntegerField(default=50_000, min_value=1, tunable=True)
    ngram_max = fields.IntegerField(default=2, min_value=1, max_value=3, tunable=True)
    min_df = fields.IntegerField(default=2, min_value=1, tunable=True)
    C = fields.FloatField(default=4.0, min_value=0.0, tunable=True)

    class Meta:
        dataset = Posts
        trainer = "sklearn"
        task = "classification"
        # Explicit, so the primary key can never be mistaken for a feature.
        features = ["text"]
        splits = {"train": 0.8, "val": 0.2}
        monitor = "accuracy"
        monitor_mode = "max"

    def build(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        return make_pipeline(
            TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, self.ngram_max),
                min_df=self.min_df,
                sublinear_tf=True,
                strip_accents="unicode",
                stop_words="english",
            ),
            LogisticRegression(C=self.C, max_iter=2000),
        )
