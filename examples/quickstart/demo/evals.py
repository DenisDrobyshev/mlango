"""Evaluation suites for the demo app."""

from mlango.evals import Eval, exact_match

from demo.datasets import Reviews
from demo.models import Sentiment


class SentimentAccuracy(Eval):
    """Does the trained classifier agree with the labels?"""

    class Meta:
        dataset = Reviews
        target = Sentiment
        input_field = "text"
        expected_field = "label"
        case_id_field = "id"
        scorers = {"correct": exact_match}
        max_cases = 100
        threshold = 1.0
