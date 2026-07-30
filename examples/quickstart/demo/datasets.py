"""Datasets for the demo app.

The data is generated in code, so a fresh project has something to train on
without downloading anything.
"""

import random

from mlango.core import fields
from mlango.data import Dataset, PythonSource

POSITIVE = [
    "great movie, loved every minute",
    "excellent film with wonderful acting",
    "brilliant story and a strong cast",
    "genuinely delightful, would watch again",
    "beautifully shot and well paced",
]
NEGATIVE = [
    "terrible movie, a waste of time",
    "awful film with bad acting",
    "boring story and a weak cast",
    "genuinely dull, would not watch again",
    "poorly shot and badly paced",
]

N_ROWS = 400


def generate_reviews():
    """Yield a deterministic set of synthetic reviews."""
    rng = random.Random(0)
    for index in range(N_ROWS):
        positive = index % 2 == 0
        phrase = rng.choice(POSITIVE if positive else NEGATIVE)
        yield {
            "id": index,
            "text": f"{phrase} ({index})",
            "label": "pos" if positive else "neg",
        }


class Reviews(Dataset):
    """Synthetic product reviews, generated in code."""

    id = fields.IntegerField()
    text = fields.TextField()
    label = fields.LabelField(["neg", "pos"])

    class Meta:
        source = PythonSource(generate_reviews, count=N_ROWS)
        # Splits hash this field, so the train/test division is stable even
        # when rows are added later.
        primary_key = "id"
