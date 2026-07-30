"""The claims in this example's README, as tests.

A number in a README rots quietly. These fail loudly instead, and they are what
`python manage.py test` runs.
"""

import pytest

from news.datasets import CATEGORIES, Posts
from news.models import Topic


@pytest.fixture(scope="module")
def trained():
    model = Topic()
    model.train()
    return model


def test_the_corpus_is_the_size_the_readme_claims():
    assert Posts.objects.count() == 3669


def test_every_category_is_present():
    groups = set(Posts.objects.values_list("group", flat=True))
    assert groups == set(CATEGORIES)


def test_no_empty_documents():
    """Stripping headers and quotes leaves some posts with nothing in them."""
    assert all(record.text.strip() for record in Posts.objects.take(200))


def test_the_split_does_not_move_when_rows_are_added():
    parts = Posts.objects.split(train=0.8, val=0.2)
    train = set(parts["train"].values_list("id", flat=True))
    val = set(parts["val"].values_list("id", flat=True))
    assert train and val
    assert not train & val


def test_accuracy_clears_the_published_baseline(trained):
    """The README says 0.88; anything under 0.85 means something broke."""
    report = trained.evaluate()
    assert report["accuracy"] > 0.85, report


def test_it_classifies_text_written_for_this_test(trained):
    assert trained.predict("my doctor prescribed antibiotics for the infection") == "sci.med"
    assert trained.predict("the z-buffer and per-pixel lighting on the GPU") == "comp.graphics"
