"""The 20 Newsgroups benchmark, declared as an mlango dataset.

scikit-learn ships the fetcher, so there is no download URL to rot and no
credentials to arrange. The first run pulls about 14 MB into scikit-learn's own
cache; every run after that is offline.

Four of the twenty groups, which is the subset the literature usually reports
on: two that are easy to tell apart and two that are not, so the number means
something.
"""

from mlango.core import fields
from mlango.data import Dataset

CATEGORIES = [
    "alt.atheism",
    "comp.graphics",
    "sci.med",
    "soc.religion.christian",
]


def load_posts():
    """Yield one record per document, train and test together.

    mlango assigns the split itself by hashing each record's key. Keeping
    scikit-learn's own train/test division would leave two sources of truth
    about which rows are held out, and only one of them would be recorded on
    the run.
    """
    from sklearn.datasets import fetch_20newsgroups

    index = 0
    for subset in ("train", "test"):
        bunch = fetch_20newsgroups(
            subset=subset,
            categories=CATEGORIES,
            # The stock text carries headers, quoted replies and signatures, and
            # a classifier will happily learn the sender's address instead of
            # the topic. Removing them is what makes the accuracy a claim about
            # the writing rather than about the metadata.
            remove=("headers", "footers", "quotes"),
            shuffle=False,
        )
        for text, target in zip(bunch.data, bunch.target, strict=True):
            index += 1
            body = text.strip()
            if not body:
                continue
            yield {
                "id": index,
                "text": body,
                "group": bunch.target_names[target],
            }


class Posts(Dataset):
    """Usenet posts from four newsgroups, labelled by group."""

    id = fields.IntegerField()
    text = fields.TextField()
    group = fields.LabelField(CATEGORIES)

    class Meta:
        source = load_posts
        primary_key = "id"
        license = "public domain"
        homepage = "http://qwone.com/~jason/20Newsgroups/"
