"""Shared fixtures.

Settings and the app registry are process-global by design — that is what makes
``manage.py`` work — so tests that touch them run against a temporary project
and clean up after themselves.
"""

from __future__ import annotations

import pytest

from mlango.core import fields
from mlango.data import Dataset, InMemorySource

ROWS = [
    {
        "id": index,
        "text": ("great movie" if index % 2 == 0 else "terrible movie") + f" {index}",
        "label": "pos" if index % 2 == 0 else "neg",
        "stars": (index % 5) + 1,
    }
    for index in range(100)
]


class Reviews(Dataset):
    """Reviews used across the test suite."""

    id = fields.IntegerField()
    text = fields.TextField()
    label = fields.LabelField(["neg", "pos"])
    stars = fields.IntegerField(min_value=1, max_value=5)

    class Meta:
        source = InMemorySource(ROWS)
        primary_key = "id"


@pytest.fixture
def project(tmp_path):
    """A configured, isolated project rooted at ``tmp_path``."""
    from mlango.agents.providers import clear_provider_cache
    from mlango.conf import settings
    from mlango.metastore.session import dispose_all
    from mlango.storage import reset_default_storage
    from mlango.training.trainer import clear_trainer_cache

    dispose_all()
    reset_default_storage()
    clear_provider_cache()
    clear_trainer_cache()

    settings.configure(
        BASE_DIR=str(tmp_path),
        SECRET_KEY="test-secret",
        METASTORE={"URL": "sqlite:///test.db"},
        STORAGE={"BACKEND": "mlango.storage.local.LocalStorage", "ROOT": "artifacts"},
        DEFAULT_PROVIDER="echo",
        DEFAULT_CALLBACKS=[],
        SEED=0,
        INSTALLED_APPS=[],
    )

    yield tmp_path

    dispose_all()
    reset_default_storage()
    clear_provider_cache()
    clear_trainer_cache()
    settings.reset()


@pytest.fixture(scope="session")
def reviews():
    """The shared dataset class.

    Session-scoped and stateless so module-scoped fixtures can depend on it —
    declaring a class twice would clash on its label.
    """
    return Reviews


@pytest.fixture(scope="session")
def sklearn_or_skip():
    return pytest.importorskip("sklearn")
