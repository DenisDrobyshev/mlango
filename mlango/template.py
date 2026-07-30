"""Project and app scaffolding.

Django ships ``.py-tpl`` files so the templates are not byte-compiled by
accident; keeping the templates as plain strings here avoids that problem
entirely, packages cleanly, and means the scaffold is reviewable in one file.

The deliberate difference from Django: ``startproject`` produces a project that
already *does something*. A newcomer runs four commands and gets an admin with a
trained model, real metrics and an agent trace in it. An empty scaffold is a
worse first five minutes, and first five minutes are the whole ballgame.
"""

from __future__ import annotations

import os
import secrets
from typing import Any

#: Replaced with the project package name.
PROJECT = "__PROJECT__"
#: Replaced with the app package name.
APP = "__APP__"
#: Replaced with a freshly generated secret.
SECRET = "__SECRET__"


# --------------------------------------------------------------------------- #
# Project files
# --------------------------------------------------------------------------- #

MANAGE_PY = '''#!/usr/bin/env python
"""Command-line entry point for this mlango project."""

import os
import sys


def main() -> None:
    os.environ.setdefault("MLANGO_SETTINGS_MODULE", "__PROJECT__.settings")
    try:
        from mlango.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Could not import mlango. Is it installed and is your virtual "
            "environment active? Try: pip install mlango"
        ) from exc
    sys.exit(execute_from_command_line(sys.argv))


if __name__ == "__main__":
    main()
'''

SETTINGS_PY = '''"""Settings for the __PROJECT__ project.

Every available setting and its default lives in
``mlango.conf.global_settings`` — override here only what differs.
"""

from pathlib import Path

# Everything relative (the SQLite file, artifacts, data files) resolves from here.
BASE_DIR = Path(__file__).resolve().parent.parent

# Keep this out of version control in production.
SECRET_KEY = "__SECRET__"

DEBUG = True

# Apps whose datasets, models, agents, evals and admin are loaded at startup.
INSTALLED_APPS = [
    "demo",
]

# Runs, metrics, artifacts, dataset/model versions and agent traces live here.
# SQLite needs no setup; point URL at Postgres when a team shares the project.
METASTORE = {
    "URL": "sqlite:///mlango.db",
}

# Where checkpoints and materialised datasets are written.
STORAGE = {
    "BACKEND": "mlango.storage.local.LocalStorage",
    "ROOT": "artifacts",
}

# Module holding `urlpatterns` for the inference API.
ROOT_ROUTECONF = "__PROJECT__.routes"

# "echo" is a deterministic offline provider: agents work with no API key, so a
# fresh checkout runs and the test suite stays free. Switch to "anthropic" and
# export ANTHROPIC_API_KEY when you want a real model.
DEFAULT_PROVIDER = "echo"
DEFAULT_AGENT_MODEL = "claude-opus-5"

# Applied to every training run. Metric recording is built in, so this list is
# purely additive.
DEFAULT_CALLBACKS = [
    "mlango.training.callbacks.ProgressBar",
]

SEED = 1337
LOG_LEVEL = "INFO"
'''

ROUTES_PY = '''"""Inference API routes — the ``urls.py`` of an mlango project."""

from mlango.serve import path

from demo.agents import Helper
from demo.models import Sentiment

urlpatterns = [
    # POST /api/predict/  {"input": "great movie"}
    path("predict/", Sentiment.as_endpoint(), name="sentiment-predict"),
    # POST /api/chat/     {"message": "hello", "session_id": "abc"}
    path("chat/", Helper.as_endpoint(), name="helper-chat"),
]
'''

PROJECT_INIT_PY = ""

GITIGNORE = """# mlango
mlango.db
mlango.db-wal
mlango.db-shm
artifacts/

# Python
__pycache__/
*.py[cod]
.venv/
venv/
env/
*.egg-info/
dist/
build/

# Tooling
.pytest_cache/
.ruff_cache/
.mypy_cache/
.coverage
htmlcov/

# Editors and OS
.idea/
.vscode/
.DS_Store
"""

REQUIREMENTS = """mlango[sklearn]
"""

PROJECT_README = """# __PROJECT__

An [mlango](https://github.com/DenisDrobyshev/mlango) project.

## Get running

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py train demo.Sentiment
python manage.py runserver
```

Then open <http://127.0.0.1:8000/admin/> — the dataset, the trained model, its
metrics and the agent traces are all there.

## What is in here

| Path | What it holds |
|---|---|
| `__PROJECT__/settings.py` | Project settings: installed apps, metastore, storage, provider |
| `__PROJECT__/routes.py` | Inference API routes |
| `demo/datasets.py` | A `Dataset` declaration |
| `demo/models.py` | A `Model` declaration with hyperparameters as fields |
| `demo/agents.py` | An `Agent` with a tool |
| `demo/evals.py` | An `Eval` suite scoring the model |
| `demo/admin.py` | Admin customisation for the dataset |

## Commands worth knowing

```bash
python manage.py check                      # validate the project
python manage.py dataset head demo.Reviews  # peek at the data
python manage.py train demo.Sentiment -p C=2.0
python manage.py runs list                  # what has run
python manage.py runs compare <id> <id>     # what changed between two runs
python manage.py evaluate demo.SentimentAccuracy
python manage.py agent demo.Helper          # interactive agent session
python manage.py traces list                # agent traces
python manage.py shell                      # shell with everything imported
```

## Using a real LLM

The project ships with the offline `echo` provider so it runs with no
credentials. To use Claude:

```bash
export ANTHROPIC_API_KEY=...
```

and set `DEFAULT_PROVIDER = "anthropic"` in `__PROJECT__/settings.py`.
"""


# --------------------------------------------------------------------------- #
# Demo app — the part that makes a fresh project non-empty
# --------------------------------------------------------------------------- #

DEMO_APPS_PY = '''"""App configuration for the demo app."""

from mlango.core import AppConfig


class DemoConfig(AppConfig):
    name = "demo"
    verbose_name = "Demo"
'''

DEMO_DATASETS_PY = '''"""Datasets for the demo app.

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
'''

DEMO_MODELS_PY = '''"""Models for the demo app."""

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
'''

DEMO_AGENTS_PY = '''"""Agents for the demo app."""

from mlango.agents import Agent, BufferMemory, tool


@tool
def classify_review(text: str) -> str:
    """Classify a product review as positive or negative.

    Args:
        text: The review text to classify.
    """
    from demo.models import Sentiment

    try:
        model = Sentiment.load()
    except LookupError:
        return "No trained model yet. Run: python manage.py train demo.Sentiment"
    return str(model.predict(text))


class Helper(Agent):
    """Answers questions about reviews and can call the trained classifier."""

    class Meta:
        system = (
            "You help analyse product reviews. When the user gives you review "
            "text and asks for a verdict, use the classify_review tool rather "
            "than guessing."
        )
        tools = [classify_review]
        memory = BufferMemory(k=20)
        max_steps = 6
'''

DEMO_EVALS_PY = '''"""Evaluation suites for the demo app."""

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
'''

TESTS_INIT_PY = ""

DEMO_TESTS_PY = '''"""Tests for the demo app.

``python manage.py test`` runs these against a throwaway metastore and artifact
store, so a test run can never touch the data or the runs you care about — the
same guarantee Django gives you with a test database.

These four are the patterns worth copying: query a dataset, train a model,
score an eval, and talk to an agent.
"""

import pytest

from demo.agents import Helper
from demo.datasets import Reviews
from demo.evals import SentimentAccuracy
from demo.models import Sentiment


def test_the_dataset_loads_and_is_labelled():
    assert Reviews.objects.count() > 0
    assert set(Reviews.objects.values_list("label", flat=True)) == {"neg", "pos"}


def test_lookups_narrow_the_queryset():
    positive = Reviews.objects.filter(label="pos")
    assert 0 < positive.count() < Reviews.objects.count()
    assert all(record.label == "pos" for record in positive.take(5))


def test_splits_do_not_overlap():
    parts = Reviews.objects.split(train=0.8, test=0.2)
    train_ids = set(parts["train"].values_list("id", flat=True))
    test_ids = set(parts["test"].values_list("id", flat=True))
    assert train_ids and test_ids
    assert train_ids & test_ids == set()


@pytest.fixture(scope="module")
def trained():
    """Train once and reuse it — training is the slow part."""
    model = Sentiment()
    model.train()
    return model


def test_training_records_a_run(trained):
    from mlango.training import recent_runs

    run = recent_runs(limit=1, kind="train")[0]
    assert run.status == "finished"
    assert run.summary.get("accuracy", 0) > 0.5


def test_the_model_predicts_a_declared_class(trained):
    assert trained.predict("an absolute delight, loved it") in {"neg", "pos"}


def test_a_saved_model_can_be_loaded_back(trained):
    assert Sentiment.load().predict("dreadful, a waste of time") in {"neg", "pos"}


def test_the_eval_passes_its_threshold(trained):
    report = SentimentAccuracy.evaluate()
    assert report.passed, report.summary()


def test_the_agent_answers():
    result = Helper().run("hello")
    assert result.output
    assert result.error == ""
'''

APP_TESTS_PY = '''"""Tests for the __APP__ app.

``python manage.py test`` runs these against a throwaway metastore and artifact
store, so nothing here can touch your real runs.
"""


def test_placeholder():
    """Replace me: query a dataset, train a model, or score an eval.

        from __APP__.datasets import MyDataset

        def test_the_dataset_loads():
            assert MyDataset.objects.count() > 0
    """
'''

DEMO_ADMIN_PY = '''"""Admin customisation for the demo app.

Everything declared shows up in the admin without being registered. Register
explicitly only to change how it is presented.
"""

from mlango import admin

from demo.datasets import Reviews


@admin.register(Reviews)
class ReviewsAdmin(admin.ObjectAdmin):
    list_display = ("id", "text", "label")
    list_filter = ("label",)
    search_fields = ("text",)
    list_per_page = 25
'''


# --------------------------------------------------------------------------- #
# Generic app template (startapp)
# --------------------------------------------------------------------------- #

APP_APPS_PY = '''"""App configuration for the __APP__ app."""

from mlango.core import AppConfig


class __CLASS__Config(AppConfig):
    name = "__APP__"
    verbose_name = "__TITLE__"

    def ready(self) -> None:
        """Runs once every app is loaded — wire signal receivers here."""
'''

APP_DATASETS_PY = '''"""Datasets for the __APP__ app."""

from mlango.core import fields  # noqa: F401
from mlango.data import Dataset, JSONLSource  # noqa: F401

# class MyDataset(Dataset):
#     """One line describing what these records are."""
#
#     id = fields.IntegerField()
#     text = fields.TextField()
#     label = fields.LabelField(["a", "b"])
#
#     class Meta:
#         source = JSONLSource("data/my_dataset.jsonl")
#         primary_key = "id"
'''

APP_MODELS_PY = '''"""Models for the __APP__ app."""

from mlango.core import fields  # noqa: F401
from mlango.training import Model  # noqa: F401

# class MyModel(Model):
#     """One line describing the approach."""
#
#     learning_rate = fields.FloatField(default=1e-3, tunable=True)
#
#     class Meta:
#         dataset = MyDataset
#         trainer = "sklearn"        # or "torch"
#         task = "classification"    # or "regression"
#         features = ["text"]
#
#     def build(self):
#         ...
'''

APP_AGENTS_PY = '''"""Agents for the __APP__ app."""

from mlango.agents import Agent, tool  # noqa: F401

# @tool
# def my_tool(query: str) -> str:
#     """What the tool does — the model reads this.
#
#     Args:
#         query: What to look for.
#     """
#     return "..."
#
#
# class MyAgent(Agent):
#     """One line describing the agent's job."""
#
#     class Meta:
#         system = "You are ..."
#         tools = [my_tool]
'''

APP_EVALS_PY = '''"""Evaluation suites for the __APP__ app."""

from mlango.evals import Eval, exact_match  # noqa: F401

# class MyEval(Eval):
#     """What "good" means for this task."""
#
#     class Meta:
#         dataset = MyCases
#         target = MyAgent
#         input_field = "question"
#         expected_field = "answer"
#         scorers = {"correct": exact_match}
#         threshold = 0.8
'''

APP_ADMIN_PY = '''"""Admin customisation for the __APP__ app.

Declared objects appear in the admin automatically; register explicitly only to
change columns, filters or search.
"""

from mlango import admin  # noqa: F401

# @admin.register(MyDataset)
# class MyDatasetAdmin(admin.ObjectAdmin):
#     list_display = ("id", "text", "label")
#     list_filter = ("label",)
#     search_fields = ("text",)
'''


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _write(path: str, content: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    return path


def _substitute(content: str, **values: str) -> str:
    for token, value in values.items():
        content = content.replace(token, value)
    return content


def render_project(name: str, target: str, *, demo: bool = True) -> list[str]:
    """Write a new project into ``target``. Returns the paths created."""
    secret = secrets.token_urlsafe(48)
    subs: dict[str, Any] = {PROJECT: name, SECRET: secret}

    tree: dict[str, str] = {
        "manage.py": MANAGE_PY,
        f"{name}/__init__.py": PROJECT_INIT_PY,
        f"{name}/settings.py": SETTINGS_PY,
        ".gitignore": GITIGNORE,
        "requirements.txt": REQUIREMENTS,
        "README.md": PROJECT_README,
    }

    if demo:
        tree[f"{name}/routes.py"] = ROUTES_PY
        tree.update(
            {
                "demo/__init__.py": "",
                "demo/apps.py": DEMO_APPS_PY,
                "demo/datasets.py": DEMO_DATASETS_PY,
                "demo/models.py": DEMO_MODELS_PY,
                "demo/agents.py": DEMO_AGENTS_PY,
                "demo/evals.py": DEMO_EVALS_PY,
                "demo/admin.py": DEMO_ADMIN_PY,
                "demo/migrations/__init__.py": "",
                "tests/__init__.py": TESTS_INIT_PY,
                "tests/test_demo.py": DEMO_TESTS_PY,
            }
        )
    else:
        tree[f"{name}/routes.py"] = (
            '"""Inference API routes — the ``urls.py`` of an mlango project."""\n\n'
            "from mlango.serve import path  # noqa: F401\n\n"
            "urlpatterns = [\n"
            '    # path("predict/", MyModel.as_endpoint()),\n'
            "]\n"
        )
        tree[f"{name}/settings.py"] = SETTINGS_PY.replace(
            '    "demo",\n', "    # Add your apps here.\n"
        )
        tree["README.md"] = PROJECT_README.replace("python manage.py train demo.Sentiment\n", "")
        # Even an empty project gets somewhere for `manage.py test` to look, so
        # the command works before anything else has been declared.
        tree["tests/__init__.py"] = TESTS_INIT_PY
        tree["tests/test_project.py"] = _substitute(APP_TESTS_PY, **{APP: name})

    created = []
    for relative, content in tree.items():
        created.append(_write(os.path.join(target, relative), _substitute(content, **subs)))
    return sorted(created)


def render_app(name: str, target: str) -> list[str]:
    """Write a new app into ``target``. Returns the paths created."""
    class_name = "".join(part.capitalize() for part in name.split("_"))
    title = name.replace("_", " ").title()
    subs = {APP: name, "__CLASS__": class_name, "__TITLE__": title}

    tree = {
        "__init__.py": "",
        "apps.py": APP_APPS_PY,
        "datasets.py": APP_DATASETS_PY,
        "models.py": APP_MODELS_PY,
        "agents.py": APP_AGENTS_PY,
        "evals.py": APP_EVALS_PY,
        "admin.py": APP_ADMIN_PY,
        "migrations/__init__.py": "",
        "tests.py": APP_TESTS_PY,
    }

    created = []
    for relative, content in tree.items():
        created.append(_write(os.path.join(target, relative), _substitute(content, **subs)))
    return sorted(created)


__all__ = ["render_project", "render_app"]
