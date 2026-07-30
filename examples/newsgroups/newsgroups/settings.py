"""Settings for the newsgroups project.

Every available setting and its default lives in
``mlango.conf.global_settings`` — override here only what differs.
"""

from pathlib import Path

# Everything relative (the SQLite file, artifacts, data files) resolves from here.
BASE_DIR = Path(__file__).resolve().parent.parent

# Keep this out of version control in production.
SECRET_KEY = "e9GdcaYlKsEltM8qCi9dyKo1pR7hBP57P5s_Y3GKMDtR0FkdxZo7_YfUbf2VfOko"

DEBUG = True

# Apps whose datasets, models, agents, evals and admin are loaded at startup.
INSTALLED_APPS = [
    "news",
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
ROOT_ROUTECONF = "newsgroups.routes"

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
