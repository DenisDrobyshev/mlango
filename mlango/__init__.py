"""mlango — a batteries-included framework for machine learning and LLM agents.

The design goal is simple: everything that made Django pleasant for web
development should be available for ML work. Declarative classes with a
``_meta``, an app registry, migrations, an auto-generated admin, management
commands and a settings module — but the nouns are datasets, models, runs,
agents and evaluations instead of tables and views.

Typical entry point::

    import mlango
    mlango.setup()
"""

__version__ = "0.1.0"

from mlango.core.registry import apps

__all__ = ["__version__", "apps", "setup", "get_version"]


def get_version() -> str:
    return __version__


def setup(settings_module: str | None = None, set_prefix: bool = True) -> None:
    """Bootstrap the framework.

    Loads the settings module, then populates the application registry, which
    in turn imports every declarative module of every installed app so that
    datasets, models, agents and evals register themselves.

    ``mlango.setup()`` is idempotent, so calling it from a script, a notebook
    or a test fixture is safe.
    """
    from mlango.conf import settings

    if settings_module is not None:
        settings.configure_from_module(settings_module)

    apps.populate(settings.INSTALLED_APPS)
