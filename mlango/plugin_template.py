"""Scaffolding for a distributable extension.

Separate from :mod:`mlango.template` because the output is a different kind of
thing: that module scaffolds a project you run, this one scaffolds a package you
publish. The packaging metadata is most of the value — the entry-point stanza is
what turns "here is a class, add it to your settings" into "pip install it" —
and getting that stanza wrong is the usual reason an extension never ships.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from typing import Any

#: Replaced with the distribution name, e.g. ``mlango-lightgbm``.
DIST = "__DIST__"
#: Replaced with the import package, e.g. ``mlango_lightgbm``.
MODULE = "__MODULE__"
#: Replaced with the name the extension registers itself under.
ENTRY = "__ENTRY__"

#: Extension points a package can ship.
KINDS = ("trainer", "provider", "storage", "source")

#: Which kinds are discovered by entry point, and under which group. Storage and
#: sources are absent on purpose: a project has one storage backend, named in
#: settings, and a source is imported and named in a declaration — neither has
#: anything for a registry to resolve.
GROUPS = {
    "trainer": "mlango.trainers",
    "provider": "mlango.providers",
}


PYPROJECT = """[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "__DIST__"
version = "0.1.0"
description = "A __KIND__ for mlango."
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
keywords = ["mlango", "__KIND__", "machine-learning"]
dependencies = [
    "mlango>=0.1",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.6"]

[project.urls]
Homepage = "https://example.com/__DIST__"
__ENTRY_POINTS__
[tool.hatch.build.targets.wheel]
packages = ["src/__MODULE__"]

[tool.ruff]
line-length = 100
"""

ENTRY_POINTS = """
[project.entry-points."__GROUP__"]
# This is what makes `pip install __DIST__` enough. mlango reads this group when
# settings are loaded, so a project can name "__ENTRY__" without editing any.
__ENTRY__ = "__MODULE__.__KIND__:__CLASS__"
"""

INIT_PY = '''"""__DIST__ — a __KIND__ for mlango."""

from __MODULE__.__KIND__ import __CLASS__

__all__ = ["__CLASS__"]
__version__ = "0.1.0"
'''

TRAINER_PY = '''"""The trainer.

A trainer knows how to fit, predict, save and load. Run tracking, metrics,
callbacks, versioning and the admin belong to the framework and are already
written, which is why this file is the whole package.
"""

from __future__ import annotations

from typing import Any

from mlango.training import Trainer


class __CLASS__(Trainer):
    name = "__ENTRY__"
    #: Checked before use, so a missing dependency is a clear message instead of
    #: an ImportError from somewhere inside fit().
    requires = ()
    extension = "bin"

    def fit(self, model, train, validation, run, callbacks, *, target="", features=None, **kwargs):
        estimator = model.build()
        x, y = train.xy(target=target or None, features=features)

        estimator.fit(x, y)

        # Metrics belong to the run, not to this backend. Logging them here is
        # what puts this trainer in the same history, admin and comparison view
        # as every other one.
        run.log_metrics({"train_score": float(estimator.score(x, y))}, epoch=0, step=0)
        callbacks.emit("on_epoch_end", run, 0, {}, model=model, trainer=self, fitted=estimator)
        return estimator

    def predict(self, model, fitted, inputs: list[Any]) -> list[Any]:
        return list(fitted.predict(inputs))

    def save(self, model, fitted, name: str) -> str:
        import joblib

        from mlango.storage import default_storage

        # writable() hands back a local path and publishes when the block exits,
        # so this keeps working when the project moves its artifacts to S3.
        with default_storage().writable(f"{name}.{self.extension}") as target:
            joblib.dump(fitted, target.path)
            return target.name

    def load(self, model, path: str):
        import joblib

        from mlango.storage import default_storage

        with default_storage().readable(path) as local:
            return joblib.load(local)

    # -- optional ------------------------------------------------------------

    def describe(self, model, fitted) -> dict[str, Any]:
        """Backend detail shown on the run page."""
        return {"backend": self.name, "estimator": type(fitted).__name__}

    def importances(self, model, fitted) -> dict[str, float] | None:
        """Feature weights, if this backend can name them.

        None is the right answer when the weights do not map onto anything a
        person would call a feature. A wrong explanation is worse than none.
        """
        return None
'''

PROVIDER_PY = '''"""The provider.

A provider has one method. The agent loop, tool dispatch, memory and tracing
belong to the framework, so changing provider cannot change how an agent
behaves — which is the point of the contract being this narrow.
"""

from __future__ import annotations

from typing import Any

from mlango.agents.providers.base import Completion, Provider


class __CLASS__(Provider):
    name = "__ENTRY__"
    requires = ()

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "",
        tools: list[dict[str, Any]] | None = None,
        system: str = "",
        **options: Any,
    ) -> Completion:
        """One turn: messages in, text and any tool calls out."""
        raise NotImplementedError(
            "Call your API here and return "
            "Completion(text=..., tool_calls=[...], usage={...})."
        )
'''

STORAGE_PY = '''"""The storage backend.

Checkpoints, materialised datasets and run outputs go through this one
interface, which is what lets a project move from a laptop to object storage by
changing a setting.

There is no entry point for storage: a project has exactly one backend, named in
settings, so discovery would have nothing to resolve.
"""

from __future__ import annotations

from typing import IO, Any

from mlango.storage.base import Storage


class __CLASS__(Storage):
    def __init__(self, root: str = "", **options: Any):
        super().__init__(root=root, **options)
        self.root = root

    def path(self, name: str) -> str:
        """Where ``name`` lives. A URL is fine when there is no local path."""
        raise NotImplementedError

    def open(self, name: str, mode: str = "rb") -> IO[Any]:
        raise NotImplementedError

    def save_bytes(self, name: str, data: bytes) -> str:
        """Store ``data`` and return the *name*, not an absolute path.

        Names are what the metastore records, and a name resolves on any machine
        that can reach this backend. A path resolves only on the one that wrote
        it, which is how a shared database ends up sharing nothing usable.
        """
        raise NotImplementedError

    def read_bytes(self, name: str) -> bytes:
        raise NotImplementedError

    def exists(self, name: str) -> bool:
        raise NotImplementedError

    def delete(self, name: str) -> None:
        raise NotImplementedError

    def size(self, name: str) -> int:
        raise NotImplementedError

    def listdir(self, prefix: str = "") -> list[str]:
        raise NotImplementedError

    # Override writable() and readable() only if this backend is not a
    # filesystem: one stages a local file and publishes it on the way out, the
    # other fetches one for the length of a block. mlango.storage.s3 is a
    # worked example.
'''

SOURCE_PY = '''"""The data source.

A source is an iterable of dicts, and optionally a count. Filtering, splitting,
fingerprinting and materialising belong to the QuerySet.

There is no entry point for sources: you import the class and name it in a
declaration, so there is nothing for discovery to resolve.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from mlango.data.sources import Source


class __CLASS__(Source):
    def __init__(self, location: str, **options: Any):
        self.location = location
        self.options = options

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield one dict per record.

        Stream rather than build a list. A dataset that does not fit in memory
        is the normal case, and everything downstream of here is already lazy.
        """
        raise NotImplementedError

    def count(self) -> int | None:
        """Rows, when that is knowable without reading them all. None otherwise."""
        return None

    def describe(self) -> dict[str, Any]:
        """Recorded on the dataset version, so a snapshot says where it came from."""
        return {"source": type(self).__name__, "location": self.location}
'''

TEST_PY = '''"""Tests for __DIST__.

An extension is only worth installing if it is exercised the way the framework
will exercise it, so these configure mlango the way a real project does.
"""

from __future__ import annotations

import pytest

from __MODULE__ import __CLASS__


@pytest.fixture
def project(tmp_path):
    """A configured, throwaway mlango project."""
    from mlango.conf import settings
    from mlango.metastore.session import dispose_all
    from mlango.storage import reset_default_storage

    dispose_all()
    reset_default_storage()
    settings.configure(
        BASE_DIR=str(tmp_path),
        METASTORE={"URL": "sqlite:///test.db"},
        STORAGE={"BACKEND": "mlango.storage.local.LocalStorage", "ROOT": "artifacts"},
        DEFAULT_PROVIDER="echo",
        INSTALLED_APPS=[],
    )
    yield tmp_path
    dispose_all()
    reset_default_storage()
    settings.reset()


def test_it_knows_its_own_name():
    assert __CLASS__.name == "__ENTRY__"
__DISCOVERY_TEST__'''

DISCOVERY_TEST = '''

def test_it_is_discovered_once_installed(project):
    """Proves the entry point in pyproject.toml is spelled correctly.

    Run after `pip install -e .` — an entry point that is never resolved is the
    commonest way for an extension to look finished and not work.
    """
    from mlango.conf import settings

    assert settings.__SETTING__.get("__ENTRY__"), (
        "Not discovered. Reinstall the package so its metadata is written: pip install -e ."
    )
'''

README = """# __DIST__

A __KIND__ for [mlango](https://github.com/DrobyshevDev/mlango).

## Install

```bash
pip install __DIST__
```

## Use

__USAGE__

## Develop

```bash
pip install -e ".[dev]"
pytest
```

## Naming

Packages extending mlango are named `mlango-<what they add>`, so they sort
together on PyPI and a reader can tell from a dependency list what each one is
for.
"""

USAGE = {
    "trainer": """```python
class MyModel(Model):
    class Meta:
        trainer = "__ENTRY__"
```

Nothing else: mlango reads the `mlango.trainers` entry point when settings load,
so the name resolves without anyone editing them. A project can still pin it, or
point it at a patched subclass:

```python
TRAINERS = {"__ENTRY__": "__MODULE__.trainer.__CLASS__"}
```""",
    "provider": """```python
class MyAgent(Agent):
    class Meta:
        provider = "__ENTRY__"
```

Or as the project default:

```python
DEFAULT_PROVIDER = "__ENTRY__"
```""",
    "storage": """```python title="settings.py"
STORAGE = {
    "BACKEND": "__MODULE__.storage.__CLASS__",
    "ROOT": "...",
}
```""",
    "source": """```python
from __MODULE__ import __CLASS__


class MyData(Dataset):
    class Meta:
        source = __CLASS__("...")
```""",
}

GITIGNORE = """__pycache__/
*.py[cod]
.venv/
venv/
build/
dist/
*.egg-info/
.pytest_cache/
.ruff_cache/
.mypy_cache/
artifacts/
*.db
"""

LICENSE = """MIT License

Copyright (c) __YEAR__ __AUTHOR__

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

BODIES = {
    "trainer": TRAINER_PY,
    "provider": PROVIDER_PY,
    "storage": STORAGE_PY,
    "source": SOURCE_PY,
}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def module_name(dist: str) -> str:
    """``mlango-lightgbm`` to ``mlango_lightgbm``."""
    return re.sub(r"[-.]+", "_", dist)


def entry_name(dist: str) -> str:
    """The name a project will type: ``mlango-lightgbm`` to ``lightgbm``."""
    stem = module_name(dist)
    return stem[len("mlango_") :] if stem.startswith("mlango_") else stem


def class_name(entry: str, kind: str) -> str:
    """``lightgbm`` + ``trainer`` to ``LightgbmTrainer``."""
    stem = "".join(part[:1].upper() + part[1:] for part in entry.split("_") if part)
    return f"{stem}{kind.capitalize()}"


def render_plugin(dist: str, target: str, *, kind: str = "trainer", author: str = "") -> list[str]:
    """Write a distributable extension into ``target``. Returns the paths created."""
    if kind not in KINDS:
        raise ValueError(f"Unknown kind {kind!r}. One of: {', '.join(KINDS)}.")

    module = module_name(dist)
    entry = entry_name(dist)
    klass = class_name(entry, kind)
    group = GROUPS.get(kind, "")

    subs: dict[str, Any] = {
        DIST: dist,
        MODULE: module,
        ENTRY: entry,
        "__CLASS__": klass,
        "__KIND__": kind,
        "__GROUP__": group,
        "__YEAR__": str(dt.date.today().year),
        "__AUTHOR__": author or "your name",
    }

    entry_points = _substitute(ENTRY_POINTS, **subs) if group else "\n"
    discovery = ""
    if group:
        setting = "TRAINERS" if kind == "trainer" else "PROVIDERS"
        discovery = _substitute(DISCOVERY_TEST, **{**subs, "__SETTING__": setting})

    tree = {
        "pyproject.toml": PYPROJECT.replace("__ENTRY_POINTS__", entry_points),
        "README.md": README.replace("__USAGE__", USAGE[kind]),
        "LICENSE": LICENSE,
        ".gitignore": GITIGNORE,
        f"src/{module}/__init__.py": INIT_PY,
        f"src/{module}/{kind}.py": BODIES[kind],
        f"tests/test_{kind}.py": TEST_PY.replace("__DISCOVERY_TEST__", discovery),
    }

    created = []
    for relative, content in tree.items():
        created.append(_write(os.path.join(target, relative), _substitute(content, **subs)))
    return sorted(created)


def _write(path: str, content: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    return path


def _substitute(content: str, **values: str) -> str:
    for token, value in values.items():
        content = content.replace(token, value)
    return content


__all__ = ["KINDS", "GROUPS", "render_plugin", "module_name", "entry_name", "class_name"]
