# Releasing

A release is driven entirely by a git tag. There is no API token stored
anywhere — PyPI publishing uses **trusted publishing**, where PyPI verifies the
workflow's identity through OIDC.

## One-time setup

### 1. Claim the name on PyPI

`mlango` was free at the time of writing. Verify before relying on it:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://pypi.org/pypi/mlango/json
# 404 means the name is available
```

### 2. Configure trusted publishing

On **PyPI** → *Your projects* → *Publishing* → *Add a pending publisher*:

| Field | Value |
|---|---|
| PyPI project name | `mlango` |
| Owner | `DrobyshevDev` |
| Repository name | `mlango` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Repeat on **TestPyPI** with environment name `testpypi`.

A pending publisher works before the project exists, so the first release needs
no manual upload.

!!! warning "The owner is part of the identity"
    Trusted publishing matches the repository that requested the token, owner
    included. Moving the repository between accounts or organisations
    invalidates the existing publisher and the release job fails at the publish
    step, having passed everything before it. Add a publisher for the new owner
    before the next tag; the old one can then be removed.

### 3. Create the GitHub environments

Repository → *Settings* → *Environments* → create `pypi` and `testpypi`. Adding
a required reviewer to `pypi` gives you a manual approval gate before anything
goes public.

> ⚠️ **The repository must be public for a normal PyPI release to make sense.**
> A wheel on PyPI is downloadable by anyone, so publishing while the source stays
> private gives users a binary they cannot inspect, audit, or contribute to — and
> the MIT licence promises they can. Flip the repository public first.

## Cutting a release

1. **Update the version** in `mlango/__init__.py`:

   ```python
   __version__ = "0.2.0"
   ```

2. **Move `## Unreleased` in `CHANGELOG.md`** to `## 0.2.0 — YYYY-MM-DD` and add a
   fresh empty `## Unreleased` above it. CI checks that the section exists.

3. **Commit, tag and push:**

   ```bash
   git commit -am "Release 0.2.0"
   git tag -a v0.2.0 -m "mlango 0.2.0"
   git push origin master
   git push origin v0.2.0
   ```

The tag triggers `release.yml`, which:

- runs lint and the full test suite;
- checks the tag matches `mlango.__version__`, so a mistyped tag fails instead
  of publishing the wrong version;
- checks `CHANGELOG.md` has a section for it;
- builds the wheel and sdist and runs `twine check --strict`;
- verifies the wheel actually contains the **admin templates** and **`py.typed`**
  — a wheel missing either is broken in a way tests would not catch;
- installs the wheel into a clean virtual environment, scaffolds a project and
  drives it through `check`, `migrate`, `train` and `evaluate`;
- publishes to PyPI;
- creates a GitHub release with the distributions attached.

## Rehearsing on TestPyPI

Before a first release, or any release you are unsure about:

*Actions* → *Release to PyPI* → *Run workflow* → target `testpypi`.

Then install from there:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            "mlango[sklearn]"
```

The extra index is needed because TestPyPI does not mirror dependencies.

## Versioning

[Semantic versioning](https://semver.org/). While the version is `0.x`:

| Change | Bump |
|---|---|
| Breaking change to a declaration, setting or command | minor (`0.1` → `0.2`) |
| New capability, backwards compatible | minor |
| Fix, docs, internals | patch (`0.1.0` → `0.1.1`) |

From `1.0` onward, a breaking change means a major bump. Document every one in
the changelog with the migration a user has to perform.

## What is public API

Anything exported from a package `__init__` is public and covered by the
versioning promise:

```python
from mlango.core import fields, apps, AppConfig
from mlango.data import Dataset, JSONLSource
from mlango.training import Model, Trainer, Callback
from mlango.agents import Agent, tool, Provider
from mlango.evals import Eval
from mlango import admin, migrations
from mlango.serve import path
```

A leading underscore, or a module not re-exported, is internal and may change in
a patch release.

## After publishing

- Check the project page renders: <https://pypi.org/project/mlango/>
- Install from a clean environment and run the four-command quickstart
- Verify the docs deployed: <https://drobyshevdev.github.io/mlango/>

## Yanking

A published version cannot be deleted, only **yanked** — pip will skip it unless
a user pins it exactly. Yank when a release is broken enough to mislead people:

```bash
# On the PyPI project page: Manage → Releases → Yank
```

Then release a fixed patch version. Do not attempt to re-upload the same version
number; PyPI rejects it, by design.
