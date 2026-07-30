# Contributing to mlango

Thanks for considering it. mlango aims to be as approachable as Django — which
means contributions to *documentation*, *error messages* and *the first five
minutes* matter as much as features.

## Development setup

```bash
git clone https://github.com/DrobyshevDev/mlango
cd mlango
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all]"
pre-commit install
pytest -q
```

The suite runs offline: agents use the `echo` provider, and the metastore is
SQLite in a temporary directory. No API key is needed to contribute.

## Before you open a pull request

```bash
ruff check mlango tests           # lint
ruff format mlango tests          # format
mypy mlango                       # types — blocking, must be clean
pytest -q --cov                   # tests, with the coverage floor enforced
mkdocs build --strict             # docs, if you touched them
```

Every one of those is blocking in CI. Two of them deserve a word:

- **mypy has to be clean.** Users type-check their own projects against these
  annotations, so an annotation that lies is a bug we shipped. If something is
  genuinely undecidable, use a narrow `# type: ignore[code]` with a comment
  saying why — not a blanket exclusion.
- **Coverage has a floor**, set in `[tool.coverage.report] fail_under` so
  `pytest --cov` enforces the same number on your laptop as on the server.
  Raise it when coverage rises; never lower it to make a red build green.

CI additionally scaffolds a fresh project and runs it end to end. If you change
the scaffold, the settings, or any command, run that path locally too:

```bash
mlango startproject /tmp/demoproject
cd /tmp/demoproject
python manage.py check && python manage.py migrate
python manage.py train demo.Sentiment
python manage.py test
python manage.py runserver
```

`startproject` ships a working `tests/` directory, so a brand-new project is
green before anyone edits it. If you change the scaffold, keep it that way.

### Maintainers: branch protection

Require the single aggregate **`CI`** check, not the individual jobs. Requiring
jobs one at a time means any job added later is not required, and a red job
quietly stops blocking merges.

## What we look for

**Errors that teach.** A message should say what went wrong, and what to do
next. Compare:

```python
raise FieldError("no such field")                                  # unhelpful
raise FieldError(                                                  # useful
    f"{opts.label} has no field named {name!r}. Available: {available}."
)
```

**Comments that explain why, not what.** The code already says what it does.
Explain the constraint a reader cannot see:

```python
# Hash the content, never the position: inserting rows must not move existing
# ones between splits, or a held-out set stops being held out.
```

**Tests that describe behaviour.** Name them after the guarantee they protect
(`test_assignment_is_stable_when_rows_are_added`), not after the method.

**No new required dependencies** in the core. Optional integrations belong
behind an extra in `pyproject.toml` and a lazy import.

## Architecture in one page

| Layer | Depends on | Never depends on |
|---|---|---|
| `core/` | nothing but stdlib | anything else in mlango |
| `metastore/` | `core` | data, training, agents |
| `data/`, `training/`, `agents/`, `evals/` | `core`, `metastore` | each other[^1] |
| `admin/`, `serve/` | everything, through `_meta` | — |
| `management/` | everything | — |

[^1]: `evals` may target a model or an agent, which is the one deliberate
    exception. If you find yourself importing across layers for a helper, the
    helper belongs in `core` — see `core/serialization.py`, which exists for
    exactly this reason.

Everything generic is written against `_meta`, not against a concrete class.
That is what lets one admin render datasets, models, agents and evals. Keep it
that way: if a feature needs `isinstance(obj, Dataset)`, look for the `_meta`
attribute it should be reading instead.

## Adding a trainer or provider

Both are single files plus a settings entry — no core changes needed.

```python
# myproject/trainers.py
from mlango.training import Trainer

class LightGBMTrainer(Trainer):
    name = "lightgbm"
    requires = ("lightgbm",)

    def fit(self, model, train, validation, run, callbacks, *, target="", features=None, **kw): ...
    def predict(self, model, fitted, inputs): ...
    def save(self, model, fitted, name): ...
    def load(self, model, path): ...
```

```python
TRAINERS = {"lightgbm": "myproject.trainers.LightGBMTrainer"}
```

If it is broadly useful, contribute it to `mlango/training/backends/` with
tests that skip cleanly when the dependency is absent
(`pytest.importorskip`).

## Documentation

Docs live in `docs/`, built with MkDocs Material. English is the source of
truth; translations sit beside each page as `page.<locale>.md` and fall back to
English when missing — see [Translating](docs/translating.md).

## Commits and pull requests

- One logical change per pull request.
- Write commit messages that explain the reasoning, not just the diff.
- Reference an issue when there is one.
- Update `CHANGELOG.md` under `## Unreleased`.

## Reporting bugs

Please include the output of:

```bash
python manage.py check
python -c "import mlango, sys; print(mlango.get_version(), sys.version)"
```

and the smallest declaration that reproduces the problem.

## Code of conduct

Participation is covered by the [Code of Conduct](CODE_OF_CONDUCT.md).
