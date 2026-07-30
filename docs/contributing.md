# Contributing

The full guide lives in
[CONTRIBUTING.md](https://github.com/DrobyshevDev/mlango/blob/master/CONTRIBUTING.md)
in the repository. The short version:

## Setup

```bash
git clone https://github.com/DrobyshevDev/mlango
cd mlango
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pre-commit install
pytest -q
```

The suite runs offline — agents use the `echo` provider and the metastore is
SQLite in a temporary directory. **No API key is needed to contribute.**

## Before opening a pull request

```bash
ruff check mlango tests
ruff format mlango tests
mypy mlango
pytest -q --cov
mkdocs build --strict     # if you touched docs
```

All five are blocking in CI. `pytest --cov` enforces the same coverage floor
locally that it does on the server — the threshold lives in `pyproject.toml`, so
there is no way to be green on a laptop and red on CI.

## The pipeline

`.github/workflows/ci.yml` (mirrored in `.gitlab-ci.yml`) runs:

| Job | What it protects |
|---|---|
| `lint` | ruff, ruff format, **mypy with no errors allowed** |
| `test` | the suite on Python 3.10–3.13, plus macOS and Windows |
| `coverage` | the floor in `[tool.coverage.report] fail_under` |
| `audit` | `pip-audit` against what actually gets installed |
| `quickstart` | `startproject` → migrate → train → evaluate → serve, for real |
| `transformers` | fine-tunes a tiny checkpoint, so that backend is exercised |
| `build` | wheel installs into a clean venv and ships `py.typed` |
| `docs` | `mkdocs build --strict` |
| `ci` | one aggregate check that fails if any job above did |

`CodeQL` runs separately, on pushes and weekly.

Require the aggregate **`CI`** check in branch protection rather than the
individual jobs. Requiring them one by one means a job added later is not
required, and a red job silently stops blocking merges.

Two rules for the coverage floor: raise it when the number rises, and never
lower it to turn a red build green. If a change genuinely cannot be covered,
say why in the pull request.

## What we look for

**Errors that teach.** A message should say what went wrong *and what to do
next*:

```python
raise FieldError(
    f"{opts.label} has no field named {name!r}. Available: {available}."
)
```

**Comments that explain why, not what.** The code says what it does; explain the
constraint a reader cannot see.

**Tests named after the guarantee they protect** —
`test_assignment_is_stable_when_rows_are_added`, not `test_split`.

**No new required dependencies in the core.** Optional integrations go behind an
extra in `pyproject.toml` with a lazy import.

## Layering

| Layer | May import | Must not import |
|---|---|---|
| `core/` | stdlib only | anything else in mlango |
| `metastore/` | `core` | data, training, agents |
| `data/`, `training/`, `agents/`, `evals/` | `core`, `metastore` | each other |
| `admin/`, `serve/` | everything, through `_meta` | — |

If you need a helper across layers, it belongs in `core` — see
`core/serialization.py`, which exists for exactly that reason.

Everything generic reads `_meta` rather than checking concrete types. That is
what lets one admin render datasets, models, agents and evals. If a feature
wants `isinstance(obj, Dataset)`, look for the `_meta` attribute it should read
instead.

## Good first contributions

- Translate a documentation page — see [Translating](translating.md)
- Improve an error message you found confusing
- Add a scorer to `mlango/evals/scorers.py`
- Add a trainer backend (a single file plus a settings entry)
- Add a `Source` for a format you use

## Reporting bugs

Include the output of `python manage.py check` and the smallest declaration that
reproduces the problem.
