# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- **Transformers trainer** (`mlango[transformers]`) — fine-tune a pretrained
  encoder for text classification or regression. The loop is mlango's own, so
  callbacks, early stopping, metric recording and run tracking behave the same
  as for any other backend; tokenisation, pretrained weights and heads come from
  Hugging Face. Two text fields become a sentence pair automatically.
- **Model presets** — `TextClassifier`, `TextRegressor`, `TabularClassifier`,
  `TabularRegressor`, `TransformerModel`. A complete declaration is now three
  lines, with every default overridable.
- **Meta options inherit.** A subclass writing its own `class Meta` keeps
  everything the parent declared and overrides only what it names. Python class
  bodies do not inherit on their own; without this, a reusable base class was
  impossible to write.
- **Agent streaming.** `Agent.stream()` yields typed events —  `Started`,
  `Thinking`, `TextChunk`, `ToolCalled`, `ToolFinished`, `StepFinished`,
  `Finished`, `Failed` — as they happen, and `Agent.as_stream_endpoint()` serves
  them as Server-Sent Events. `run()` and `stream()` share one loop
  implementation, so they cannot disagree about what the agent did.
- **Data sources**: `ParquetSource` (streamed in row-group batches, count from
  the file footer), `SQLSource` (server-side cursor, defaults to the metastore),
  `HuggingFaceSource`, and `DatasetVersionSource` for pinning a derived dataset
  to an exact upstream snapshot.
- **`Registry.unregister()`** and a documented registry-isolation pattern, so
  tests and notebooks can redeclare a class.
- `py.typed`, so downstream type checkers honour mlango's annotations.
- Release workflow using PyPI trusted publishing, with the tag checked against
  `__version__`, the changelog checked for a matching section, and the built
  wheel verified to contain the admin templates and `py.typed`.
- **`startproject` ships tests.** A new project comes with a working `tests/`
  directory — eight tests covering datasets, training, evaluation and agents —
  so it is green before anyone edits it and `manage.py test` works immediately.
  `startapp` writes a `tests.py` to match.
- **Documentation complete in English and Russian** — all 14 pages in both.
- **The type surface is now checked.** `mypy mlango` is clean and blocking in
  CI. Generic subsystems that took a bare `type` now name the family they mean
  (`mlango.core.typing`), and declared fields read as values inside your own
  `build()` rather than as `Field` objects, so user projects type-check too.

### Changed

- Coverage is 85% and gated: the floor lives in `pyproject.toml`, so
  `pytest --cov` enforces the same number locally as on CI.
- CI additionally runs `pip-audit`, CodeQL and Dependabot, defaults to read-only
  token permissions, and exposes one aggregate `CI` check to require in branch
  protection — so a job added later cannot silently stop blocking merges.
- A management command now accepts settings supplied by `settings.configure()`,
  not only `MLANGO_SETTINGS_MODULE`. Notebooks and test suites could not run a
  command at all before this.
- A missing run, trace or object in the admin answers `404` instead of `200`.

### Fixed

- A configuration mistake in a transformers preset now reports the mistake
  rather than surfacing as `ModuleNotFoundError: transformers`.
- **A streaming endpoint no longer breaks the whole OpenAPI schema.** The
  handler's `StreamingResponse` annotation was unresolvable, so one stream route
  took `/api/openapi.json` and the docs page down with it.
- **scikit-learn inference now agrees with training on input shape.** A
  single-feature model built a 2-D array from a request's dict while training
  passed raw values, so a text pipeline failed on every served request; with
  several features the column order followed the payload's own keys rather than
  the declaration.
- A mistyped label in an admin URL is a `404` page, not a `500`.
- The admin's version-promotion redirect no longer assumes the default mount
  point, so it works when the admin is mounted elsewhere.
- `get_run()` eager-loads its collections, so reading `run.artifacts` outside
  the session no longer raises `DetachedInstanceError`.
- `manage.py test` cannot leave `BASE_DIR` pointing at the sandbox it deleted.
  The "no tests found" path redirected settings before it failed, poisoning
  every later command in the same process.
- Metrics and probabilities read the declared feature order rather than a sorted
  copy of the request's keys.

## 0.1.0 — 2026-07-30

First release. mlango applies Django's design philosophy to machine learning,
analytics and LLM agents: declarative classes, migrations, an auto-generated
admin, and a `manage.py` that ties it together.

### Core

- Lazy settings resolved from `MLANGO_SETTINGS_MODULE`, with every default
  documented in `mlango.conf.global_settings`. An unknown setting is an error,
  not a silent no-op.
- App registry that autodiscovers `datasets.py`, `models.py`, `agents.py`,
  `evals.py`, `admin.py` and `signals.py` in every installed app.
- Declarative metaclass producing a `_meta` that the admin, migrations, CLI and
  API are all written against.
- 17 field types serving four jobs: dataset schemas, model hyperparameters,
  agent configuration and inference input validation.
- Signal dispatcher whose receivers cannot take a run down.

### Metastore and migrations

- Nine tables recording runs, metrics, artifacts, dataset versions, model
  versions, agent traces, spans, evaluation results and applied migrations.
- SQLite by default; the same schema runs on Postgres via one setting.
- Generated, reviewable migration files. The autodetector is deliberately
  conservative and never guesses a rename.

### Data

- `Dataset` with a lazy, composable queryset and Django-style `field__lookup`.
- Splits assigned by hashing each record's key, so adding rows never moves
  existing ones between train and test.
- Content-addressed materialisation that deduplicates identical snapshots, and
  distinguishes a schema change from a data change.

### Training

- `Model` with hyperparameters as validated fields, recorded on every run.
- Pluggable trainers behind one contract; scikit-learn and PyTorch included.
- Run tracking that captures seed, device, git commit, host, Python version and
  a data fingerprint.
- Metric recording built into the framework rather than a callback, so
  customising `DEFAULT_CALLBACKS` never costs run history.
- Callbacks for early stopping, checkpointing, progress and CI thresholds.
- Model registry with promotable stages.
- Hyperparameter sweeps (`manage.py sweep`) over grid or random search, with the
  winning version promotable in the same command.

### Agents

- Declarative `Agent` owning the tool-use loop, tool dispatch, retries and usage
  accounting.
- `@tool` deriving JSON Schema from type hints and Google-style docstrings.
- Four memory backends, including one that rebuilds history from traces.
- Anthropic provider and a deterministic offline provider, so the framework and
  its tests run with no credentials.
- Step-by-step tracing into ordered spans.

### Evaluation

- Declarative `Eval` with 13 scorers, including an LLM judge and a scorer that
  asserts on which tools an agent reached for.
- Per-case results persisted, so a regression is a diff between two runs.

### Admin and serving

- Server-rendered admin with no build step: data previews with filters and
  search, run history with inline SVG metric charts, side-by-side run
  comparison, version promotion and a trace viewer.
- Everything declared appears without registration; register to customise.
- Optional HTTP Basic auth, with `manage.py check` warning when the admin is
  open and `DEBUG` is off.
- Inference API deriving OpenAPI schemas from the declarations.

### Command line

- Sixteen commands: `startproject`, `startapp`, `check`, `makemigrations`,
  `migrate`, `showmigrations`, `train`, `sweep`, `evaluate`, `agent`, `runs`,
  `traces`, `dataset`, `shell`, `test`, `runserver`.
- Apps can ship their own commands, and override built-ins.
- `python -m mlango` works when the console script is not on `PATH`.

### Getting started

- `mlango startproject` scaffolds a project that already works — a dataset, a
  trained model, an agent and an eval — so the admin is populated the first time
  you open it. `--bare` skips the demo.
- Documentation in English and Russian, with a structure that accepts new
  languages one file at a time.
