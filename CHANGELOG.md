# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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
