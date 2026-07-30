# Concepts

## A framework, not a library

A library is something you call. A framework calls you. That inversion is the
whole point of mlango, and it is what buys every convenience in these docs.

When you run:

```bash
python manage.py train reviews.Sentiment -p C=2.0
```

the framework resolves your class from the registry, opens a run, seeds every
RNG, splits the data, calls **your** `build()`, drives the loop, records metrics,
captures the git commit, saves the artifact and registers a version. You supplied
a `build()` method and some field declarations.

If mlango were a library, you would still be writing the run loop, the tracking
schema, the admin and the CLI. Being a framework is what removes them.

## The four declarations

Everything you write is one of four families:

| Family | Declares | Lives in |
|---|---|---|
| `Dataset` | What a record looks like and where records come from | `<app>/datasets.py` |
| `Model` | Hyperparameters and how to build an estimator | `<app>/models.py` |
| `Agent` | Model, system prompt, tools, memory | `<app>/agents.py` |
| `Eval` | What "good" means, and how to score it | `<app>/evals.py` |

They share one machinery: a metaclass collects your fields, builds a `_meta`
object, and registers the class.

## `_meta` is the contract

Every declared class gets a `_meta` describing it:

```python
>>> Reviews._meta.label
'reviews.Reviews'
>>> Reviews._meta.field_names
['id', 'text', 'label']
>>> Reviews._meta.target_fields
[<LabelField: label>]
>>> Reviews._meta.fingerprint()
'11b053a7043a'
```

This matters more than it looks. Everything generic in the framework — the
admin, migrations, the CLI, the API schema — is written against `_meta` rather
than against `Dataset` or `Model` specifically. That is the single trick that
lets **one** admin render four different kinds of object, and why adding a fifth
family would not mean rewriting the admin.

## Fields do four jobs

One field system serves the schema of a dataset record, the hyperparameters of a
model, the configuration of an agent, and the input contract of an inference
endpoint:

```python
text = fields.TextField(max_length=5000)            # a dataset column
learning_rate = fields.FloatField(default=1e-3)     # a hyperparameter
temperature_hint = fields.ChoiceField(["low", "high"])  # agent config
```

Because they are the same objects, a hyperparameter is automatically validated,
defaulted, recorded on every run, introspectable in the admin, and sweepable —
without you writing any of that.

## Apps

An app is an importable package grouping declarations for one problem domain:
`reviews`, `recsys`, `support`. Apps are the unit of reuse and the unit of
migration.

```python
INSTALLED_APPS = ["reviews", "support"]
```

At startup the registry imports each app's `datasets.py`, `models.py`,
`agents.py`, `evals.py`, `admin.py` and `signals.py` — so declaring a class is
enough to make it exist. Nothing needs to be registered by hand.

```python
from mlango.core import apps

apps.get_dataset("reviews.Reviews")
apps.get_registered("model")
apps.summary()
```

Add an `apps.py` when you want a verbose name or startup wiring:

```python
from mlango.core import AppConfig

class ReviewsConfig(AppConfig):
    name = "reviews"
    verbose_name = "Customer reviews"

    def ready(self) -> None:
        from mlango.core.signals import run_finished
        run_finished.connect(notify_slack)
```

## Settings

One module, resolved from `MLANGO_SETTINGS_MODULE`. Every setting and its
default is documented in `mlango.conf.global_settings`; override only what
differs. A typo is an error rather than a silent no-op.

Backends are swapped by setting, not by rewriting code:

```python
METASTORE = {"URL": "postgresql://user@host/mlango"}
STORAGE = {"BACKEND": "myproject.storage.S3Storage"}
TRAINERS = {"lightgbm": "myproject.trainers.LightGBMTrainer"}
PROVIDERS = {"vllm": "myproject.providers.VLLMProvider"}
```

See [Settings](settings.md).

## The metastore

One database records everything that happened: runs, metrics, artifacts, dataset
versions, model versions, agent traces, spans and eval results. SQLite by
default, so a fresh project needs no infrastructure; the same schema runs on
Postgres when a team shares the project.

This is why "what produced this number?" is always answerable. A run row carries
its parameters, its data fingerprint, its git commit, its seed and its device.

```python
from mlango.training import recent_runs, get_run

run = get_run("7c8f1020")
run.params["_data_fingerprint"]
run.git_commit
```

## Signals

Instrumentation hooks, in Django's shape — `fn(sender, **kwargs)`:

```python
from mlango.core.signals import epoch_finished, run_finished, tool_called

@receiver(run_finished)
def on_finish(sender, run, status, **kwargs):
    ...
```

A receiver that raises is logged and skipped rather than taking the run down:
instrumentation must never be the reason a six-hour training job dies.

## Two identities for data

mlango distinguishes two things that are easy to conflate:

- **Schema fingerprint** — a hash of the declared fields. Changes when you
  change the declaration.
- **Content hash** — a hash of the actual rows. Changes when the data changes.

A materialised dataset version records both, which is what makes
"same schema, different data" and "same data, different schema" distinguishable
six months later.

## Where things live

```
myproject/
├── manage.py              # the command line for this project
├── myproject/
│   ├── settings.py        # one settings module
│   └── routes.py          # inference API routes
└── reviews/               # an app
    ├── apps.py            # optional app configuration
    ├── datasets.py
    ├── models.py
    ├── agents.py
    ├── evals.py
    ├── admin.py
    └── migrations/
        └── 0001_initial.py
```
