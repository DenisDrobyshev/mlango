# Extending mlango

Every part of mlango that touches the outside world is a small contract behind a
name in settings. That is not decoration: it is what lets a project swap the
thing without forking the framework, and it is what lets you publish the thing
without asking anyone's permission.

## The extension points

| Point | Contract | Registered by |
|---|---|---|
| Trainer | `fit`, `predict`, `save`, `load` | `TRAINERS` or the `mlango.trainers` entry point |
| LLM provider | one method: `complete()` | `PROVIDERS` or the `mlango.providers` entry point |
| Storage backend | `path`, `open`, `save_bytes`, `read_bytes`, `exists`, `delete`, `size`, `listdir` | `STORAGE["BACKEND"]` |
| Data source | iterable of dicts, optional `count()` | named in a `Dataset.Meta.source` |
| Training callback | any subset of the `Callback` hooks | `DEFAULT_CALLBACKS`, or per run |
| Serving middleware | ASGI middleware | `SERVE_MIDDLEWARE` |
| Management command | `Command(BaseCommand)` | `<app>/management/commands/` |

The contracts are narrow on purpose. A provider has exactly one method because
the agent loop, tool dispatch, memory and tracing belong to the framework —
changing provider must not be able to change how an agent behaves.

## Start from a working package

```bash
mlango startplugin mlango-lightgbm --kind trainer
```

```
mlango-lightgbm/
├── pyproject.toml          # entry point already declared
├── README.md
├── LICENSE
├── src/mlango_lightgbm/
│   ├── __init__.py
│   └── trainer.py          # the contract, with the interesting parts commented
└── tests/test_trainer.py   # including one that proves the entry point resolves
```

`--kind` is `trainer`, `provider`, `storage` or `source`. It needs no project:
writing an extension should not require inventing somewhere to write it.

## Discovery

A package advertises itself with a standard entry point:

```toml title="pyproject.toml"
[project.entry-points."mlango.trainers"]
lightgbm = "mlango_lightgbm.trainer:LightGBMTrainer"
```

After `pip install mlango-lightgbm`, this works with no settings change at all:

```python
class Urgency(Model):
    class Meta:
        trainer = "lightgbm"
```

Three sources are merged, in this order:

1. mlango's own defaults
2. installed packages, through entry points
3. the project's `TRAINERS` / `PROVIDERS`

The project wins, always. Pointing `TRAINERS["lightgbm"]` at a patched subclass
has to be possible without uninstalling the package that named it — an
extension you cannot override is a worse deal than the dotted path it replaced.

Nothing is imported during discovery. Only names and paths are read, and
resolution stays lazy, so a broken plugin fails when something asks for it
rather than at start-up. `manage.py check` prints what came from where:

```
Backends
  trainer    sklearn: available
  trainer    lightgbm: available (plugin)
  plugin     TRAINERS['lightgbm'] = mlango_lightgbm.trainer.LightGBMTrainer
```

Storage backends and data sources have no entry point, and that is not an
oversight: a project has exactly one storage backend, named in settings, and a
source is imported and named in a declaration. Neither leaves anything for a
registry to resolve.

## Naming

Name the distribution `mlango-<what it adds>`:

- `mlango-lightgbm`, `mlango-catboost` — trainers
- `mlango-openai`, `mlango-ollama` — providers
- `mlango-gcs`, `mlango-azure` — storage
- `mlango-snowflake`, `mlango-bigquery` — sources

A convention, not a rule. It costs nothing and it means a reader can tell from a
dependency list what each line is for, and a search for `mlango-` finds the
ecosystem rather than a subset of it.

The import package follows: `mlango_lightgbm`. The registered name drops the
prefix: `"lightgbm"`.

## Writing a trainer

The one worth showing, because it looks like more work than it is:

```python
from mlango.training import Trainer


class LightGBMTrainer(Trainer):
    name = "lightgbm"
    requires = ("lightgbm",)      # checked before use, so a missing dep is a clear message
    extension = "txt"

    def fit(self, model, train, validation, run, callbacks, *, target="", features=None, **kw):
        booster = model.build()
        x, y = train.xy(target=target, features=features)
        booster.fit(x, y)
        run.log_metrics({"train_score": booster.score(x, y)}, epoch=0, step=0)
        return booster

    def predict(self, model, fitted, inputs):
        return list(fitted.predict(inputs))

    def save(self, model, fitted, name):
        from mlango.storage import default_storage

        with default_storage().writable(f"{name}.{self.extension}") as target:
            fitted.save_model(target.path)
            return target.name

    def load(self, model, path):
        import lightgbm

        from mlango.storage import default_storage

        with default_storage().readable(path) as local:
            return lightgbm.Booster(model_file=local)
```

That is the whole thing. Run tracking, metric history, versioning, promotion,
sweeps, the admin, the inference endpoint and `manage.py train` are already
written and do not know which backend produced a number.

Two details worth copying:

- **`writable()` and `readable()`, not `path()`.** They hand back a local path
  and publish on the way out, so the same trainer works when the project moves
  its artifacts to S3. `save()` returns the storage *name* — an absolute path in
  the metastore only resolves on the machine that wrote it.
- **Log metrics through `run`.** Metrics belong to the run, not the backend.
  That is what puts a third-party trainer in the same history, admin and
  comparison view as the built-in ones.

Optional hooks: `predict_proba()`, `describe()` (shown on the run page) and
`importances()` (feature weights — return `None` when the weights do not
correspond to anything a person would call a feature; see
[Feature importance](models.md#feature-importance)).

## Testing an extension

The scaffold ships a fixture that configures mlango the way a project does, so
tests exercise the code the way the framework will:

```python
settings.configure(
    BASE_DIR=str(tmp_path),
    METASTORE={"URL": "sqlite:///test.db"},
    STORAGE={"BACKEND": "mlango.storage.local.LocalStorage", "ROOT": "artifacts"},
    INSTALLED_APPS=[],
)
```

It also ships a test that asserts the entry point resolves after installation.
Keep it. An entry point that is never resolved is the commonest way for an
extension to look finished and do nothing.

## Publishing

Nothing mlango-specific:

```bash
pip install build twine
python -m build
twine upload dist/*
```

Then open an issue on
[the tracker](https://github.com/DrobyshevDev/mlango/issues) so it can be listed.
There is no approval process and no registry to be admitted to — the entry point
is the whole mechanism, and it works whether anyone has heard of your package or
not.
