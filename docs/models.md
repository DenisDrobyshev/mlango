# Models

A `Model` declares hyperparameters and how to build an estimator. The framework
owns the run: seeding, splitting, the loop, metrics, artifacts, versioning.

```python
from mlango.core import fields
from mlango.training import Model

from reviews.datasets import Reviews


class Sentiment(Model):
    """TF-IDF into logistic regression."""

    max_features = fields.IntegerField(default=20_000, min_value=1, tunable=True)
    C = fields.FloatField(default=1.0, min_value=0.0, tunable=True)

    class Meta:
        dataset = Reviews
        trainer = "sklearn"
        task = "classification"
        features = ["text"]
        splits = {"train": 0.8, "val": 0.2}

    def build(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        return make_pipeline(
            TfidfVectorizer(max_features=self.max_features),
            LogisticRegression(C=self.C),
        )
```

## Hyperparameters are fields

Because they are the same `Field` objects a dataset uses, you get validation,
defaults, introspection, recording and sweeping for free:

```python
>>> Sentiment(C=-1).full_clean()
ValidationError: C: Value -1.0 is below the minimum of 0.0.
```

Every run records the cleaned values, so "what settings produced this number?" is
answerable from the metastore rather than from memory.

Mark a field `tunable=True` to make it part of the default sweep space.

## `Meta` options

| Option | Purpose |
|---|---|
| `dataset` | The dataset class (or its label) to train on |
| `trainer` | A key from the `TRAINERS` setting: `"sklearn"`, `"torch"`, … |
| `task` | `"classification"` or `"regression"` — picks the metric bundle |
| `target` | Which field to predict; inferred when the dataset has exactly one |
| `features` | Which fields feed the model. **Set this.** |
| `exclude` | Fields to drop when `features` is not given |
| `splits` | Ratios, default `{"train": 0.8, "val": 0.2}` |
| `monitor`, `monitor_mode` | Metric that early stopping and sweeps rank by |

!!! warning "Declare `features`"
    Without `features`, every non-target field becomes an input. The primary key
    is excluded automatically, but anything else — a row number, a leaked label,
    a timestamp that encodes the answer — is not. Being explicit is one line and
    prevents a whole category of silent leakage.

## Training

```python
model = Sentiment(C=2.0)
run = model.train(tags=["baseline"])
```

```bash
python manage.py train reviews.Sentiment -p C=2.0 -p max_features=5000 --tag baseline
```

What the framework does around your `build()`:

1. Resolves the dataset and the feature and target fields
2. Seeds `random`, `numpy` and `torch` from `settings.SEED`
3. Splits the data deterministically
4. Opens a run and records the parameters and the data fingerprint
5. Captures the git commit, host, Python version and device
6. Calls `build()` and hands the result to the trainer
7. Records metrics as they arrive
8. Evaluates on the validation split
9. Saves the artifact and registers a version

Useful arguments:

| Argument | Effect |
|---|---|
| `materialize=True` | Freeze the training view into a dataset version first |
| `register=False` | Train without adding to the model registry |
| `queryset=...` | Train on an explicit queryset instead of the whole dataset |
| `splits={...}` | Override the split ratios |
| `callbacks=[...]` | Add callbacks for this run |
| `seed=...` | Override the seed |

## Inference

```python
model.predict("great movie")                 # one input
model.predict(["great movie", "awful film"]) # a batch
model.predict_proba("great movie")           # {'negative': 0.04, 'positive': 0.96}
model.evaluate(Reviews.objects.take(100))    # a metric report
```

## The registry

Every training run registers a version:

```python
Sentiment.versions()                       # newest first
Sentiment.load()                           # latest
Sentiment.load(version=3)                  # a specific one
Sentiment.load(stage="production")
Sentiment.production()                     # shorthand
Sentiment.promote(3, "production")         # demotes the incumbent to archived
```

Stages are `none`, `staging`, `production`, `archived`. Promotion is one click in
the admin, or one call here.

## Feature importance

Registration also records what the fit weighted, so a version can explain itself
without being loaded:

```python
Sentiment.load(version=3)._version.importances
# {'delightful': 2.44, 'dull': -2.16, 'brilliant': 2.05, ...}
```

```bash
python manage.py explain reviews.Sentiment
```

The model page in the admin charts the same numbers. Where the names come from
depends on the pipeline: a vectoriser is asked for its own
`get_feature_names_out()`, so a text model reports words rather than column
indices, and everything else falls back to the declared fields.

Backends opt in by implementing one method:

```python
class LightGBMTrainer(Trainer):
    def importances(self, model, fitted):
        names = model.get_features()
        return dict(zip(names, fitted.feature_importance(), strict=True))
```

Returning `None` — the default — is the right answer for a backend whose weights
do not correspond to anything a person would recognise as a feature, which is
most neural networks. A wrong explanation is worse than none, so no backend is
asked to invent one.

## Sweeps

Fields marked `tunable` define a default space:

```bash
python manage.py sweep reviews.Sentiment
python manage.py sweep reviews.Sentiment -p C=0.25,1,4 -p max_features=500,5000
python manage.py sweep reviews.Sentiment --strategy random --trials 20 --seed 0
python manage.py sweep reviews.Sentiment -p C=0.5,1,2 --promote-best production
```

```python
result = Sentiment.sweep({"C": [0.5, 1.0, 2.0]}, metric="accuracy", mode="max")
result.best.params        # {'C': 1.0}
result.ranked()           # every completed trial, best first
```

One parent run holds the search; each trial is a full child run with its own
record. A failing trial is recorded and the sweep continues.

## Presets

The recurring shapes are already written. Django ships generic views so the
ninetieth CRUD page is three lines; these are the same idea:

```python
from mlango.training import TextClassifier

class Sentiment(TextClassifier):
    """Fine-tune a pretrained encoder on the reviews."""

    class Meta:
        dataset = Reviews
        features = ["text"]
```

That is a complete declaration. `base_model`, `learning_rate`, `epochs`,
`batch_size`, `max_length`, `weight_decay`, `warmup_ratio` and `build()` come
from the preset, and every one of them is overridable.

| Preset | Trainer | For |
|---|---|---|
| `TextClassifier` | `transformers` | Fine-tuning a pretrained encoder to classify text |
| `TextRegressor` | `transformers` | Predicting a continuous value from text |
| `TabularClassifier` | `torch` | A small feed-forward net over numeric columns |
| `TabularRegressor` | `torch` | The same, predicting a number |
| `TransformerModel` | `transformers` | The shared base, if you want a different head |

!!! note "Meta options inherit"
    A subclass writing its own `class Meta` keeps everything the parent declared
    — `trainer`, `task`, `monitor` — and overrides only what it names. Python
    class bodies do not inherit on their own, so mlango merges them; without
    that, a reusable base class would be impossible to write.

```bash
pip install "mlango[transformers]"
python manage.py train reviews.Sentiment -p epochs=2 -p learning_rate=3e-5
```

The fine-tuning loop is mlango's own, not `transformers.Trainer` — so callbacks,
early stopping, metric recording and run tracking behave identically whichever
backend a project picked. What is borrowed is tokenisation, pretrained weights
and the model heads.

Override only what is genuinely model-specific:

| Method | Replaces |
|---|---|
| `encode_batch(records, target)` | Tokenising one or two text fields |
| `configure_optimizer(module)` | AdamW with no decay on biases and layer norms |
| `build()` | The head chosen from the target field's classes |

Two text fields become a sentence pair automatically, which covers entailment,
similarity and question-answer scoring:

```python
class Entailment(TextClassifier):
    class Meta:
        dataset = Pairs
        features = ["premise", "hypothesis"]
```

## Callbacks

The middleware of the training loop:

```python
from mlango.training import Checkpoint, EarlyStopping, MetricThreshold, ProgressBar

model.train(callbacks=[
    ProgressBar(),
    EarlyStopping(monitor="val_loss", patience=3),
    Checkpoint(monitor="val_accuracy", mode="max"),
    MetricThreshold("accuracy", minimum=0.85),   # fail the run in CI
])
```

Add defaults for every run with the `DEFAULT_CALLBACKS` setting.

!!! note "Metric recording is not a callback"
    Metrics are written by the framework itself, so emptying `DEFAULT_CALLBACKS`
    never costs you run history. Callbacks are purely additive.

Write your own by overriding the hooks you care about:

```python
from mlango.training import Callback


class NotifySlack(Callback):
    def on_train_end(self, run, model, **kwargs):
        post_to_slack(f"{model._meta.label} finished: {run.refresh().summary}")
```

A callback that raises is logged and skipped — instrumentation must never be the
reason a long job dies.

## Trainers

A trainer knows how to fit, predict, save and load. Everything else is the
framework's, which is why adding one is a small file:

```python
from mlango.training import Trainer


class LightGBMTrainer(Trainer):
    name = "lightgbm"
    requires = ("lightgbm",)
    extension = "txt"

    def fit(self, model, train, validation, run, callbacks, *, target="", features=None, **kw):
        booster = model.build()
        x, y = train.xy(target=target, features=features)
        booster.fit(x, y)
        run.log_metrics({"train_score": booster.score(x, y)}, step=0)
        return booster

    def predict(self, model, fitted, inputs):
        return list(fitted.predict(inputs))

    def save(self, model, fitted, name):
        from mlango.storage import default_storage

        path = default_storage().path(f"{name}.{self.extension}")
        fitted.save_model(path)
        return path

    def load(self, model, path):
        import lightgbm

        return lightgbm.Booster(model_file=path)
```

```python
TRAINERS = {"lightgbm": "myproject.trainers.LightGBMTrainer"}
```

### PyTorch

`build()` returns an `nn.Module`. The loop, device placement, batching, metrics
and checkpointing are the framework's:

```python
class Classifier(Model):
    epochs = fields.IntegerField(default=20)
    batch_size = fields.IntegerField(default=64)
    learning_rate = fields.FloatField(default=1e-3, tunable=True)

    class Meta:
        dataset = Tabular
        trainer = "torch"
        task = "classification"
        features = ["x1", "x2", "x3"]

    def build(self):
        import torch.nn as nn

        return nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 2))
```

Override the parts that are genuinely model-specific:

| Method | Replaces |
|---|---|
| `encode_batch(records, target)` | The default numeric-tabular encoder |
| `configure_optimizer(module)` | Adam at `learning_rate` |
| `loss_fn()` | Cross-entropy or MSE, chosen by target type |

`settings.DEVICE` defaults to `"auto"`, which uses CUDA when it is available.

## Serving

```python
from mlango.serve import path

urlpatterns = [
    path("predict/", Sentiment.as_endpoint(stage="production")),
]
```

The version is loaded once, on the first request. See [Serving](serving.md).
