# Notebooks

Everything else in these docs assumes a `manage.py`. You do not need one.

```python
import mlango
mlango.notebook()
```

That is the whole setup. Declare a dataset and a model in the next cell, call
`train()`, and the run is recorded with its seed, its metrics, its artifacts and
the git commit, exactly as it would be inside a project.

## A complete session

```python
import mlango
mlango.notebook()
```

```python
from mlango.core import fields
from mlango.data import Dataset, CSVSource

class Reviews(Dataset):
    id = fields.IntegerField()
    text = fields.TextField()
    label = fields.LabelField(["neg", "pos"])

    class Meta:
        source = CSVSource("reviews.csv")
        primary_key = "id"

Reviews.objects.filter(label="pos").count()
```

```python
from mlango.training import Model

class Sentiment(Model):
    C = fields.FloatField(default=1.0, tunable=True)

    class Meta:
        dataset = Reviews
        trainer = "sklearn"
        task = "classification"
        features = ["text"]

    def build(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        return make_pipeline(TfidfVectorizer(), LogisticRegression(C=self.C))

run = Sentiment().train()
run.refresh().summary
```

```python
Sentiment.load().predict("loved every minute of it")
```

No `migrate`: the metastore creates its tables the first time something writes
to one.

## Re-running a cell

Re-running a cell runs the class body again, which declares the class a second
time. mlango replaces the earlier declaration and carries on, so editing a
`Meta` option and re-running is the normal way to work.

The uniqueness check that makes two apps unable to claim one label is still
there. It only fires when the two declarations come from **different** modules,
which is a genuine collision rather than an edit.

## Why the boilerplate is gone

`mlango.notebook()` is `settings.configure()` with defaults chosen for this
situation, followed by `mlango.setup()`:

| Setting | What it uses | Why |
|---|---|---|
| `BASE_DIR` | the working directory | Runs and artifacts land beside the notebook |
| `METASTORE` | `sqlite:///mlango.db` | One file, no server |
| `STORAGE` | a local `artifacts/` directory | Checkpoints somewhere you can find |
| `DEFAULT_PROVIDER` | `echo` | Agents run with no API key |

Override any of them:

```python
mlango.notebook(SEED=7, DEFAULT_PROVIDER="anthropic")
mlango.notebook(base_dir="/data/experiments")
```

Calling it twice is harmless, which matters because the first cell is the one
people re-run most.

## Opening the admin on a notebook's work

The database it wrote is an ordinary mlango metastore, so a project pointed at
the same directory shows every run, chart and artifact from the notebook:

```bash
mlango startproject dashboard --bare
cd dashboard
# point METASTORE at the notebook's mlango.db, then
python manage.py runserver
```

This is the argument for using a framework in a notebook at all. Exploratory
work usually evaporates; here it lands in the same store as everything else, and
the run that produced a number six months ago still says which data it read.

## Moving from a notebook into a project

When the notebook has earned a home, the declarations move unchanged: the class
bodies are already what a project's `datasets.py` and `models.py` contain.

```bash
mlango startproject myproject --bare
cd myproject
python manage.py startapp reviews
```

Paste the classes into `reviews/datasets.py` and `reviews/models.py`, add
`"reviews"` to `INSTALLED_APPS`, and run `manage.py makemigrations` to record
the schema. Nothing about the declarations changes, which is the point of them
being declarations.
