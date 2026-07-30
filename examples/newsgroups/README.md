# 20 Newsgroups

A real benchmark, start to finish, in about fifty lines of declaration.

The demo project that `mlango startproject` gives you trains on 400 synthetic
strings, which proves the wiring works and nothing else. This one uses a dataset
with published baselines so the number can be checked against something.

## The result

| | |
|---|---|
| Documents | **3,669** across four newsgroups |
| Held-out accuracy | **0.8832** |
| Macro F1 | **0.8776** |
| Training time | **12 s** on a laptop CPU |
| After a six-point sweep | **0.8886** (`C=16.0, ngram_max=2`) |

Measured on the run below, not quoted from anywhere. Headers, signatures and
quoted replies are stripped from the text, which matters: leave them in and a
classifier reaches the high nineties by learning who posts where, and the number
stops being about the writing.

## Run it yourself

```bash
pip install "mlango[sklearn]"
cd examples/newsgroups

python manage.py migrate
python manage.py train news.Topic --tag baseline
```

The first run downloads about 14 MB into scikit-learn's cache. Everything after
that is offline.

```
accuracy         0.8832
f1_macro         0.8776
f1_weighted      0.8822
precision_macro  0.8801
recall_macro     0.8767
support          745

Registered news.Topic@v1
```

Then score text nobody wrote for the benchmark:

```bash
python manage.py predict news.Topic \
  "my doctor prescribed antibiotics but the infection came back after a week" \
  "the renderer uses a z-buffer and per-pixel lighting on the GPU"
```

```
input                                                         prediction
------------------------------------------------------------  -------------
my doctor prescribed antibiotics but the infection came bac…  sci.med
the renderer uses a z-buffer and per-pixel lighting on the …  comp.graphics
```

And search the space, since the hyperparameters are already declared `tunable`:

```bash
python manage.py sweep news.Topic -p C=1.0,4.0,16.0 -p ngram_max=1,2 \
    --metric accuracy --mode max
```

```
rank  trial  C     ngram_max  accuracy  run
1     6      16.0  2          0.8886    70d151e9
2     1      1.0   1          0.8832    86f51a57
3     4      4.0   2          0.8832    75c56a32
...
Best: accuracy=0.8886 with C=16.0, ngram_max=2
```

Six runs, one parent, forty seconds, every trial kept with its own metrics.

## What the declaration costs

Three files. `news/datasets.py` says what a record is and where records come
from; `news/models.py` says how to turn text into a prediction;
`news/evals.py` says what counts as good enough.

```python
class Posts(Dataset):
    """Usenet posts from four newsgroups, labelled by group."""

    id = fields.IntegerField()
    text = fields.TextField()
    group = fields.LabelField(CATEGORIES)

    class Meta:
        source = load_posts
        primary_key = "id"
```

Nothing here mentions the admin, the API, migrations, the run record or the
sweep. They all read the same declaration.

```bash
python manage.py evaluate news.TopicAccuracy   # 485/500 cases passed
python manage.py runserver                     # admin + a documented API
```

## Two things worth noticing

**The split is mlango's, not scikit-learn's.** The loader yields the train and
test sets together and lets the framework assign the split by hashing each
record's `id`. Keeping scikit-learn's own division would leave two sources of
truth about which rows are held out, and only one of them would be recorded on
the run. Because assignment is by hash, adding documents next month does not
move existing ones across the boundary.

**The evaluation floor is a floor.** `TopicAccuracy` fails below 0.80, which
catches a real regression while tolerating the run-to-run variation any split
introduces. In CI:

```bash
python manage.py evaluate news.TopicAccuracy --min-pass-rate 0.9
```
