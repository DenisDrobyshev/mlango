# Datasets

A `Dataset` declares what a record looks like and where records come from.
Everything else — validation, querying, splitting, versioning, the admin preview
— follows from that declaration.

```python
from mlango.core import fields
from mlango.data import Dataset, JSONLSource


class Reviews(Dataset):
    """Customer product reviews."""

    id = fields.IntegerField()
    text = fields.TextField(max_length=5000)
    label = fields.LabelField(["negative", "positive"])
    stars = fields.IntegerField(min_value=1, max_value=5)

    class Meta:
        source = JSONLSource("data/reviews.jsonl")
        primary_key = "id"
```

## `Meta` options

| Option | Purpose |
|---|---|
| `source` | Where records come from: a `Source`, a callable, or a list |
| `primary_key` | Field used for stable split assignment and excluded from features |
| `split_salt` | Change it to reshuffle split assignment deliberately |
| `verbose_name`, `verbose_name_plural` | Admin labels |
| `description` | Defaults to the class docstring's first paragraph |
| `license`, `homepage` | Provenance, shown in the admin |

## Sources

| Source | Reads |
|---|---|
| `JSONLSource(path)` | One JSON object per line — the default interchange format |
| `JSONSource(path, key=None)` | A JSON array, or an object whose `key` holds one |
| `CSVSource(path, delimiter=",")` | Delimited text; fields coerce the strings |
| `DirectorySource(root)` | `root/<class>/<file>` layouts for images and audio |
| `InMemorySource(rows)` | A list of dicts — fixtures and tests |
| `PythonSource(factory)` | A callable yielding records — generated or scraped data |
| `ChainSource(*sources)` | Several sources concatenated — shards, multiple dumps |

Or override `records()` and skip `source` entirely:

```python
class Reviews(Dataset):
    text = fields.TextField()

    @classmethod
    def records(cls):
        for row in fetch_from_warehouse():
            yield {"text": row.body}
```

Writing your own source is a two-method class:

```python
from mlango.data import Source


class WarehouseSource(Source):
    def __init__(self, query: str):
        self.query = query

    def __iter__(self):
        yield from run_query(self.query)

    def count(self):
        return None          # unknown is fine

    def describe(self):
        return {"type": "WarehouseSource", "query": self.query}
```

`describe()` is recorded alongside every run, so keep it informative and free of
credentials.

## The queryset

`Dataset.objects` hands out a lazy, immutable queryset. Nothing is read until you
iterate, so the same expression works over ten rows in a test and ten million in
production.

### Filtering

Lookups use Django's `field__op` spelling:

```python
Reviews.objects.filter(label="positive")
Reviews.objects.filter(stars__gte=4)
Reviews.objects.exclude(text__icontains="spam")
Reviews.objects.filter(label__in=["positive", "neutral"])
Reviews.objects.filter(text__isnull=False)
```

| Lookup | Meaning |
|---|---|
| `exact` (default), `iexact`, `ne` | Equality |
| `gt`, `gte`, `lt`, `lte` | Ordering |
| `in` | Membership |
| `contains`, `icontains`, `startswith`, `endswith` | Substrings |
| `isnull` | Presence |
| `len`, `len_gt`, `len_lt` | Length |
| `regex` | Regular expression |

A typo is caught immediately rather than silently matching nothing:

```python
>>> Reviews.objects.filter(lable="positive")
FieldError: Reviews has no field(s) lable. Available: id, label, stars, text.
```

For anything lookups cannot express, use a predicate:

```python
Reviews.objects.where(lambda r: detect_language(r.text) == "en")
```

### Transforming

```python
Reviews.objects.map(lambda r: {"text": r.text.lower(), "label": r.label})
Reviews.objects.annotate(length=lambda r: len(r.text))
Reviews.objects.only("text", "label")
Reviews.objects.defer("raw_html")
Reviews.objects.rename(body="text")
```

### Ordering, sampling, slicing

```python
Reviews.objects.order_by("-stars", "id")
Reviews.objects.shuffle(seed=0)      # seeded, so it is reproducible
Reviews.objects.distinct("label")
Reviews.objects.take(100).skip(10)
Reviews.objects[10:20]
Reviews.objects.repeat(3)            # oversampling
```

`order_by` and `shuffle` must see every record, so they materialise. Everything
else streams.

### Validating

```python
Reviews.objects.validate()   # raise on the first row that breaks the declaration
Reviews.objects.clean()      # validate and coerce to the declared types
```

`clean()` is what turns `"5"` from a CSV into `5`.

### Terminal operations

```python
Reviews.objects.count()
Reviews.objects.first()
Reviews.objects.get(id=42)                  # raises unless exactly one matches
Reviews.objects.exists()
Reviews.objects.all()                       # a list of records
Reviews.objects.values("id", "label")
Reviews.objects.values_list("id", flat=True)
Reviews.objects.columns("text", "label")    # column-oriented
Reviews.objects.xy(features=["text"])       # (inputs, targets)
Reviews.objects.to_pandas()                 # if pandas is installed

for batch in Reviews.objects.batch(32):
    ...
```

### Records

Iteration yields a `Record`, which answers to both styles:

```python
record = Reviews.objects.first()
record.text        # attribute access
record["text"]     # mapping access
```

### Caching

```python
train = Reviews.objects.filter(label="positive").cache()
```

Evaluate once, then reuse across epochs.

## Splits that stay put

```python
parts = Reviews.objects.split(train=0.8, val=0.1, test=0.1)
train, val, test = parts["train"], parts["val"], parts["test"]
```

Assignment hashes each record's `primary_key` (or its content, when no key is
declared) — **not** its position. So:

- adding rows never moves existing rows between splits;
- duplicate records land in the same split, which avoids train/test leakage;
- the division is identical across machines and runs.

That is the property that makes a held-out set still trustworthy six months
later. To reshuffle deliberately, change `Meta.split_salt`.

## Versioning

Two identities, deliberately separate:

| Identity | Changes when | Where |
|---|---|---|
| Schema fingerprint | the declared fields change | `_meta.fingerprint()` |
| Content hash | the rows change | computed while materialising |

Freeze the current view into a numbered snapshot:

```python
version = Reviews.materialize(
    Reviews.objects.filter(label="positive"),
    notes="positives only, for the imbalance experiment",
)
version.ref          # 'reviews.Reviews@v3'
version.row_count
version.content_hash
```

```bash
python manage.py dataset materialize reviews.Reviews
python manage.py dataset versions reviews.Reviews
```

Re-materialising identical data **returns the existing version** rather than
piling up duplicates, so a nightly job is cheap when nothing changed. Pass
`force=True` to override.

Read a snapshot back:

```python
Reviews.load_version()      # latest
Reviews.load_version(2)     # a specific one
```

Train against a frozen version so the run is reproducible even if the upstream
source moves:

```bash
python manage.py train reviews.Sentiment --materialize
```

## Inspecting from the terminal

```bash
python manage.py dataset list
python manage.py dataset show reviews.Reviews
python manage.py dataset head reviews.Reviews -n 20
python manage.py dataset validate reviews.Reviews
python manage.py dataset versions reviews.Reviews
```

## Reusing a declaration

Abstract bases contribute their fields:

```python
class Timestamped(Dataset):
    created_at = fields.DateTimeField(null=True)

    class Meta:
        abstract = True


class Reviews(Timestamped):
    text = fields.TextField()
    # fields are: created_at, text
```

An abstract dataset is not registered and never appears in the admin.
