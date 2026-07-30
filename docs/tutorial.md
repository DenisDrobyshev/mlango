# Tutorial

We are going to build a project that classifies support tickets by urgency, then
put an agent in front of it. By the end you will have a tracked training run, a
promotable model version, an evaluation suite, an inference API and an admin
showing all of it.

Everything runs locally with no API key.

## 1. Create the project

```bash
pip install "mlango[sklearn]"
mlango startproject helpdesk --bare
cd helpdesk
```

`--bare` skips the demo app, because we are writing our own. (Drop it and you get
a working example to read.)

```bash
python manage.py startapp tickets
```

Add the app to `helpdesk/settings.py`:

```python title="helpdesk/settings.py" hl_lines="2"
INSTALLED_APPS = [
    "tickets",
]
```

Then check the project:

```bash
python manage.py check
```

## 2. Declare a dataset

Save some data as `data/tickets.jsonl` — one JSON object per line:

```json title="data/tickets.jsonl"
{"id": 1, "subject": "Cannot log in at all", "urgency": "high"}
{"id": 2, "subject": "Typo on the pricing page", "urgency": "low"}
{"id": 3, "subject": "Payments failing for everyone", "urgency": "high"}
{"id": 4, "subject": "Feature idea for exports", "urgency": "low"}
```

Now declare what a record *is*:

```python title="tickets/datasets.py"
from mlango.core import fields
from mlango.data import Dataset, JSONLSource


class Tickets(Dataset):
    """Support tickets, labelled by urgency."""

    id = fields.IntegerField()
    subject = fields.TextField(max_length=500)
    urgency = fields.LabelField(["low", "high"])

    class Meta:
        source = JSONLSource("data/tickets.jsonl")
        primary_key = "id"
```

Three things follow from those six lines.

**The data is validated against the declaration:**

```bash
python manage.py dataset validate tickets.Tickets
python manage.py dataset head tickets.Tickets
```

If a row has `"urgency": "urgent"`, validation says so and names the row.

**You can query it lazily:**

```python
Tickets.objects.filter(urgency="high").count()
Tickets.objects.filter(subject__icontains="login").take(5).all()
Tickets.objects.shuffle(seed=0).split(train=0.8, test=0.2)
```

**Splits are stable.** Assignment hashes each record's `id`, so adding tickets
next month does not move existing ones between train and test.

## 3. Record the schema in a migration

```bash
python manage.py makemigrations
python manage.py migrate
```

Open `tickets/migrations/0001_initial.py` — it is ordinary, readable Python. Add
a field later and `makemigrations` writes the diff, so six months from now you
can tell what a stored dataset version actually meant.

## 4. Declare a model

```python title="tickets/models.py"
from mlango.core import fields
from mlango.training import Model

from tickets.datasets import Tickets


class Urgency(Model):
    """TF-IDF over the subject line, into logistic regression."""

    max_features = fields.IntegerField(default=5000, min_value=1, tunable=True)
    C = fields.FloatField(default=1.0, min_value=0.0, tunable=True)

    class Meta:
        dataset = Tickets
        trainer = "sklearn"
        task = "classification"
        features = ["subject"]

    def build(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        return make_pipeline(
            TfidfVectorizer(max_features=self.max_features),
            LogisticRegression(C=self.C, max_iter=1000),
        )
```

!!! warning "Why `features` is explicit"
    Without it, `id` would be treated as an input. Declaring `features` — or
    relying on `primary_key` to exclude the id — is what stops a model quietly
    learning from a row number.

Hyperparameters are **fields**, which means they are validated, defaulted,
recorded on every run, and sweepable. You did not write a config parser.

## 5. Train

```bash
python manage.py train tickets.Urgency -p C=2.0 --tag baseline
```

The framework opened a run, seeded every RNG, split the data, called your
`build()`, recorded metrics, captured the git commit, saved the artifact and
registered version 1.

```bash
python manage.py runs list
python manage.py runs show <run-id>
```

Note `_data_fingerprint` in the parameters. Two runs with the same fingerprint
saw the same data view — which is what makes a comparison meaningful.

## 6. Search the space

Both hyperparameters are marked `tunable`, so:

```bash
python manage.py sweep tickets.Urgency -p C=0.25,1,4 -p max_features=500,5000
```

One parent run, six child runs, a ranked table. Add `--promote-best production`
and the winner is promoted in the same command.

## 7. Say what "good" means

```python title="tickets/evals.py"
from mlango.evals import Eval, exact_match

from tickets.datasets import Tickets
from tickets.models import Urgency


class UrgencyAccuracy(Eval):
    """Does the classifier agree with the labelled urgency?"""

    class Meta:
        dataset = Tickets
        target = Urgency
        input_field = "subject"
        expected_field = "urgency"
        case_id_field = "id"
        scorers = {"correct": exact_match}
        threshold = 1.0
```

```bash
python manage.py evaluate tickets.UrgencyAccuracy --show-failures
```

Every case is stored, so a regression is a diff between two runs rather than a
number someone remembers. In CI:

```bash
python manage.py evaluate tickets.UrgencyAccuracy --min-pass-rate 0.9
```

## 8. Put an agent in front

```python title="tickets/agents.py"
from mlango.agents import Agent, BufferMemory, tool


@tool
def triage(subject: str) -> str:
    """Classify a ticket subject as low or high urgency.

    Args:
        subject: The ticket's subject line.
    """
    from tickets.models import Urgency

    try:
        model = Urgency.load(stage="production")
    except LookupError:
        return "No production model yet. Run the sweep with --promote-best production."
    return str(model.predict(subject))


class Triage(Agent):
    """Triages incoming tickets, using the trained classifier."""

    class Meta:
        system = (
            "You triage support tickets. When given a subject line, call the "
            "triage tool rather than guessing, then explain the result briefly."
        )
        tools = [triage]
        memory = BufferMemory(k=20)
```

The tool's JSON schema comes from the type hints and the docstring — you describe
it once.

```bash
python manage.py agent tickets.Triage
```

The project ships with the offline `echo` provider so this runs with no
credentials. For a real model, set `DEFAULT_PROVIDER = "anthropic"` and export
`ANTHROPIC_API_KEY`.

Every call is traced:

```bash
python manage.py traces list
python manage.py traces show <trace-id> -v 2
```

## 9. Serve it

```python title="helpdesk/routes.py"
from mlango.serve import path

from tickets.agents import Triage
from tickets.models import Urgency

urlpatterns = [
    path("predict/", Urgency.as_endpoint(stage="production")),
    path("triage/", Triage.as_endpoint()),
]
```

```bash
python manage.py runserver
```

- Admin: <http://127.0.0.1:8000/admin/>
- API docs: <http://127.0.0.1:8000/api/docs>

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H 'Content-Type: application/json' \
  -d '{"input": "Payments are down for all customers"}'
```

## 10. Look at what you built

Open the admin. Without writing a single template you have:

- **Tickets** — the data, with a filter on `urgency` and search over `subject`
- **Urgency** — every version, its metrics, and one-click promotion
- **Runs** — history with metric charts, and side-by-side comparison
- **Traces** — each agent call, step by step

Customise the presentation when you want to:

```python title="tickets/admin.py"
from mlango import admin

from tickets.datasets import Tickets


@admin.register(Tickets)
class TicketsAdmin(admin.ObjectAdmin):
    list_display = ("id", "subject", "urgency")
    list_filter = ("urgency",)
    search_fields = ("subject",)
```

## What to read next

- **[Concepts](concepts.md)** — why `_meta` is the contract everything reads
- **[Datasets](datasets.md)** — the full queryset and versioning story
- **[Models](models.md)** — trainers, callbacks and the registry
- **[Agents](agents.md)** — tools, memory, providers and tracing
- **[Settings](settings.md)** — every knob and its default
