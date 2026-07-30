# Introduction

**mlango is a batteries-included framework for machine learning, analytics and
LLM agents, built on Django's philosophy.**

ML projects have a way of turning into a pile of scripts: one that loads data,
one that trains, a notebook that produced the number in last quarter's slide
deck, and a `checkpoints/` directory nobody can map back to a commit.

Web development had exactly this problem. Django solved it — not by shipping a
better library, but by being a **framework**: a project layout, a settings
module, declarative classes, migrations, an auto-generated admin, and a
`manage.py` that ties it together. mlango applies that answer to ML.

You declare datasets, models, agents and evaluations. The framework runs them,
versions them, records them, and shows them to you.

Two things make that different from the tools you already have. One class body
becomes an admin page, a documented API endpoint, a migration and a CLI target
at once — you wire up none of it. And **agents are a first-class family beside
models**, sharing one metastore, one admin and one evaluation system, rather
than living in a separate stack.

!!! question "Already using MLflow, Kedro, W&B or LangChain?"
    [**mlango and the alternatives**](comparison.md) says where each of them
    sits, what genuinely overlaps, and — set out plainly — when mlango is the
    wrong choice.

## Install

```bash
pip install "mlango[sklearn]"
```

| Extra | Brings in |
|---|---|
| `sklearn` | scikit-learn trainer |
| `torch` | PyTorch trainer |
| `anthropic` | Claude provider for agents |
| `dev` | pytest, ruff, mypy |
| `all` | everything above |

## Five minutes from nothing

```bash
mlango startproject myproject
cd myproject
python manage.py migrate
python manage.py train demo.Sentiment
python manage.py runserver
```

Open <http://127.0.0.1:8000/admin/>.

Unlike a bare scaffold, a fresh mlango project **already contains a working
example** — a dataset, a trained model with real metrics, an agent with a tool,
and an eval suite. The admin has something in it the first time you look, which
is the difference between "I see how this works" and "now what?".

Getting there needs no configuration: the metastore is SQLite, artifacts go to a
local directory, and agents run on an offline provider that needs no API key.

!!! tip "Prefer an empty project?"
    `mlango startproject myproject --bare` skips the demo app.

## What a declaration looks like

=== "Dataset"

    ```python
    # reviews/datasets.py
    from mlango.core import fields
    from mlango.data import Dataset, JSONLSource

    class Reviews(Dataset):
        """Customer product reviews."""

        id = fields.IntegerField()
        text = fields.TextField()
        label = fields.LabelField(["negative", "positive"])

        class Meta:
            source = JSONLSource("data/reviews.jsonl")
            primary_key = "id"
    ```

=== "Model"

    ```python
    # reviews/models.py
    from mlango.core import fields
    from mlango.training import Model
    from reviews.datasets import Reviews

    class Sentiment(Model):
        """TF-IDF into logistic regression."""

        max_features = fields.IntegerField(default=20_000, tunable=True)
        C = fields.FloatField(default=1.0, min_value=0.0, tunable=True)

        class Meta:
            dataset = Reviews
            trainer = "sklearn"
            task = "classification"
            features = ["text"]

        def build(self):
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            return make_pipeline(
                TfidfVectorizer(max_features=self.max_features),
                LogisticRegression(C=self.C),
            )
    ```

=== "Agent"

    ```python
    # support/agents.py
    from mlango.agents import Agent, BufferMemory, tool

    @tool
    def search_docs(query: str, limit: int = 5) -> list[str]:
        """Search the product documentation.

        Args:
            query: What to search for.
            limit: Maximum number of results.
        """
        return retrieve(query, limit)

    class Support(Agent):
        """Answers product questions from the docs."""

        class Meta:
            model = "claude-opus-5"
            system = "You are a support engineer. Cite the docs you used."
            tools = [search_docs]
            memory = BufferMemory(k=20)
    ```

=== "Eval"

    ```python
    # support/evals.py
    from mlango.evals import Eval, contains_all, token_f1

    class AnswerQuality(Eval):
        """Does the agent answer from the docs?"""

        class Meta:
            dataset = SupportCases
            target = Support
            input_field = "question"
            expected_field = "answer"
            scorers = {"overlap": token_f1, "cited": contains_all("docs/")}
            threshold = 0.6
    ```

Then:

```bash
python manage.py train reviews.Sentiment -p C=2.0
```

That one command resolves your class, opens a tracked run, seeds every RNG,
splits the data deterministically, calls your `build()`, drives the training
loop, records metrics, captures the git commit, saves the artifact, and
registers a promotable model version. You wrote `build()` and four field
declarations.

## Where to go next

- **[Tutorial](tutorial.md)** — build a project end to end
- **[Concepts](concepts.md)** — how declarations, `_meta` and apps fit together
- **[Datasets](datasets.md)**, **[Models](models.md)**, **[Agents](agents.md)**, **[Evaluations](evals.md)** — the four building blocks
- **[Admin](admin.md)** — the interface you did not have to build
