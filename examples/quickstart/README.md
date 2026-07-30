# quickstart

An [mlango](https://github.com/DrobyshevDev/mlango) project.

## Get running

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py train demo.Sentiment
python manage.py runserver
```

Then open <http://127.0.0.1:8000/admin/> — the dataset, the trained model, its
metrics and the agent traces are all there.

## What is in here

| Path | What it holds |
|---|---|
| `quickstart/settings.py` | Project settings: installed apps, metastore, storage, provider |
| `quickstart/routes.py` | Inference API routes |
| `demo/datasets.py` | A `Dataset` declaration |
| `demo/models.py` | A `Model` declaration with hyperparameters as fields |
| `demo/agents.py` | An `Agent` with a tool |
| `demo/evals.py` | An `Eval` suite scoring the model |
| `demo/admin.py` | Admin customisation for the dataset |

## Commands worth knowing

```bash
python manage.py check                      # validate the project
python manage.py dataset head demo.Reviews  # peek at the data
python manage.py train demo.Sentiment -p C=2.0
python manage.py runs list                  # what has run
python manage.py runs compare <id> <id>     # what changed between two runs
python manage.py evaluate demo.SentimentAccuracy
python manage.py agent demo.Helper          # interactive agent session
python manage.py traces list                # agent traces
python manage.py shell                      # shell with everything imported
```

## Using a real LLM

The project ships with the offline `echo` provider so it runs with no
credentials. To use Claude:

```bash
export ANTHROPIC_API_KEY=...
```

and set `DEFAULT_PROVIDER = "anthropic"` in `quickstart/settings.py`.
