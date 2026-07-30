# Command line

Every project gets a `manage.py`. The `mlango` script does the same job before a
project exists, and `python -m mlango` works when the script is not on `PATH`.

```bash
python manage.py help
python manage.py help train
```

## The commands

### Getting started

| Command | Does |
|---|---|
| `mlango startproject NAME [DIR]` | Scaffold a project that already works. `--bare` skips the demo app |
| `manage.py startapp NAME` | Scaffold an app: datasets, models, agents, evals, admin, migrations |
| `manage.py check` | Validate settings, backends, wiring, migrations and the admin |

### Data

```bash
python manage.py dataset list
python manage.py dataset show reviews.Reviews
python manage.py dataset head reviews.Reviews -n 20
python manage.py dataset validate reviews.Reviews
python manage.py dataset materialize reviews.Reviews --notes "nightly snapshot"
python manage.py dataset versions reviews.Reviews
```

### Migrations

```bash
python manage.py makemigrations [app] [-n NAME] [--dry-run] [--empty]
python manage.py migrate [app] [--plan] [--fake]
python manage.py showmigrations [app]
```

### Training

```bash
python manage.py train reviews.Sentiment -p C=2.0 -p max_features=5000 \
    --tag baseline --notes "first attempt" --materialize

python manage.py sweep reviews.Sentiment -p C=0.25,1,4 \
    --strategy grid --metric accuracy --mode max --promote-best production
```

| Flag | Effect |
|---|---|
| `-p NAME=VALUE` | Override a hyperparameter. Repeatable |
| `--dataset LABEL` | Train on a different dataset |
| `--tag TAG` | Tag the run. Repeatable |
| `--seed N` | Override the seed |
| `--materialize` | Freeze the training view into a dataset version first |
| `--no-register` | Train without adding to the registry |

### Evaluation

```bash
python manage.py evaluate support.AnswerQuality
python manage.py evaluate support.AnswerQuality --show-failures
python manage.py evaluate support.AnswerQuality --min-pass-rate 0.9
```

### Agents

```bash
python manage.py agent support.Support                         # interactive
python manage.py agent support.Support "how do I ...?"          # one shot
python manage.py agent support.Support "..." --show-steps       # print tool calls
python manage.py agent support.Support "..." --session user-42  # with memory
```

### Inspecting what happened

```bash
python manage.py runs list --kind train --status finished -n 20
python manage.py runs show 7c8f1020
python manage.py runs compare 7c8f1020 c089b7e6

python manage.py traces list --agent support.Support
python manage.py traces show a1b2c3d4 -v 2
```

### Development

```bash
python manage.py runserver              # 127.0.0.1:8000
python manage.py runserver 8080
python manage.py runserver 0.0.0.0:8080 --reload
python manage.py runserver --no-admin

python manage.py shell                  # IPython when available
python manage.py shell -c "print(Reviews.objects.count())"

python manage.py test                   # pytest, against a throwaway metastore
python manage.py test -k splits -x
python manage.py test --coverage
```

`manage.py test` points the metastore and artifact store at a temporary
directory for the duration of the run, so a test can never touch real data —
the same idea as Django creating a test database.

## Common flags

Available on every command:

| Flag | Effect |
|---|---|
| `--settings MODULE` | Use a different settings module for this run |
| `-v 0..3` | Quiet, normal, verbose, very verbose |
| `--traceback` | Show the full traceback instead of a message |

## The shell

`manage.py shell` pre-imports every declared object plus a few helpers:

```python
>>> Reviews.objects.filter(label="positive").count()
1284
>>> Sentiment.versions()
[<ModelVersion reviews.Sentiment@v2 stage=production>, ...]
>>> recent_runs(limit=3)
>>> get_trace("a1b2c3d4").spans
>>> apps.summary()
```

## Your own commands

Drop a module in `<app>/management/commands/` and it appears in
`manage.py help` — including one that **overrides a built-in**, which is how a
project customises `train` without forking the framework.

```python title="reviews/management/commands/import_reviews.py"
from mlango.management import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Import reviews from the warehouse."

    def add_arguments(self, parser):
        parser.add_argument("since", help="ISO date to import from.")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, **options):
        rows = fetch_since(options["since"])
        if not rows:
            raise CommandError(f"Nothing to import since {options['since']}.")

        self.table(
            ["id", "subject"],
            [[r["id"], r["subject"]] for r in rows[:10]],
        )
        if options["dry_run"]:
            self.warn("Dry run: nothing written.")
            return
        write(rows)
        self.ok(f"Imported {len(rows)} review(s).")
```

Helpers available on `self`:

| Helper | Prints |
|---|---|
| `self.write(msg, level=1)` | A line, respecting `-v` |
| `self.ok(msg)` / `self.warn(msg)` | Green / yellow |
| `self.stderr(msg)` | To stderr |
| `self.table(headers, rows)` | An aligned table |
| `self.style.bold(...)` etc. | Colour, disabled when output is redirected |

Raise `CommandError` for anything the user should see as a message rather than a
traceback. Set `requires_apps = False` for a command that must run before apps
load, and `requires_settings = False` for one that runs without a project.
