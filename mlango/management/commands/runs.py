"""``manage.py runs`` — browse and compare runs from the terminal."""

from __future__ import annotations

import json
from typing import Any

from mlango.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "List, show and compare runs."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "action", nargs="?", default="list", choices=["list", "show", "compare"]
        )
        parser.add_argument("reference", nargs="*", help="Run id(s) for 'show' and 'compare'.")
        parser.add_argument("--kind", help="Filter by kind: train, eval, agent, sweep.")
        parser.add_argument("--status", help="Filter by status.")
        parser.add_argument("--target", help="Filter by target label.")
        parser.add_argument("-n", "--limit", type=int, default=20, help="How many to list.")

    def handle(self, **options: Any) -> None:
        action = options["action"]
        if action == "list":
            self._list(options)
        elif action == "show":
            self._show(options)
        else:
            self._compare(options)

    # -- actions -------------------------------------------------------------

    def _list(self, options: dict[str, Any]) -> None:
        from mlango.training.run import recent_runs

        runs = recent_runs(
            limit=options["limit"], kind=options.get("kind"), target=options.get("target")
        )
        if options.get("status"):
            runs = [r for r in runs if r.status == options["status"]]

        self.table(
            ["run", "kind", "target", "status", "started", "duration", "summary"],
            [
                [
                    run.short_id,
                    run.kind,
                    run.target,
                    run.status,
                    run.started_at.strftime("%m-%d %H:%M"),
                    f"{run.duration_s:.1f}s" if run.duration_s else "-",
                    " ".join(
                        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in list((run.summary or {}).items())[:3]
                    ),
                ]
                for run in runs
            ],
        )

    def _show(self, options: dict[str, Any]) -> None:
        from mlango.training.run import get_run, metric_history, metric_keys

        if not options["reference"]:
            raise CommandError("'show' needs a run id.")
        run = get_run(options["reference"][0])
        if run is None:
            raise CommandError(f"No run matches {options['reference'][0]!r}.")

        duration = f"{run.duration_s:.2f}s" if run.duration_s else "-"
        dirty = " (dirty)" if run.git_dirty else ""

        self.write(self.style.bold(f"Run {run.short_id} - {run.kind}:{run.target}"))
        self.write(f"  status     {run.status}")
        self.write(f"  started    {run.started_at}")
        self.write(f"  duration   {duration}")
        self.write(f"  seed       {run.seed}")
        self.write(f"  device     {run.device or '-'}")
        self.write(f"  git        {(run.git_commit or '-')[:12]}{dirty}")
        if run.tags:
            self.write(f"  tags       {', '.join(run.tags)}")
        if run.error:
            self.write("")
            self.write(self.style.error(run.error.strip().splitlines()[-1]))

        self.write("")
        self.write(self.style.bold("Parameters"))
        self.write(json.dumps(run.params or {}, indent=2, default=str))

        self.write("")
        self.write(self.style.bold("Metrics"))
        rows = []
        for key in metric_keys(run.id):
            history = metric_history(run.id, key)
            last = history[-1][1] if history else None
            rows.append([key, len(history), f"{last:.4f}" if last is not None else "-"])
        self.table(["metric", "points", "last"], rows)

    def _compare(self, options: dict[str, Any]) -> None:
        from mlango.training.run import get_run

        if len(options["reference"]) < 2:
            raise CommandError("'compare' needs at least two run ids.")

        runs = []
        for reference in options["reference"]:
            run = get_run(reference)
            if run is None:
                raise CommandError(f"No run matches {reference!r}.")
            runs.append(run)

        metric_keys = sorted({k for run in runs for k in (run.summary or {})})
        param_keys = sorted(
            {k for run in runs for k in (run.params or {}) if not k.startswith("_")}
        )

        headers = ["field", *[run.short_id for run in runs]]
        rows: list[list[Any]] = [
            ["target", *[run.target for run in runs]],
            ["status", *[run.status for run in runs]],
        ]
        for key in param_keys:
            rows.append([key, *[str((run.params or {}).get(key, "-"))[:20] for run in runs]])
        for key in metric_keys:
            values = []
            for run in runs:
                value = (run.summary or {}).get(key)
                values.append(f"{value:.4f}" if isinstance(value, float) else str(value or "-"))
            rows.append([key, *values])

        self.table(headers, rows)
