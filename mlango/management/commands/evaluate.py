"""``manage.py evaluate`` — run a declared eval suite."""

from __future__ import annotations

from typing import Any

from mlango.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Run a declared evaluation suite and record the results."

    def add_arguments(self, parser) -> None:
        parser.add_argument("eval", help="Eval label, e.g. support.AnswerQuality.")
        parser.add_argument("--name", default="", help="Name for the run.")
        parser.add_argument("--notes", default="", help="Free-text notes for the run.")
        parser.add_argument("--tag", action="append", default=[], help="Tag the run. Repeatable.")
        parser.add_argument(
            "--show-failures",
            action="store_true",
            help="Print each failing case after the summary.",
        )
        parser.add_argument(
            "--min-pass-rate",
            type=float,
            help="Exit non-zero if the pass rate falls below this. Useful in CI.",
        )

    def handle(self, **options: Any) -> None:
        from mlango.core.registry import apps

        eval_class = apps.get_eval(options["eval"])
        self.write(self.style.bold(f"Evaluating {eval_class._meta.label}"))

        report = eval_class().run(
            name=options["name"],
            notes=options["notes"],
            tags=options["tag"] or None,
            progress=self.verbosity >= 2,
        )

        summary = report.summary()
        self.write("")
        self.table(
            ["metric", "value"],
            [
                [key, f"{value:.4f}" if isinstance(value, float) else value]
                for key, value in summary.items()
            ],
        )

        if options["show_failures"] and report.failures():
            self.write("")
            self.write(self.style.bold("Failures"))
            for case in report.failures():
                self.write(f"  case {case['case_id']}")
                self.write(self.style.dim(f"    output   {str(case.get('output'))[:120]}"))
                if case.get("expected") is not None:
                    self.write(self.style.dim(f"    expected {str(case.get('expected'))[:120]}"))
                if case.get("error"):
                    self.write(self.style.error(f"    error    {case['error']}"))
                self.write(self.style.dim(f"    scores   {case.get('scores')}"))

        self.write("")
        self.write(f"run {report.run.uuid}")

        minimum = options.get("min_pass_rate")
        if minimum is not None and report.pass_rate < minimum:
            raise CommandError(
                f"Pass rate {report.pass_rate:.1%} is below the required {minimum:.1%}."
            )
        self.ok(f"{report.passed}/{report.total} cases passed.")
