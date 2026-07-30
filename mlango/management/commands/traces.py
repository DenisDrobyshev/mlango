"""``manage.py traces`` — inspect agent traces from the terminal."""

from __future__ import annotations

from typing import Any

from mlango.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "List agent traces, or replay one step by step."

    def add_arguments(self, parser) -> None:
        parser.add_argument("action", nargs="?", default="list", choices=["list", "show"])
        parser.add_argument("reference", nargs="?", help="Trace id for 'show'.")
        parser.add_argument("--agent", help="Filter by agent label.")
        parser.add_argument("-n", "--limit", type=int, default=20, help="How many to list.")

    def handle(self, **options: Any) -> None:
        from mlango.agents.tracing import get_trace, recent_traces

        if options["action"] == "list":
            traces = recent_traces(limit=options["limit"], agent=options.get("agent"))
            self.table(
                ["trace", "agent", "session", "steps", "tokens", "status", "duration", "input"],
                [
                    [
                        t.short_id,
                        t.agent.rpartition(".")[2],
                        t.session_id or "-",
                        t.steps,
                        t.total_tokens,
                        t.status,
                        f"{t.duration_s:.2f}s" if t.duration_s else "-",
                        (t.input or "")[:40],
                    ]
                    for t in traces
                ],
            )
            return

        if not options.get("reference"):
            raise CommandError("'show' needs a trace id.")
        trace = get_trace(options["reference"])
        if trace is None:
            raise CommandError(f"No trace matches {options['reference']!r}.")

        duration = f"{trace.duration_s:.2f}s" if trace.duration_s else "-"
        self.write(self.style.bold(f"Trace {trace.short_id} - {trace.agent}"))
        self.write(f"  status   {trace.status}")
        self.write(f"  steps    {trace.steps}")
        self.write(f"  tokens   in {trace.input_tokens}, out {trace.output_tokens}")
        self.write(f"  duration {duration}")
        self.write("")
        self.write(self.style.bold("Input"))
        self.write(f"  {trace.input}")
        self.write("")
        self.write(self.style.bold("Output"))
        self.write(f"  {trace.output or '(none)'}")

        if trace.error:
            self.write("")
            self.write(self.style.error(trace.error))

        self.write("")
        self.write(self.style.bold("Steps"))
        for span in trace.spans:
            marker = self.style.error("x") if span.error else self.style.success("v")
            elapsed = f"{span.duration_s:.3f}s" if span.duration_s else "-"
            self.write(f"  {marker} [{span.kind}] {span.name}  {elapsed}")
            self.write(self.style.dim(f"      in  {str(span.input)[:160]}"), level=2)
            self.write(self.style.dim(f"      out {str(span.output)[:160]}"), level=2)
            if span.error:
                self.write(self.style.error(f"      err {span.error}"))
