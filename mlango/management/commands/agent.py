"""``manage.py agent`` — run a declared agent from the terminal."""

from __future__ import annotations

import sys
from typing import Any

from mlango.management.base import BaseCommand


class Command(BaseCommand):
    help = "Send a message to a declared agent, or start an interactive session."

    def add_arguments(self, parser) -> None:
        parser.add_argument("agent", help="Agent label, e.g. support.SupportAgent.")
        parser.add_argument("message", nargs="*", help="The message. Omit for interactive mode.")
        parser.add_argument("--session", default="", help="Session id, for memory continuity.")
        parser.add_argument("--max-steps", type=int, help="Override the step limit.")
        parser.add_argument(
            "--show-steps", action="store_true", help="Print each tool call as it happens."
        )

    def handle(self, **options: Any) -> None:
        from mlango.core.registry import apps

        agent_class = apps.get_agent(options["agent"])
        agent = agent_class()

        if options["show_steps"]:
            self._wire_step_output(agent_class)

        message = " ".join(options["message"]).strip()
        if message:
            self._once(agent, message, options)
            return
        self._interactive(agent, options)

    # -- modes ---------------------------------------------------------------

    def _once(self, agent: Any, message: str, options: dict[str, Any]) -> None:
        result = agent.run(
            message, session_id=options["session"], max_steps=options.get("max_steps")
        )
        self.write(result.output)
        self.write("")
        self.write(
            self.style.dim(
                f"steps {result.steps} · tokens {result.usage.total_tokens} "
                f"· trace {result.trace_uuid[:8]}"
            )
        )
        if result.error:
            self.warn(result.error)

    def _interactive(self, agent: Any, options: dict[str, Any]) -> None:
        label = type(agent)._meta.label
        session = options["session"] or "cli"
        self.write(self.style.bold(f"{label} — interactive"))
        self.write(self.style.dim("Type a message, or 'exit' to leave.\n"))

        while True:
            try:
                message = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.write("")
                return
            if message.lower() in {"exit", "quit", ":q"}:
                return
            if not message:
                continue

            result = agent.run(message, session_id=session, max_steps=options.get("max_steps"))
            self.write(f"{label.rpartition('.')[2]}> {result.output}")
            if result.error:
                self.warn(f"  {result.error}")
            self.write(
                self.style.dim(
                    f"  [{result.steps} step(s), {result.usage.total_tokens} tokens, "
                    f"trace {result.trace_uuid[:8]}]"
                )
            )
            self.write("")

    # -- live step output ----------------------------------------------------

    def _wire_step_output(self, agent_class: type) -> None:
        from mlango.core.signals import tool_called

        style = self.style

        def on_tool(sender, agent, tool, arguments, **kwargs):
            print(style.dim(f"  → {tool.name}({arguments})"), file=sys.stderr)

        # weak=False keeps the closure alive for the length of the command.
        tool_called.connect(on_tool, sender=agent_class, weak=False)
