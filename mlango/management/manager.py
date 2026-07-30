"""Command discovery and dispatch.

``manage.py`` and the ``mlango`` console script both route through here. Apps
can ship their own commands in ``<app>/management/commands/`` and they show up
in ``manage.py help`` automatically, exactly as in Django.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys

from mlango.management.base import BaseCommand, CommandError

BUILTIN_PACKAGE = "mlango.management.commands"


def builtin_commands() -> dict[str, str]:
    """``{name: module path}`` for the framework's own commands."""
    package = importlib.import_module(BUILTIN_PACKAGE)
    return {
        name: f"{BUILTIN_PACKAGE}.{name}"
        for _finder, name, is_pkg in pkgutil.iter_modules(package.__path__)
        if not is_pkg and not name.startswith("_")
    }


def app_commands() -> dict[str, str]:
    """Commands contributed by installed apps.

    Requires the registry, so it is only consulted once settings exist. An app
    command with the same name as a built-in wins, which is how a project
    customises ``train`` without forking the framework.
    """
    from mlango.core.module_loading import module_has_submodule
    from mlango.core.registry import apps

    found: dict[str, str] = {}
    for config in apps.get_app_configs():
        if not module_has_submodule(config.module, "management"):
            continue
        try:
            package = importlib.import_module(f"{config.name}.management.commands")
        except ImportError:
            continue
        for _finder, name, is_pkg in pkgutil.iter_modules(package.__path__):
            if not is_pkg and not name.startswith("_"):
                found[name] = f"{config.name}.management.commands.{name}"
    return found


def all_commands(*, include_apps: bool = True) -> dict[str, str]:
    commands = builtin_commands()
    if include_apps and _settings_available():
        try:
            import mlango

            mlango.setup()
            commands.update(app_commands())
        except Exception:
            # A broken project must still be able to run `help` and `check`.
            pass
    return dict(sorted(commands.items()))


def load_command(name: str, module_path: str) -> BaseCommand:
    module = importlib.import_module(module_path)
    command_class = getattr(module, "Command", None)
    if command_class is None or not issubclass(command_class, BaseCommand):
        raise CommandError(f"{module_path} does not define a Command(BaseCommand) class.")
    return command_class(name)


def execute_from_command_line(argv: list[str] | None = None) -> int:
    """Entry point used by ``manage.py``."""
    argv = list(argv if argv is not None else sys.argv)
    prog = os.path.basename(argv[0]) if argv else "manage.py"
    args = argv[1:]

    if not args or args[0] in {"help", "-h", "--help"}:
        target = args[1] if len(args) > 1 else None
        return _help(prog, target)

    if args[0] in {"--version", "version"}:
        import mlango

        print(mlango.get_version())
        return 0

    name, rest = args[0], args[1:]
    commands = all_commands()
    if name not in commands:
        suggestion = _closest(name, commands)
        hint = f" Did you mean {suggestion!r}?" if suggestion else ""
        print(
            f"Unknown command {name!r}.{hint}\nRun '{prog} help' to list the available commands.",
            file=sys.stderr,
        )
        return 1

    command = load_command(name, commands[name])
    return command.run_from_argv(rest, prog=prog)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``mlango`` console script."""
    argv = list(argv if argv is not None else sys.argv)
    if len(argv) > 1 and argv[1] == "startproject":
        # startproject runs before a settings module exists.
        command = load_command("startproject", f"{BUILTIN_PACKAGE}.startproject")
        return command.run_from_argv(argv[2:], prog="mlango")
    return execute_from_command_line(argv)


# --------------------------------------------------------------------------- #
# Help
# --------------------------------------------------------------------------- #


def _help(prog: str, target: str | None) -> int:
    commands = all_commands()

    if target:
        if target not in commands:
            print(f"Unknown command {target!r}.", file=sys.stderr)
            return 1
        load_command(target, commands[target]).create_parser(prog).print_help()
        return 0

    import mlango

    print(f"mlango {mlango.get_version()}\n")
    print(f"Usage: {prog} <command> [options]\n")
    print("Available commands:")
    width = max((len(n) for n in commands), default=0)
    for name, module_path in commands.items():
        summary = _summary(name, module_path)
        print(f"  {name.ljust(width)}  {summary}")
    print(f"\nRun '{prog} help <command>' for a command's options.")
    return 0


def _summary(name: str, module_path: str) -> str:
    try:
        module = importlib.import_module(module_path)
        return (module.Command.help or "").strip().split("\n")[0]
    except Exception:
        return ""


def _closest(name: str, commands: dict[str, str]) -> str | None:
    import difflib

    matches = difflib.get_close_matches(name, list(commands), n=1, cutoff=0.6)
    return matches[0] if matches else None


def _settings_available() -> bool:
    from mlango.conf import ENVIRONMENT_VARIABLE, settings

    return bool(os.environ.get(ENVIRONMENT_VARIABLE) or settings.configured)


__all__ = ["execute_from_command_line", "main", "all_commands", "load_command"]
