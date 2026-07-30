#!/usr/bin/env python
"""Command-line entry point for this mlango project."""

import os
import sys


def main() -> None:
    os.environ.setdefault("MLANGO_SETTINGS_MODULE", "quickstart.settings")
    try:
        from mlango.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Could not import mlango. Is it installed and is your virtual "
            "environment active? Try: pip install mlango"
        ) from exc
    sys.exit(execute_from_command_line(sys.argv))


if __name__ == "__main__":
    main()
