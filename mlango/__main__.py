"""Support ``python -m mlango`` as an alias for the ``mlango`` console script.

Useful when the entry point is not on PATH — a fresh virtual environment, a
container, or a CI step that installed the wheel but not the scripts directory.
"""

import sys

from mlango.management.manager import main

if __name__ == "__main__":
    # argv[0] is the module path under -m; rewrite it so help text reads
    # "mlango <command>" rather than "__main__.py <command>".
    argv = ["mlango", *sys.argv[1:]]
    sys.exit(main(argv))
