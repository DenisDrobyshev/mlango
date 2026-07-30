"""``manage.py test`` — run the project's tests against a throwaway metastore."""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any

from mlango.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Run the project's tests with a temporary metastore and artifact store."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "target",
            nargs="*",
            help="Paths or node ids to pass to pytest. Defaults to ./tests.",
        )
        parser.add_argument(
            "-k", "--keyword", help="Only run tests whose name matches this expression."
        )
        parser.add_argument(
            "-x", "--exitfirst", action="store_true", help="Stop after the first failure."
        )
        parser.add_argument(
            "--coverage", action="store_true", help="Report coverage for the project."
        )
        parser.add_argument(
            "--keep",
            action="store_true",
            help="Leave the temporary metastore on disk for inspection.",
        )

    def handle(self, **options: Any) -> None:
        try:
            import pytest
        except ImportError as exc:
            raise CommandError(
                "pytest is not installed. Install it with: pip install 'mlango[dev]'"
            ) from exc

        from mlango.conf import settings
        from mlango.metastore.session import dispose_all
        from mlango.storage import reset_default_storage

        # Django creates a test database so a test run cannot touch real data.
        # The same idea, applied to the metastore and the artifact store.
        sandbox = tempfile.mkdtemp(prefix="mlango-test-")
        original = {
            "METASTORE": settings.METASTORE,
            "STORAGE": settings.STORAGE,
            "BASE_DIR": settings.BASE_DIR,
        }

        dispose_all()
        reset_default_storage()
        settings.METASTORE = {**settings.METASTORE, "URL": "sqlite:///test-metastore.db"}
        settings.STORAGE = {**settings.STORAGE, "ROOT": "artifacts"}
        settings.BASE_DIR = sandbox

        targets = options["target"] or (["tests"] if os.path.isdir("tests") else [])
        if not targets:
            raise CommandError(
                "No tests found. Create a ./tests directory, or name a path: "
                "manage.py test myapp/tests.py"
            )

        argv = [*targets]
        if options.get("keyword"):
            argv += ["-k", options["keyword"]]
        if options["exitfirst"]:
            argv.append("-x")
        if options["coverage"]:
            argv += ["--cov", "--cov-report=term-missing"]
        if self.verbosity >= 2:
            argv.append("-v")
        elif self.verbosity == 0:
            argv.append("-q")

        self.write(self.style.dim(f"metastore: sqlite in {sandbox}"))
        self.write("")

        try:
            exit_code = pytest.main(argv)
        finally:
            dispose_all()
            reset_default_storage()
            for key, value in original.items():
                setattr(settings, key, value)
            if options["keep"]:
                self.write(self.style.dim(f"\nSandbox kept at {sandbox}"))
            else:
                shutil.rmtree(sandbox, ignore_errors=True)

        if exit_code != 0:
            raise CommandError(
                f"Tests failed (pytest exit code {exit_code}).", returncode=exit_code
            )
        self.ok("Tests passed.")
