"""The command line: discovery, the base class, and every built-in command.

These tests run commands the way a user does — through ``run_from_argv`` with an
argv list — so the argument parsing, the error handling and the exit codes are
covered, not just the ``handle`` bodies.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from mlango.management.base import BaseCommand, CommandError, LabelCommand, Style
from mlango.management.manager import (
    all_commands,
    builtin_commands,
    execute_from_command_line,
    load_command,
)

pytestmark = pytest.mark.usefixtures("isolated_registry")


# --------------------------------------------------------------------------- #
# A scaffolded project, built once per module and reused
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def scaffold(tmp_path_factory):
    """A real project on disk, driven through its own ``manage.py``.

    A subprocess is the honest way to test the CLI: it exercises ``manage.py``,
    the settings-module discovery and the exit codes exactly as a user hits them,
    with no shared interpreter state to leak between commands.
    """
    root = tmp_path_factory.mktemp("cli")
    project = root / "demoproject"

    from mlango.template import render_project

    render_project("demoproject", str(project), demo=True)
    return project


def manage(project, *args, expect_success: bool = True) -> subprocess.CompletedProcess:
    """Run ``manage.py`` in a scaffolded project and return the result."""
    result = subprocess.run(
        [sys.executable, "manage.py", *args],
        cwd=str(project),
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if expect_success and result.returncode != 0:
        pytest.fail(
            f"manage.py {' '.join(args)} exited {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


@pytest.fixture(scope="module")
def migrated(scaffold):
    """The project with its metastore created and migrations applied."""
    manage(scaffold, "migrate", "-v", "0")
    manage(scaffold, "makemigrations", "-v", "0")
    manage(scaffold, "migrate", "-v", "0")
    return scaffold


@pytest.fixture(scope="module")
def trained(migrated):
    """The project with a trained, registered model."""
    pytest.importorskip("sklearn")
    manage(migrated, "train", "demo.Sentiment", "-v", "0")
    return migrated


# --------------------------------------------------------------------------- #
# Discovery and the base class
# --------------------------------------------------------------------------- #


class TestDiscovery:
    def test_every_builtin_is_found(self):
        found = builtin_commands()
        expected = {
            "agent",
            "check",
            "dataset",
            "evaluate",
            "inspectdata",
            "makemigrations",
            "migrate",
            "predict",
            "runs",
            "runserver",
            "shell",
            "showmigrations",
            "startapp",
            "startproject",
            "sweep",
            "test",
            "traces",
            "train",
        }
        assert expected <= set(found)

    def test_each_module_defines_a_command(self):
        for name, path in builtin_commands().items():
            command = load_command(name, path)
            assert isinstance(command, BaseCommand)
            assert command.help, f"{name} has no help text"

    def test_loading_a_module_without_a_command_is_reported(self):
        with pytest.raises(CommandError, match="does not define a Command"):
            load_command("nope", "mlango.core.fields")

    def test_all_commands_is_sorted(self):
        names = list(all_commands(include_apps=False))
        assert names == sorted(names)

    def test_unknown_command_suggests_a_close_match(self, capsys):
        code = execute_from_command_line(["manage.py", "migrat"])
        assert code == 1
        assert "Did you mean 'migrate'" in capsys.readouterr().err

    def test_help_lists_the_commands(self, capsys):
        assert execute_from_command_line(["manage.py", "help"]) == 0
        out = capsys.readouterr().out
        assert "Available commands" in out
        assert "startproject" in out

    def test_help_for_one_command(self, capsys):
        assert execute_from_command_line(["manage.py", "help", "train"]) == 0
        assert "--param" in capsys.readouterr().out

    def test_help_for_an_unknown_command_fails(self, capsys):
        assert execute_from_command_line(["manage.py", "help", "nope"]) == 1

    def test_version(self, capsys):
        import mlango

        assert execute_from_command_line(["manage.py", "--version"]) == 0
        assert capsys.readouterr().out.strip() == mlango.get_version()

    def test_no_arguments_shows_help(self, capsys):
        assert execute_from_command_line(["manage.py"]) == 0
        assert "Available commands" in capsys.readouterr().out


class TestBaseCommand:
    def test_command_error_sets_the_exit_code(self, capsys):
        class Failing(BaseCommand):
            help = "Always fails."
            requires_settings = False
            requires_apps = False

            def handle(self, **options):
                raise CommandError("nope", returncode=3)

        assert Failing("failing").run_from_argv([]) == 3
        assert "error: nope" in capsys.readouterr().err

    def test_traceback_flag_re_raises(self):
        class Failing(BaseCommand):
            requires_settings = False
            requires_apps = False

            def handle(self, **options):
                raise CommandError("nope")

        with pytest.raises(CommandError):
            Failing("failing").run_from_argv(["--traceback"])

    def test_keyboard_interrupt_exits_130(self, capsys):
        class Interrupted(BaseCommand):
            requires_settings = False
            requires_apps = False

            def handle(self, **options):
                raise KeyboardInterrupt

        assert Interrupted("interrupted").run_from_argv([]) == 130

    def test_mlango_errors_are_reported_without_a_traceback(self, capsys):
        from mlango.core.exceptions import ImproperlyConfigured

        class Broken(BaseCommand):
            requires_settings = False
            requires_apps = False

            def handle(self, **options):
                raise ImproperlyConfigured("bad wiring")

        assert Broken("broken").run_from_argv([]) == 1
        assert "ImproperlyConfigured: bad wiring" in capsys.readouterr().err

    def test_handle_must_be_implemented(self):
        class Empty(BaseCommand):
            requires_settings = False
            requires_apps = False

        with pytest.raises(NotImplementedError):
            Empty("empty").execute(verbosity=1)

    def test_missing_settings_is_explained(self, monkeypatch):
        from mlango.conf import ENVIRONMENT_VARIABLE

        monkeypatch.delenv(ENVIRONMENT_VARIABLE, raising=False)

        class NeedsSettings(BaseCommand):
            requires_apps = False

            def handle(self, **options):
                return None

        with pytest.raises(CommandError, match="Settings are not configured"):
            NeedsSettings("needs").execute(verbosity=1, settings=None)

    def test_verbosity_gates_output(self, capsys):
        command = BaseCommand("quiet")
        command.verbosity = 0
        command.write("hidden")
        command.verbosity = 2
        command.write("shown", level=2)
        out = capsys.readouterr().out
        assert "hidden" not in out
        assert "shown" in out

    def test_table_aligns_columns(self, capsys):
        BaseCommand("t").table(["a", "bbbb"], [[1, 2], [333, 4]])
        lines = capsys.readouterr().out.splitlines()
        assert lines[0].startswith("a  ")
        assert len(lines) == 4

    def test_table_tolerates_a_ragged_row(self, capsys):
        # A display glitch must not fail the whole command.
        BaseCommand("t").table(["a", "b", "c"], [[1], [1, 2, 3, 4]])
        assert len(capsys.readouterr().out.splitlines()) == 4

    def test_table_says_when_there_is_nothing(self, capsys):
        BaseCommand("t").table(["a"], [])
        assert "nothing to show" in capsys.readouterr().out

    def test_label_command_visits_every_label(self):
        seen = []

        class Visitor(LabelCommand):
            requires_settings = False
            requires_apps = False

            def handle_label(self, label, **options):
                seen.append(label)

        Visitor("visitor").execute(label=["a", "b"], verbosity=1)
        assert seen == ["a", "b"]

    def test_style_can_be_disabled(self):
        plain = Style(enabled=False)
        assert plain.success("ok") == "ok"
        assert "\033[" in Style(enabled=True).success("ok")


# --------------------------------------------------------------------------- #
# Scaffolding
# --------------------------------------------------------------------------- #


class TestStartProject:
    def test_creates_a_runnable_project(self, scaffold):
        for relative in (
            "manage.py",
            "demoproject/settings.py",
            "demoproject/routes.py",
            "demo/datasets.py",
            "demo/models.py",
            "demo/agents.py",
            "demo/evals.py",
            "demo/admin.py",
            ".gitignore",
            "README.md",
            "requirements.txt",
            # A project that ships no tests teaches people not to write any.
            "Dockerfile",
            ".dockerignore",
            "compose.yaml",
            "tests/__init__.py",
            "tests/test_demo.py",
        ):
            assert (scaffold / relative).exists(), relative

    def test_the_asgi_entry_point_builds_an_app(self, scaffold):
        """What a production server points at. Django ships one; so does this."""
        result = manage(
            scaffold,
            "shell",
            "-c",
            "from demoproject.asgi import application; print('app:', application.title)",
        )
        assert "app:" in result.stdout

    def test_the_metastore_url_can_come_from_the_environment(self, scaffold):
        """compose.yaml sets DATABASE_URL, so settings.py has to read it."""
        import os
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "manage.py", "check"],
            cwd=str(scaffold),
            capture_output=True,
            text=True,
            timeout=300,
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "DATABASE_URL": "postgresql+psycopg://u:p@db:5432/mlango",
            },
        )
        assert "postgresql+psycopg://u:p@db:5432/mlango" in result.stdout

    def test_debug_can_be_turned_off_without_editing_a_file(self, scaffold):
        import os
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "manage.py", "check"],
            cwd=str(scaffold),
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "MLANGO_DEBUG": "0"},
        )
        assert "DEBUG is on" not in result.stdout

    def test_the_container_files_name_the_project(self, scaffold):
        """A placeholder that survives templating produces an unbuildable image."""
        for name in ("Dockerfile", "compose.yaml"):
            body = (scaffold / name).read_text(encoding="utf-8")
            assert "__PROJECT__" not in body, name
            assert "demoproject" in body, name

    def test_compose_is_valid_yaml(self, scaffold):
        yaml = pytest.importorskip("yaml")

        config = yaml.safe_load((scaffold / "compose.yaml").read_text(encoding="utf-8"))
        assert set(config["services"]) == {"db", "web"}
        assert config["services"]["web"]["depends_on"]["db"]["condition"] == "service_healthy"

    def test_dockerignore_excludes_local_state(self, scaffold):
        """Copying a developer's SQLite file into an image ships stale runs."""
        body = (scaffold / ".dockerignore").read_text(encoding="utf-8")
        for pattern in ("mlango.db", "artifacts/", "__pycache__", ".venv"):
            assert pattern in body, pattern

    def test_the_secret_key_is_generated(self, scaffold):
        import re

        settings = (scaffold / "demoproject" / "settings.py").read_text(encoding="utf-8")
        assert "__SECRET__" not in settings

        # The environment wins, so a deployment never edits this file; the
        # generated value is the fallback a fresh checkout runs on.
        match = re.search(
            r'SECRET_KEY = os\.environ\.get\("MLANGO_SECRET_KEY", "([^"]+)"\)', settings
        )
        assert match, settings
        assert len(match.group(1)) >= 32

    def test_two_projects_get_different_secrets(self, tmp_path):
        import re

        from mlango.template import render_project

        secrets = []
        for name in ("alpha", "beta"):
            render_project(name, str(tmp_path / name), demo=False)
            body = (tmp_path / name / name / "settings.py").read_text(encoding="utf-8")
            secrets.append(re.search(r'"MLANGO_SECRET_KEY", "([^"]+)"', body).group(1))

        assert secrets[0] != secrets[1]

    def test_no_placeholder_survives(self, scaffold):
        for path in scaffold.rglob("*.py"):
            body = path.read_text(encoding="utf-8")
            assert "__PROJECT__" not in body, path
            assert "__APP__" not in body, path

    def test_bare_skips_the_demo(self, tmp_path):
        from mlango.template import render_project

        render_project("bareproject", str(tmp_path / "bare"), demo=False)
        assert not (tmp_path / "bare" / "demo").exists()
        assert (tmp_path / "bare" / "bareproject" / "settings.py").exists()
        # `manage.py test` needs somewhere to look even in an empty project.
        assert (tmp_path / "bare" / "tests" / "test_project.py").exists()

    def test_an_invalid_name_is_rejected(self, tmp_path, capsys):
        command = load_command("startproject", "mlango.management.commands.startproject")
        assert command.run_from_argv(["Not-Valid", str(tmp_path / "x")]) == 1
        assert "not a valid project name" in capsys.readouterr().err

    def test_a_reserved_name_is_rejected(self, tmp_path, capsys):
        command = load_command("startproject", "mlango.management.commands.startproject")
        assert command.run_from_argv(["mlango", str(tmp_path / "y")]) == 1
        assert "shadow an existing module" in capsys.readouterr().err

    def test_a_non_empty_directory_is_refused(self, tmp_path, capsys):
        target = tmp_path / "taken"
        target.mkdir()
        (target / "file.txt").write_text("x")

        command = load_command("startproject", "mlango.management.commands.startproject")
        assert command.run_from_argv(["ok", str(target)]) == 1
        assert "already exists and is not empty" in capsys.readouterr().err


class TestStartApp:
    def test_creates_the_app_layout(self, scaffold):
        manage(scaffold, "startapp", "extra")
        for relative in (
            "__init__.py",
            "apps.py",
            "datasets.py",
            "models.py",
            "agents.py",
            "evals.py",
            "admin.py",
            "migrations/__init__.py",
            "tests.py",
        ):
            assert (scaffold / "extra" / relative).exists(), relative

    def test_the_config_class_is_named_after_the_app(self, scaffold):
        manage(scaffold, "startapp", "two_words")
        body = (scaffold / "two_words" / "apps.py").read_text(encoding="utf-8")
        assert "class TwoWordsConfig(AppConfig)" in body
        assert 'name = "two_words"' in body
        assert 'verbose_name = "Two Words"' in body

    def test_an_invalid_name_is_rejected(self, scaffold):
        result = manage(scaffold, "startapp", "Bad-Name", expect_success=False)
        assert result.returncode == 1
        assert "not a valid app name" in result.stderr


# --------------------------------------------------------------------------- #
# Migrations through the CLI
# --------------------------------------------------------------------------- #


class TestMigrationCommands:
    def test_migrate_creates_the_metastore(self, scaffold):
        result = manage(scaffold, "migrate")
        assert "metastore tables ready" in result.stdout
        assert (scaffold / "mlango.db").exists()

    def test_makemigrations_writes_a_file(self, scaffold):
        result = manage(scaffold, "makemigrations")
        assert "0001_initial.py" in result.stdout
        assert (scaffold / "demo" / "migrations" / "0001_initial.py").exists()

    def test_the_generated_migration_is_valid_python(self, scaffold):
        import ast

        manage(scaffold, "makemigrations", "-v", "0")
        body = (scaffold / "demo" / "migrations" / "0001_initial.py").read_text(encoding="utf-8")
        ast.parse(body)

    def test_makemigrations_is_idempotent(self, migrated):
        result = manage(migrated, "makemigrations")
        assert "No changes detected" in result.stdout

    def test_dry_run_writes_nothing(self, scaffold, tmp_path):
        # A fresh project so there is something to detect.
        from mlango.template import render_project

        target = tmp_path / "dry"
        render_project("dryproject", str(target), demo=True)
        manage(target, "migrate", "-v", "0")

        result = manage(target, "makemigrations", "--dry-run")
        assert "Dry run" in result.stdout
        assert not list((target / "demo" / "migrations").glob("0001_*.py"))

    def test_migrate_plan_shows_without_applying(self, scaffold, tmp_path):
        from mlango.template import render_project

        target = tmp_path / "planned"
        render_project("plannedproject", str(target), demo=True)
        manage(target, "migrate", "-v", "0")
        manage(target, "makemigrations", "-v", "0")

        result = manage(target, "migrate", "--plan")
        assert "Plan" in result.stdout
        assert "Create dataset Reviews" in result.stdout

        # Still pending, because --plan applies nothing.
        assert "[ ]" in manage(target, "showmigrations").stdout

    def test_showmigrations_marks_applied(self, migrated):
        result = manage(migrated, "showmigrations")
        assert "demo" in result.stdout
        assert "[X] 0001_initial" in result.stdout

    def test_showmigrations_can_list_operations(self, migrated):
        result = manage(migrated, "showmigrations", "-v", "2")
        assert "Create dataset Reviews" in result.stdout

    def test_migrate_twice_is_a_no_op(self, migrated):
        assert "No migrations to apply" in manage(migrated, "migrate").stdout

    def test_empty_migration_needs_an_app(self, migrated):
        result = manage(migrated, "makemigrations", "--empty", expect_success=False)
        assert result.returncode == 1
        assert "--empty needs an app label" in result.stderr


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #


class TestCheck:
    def test_reports_the_whole_project(self, migrated):
        result = manage(migrated, "check")
        for heading in ("Project", "Metastore", "Backends", "Wiring", "Migrations", "Admin"):
            assert heading in result.stdout
        assert "datasets: 1" in result.stdout
        assert "Check complete" in result.stdout

    def test_resolves_dotted_paths(self, migrated):
        result = manage(migrated, "check")
        assert "DEFAULT_CALLBACKS" in result.stdout
        assert "SERVE_MIDDLEWARE" in result.stdout
        assert "2 endpoint(s)" in result.stdout

    def test_a_broken_dotted_path_is_an_error(self, scaffold, tmp_path):
        from mlango.template import render_project

        target = tmp_path / "brokenwiring"
        render_project("brokenproject", str(target), demo=True)

        settings = target / "brokenproject" / "settings.py"
        settings.write_text(
            settings.read_text(encoding="utf-8").replace(
                '"mlango.training.callbacks.ProgressBar"',
                '"mlango.training.callbacks.NoSuchCallback"',
            ),
            encoding="utf-8",
        )

        result = manage(target, "check", expect_success=False)
        assert result.returncode == 1
        assert "cannot be imported" in result.stdout or "cannot be imported" in result.stderr

    def test_fail_level_warning_exits_non_zero(self, migrated):
        # The scaffold ships DEBUG = True, which is a warning.
        result = manage(migrated, "check", "--fail-level", "warning", expect_success=False)
        assert result.returncode == 1


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


class TestDatasetCommand:
    def test_list(self, migrated):
        result = manage(migrated, "dataset", "list")
        assert "demo.Reviews" in result.stdout

    def test_show_prints_the_declaration(self, migrated):
        result = manage(migrated, "dataset", "show", "demo.Reviews")
        assert "LabelField" in result.stdout
        assert "primary_key" in result.stdout or "targets" in result.stdout

    def test_head_prints_rows(self, migrated):
        result = manage(migrated, "dataset", "head", "demo.Reviews", "-n", "3")
        assert result.stdout.count("\n") >= 4

    def test_validate_passes_on_good_data(self, migrated):
        result = manage(migrated, "dataset", "validate", "demo.Reviews")
        assert "validated against" in result.stdout

    def test_materialize_then_versions(self, migrated):
        materialised = manage(migrated, "dataset", "materialize", "demo.Reviews").stdout
        assert "row(s)" in materialised
        assert "content" in materialised
        assert "v1" in manage(migrated, "dataset", "versions", "demo.Reviews").stdout

    def test_an_action_needing_a_label_says_so(self, migrated):
        result = manage(migrated, "dataset", "head", expect_success=False)
        assert result.returncode == 1
        assert "needs a dataset label" in result.stderr

    def test_an_unknown_dataset_lists_alternatives(self, migrated):
        result = manage(migrated, "dataset", "show", "demo.Nope", expect_success=False)
        assert result.returncode == 1
        assert "Registered datasets" in result.stderr


# --------------------------------------------------------------------------- #
# Training, sweeping, evaluating
# --------------------------------------------------------------------------- #


class TestTrainCommand:
    def test_trains_and_registers(self, trained):
        result = manage(trained, "train", "demo.Sentiment", "-p", "C=2.0", "--tag", "cli")
        assert "Registered" in result.stdout
        assert "accuracy" in result.stdout

    def test_an_unknown_parameter_lists_the_real_ones(self, trained):
        result = manage(trained, "train", "demo.Sentiment", "-p", "nope=1", expect_success=False)
        assert result.returncode == 1
        assert "has no hyperparameter" in result.stderr

    def test_a_malformed_parameter_is_reported(self, trained):
        result = manage(trained, "train", "demo.Sentiment", "-p", "justname", expect_success=False)
        assert result.returncode == 1
        assert "NAME=VALUE" in result.stderr

    def test_an_out_of_range_parameter_is_rejected(self, trained):
        result = manage(trained, "train", "demo.Sentiment", "-p", "C=-5", expect_success=False)
        assert result.returncode == 1
        assert "minimum" in result.stderr

    def test_no_register_skips_the_registry(self, trained):
        result = manage(trained, "train", "demo.Sentiment", "--no-register")
        assert "Registered" not in result.stdout


class TestSweepCommand:
    def test_grid_search_ranks_the_trials(self, trained):
        result = manage(
            trained,
            "sweep",
            "demo.Sentiment",
            "-p",
            "C=0.5,2.0",
            "--metric",
            "accuracy",
            "--mode",
            "max",
        )
        assert "trial 1" in result.stdout
        assert "trial 2" in result.stdout
        assert "Best:" in result.stdout

    def test_random_search_needs_a_trial_count(self, trained):
        result = manage(
            trained,
            "sweep",
            "demo.Sentiment",
            "-p",
            "C=0.5,1,2",
            "--strategy",
            "random",
            expect_success=False,
        )
        assert result.returncode == 1

    def test_random_search_with_a_seed(self, trained):
        result = manage(
            trained,
            "sweep",
            "demo.Sentiment",
            "-p",
            "C=0.5,1,2",
            "--strategy",
            "random",
            "--trials",
            "2",
            "--seed",
            "0",
            "--metric",
            "accuracy",
            "--mode",
            "max",
        )
        assert "Best:" in result.stdout

    def test_promote_best_promotes(self, trained):
        result = manage(
            trained,
            "sweep",
            "demo.Sentiment",
            "-p",
            "C=1.0",
            "--metric",
            "accuracy",
            "--mode",
            "max",
            "--promote-best",
            "production",
        )
        assert "Promoted" in result.stdout

    def test_an_unknown_parameter_is_reported(self, trained):
        result = manage(trained, "sweep", "demo.Sentiment", "-p", "nope=1,2", expect_success=False)
        assert result.returncode == 1
        assert "has no hyperparameter" in result.stderr


class TestEvaluateCommand:
    def test_runs_and_reports(self, trained):
        result = manage(trained, "evaluate", "demo.SentimentAccuracy")
        assert "pass_rate" in result.stdout
        assert "cases passed" in result.stdout

    def test_min_pass_rate_can_fail_the_command(self, trained):
        result = manage(
            trained,
            "evaluate",
            "demo.SentimentAccuracy",
            "--min-pass-rate",
            "1.01",
            expect_success=False,
        )
        assert result.returncode == 1
        assert "below the required" in result.stderr

    def test_show_failures_is_accepted(self, trained):
        manage(trained, "evaluate", "demo.SentimentAccuracy", "--show-failures")


# --------------------------------------------------------------------------- #
# Agents and inspection
# --------------------------------------------------------------------------- #


class TestAgentCommand:
    def test_a_single_message(self, migrated):
        result = manage(migrated, "agent", "demo.Helper", "hello there")
        assert "echo:" in result.stdout
        assert "trace" in result.stdout

    def test_show_steps_prints_tool_calls(self, trained):
        result = manage(
            trained,
            "agent",
            "demo.Helper",
            'use classify_review {"text": "great movie 2"}',
            "--show-steps",
        )
        assert "classify_review" in result.stdout + result.stderr

    def test_an_unknown_agent_is_reported(self, migrated):
        result = manage(migrated, "agent", "demo.Nope", "hi", expect_success=False)
        assert result.returncode == 1
        assert "Registered agents" in result.stderr


class TestRunsCommand:
    def test_list(self, trained):
        result = manage(trained, "runs", "list")
        assert "train" in result.stdout
        assert "demo.Sentiment" in result.stdout

    def test_filters(self, trained):
        assert "eval" not in manage(trained, "runs", "list", "--kind", "train").stdout.split(
            "kind"
        )[-1].replace("evaluate", "")

    def test_show_needs_an_id(self, trained):
        result = manage(trained, "runs", "show", expect_success=False)
        assert result.returncode == 1
        assert "needs a run id" in result.stderr

    def test_show_prints_the_record(self, trained):
        listing = manage(trained, "runs", "list", "-n", "1").stdout.splitlines()
        run_id = listing[2].split()[0]

        result = manage(trained, "runs", "show", run_id)
        assert "status" in result.stdout
        assert "Parameters" in result.stdout
        assert "Metrics" in result.stdout

    def test_compare_needs_two(self, trained):
        result = manage(trained, "runs", "compare", "abc", expect_success=False)
        assert result.returncode == 1
        assert "at least two" in result.stderr

    def test_compare_lines_runs_up(self, trained):
        listing = manage(trained, "runs", "list", "-n", "5").stdout.splitlines()
        ids = [line.split()[0] for line in listing[2:4]]

        result = manage(trained, "runs", "compare", *ids)
        assert "target" in result.stdout
        assert "status" in result.stdout

    def test_an_unknown_run_is_reported(self, trained):
        result = manage(trained, "runs", "show", "ffffffff", expect_success=False)
        assert result.returncode == 1
        assert "No run matches" in result.stderr


class TestTracesCommand:
    def test_list(self, migrated):
        manage(migrated, "agent", "demo.Helper", "hello", "-v", "0")
        result = manage(migrated, "traces", "list")
        assert "Helper" in result.stdout

    def test_show(self, migrated):
        manage(migrated, "agent", "demo.Helper", "hello", "-v", "0")
        listing = manage(migrated, "traces", "list", "-n", "1").stdout.splitlines()
        trace_id = listing[2].split()[0]

        result = manage(migrated, "traces", "show", trace_id, "-v", "2")
        assert "Input" in result.stdout
        assert "Steps" in result.stdout

    def test_show_needs_an_id(self, migrated):
        result = manage(migrated, "traces", "show", expect_success=False)
        assert result.returncode == 1
        assert "needs a trace id" in result.stderr

    def test_an_unknown_trace_is_reported(self, migrated):
        result = manage(migrated, "traces", "show", "ffffffff", expect_success=False)
        assert result.returncode == 1
        assert "No trace matches" in result.stderr


# --------------------------------------------------------------------------- #
# shell, test, runserver
# --------------------------------------------------------------------------- #


class TestShellCommand:
    def test_declared_objects_are_pre_imported(self, migrated):
        result = manage(
            migrated,
            "shell",
            "-c",
            "print('rows', Reviews.objects.count()); print('agent', Helper._meta.label)",
        )
        assert "rows 400" in result.stdout
        assert "agent demo.Helper" in result.stdout

    def test_helpers_are_available(self, trained):
        result = manage(
            trained,
            "shell",
            "-c",
            "print('runs', len(recent_runs())); print('apps', apps.summary()['apps'])",
        )
        assert "runs" in result.stdout
        assert "demo" in result.stdout


class TestInspectData:
    @pytest.fixture(scope="class")
    def with_csv(self, scaffold):
        """A CSV that looks like something a user would actually bring."""
        path = scaffold / "incoming.csv"
        lines = ["id,body,stars,country,verified,label"]
        for index in range(40):
            positive = index % 2 == 0
            lines.append(
                f"{index + 1},"
                f'"a review of some length written out here, number {index}",'
                f"{(index % 5) + 1},"
                f"{'GB' if positive else 'US'},"
                f"{'true' if positive else 'false'},"
                f"{'pos' if positive else 'neg'}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return scaffold, path

    def test_it_prints_a_declaration_that_parses(self, with_csv):
        import ast

        project, _path = with_csv
        result = manage(project, "inspectdata", "incoming.csv")

        declaration = result.stdout.split("\nRead ")[0]
        ast.parse(declaration)
        assert "class Incoming(Dataset):" in declaration

    def test_the_types_it_infers(self, with_csv):
        project, _path = with_csv
        out = manage(project, "inspectdata", "incoming.csv").stdout

        assert "id = IntegerField(min_value=1, max_value=40)" in out
        assert "body = TextField()" in out
        assert "stars = IntegerField(min_value=1, max_value=5)" in out
        assert "verified = BooleanField()" in out
        assert 'label = LabelField(["neg", "pos"])' in out
        assert 'primary_key = "id"' in out

    def test_the_summary_table_explains_the_choices(self, with_csv):
        project, _path = with_csv
        out = manage(project, "inspectdata", "incoming.csv").stdout

        assert "distinct" in out
        assert "primary_key  id" in out
        assert "target       label" in out

    def test_a_custom_class_name(self, with_csv):
        project, _path = with_csv
        out = manage(project, "inspectdata", "incoming.csv", "--name", "Feedback").stdout
        assert "class Feedback(Dataset):" in out

    def test_the_sample_size_is_respected(self, with_csv):
        project, _path = with_csv
        out = manage(project, "inspectdata", "incoming.csv", "-n", "4").stdout
        assert "Read 4 rows." in out
        # A range from four rows, not forty.
        assert "id = IntegerField(min_value=1, max_value=4)" in out

    def test_a_missing_file_is_a_message_not_a_traceback(self, scaffold):
        result = manage(scaffold, "inspectdata", "nope.csv", expect_success=False)
        assert result.returncode == 1
        assert "No such file" in result.stderr
        assert "Traceback" not in result.stderr

    def test_an_unsupported_extension_lists_what_works(self, scaffold):
        (scaffold / "book.xlsx").write_text("not really a spreadsheet", encoding="utf-8")
        result = manage(scaffold, "inspectdata", "book.xlsx", expect_success=False)
        assert "Recognised extensions" in result.stderr
        assert ".csv" in result.stderr

    def test_write_needs_an_app(self, with_csv):
        project, _path = with_csv
        result = manage(project, "inspectdata", "incoming.csv", "--write", expect_success=False)
        assert "--write needs --app" in result.stderr

    def test_write_refuses_to_clobber_real_declarations(self, with_csv):
        project, _path = with_csv
        result = manage(
            project,
            "inspectdata",
            "incoming.csv",
            "--write",
            "--app",
            "demo",
            expect_success=False,
        )
        assert "already declares something" in result.stderr
        assert "--force" in result.stderr

    def test_write_into_a_fresh_app(self, with_csv):
        project, _path = with_csv
        manage(project, "startapp", "incoming_app", "-v", "0")
        manage(
            project,
            "inspectdata",
            "incoming.csv",
            "--write",
            "--app",
            "incoming_app",
            "--force",
        )

        written = (project / "incoming_app" / "datasets.py").read_text(encoding="utf-8")
        assert "class Incoming(Dataset):" in written
        assert "for the incoming_app app" in written

    def test_it_runs_before_anything_is_declared(self, tmp_path):
        """inspectdata exists to be used on a bare project, so it must not need apps."""
        from mlango.template import render_project

        bare = tmp_path / "bare"
        render_project("bareproject", str(bare), demo=False)
        (bare / "rows.jsonl").write_text(
            '{"id": 1, "label": "a"}\n{"id": 2, "label": "b"}\n', encoding="utf-8"
        )

        out = manage(bare, "inspectdata", "rows.jsonl").stdout
        assert "class Rows(Dataset):" in out
        assert "JSONLSource" in out


class TestPredict:
    def test_a_literal_input(self, trained):
        out = manage(trained, "predict", "demo.Sentiment", "an absolute delight").stdout
        assert "prediction" in out
        assert "1 prediction(s)." in out

    def test_several_literals(self, trained):
        out = manage(trained, "predict", "demo.Sentiment", "wonderful", "dreadful").stdout
        assert "2 prediction(s)." in out

    def test_probabilities_on_request(self, trained):
        out = manage(trained, "predict", "demo.Sentiment", "wonderful", "--proba").stdout
        assert "probabilities" in out
        assert "pos" in out and "neg" in out

    def test_the_loaded_version_is_reported(self, trained):
        import re

        out = manage(trained, "predict", "demo.Sentiment", "wonderful").stdout
        # Not a fixed number: earlier tests in this module register versions too.
        assert re.search(r"demo\.Sentiment@v\d+", out)
        # Every version starts at "none"; saying so on every call is noise.
        assert "stage=none" not in out

    def test_an_explicit_version_can_be_asked_for(self, trained):
        out = manage(trained, "predict", "demo.Sentiment", "wonderful", "--version", "1").stdout
        assert "demo.Sentiment@v1" in out

    def test_a_version_that_does_not_exist(self, trained):
        result = manage(
            trained,
            "predict",
            "demo.Sentiment",
            "wonderful",
            "--version",
            "999",
            expect_success=False,
        )
        assert result.returncode == 1
        assert "Traceback" not in result.stderr

    def test_scoring_the_declared_dataset(self, trained):
        out = manage(trained, "predict", "demo.Sentiment", "--dataset", "-n", "5").stdout
        assert "5 prediction(s)." in out
        assert "id" in out

    def test_filtering_the_dataset(self, trained):
        out = manage(
            trained, "predict", "demo.Sentiment", "--dataset", "--filter", "label=pos", "-n", "3"
        ).stdout
        assert "3 prediction(s)." in out

    def test_a_filter_that_matches_nothing_says_what_was_applied(self, trained):
        result = manage(
            trained,
            "predict",
            "demo.Sentiment",
            "--dataset",
            "--filter",
            "label=nonexistent",
            expect_success=False,
        )
        assert "Filters applied: label=nonexistent" in result.stderr
        assert "dataset head" in result.stderr

    def test_a_malformed_filter(self, trained):
        result = manage(
            trained,
            "predict",
            "demo.Sentiment",
            "--dataset",
            "--filter",
            "label",
            expect_success=False,
        )
        assert "FIELD=VALUE" in result.stderr

    def test_an_unknown_filter_field(self, trained):
        result = manage(
            trained,
            "predict",
            "demo.Sentiment",
            "--dataset",
            "--filter",
            "nope=1",
            expect_success=False,
        )
        assert "has no field" in result.stderr

    def test_scoring_a_file(self, trained):
        path = trained / "to_score.jsonl"
        path.write_text(
            '{"id": 1, "text": "wonderful and warm"}\n{"id": 2, "text": "dreadful, dull"}\n',
            encoding="utf-8",
        )
        out = manage(trained, "predict", "demo.Sentiment", "--file", "to_score.jsonl").stdout
        assert "2 prediction(s)." in out

    def test_a_file_missing_the_model_features_says_which(self, trained):
        path = trained / "wrong_columns.jsonl"
        path.write_text('{"id": 1, "body": "wonderful"}\n', encoding="utf-8")

        result = manage(
            trained,
            "predict",
            "demo.Sentiment",
            "--file",
            "wrong_columns.jsonl",
            expect_success=False,
        )
        assert "needs text" in result.stderr
        assert "Columns found: body, id" in result.stderr
        assert "Traceback" not in result.stderr

    def test_jsonl_output_is_machine_readable(self, trained):
        import json

        out = manage(
            trained, "predict", "demo.Sentiment", "--dataset", "-n", "3", "--format", "jsonl"
        ).stdout
        rows = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
        assert len(rows) == 3
        assert {"id", "input", "prediction"} <= set(rows[0])

    def test_csv_output(self, trained):
        out = manage(
            trained, "predict", "demo.Sentiment", "--dataset", "-n", "2", "--format", "csv"
        ).stdout
        assert "id,input,prediction" in out

    def test_writing_to_a_file(self, trained):
        manage(
            trained,
            "predict",
            "demo.Sentiment",
            "--dataset",
            "-n",
            "4",
            "--format",
            "jsonl",
            "--output",
            "scored.jsonl",
        )
        written = (trained / "scored.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(written) == 4

    def test_a_file_target_never_gets_a_terminal_table(self, trained):
        """--output with the default format must still write data, not a table."""
        import json

        manage(
            trained, "predict", "demo.Sentiment", "--dataset", "-n", "2", "--output", "plain.jsonl"
        )
        lines = (trained / "plain.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["prediction"]

    def test_an_untrained_model_says_how_to_train_it(self, migrated):
        result = manage(migrated, "predict", "demo.Untrained", expect_success=False)
        assert result.returncode == 1
        assert "No model named" in result.stderr

    def test_no_input_at_all_says_what_to_pass(self, trained):
        result = manage(trained, "predict", "demo.Sentiment", expect_success=False)
        assert "--dataset" in result.stderr

    def test_conflicting_inputs_are_refused(self, trained):
        result = manage(
            trained, "predict", "demo.Sentiment", "hello", "--dataset", expect_success=False
        )
        assert "not both" in result.stderr


class TestTestCommand:
    def test_the_scaffolded_tests_pass(self, migrated):
        """A new project must be green before anyone edits it."""
        result = manage(migrated, "test")
        assert "Tests passed" in result.stdout
        assert "8 passed" in result.stdout

    def test_reports_when_there_are_no_tests(self, migrated):
        import shutil

        shipped = migrated / "tests"
        moved = migrated / "tests-moved"
        shutil.move(str(shipped), str(moved))
        try:
            result = manage(migrated, "test", expect_success=False)
            assert result.returncode == 1
            assert "No tests found" in result.stderr
        finally:
            shutil.move(str(moved), str(shipped))

    def test_runs_a_project_test_against_a_sandbox(self, migrated):
        tests_dir = migrated / "tests"
        added = tests_dir / "test_sandbox.py"
        added.write_text(
            "from demo.datasets import Reviews\n"
            "\n"
            "def test_rows():\n"
            "    assert Reviews.objects.count() == 400\n"
            "\n"
            "def test_metastore_is_a_sandbox():\n"
            "    from mlango.metastore.session import metastore_url\n"
            "    assert 'test-metastore' in metastore_url()\n",
            encoding="utf-8",
        )
        try:
            result = manage(migrated, "test")
            assert "Tests passed" in result.stdout
        finally:
            added.unlink()

    def test_a_failing_project_test_fails_the_command(self, migrated):
        tests_dir = migrated / "tests"
        (tests_dir / "test_broken.py").write_text(
            "def test_fails():\n    assert False\n", encoding="utf-8"
        )
        try:
            result = manage(migrated, "test", expect_success=False)
            assert result.returncode != 0
        finally:
            (tests_dir / "test_broken.py").unlink()


class TestRunserverWiring:
    def test_address_parsing(self):
        from mlango.management.commands.runserver import _resolve_address

        class FakeSettings:
            SERVE_HOST = "127.0.0.1"
            SERVE_PORT = 8000

        settings = FakeSettings()
        assert _resolve_address({"addrport": "9000"}, settings) == ("127.0.0.1", 9000)
        assert _resolve_address({"addrport": "0.0.0.0:9001"}, settings) == ("0.0.0.0", 9001)
        assert _resolve_address({"host": "1.2.3.4", "port": 80}, settings) == ("1.2.3.4", 80)
        assert _resolve_address({}, settings) == ("127.0.0.1", 8000)


# --------------------------------------------------------------------------- #
# The whole path, as advertised
# --------------------------------------------------------------------------- #


class TestQuickstartPromise:
    @pytest.mark.slow
    def test_four_commands_from_nothing(self, tmp_path):
        """The README promises this exact sequence. It has to keep working."""
        pytest.importorskip("sklearn")

        from mlango.template import render_project

        project = tmp_path / "promise"
        render_project("promiseproject", str(project), demo=True)

        manage(project, "migrate")
        manage(project, "train", "demo.Sentiment")

        # The admin would be empty if any of this silently did nothing.
        assert "demo.Sentiment" in manage(project, "runs", "list").stdout

        health = manage(
            project,
            "shell",
            "-c",
            "import json; from mlango.core.registry import apps; print(json.dumps(apps.summary()))",
        ).stdout
        summary = json.loads(health.strip().splitlines()[-1])
        assert summary["counts"] == {"dataset": 1, "model": 1, "agent": 1, "eval": 1}
