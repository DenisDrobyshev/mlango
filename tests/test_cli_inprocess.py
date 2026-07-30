"""The same commands, run in-process.

``test_cli.py`` drives ``manage.py`` in a subprocess, which is the honest way to
prove the CLI works as a user hits it — settings discovery, argv parsing, exit
codes. Coverage cannot see into those subprocesses, and a fast feedback loop
matters too, so the command bodies are also exercised here directly.

Both layers earn their keep: if only these existed, a broken ``manage.py`` or a
mis-wired settings module would go unnoticed.
"""

from __future__ import annotations

import sys

import pytest

from mlango.management.manager import load_command

BUILTIN = "mlango.management.commands"


def run(name: str, *argv: str) -> int:
    """Run a built-in command in this process and return its exit code."""
    command = load_command(name, f"{BUILTIN}.{name}")
    return command.run_from_argv(list(argv))


@pytest.fixture(scope="module")
def live_project(tmp_path_factory):
    """A scaffolded project loaded into *this* interpreter.

    The registry and settings are process-global, so the fixture snapshots both,
    installs the project, and restores everything afterwards — otherwise the rest
    of the suite would run against a half-replaced registry.
    """
    import os

    from mlango.conf import ENVIRONMENT_VARIABLE, settings
    from mlango.core.registry import apps
    from mlango.metastore.session import dispose_all
    from mlango.storage import reset_default_storage
    from mlango.template import render_project

    root = tmp_path_factory.mktemp("inproc")
    project = root / "liveproject"
    render_project("liveproject", str(project), demo=True)

    # Snapshot the global state this fixture is about to replace.
    saved_objects = {kind: dict(entries) for kind, entries in apps._objects.items()}
    saved_configs = dict(apps.app_configs)
    saved_ready = apps.ready
    saved_env = os.environ.get(ENVIRONMENT_VARIABLE)
    saved_path = list(sys.path)
    saved_modules = set(sys.modules)

    sys.path.insert(0, str(project))
    os.environ[ENVIRONMENT_VARIABLE] = "liveproject.settings"

    dispose_all()
    reset_default_storage()
    settings.reset()
    apps.clear()

    import mlango

    mlango.setup()

    yield project

    # Restore, so later modules see the registry they declared into.
    dispose_all()
    reset_default_storage()
    settings.reset()
    apps.clear()

    apps._objects.update(saved_objects)
    apps.app_configs.update(saved_configs)
    apps.ready = saved_ready

    for module in set(sys.modules) - saved_modules:
        if module.split(".")[0] in {"liveproject", "demo"}:
            sys.modules.pop(module, None)

    sys.path[:] = saved_path
    if saved_env is None:
        os.environ.pop(ENVIRONMENT_VARIABLE, None)
    else:
        os.environ[ENVIRONMENT_VARIABLE] = saved_env


@pytest.fixture(scope="module")
def live_migrated(live_project):
    run("migrate", "-v", "0")
    run("makemigrations", "-v", "0")
    run("migrate", "-v", "0")
    return live_project


@pytest.fixture(scope="module")
def live_trained(live_migrated):
    pytest.importorskip("sklearn")
    assert run("train", "demo.Sentiment", "-v", "0") == 0
    return live_migrated


class TestMigrate:
    def test_creates_the_metastore(self, live_project, capsys):
        assert run("migrate") == 0
        assert "metastore tables ready" in capsys.readouterr().out

    def test_makemigrations_then_migrate(self, live_project, capsys):
        assert run("makemigrations") == 0
        assert "0001_initial" in capsys.readouterr().out

        assert run("migrate") == 0
        assert "applied" in capsys.readouterr().out.lower()

    def test_showmigrations(self, live_migrated, capsys):
        assert run("showmigrations") == 0
        out = capsys.readouterr().out
        assert "demo" in out
        assert "0001_initial" in out

    def test_plan_applies_nothing(self, live_migrated, capsys):
        assert run("migrate", "--plan") == 0
        assert "No migrations to apply" in capsys.readouterr().out

    def test_an_unknown_app_is_reported(self, live_migrated, capsys):
        assert run("makemigrations", "nosuchapp") == 1
        assert "No installed app" in capsys.readouterr().err


class TestCheck:
    def test_reports_every_section(self, live_migrated, capsys):
        assert run("check") == 0
        out = capsys.readouterr().out
        for heading in ("Project", "Metastore", "Backends", "Wiring", "Migrations", "Admin"):
            assert heading in out

    def test_admin_auth_status_is_reported(self, live_migrated, capsys):
        assert run("check") == 0
        assert "auth" in capsys.readouterr().out

    def test_fail_level_warning(self, live_migrated, capsys):
        # The scaffold ships DEBUG = True, which is reported as a warning.
        assert run("check", "--fail-level", "warning") == 1
        assert "warning(s) found" in capsys.readouterr().err


class TestDataset:
    def test_list(self, live_migrated, capsys):
        assert run("dataset", "list") == 0
        assert "demo.Reviews" in capsys.readouterr().out

    def test_show(self, live_migrated, capsys):
        assert run("dataset", "show", "demo.Reviews") == 0
        out = capsys.readouterr().out
        assert "LabelField" in out
        assert "targets" in out

    def test_head(self, live_migrated, capsys):
        assert run("dataset", "head", "demo.Reviews", "-n", "2") == 0
        assert capsys.readouterr().out.count("\n") >= 4

    def test_validate(self, live_migrated, capsys):
        assert run("dataset", "validate", "demo.Reviews") == 0
        assert "validated against" in capsys.readouterr().out

    def test_materialize_and_versions(self, live_migrated, capsys):
        assert run("dataset", "materialize", "demo.Reviews", "--notes", "in-process") == 0
        assert "row(s)" in capsys.readouterr().out

        assert run("dataset", "versions", "demo.Reviews") == 0
        assert "v1" in capsys.readouterr().out

    def test_force_creates_another_version(self, live_migrated, capsys):
        run("dataset", "materialize", "demo.Reviews", "-v", "0")
        capsys.readouterr()
        assert run("dataset", "materialize", "demo.Reviews", "--force") == 0
        assert "v" in capsys.readouterr().out

    def test_a_missing_label_is_reported(self, live_migrated, capsys):
        assert run("dataset", "head") == 1
        assert "needs a dataset label" in capsys.readouterr().err


class TestTrain:
    def test_trains_and_reports(self, live_trained, capsys):
        assert run("train", "demo.Sentiment", "-p", "C=1.5", "--tag", "inproc") == 0
        out = capsys.readouterr().out
        assert "Registered" in out
        assert "accuracy" in out

    def test_notes_and_seed_are_accepted(self, live_trained, capsys):
        assert run("train", "demo.Sentiment", "--notes", "why", "--seed", "7", "-v", "0") == 0

    def test_materialize_first(self, live_trained, capsys):
        assert run("train", "demo.Sentiment", "--materialize", "-v", "0") == 0

    def test_an_unknown_model_is_reported(self, live_trained, capsys):
        assert run("train", "demo.Nope") == 1
        assert "Registered models" in capsys.readouterr().err

    def test_a_bad_parameter_is_reported(self, live_trained, capsys):
        assert run("train", "demo.Sentiment", "-p", "C=notanumber") == 1
        assert "--param C" in capsys.readouterr().err


class TestSweep:
    def test_grid(self, live_trained, capsys):
        assert (
            run(
                "sweep",
                "demo.Sentiment",
                "-p",
                "C=0.5,2.0",
                "--metric",
                "accuracy",
                "--mode",
                "max",
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "trial 1" in out
        assert "Best:" in out

    def test_default_space_from_tunable_fields(self, live_trained, capsys):
        assert run("sweep", "demo.Sentiment", "--trials", "2", "-v", "0") == 0

    def test_random_without_trials_is_reported(self, live_trained, capsys):
        assert run("sweep", "demo.Sentiment", "-p", "C=1,2", "--strategy", "random") == 1

    def test_an_empty_value_list_is_reported(self, live_trained, capsys):
        assert run("sweep", "demo.Sentiment", "-p", "C=") == 1
        assert "lists no values" in capsys.readouterr().err


class TestEvaluate:
    def test_runs(self, live_trained, capsys):
        assert run("evaluate", "demo.SentimentAccuracy") == 0
        out = capsys.readouterr().out
        assert "pass_rate" in out
        assert "cases passed" in out

    def test_show_failures(self, live_trained, capsys):
        assert run("evaluate", "demo.SentimentAccuracy", "--show-failures", "-v", "2") == 0

    def test_min_pass_rate_gate(self, live_trained, capsys):
        assert run("evaluate", "demo.SentimentAccuracy", "--min-pass-rate", "1.01") == 1
        assert "below the required" in capsys.readouterr().err

    def test_an_unknown_eval_is_reported(self, live_trained, capsys):
        assert run("evaluate", "demo.Nope") == 1


class TestAgent:
    def test_one_message(self, live_migrated, capsys):
        assert run("agent", "demo.Helper", "hello", "there") == 0
        out = capsys.readouterr().out
        assert "echo:" in out
        assert "trace" in out

    def test_show_steps(self, live_trained, capsys):
        assert (
            run(
                "agent",
                "demo.Helper",
                'use classify_review {"text": "great movie 2"}',
                "--show-steps",
            )
            == 0
        )
        captured = capsys.readouterr()
        # Step output goes to stderr so it does not pollute piped answers.
        assert "classify_review" in captured.out + captured.err

    def test_max_steps_override(self, live_migrated, capsys):
        assert run("agent", "demo.Helper", "hello", "--max-steps", "2", "-v", "0") == 0

    def test_session_is_accepted(self, live_migrated, capsys):
        assert run("agent", "demo.Helper", "hello", "--session", "s1", "-v", "0") == 0


class TestRuns:
    def test_list(self, live_trained, capsys):
        assert run("runs", "list") == 0
        assert "demo.Sentiment" in capsys.readouterr().out

    def test_filter_by_kind_and_status(self, live_trained, capsys):
        assert run("runs", "list", "--kind", "train", "--status", "finished") == 0

    def test_filter_by_target(self, live_trained, capsys):
        assert run("runs", "list", "--target", "demo.Sentiment") == 0
        assert "demo.Sentiment" in capsys.readouterr().out

    def test_show(self, live_trained, capsys):
        from mlango.training import recent_runs

        run_id = recent_runs(limit=1)[0].uuid
        assert run("runs", "show", run_id) == 0
        out = capsys.readouterr().out
        assert "Parameters" in out
        assert "Metrics" in out

    def test_compare(self, live_trained, capsys):
        from mlango.training import recent_runs

        runs = recent_runs(limit=2)
        assert run("runs", "compare", runs[0].uuid, runs[1].uuid) == 0
        assert "target" in capsys.readouterr().out

    def test_compare_needs_two(self, live_trained, capsys):
        assert run("runs", "compare", "abc") == 1

    def test_an_unknown_run(self, live_trained, capsys):
        assert run("runs", "show", "ffffffffff") == 1


class TestTraces:
    def test_list(self, live_migrated, capsys):
        run("agent", "demo.Helper", "hello", "-v", "0")
        capsys.readouterr()

        assert run("traces", "list") == 0
        assert "Helper" in capsys.readouterr().out

    def test_filter_by_agent(self, live_migrated, capsys):
        run("agent", "demo.Helper", "hello", "-v", "0")
        capsys.readouterr()

        assert run("traces", "list", "--agent", "demo.Helper") == 0

    def test_show(self, live_migrated, capsys):
        from mlango.agents.tracing import recent_traces

        run("agent", "demo.Helper", "hello", "-v", "0")
        capsys.readouterr()

        trace = recent_traces(limit=1)[0]
        assert run("traces", "show", trace.uuid, "-v", "2") == 0
        out = capsys.readouterr().out
        assert "Input" in out
        assert "Steps" in out

    def test_an_unknown_trace(self, live_migrated, capsys):
        assert run("traces", "show", "ffffffffff") == 1


class TestShell:
    def test_runs_code(self, live_migrated, capsys):
        assert run("shell", "-c", "print('rows', Reviews.objects.count())") == 0
        assert "rows 400" in capsys.readouterr().out

    def test_helpers_are_present(self, live_trained, capsys):
        assert run("shell", "-c", "print('n', len(recent_runs()))") == 0
        assert "n " in capsys.readouterr().out

    def test_the_banner_lists_declarations(self, live_migrated):
        command = load_command("shell", f"{BUILTIN}.shell")
        banner = command._banner()
        assert "datasets: Reviews" in banner
        assert "models: Sentiment" in banner
        assert "helpers:" in banner

    def test_the_namespace_holds_every_declaration(self, live_migrated):
        command = load_command("shell", f"{BUILTIN}.shell")
        namespace = command._namespace()
        assert {"Reviews", "Sentiment", "Helper", "SentimentAccuracy"} <= set(namespace)
        assert {"apps", "settings", "recent_runs", "get_trace"} <= set(namespace)


class TestStartApp:
    def test_creates_an_app(self, live_project, capsys):
        assert run("startapp", "inproc_app") == 0
        assert (live_project / "inproc_app" / "datasets.py").exists()
        assert "Next steps" in capsys.readouterr().out

    def test_a_reserved_name_is_refused(self, live_project, capsys):
        assert run("startapp", "admin") == 1
        assert "shadow an existing module" in capsys.readouterr().err

    def test_a_non_empty_target_is_refused(self, live_project, capsys):
        assert run("startapp", "demo") == 1
        assert "already exists and is not empty" in capsys.readouterr().err


class TestTestCommand:
    def test_a_scaffolded_project_is_green_before_it_is_edited(self, live_project, capsys):
        """startproject ships tests, and they must pass on a fresh checkout."""
        assert run("test") == 0

        out = capsys.readouterr().out
        assert "passed" in out
        assert "Tests passed." in out

    def test_no_tests_is_reported(self, live_project, capsys):
        """With the directory gone, the message has to say where to put one."""
        import shutil

        shipped = live_project / "tests"
        moved = live_project / "tests-moved"
        shutil.move(str(shipped), str(moved))
        try:
            assert run("test") == 1
            assert "No tests found" in capsys.readouterr().err
        finally:
            shutil.move(str(moved), str(shipped))

    def test_a_keyword_selects_a_subset(self, live_project, capsys):
        assert run("test", "-k", "dataset_loads") == 0
        assert "1 passed" in capsys.readouterr().out

    def test_a_failed_run_leaves_settings_alone(self, live_project, capsys):
        """The sandbox must not outlive the command.

        The redirection happened before the "no tests" check, so that error path
        left BASE_DIR pointing at a temp directory the command then deleted —
        and every later command in the process read from it.
        """
        import shutil

        from mlango.conf import settings

        before = str(settings.BASE_DIR)
        shipped = live_project / "tests"
        moved = live_project / "tests-away"
        shutil.move(str(shipped), str(moved))
        try:
            assert run("test") == 1
            assert str(settings.BASE_DIR) == before
            assert "test-metastore" not in settings.METASTORE["URL"]
        finally:
            shutil.move(str(moved), str(shipped))
