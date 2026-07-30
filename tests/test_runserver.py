"""``manage.py runserver`` and ``mlango.serve.api.run``.

uvicorn is stubbed out: starting a real server would block, and what matters is
that the right host, port and app reach it.
"""

from __future__ import annotations

import pytest

from mlango.management.commands.runserver import _resolve_address

pytestmark = pytest.mark.usefixtures("isolated_registry")


@pytest.fixture
def uvicorn_calls(monkeypatch):
    """Record what would have been served instead of serving it."""
    import uvicorn

    calls: list[dict] = []

    def fake_run(app, **kwargs):
        calls.append({"app": app, **kwargs})

    monkeypatch.setattr(uvicorn, "run", fake_run)
    return calls


class TestAddressResolution:
    def test_settings_provide_the_default(self, project):
        from mlango.conf import settings

        host, port = _resolve_address({}, settings)
        assert host == settings.SERVE_HOST
        assert port == int(settings.SERVE_PORT)

    def test_a_bare_port(self, project):
        from mlango.conf import settings

        assert _resolve_address({"addrport": "9001"}, settings)[1] == 9001

    def test_host_and_port_together(self, project):
        from mlango.conf import settings

        assert _resolve_address({"addrport": "0.0.0.0:9001"}, settings) == ("0.0.0.0", 9001)

    def test_a_leading_colon_keeps_the_default_host(self, project):
        from mlango.conf import settings

        host, port = _resolve_address({"addrport": ":9001"}, settings)
        assert host == settings.SERVE_HOST
        assert port == 9001

    def test_an_ipv6_style_address_takes_the_last_colon(self, project):
        from mlango.conf import settings

        assert _resolve_address({"addrport": "::1:9001"}, settings) == ("::1", 9001)

    def test_explicit_flags_are_used(self, project):
        from mlango.conf import settings

        assert _resolve_address({"host": "1.2.3.4", "port": 8123}, settings) == ("1.2.3.4", 8123)

    def test_addrport_wins_over_the_port_flag(self, project):
        """The positional argument is what Django users type; it comes last."""
        from mlango.conf import settings

        assert _resolve_address({"port": 8000, "addrport": "9999"}, settings)[1] == 9999


class TestServeRun:
    def test_the_app_is_built_and_handed_to_uvicorn(self, project, uvicorn_calls):
        from mlango.serve.api import run

        run(host="127.0.0.1", port=8123)

        assert len(uvicorn_calls) == 1
        call = uvicorn_calls[0]
        assert call["host"] == "127.0.0.1"
        assert call["port"] == 8123
        assert call["app"].title

    def test_reload_passes_an_import_string_not_an_app(self, project, uvicorn_calls):
        """A reloading worker has to rebuild the app itself."""
        from mlango.serve.api import run

        run(host="127.0.0.1", port=8124, reload=True)

        call = uvicorn_calls[0]
        assert call["app"] == "mlango.serve.api:create_app"
        assert call["factory"] is True
        assert call["reload"] is True

    def test_settings_supply_host_and_port(self, project, uvicorn_calls):
        from mlango.conf import settings
        from mlango.serve.api import run

        settings.SERVE_HOST = "0.0.0.0"
        settings.SERVE_PORT = 9100
        run()

        assert uvicorn_calls[0]["host"] == "0.0.0.0"
        assert uvicorn_calls[0]["port"] == 9100

    def test_the_log_level_follows_settings(self, project, uvicorn_calls):
        from mlango.conf import settings
        from mlango.serve.api import run

        settings.LOG_LEVEL = "WARNING"
        run()
        assert uvicorn_calls[0]["log_level"] == "warning"


class TestRunserverCommand:
    def _run(self, argv):
        from mlango.management.manager import execute_from_command_line

        return execute_from_command_line(["manage.py", *argv])

    def test_the_banner_lists_what_is_declared(self, project, uvicorn_calls, capsys):
        assert self._run(["runserver", "8321"]) == 0

        out = capsys.readouterr().out
        assert "mlango development server" in out
        assert "declared" in out
        assert "http://127.0.0.1:8321/admin/" in out
        assert "http://127.0.0.1:8321/api/docs" in out
        assert uvicorn_calls[0]["port"] == 8321

    def test_no_routes_says_where_to_add_them(self, project, uvicorn_calls, capsys):
        self._run(["runserver"])
        assert "ROOT_ROUTECONF" in capsys.readouterr().out

    def test_declared_routes_are_listed(self, project, uvicorn_calls, capsys, monkeypatch):
        import sys
        import types

        from mlango.conf import settings
        from mlango.serve import path
        from mlango.serve.routing import Endpoint

        module = types.ModuleType("banner_routes")
        module.urlpatterns = [
            path("predict/", Endpoint(kind="model", label="demo.Sentiment", handler=lambda: None))
        ]
        monkeypatch.setitem(sys.modules, "banner_routes", module)
        settings.ROOT_ROUTECONF = "banner_routes"

        self._run(["runserver"])
        assert "POST /api/predict/ → demo.Sentiment" in capsys.readouterr().out

    def test_no_admin_hides_the_admin_and_disables_it(self, project, uvicorn_calls, capsys):
        from mlango.conf import settings

        self._run(["runserver", "--no-admin"])

        assert "/admin/" not in capsys.readouterr().out
        assert settings.ADMIN_ENABLED is False

    def test_reload_is_forwarded(self, project, uvicorn_calls):
        self._run(["runserver", "--reload"])
        assert uvicorn_calls[0]["reload"] is True

    def test_the_host_flag_is_used(self, project, uvicorn_calls, capsys):
        self._run(["runserver", "--host", "0.0.0.0", "--port", "8444"])

        assert uvicorn_calls[0]["host"] == "0.0.0.0"
        assert "http://0.0.0.0:8444/api/docs" in capsys.readouterr().out

    def test_programmatic_settings_are_enough_to_run_a_command(self, project, capsys):
        """settings.configure() is a documented way in; commands must accept it.

        Requiring MLANGO_SETTINGS_MODULE as well left notebooks and test suites
        unable to run any command, even with settings fully configured.
        """
        import os

        from mlango.conf import ENVIRONMENT_VARIABLE

        assert ENVIRONMENT_VARIABLE not in os.environ
        assert self._run(["check"]) == 0

    def test_without_settings_at_all_the_message_says_what_to_do(self, capsys):
        from mlango.conf import settings

        settings.reset()
        assert self._run(["check"]) == 1
        assert "MLANGO_SETTINGS_MODULE" in capsys.readouterr().err
