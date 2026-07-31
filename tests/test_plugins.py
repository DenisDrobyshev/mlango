"""Entry-point discovery: what makes ``pip install mlango-lightgbm`` enough."""

from __future__ import annotations

import pytest

from mlango.core import plugins


class FakeEntryPoint:
    def __init__(self, name: str, value: str):
        self.name = name
        self.value = value


@pytest.fixture(autouse=True)
def clean_cache():
    plugins.clear_cache()
    yield
    plugins.clear_cache()


def install(monkeypatch, **groups: list[FakeEntryPoint]):
    """Pretend the named entry-point groups are advertised by installed packages."""

    def fake_entry_points(*, group: str):
        return groups.get(group, [])

    monkeypatch.setattr(plugins, "entry_points", fake_entry_points)


class TestDiscovery:
    def test_nothing_installed_finds_nothing(self, monkeypatch):
        install(monkeypatch)
        assert plugins.discover("mlango.trainers") == {}

    def test_an_entry_point_becomes_a_dotted_path(self, monkeypatch):
        """Packaging writes module:Object; mlango settings are written module.Object."""
        install(
            monkeypatch,
            **{"mlango.trainers": [FakeEntryPoint("lightgbm", "mlango_lightgbm.trainer:Trainer")]},
        )
        assert plugins.discover("mlango.trainers") == {
            "lightgbm": "mlango_lightgbm.trainer.Trainer"
        }

    def test_only_the_first_colon_is_replaced(self, monkeypatch):
        """`module:Outer.Inner` is a legal entry point and must survive intact."""
        install(
            monkeypatch,
            **{"mlango.providers": [FakeEntryPoint("x", "pkg.mod:Outer.Inner")]},
        )
        assert plugins.discover("mlango.providers")["x"] == "pkg.mod.Outer.Inner"

    def test_the_scan_happens_once(self, monkeypatch):
        calls = []

        def counting(*, group):
            calls.append(group)
            return []

        monkeypatch.setattr(plugins, "entry_points", counting)
        plugins.discover("mlango.trainers")
        plugins.discover("mlango.trainers")
        assert calls == ["mlango.trainers"]

    def test_a_broken_distribution_does_not_stop_start_up(self, monkeypatch, caplog):
        """One bad package on the machine must not make every project unusable."""
        import logging

        def explode(*, group):
            raise RuntimeError("a metadata file is corrupt")

        monkeypatch.setattr(plugins, "entry_points", explode)
        with caplog.at_level(logging.WARNING, logger="mlango.plugins"):
            assert plugins.discover("mlango.trainers") == {}
        assert "mlango.trainers" in caplog.text

    def test_installed_reports_every_group(self, monkeypatch):
        install(monkeypatch, **{"mlango.trainers": [FakeEntryPoint("a", "pkg:A")]})
        assert plugins.installed() == {"TRAINERS": {"a": "pkg.A"}, "PROVIDERS": {}}


class TestPrecedence:
    def test_a_plugin_extends_the_defaults(self, monkeypatch):
        install(monkeypatch, **{"mlango.trainers": [FakeEntryPoint("lightgbm", "pkg:LGB")]})
        merged = plugins.merged("TRAINERS", {"sklearn": "builtin.Sklearn"}, {})
        assert merged == {"sklearn": "builtin.Sklearn", "lightgbm": "pkg.LGB"}

    def test_the_project_beats_a_plugin(self, monkeypatch):
        """Pointing a name at a patched subclass must not need an uninstall."""
        install(monkeypatch, **{"mlango.trainers": [FakeEntryPoint("lightgbm", "pkg:LGB")]})
        merged = plugins.merged("TRAINERS", {}, {"lightgbm": "myproject.trainers.Patched"})
        assert merged["lightgbm"] == "myproject.trainers.Patched"

    def test_a_plugin_beats_a_framework_default(self, monkeypatch):
        install(monkeypatch, **{"mlango.providers": [FakeEntryPoint("echo", "pkg:LoudEcho")]})
        assert plugins.merged("PROVIDERS", {"echo": "builtin.Echo"}, {})["echo"] == "pkg.LoudEcho"

    def test_a_setting_with_no_group_is_left_alone(self, monkeypatch):
        install(monkeypatch, **{"mlango.trainers": [FakeEntryPoint("a", "pkg:A")]})
        assert plugins.merged("SOMETHING_ELSE", {"x": "y"}, {}) == {"x": "y"}


class TestThroughSettings:
    def test_a_discovered_trainer_is_usable_without_editing_settings(self, monkeypatch, tmp_path):
        """The whole point, end to end: install a package, name the trainer."""
        pytest.importorskip("sklearn")

        from mlango.conf import settings
        from mlango.metastore.session import dispose_all
        from mlango.storage import reset_default_storage
        from mlango.training.trainer import clear_trainer_cache, get_trainer

        install(
            monkeypatch,
            **{
                "mlango.trainers": [
                    FakeEntryPoint(
                        "vendored", "mlango.training.backends.sklearn_backend:SklearnTrainer"
                    )
                ]
            },
        )

        dispose_all()
        reset_default_storage()
        clear_trainer_cache()
        settings.configure(BASE_DIR=str(tmp_path), INSTALLED_APPS=[])
        try:
            assert "vendored" in settings.TRAINERS
            assert get_trainer("vendored").name == "sklearn"
        finally:
            clear_trainer_cache()
            settings.reset()

    def test_the_builtin_registry_is_unchanged_when_nothing_is_installed(
        self, monkeypatch, tmp_path
    ):
        from mlango.conf import global_settings, settings

        install(monkeypatch)
        settings.configure(BASE_DIR=str(tmp_path), INSTALLED_APPS=[])
        try:
            assert settings.TRAINERS == dict(global_settings.TRAINERS)
        finally:
            settings.reset()


class TestScaffolding:
    """``mlango startplugin`` — the packaging is the part people get wrong."""

    def test_names_are_derived_from_the_distribution(self):
        from mlango.plugin_template import class_name, entry_name, module_name

        assert module_name("mlango-lightgbm") == "mlango_lightgbm"
        assert entry_name("mlango-lightgbm") == "lightgbm"
        assert class_name("lightgbm", "trainer") == "LightgbmTrainer"

    def test_a_name_without_the_prefix_still_works(self):
        from mlango.plugin_template import entry_name, module_name

        assert module_name("my-thing") == "my_thing"
        assert entry_name("my-thing") == "my_thing"

    def test_a_trainer_package_declares_its_entry_point(self, tmp_path):
        from mlango.plugin_template import render_plugin

        render_plugin("mlango-lightgbm", str(tmp_path), kind="trainer")
        pyproject = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")

        assert '[project.entry-points."mlango.trainers"]' in pyproject
        assert 'lightgbm = "mlango_lightgbm.trainer:LightgbmTrainer"' in pyproject

    def test_a_provider_package_uses_the_provider_group(self, tmp_path):
        from mlango.plugin_template import render_plugin

        render_plugin("mlango-cohere", str(tmp_path), kind="provider")
        pyproject = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")

        assert '[project.entry-points."mlango.providers"]' in pyproject
        assert 'cohere = "mlango_cohere.provider:CohereProvider"' in pyproject

    @pytest.mark.parametrize("kind", ["storage", "source"])
    def test_kinds_with_nothing_to_discover_declare_no_entry_point(self, kind, tmp_path):
        """A project has one storage backend and imports its sources by name."""
        from mlango.plugin_template import render_plugin

        render_plugin("mlango-thing", str(tmp_path), kind=kind)
        pyproject = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")

        assert "entry-points" not in pyproject
        assert (tmp_path / f"src/mlango_thing/{kind}.py").exists()

    def test_every_placeholder_is_replaced(self, tmp_path):
        """A surviving token ships a package that names a class nobody wrote."""
        from mlango.plugin_template import KINDS, render_plugin

        tokens = (
            "__DIST__",
            "__MODULE__",
            "__ENTRY__",
            "__CLASS__",
            "__KIND__",
            "__GROUP__",
            "__YEAR__",
            "__AUTHOR__",
            "__USAGE__",
            "__ENTRY_POINTS__",
            "__DISCOVERY_TEST__",
            "__SETTING__",
        )
        for index, kind in enumerate(KINDS):
            target = tmp_path / f"pkg{index}"
            for path in render_plugin(f"mlango-{kind}test", str(target), kind=kind):
                text = open(path, encoding="utf-8").read()
                left = [token for token in tokens if token in text]
                assert not left, f"{path} kept {', '.join(left)}"

    def test_the_generated_package_is_importable_python(self, tmp_path):
        import ast

        from mlango.plugin_template import KINDS, render_plugin

        for index, kind in enumerate(KINDS):
            target = tmp_path / f"pkg{index}"
            for path in render_plugin(f"mlango-{kind}test", str(target), kind=kind):
                if path.endswith(".py"):
                    ast.parse(open(path, encoding="utf-8").read())

    def test_the_pyproject_parses(self, tmp_path):
        from mlango.plugin_template import render_plugin

        render_plugin("mlango-lightgbm", str(tmp_path), kind="trainer")
        try:
            import tomllib
        except ImportError:
            pytest.skip("tomllib is 3.11+")

        with open(tmp_path / "pyproject.toml", "rb") as fh:
            parsed = tomllib.load(fh)

        assert parsed["project"]["name"] == "mlango-lightgbm"
        assert parsed["project"]["entry-points"]["mlango.trainers"] == {
            "lightgbm": "mlango_lightgbm.trainer:LightgbmTrainer"
        }

    def test_the_declared_entry_point_is_the_one_discovery_reads(self, tmp_path, monkeypatch):
        """The scaffold and the loader must agree, or the package silently does nothing."""
        from mlango.plugin_template import GROUPS as SCAFFOLD_GROUPS

        assert set(SCAFFOLD_GROUPS.values()) <= set(plugins.GROUPS.values())

    def test_an_unknown_kind_is_refused(self, tmp_path):
        from mlango.plugin_template import render_plugin

        with pytest.raises(ValueError, match="Unknown kind"):
            render_plugin("mlango-x", str(tmp_path), kind="wizard")
