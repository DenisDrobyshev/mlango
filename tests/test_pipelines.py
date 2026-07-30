"""The two CI definitions have to agree with each other.

``.gitlab-ci.yml`` says at the top that it mirrors ``.github/workflows/ci.yml``
and that the two must be kept in step, because a contributor may only ever see
one of them. A comment did not achieve that: the GitHub audit job was fixed and
the GitLab one was left with the flags that had just been proven wrong, so the
GitLab pipeline failed and took the docs deployment down with it.

These tests are that instruction made enforceable. They check the handful of
things that actually have to match, not the file layout — the two systems
express jobs differently and always will.
"""

from __future__ import annotations

import pathlib
import re

import pytest

yaml = pytest.importorskip("yaml")

ROOT = pathlib.Path(__file__).resolve().parent.parent
GITHUB = ROOT / ".github" / "workflows" / "ci.yml"
GITLAB = ROOT / ".gitlab-ci.yml"


class _Loader(yaml.SafeLoader):
    """SafeLoader that tolerates mkdocs' Python-callable tags.

    mkdocs.yml legitimately carries ``!!python/name:...`` to wire up the mermaid
    fence. Reading it is not executing it, so the tag becomes its own name and
    the rest of the document parses.
    """


_Loader.add_multi_constructor(
    "tag:yaml.org,2002:python/name:", lambda loader, suffix, node: suffix
)


def _load(path: pathlib.Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)


def _github_commands() -> str:
    """Every shell line in the GitHub workflow, as one blob."""
    workflow = _load(GITHUB)
    lines = []
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if isinstance(step.get("run"), str):
                lines.append(step["run"])
    return "\n".join(lines)


def _gitlab_commands() -> str:
    """Every script line in the GitLab pipeline, as one blob."""
    pipeline = _load(GITLAB)
    lines = []
    for job in pipeline.values():
        if not isinstance(job, dict):
            continue
        for key in ("before_script", "script", "after_script"):
            entries = job.get(key) or []
            if isinstance(entries, str):
                entries = [entries]
            lines.extend(str(entry) for entry in entries)
    return "\n".join(lines)


@pytest.fixture(scope="module")
def commands() -> tuple[str, str]:
    return _github_commands(), _gitlab_commands()


class TestBothFilesExist:
    def test_they_parse(self):
        assert _load(GITHUB)["jobs"]
        assert _load(GITLAB)

    def test_the_gitlab_file_says_it_is_a_mirror(self):
        """The claim in the header is what these tests are enforcing."""
        header = GITLAB.read_text(encoding="utf-8")[:400]
        assert "ci.yml" in header


class TestInvariants:
    def test_mypy_is_blocking_in_both(self, commands):
        """Advisory type checking is how 87 errors accumulated the first time."""
        for blob in commands:
            assert re.search(r"^\s*-?\s*mypy mlango\s*$", blob, re.MULTILINE), blob
            assert "mypy mlango || true" not in blob
            assert "mypy mlango || exit 0" not in blob

    def test_ruff_runs_in_both(self, commands):
        for blob in commands:
            assert "ruff check mlango tests" in blob
            assert "ruff format --check mlango tests" in blob

    def test_the_audit_flags_match(self, commands):
        """--strict cannot be combined with --skip-editable; neither may use it."""
        for blob in commands:
            assert "pip-audit --desc --skip-editable" in blob
            assert "pip-audit --strict" not in blob

    def test_both_audit_against_a_current_pip(self, commands):
        """An old pip in a base image has its own advisories and fails the job.

        They are genuine, but they belong to the image rather than to anything a
        user installs — GitLab's python:3.12 shipped pip 25.0.1 with six.
        """
        for blob in commands:
            assert "pip install --upgrade pip" in blob

    def test_the_docs_build_is_strict_in_both(self, commands):
        for blob in commands:
            assert "mkdocs build --strict" in blob

    def test_the_coverage_gate_is_not_given_a_threshold_on_the_command_line(self, commands):
        """fail_under lives in pyproject.toml so a laptop enforces the same number."""
        for blob in commands:
            assert "--cov-fail-under" not in blob

    def test_py_typed_is_verified_in_both(self, commands):
        for blob in commands:
            assert "py.typed" in blob


class TestQuickstart:
    def test_both_run_the_scaffold_s_own_tests(self, commands):
        for blob in commands:
            assert "manage.py test" in blob

    def test_both_install_dev_for_the_quickstart(self, commands):
        """`manage.py test` needs pytest, so the sklearn extra alone is not enough."""
        github, gitlab = commands
        assert ".[sklearn,dev]" in github
        assert ".[sklearn,dev]" in gitlab

    def test_neither_installs_sklearn_without_dev_anywhere(self, commands):
        for blob in commands:
            bare = re.findall(r'"\.\[sklearn\]"', blob)
            assert not bare, bare

    def test_both_exercise_the_same_commands(self, commands):
        for verb in (
            "startproject",
            "check",
            "migrate",
            "makemigrations",
            "train",
            "evaluate",
            "sweep",
            "agent",
            "runs list",
            "traces list",
            "dataset head",
            "dataset materialize",
            "runserver",
        ):
            for blob in commands:
                assert verb in blob, f"{verb} missing"


class TestPythonMatrix:
    def test_the_versions_match(self):
        """A version tested on one host and not the other is a gap nobody sees."""
        github = _load(GITHUB)
        entries = github["jobs"]["test"]["strategy"]["matrix"]["include"]
        on_github = {str(entry["python"]) for entry in entries}

        gitlab = _load(GITLAB)
        on_gitlab = set(gitlab["test"]["parallel"]["matrix"][0]["PYTHON_VERSION"])

        assert on_github >= on_gitlab
        # GitLab has no macOS or Windows runners here, so GitHub covers more.
        # What must hold is that every interpreter GitLab claims is also on
        # GitHub, and that both cover the range pyproject.toml promises.
        assert on_gitlab == {"3.10", "3.11", "3.12", "3.13"}

    def test_the_matrix_covers_what_the_package_claims_to_support(self):
        # Read with a regex rather than tomllib, which is only stdlib from 3.11
        # — and 3.10 is one of the versions this very test is about.
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        claimed = set(re.findall(r"Programming Language :: Python :: (3\.\d+)", text))
        assert claimed, "pyproject.toml declares no Python version classifiers"

        entries = _load(GITHUB)["jobs"]["test"]["strategy"]["matrix"]["include"]
        tested = {str(entry["python"]) for entry in entries}
        assert claimed <= tested, f"claimed but untested: {sorted(claimed - tested)}"


class TestGitHubSpecific:
    def test_the_aggregate_check_needs_every_other_job(self):
        """Branch protection requires one check; it has to depend on all of them."""
        jobs = _load(GITHUB)["jobs"]
        aggregate = jobs["ci"]
        others = {name for name in jobs if name != "ci"}
        assert set(aggregate["needs"]) == others

    def test_the_aggregate_check_fails_on_a_failed_job(self):
        jobs = _load(GITHUB)["jobs"]
        body = yaml.safe_dump(jobs["ci"])
        assert "failure" in body
        assert "cancelled" in body

    def test_permissions_default_to_read_only(self):
        assert _load(GITHUB)["permissions"] == {"contents": "read"}

    def test_jobs_needing_the_repository_visibility_say_so(self):
        """CodeQL and Pages cannot work on a private repository."""
        codeql = _load(ROOT / ".github" / "workflows" / "codeql.yml")
        assert "visibility == 'public'" in str(codeql["jobs"]["analyze"]["if"])

        docs = _load(ROOT / ".github" / "workflows" / "docs.yml")
        assert "visibility == 'public'" in str(docs["jobs"]["deploy"]["if"])


class TestReadme:
    """The README is the PyPI landing page, so its links are load-bearing.

    They were not: relative links resolved against pypi.org and 404ed, and the
    CI badge rendered as a broken image because it points at a workflow only a
    public repository exposes. The English and Russian copies then drifted from
    each other while that was being fixed, which is the same failure the
    pipeline tests above exist to catch.
    """

    @pytest.fixture(scope="class")
    def readmes(self) -> dict[str, str]:
        return {
            name: (ROOT / name).read_text(encoding="utf-8")
            for name in ("README.md", "README.ru.md")
        }

    def test_no_relative_links(self, readmes):
        """PyPI renders the README standalone; a relative link goes nowhere."""
        for name, text in readmes.items():
            relative = re.findall(r"\]\((?!https?:|#|mailto:)([^)]+)\)", text)
            assert not relative, f"{name}: {relative}"

    def test_both_carry_the_same_badges(self, readmes):
        """Including the CI badge, which only renders while the repo is public."""
        badges = {
            name: sorted(
                set(re.findall(r"img\.shields\.io/([^)\s]+)", text))
                | set(re.findall(r"(actions/workflows/[^)\s]+)", text))
            )
            for name, text in readmes.items()
        }
        assert badges["README.md"] == badges["README.ru.md"]
        assert badges["README.md"], "no badges at all"

    def test_both_list_the_same_manage_py_commands(self, readmes):
        """The command list is the tour of the CLI; a stale copy undersells it."""
        commands = {
            name: sorted(set(re.findall(r"manage\.py ([a-z]+)", text)))
            for name, text in readmes.items()
        }
        assert commands["README.md"] == commands["README.ru.md"]

    def test_both_link_to_the_documentation_site(self, readmes):
        for name, text in readmes.items():
            assert "drobyshevdev.github.io/mlango" in text, name

    def test_each_points_at_the_other(self, readmes):
        assert "README.ru.md" in readmes["README.md"]
        assert "README.md" in readmes["README.ru.md"]

    def test_both_show_the_install_command(self, readmes):
        for name, text in readmes.items():
            assert 'pip install "mlango[sklearn]"' in text, name


class TestDocumentation:
    """The project claims complete English and Russian documentation.

    That claim slipped twice while pages were being added, both times
    unnoticed, because an untranslated page falls back to English and nothing
    visibly breaks. These make it visible.
    """

    @pytest.fixture(scope="class")
    def pages(self) -> tuple[set[str], set[str]]:
        docs = ROOT / "docs"
        english = {p.stem for p in docs.glob("*.md") if ".ru." not in p.name}
        russian = {p.name[: -len(".ru.md")] for p in docs.glob("*.ru.md")}
        return english, russian

    def test_every_page_exists_in_both_languages(self, pages):
        english, russian = pages
        assert not english - russian, f"no Russian: {sorted(english - russian)}"
        assert not russian - english, f"no English: {sorted(russian - english)}"

    def test_every_page_is_in_the_navigation(self, pages):
        english, _ = pages
        nav = yaml.safe_dump(_load(ROOT / "mkdocs.yml")["nav"])
        missing = [name for name in english if f"{name}.md" not in nav]
        assert not missing, f"written but unreachable: {sorted(missing)}"

    def test_the_navigation_labels_are_all_translated(self):
        config = _load(ROOT / "mkdocs.yml")
        russian = next(
            locale for locale in config["plugins"][1]["i18n"]["languages"]
            if locale["locale"] == "ru"
        )
        translated = set(russian["nav_translations"])

        labels: set[str] = set()
        for section in config["nav"]:
            for heading, entries in section.items():
                labels.add(heading)
                labels.update(k for entry in entries for k in entry)

        assert not labels - translated, f"untranslated labels: {sorted(labels - translated)}"

    def test_both_readmes_explain_the_architecture(self):
        """It is the question a reader has before any feature list."""
        for name in ("README.md", "README.ru.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            assert "_meta" in text, name
            assert "manage.py train" in text, name


class TestGitLabSpecific:
    def test_pages_builds_into_the_directory_gitlab_serves(self):
        """The job name and `public/` are fixed by GitLab, not by us."""
        pipeline = _load(GITLAB)
        pages = pipeline["pages"]

        assert "--site-dir public" in "\n".join(pages["script"])
        assert pages["artifacts"]["paths"] == ["public/"]

    def test_pages_only_publishes_from_the_default_branch(self):
        pages = _load(GITLAB)["pages"]
        assert "CI_DEFAULT_BRANCH" in str(pages["rules"])

    def test_the_strict_docs_build_still_runs_off_the_default_branch(self):
        """Otherwise a broken link would only be caught after merging."""
        docs = _load(GITLAB)["docs"]
        assert "!=" in str(docs["rules"])
