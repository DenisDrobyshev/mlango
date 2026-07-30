"""Scorers and the evaluation runner."""

from __future__ import annotations

import pytest

from mlango.agents import Agent, tool
from mlango.core import fields
from mlango.data import Dataset, InMemorySource
from mlango.evals import (
    Eval,
    contains,
    contains_all,
    exact_match,
    iexact_match,
    json_equals,
    json_subset,
    length_between,
    not_contains,
    numeric_close,
    regex_match,
    token_f1,
    used_tool,
)


class TestScorers:
    def test_exact_and_case_insensitive(self):
        assert exact_match(" a ", "a") is True
        assert exact_match("A", "a") is False
        assert iexact_match("A", "a") is True

    def test_contains(self):
        assert contains("the QUICK fox", "quick") is True
        assert contains("the fox", "quick") is False

    def test_contains_all_is_a_fraction(self):
        scorer = contains_all("alpha", "beta")
        assert scorer("alpha beta", None) == 1.0
        assert scorer("alpha only", None) == 0.5
        assert scorer("neither", None) == 0.0

    def test_not_contains(self):
        assert not_contains("secret")("clean text", None) is True
        assert not_contains("secret")("has a secret", None) is False

    def test_regex(self):
        assert regex_match(r"\d{3}")("abc 123", None) is True
        assert regex_match(r"\d{3}")("abc", None) is False

    def test_json_equals_ignores_key_order(self):
        assert json_equals('{"b": 2, "a": 1}', {"a": 1, "b": 2}) is True

    def test_json_subset_is_a_fraction(self):
        assert json_subset({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2}) == 1.0
        assert json_subset({"a": 1}, {"a": 1, "b": 2}) == 0.5

    def test_numeric_close(self):
        assert numeric_close(0.01)("1.005", 1.0) is True
        assert numeric_close(0.001)("1.005", 1.0) is False
        assert numeric_close()("not a number", 1.0) is False

    def test_length_between(self):
        assert length_between(2, 5)("abc", None) is True
        assert length_between(4)("abc", None) is False

    def test_token_f1(self):
        assert token_f1("the quick fox", "the quick fox") == 1.0
        assert token_f1("nothing shared", "totally different") == 0.0
        assert 0 < token_f1("the quick fox", "the slow fox") < 1

    def test_used_tool_reads_the_run(self):
        class FakeRun:
            tools_used = ["search"]

        scorer = used_tool("search")
        assert scorer.wants_run is True
        assert scorer(FakeRun(), None) is True
        assert used_tool("other")(FakeRun(), None) is False


@tool
def lookup(query: str) -> str:
    """Look something up.

    Args:
        query: What to look up.
    """
    return f"docs/{query}.md explains it"


class Cases(Dataset):
    """Question/answer pairs for the eval suite."""

    id = fields.IntegerField()
    question = fields.TextField()
    answer = fields.TextField()

    class Meta:
        primary_key = "id"
        source = InMemorySource(
            [
                {"id": 1, "question": 'use lookup {"query": "keys"}', "answer": "docs/keys.md explains it"},
                {"id": 2, "question": "hello", "answer": "something else entirely"},
            ]
        )


class Assistant(Agent):
    """An agent under evaluation."""

    class Meta:
        tools = [lookup]


class Quality(Eval):
    """Does the assistant answer from the docs?"""

    class Meta:
        dataset = Cases
        target = Assistant
        input_field = "question"
        expected_field = "answer"
        case_id_field = "id"
        scorers = {"overlap": token_f1, "cited": contains_all("docs/"), "searched": used_tool("lookup")}
        threshold = 0.5


class TestEvalRunner:
    def test_produces_a_report(self, project):
        report = Quality.evaluate()
        assert report.total == 2
        assert report.passed == 1
        assert report.failed == 1
        assert report.pass_rate == 0.5

    def test_mean_scores_are_computed(self, project):
        report = Quality.evaluate()
        means = report.mean_scores()
        assert set(means) == {"overlap", "cited", "searched"}
        assert 0.0 <= means["overlap"] <= 1.0

    def test_failures_are_listed(self, project):
        failures = Quality.evaluate().failures()
        assert [case["case_id"] for case in failures] == ["2"]

    def test_results_are_persisted(self, project):
        from sqlalchemy import select

        from mlango.metastore import EvalResult, session_scope

        Quality.evaluate()
        with session_scope() as session:
            rows = list(session.execute(select(EvalResult)).scalars())
        assert len(rows) == 2
        assert {r.case_id for r in rows} == {"1", "2"}
        assert all(set(r.scores) == {"overlap", "cited", "searched"} for r in rows)

    def test_a_run_is_recorded(self, project):
        report = Quality.evaluate()
        record = report.run.refresh()
        assert record.kind == "eval"
        assert record.status == "finished"
        assert record.summary["total"] == 2

    def test_trace_is_linked_for_agent_targets(self, project):
        report = Quality.evaluate()
        assert all(case["trace_uuid"] for case in report.cases)

    def test_max_cases_limits_the_run(self, project):
        class Limited(Eval):
            """Only the first case."""

            class Meta:
                dataset = Cases
                target = Assistant
                input_field = "question"
                expected_field = "answer"
                scorers = {"overlap": token_f1}
                max_cases = 1

        assert Limited.evaluate().total == 1

    def test_a_broken_case_is_recorded_not_raised(self, project):
        def explodes(output, expected):
            raise ZeroDivisionError("scorer bug")

        class Fragile(Eval):
            """Has a broken scorer."""

            class Meta:
                dataset = Cases
                target = Assistant
                input_field = "question"
                scorers = {"broken": explodes}

        report = Fragile.evaluate()
        assert report.errored == 2
        assert report.passed == 0
        assert all("ZeroDivisionError" in case["error"] for case in report.cases)

    def test_missing_dataset_is_reported(self):
        from mlango.core.exceptions import ImproperlyConfigured

        class NoDataset(Eval):
            """Incomplete on purpose."""

        with pytest.raises(ImproperlyConfigured, match="Meta.dataset"):
            NoDataset.get_dataset()

    def test_missing_scorers_is_reported(self, project):
        from mlango.core.exceptions import ImproperlyConfigured

        class NoScorers(Eval):
            """Incomplete on purpose."""

            class Meta:
                dataset = Cases
                target = Assistant
                input_field = "question"

        with pytest.raises(ImproperlyConfigured, match="no scorers"):
            NoScorers().score(Cases.objects.first(), "output")


class TestModelTargets:
    def test_a_model_can_be_evaluated(self, project, reviews, sklearn_or_skip):
        from mlango.training import Model

        class Tiny(Model):
            """Trained for the eval test."""

            class Meta:
                dataset = reviews
                trainer = "sklearn"
                task = "classification"
                features = ["text"]

            def build(self):
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline

                return make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=500))

        Tiny().train()

        class Accuracy(Eval):
            """Does the classifier agree with the labels?"""

            class Meta:
                dataset = reviews
                target = Tiny
                input_field = "text"
                expected_field = "label"
                case_id_field = "id"
                scorers = {"correct": exact_match}
                max_cases = 20

        report = Accuracy.evaluate()
        assert report.total == 20
        assert report.pass_rate == 1.0
