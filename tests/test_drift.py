"""Profiling a training split, and measuring how far production has moved."""

from __future__ import annotations

import math
import random

import pytest

from mlango.training import drift


def numbers(n: int, low: float, high: float, seed: int = 0) -> list[dict[str, float]]:
    rng = random.Random(seed)
    return [{"x": rng.uniform(low, high)} for _ in range(n)]


class TestProfile:
    def test_a_numeric_column_gets_quantile_buckets(self):
        profile = drift.profile([{"x": float(i)} for i in range(100)], ["x"])["x"]
        assert profile["kind"] == "numeric"
        assert profile["min"] == 0.0
        assert profile["max"] == 99.0
        assert len(profile["counts"]) == len(profile["edges"]) + 1
        assert sum(profile["counts"]) == 100

    def test_buckets_hold_equal_counts_not_equal_widths(self):
        """Equal-width buckets on a skewed column put everything in one of them."""
        skewed = [{"x": float(i) ** 3} for i in range(200)]
        counts = drift.profile(skewed, ["x"])["x"]["counts"]
        assert max(counts) - min(counts) <= 1

    def test_a_categorical_column_counts_its_values(self):
        rows = [{"label": "pos" if i % 4 else "neg"} for i in range(100)]
        profile = drift.profile(rows, ["label"])["label"]
        assert profile["kind"] == "categorical"
        assert profile["values"] == {"pos": 75, "neg": 25}

    def test_free_text_is_profiled_by_length(self):
        rows = [{"body": f"row number {i} is unique"} for i in range(200)]
        profile = drift.profile(rows, ["body"])["body"]
        assert profile["kind"] == "text"
        assert profile["distinct"] == 200
        assert profile["min"] > 0

    def test_booleans_are_categories_not_numbers(self):
        """True is not 1.0: quantile buckets over two values measure nothing."""
        rows = [{"flag": bool(i % 2)} for i in range(50)]
        assert drift.profile(rows, ["flag"])["flag"]["kind"] == "categorical"

    def test_missing_values_are_counted_separately(self):
        rows = [{"x": 1.0}, {"x": None}, {}]
        profile = drift.profile(rows, ["x"])["x"]
        assert profile["count"] == 3
        assert profile["missing"] == 2

    def test_bare_values_are_filed_under_the_only_column(self):
        """A single-feature model hands the trainer values, not dicts."""
        profile = drift.profile(["a", "b", "a"], ["text"])["text"]
        assert profile["kind"] == "categorical"
        assert profile["values"]["a"] == 2

    def test_a_column_of_one_repeated_value_does_not_crash(self):
        profile = drift.profile([{"x": 5.0}] * 20, ["x"])["x"]
        assert profile["edges"] == []
        assert profile["counts"] == [20]

    def test_no_records_profiles_nothing(self):
        assert drift.profile([], ["x"]) == {}


class TestCompare:
    def test_the_same_distribution_scores_near_zero(self):
        baseline = drift.profile(numbers(500, 0, 10, seed=1), ["x"])
        scores = drift.compare(baseline, numbers(500, 0, 10, seed=2))
        assert scores["x"]["psi"] < drift.STABLE
        assert scores["x"]["verdict"] == "stable"

    def test_a_shifted_distribution_is_significant(self):
        baseline = drift.profile(numbers(500, 0, 10, seed=1), ["x"])
        scores = drift.compare(baseline, numbers(500, 20, 30, seed=2))
        assert scores["x"]["psi"] > drift.SIGNIFICANT
        assert scores["x"]["verdict"] == "significant"

    def test_a_categorical_flip_is_caught(self):
        baseline = drift.profile(
            [{"label": "pos" if i % 2 else "neg"} for i in range(200)], ["label"]
        )
        scores = drift.compare(baseline, [{"label": "pos"} for _ in range(200)])
        assert scores["label"]["verdict"] == "significant"

    def test_an_unseen_category_moves_the_score(self):
        baseline = drift.profile([{"label": "a"} for _ in range(100)], ["label"])
        scores = drift.compare(baseline, [{"label": "b"} for _ in range(100)])
        assert scores["label"]["psi"] > drift.SIGNIFICANT

    def test_the_observed_column_is_read_as_the_baseline_read_it(self):
        """Production text that starts repeating is still text, not a category.

        Inferring the kind twice would return no score for exactly the case
        worth catching: an input that has collapsed to a few values.
        """
        baseline = drift.profile(
            [{"body": f"a distinct sentence {i}"} for i in range(200)], ["body"]
        )
        assert baseline["body"]["kind"] == "text"

        scores = drift.compare(baseline, [{"body": "ok"} for _ in range(200)])
        assert scores["body"]["kind"] == "text"
        assert scores["body"]["verdict"] == "significant"

    def test_a_column_the_observations_lack_is_skipped(self):
        """A payload missing a feature is a caller bug, not drift."""
        baseline = drift.profile([{"x": 1.0, "y": 2.0}] * 50, ["x", "y"])
        scores = drift.compare(baseline, [{"x": 1.0}] * 50)
        assert set(scores) == {"x"}

    def test_a_column_that_changed_meaning_reports_nothing(self):
        baseline = drift.profile([{"x": float(i)} for i in range(100)], ["x"])
        assert drift.compare(baseline, [{"x": "not a number"}] * 100) == {}

    def test_the_row_count_is_carried_through(self):
        baseline = drift.profile(numbers(200, 0, 5), ["x"])
        assert drift.compare(baseline, numbers(37, 0, 5, seed=9))["x"]["count"] == 37

    def test_an_empty_bucket_does_not_produce_infinity(self):
        """PSI divides by proportions; one empty bucket must not blow it up."""
        baseline = drift.profile([{"x": float(i)} for i in range(100)], ["x"])
        score = drift.compare(baseline, [{"x": 0.0}] * 100)["x"]["psi"]
        assert math.isfinite(score)
        assert score > drift.SIGNIFICANT


class TestVerdict:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (0.0, "stable"),
            (0.099, "stable"),
            (0.1, "moderate"),
            (0.24, "moderate"),
            (0.25, "significant"),
            (12.0, "significant"),
        ],
    )
    def test_thresholds(self, score, expected):
        assert drift.verdict(score) == expected
