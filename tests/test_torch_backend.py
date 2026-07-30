"""The PyTorch backend.

Kept deliberately small — a few epochs over a handful of rows on the CPU. The
point is that the contract holds (shapes agree between training and inference,
callbacks fire, checkpoints reload), not that the network learns anything.
"""

from __future__ import annotations

import pytest

from mlango.core import fields
from mlango.data import Dataset, InMemorySource

pytestmark = pytest.mark.usefixtures("isolated_registry")

ROWS = [
    {
        "id": index,
        "x1": float(index % 7),
        "x2": float((index * 3) % 5),
        "label": "high" if index % 2 else "low",
        "score": float(index) / 10.0,
    }
    for index in range(40)
]


@pytest.fixture(scope="module")
def torch_or_skip():
    return pytest.importorskip("torch")


@pytest.fixture
def tabular(project, torch_or_skip):
    """A tiny classifier over two numeric features."""
    from mlango.training import Model

    class Rows(Dataset):
        id = fields.IntegerField()
        x1 = fields.FloatField()
        x2 = fields.FloatField()
        label = fields.LabelField(["low", "high"])

        class Meta:
            source = InMemorySource(ROWS)
            primary_key = "id"

    class Net(Model):
        """Two inputs, two classes."""

        epochs = fields.IntegerField(default=2)
        batch_size = fields.IntegerField(default=8)

        class Meta:
            dataset = Rows
            trainer = "torch"
            task = "classification"
            features = ["x1", "x2"]

        def build(self):
            import torch.nn as nn

            return nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))

    return Rows, Net


class TestTraining:
    def test_it_trains_and_records_a_run(self, tabular):
        _rows, net = tabular
        run = net().train()

        record = run.refresh()
        assert record.status == "finished"
        assert record.params["_features"] == ["x1", "x2"]
        assert record.summary

    def test_metrics_are_recorded_per_epoch(self, tabular):
        from mlango.training.run import metric_history

        _rows, net = tabular
        run = net(epochs=3).train()

        history = metric_history(run.refresh().id, "loss")
        assert len(history) == 3
        assert all(value > 0 for _step, value in history)

    def test_the_device_is_recorded(self, tabular):
        _rows, net = tabular
        record = net().train().refresh()
        assert record.device in {"cpu", "cuda"}


class TestInference:
    @pytest.fixture
    def trained(self, tabular):
        rows, net = tabular
        model = net()
        model.train()
        return rows, model

    def test_a_dict_input(self, trained):
        _rows, model = trained
        assert model.predict({"x1": 1.0, "x2": 2.0}) in {"low", "high"}

    def test_a_batch_of_dicts(self, trained):
        _rows, model = trained
        out = model.predict([{"x1": 1.0, "x2": 2.0}, {"x1": 5.0, "x2": 0.0}])
        assert len(out) == 2
        assert all(value in {"low", "high"} for value in out)

    def test_probabilities_cover_every_class(self, trained):
        _rows, model = trained
        proba = model.predict_proba({"x1": 1.0, "x2": 2.0})
        assert set(proba) == {"low", "high"}
        assert abs(sum(proba.values()) - 1.0) < 1e-5

    def test_predict_does_not_touch_the_caller_s_data(self, trained):
        """The rows belong to the caller.

        _encode needs the target key present even at inference, where the answer
        is the thing being asked for. Filling it in on the caller's own dict left
        a fabricated label sitting in their data, indistinguishable from ground
        truth in whatever they did with the row next.
        """
        _rows, model = trained
        row = {"x1": 1.0, "x2": 2.0}

        model.predict([row])
        assert row == {"x1": 1.0, "x2": 2.0}

        model.predict_proba([row])
        assert row == {"x1": 1.0, "x2": 2.0}

    def test_a_supplied_target_is_left_alone(self, trained):
        """A row that already carries a label must not have it overwritten."""
        _rows, model = trained
        row = {"x1": 1.0, "x2": 2.0, "label": "high"}

        model.predict([row])
        assert row["label"] == "high"

    def test_a_saved_model_reloads_and_predicts(self, trained):
        rows, model = trained
        reloaded = type(model).load()
        assert reloaded.predict({"x1": 1.0, "x2": 2.0}) in {"low", "high"}


class TestRegression:
    @pytest.fixture
    def regressor(self, project, torch_or_skip):
        from mlango.training import Model

        class Points(Dataset):
            id = fields.IntegerField()
            x1 = fields.FloatField()
            score = fields.TargetField()

            class Meta:
                source = InMemorySource(ROWS)
                primary_key = "id"

        class Line(Model):
            epochs = fields.IntegerField(default=2)

            class Meta:
                dataset = Points
                trainer = "torch"
                task = "regression"
                features = ["x1"]

            def build(self):
                import torch.nn as nn

                return nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 1))

        return Line

    def test_a_continuous_target(self, regressor):
        model = regressor()
        model.train()
        assert isinstance(model.predict({"x1": 2.0}), float)

    def test_probabilities_are_absent_for_regression(self, regressor):
        model = regressor()
        model.train()
        assert model.predict_proba({"x1": 2.0}) is None

    def test_a_scalar_input_is_accepted_for_a_single_feature(self, regressor):
        model = regressor()
        model.train()
        assert isinstance(model.predict(2.0), float)

    def test_the_loss_pairs_row_wise(self, regressor, recwarn):
        """A regression head outputs (N, 1) against targets of (N,).

        MSELoss broadcasts that into an (N, N) matrix, so the recorded loss
        compares every prediction with every target and the gradients are wrong.
        Torch only warns, and a plausible-looking curve is worse than a crash.
        """
        regressor().train()

        broadcasts = [w for w in recwarn.list if "different to the input size" in str(w.message)]
        assert not broadcasts, [str(w.message) for w in broadcasts]

    def test_the_loss_is_the_mean_squared_error_it_claims_to_be(self, torch_or_skip):
        """Pin the number, not just the absence of a warning.

        Perfect predictions, so the paired loss is exactly zero. Broadcasting
        compares row 0's prediction with row 1's target as well, and reports a
        loss of 2.0 for a model that got everything right.
        """
        import torch

        from mlango.training.backends.torch_backend import _align

        outputs = torch.tensor([[1.0], [3.0]])
        targets = torch.tensor([1.0, 3.0])

        paired = torch.nn.MSELoss()(_align(outputs, targets), targets)
        assert float(paired) == pytest.approx(0.0)

        with pytest.warns(UserWarning, match="different to the input size"):
            broadcast = torch.nn.MSELoss()(outputs, targets)
        assert float(broadcast) == pytest.approx(2.0)

    def test_align_leaves_a_classification_head_alone(self, torch_or_skip):
        """Class logits are (N, C) against indices of (N,) — that pairing is correct."""
        import torch

        from mlango.training.backends.torch_backend import _align

        logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        indices = torch.tensor([1, 0])
        assert _align(logits, indices).shape == logits.shape


class TestEncodingErrors:
    def test_a_non_numeric_feature_says_to_override_encode_batch(self, project, torch_or_skip):
        from mlango.core.exceptions import RunError
        from mlango.training import Model

        class Texts(Dataset):
            id = fields.IntegerField()
            text = fields.TextField()
            label = fields.LabelField(["a", "b"])

            class Meta:
                source = InMemorySource(
                    [
                        {"id": i, "text": f"row {i}", "label": "a" if i % 2 else "b"}
                        for i in range(8)
                    ]
                )
                primary_key = "id"

        class Net(Model):
            class Meta:
                dataset = Texts
                trainer = "torch"
                task = "classification"
                features = ["text"]

            def build(self):
                import torch.nn as nn

                return nn.Linear(1, 2)

        with pytest.raises(RunError, match="encode_batch"):
            Net().train()
