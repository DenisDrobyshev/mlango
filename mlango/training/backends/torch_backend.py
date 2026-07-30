"""PyTorch backend.

``build()`` returns an ``nn.Module``; the training loop, device placement,
metric logging and checkpointing are the framework's. A model customises the
parts that are genuinely model-specific by overriding ``encode_batch``,
``configure_optimizer`` or ``loss_fn``.

Device follows ``settings.DEVICE`` and defaults to CUDA when it is available —
long CPU training loops are both slow and, on some Windows builds, unstable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mlango.core.exceptions import RunError
from mlango.core.signals import epoch_finished, epoch_started
from mlango.training import metrics as metric_lib
from mlango.training.trainer import Trainer

#: Hyperparameter field names the loop looks for on the model.
EPOCHS = "epochs"
BATCH_SIZE = "batch_size"
LEARNING_RATE = "learning_rate"


class TorchTrainer(Trainer):
    name = "torch"
    requires = ("torch",)
    extension = "pt"

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _param(model: Any, name: str, default: Any) -> Any:
        return getattr(model, name, default) if model._meta.has_field(name) else default

    def _encode(
        self, model: Any, records: list[Any], target: str, features: list[str] | None = None
    ) -> tuple[Any, Any]:
        """Turn records into tensors.

        A model that overrides ``encode_batch`` owns this entirely. The default
        handles the common tabular case: numeric feature fields stacked into a
        float matrix, targets as class indices or floats.
        """
        import torch

        custom = getattr(model, "encode_batch", None)
        if callable(custom):
            return custom(records, target)

        dataset_class = model.get_dataset()
        names = features or model.get_features(dataset_class)
        target_field = dataset_class._meta.get_field(target)

        try:
            matrix = np.asarray(
                [[float(record.get(name)) for name in names] for record in records],
                dtype=np.float32,
            )
        except (TypeError, ValueError) as exc:
            raise RunError(
                f"{type(model)._meta.label}: the default encoder needs numeric feature fields. "
                f"Override encode_batch(records, target) to handle {names}."
            ) from exc

        raw_targets = [record.get(target) for record in records]
        classes = getattr(target_field, "classes", None)
        if classes:
            targets = torch.tensor(
                [classes.index(value) for value in raw_targets], dtype=torch.long
            )
        else:
            targets = torch.tensor([float(value) for value in raw_targets], dtype=torch.float32)
        return torch.from_numpy(matrix), targets

    def _optimizer(self, model: Any, module: Any):
        import torch

        custom = getattr(model, "configure_optimizer", None)
        if callable(custom):
            return custom(module)
        lr = float(self._param(model, LEARNING_RATE, 1e-3))
        return torch.optim.Adam(module.parameters(), lr=lr)

    def _loss(self, model: Any, target_field: Any):
        import torch

        custom = getattr(model, "loss_fn", None)
        if callable(custom):
            return custom()
        if getattr(target_field, "classes", None):
            return torch.nn.CrossEntropyLoss()
        return torch.nn.MSELoss()

    # -- training ------------------------------------------------------------

    def fit(
        self,
        model,
        train,
        validation,
        run,
        callbacks,
        *,
        target: str = "",
        features: list[str] | None = None,
        **kwargs: Any,
    ):
        import torch

        module = model.build()
        if not isinstance(module, torch.nn.Module):
            raise RunError(
                f"{type(model)._meta.label}.build() returned {type(module).__name__}; the torch "
                f"trainer expects a torch.nn.Module."
            )

        device = torch.device(kwargs.get("device") or self.resolve_device())
        module.to(device)

        dataset_class = model.get_dataset()
        target = target or model.get_target(dataset_class)
        target_field = dataset_class._meta.get_field(target)
        features = features or model.get_features(dataset_class)

        epochs = int(kwargs.get(EPOCHS) or self._param(model, EPOCHS, 10))
        batch_size = int(kwargs.get(BATCH_SIZE) or self._param(model, BATCH_SIZE, 32))
        optimizer = self._optimizer(model, module)
        criterion = self._loss(model, target_field)

        train_records = list(train)
        if not train_records:
            raise RunError("The training split is empty; check the dataset source and filters.")
        validation_records = list(validation) if validation is not None else []

        step = 0
        for epoch in range(1, epochs + 1):
            epoch_started.send(sender=type(model), run=run, epoch=epoch)
            callbacks.emit("on_epoch_begin", run, epoch, model=model)

            module.train()
            total_loss, seen = 0.0, 0
            for start in range(0, len(train_records), batch_size):
                batch = train_records[start : start + batch_size]
                inputs, targets = self._encode(model, batch, target, features)
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = module(inputs)
                loss = criterion(_align(outputs, targets), targets)
                loss.backward()
                optimizer.step()

                step += 1
                total_loss += float(loss.item()) * len(batch)
                seen += len(batch)
                if step % 50 == 0:
                    run.log_metric("batch_loss", float(loss.item()), step=step, split="train")
                    callbacks.emit("on_batch_end", run, step, {"loss": float(loss.item())})

            epoch_metrics = {"loss": total_loss / max(seen, 1)}
            if validation_records:
                epoch_metrics.update(
                    self._validate(
                        model, module, criterion, validation_records, target, features, device
                    )
                )

            # Recording metrics is the framework's job, not a callback's.
            run.log_metrics(epoch_metrics, epoch=epoch, step=epoch)
            callbacks.emit(
                "on_epoch_end",
                run,
                epoch,
                epoch_metrics,
                model=model,
                trainer=self,
                fitted=module,
            )
            epoch_finished.send(sender=type(model), run=run, epoch=epoch, metrics=epoch_metrics)

            if run.should_stop:
                run.log_metric("stopped_epoch", epoch, step=epoch)
                break

        module.eval()
        return module

    def _validate(
        self, model, module, criterion, records, target, features, device
    ) -> dict[str, float]:
        import torch

        module.eval()
        total, seen = 0.0, 0
        predictions: list[Any] = []
        truth: list[Any] = []

        with torch.no_grad():
            for start in range(0, len(records), 256):
                batch = records[start : start + 256]
                inputs, targets = self._encode(model, batch, target, features)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = module(inputs)
                total += float(criterion(_align(outputs, targets), targets).item()) * len(batch)
                seen += len(batch)
                predictions.extend(_decode(model, outputs, target))
                truth.extend(record.get(target) for record in batch)

        metrics = {"val_loss": total / max(seen, 1)}
        report = metric_lib.report_for_task(model.get_task(), truth, predictions)
        metrics.update({f"val_{k}": v for k, v in metric_lib.flatten_report(report).items()})
        metrics.update(
            {k: v for k, v in metric_lib.flatten_report(report).items() if k in {"accuracy", "r2"}}
        )
        return metrics

    # -- inference -----------------------------------------------------------

    def predict(self, model, fitted, inputs: list[Any]) -> list[Any]:
        import torch

        dataset_class = model.get_dataset()
        target = model.get_target(dataset_class)
        features = model.get_features(dataset_class)
        placeholder = _placeholder(dataset_class._meta.get_field(target))
        records = _as_records(inputs, features, target, placeholder)

        device = next(fitted.parameters()).device
        encoded, _targets = self._encode(model, records, target, features)
        fitted.eval()
        with torch.no_grad():
            outputs = fitted(encoded.to(device))
        return _decode(model, outputs, target)

    def predict_proba(self, model, fitted, inputs: list[Any]):
        import torch

        dataset_class = model.get_dataset()
        target = model.get_target(dataset_class)
        target_field = dataset_class._meta.get_field(target)
        classes = getattr(target_field, "classes", None)
        if not classes:
            return None

        features = model.get_features(dataset_class)
        records = _as_records(inputs, features, target, classes[0])

        device = next(fitted.parameters()).device
        encoded, _ = self._encode(model, records, target, features)
        fitted.eval()
        with torch.no_grad():
            logits = fitted(encoded.to(device))
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        return [dict(zip(classes, (float(p) for p in row), strict=True)) for row in probabilities]

    # -- persistence ---------------------------------------------------------

    def save(self, model, fitted, name: str) -> str:
        import torch

        from mlango.storage import default_storage

        path = default_storage().path(f"{name}.{self.extension}")
        torch.save({"state_dict": fitted.state_dict(), "params": model.to_dict()}, path)
        return path

    def load(self, model, path: str):
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        module = model.build()
        module.load_state_dict(payload["state_dict"])
        module.eval()
        return module

    def describe(self, model, fitted) -> dict[str, Any]:
        total = sum(p.numel() for p in fitted.parameters())
        trainable = sum(p.numel() for p in fitted.parameters() if p.requires_grad)
        return {
            "backend": self.name,
            "module": type(fitted).__name__,
            "parameters": int(total),
            "trainable_parameters": int(trainable),
            "device": str(next(fitted.parameters()).device),
        }


def _align(outputs: Any, targets: Any) -> Any:
    """Drop a trailing singleton dimension so a regression loss pairs row-wise.

    A regression head is almost always ``nn.Linear(..., 1)``, which gives
    ``(N, 1)`` against targets of ``(N,)``. MSELoss broadcasts that into an
    ``(N, N)`` matrix and returns a number that looks like a loss but compares
    every prediction against every target — the run reports a plausible curve
    while the gradients are wrong. Torch warns about it; a framework should not
    make the user read warnings to find out its training loop is broken.
    """
    if targets.dim() == 1 and outputs.dim() == 2 and outputs.shape[-1] == 1:
        return outputs.squeeze(-1)
    return outputs


def _decode(model: Any, outputs: Any, target: str) -> list[Any]:
    """Logits or regression outputs back into declared label values."""
    import torch

    dataset_class = model.get_dataset()
    target_field = dataset_class._meta.get_field(target)
    classes = getattr(target_field, "classes", None)
    if classes:
        indices = torch.argmax(outputs, dim=-1).cpu().tolist()
        return [classes[i] for i in indices]
    return [float(v) for v in outputs.detach().cpu().reshape(-1).tolist()]


def _placeholder(target_field: Any) -> Any:
    classes = getattr(target_field, "classes", None)
    return classes[0] if classes else 0.0


def _as_records(
    inputs: list[Any], features: list[str], target: str, placeholder: Any
) -> list[dict]:
    """Inputs as records, each carrying a stand-in target value.

    ``_encode`` builds both the feature matrix and the target tensor in one pass,
    so a record needs the target key present even at inference time, where the
    answer is what we are asking for. The stand-in is discarded.

    Copied, never mutated in place: the dicts belong to the caller, and writing a
    fabricated label into them would leave a value that looks like ground truth
    in whatever they do with the rows next.
    """
    records = []
    for value in inputs:
        record = dict(value) if isinstance(value, dict) else {features[0]: value}
        record.setdefault(target, placeholder)
        records.append(record)
    return records
