"""Hugging Face transformers backend — fine-tuning a pretrained model.

The loop here is mlango's own, not ``transformers.Trainer``. That is deliberate:
using theirs would mean callbacks, early stopping, metric recording and run
tracking behave differently depending on which backend a project picked, and
"the framework owns the run" would stop being true. What we borrow is the part
worth borrowing — tokenisation, pretrained weights, and the model heads.

    class Sentiment(TextClassifier):
        class Meta:
            dataset = Reviews
            features = ["text"]

That is the whole declaration; see :mod:`mlango.training.presets`.
"""

from __future__ import annotations

import logging
from typing import Any

from mlango.core.exceptions import RunError
from mlango.core.signals import epoch_finished, epoch_started
from mlango.training import metrics as metric_lib
from mlango.training.trainer import Trainer

logger = logging.getLogger("mlango.training.transformers")

#: Hyperparameter field names the loop reads off the model when present.
BASE_MODEL = "base_model"
EPOCHS = "epochs"
BATCH_SIZE = "batch_size"
LEARNING_RATE = "learning_rate"
MAX_LENGTH = "max_length"
WARMUP_RATIO = "warmup_ratio"
WEIGHT_DECAY = "weight_decay"
GRADIENT_ACCUMULATION = "gradient_accumulation"
FREEZE_BASE = "freeze_base"


class TransformersTrainer(Trainer):
    """Fine-tunes a pretrained transformer for classification or regression."""

    name = "transformers"
    requires = ("transformers", "torch")
    extension = "dir"

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _param(model: Any, name: str, default: Any) -> Any:
        return getattr(model, name, default) if model._meta.has_field(name) else default

    def _tokenizer(self, model: Any) -> Any:
        """Cache the tokenizer on the model instance — loading is not free."""
        cached = getattr(model, "_tokenizer", None)
        if cached is not None:
            return cached

        from transformers import AutoTokenizer

        base = self._param(model, BASE_MODEL, None)
        if not base:
            raise RunError(
                f"{type(model)._meta.label} needs a `base_model` field naming a pretrained "
                f"checkpoint, e.g. base_model = CharField(default='distilbert-base-uncased')."
            )
        tokenizer = AutoTokenizer.from_pretrained(str(base))
        model._tokenizer = tokenizer
        return tokenizer

    def _text_of(self, record: Any, features: list[str]) -> str | tuple[str, str]:
        """One or two text fields; two become a sentence pair."""
        if len(features) == 1:
            return str(record.get(features[0]) or "")
        if len(features) == 2:
            return str(record.get(features[0]) or ""), str(record.get(features[1]) or "")
        raise RunError(
            f"The transformers trainer takes one text field, or two for a sentence pair; "
            f"got {features}. Narrow Meta.features, or override encode_batch."
        )

    def _encode(self, model: Any, records: list[Any], target: str, features: list[str]) -> Any:
        """Tokenise a batch and attach labels."""
        import torch

        custom = getattr(model, "encode_batch", None)
        if callable(custom):
            return custom(records, target)

        tokenizer = self._tokenizer(model)
        max_length = int(self._param(model, MAX_LENGTH, 256))

        texts = [self._text_of(r, features) for r in records]
        if features and len(features) == 2:
            first = [t[0] for t in texts]
            second = [t[1] for t in texts]
            batch = tokenizer(
                first,
                second,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
        else:
            batch = tokenizer(
                list(texts),
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )

        dataset_class = model.get_dataset()
        target_field = dataset_class._meta.get_field(target)
        classes = getattr(target_field, "classes", None)
        raw = [r.get(target) for r in records]

        if classes:
            unknown = sorted({v for v in raw if v not in classes})
            if unknown:
                raise RunError(
                    f"{target!r} contains label(s) {unknown} that {target_field.name} does not "
                    f"declare. Declared classes: {classes}."
                )
            batch["labels"] = torch.tensor([classes.index(v) for v in raw], dtype=torch.long)
        else:
            batch["labels"] = torch.tensor([float(v) for v in raw], dtype=torch.float32)
        return batch

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
                f"{type(model)._meta.label}.build() returned {type(module).__name__}; the "
                f"transformers trainer expects a torch.nn.Module, usually from "
                f"AutoModelForSequenceClassification.from_pretrained(...)."
            )

        device = torch.device(kwargs.get("device") or self.resolve_device())
        module.to(device)

        dataset_class = model.get_dataset()
        target = target or model.get_target(dataset_class)
        features = features or model.get_features(dataset_class)

        epochs = int(kwargs.get(EPOCHS) or self._param(model, EPOCHS, 3))
        batch_size = int(kwargs.get(BATCH_SIZE) or self._param(model, BATCH_SIZE, 16))
        accumulation = max(1, int(self._param(model, GRADIENT_ACCUMULATION, 1)))

        if self._param(model, FREEZE_BASE, False):
            _freeze_base(module)

        optimizer, scheduler = self._optimizer_and_schedule(
            model, module, epochs=epochs, batch_size=batch_size, train=train
        )

        train_records = list(train)
        if not train_records:
            raise RunError("The training split is empty; check the dataset source and filters.")
        validation_records = list(validation) if validation is not None else []

        run.update(device=str(device))
        logger.info(
            "Fine-tuning %s on %s for %s epoch(s), batch %s, device %s",
            self._param(model, BASE_MODEL, "?"),
            len(train_records),
            epochs,
            batch_size,
            device,
        )

        step = 0
        for epoch in range(1, epochs + 1):
            epoch_started.send(sender=type(model), run=run, epoch=epoch)
            callbacks.emit("on_epoch_begin", run, epoch, model=model)

            module.train()
            total_loss, seen = 0.0, 0
            optimizer.zero_grad(set_to_none=True)

            for index, start in enumerate(range(0, len(train_records), batch_size)):
                chunk = train_records[start : start + batch_size]
                batch = self._encode(model, chunk, target, features)
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = module(**batch)
                loss = outputs.loss / accumulation
                loss.backward()

                if (index + 1) % accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                step += 1
                total_loss += float(outputs.loss.item()) * len(chunk)
                seen += len(chunk)

                if step % 20 == 0:
                    run.log_metric(
                        "batch_loss", float(outputs.loss.item()), step=step, split="train"
                    )
                    callbacks.emit("on_batch_end", run, step, {"loss": float(outputs.loss.item())})

            epoch_metrics = {"loss": total_loss / max(seen, 1)}
            if scheduler is not None:
                epoch_metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
            if validation_records:
                epoch_metrics.update(
                    self._validate(model, module, validation_records, target, features, device)
                )

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

    def _optimizer_and_schedule(
        self, model: Any, module: Any, *, epochs: int, batch_size: int, train: Any
    ) -> tuple[Any, Any]:
        import torch

        custom = getattr(model, "configure_optimizer", None)
        if callable(custom):
            return custom(module), None

        lr = float(self._param(model, LEARNING_RATE, 2e-5))
        decay = float(self._param(model, WEIGHT_DECAY, 0.01))

        # No weight decay on biases and layer norms — the standard recipe for
        # transformer fine-tuning, and easy to get wrong by omission.
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        groups = [
            {
                "params": [
                    p
                    for n, p in module.named_parameters()
                    if p.requires_grad and not any(k in n for k in no_decay)
                ],
                "weight_decay": decay,
            },
            {
                "params": [
                    p
                    for n, p in module.named_parameters()
                    if p.requires_grad and any(k in n for k in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(groups, lr=lr)

        warmup_ratio = float(self._param(model, WARMUP_RATIO, 0.06))
        try:
            from transformers import get_linear_schedule_with_warmup

            rows = train.count()
            total_steps = max(1, (rows // max(1, batch_size)) * epochs)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(total_steps * warmup_ratio),
                num_training_steps=total_steps,
            )
        except Exception:  # pragma: no cover - scheduler is a nicety, not a requirement
            logger.debug("Could not build a warmup schedule; training without one.", exc_info=True)
            scheduler = None
        return optimizer, scheduler

    def _validate(self, model, module, records, target, features, device) -> dict[str, float]:
        import torch

        module.eval()
        total, seen = 0.0, 0
        predictions: list[Any] = []
        truth: list[Any] = []

        with torch.no_grad():
            for start in range(0, len(records), 64):
                chunk = records[start : start + 64]
                batch = self._encode(model, chunk, target, features)
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = module(**batch)
                total += float(outputs.loss.item()) * len(chunk)
                seen += len(chunk)
                predictions.extend(_decode(model, outputs.logits, target))
                truth.extend(r.get(target) for r in chunk)

        report = metric_lib.report_for_task(model.get_task(), truth, predictions)
        flat = metric_lib.flatten_report(report)
        out = {"val_loss": total / max(seen, 1)}
        out.update({f"val_{k}": v for k, v in flat.items()})
        out.update({k: v for k, v in flat.items() if k in {"accuracy", "f1_macro", "r2"}})
        return out

    # -- inference -----------------------------------------------------------

    def predict(self, model, fitted, inputs: list[Any]) -> list[Any]:
        import torch

        dataset_class = model.get_dataset()
        target = model.get_target(dataset_class)
        features = model.get_features(dataset_class)
        records = _as_records(inputs, features, target, dataset_class)

        device = next(fitted.parameters()).device
        fitted.eval()
        out: list[Any] = []
        with torch.no_grad():
            for start in range(0, len(records), 64):
                chunk = records[start : start + 64]
                batch = self._encode(model, chunk, target, features)
                batch.pop("labels", None)
                batch = {k: v.to(device) for k, v in batch.items()}
                out.extend(_decode(model, fitted(**batch).logits, target))
        return out

    def predict_proba(self, model, fitted, inputs: list[Any]):
        import torch

        dataset_class = model.get_dataset()
        target = model.get_target(dataset_class)
        classes = getattr(dataset_class._meta.get_field(target), "classes", None)
        if not classes:
            return None

        features = model.get_features(dataset_class)
        records = _as_records(inputs, features, target, dataset_class)

        device = next(fitted.parameters()).device
        fitted.eval()
        out: list[Any] = []
        with torch.no_grad():
            for start in range(0, len(records), 64):
                chunk = records[start : start + 64]
                batch = self._encode(model, chunk, target, features)
                batch.pop("labels", None)
                batch = {k: v.to(device) for k, v in batch.items()}
                probabilities = torch.softmax(fitted(**batch).logits, dim=-1).cpu().numpy()
                out.extend(
                    dict(zip(classes, (float(p) for p in row), strict=True))
                    for row in probabilities
                )
        return out

    # -- persistence ---------------------------------------------------------

    def save(self, model, fitted, name: str) -> str:
        """Save in the Hugging Face layout, so the artifact is portable."""
        from mlango.storage import default_storage

        with default_storage().writable(f"{name}/model", directory=True) as target:
            fitted.save_pretrained(target.path)
            self._tokenizer(model).save_pretrained(target.path)
            return target.name

    def load(self, model, path: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from mlango.storage import default_storage

        with default_storage().readable(path) as local:
            module = AutoModelForSequenceClassification.from_pretrained(local)
            model._tokenizer = AutoTokenizer.from_pretrained(local)
        module.eval()
        return module

    # -- introspection -------------------------------------------------------

    def describe(self, model, fitted) -> dict[str, Any]:
        total = sum(p.numel() for p in fitted.parameters())
        trainable = sum(p.numel() for p in fitted.parameters() if p.requires_grad)
        return {
            "backend": self.name,
            "base_model": self._param(model, BASE_MODEL, ""),
            "architecture": type(fitted).__name__,
            "parameters": int(total),
            "trainable_parameters": int(trainable),
            "device": str(next(fitted.parameters()).device),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _freeze_base(module: Any) -> None:
    """Train only the classification head.

    Much faster and often enough for a small dataset, where full fine-tuning
    overfits.
    """
    base = getattr(module, "base_model", None)
    if base is None:
        logger.warning("freeze_base is set but the module exposes no base_model; ignoring.")
        return
    for parameter in base.parameters():
        parameter.requires_grad = False


def _decode(model: Any, logits: Any, target: str) -> list[Any]:
    import torch

    target_field = model.get_dataset()._meta.get_field(target)
    classes = getattr(target_field, "classes", None)
    if classes:
        return [classes[i] for i in torch.argmax(logits, dim=-1).cpu().tolist()]
    return [float(v) for v in logits.detach().cpu().reshape(-1).tolist()]


def _as_records(
    inputs: list[Any], features: list[str], target: str, dataset_class: Any
) -> list[Any]:
    """Accept raw strings or dicts, and add a placeholder label for tokenisation."""
    from mlango.data.query import Record

    target_field = dataset_class._meta.get_field(target)
    classes = getattr(target_field, "classes", None)
    placeholder = classes[0] if classes else 0.0

    records = []
    for value in inputs:
        record = Record(value) if isinstance(value, dict) else Record({features[0]: value})
        record.setdefault(target, placeholder)
        records.append(record)
    return records
