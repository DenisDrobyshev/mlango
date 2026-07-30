"""Ready-made model bases.

Django ships generic views so that the ninetieth CRUD page is three lines rather
than thirty. These are the same idea for the shapes that recur in ML work: the
hyperparameters, the `build()` and the sensible defaults are already written, and
you supply only what is actually specific to your problem.

    class Sentiment(TextClassifier):
        \"\"\"Sentiment over product reviews.\"\"\"

        class Meta:
            dataset = Reviews
            features = ["text"]

Every field remains overridable, because a preset that cannot be adjusted is a
cage rather than a convenience.
"""

from __future__ import annotations

from typing import Any

from mlango.core import fields
from mlango.core.exceptions import ImproperlyConfigured
from mlango.training.model import Model


class TransformerModel(Model):
    """Shared configuration for fine-tuning a pretrained transformer."""

    #: Any checkpoint on the Hugging Face hub, or a local directory.
    base_model = fields.CharField(default="distilbert-base-uncased")
    learning_rate = fields.FloatField(default=2e-5, min_value=0.0, tunable=True)
    epochs = fields.IntegerField(default=3, min_value=1, tunable=True)
    batch_size = fields.IntegerField(default=16, min_value=1, tunable=True)
    max_length = fields.IntegerField(default=256, min_value=8)
    weight_decay = fields.FloatField(default=0.01, min_value=0.0)
    warmup_ratio = fields.FloatField(default=0.06, min_value=0.0, max_value=1.0)
    gradient_accumulation = fields.IntegerField(default=1, min_value=1)
    #: Train only the head. Faster, and often better on a small dataset.
    freeze_base = fields.BooleanField(default=False)

    class Meta:
        abstract = True
        trainer = "transformers"

    # -- the part a preset exists to write for you ---------------------------

    def build(self) -> Any:
        # Validate the declaration before importing transformers: a
        # configuration mistake should say so, not surface as a missing
        # dependency the user may not even need to install to see the problem.
        dataset_class = self.get_dataset()
        target_field = dataset_class._meta.get_field(self.get_target(dataset_class))
        classes = getattr(target_field, "classes", None)

        if self.get_task() == "regression":
            num_labels, problem_type = 1, "regression"
        else:
            if not classes:
                raise ImproperlyConfigured(
                    f"{type(self)._meta.label}: the target field "
                    f"{target_field.name!r} must declare its classes, e.g. "
                    f"LabelField(['negative', 'positive']), so the classification head "
                    f"knows how many outputs to build."
                )
            num_labels, problem_type = len(classes), "single_label_classification"

        from transformers import AutoConfig, AutoModelForSequenceClassification

        config = AutoConfig.from_pretrained(
            str(self.base_model),
            num_labels=num_labels,
            problem_type=problem_type,
        )
        # Labels round-trip into the saved config, so an exported checkpoint is
        # self-describing outside mlango too.
        if classes and self.get_task() != "regression":
            config.id2label = dict(enumerate(classes))
            config.label2id = {label: index for index, label in enumerate(classes)}

        return AutoModelForSequenceClassification.from_pretrained(
            str(self.base_model), config=config
        )


class TextClassifier(TransformerModel):
    """Fine-tune a pretrained encoder to classify text.

    Declare the dataset and which field holds the text; everything else has a
    working default.
    """

    class Meta:
        abstract = True
        trainer = "transformers"
        task = "classification"
        monitor = "val_accuracy"
        monitor_mode = "max"


class TextRegressor(TransformerModel):
    """Fine-tune a pretrained encoder to predict a continuous value from text."""

    class Meta:
        abstract = True
        trainer = "transformers"
        task = "regression"
        monitor = "val_loss"
        monitor_mode = "min"


class TabularClassifier(Model):
    """A small feed-forward network over numeric columns.

    The honest default for tabular data is gradient boosting, not a neural net —
    reach for this when you want a torch model in the loop, or as a baseline.
    """

    hidden_size = fields.IntegerField(default=64, min_value=1, tunable=True)
    depth = fields.IntegerField(default=2, min_value=1, max_value=8, tunable=True)
    dropout = fields.FloatField(default=0.1, min_value=0.0, max_value=0.9, tunable=True)
    learning_rate = fields.FloatField(default=1e-3, min_value=0.0, tunable=True)
    epochs = fields.IntegerField(default=30, min_value=1)
    batch_size = fields.IntegerField(default=64, min_value=1)

    class Meta:
        abstract = True
        trainer = "torch"
        task = "classification"
        monitor = "val_accuracy"
        monitor_mode = "max"

    def build(self) -> Any:
        import torch.nn as nn

        dataset_class = self.get_dataset()
        features = self.get_features(dataset_class)
        target_field = dataset_class._meta.get_field(self.get_target(dataset_class))
        classes = getattr(target_field, "classes", None)

        if self.get_task() != "regression" and not classes:
            raise ImproperlyConfigured(
                f"{type(self)._meta.label}: the target field {target_field.name!r} must "
                f"declare its classes so the output layer knows how wide to be."
            )
        outputs = 1 if self.get_task() == "regression" else len(classes or [])

        layers: list[Any] = []
        width = len(features)
        for _ in range(int(self.depth)):
            layers += [nn.Linear(width, int(self.hidden_size)), nn.ReLU()]
            if self.dropout:
                layers.append(nn.Dropout(float(self.dropout)))
            width = int(self.hidden_size)
        layers.append(nn.Linear(width, outputs))
        return nn.Sequential(*layers)


class TabularRegressor(TabularClassifier):
    """The same network, predicting a continuous value."""

    class Meta:
        abstract = True
        trainer = "torch"
        task = "regression"
        monitor = "val_loss"
        monitor_mode = "min"


__all__ = [
    "TransformerModel",
    "TextClassifier",
    "TextRegressor",
    "TabularClassifier",
    "TabularRegressor",
]
