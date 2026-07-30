"""Type aliases for the four declarative families.

Every generic subsystem in mlango is handed a *declared class* rather than an
instance — ``model_endpoint(Sentiment)``, ``run_sweep(Sentiment, ...)``,
``DataQuerySet(Reviews)``. Writing that as a bare ``type`` throws away the only
information that matters, so a reader cannot tell which family is expected and a
type checker cannot see ``_meta`` at all.

These aliases name the contract once. They resolve to the real classes only while
type checking, which keeps them free of the import cycles that would follow from
``data`` importing ``training`` importing ``serve`` at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlango.agents.agent import Agent
    from mlango.core.base import Declarative
    from mlango.data.dataset import Dataset
    from mlango.evals.base import Eval
    from mlango.training.model import Model

    #: Any class built by the declarative metaclass — it has ``_meta``.
    DeclarativeClass = type[Declarative]
    DatasetClass = type[Dataset]
    ModelClass = type[Model]
    AgentClass = type[Agent]
    EvalClass = type[Eval]
else:  # pragma: no cover - aliases collapse to `type` at runtime
    DeclarativeClass = type
    DatasetClass = type
    ModelClass = type
    AgentClass = type
    EvalClass = type


__all__ = [
    "DeclarativeClass",
    "DatasetClass",
    "ModelClass",
    "AgentClass",
    "EvalClass",
]
