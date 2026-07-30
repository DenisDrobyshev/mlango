"""Turning declared models and agents into HTTP endpoints.

The request and response shapes are pydantic models, so FastAPI's ``/api/docs``
describes each endpoint without anyone writing OpenAPI by hand — the schema is
derived from the same declaration that trains the model.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from pydantic import BaseModel, Field

from mlango.core.exceptions import RunError
from mlango.serve.routing import Endpoint

logger = logging.getLogger("mlango.serve")


class PredictRequest(BaseModel):
    """One input, or a batch of them."""

    input: Any | None = Field(default=None, description="A single input record or value.")
    inputs: list[Any] | None = Field(default=None, description="A batch of input records.")
    proba: bool = Field(default=False, description="Also return class probabilities.")


class PredictResponse(BaseModel):
    model: str
    version: int | None = None
    predictions: list[Any]
    probabilities: list[Any] | None = None


class ChatRequest(BaseModel):
    message: str = Field(description="The user's message.")
    session_id: str = Field(default="", description="Conversation id for memory continuity.")


class ChatResponse(BaseModel):
    agent: str
    output: str
    steps: int
    trace: str = ""
    tools_used: list[str] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    error: str = ""


class _LazyModel:
    """Loads a registered model version once, on first request."""

    def __init__(self, model_class: type, version: int | None, stage: str | None):
        self.model_class = model_class
        self.version = version
        self.stage = stage
        self._instance: Any = None
        self._lock = threading.Lock()

    def get(self) -> Any:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self.model_class.load(version=self.version, stage=self.stage)
                    logger.info(
                        "Loaded %s for serving", getattr(self._instance._version, "ref", "?")
                    )
        return self._instance

    def reset(self) -> None:
        with self._lock:
            self._instance = None


def model_endpoint(
    model_class: type, *, version: int | None = None, stage: str | None = None
) -> Endpoint:
    """Build a prediction endpoint for a declared model."""
    loader = _LazyModel(model_class, version, stage)
    label = model_class._meta.label

    def handler(payload: PredictRequest) -> PredictResponse:
        if payload.inputs is None and payload.input is None:
            raise RunError("Send either `input` (one record) or `inputs` (a batch).")

        batch = payload.inputs if payload.inputs is not None else [payload.input]
        model = loader.get()
        predictions = model.predict(batch)
        probabilities = model.predict_proba(batch) if payload.proba else None

        return PredictResponse(
            model=label,
            version=getattr(model._version, "version", None),
            predictions=list(predictions),
            probabilities=list(probabilities) if probabilities is not None else None,
        )

    handler.__name__ = f"predict_{label.replace('.', '_')}"
    return Endpoint(
        kind="model",
        label=label,
        handler=handler,
        summary=f"Predict with {label}",
        description=model_class._meta.description or f"Run inference with {label}.",
        meta={
            "task": model_class.get_task(),
            "version": version,
            "stage": stage,
            "features": _safe(model_class.get_features),
        },
    )


def agent_endpoint(agent_class: type, **agent_kwargs: Any) -> Endpoint:
    """Build a chat endpoint for a declared agent."""
    label = agent_class._meta.label

    def handler(payload: ChatRequest) -> ChatResponse:
        agent = agent_class(**agent_kwargs)
        result = agent.run(payload.message, session_id=payload.session_id)
        return ChatResponse(
            agent=label,
            output=result.output,
            steps=result.steps,
            trace=result.trace_uuid,
            tools_used=result.tools_used,
            usage=result.usage.describe(),
            error=result.error,
        )

    handler.__name__ = f"chat_{label.replace('.', '_')}"
    instance = agent_class()
    return Endpoint(
        kind="agent",
        label=label,
        handler=handler,
        summary=f"Chat with {label}",
        description=agent_class._meta.description or f"Send a message to {label}.",
        meta={
            "model": agent_class.get_model(),
            "tools": instance.get_tools().names(),
            "max_steps": agent_class.get_max_steps(),
        },
    )


def _safe(fn: Any) -> Any:
    """Call an introspection helper, tolerating an incomplete declaration."""
    try:
        return fn()
    except Exception:
        return None


__all__ = [
    "model_endpoint",
    "agent_endpoint",
    "PredictRequest",
    "PredictResponse",
    "ChatRequest",
    "ChatResponse",
]
