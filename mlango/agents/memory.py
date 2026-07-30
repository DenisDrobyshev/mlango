"""Conversation memory.

Memory is deliberately a separate, swappable object rather than something the
agent owns: the same agent should be able to run stateless in a batch job, keep
a short buffer in a web request, and read a full persisted history in a support
console — without its own code changing.
"""

from __future__ import annotations

import abc
import threading
from collections import defaultdict, deque
from typing import Any


class Memory(abc.ABC):
    """The contract every memory backend implements."""

    @abc.abstractmethod
    def load(self, session_id: str) -> list[dict[str, Any]]:
        """Prior messages for a session, oldest first."""

    @abc.abstractmethod
    def append(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Record new messages at the end of a session."""

    def clear(self, session_id: str) -> None:  # noqa: B027 - optional by design
        """Forget a session.

        Deliberately concrete and empty: a backend with nothing to forget (a
        stateless one, or one whose store is owned elsewhere) should not be
        forced to write a no-op override.
        """

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__}

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


class NullMemory(Memory):
    """No history at all — every run starts fresh. The default."""

    def load(self, session_id: str) -> list[dict[str, Any]]:
        return []

    def append(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        return None


class BufferMemory(Memory):
    """Keeps the last ``k`` messages per session, in process.

    Fast and simple, but per-process: a second worker will not see it. Use
    :class:`MetastoreMemory` when the history has to survive a restart.
    """

    def __init__(self, k: int = 20):
        self.k = k
        self._store: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=self.k))
        self._lock = threading.Lock()

    def load(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._store.get(session_id, ()))

    def append(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        with self._lock:
            buffer = self._store[session_id]
            for message in messages:
                buffer.append(message)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

    def describe(self) -> dict[str, Any]:
        return {"type": "BufferMemory", "k": self.k}


class MetastoreMemory(Memory):
    """Rebuilds history from the traces already recorded in the metastore.

    Nothing extra is stored: since every agent run is traced anyway, the
    conversation can be reconstructed from those rows. That keeps one source of
    truth and means the admin's trace view and the agent's memory can never
    disagree.
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns

    def load(self, session_id: str) -> list[dict[str, Any]]:
        from sqlalchemy import select

        from mlango.metastore.models import RunStatus, Trace
        from mlango.metastore.session import session_scope

        if not session_id:
            return []

        with session_scope() as session:
            rows = list(
                session.execute(
                    select(Trace)
                    .where(Trace.session_id == session_id, Trace.status == RunStatus.FINISHED)
                    .order_by(Trace.started_at.desc())
                    .limit(self.max_turns)
                ).scalars()
            )

        messages: list[dict[str, Any]] = []
        for trace in reversed(rows):
            if trace.input:
                messages.append({"role": "user", "content": trace.input})
            if trace.output:
                messages.append({"role": "assistant", "content": trace.output})
        return messages

    def append(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """No-op: the tracer already persists the turn."""

    def clear(self, session_id: str) -> None:
        from sqlalchemy import select

        from mlango.metastore.models import Trace
        from mlango.metastore.session import session_scope

        with session_scope() as session:
            for trace in session.execute(
                select(Trace).where(Trace.session_id == session_id)
            ).scalars():
                session.delete(trace)

    def describe(self) -> dict[str, Any]:
        return {"type": "MetastoreMemory", "max_turns": self.max_turns}


class WindowMemory(BufferMemory):
    """A buffer that always keeps the first message (usually the task brief).

    Truncating from the front is the usual way a long conversation loses the
    instruction that started it; this keeps the anchor and drops the middle.
    """

    def __init__(self, k: int = 20, keep_first: int = 1):
        super().__init__(k=k)
        self.keep_first = keep_first
        self._anchors: dict[str, list[dict[str, Any]]] = {}

    def load(self, session_id: str) -> list[dict[str, Any]]:
        anchors = self._anchors.get(session_id, [])
        recent = super().load(session_id)
        seen = {id(m) for m in anchors}
        return anchors + [m for m in recent if id(m) not in seen]

    def append(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        stored = self._anchors.setdefault(session_id, [])
        for message in messages:
            if len(stored) < self.keep_first:
                stored.append(message)
        super().append(session_id, messages)

    def clear(self, session_id: str) -> None:
        self._anchors.pop(session_id, None)
        super().clear(session_id)

    def describe(self) -> dict[str, Any]:
        return {"type": "WindowMemory", "k": self.k, "keep_first": self.keep_first}


__all__ = ["Memory", "NullMemory", "BufferMemory", "WindowMemory", "MetastoreMemory"]
