"""Coercing arbitrary Python values into something a JSON column accepts.

Lives in ``core`` because three unrelated layers need it — runs, traces and
eval results all write JSON columns. Previously each imported a private helper
from ``training.run``, which quietly made the agents layer depend on the
training layer for no reason.
"""

from __future__ import annotations

from typing import Any


def jsonable(payload: Any) -> Any:
    """Return a JSON-serialisable view of ``payload``.

    Never raises: anything the encoder cannot represent falls back to ``repr``.
    Metadata is not worth failing a run over.
    """
    if isinstance(payload, dict):
        return {str(k): jsonable(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in payload]
    if isinstance(payload, (str, int, float, bool, type(None))):
        return payload

    # numpy scalars, arrays, datetimes and anything else with a standard
    # conversion hook.
    for attr in ("tolist", "item", "isoformat"):
        method = getattr(payload, attr, None)
        if callable(method):
            try:
                return jsonable(method())
            except (TypeError, ValueError):
                break
    return repr(payload)


__all__ = ["jsonable"]
