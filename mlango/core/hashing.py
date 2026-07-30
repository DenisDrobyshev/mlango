"""Stable fingerprints.

Reproducibility is the point of the metastore, so every fingerprint here has to
be deterministic across processes and Python versions: canonical JSON with
sorted keys, then SHA-256. Anything the encoder cannot represent falls back to
``repr`` rather than raising, because a slightly coarse fingerprint beats a
crashed run.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


class _CanonicalEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        for attr in ("tolist", "isoformat"):
            method = getattr(o, attr, None)
            if callable(method):
                return method()
        if isinstance(o, (set, frozenset)):
            return sorted(map(str, o))
        if isinstance(o, bytes):
            return o.decode("utf-8", "replace")
        if callable(o):
            return f"{getattr(o, '__module__', '?')}.{getattr(o, '__qualname__', repr(o))}"
        return repr(o)


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, cls=_CanonicalEncoder
    )


def fingerprint(payload: Any, *, length: int = 12) -> str:
    """A short, stable hex digest of any JSON-ish payload."""
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[:length]


def full_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_digest(path: str, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's contents, streamed so large checkpoints are fine."""
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            sha.update(chunk)
    return sha.hexdigest()
