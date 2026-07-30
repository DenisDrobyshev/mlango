"""Inference middleware.

The same idea as Django's middleware stack, applied to prediction requests: a
list of small classes in settings, each wrapping the next, configured
outermost-first. Auth, rate limiting, guardrails and logging belong here rather
than inside a model.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("mlango.serve")


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Log method, path, status and duration for every request."""

    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "%s %s -> %s (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        response.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
        return response


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Require a shared key on inference routes.

    Reads ``settings.SERVE_API_KEYS``. The admin is left alone so a local
    dashboard keeps working; put a real identity provider in front of anything
    public.
    """

    def __init__(self, app: Any, header: str = "X-API-Key"):
        super().__init__(app)
        self.header = header

    async def dispatch(self, request: Request, call_next):
        from mlango.conf import settings

        keys = set(getattr(settings, "SERVE_API_KEYS", []) or [])
        if not keys or not request.url.path.startswith("/api"):
            return await call_next(request)

        provided = request.headers.get(self.header, "")
        if provided not in keys:
            return JSONResponse(
                {"detail": f"A valid {self.header} header is required."}, status_code=401
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """A simple fixed-window limit per client address.

    In-process and therefore per-worker: enough to stop a runaway script,
    not a substitute for a gateway in front of a real deployment.
    """

    def __init__(self, app: Any, limit: int = 120, window_s: int = 60):
        super().__init__(app)
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith("/api"):
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        hits = self._hits[client]
        while hits and now - hits[0] > self.window_s:
            hits.popleft()

        if len(hits) >= self.limit:
            retry_after = int(self.window_s - (now - hits[0])) + 1
            return JSONResponse(
                {"detail": "Rate limit exceeded."},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

        hits.append(now)
        response: Response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.limit - len(hits)))
        return response


class GuardrailMiddleware(BaseHTTPMiddleware):
    """Reject request bodies containing blocked terms.

    A coarse first line of defence for agent endpoints. Configure with
    ``settings.SERVE_BLOCKED_TERMS``.
    """

    async def dispatch(self, request: Request, call_next):
        from mlango.conf import settings

        blocked = [t.casefold() for t in getattr(settings, "SERVE_BLOCKED_TERMS", []) or []]
        if blocked and request.method == "POST" and request.url.path.startswith("/api"):
            body = await request.body()
            haystack = body.decode("utf-8", "ignore").casefold()
            hit = next((term for term in blocked if term in haystack), None)
            if hit:
                logger.warning("Blocked request to %s containing %r", request.url.path, hit)
                return JSONResponse(
                    {"detail": "The request was rejected by a content guardrail."},
                    status_code=400,
                )
        return await call_next(request)


def build_middleware() -> list[Any]:
    """Resolve ``settings.SERVE_MIDDLEWARE`` into classes, outermost first.

    Typed loosely on purpose: the stack is dotted paths in settings, so what
    comes back is whatever the project named. ``check`` is what reports a class
    that is not usable as middleware, with the setting's name attached.
    """
    from mlango.conf import settings
    from mlango.core.module_loading import import_string

    return [import_string(dotted) for dotted in settings.SERVE_MIDDLEWARE]


__all__ = [
    "RequestLogMiddleware",
    "ApiKeyMiddleware",
    "RateLimitMiddleware",
    "GuardrailMiddleware",
    "build_middleware",
]
