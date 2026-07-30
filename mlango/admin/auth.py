"""Admin authentication.

Django ships a full auth system because a web app needs users anyway. An ML
project usually does not, so mlango takes the smaller honest option: HTTP Basic
auth guarding the admin, off by default for local work, and a ``check`` warning
if you leave it off with ``DEBUG = False``.

For anything real, put the admin behind your existing identity provider — an
SSO proxy knows about your org, and this module never will.
"""

from __future__ import annotations

import base64
import hmac
import logging
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

logger = logging.getLogger("mlango.admin")

REALM = "mlango admin"


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """Require a username and password when ``ADMIN_PASSWORD`` is set."""

    async def dispatch(self, request: Request, call_next):
        from mlango.conf import settings

        password = getattr(settings, "ADMIN_PASSWORD", "") or ""
        if not password:
            return await call_next(request)

        username = getattr(settings, "ADMIN_USERNAME", "admin") or "admin"
        header = request.headers.get("Authorization", "")

        if not _matches(header, username, password):
            logger.warning(
                "Rejected an unauthenticated admin request from %s",
                request.client.host if request.client else "unknown",
            )
            return _challenge()

        return await call_next(request)


def _challenge() -> Response:
    return PlainTextResponse(
        "Authentication required.",
        status_code=401,
        headers={"WWW-Authenticate": f'Basic realm="{REALM}", charset="UTF-8"'},
    )


def _matches(header: str, username: str, password: str) -> bool:
    scheme, _, encoded = header.partition(" ")
    if scheme.lower() != "basic" or not encoded:
        return False
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False

    supplied_user, _, supplied_password = decoded.partition(":")
    # compare_digest on both halves so the response time does not reveal which
    # of the two was wrong.
    user_ok = hmac.compare_digest(supplied_user.encode(), username.encode())
    password_ok = hmac.compare_digest(supplied_password.encode(), password.encode())
    return user_ok and password_ok


def auth_configured() -> bool:
    from mlango.conf import settings

    return bool(getattr(settings, "ADMIN_PASSWORD", ""))


def describe() -> dict[str, Any]:
    from mlango.conf import settings

    return {
        "enabled": auth_configured(),
        "username": getattr(settings, "ADMIN_USERNAME", "admin"),
    }


__all__ = ["BasicAuthMiddleware", "auth_configured", "describe", "REALM"]
