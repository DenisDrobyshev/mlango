"""URL routing for the inference API.

A project's ``routes.py`` looks like Django's ``urls.py``::

    from mlango.serve import path

    urlpatterns = [
        path("predict/", Sentiment.as_endpoint()),
        path("chat/", SupportAgent.as_endpoint(), name="support-chat"),
    ]

The endpoint objects know their own input and output shapes, so FastAPI's
generated OpenAPI docs describe the model's fields without anyone writing a
schema by hand.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mlango.core.exceptions import ImproperlyConfigured
from mlango.core.module_loading import import_string


@dataclass
class Endpoint:
    """One servable operation."""

    kind: str
    label: str
    handler: Callable[..., Any]
    summary: str = ""
    description: str = ""
    methods: tuple[str, ...] = ("POST",)
    #: Extra detail shown in the admin.
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<Endpoint {self.kind}:{self.label}>"


@dataclass
class Route:
    route: str
    endpoint: Endpoint
    name: str = ""

    @property
    def path(self) -> str:
        return "/" + self.route.lstrip("/")

    def __repr__(self) -> str:
        return f"<Route {self.path} -> {self.endpoint.label}>"


def path(route: str, endpoint: Endpoint, *, name: str = "") -> Route:
    """Bind a route to an endpoint."""
    if not isinstance(endpoint, Endpoint):
        raise ImproperlyConfigured(
            f"path({route!r}, ...) expects an Endpoint — use Model.as_endpoint() or "
            f"Agent.as_endpoint(), got {type(endpoint).__name__}."
        )
    return Route(route=route, endpoint=endpoint, name=name or endpoint.label)


def include(module_path: str) -> list[Route]:
    """Pull ``urlpatterns`` in from another module."""
    module = import_string(f"{module_path}.urlpatterns")
    return list(module)


def load_routes() -> list[Route]:
    """Resolve ``settings.ROOT_ROUTECONF`` into a list of routes."""
    from mlango.conf import settings

    conf = settings.ROOT_ROUTECONF
    if not conf:
        return []
    try:
        patterns = import_string(f"{conf}.urlpatterns")
    except ImproperlyConfigured as exc:
        raise ImproperlyConfigured(
            f"ROOT_ROUTECONF is {conf!r} but it has no `urlpatterns` list: {exc}"
        ) from exc

    routes: list[Route] = []
    for entry in patterns:
        if isinstance(entry, Route):
            routes.append(entry)
        elif isinstance(entry, list):
            routes.extend(entry)
        else:
            raise ImproperlyConfigured(
                f"{conf}.urlpatterns contains {entry!r}; expected path(...) results."
            )
    return routes


__all__ = ["path", "include", "load_routes", "Route", "Endpoint"]
