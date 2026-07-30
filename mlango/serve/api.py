"""Building the ASGI application.

``manage.py runserver`` serves one app that carries both the admin and the
inference API, the same way ``django-admin runserver`` serves the admin
alongside a project's own views.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.requests import Request

from mlango.core.exceptions import MlangoError, ValidationError
from mlango.serve.middleware import build_middleware
from mlango.serve.routing import Route, load_routes

logger = logging.getLogger("mlango.serve")


def create_app(*, include_admin: bool | None = None, routes: list[Route] | None = None) -> FastAPI:
    """Build the project's ASGI app."""
    import mlango
    from mlango.conf import settings

    mlango.setup()

    app = FastAPI(
        title=settings.ADMIN_SITE_TITLE,
        version=mlango.get_version(),
        description="Inference API and administration for this mlango project.",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    for middleware_class in reversed(build_middleware()):
        app.add_middleware(middleware_class)

    _install_error_handlers(app)
    _install_health(app)

    for route in routes if routes is not None else load_routes():
        _mount_route(app, route)

    if settings.ADMIN_ENABLED if include_admin is None else include_admin:
        from mlango.admin.app import build_admin_app

        app.mount(settings.ADMIN_URL, build_admin_app(), name="admin")
        logger.info("Admin mounted at %s", settings.ADMIN_URL)

        @app.get("/", include_in_schema=False)
        def _root() -> RedirectResponse:
            return RedirectResponse(settings.ADMIN_URL + "/")

    return app


def _mount_route(app: FastAPI, route: Route) -> None:
    endpoint = route.endpoint
    app.add_api_route(
        f"/api{route.path}",
        endpoint.handler,
        methods=list(endpoint.methods),
        name=route.name,
        summary=endpoint.summary,
        description=endpoint.description,
        tags=[endpoint.kind],
    )
    logger.info("Route /api%s -> %s", route.path, endpoint.label)


def _install_health(app: FastAPI) -> None:
    @app.get("/api/health", tags=["meta"], summary="Liveness and registry snapshot")
    def health() -> dict[str, Any]:
        import mlango
        from mlango.core.registry import apps
        from mlango.metastore.session import metastore_ready

        return {
            "status": "ok",
            "version": mlango.get_version(),
            "metastore": metastore_ready(),
            **apps.summary(),
        }


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ValidationError)
    async def _validation(request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse({"detail": exc.errors}, status_code=422)

    @app.exception_handler(LookupError)
    async def _lookup(request: Request, exc: LookupError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(MlangoError)
    async def _mlango(request: Request, exc: MlangoError) -> JSONResponse:
        logger.exception("Request to %s failed", request.url.path)
        return JSONResponse({"detail": str(exc)}, status_code=400)


def run(host: str | None = None, port: int | None = None, *, reload: bool = False) -> None:
    """Start uvicorn. Used by ``manage.py runserver``."""
    import uvicorn

    from mlango.conf import settings

    host = host or settings.SERVE_HOST
    port = port or settings.SERVE_PORT

    if reload:
        # Reload needs an import string so the worker can rebuild the app.
        uvicorn.run(
            "mlango.serve.api:create_app",
            factory=True,
            host=host,
            port=int(port),
            reload=True,
            log_level=settings.LOG_LEVEL.lower(),
        )
        return

    uvicorn.run(create_app(), host=host, port=int(port), log_level=settings.LOG_LEVEL.lower())


__all__ = ["create_app", "run"]
