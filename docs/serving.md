# Serving

`manage.py runserver` serves the admin and a documented inference API together —
the same way `django-admin runserver` serves the admin alongside your views.

## Routes

A project's `routes.py` is its `urls.py`:

```python title="myproject/routes.py"
from mlango.serve import path

from reviews.models import Sentiment
from support.agents import Support

urlpatterns = [
    path("predict/", Sentiment.as_endpoint(stage="production"), name="sentiment"),
    path("chat/", Support.as_endpoint(), name="support"),
]
```

```python title="myproject/settings.py"
ROOT_ROUTECONF = "myproject.routes"
```

Routes are mounted under `/api`, so `path("predict/")` serves
`POST /api/predict/`.

Split them across apps with `include`:

```python
from mlango.serve import include, path

urlpatterns = [
    *include("reviews.routes"),
    *include("support.routes"),
]
```

## Model endpoints

```python
Sentiment.as_endpoint()                     # latest registered version
Sentiment.as_endpoint(version=3)            # pinned
Sentiment.as_endpoint(stage="production")   # whatever is promoted
```

The version is loaded once, lazily, on the first request — so starting the server
does not require a trained model, and a fresh promotion is picked up by a
restart.

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H 'Content-Type: application/json' \
  -d '{"input": "great movie"}'

curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H 'Content-Type: application/json' \
  -d '{"inputs": ["great movie", "awful film"], "proba": true}'
```

```json
{
  "model": "reviews.Sentiment",
  "version": 2,
  "predictions": ["positive", "negative"],
  "probabilities": [
    {"negative": 0.04, "positive": 0.96},
    {"negative": 0.95, "positive": 0.05}
  ]
}
```

## Agent endpoints

```bash
curl -X POST http://127.0.0.1:8000/api/chat/ \
  -H 'Content-Type: application/json' \
  -d '{"message": "How do I rotate an API key?", "session_id": "user-42"}'
```

```json
{
  "agent": "support.Support",
  "output": "Rotate it in Settings → API keys…",
  "steps": 2,
  "trace": "a1b2c3d4…",
  "tools_used": ["search_docs"],
  "usage": {"input_tokens": 1840, "output_tokens": 96, "total_tokens": 1936}
}
```

`session_id` is what gives the agent continuity across requests, via its memory
backend.

## Documented for free

Request and response shapes are pydantic models, so `/api/docs` describes every
endpoint without anyone writing OpenAPI:

- Swagger UI — `/api/docs`
- ReDoc — `/api/redoc`
- Schema — `/api/openapi.json`

## Health

```bash
curl http://127.0.0.1:8000/api/health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "metastore": true,
  "apps": ["reviews", "support"],
  "counts": {"dataset": 2, "model": 1, "agent": 1, "eval": 1}
}
```

Useful as a readiness probe: it confirms the app booted *and* that the registry
and metastore are reachable.

## Middleware

A stack configured in settings, outermost first:

```python
SERVE_MIDDLEWARE = [
    "mlango.serve.middleware.RequestLogMiddleware",
    "mlango.serve.middleware.RateLimitMiddleware",
    "mlango.serve.middleware.ApiKeyMiddleware",
    "mlango.serve.middleware.GuardrailMiddleware",
]
```

| Middleware | Does |
|---|---|
| `RequestLogMiddleware` | Logs method, path, status and duration; adds `X-Response-Time-Ms` |
| `ApiKeyMiddleware` | Requires an `X-API-Key` from `SERVE_API_KEYS` on `/api` routes |
| `RateLimitMiddleware` | Fixed-window limit per client address |
| `GuardrailMiddleware` | Rejects bodies containing `SERVE_BLOCKED_TERMS` |

Write your own as ordinary Starlette middleware:

```python
from starlette.middleware.base import BaseHTTPMiddleware


class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request.state.tenant = request.headers.get("X-Tenant", "default")
        return await call_next(request)
```

!!! warning "In-process limits"
    `RateLimitMiddleware` counts per worker. It stops a runaway script; it is not
    a substitute for a gateway.

## Errors

| Raised | Status | Body |
|---|---|---|
| `ValidationError` | 422 | Per-field messages |
| `LookupError` | 404 | The message, e.g. "has no registered version" |
| Any `MlangoError` | 400 | The message |

So requesting a model that was never trained returns a 404 explaining exactly
that, rather than a 500.

## Deployment

`runserver` is for development. In production, point an ASGI server at the app
factory:

```bash
uvicorn "mlango.serve.api:create_app" --factory --host 0.0.0.0 --port 8000 --workers 4
```

```bash
gunicorn "mlango.serve.api:create_app()" -k uvicorn.workers.UvicornWorker -w 4
```

Set `MLANGO_SETTINGS_MODULE` in the environment, and before you go public:

- `DEBUG = False`
- `SECRET_KEY` from your secret store
- `ADMIN_PASSWORD`, or the admin behind your identity provider
- `SERVE_API_KEYS`, or auth terminated at the gateway
- `METASTORE` pointing at Postgres if more than one worker writes runs
- `STORAGE` pointing at shared storage if workers must see each other's artifacts

See [SECURITY.md](https://github.com/DenisDrobyshev/mlango/blob/master/SECURITY.md)
for the full list of development defaults you must change.
