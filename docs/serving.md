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
  "version": "0.2.0",
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

`runserver` is for development: one process, autoreload, no worker management.

`startproject` writes an `asgi.py`, the same way a Django project has one. It is
what a production server points at:

```bash
uvicorn myproject.asgi:application --host 0.0.0.0 --port 8000 --workers 4
gunicorn myproject.asgi:application -k uvicorn.workers.UvicornWorker -w 4
```

`application` is built at import, so the registry is populated and every declared
model is resolvable before the first request arrives rather than during it.

### In a container

`startproject` also writes a `Dockerfile`, a `.dockerignore` and a
`compose.yaml`. Nothing to research:

```bash
docker build -t myproject .
docker run -p 8000:8000 -e MLANGO_SECRET_KEY=... myproject
```

```bash
docker compose up --build      # Postgres for the metastore, one web process
```

The image is two-stage, runs as a non-root user, and its `HEALTHCHECK` calls
`/api/health` — which reports the registry and whether the metastore is
reachable, so a container that starts but cannot resolve its settings is marked
unhealthy rather than passing traffic.

`.dockerignore` excludes `mlango.db` and `artifacts/`, because copying a
developer's SQLite file into an image is how stale runs reach production.

### Configuration comes from the environment

The scaffolded `settings.py` reads what a deployment has to change, so a
container never edits a file:

| Variable | Effect |
|---|---|
| `MLANGO_SETTINGS_MODULE` | Which settings module to load |
| `MLANGO_SECRET_KEY` | Overrides the generated development key |
| `MLANGO_DEBUG=0` | Turns off debug |
| `DATABASE_URL` | Points the metastore at Postgres |

Before you go public:

- `DEBUG = False`
- `SECRET_KEY` from your secret store
- `ADMIN_PASSWORD`, or the admin behind your identity provider
- `SERVE_API_KEYS`, or auth terminated at the gateway
- `METASTORE` pointing at Postgres if more than one worker writes runs
- `STORAGE` pointing at shared storage if workers must see each other's artifacts

## Training somewhere else { #training-somewhere-else }

A laptop is a fine place to declare a model and a poor place to fit one. What
makes the GPU box a *part of the project* rather than a machine you copy files
off is two settings:

```python title="myproject/settings.py"
METASTORE = {"URL": os.environ["DATABASE_URL"]}          # shared history
STORAGE = {
    "BACKEND": "mlango.storage.s3.S3Storage",            # shared artifacts
    "ROOT": "s3://my-bucket/mlango",
}
```

```bash
pip install "mlango[s3]"
```

Then the workflow is the same one you already know, run somewhere else:

```bash
# on the machine with the GPU
python manage.py train reviews.Sentiment --tag overnight
```

```bash
# back on the laptop
python manage.py runs list
python manage.py explain reviews.Sentiment
python manage.py predict reviews.Sentiment "loved it"
python manage.py runserver
```

Nothing had to be copied. The run, its metrics, its parameters and its
reproducibility record are rows the laptop can read; the artifact is an object
it can fetch; and the admin shows the training you did not watch.

What makes this work is that artifacts are recorded by **storage-relative
name** — `models/reviews/Sentiment/<run>.joblib`, never
`/home/gpu/artifacts/models/...`. A run that wrote an absolute path into a
shared metastore leaves a row that only one machine can resolve, which is a
subtle way for a shared database to be useless. Versions registered before this
was true still carry an absolute path, and still load, on the machine that
wrote them.

The `ENDPOINT_URL` option points the same backend at MinIO, Cloudflare R2 or
Backblaze B2, so "S3" here means the protocol rather than the vendor.
Credentials are boto3's — environment variables, instance roles, profiles —
which is why mlango has no credential settings of its own to leak.

!!! note "What this is not"
    There is no job scheduler and no cluster support. mlango does not launch the
    training; you do, with ssh, a Slurm script, a GitHub Actions runner or
    whatever you already use. What it provides is the part that makes the result
    reach you.

See [SECURITY.md](https://github.com/DrobyshevDev/mlango/blob/master/SECURITY.md)
for the full list of development defaults you must change.
