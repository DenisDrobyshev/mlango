# Развёртывание

`manage.py runserver` отдаёт админку и документированный inference API вместе —
так же, как `django-admin runserver` отдаёт админку рядом с вашими вьюхами.

## Маршруты

`routes.py` проекта — это его `urls.py`:

```python title="myproject/routes.py"
from mlango.serve import path

from reviews.models import Sentiment
from support.agents import Support

urlpatterns = [
    path("predict/", Sentiment.as_endpoint(stage="production"), name="sentiment"),
    path("chat/", Support.as_endpoint(), name="support"),
    path("chat/stream/", Support.as_stream_endpoint(), name="support-stream"),
]
```

```python title="myproject/settings.py"
ROOT_ROUTECONF = "myproject.routes"
```

Маршруты монтируются под `/api`, поэтому `path("predict/")` отдаёт
`POST /api/predict/`.

Разнести их по приложениям — через `include`:

```python
from mlango.serve import include, path

urlpatterns = [
    *include("reviews.routes"),
    *include("support.routes"),
]
```

## Эндпоинты моделей

```python
Sentiment.as_endpoint()                     # последняя зарегистрированная версия
Sentiment.as_endpoint(version=3)            # закреплённая
Sentiment.as_endpoint(stage="production")   # та, что промоутнута
```

Версия загружается один раз, лениво, при первом запросе — поэтому запуск сервера
не требует обученной модели, а свежий промоут подхватывается перезапуском.

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H 'Content-Type: application/json' \
  -d '{"input": "отличный фильм"}'

curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H 'Content-Type: application/json' \
  -d '{"inputs": ["отличный фильм", "ужасное кино"], "proba": true}'
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

## Эндпоинты агентов

```bash
curl -X POST http://127.0.0.1:8000/api/chat/ \
  -H 'Content-Type: application/json' \
  -d '{"message": "Как перевыпустить API-ключ?", "session_id": "user-42"}'
```

```json
{
  "agent": "support.Support",
  "output": "Перевыпустите его в Настройки → API-ключи…",
  "steps": 2,
  "trace": "a1b2c3d4…",
  "tools_used": ["search_docs"],
  "usage": {"input_tokens": 1840, "output_tokens": 96, "total_tokens": 1936}
}
```

`session_id` — это то, что даёт агенту непрерывность между запросами через его
бэкенд памяти.

Для интерфейса, который показывает прогресс, есть стриминговый вариант на
Server-Sent Events — см. [Агенты](agents.md#_9).

## Документация бесплатно

Формы запроса и ответа — pydantic-модели, поэтому `/api/docs` описывает каждый
эндпоинт без единой строки OpenAPI:

- Swagger UI — `/api/docs`
- ReDoc — `/api/redoc`
- Схема — `/api/openapi.json`

## Здоровье

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

Годится как readiness-проба: подтверждает и что приложение поднялось, *и* что
реестр с метастором доступны.

## Middleware

Стек настраивается в настройках, снаружи внутрь:

```python
SERVE_MIDDLEWARE = [
    "mlango.serve.middleware.RequestLogMiddleware",
    "mlango.serve.middleware.RateLimitMiddleware",
    "mlango.serve.middleware.ApiKeyMiddleware",
    "mlango.serve.middleware.GuardrailMiddleware",
]
```

| Middleware | Делает |
|---|---|
| `RequestLogMiddleware` | Логирует метод, путь, статус и время; добавляет `X-Response-Time-Ms` |
| `ApiKeyMiddleware` | Требует `X-API-Key` из `SERVE_API_KEYS` на маршрутах `/api` |
| `RateLimitMiddleware` | Ограничение в фиксированном окне по адресу клиента |
| `GuardrailMiddleware` | Отклоняет тела запросов с `SERVE_BLOCKED_TERMS` |

Своё — обычным middleware Starlette:

```python
from starlette.middleware.base import BaseHTTPMiddleware


class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request.state.tenant = request.headers.get("X-Tenant", "default")
        return await call_next(request)
```

!!! warning "Ограничения в процессе"
    `RateLimitMiddleware` считает на воркер. Он остановит сорвавшийся скрипт, но
    не заменяет шлюз.

## Ошибки

| Брошено | Статус | Тело |
|---|---|---|
| `ValidationError` | 422 | Сообщения по полям |
| `LookupError` | 404 | Сообщение, например «нет зарегистрированной версии» |
| Любая `MlangoError` | 400 | Сообщение |

Поэтому запрос к модели, которая никогда не обучалась, возвращает 404 с точным
объяснением, а не 500.

## Продакшн

`runserver` — для разработки. В продакшне направьте ASGI-сервер на фабрику
приложения:

```bash
uvicorn "mlango.serve.api:create_app" --factory --host 0.0.0.0 --port 8000 --workers 4
```

```bash
gunicorn "mlango.serve.api:create_app()" -k uvicorn.workers.UvicornWorker -w 4
```

Установите `MLANGO_SETTINGS_MODULE` в окружении, и прежде чем выходить в публичный
доступ:

- `DEBUG = False`
- `SECRET_KEY` из вашего хранилища секретов
- `ADMIN_PASSWORD`, либо админка за провайдером идентичности
- `SERVE_API_KEYS`, либо аутентификация на шлюзе
- `METASTORE` на Postgres, если раны пишет больше одного воркера
- `STORAGE` на общее хранилище, если воркеры должны видеть артефакты друг друга

Полный список дефолтов разработки, которые нужно изменить, — в
[SECURITY.md](https://github.com/DenisDrobyshev/mlango/blob/master/SECURITY.md).
