# Учебник

Мы соберём проект, который классифицирует тикеты поддержки по срочности, а потом
поставим перед ним агента. К концу у вас будет отслеживаемый ран обучения,
версия модели, готовая к промоуту, набор оценок, inference API и админка, которая
показывает всё это.

Всё работает локально, без API-ключа.

## 1. Создать проект

```bash
pip install "mlango[sklearn]"
mlango startproject helpdesk --bare
cd helpdesk
```

`--bare` пропускает демо-приложение, потому что мы пишем своё. (Уберите флаг — и
получите рабочий пример, который можно читать.)

```bash
python manage.py startapp tickets
```

Добавьте приложение в `helpdesk/settings.py`:

```python title="helpdesk/settings.py" hl_lines="2"
INSTALLED_APPS = [
    "tickets",
]
```

Затем проверьте проект:

```bash
python manage.py check
```

## 2. Объявить датасет

Сохраните данные как `data/tickets.jsonl` — один JSON-объект на строку:

```json title="data/tickets.jsonl"
{"id": 1, "subject": "Не могу войти совсем", "urgency": "high"}
{"id": 2, "subject": "Опечатка на странице тарифов", "urgency": "low"}
{"id": 3, "subject": "Платежи падают у всех", "urgency": "high"}
{"id": 4, "subject": "Идея по экспорту данных", "urgency": "low"}
```

Теперь объявите, чем *является* запись. Можно написать это руками — но файл уже
есть, так что пусть mlango прочитает его и напишет черновик сам:

```bash
python manage.py inspectdata data/tickets.jsonl --name Tickets
```

```python title="tickets/datasets.py"
from mlango.core import fields
from mlango.data import Dataset, JSONLSource


class Tickets(Dataset):
    """Тикеты поддержки, размеченные по срочности."""

    id = fields.IntegerField()
    subject = fields.TextField(max_length=500)
    urgency = fields.LabelField(["low", "high"])

    class Meta:
        source = JSONLSource("data/tickets.jsonl")
        primary_key = "id"
```

`inspectdata` читает выборку, подбирает тип поля для каждой колонки, замечает,
что `id` уникален, и делает его первичным ключом, а в `urgency` узнаёт метку.
Вставьте вывод в `tickets/datasets.py` — или добавьте `--write --app tickets` и
обойдитесь без вставки.

Это черновик, а не истина, и выше — черновик после одной правки. На четырёх
коротких строках команда предложит `subject = CharField(max_length=32)`, потому
что ничего длиннее не видела; настоящие тикеты — это проза, так что расширьте до
`TextField(max_length=500)`. Всё угаданное помечено комментарием — ради этого
вывод и стоит читать, а не принимать на веру.

Из этих шести строк следуют три вещи.

**Данные валидируются против декларации:**

```bash
python manage.py dataset validate tickets.Tickets
python manage.py dataset head tickets.Tickets
```

Если в строке будет `"urgency": "urgent"`, валидация скажет об этом и назовёт
номер строки.

**Их можно лениво запрашивать:**

```python
Tickets.objects.filter(urgency="high").count()
Tickets.objects.filter(subject__icontains="войти").take(5).all()
Tickets.objects.shuffle(seed=0).split(train=0.8, test=0.2)
```

**Сплиты стабильны.** Назначение хеширует `id` каждой записи, поэтому добавление
тикетов в следующем месяце не перемещает существующие между train и test.

## 3. Зафиксировать схему в миграции

```bash
python manage.py makemigrations
python manage.py migrate
```

Откройте `tickets/migrations/0001_initial.py` — это обычный читаемый Python.
Добавите поле позже — `makemigrations` запишет разницу, и через полгода вы
сможете сказать, что именно означала сохранённая версия датасета.

## 4. Объявить модель

```python title="tickets/models.py"
from mlango.core import fields
from mlango.training import Model

from tickets.datasets import Tickets


class Urgency(Model):
    """TF-IDF по теме тикета, затем логистическая регрессия."""

    max_features = fields.IntegerField(default=5000, min_value=1, tunable=True)
    C = fields.FloatField(default=1.0, min_value=0.0, tunable=True)

    class Meta:
        dataset = Tickets
        trainer = "sklearn"
        task = "classification"
        features = ["subject"]

    def build(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        return make_pipeline(
            TfidfVectorizer(max_features=self.max_features),
            LogisticRegression(C=self.C, max_iter=1000),
        )
```

!!! warning "Почему `features` указан явно"
    Без него `id` попал бы во входы модели. Явный `features` — или расчёт на то,
    что `primary_key` исключит идентификатор — это то, что не даёт модели тихо
    учиться на номере строки.

Гиперпараметры — это **поля**, а значит они валидируются, имеют значения по
умолчанию, записываются в каждый ран и участвуют в свипах. Парсер конфигов вы не
писали.

## 5. Обучить

```bash
python manage.py train tickets.Urgency -p C=2.0 --tag baseline
```

Фреймворк открыл ран, поставил сид всем генераторам, порезал данные, вызвал ваш
`build()`, записал метрики, зафиксировал git-коммит, сохранил артефакт и
зарегистрировал версию 1.

```bash
python manage.py runs list
python manage.py runs show <run-id>
```

Обратите внимание на `_data_fingerprint` в параметрах. Два рана с одинаковым
отпечатком видели одно и то же представление данных — именно это делает
сравнение осмысленным.

Попробуйте на том, чего модель не видела, — сервер поднимать не нужно:

```bash
python manage.py predict tickets.Urgency "страница оплаты не работает ни у кого"
```

`predict` загружает зарегистрированную версию — тот же артефакт, который отдавал
бы API, — то есть здесь виден ровно продакшен-ответ. Ещё он умеет `--dataset`,
чтобы оценить объявленные данные, и `--file` для пакета:

```bash
python manage.py predict tickets.Urgency --dataset --filter urgency=high -n 5
python manage.py predict tickets.Urgency --file inbox.jsonl --format jsonl --output scored.jsonl
```

## 6. Обойти пространство параметров

Оба гиперпараметра помечены `tunable`, поэтому:

```bash
python manage.py sweep tickets.Urgency -p C=0.25,1,4 -p max_features=500,5000
```

Один родительский ран, шесть дочерних, ранжированная таблица. Добавьте
`--promote-best production` — и победитель будет промоутнут той же командой.

## 7. Сказать, что значит «хорошо»

```python title="tickets/evals.py"
from mlango.evals import Eval, exact_match

from tickets.datasets import Tickets
from tickets.models import Urgency


class UrgencyAccuracy(Eval):
    """Согласен ли классификатор с размеченной срочностью?"""

    class Meta:
        dataset = Tickets
        target = Urgency
        input_field = "subject"
        expected_field = "urgency"
        case_id_field = "id"
        scorers = {"correct": exact_match}
        threshold = 1.0
```

```bash
python manage.py evaluate tickets.UrgencyAccuracy --show-failures
```

Каждый кейс сохраняется, поэтому регрессия — это дифф между двумя ранами, а не
цифра, которую кто-то помнит. В CI:

```bash
python manage.py evaluate tickets.UrgencyAccuracy --min-pass-rate 0.9
```

## 8. Поставить впереди агента

```python title="tickets/agents.py"
from mlango.agents import Agent, BufferMemory, tool


@tool
def triage(subject: str) -> str:
    """Определить срочность тикета: low или high.

    Args:
        subject: Тема тикета.
    """
    from tickets.models import Urgency

    try:
        model = Urgency.load(stage="production")
    except LookupError:
        return "Продакшн-модели пока нет. Запустите свип с --promote-best production."
    return str(model.predict(subject))


class Triage(Agent):
    """Разбирает входящие тикеты, используя обученный классификатор."""

    class Meta:
        system = (
            "Ты разбираешь тикеты поддержки. Получив тему, вызывай инструмент "
            "triage, а не угадывай, затем коротко поясни результат."
        )
        tools = [triage]
        memory = BufferMemory(k=20)
```

JSON-схема инструмента берётся из аннотаций типов и докстроки — вы описываете
его один раз.

```bash
python manage.py agent tickets.Triage
```

Проект идёт с офлайн-провайдером `echo`, поэтому это работает без учётных
данных. Для настоящей модели поставьте `DEFAULT_PROVIDER = "anthropic"` и
экспортируйте `ANTHROPIC_API_KEY`.

Каждый вызов трассируется:

```bash
python manage.py traces list
python manage.py traces show <trace-id> -v 2
```

## 9. Развернуть

```python title="helpdesk/routes.py"
from mlango.serve import path

from tickets.agents import Triage
from tickets.models import Urgency

urlpatterns = [
    path("predict/", Urgency.as_endpoint(stage="production")),
    path("triage/", Triage.as_endpoint()),
    path("triage/stream/", Triage.as_stream_endpoint()),
]
```

```bash
python manage.py runserver
```

- Админка: <http://127.0.0.1:8000/admin/>
- Документация API: <http://127.0.0.1:8000/api/docs>

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H 'Content-Type: application/json' \
  -d '{"input": "Платежи упали у всех клиентов"}'
```

## 10. Посмотреть, что получилось

Откройте админку. Не написав ни одного шаблона, вы получили:

- **Tickets** — данные с фильтром по `urgency` и поиском по `subject`
- **Urgency** — все версии, их метрики и промоут в один клик
- **Runs** — историю с графиками метрик и сравнение рядом
- **Traces** — каждый вызов агента, пошагово

Настроить вид, когда захочется:

```python title="tickets/admin.py"
from mlango import admin

from tickets.datasets import Tickets


@admin.register(Tickets)
class TicketsAdmin(admin.ObjectAdmin):
    list_display = ("id", "subject", "urgency")
    list_filter = ("urgency",)
    search_fields = ("subject",)
```

## Что читать дальше

- **[Концепции](concepts.md)** — почему `_meta` это контракт, который читают все
- **[Датасеты](datasets.md)** — полная история queryset и версионирования
- **[Модели](models.md)** — тренеры, пресеты, колбэки и реестр
- **[Агенты](agents.md)** — инструменты, память, провайдеры, стриминг, трейсинг
- **[Настройки](settings.md)** — все параметры и их значения по умолчанию
