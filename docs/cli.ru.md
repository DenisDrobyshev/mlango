# Командная строка

В каждом проекте есть `manage.py`. Скрипт `mlango` делает то же самое, когда
проекта ещё нет, а `python -m mlango` работает, если скрипта нет в `PATH`.

```bash
python manage.py help
python manage.py help train
```

## Команды

### Начало работы

| Команда | Что делает |
|---|---|
| `mlango startproject NAME [DIR]` | Создаёт проект, который уже работает. `--bare` пропускает демо-приложение |
| `manage.py startapp NAME` | Создаёт приложение: datasets, models, agents, evals, admin, migrations, tests |
| `manage.py check` | Проверяет настройки, бэкенды, связи, миграции и админку |

### Данные

```bash
python manage.py dataset list
python manage.py dataset show reviews.Reviews
python manage.py dataset head reviews.Reviews -n 20
python manage.py dataset validate reviews.Reviews
python manage.py dataset materialize reviews.Reviews --notes "ночной снимок"
python manage.py dataset versions reviews.Reviews
```

### Миграции

```bash
python manage.py makemigrations [app] [-n NAME] [--dry-run] [--empty]
python manage.py migrate [app] [--plan] [--fake]
python manage.py showmigrations [app]
```

### Обучение

```bash
python manage.py train reviews.Sentiment -p C=2.0 -p max_features=5000 \
    --tag baseline --notes "первая попытка" --materialize

python manage.py sweep reviews.Sentiment -p C=0.25,1,4 \
    --strategy grid --metric accuracy --mode max --promote-best production
```

| Флаг | Что делает |
|---|---|
| `-p NAME=VALUE` | Переопределяет гиперпараметр. Можно повторять |
| `--dataset LABEL` | Обучает на другом датасете |
| `--tag TAG` | Помечает запуск тегом. Можно повторять |
| `--seed N` | Переопределяет seed |
| `--materialize` | Сначала фиксирует обучающую выборку как версию датасета |
| `--no-register` | Обучает, не добавляя в реестр версий |

### Оценка

```bash
python manage.py evaluate support.AnswerQuality
python manage.py evaluate support.AnswerQuality --show-failures
python manage.py evaluate support.AnswerQuality --min-pass-rate 0.9
```

### Агенты

```bash
python manage.py agent support.Support                          # интерактивно
python manage.py agent support.Support "как мне ...?"            # один запрос
python manage.py agent support.Support "..." --show-steps        # показать вызовы инструментов
python manage.py agent support.Support "..." --session user-42   # с памятью
```

### Что уже произошло

```bash
python manage.py runs list --kind train --status finished -n 20
python manage.py runs show 7c8f1020
python manage.py runs compare 7c8f1020 c089b7e6

python manage.py traces list --agent support.Support
python manage.py traces show a1b2c3d4 -v 2
```

### Разработка

```bash
python manage.py runserver              # 127.0.0.1:8000
python manage.py runserver 8080
python manage.py runserver 0.0.0.0:8080 --reload
python manage.py runserver --no-admin

python manage.py shell                  # IPython, если установлен
python manage.py shell -c "print(Reviews.objects.count())"

python manage.py test                   # pytest, на одноразовом метахранилище
python manage.py test -k splits -x
python manage.py test --coverage
```

`manage.py test` на время прогона переводит метахранилище и хранилище артефактов
во временный каталог, поэтому тест физически не может задеть настоящие данные —
та же идея, что и тестовая база в Django.

`startproject` создаёт готовый каталог `tests/`, так что новый проект зелёный
ещё до первой правки: есть с чего начать и есть что скопировать.

## Общие флаги

Доступны в каждой команде:

| Флаг | Что делает |
|---|---|
| `--settings MODULE` | Использовать другой модуль настроек для этого запуска |
| `-v 0..3` | Тихо, обычно, подробно, очень подробно |
| `--traceback` | Показать полный traceback вместо сообщения |

## Оболочка

`manage.py shell` заранее импортирует все объявленные объекты и несколько
вспомогательных функций:

```python
>>> Reviews.objects.filter(label="positive").count()
1284
>>> Sentiment.versions()
[<ModelVersion reviews.Sentiment@v2 stage=production>, ...]
>>> recent_runs(limit=3)
>>> get_trace("a1b2c3d4").spans
>>> apps.summary()
```

## Свои команды

Положите модуль в `<app>/management/commands/`, и он появится в
`manage.py help` — включая команду, которая **переопределяет встроенную**.
Именно так проект настраивает `train` под себя, не форкая фреймворк.

```python title="reviews/management/commands/import_reviews.py"
from mlango.management import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Импортировать отзывы из хранилища."

    def add_arguments(self, parser):
        parser.add_argument("since", help="Дата в формате ISO, с которой импортировать.")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, **options):
        rows = fetch_since(options["since"])
        if not rows:
            raise CommandError(f"Нечего импортировать с {options['since']}.")

        self.table(
            ["id", "subject"],
            [[r["id"], r["subject"]] for r in rows[:10]],
        )
        if options["dry_run"]:
            self.warn("Пробный запуск: ничего не записано.")
            return
        write(rows)
        self.ok(f"Импортировано отзывов: {len(rows)}.")
```

Что доступно на `self`:

| Метод | Печатает |
|---|---|
| `self.write(msg, level=1)` | Строку, с учётом `-v` |
| `self.ok(msg)` / `self.warn(msg)` | Зелёным / жёлтым |
| `self.stderr(msg)` | В stderr |
| `self.table(headers, rows)` | Выровненную таблицу |
| `self.style.bold(...)` и т. д. | Цвет, отключается при перенаправлении вывода |

Бросайте `CommandError` там, где пользователь должен увидеть сообщение, а не
traceback. Поставьте `requires_apps = False` для команды, которая должна
работать до загрузки приложений, и `requires_settings = False` — для той,
что работает вообще без проекта.
