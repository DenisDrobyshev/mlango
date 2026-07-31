# Расширение mlango

Каждая часть mlango, соприкасающаяся с внешним миром, — это небольшой контракт
за именем в настройках. Это не украшение: именно так проект подменяет нужное, не
форкая фреймворк, и именно так вы публикуете своё, ни у кого не спрашивая
разрешения.

## Точки расширения

| Точка | Контракт | Чем регистрируется |
|---|---|---|
| Тренер | `fit`, `predict`, `save`, `load` | `TRAINERS` или entry point `mlango.trainers` |
| LLM-провайдер | один метод: `complete()` | `PROVIDERS` или entry point `mlango.providers` |
| Хранилище артефактов | `path`, `open`, `save_bytes`, `read_bytes`, `exists`, `delete`, `size`, `listdir` | `STORAGE["BACKEND"]` |
| Источник данных | итерируемое из словарей, по желанию `count()` | указывается в `Dataset.Meta.source` |
| Колбэк обучения | любое подмножество хуков `Callback` | `DEFAULT_CALLBACKS` или на запуск |
| Middleware сервинга | ASGI middleware | `SERVE_MIDDLEWARE` |
| Команда | `Command(BaseCommand)` | `<app>/management/commands/` |

Контракты узкие намеренно. У провайдера ровно один метод, потому что цикл
агента, диспетчеризация инструментов, память и трейсинг принадлежат фреймворку:
смена провайдера не должна уметь менять поведение агента.

## Начните с рабочего пакета

```bash
mlango startplugin mlango-lightgbm --kind trainer
```

```
mlango-lightgbm/
├── pyproject.toml          # entry point уже объявлен
├── README.md
├── LICENSE
├── src/mlango_lightgbm/
│   ├── __init__.py
│   └── trainer.py          # контракт, с комментариями в интересных местах
└── tests/test_trainer.py   # включая тест, что entry point разрешается
```

`--kind` — это `trainer`, `provider`, `storage` или `source`. Проект не нужен:
чтобы написать расширение, не должно требоваться сначала придумать, где его
писать.

## Обнаружение

Пакет заявляет о себе стандартным entry point:

```toml title="pyproject.toml"
[project.entry-points."mlango.trainers"]
lightgbm = "mlango_lightgbm.trainer:LightGBMTrainer"
```

После `pip install mlango-lightgbm` это работает вообще без правки настроек:

```python
class Urgency(Model):
    class Meta:
        trainer = "lightgbm"
```

Сливаются три источника, в таком порядке:

1. значения по умолчанию самого mlango;
2. установленные пакеты — через entry points;
3. `TRAINERS` / `PROVIDERS` проекта.

Проект побеждает всегда. Направить `TRAINERS["lightgbm"]` на пропатченный
подкласс должно быть можно без удаления пакета, который дал это имя: расширение,
которое нельзя переопределить, — сделка хуже, чем строка с путём, которую оно
заменило.

Во время обнаружения ничего не импортируется. Читаются только имена и пути, а
разрешение остаётся ленивым, поэтому сломанный плагин падает тогда, когда его
попросят, а не на старте. `manage.py check` печатает, что откуда пришло:

```
Backends
  trainer    sklearn: available
  trainer    lightgbm: available (plugin)
  plugin     TRAINERS['lightgbm'] = mlango_lightgbm.trainer.LightGBMTrainer
```

У хранилищ и источников данных entry point нет, и это не упущение: у проекта
ровно один бэкенд хранилища, названный в настройках, а источник импортируется и
называется в объявлении. Ни там, ни там реестру нечего разрешать.

## Именование

Называйте дистрибутив `mlango-<что он добавляет>`:

- `mlango-lightgbm`, `mlango-catboost` — тренеры;
- `mlango-openai`, `mlango-ollama` — провайдеры;
- `mlango-gcs`, `mlango-azure` — хранилища;
- `mlango-snowflake`, `mlango-bigquery` — источники.

Это соглашение, а не правило. Оно ничего не стоит и позволяет по списку
зависимостей понять, зачем там каждая строка, а поиску по `mlango-` — найти
экосистему целиком, а не её часть.

Имя импортируемого пакета следует за ним: `mlango_lightgbm`. Регистрируемое имя
теряет префикс: `"lightgbm"`.

## Как написать тренер

Показать стоит именно его — он выглядит более трудоёмким, чем есть:

```python
from mlango.training import Trainer


class LightGBMTrainer(Trainer):
    name = "lightgbm"
    requires = ("lightgbm",)      # проверяется до использования: понятное сообщение вместо ImportError
    extension = "txt"

    def fit(self, model, train, validation, run, callbacks, *, target="", features=None, **kw):
        booster = model.build()
        x, y = train.xy(target=target, features=features)
        booster.fit(x, y)
        run.log_metrics({"train_score": booster.score(x, y)}, epoch=0, step=0)
        return booster

    def predict(self, model, fitted, inputs):
        return list(fitted.predict(inputs))

    def save(self, model, fitted, name):
        from mlango.storage import default_storage

        with default_storage().writable(f"{name}.{self.extension}") as target:
            fitted.save_model(target.path)
            return target.name

    def load(self, model, path):
        import lightgbm

        from mlango.storage import default_storage

        with default_storage().readable(path) as local:
            return lightgbm.Booster(model_file=local)
```

Это всё. Учёт запусков, история метрик, версионирование, промоут, свипы,
админка, инференс-эндпоинт и `manage.py train` уже написаны и не знают, какой
бэкенд произвёл число.

Две детали, которые стоит перенять:

- **`writable()` и `readable()`, а не `path()`.** Они дают локальный путь и
  публикуют результат на выходе, поэтому тот же тренер работает, когда проект
  переезжает на S3. `save()` возвращает *имя* в хранилище: абсолютный путь в
  метасторе разрешается только на той машине, которая его записала.
- **Метрики — через `run`.** Метрики принадлежат запуску, а не бэкенду. Именно
  это помещает сторонний тренер в ту же историю, админку и таблицу сравнения,
  что и встроенные.

Необязательные хуки: `predict_proba()`, `describe()` (показывается на странице
запуска) и `importances()` (веса признаков — возвращайте `None`, когда веса не
соответствуют ничему, что человек назвал бы признаком; см.
[Важность признаков](models.md#_6)).

## Как тестировать расширение

Скаффолд поставляет фикстуру, которая настраивает mlango так же, как это делает
проект, — чтобы тесты гоняли код так, как его будет гонять фреймворк:

```python
settings.configure(
    BASE_DIR=str(tmp_path),
    METASTORE={"URL": "sqlite:///test.db"},
    STORAGE={"BACKEND": "mlango.storage.local.LocalStorage", "ROOT": "artifacts"},
    INSTALLED_APPS=[],
)
```

Он поставляет и тест, проверяющий, что entry point разрешается после установки.
Оставьте его. Entry point, который никто ни разу не разрешил, — самый частый
способ для расширения выглядеть законченным и не делать ничего.

## Публикация

Ничего специфичного для mlango:

```bash
pip install build twine
python -m build
twine upload dist/*
```

Потом заведите issue в
[трекере](https://github.com/DrobyshevDev/mlango/issues), чтобы пакет попал в
список. Процедуры одобрения нет, и реестра, в который надо быть принятым, тоже:
entry point — это весь механизм, и он работает независимо от того, слышал ли
кто-нибудь про ваш пакет.
