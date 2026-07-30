# Модели

`Model` объявляет гиперпараметры и то, как собрать оценщик. Ран принадлежит
фреймворку: сиды, сплиты, цикл, метрики, артефакты, версионирование.

```python
from mlango.core import fields
from mlango.training import Model

from reviews.datasets import Reviews


class Sentiment(Model):
    """TF-IDF и логистическая регрессия."""

    max_features = fields.IntegerField(default=20_000, min_value=1, tunable=True)
    C = fields.FloatField(default=1.0, min_value=0.0, tunable=True)

    class Meta:
        dataset = Reviews
        trainer = "sklearn"
        task = "classification"
        features = ["text"]
        splits = {"train": 0.8, "val": 0.2}

    def build(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        return make_pipeline(
            TfidfVectorizer(max_features=self.max_features),
            LogisticRegression(C=self.C),
        )
```

## Гиперпараметры — это поля

Поскольку это те же объекты `Field`, что использует датасет, вы бесплатно
получаете валидацию, значения по умолчанию, интроспекцию, запись и свипы:

```python
>>> Sentiment(C=-1).full_clean()
ValidationError: C: Value -1.0 is below the minimum of 0.0.
```

Каждый ран записывает приведённые значения, поэтому «какие настройки дали эту
цифру?» отвечается из метастора, а не по памяти.

Помеченное `tunable=True` поле входит в пространство свипа по умолчанию.

## Опции `Meta`

| Опция | Назначение |
|---|---|
| `dataset` | Класс датасета (или его label), на котором обучаться |
| `trainer` | Ключ из настройки `TRAINERS`: `"sklearn"`, `"torch"`, `"transformers"` |
| `task` | `"classification"` или `"regression"` — определяет набор метрик |
| `target` | Какое поле предсказывать; выводится, если у датасета ровно одна цель |
| `features` | Какие поля подаются модели. **Указывайте это.** |
| `exclude` | Поля, которые выбросить, если `features` не задан |
| `splits` | Пропорции, по умолчанию `{"train": 0.8, "val": 0.2}` |
| `monitor`, `monitor_mode` | Метрика, по которой ранжируют раннюю остановку и свипы |

!!! warning "Объявляйте `features`"
    Без `features` входом становится каждое нецелевое поле. Первичный ключ
    исключается автоматически, но всё остальное — номер строки, протёкшая метка,
    метка времени, кодирующая ответ — нет. Явность это одна строка, которая
    предотвращает целый класс молчаливых утечек.

## Обучение

```python
model = Sentiment(C=2.0)
run = model.train(tags=["baseline"])
```

```bash
python manage.py train reviews.Sentiment -p C=2.0 -p max_features=5000 --tag baseline
```

Что фреймворк делает вокруг вашего `build()`:

1. Разрешает датасет, поля признаков и цели
2. Ставит сид `random`, `numpy` и `torch` из `settings.SEED`
3. Детерминированно режет данные
4. Открывает ран и записывает параметры и отпечаток данных
5. Фиксирует git-коммит, хост, версию Python и устройство
6. Вызывает `build()` и передаёт результат тренеру
7. Записывает метрики по мере поступления
8. Оценивает на валидационном сплите
9. Сохраняет артефакт и регистрирует версию

Полезные аргументы:

| Аргумент | Эффект |
|---|---|
| `materialize=True` | Сначала заморозить обучающее представление в версию датасета |
| `register=False` | Обучить без добавления в реестр моделей |
| `queryset=...` | Обучаться на явном queryset вместо всего датасета |
| `splits={...}` | Переопределить пропорции сплитов |
| `callbacks=[...]` | Добавить колбэки для этого рана |
| `seed=...` | Переопределить сид |

## Инференс

```python
model.predict("отличный фильм")                     # один вход
model.predict(["отличный фильм", "ужасное кино"])   # батч
model.predict_proba("отличный фильм")               # {'negative': 0.04, 'positive': 0.96}
model.evaluate(Reviews.objects.take(100))           # отчёт по метрикам
```

## Реестр

Каждый ран обучения регистрирует версию:

```python
Sentiment.versions()                       # новые первыми
Sentiment.load()                           # последнюю
Sentiment.load(version=3)                  # конкретную
Sentiment.load(stage="production")
Sentiment.production()                     # то же короче
Sentiment.promote(3, "production")         # действующую переводит в archived
```

Стадии: `none`, `staging`, `production`, `archived`. Промоут — один клик в
админке или один вызов здесь.

## Пресеты

Повторяющиеся формы уже написаны. Django поставляет generic views, чтобы
девяностая CRUD-страница занимала три строки; здесь та же идея:

```python
from mlango.training import TextClassifier

class Sentiment(TextClassifier):
    """Дообучение предобученного энкодера на отзывах."""

    class Meta:
        dataset = Reviews
        features = ["text"]
```

Это полная декларация. `base_model`, `learning_rate`, `epochs`, `batch_size`,
`max_length`, `weight_decay`, `warmup_ratio` и `build()` приходят из пресета, и
каждое из них можно переопределить.

| Пресет | Тренер | Для чего |
|---|---|---|
| `TextClassifier` | `transformers` | Дообучение энкодера для классификации текста |
| `TextRegressor` | `transformers` | Предсказание непрерывной величины из текста |
| `TabularClassifier` | `torch` | Небольшая полносвязная сеть по числовым колонкам |
| `TabularRegressor` | `torch` | То же, но предсказывает число |
| `TransformerModel` | `transformers` | Общая база, если нужна другая голова |

!!! note "Meta-опции наследуются"
    Дочерний класс со своим `class Meta` сохраняет всё, что объявил родитель —
    `trainer`, `task`, `monitor` — и переопределяет только то, что назвал сам. В
    Python тела классов сами по себе не наследуются, поэтому mlango их сливает;
    без этого переиспользуемый базовый класс написать невозможно.

```bash
pip install "mlango[transformers]"
python manage.py train reviews.Sentiment -p epochs=2 -p learning_rate=3e-5
```

Цикл дообучения — собственный mlango, а не `transformers.Trainer`, поэтому
колбэки, ранняя остановка, запись метрик и трекинг ранов ведут себя одинаково,
какой бы бэкенд проект ни выбрал. Заимствуется то, что стоит заимствовать:
токенизация, предобученные веса и головы моделей.

Переопределяйте только то, что действительно специфично для модели:

| Метод | Заменяет |
|---|---|
| `encode_batch(records, target)` | Токенизацию одного или двух текстовых полей |
| `configure_optimizer(module)` | AdamW без decay на смещениях и layer norm |
| `build()` | Голову, выбранную по классам целевого поля |

Два текстовых поля автоматически становятся парой предложений, что покрывает
entailment, схожесть и оценку пар «вопрос-ответ»:

```python
class Entailment(TextClassifier):
    class Meta:
        dataset = Pairs
        features = ["premise", "hypothesis"]
```

## Свипы

Поля, помеченные `tunable`, задают пространство по умолчанию:

```bash
python manage.py sweep reviews.Sentiment
python manage.py sweep reviews.Sentiment -p C=0.25,1,4 -p max_features=500,5000
python manage.py sweep reviews.Sentiment --strategy random --trials 20 --seed 0
python manage.py sweep reviews.Sentiment -p C=0.5,1,2 --promote-best production
```

```python
result = Sentiment.sweep({"C": [0.5, 1.0, 2.0]}, metric="accuracy", mode="max")
result.best.params        # {'C': 1.0}
result.ranked()           # все завершённые трейлы, лучшие первыми
```

Один родительский ран держит поиск; каждый трейл — полноценный дочерний ран со
своей записью. Упавший трейл записывается, и свип продолжается.

## Колбэки

Middleware цикла обучения:

```python
from mlango.training import Checkpoint, EarlyStopping, MetricThreshold, ProgressBar

model.train(callbacks=[
    ProgressBar(),
    EarlyStopping(monitor="val_loss", patience=3),
    Checkpoint(monitor="val_accuracy", mode="max"),
    MetricThreshold("accuracy", minimum=0.85),   # уронить ран в CI
])
```

Добавить колбэки для каждого рана — настройкой `DEFAULT_CALLBACKS`.

!!! note "Запись метрик — не колбэк"
    Метрики пишет сам фреймворк, поэтому опустошение `DEFAULT_CALLBACKS` никогда
    не лишит вас истории ранов. Колбэки только добавляют.

Свой колбэк — переопределите нужные хуки:

```python
from mlango.training import Callback


class NotifySlack(Callback):
    def on_train_end(self, run, model, **kwargs):
        post_to_slack(f"{model._meta.label} завершил: {run.refresh().summary}")
```

Колбэк, бросивший исключение, логируется и пропускается — инструментирование не
должно быть причиной падения долгой задачи.

## Тренеры

Тренер знает, как обучить, предсказать, сохранить и загрузить. Всё остальное —
фреймворка, поэтому добавление тренера это небольшой файл:

```python
from mlango.training import Trainer


class LightGBMTrainer(Trainer):
    name = "lightgbm"
    requires = ("lightgbm",)
    extension = "txt"

    def fit(self, model, train, validation, run, callbacks, *, target="", features=None, **kw):
        booster = model.build()
        x, y = train.xy(target=target, features=features)
        booster.fit(x, y)
        run.log_metrics({"train_score": booster.score(x, y)}, step=0)
        return booster

    def predict(self, model, fitted, inputs):
        return list(fitted.predict(inputs))

    def save(self, model, fitted, name):
        from mlango.storage import default_storage

        path = default_storage().path(f"{name}.{self.extension}")
        fitted.save_model(path)
        return path

    def load(self, model, path):
        import lightgbm

        return lightgbm.Booster(model_file=path)
```

```python
TRAINERS = {"lightgbm": "myproject.trainers.LightGBMTrainer"}
```

### PyTorch

`build()` возвращает `nn.Module`. Цикл, размещение на устройстве, батчинг,
метрики и чекпоинты — фреймворка:

```python
class Classifier(Model):
    epochs = fields.IntegerField(default=20)
    batch_size = fields.IntegerField(default=64)
    learning_rate = fields.FloatField(default=1e-3, tunable=True)

    class Meta:
        dataset = Tabular
        trainer = "torch"
        task = "classification"
        features = ["x1", "x2", "x3"]

    def build(self):
        import torch.nn as nn

        return nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 2))
```

`settings.DEVICE` по умолчанию `"auto"`, что использует CUDA, когда она доступна.

## Развёртывание

```python
from mlango.serve import path

urlpatterns = [
    path("predict/", Sentiment.as_endpoint(stage="production")),
]
```

Версия загружается один раз, при первом запросе. См. [Развёртывание](serving.md).
