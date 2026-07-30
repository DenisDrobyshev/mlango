# Датасеты

`Dataset` объявляет, как выглядит запись и откуда записи берутся. Всё остальное —
валидация, запросы, сплиты, версионирование, предпросмотр в админке — следует из
этой декларации.

```python
from mlango.core import fields
from mlango.data import Dataset, JSONLSource


class Reviews(Dataset):
    """Отзывы покупателей о товарах."""

    id = fields.IntegerField()
    text = fields.TextField(max_length=5000)
    label = fields.LabelField(["negative", "positive"])
    stars = fields.IntegerField(min_value=1, max_value=5)

    class Meta:
        source = JSONLSource("data/reviews.jsonl")
        primary_key = "id"
```

## Опции `Meta`

| Опция | Назначение |
|---|---|
| `source` | Откуда берутся записи: `Source`, вызываемый объект или список |
| `primary_key` | Поле для стабильных сплитов; исключается из признаков |
| `split_salt` | Измените, чтобы намеренно перетасовать сплиты |
| `verbose_name`, `verbose_name_plural` | Подписи в админке |
| `description` | По умолчанию — первый абзац докстроки класса |
| `license`, `homepage` | Происхождение данных, показывается в админке |

!!! tip "Файл уже есть? Не пишите это руками"
    ```bash
    python manage.py inspectdata data/reviews.jsonl
    ```
    прочитает выборку и напечатает объявление вроде того, что выше, — с типами
    полей, диапазонами, классами и первичным ключом. См.
    [Свои данные](cli.md#bringing-your-own-data).

## Источники

| Источник | Читает |
|---|---|
| `JSONLSource(path)` | По одному JSON-объекту на строку — формат обмена по умолчанию |
| `JSONSource(path, key=None)` | JSON-массив или объект, у которого массив под `key` |
| `CSVSource(path, delimiter=",")` | Разделённый текст; поля приводят строки к типам |
| `DirectorySource(root)` | Раскладку `root/<класс>/<файл>` для изображений и звука |
| `InMemorySource(rows)` | Список словарей — фикстуры и тесты |
| `PythonSource(factory)` | Вызываемый объект, отдающий записи — сгенерированные данные |
| `ChainSource(*sources)` | Несколько источников подряд — шарды, несколько дампов |
| `ParquetSource(path)` | Колоночные данные, потоком по row-group |
| `SQLSource(query, url=None)` | Всё, до чего дотягивается SQLAlchemy; по умолчанию метастор |
| `HuggingFaceSource(path, split=...)` | Сплит с Hugging Face hub, опционально потоком |
| `DatasetVersionSource(label, version)` | Замороженный снапшот другого датасета |

Трём последним нужен extra:

```bash
pip install "mlango[parquet]"        # ParquetSource
pip install "mlango[huggingface]"    # HuggingFaceSource
```

```python
class Reviews(Dataset):
    text = fields.TextField()
    label = fields.LabelField(["neg", "pos"])

    class Meta:
        source = ParquetSource("data/reviews.parquet", columns=["text", "label"])
```

`ParquetSource.count()` читает футер файла, поэтому подсчёт файла на сто
миллионов строк мгновенный. `SQLSource` стримит серверным курсором, а не
загружает результат целиком.

`DatasetVersionSource` — это способ привязать производный датасет к точному
состоянию источника вместо того, чтобы следовать за тем, что источник говорит
сегодня:

```python
class BalancedReviews(Dataset):
    """Построен на закреплённом снапшоте, поэтому вывод воспроизводим."""

    text = fields.TextField()
    label = fields.LabelField(["neg", "pos"])

    class Meta:
        source = DatasetVersionSource("reviews.Reviews", version=3)
```

Или переопределите `records()` и обойдитесь без `source`:

```python
class Reviews(Dataset):
    text = fields.TextField()

    @classmethod
    def records(cls):
        for row in fetch_from_warehouse():
            yield {"text": row.body}
```

Свой источник — это класс с двумя методами:

```python
from mlango.data import Source


class WarehouseSource(Source):
    def __init__(self, query: str):
        self.query = query

    def __iter__(self):
        yield from run_query(self.query)

    def count(self):
        return None          # «неизвестно» — нормальный ответ

    def describe(self):
        return {"type": "WarehouseSource", "query": self.query}
```

`describe()` записывается рядом с каждым раном, поэтому пусть он будет
информативным и без учётных данных.

## QuerySet

`Dataset.objects` выдаёт ленивый неизменяемый queryset. Ничего не читается, пока
вы не начнёте итерировать, поэтому одно и то же выражение работает на десяти
строках в тесте и на десяти миллионах в продакшне.

### Фильтрация

Лукапы используют джанговское написание `field__op`:

```python
Reviews.objects.filter(label="positive")
Reviews.objects.filter(stars__gte=4)
Reviews.objects.exclude(text__icontains="спам")
Reviews.objects.filter(label__in=["positive", "neutral"])
Reviews.objects.filter(text__isnull=False)
```

| Лукап | Значение |
|---|---|
| `exact` (по умолчанию), `iexact`, `ne` | Равенство |
| `gt`, `gte`, `lt`, `lte` | Порядок |
| `in` | Вхождение |
| `contains`, `icontains`, `startswith`, `endswith` | Подстроки |
| `isnull` | Наличие |
| `len`, `len_gt`, `len_lt` | Длина |
| `regex` | Регулярное выражение |

Опечатка ловится сразу, а не приводит к молчаливому нулю совпадений:

```python
>>> Reviews.objects.filter(lable="positive")
FieldError: Reviews has no field(s) lable. Available: id, label, stars, text.
```

Для того, что лукапами не выразить, используйте предикат:

```python
Reviews.objects.where(lambda r: detect_language(r.text) == "ru")
```

### Преобразование

```python
Reviews.objects.map(lambda r: {"text": r.text.lower(), "label": r.label})
Reviews.objects.annotate(length=lambda r: len(r.text))
Reviews.objects.only("text", "label")
Reviews.objects.defer("raw_html")
Reviews.objects.rename(body="text")
```

### Порядок, выборка, срезы

```python
Reviews.objects.order_by("-stars", "id")
Reviews.objects.shuffle(seed=0)      # с сидом, значит воспроизводимо
Reviews.objects.distinct("label")
Reviews.objects.take(100).skip(10)
Reviews.objects[10:20]
Reviews.objects.repeat(3)            # оверсэмплинг
```

`order_by` и `shuffle` должны увидеть все записи, поэтому материализуют данные.
Всё остальное стримит.

### Валидация

```python
Reviews.objects.validate()   # упасть на первой строке, нарушающей декларацию
Reviews.objects.clean()      # валидировать и привести к объявленным типам
```

`clean()` — это то, что превращает `"5"` из CSV в `5`.

### Терминальные операции

```python
Reviews.objects.count()
Reviews.objects.first()
Reviews.objects.get(id=42)                  # ошибка, если совпадений не ровно одно
Reviews.objects.exists()
Reviews.objects.all()                       # список записей
Reviews.objects.values("id", "label")
Reviews.objects.values_list("id", flat=True)
Reviews.objects.columns("text", "label")    # по колонкам
Reviews.objects.xy(features=["text"])       # (входы, цели)
Reviews.objects.to_pandas()                 # если установлен pandas

for batch in Reviews.objects.batch(32):
    ...
```

### Записи

Итерация выдаёт `Record`, который отвечает на оба стиля обращения:

```python
record = Reviews.objects.first()
record.text        # как атрибут
record["text"]     # как ключ
```

### Кеширование

```python
train = Reviews.objects.filter(label="positive").cache()
```

Вычислить один раз и переиспользовать между эпохами.

## Сплиты, которые не сдвигаются

```python
parts = Reviews.objects.split(train=0.8, val=0.1, test=0.1)
train, val, test = parts["train"], parts["val"], parts["test"]
```

Назначение хеширует `primary_key` каждой записи (или её содержимое, если ключ не
объявлен), а **не** позицию. Поэтому:

- добавление строк никогда не перемещает существующие между сплитами;
- дубликаты попадают в один сплит, что исключает утечку между train и test;
- разделение одинаково на всех машинах и во всех ранах.

Именно это свойство делает отложенную выборку по-прежнему честной через полгода.
Чтобы перетасовать намеренно, измените `Meta.split_salt`.

## Версионирование

Две идентичности, намеренно раздельные:

| Идентичность | Меняется, когда | Где |
|---|---|---|
| Отпечаток схемы | меняются объявленные поля | `_meta.fingerprint()` |
| Хеш содержимого | меняются строки | считается при материализации |

Заморозить текущее представление в нумерованный снапшот:

```python
version = Reviews.materialize(
    Reviews.objects.filter(label="positive"),
    notes="только положительные, для эксперимента с дисбалансом",
)
version.ref          # 'reviews.Reviews@v3'
version.row_count
version.content_hash
```

```bash
python manage.py dataset materialize reviews.Reviews
python manage.py dataset versions reviews.Reviews
```

Повторная материализация одинаковых данных **возвращает существующую версию**, а
не копит дубликаты, поэтому ночная задача дешёва, когда ничего не изменилось.
Передайте `force=True`, чтобы обойти это.

Прочитать снапшот обратно:

```python
Reviews.load_version()      # последнюю
Reviews.load_version(2)     # конкретную
```

Обучаться на замороженной версии, чтобы ран был воспроизводим даже если источник
сдвинется:

```bash
python manage.py train reviews.Sentiment --materialize
```

## Осмотр из терминала

```bash
python manage.py dataset list
python manage.py dataset show reviews.Reviews
python manage.py dataset head reviews.Reviews -n 20
python manage.py dataset validate reviews.Reviews
python manage.py dataset versions reviews.Reviews
```

## Переиспользование декларации

Абстрактные базовые классы отдают свои поля:

```python
class Timestamped(Dataset):
    created_at = fields.DateTimeField(null=True)

    class Meta:
        abstract = True


class Reviews(Timestamped):
    text = fields.TextField()
    # поля: created_at, text
```

Абстрактный датасет не регистрируется и никогда не появляется в админке.
