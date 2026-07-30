# Ноутбуки

Вся остальная документация предполагает `manage.py`. Он не обязателен.

```python
import mlango
mlango.notebook()
```

Это вся настройка. Объявите в следующей ячейке датасет и модель, вызовите
`train()` — и ран запишется с сидом, метриками, артефактами и git-коммитом,
ровно как внутри проекта.

## Полная сессия

```python
import mlango
mlango.notebook()
```

```python
from mlango.core import fields
from mlango.data import Dataset, CSVSource

class Reviews(Dataset):
    id = fields.IntegerField()
    text = fields.TextField()
    label = fields.LabelField(["neg", "pos"])

    class Meta:
        source = CSVSource("reviews.csv")
        primary_key = "id"

Reviews.objects.filter(label="pos").count()
```

```python
from mlango.training import Model

class Sentiment(Model):
    C = fields.FloatField(default=1.0, tunable=True)

    class Meta:
        dataset = Reviews
        trainer = "sklearn"
        task = "classification"
        features = ["text"]

    def build(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        return make_pipeline(TfidfVectorizer(), LogisticRegression(C=self.C))

run = Sentiment().train()
run.refresh().summary
```

```python
Sentiment.load().predict("понравилось от начала до конца")
```

Никакого `migrate`: таблицы метастора создаются при первой же записи.

## Повторный запуск ячейки

Повторный запуск ячейки выполняет тело класса заново, то есть объявляет класс
второй раз. mlango заменяет предыдущее объявление и продолжает работу — поэтому
поправить опцию `Meta` и перезапустить ячейку это нормальный рабочий приём.

Проверка уникальности, которая не даёт двум приложениям занять одну метку,
никуда не делась. Она срабатывает, только когда объявления пришли из **разных**
модулей, — а это настоящий конфликт, а не правка.

## Откуда исчез шаблонный код

`mlango.notebook()` — это `settings.configure()` со значениями, подобранными под
эту ситуацию, и следом `mlango.setup()`:

| Настройка | Что берётся | Зачем |
|---|---|---|
| `BASE_DIR` | рабочий каталог | Раны и артефакты ложатся рядом с ноутбуком |
| `METASTORE` | `sqlite:///mlango.db` | Один файл, без сервера |
| `STORAGE` | локальная папка `artifacts/` | Чекпоинты там, где их найдёшь |
| `DEFAULT_PROVIDER` | `echo` | Агенты работают без API-ключа |

Любое можно переопределить:

```python
mlango.notebook(SEED=7, DEFAULT_PROVIDER="anthropic")
mlango.notebook(base_dir="/data/experiments")
```

Вызвать дважды безвредно — а это важно, потому что первую ячейку перезапускают
чаще всего.

## Открыть админку на том, что сделано в ноутбуке

Записанная база — обычный метастор mlango, поэтому проект, направленный в тот же
каталог, покажет все раны, графики и артефакты из ноутбука:

```bash
mlango startproject dashboard --bare
cd dashboard
# направьте METASTORE на mlango.db ноутбука, затем
python manage.py runserver
```

Это и есть довод в пользу фреймворка в ноутбуке. Исследовательская работа обычно
испаряется; здесь она попадает в то же хранилище, что и всё остальное, и ран,
давший цифру полгода назад, по-прежнему говорит, какие данные читал.

## Переезд из ноутбука в проект

Когда наработкам пора обрести дом, декларации переезжают без изменений: тела
классов — это уже то, что лежит в `datasets.py` и `models.py` проекта.

```bash
mlango startproject myproject --bare
cd myproject
python manage.py startapp reviews
```

Вставьте классы в `reviews/datasets.py` и `reviews/models.py`, добавьте
`"reviews"` в `INSTALLED_APPS` и выполните `manage.py makemigrations`, чтобы
зафиксировать схему. В самих декларациях не меняется ничего — ради этого они и
декларации.
