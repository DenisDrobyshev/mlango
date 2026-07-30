# Введение

**mlango — фреймворк «с батарейками» для машинного обучения, аналитики и
LLM-агентов, построенный на философии Django.**

ML-проекты имеют свойство превращаться в свалку скриптов: один загружает данные,
другой обучает, где-то лежит ноутбук, который выдал цифру для презентации за
прошлый квартал, и папка `checkpoints/`, которую уже никто не сопоставит с
коммитом.

В веб-разработке была ровно такая же проблема. Django её решил — не тем, что
выпустил более удачную библиотеку, а тем, что стал **фреймворком**: структура
проекта, модуль настроек, декларативные классы, миграции, автоматическая
админка и `manage.py`, который всё это связывает. mlango применяет этот ответ к
машинному обучению.

Вы объявляете датасеты, модели, агентов и оценки. Фреймворк их запускает,
версионирует, записывает и показывает вам.

Две вещи отличают это от инструментов, которые у вас уже есть. Одно тело класса
разом становится страницей админки, документированным эндпоинтом API, миграцией
и целью для CLI — связывать не нужно ничего. И **агенты — полноправное семейство
рядом с моделями**: общий метастор, общая админка, общая система оценки, а не
отдельный стек.

!!! question "Уже пользуетесь MLflow, Kedro, W&B или LangChain?"
    [**mlango и альтернативы**](comparison.md) — где находится каждый из них,
    что действительно пересекается и, сказанное прямо, когда mlango не подходит.

## Установка

```bash
pip install "mlango[sklearn]"
```

| Extra | Что добавляет |
|---|---|
| `sklearn` | тренер на scikit-learn |
| `torch` | тренер на PyTorch |
| `anthropic` | провайдер Claude для агентов |
| `dev` | pytest, ruff, mypy |
| `all` | всё перечисленное |

## Пять минут с нуля

```bash
mlango startproject myproject
cd myproject
python manage.py migrate
python manage.py train demo.Sentiment
python manage.py runserver
```

Откройте <http://127.0.0.1:8000/admin/>.

В отличие от пустого каркаса, свежий проект mlango **уже содержит рабочий
пример** — датасет, обученную модель с настоящими метриками, агента с
инструментом и набор оценок. В админке с первого взгляда есть что смотреть, и
это разница между «понятно, как это работает» и «а дальше-то что?».

Чтобы дойти до этой точки, настраивать ничего не нужно: метастор — SQLite,
артефакты пишутся в локальную папку, а агенты работают на офлайн-провайдере,
которому не нужен API-ключ.

!!! tip "Нужен пустой проект?"
    `mlango startproject myproject --bare` создаст проект без демо-приложения.

## Как выглядит декларация

=== "Датасет"

    ```python
    # reviews/datasets.py
    from mlango.core import fields
    from mlango.data import Dataset, JSONLSource

    class Reviews(Dataset):
        """Отзывы покупателей о товарах."""

        id = fields.IntegerField()
        text = fields.TextField()
        label = fields.LabelField(["negative", "positive"])

        class Meta:
            source = JSONLSource("data/reviews.jsonl")
            primary_key = "id"
    ```

=== "Модель"

    ```python
    # reviews/models.py
    from mlango.core import fields
    from mlango.training import Model
    from reviews.datasets import Reviews

    class Sentiment(Model):
        """TF-IDF и логистическая регрессия."""

        max_features = fields.IntegerField(default=20_000, tunable=True)
        C = fields.FloatField(default=1.0, min_value=0.0, tunable=True)

        class Meta:
            dataset = Reviews
            trainer = "sklearn"
            task = "classification"
            features = ["text"]

        def build(self):
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            return make_pipeline(
                TfidfVectorizer(max_features=self.max_features),
                LogisticRegression(C=self.C),
            )
    ```

=== "Агент"

    ```python
    # support/agents.py
    from mlango.agents import Agent, BufferMemory, tool

    @tool
    def search_docs(query: str, limit: int = 5) -> list[str]:
        """Поиск по документации продукта.

        Args:
            query: Что искать.
            limit: Максимум результатов.
        """
        return retrieve(query, limit)

    class Support(Agent):
        """Отвечает на вопросы о продукте по документации."""

        class Meta:
            model = "claude-opus-5"
            system = "Ты инженер поддержки. Ссылайся на использованные разделы документации."
            tools = [search_docs]
            memory = BufferMemory(k=20)
    ```

=== "Оценка"

    ```python
    # support/evals.py
    from mlango.evals import Eval, contains_all, token_f1

    class AnswerQuality(Eval):
        """Отвечает ли агент именно по документации?"""

        class Meta:
            dataset = SupportCases
            target = Support
            input_field = "question"
            expected_field = "answer"
            scorers = {"overlap": token_f1, "cited": contains_all("docs/")}
            threshold = 0.6
    ```

Дальше:

```bash
python manage.py train reviews.Sentiment -p C=2.0
```

Эта одна команда находит ваш класс, открывает отслеживаемый ран, ставит сид всем
генераторам случайных чисел, детерминированно режет данные на выборки, вызывает
ваш `build()`, ведёт цикл обучения, пишет метрики, фиксирует git-коммит,
сохраняет артефакт и регистрирует версию модели, готовую к промоуту. Вы написали
`build()` и четыре объявления полей.

## Куда дальше

- **[Учебник](tutorial.md)** — собрать проект от начала до конца
- **[Концепции](concepts.md)** — как связаны декларации, `_meta` и приложения
- **[Датасеты](datasets.md)**, **[Модели](models.md)**, **[Агенты](agents.md)**, **[Оценка](evals.md)** — четыре строительных блока
- **[Админка](admin.md)** — интерфейс, который не пришлось писать
