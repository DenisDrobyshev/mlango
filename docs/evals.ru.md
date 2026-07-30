# Оценка

`Eval` объявляет, что значит «хорошо» и как это измерить. Запуск даёт
отслеживаемый ран плюс сохранённый результат по каждому кейсу, поэтому регрессия
— это дифф между двумя ранами, а не цифра, которую кто-то помнит.

```python
from mlango.evals import Eval, contains_all, token_f1, used_tool

from support.agents import Support
from support.datasets import SupportCases


class AnswerQuality(Eval):
    """Отвечает ли агент именно по документации?"""

    class Meta:
        dataset = SupportCases
        target = Support
        input_field = "question"
        expected_field = "answer"
        case_id_field = "id"
        scorers = {
            "overlap": token_f1,
            "cited": contains_all("docs/"),
            "searched": used_tool("search_docs"),
        }
        threshold = 0.6
```

```bash
python manage.py evaluate support.AnswerQuality
python manage.py evaluate support.AnswerQuality --show-failures
python manage.py evaluate support.AnswerQuality --min-pass-rate 0.9   # для CI
```

## Опции `Meta`

| Опция | Назначение |
|---|---|
| `dataset` | Откуда берутся кейсы |
| `target` | `Agent`, `Model` или любой вызываемый объект |
| `input_field` | Поле со входом, по умолчанию `"input"` |
| `expected_field` | Поле с эталонным ответом, если он есть |
| `case_id_field` | Поле, идентифицирующее кейс, чтобы результаты сходились между ранами |
| `scorers` | `{имя: скорер}` — каждый записывается отдельно |
| `threshold` | Средний балл, начиная с которого кейс считается пройденным |
| `max_cases` | Ограничить число кейсов |
| `fail_fast` | Остановиться на первой ошибке вместо записи и продолжения |

## Скореры

Скорер принимает `(output, expected)` и возвращает float в `[0, 1]` или bool.

| Скорер | Проверяет |
|---|---|
| `exact_match`, `iexact_match` | Равенство, опционально без учёта регистра |
| `contains` | Ожидаемый текст встречается в выводе |
| `contains_all(*needles)` | Долю присутствующих подстрок |
| `not_contains(*needles)` | Что ни одна не встречается |
| `regex_match(pattern)` | Совпадение с шаблоном |
| `json_equals` | Структурное совпадение двух JSON |
| `json_subset` | Долю ожидаемых ключей с совпадающими значениями |
| `numeric_close(tolerance)` | Числа в пределах допуска |
| `length_between(min, max)` | Длину вывода в диапазоне |
| `token_f1` | F1 по пересечению слов — дешёвая замена сходству |
| `used_tool(name)` | Что агент обратился к конкретному инструменту |
| `llm_judge(agent, rubric=...)` | Оценку другим агентом по рубрике |

Объявлять несколько — это и есть смысл: когда ран регрессирует, видно, **какой
именно критерий** сломался.

```python
scorers = {
    "correct": exact_match,
    "concise": length_between(maximum=400),
    "no_hedging": not_contains("как ИИ", "я не могу"),
}
```

### Оценивать поведение, а не только текст

`used_tool` получает весь `AgentRun`, а не его текст, поэтому оценка может
проверять, что агент *сделал*:

```python
scorers = {"searched": used_tool("search_docs")}
```

Любой скорер может подписаться на это, выставив `fn.wants_run = True`.

### Судья на LLM

```python
from mlango.evals import llm_judge

from support.agents import Judge

scorers = {
    "helpfulness": llm_judge(
        Judge(),
        rubric="Поставь 1, если ответ решает вопрос, и 0, если нет.",
    )
}
```

Судью просят вернуть только число, а всё непарсящееся получает 0, а не бросает
исключение — одно запутавшееся суждение не должно ронять весь прогон.

### Свой скорер

```python
def mentions_price(output, expected) -> bool:
    """True, когда ответ приводит сумму из эталона."""
    return any(t in str(output) for t in str(expected).split() if t.startswith("₽"))
```

## Своя логика

Переопределяйте хуки, когда декларации недостаточно:

```python
class AnswerQuality(Eval):
    class Meta:
        dataset = SupportCases
        target = Support
        input_field = "question"

    def cases(self):
        """Только сложные."""
        return SupportCases.objects.filter(difficulty="hard")

    def predict(self, case):
        """Запустить цель как вам угодно."""
        return Support().run(case.question, session_id=f"eval-{case.id}")

    def score(self, case, output) -> dict:
        return {"cited": case.source in output.output}

    def decide(self, scores) -> bool | None:
        """Превратить оценки скореров в pass или fail."""
        return scores["cited"]
```

## Чтение отчёта

```python
report = AnswerQuality.evaluate()

report.total
report.passed
report.pass_rate
report.mean_scores()      # {'overlap': 0.71, 'cited': 0.5}
report.failures()         # все непройденные кейсы
report.run.uuid
```

Каждый кейс сохраняется со входами, выводом, ожидаемым значением, оценками по
каждому скореру и — для агентов — трейсом, который он породил. Поэтому падение в
админке ведёт прямо к пошаговому трейсу, вызвавшему его.

## Ошибки записываются, а не роняют

Кейс, бросивший исключение, записывается со стектрейсом, и прогон продолжается.
Один сломанный кейс не должен выбрасывать результаты остальных девяноста девяти.
Поставьте `fail_fast = True`, если предпочитаете остановиться.

## В непрерывной интеграции

```bash
python manage.py evaluate support.AnswerQuality --min-pass-rate 0.9
```

Возвращает ненулевой код ниже порога, чего достаточно, чтобы уронить пайплайн.
Для обучения моделей аналогичный предохранитель — колбэк:

```python
from mlango.training import MetricThreshold

model.train(callbacks=[MetricThreshold("accuracy", minimum=0.85)])
```
