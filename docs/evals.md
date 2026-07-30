# Evaluations

An `Eval` declares what "good" means and how to measure it. Running one produces
a tracked run plus a stored result per case, so a regression is a diff between
two runs rather than a number someone remembers.

```python
from mlango.evals import Eval, contains_all, token_f1, used_tool

from support.agents import Support
from support.datasets import SupportCases


class AnswerQuality(Eval):
    """Does the agent answer from the docs?"""

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
python manage.py evaluate support.AnswerQuality --min-pass-rate 0.9   # for CI
```

## `Meta` options

| Option | Purpose |
|---|---|
| `dataset` | Where the cases come from |
| `target` | An `Agent`, a `Model`, or any callable |
| `input_field` | Field holding the input, default `"input"` |
| `expected_field` | Field holding the reference answer, if there is one |
| `case_id_field` | Field identifying a case, so results line up across runs |
| `scorers` | `{name: scorer}` — each is recorded separately |
| `threshold` | Mean score at or above which a case passes |
| `max_cases` | Cap the number of cases |
| `fail_fast` | Stop at the first error instead of recording and continuing |

## Scorers

A scorer takes `(output, expected)` and returns a float in `[0, 1]` or a bool.

| Scorer | Checks |
|---|---|
| `exact_match`, `iexact_match` | Equality, optionally case-insensitive |
| `contains` | The expected text appears in the output |
| `contains_all(*needles)` | Fraction of the needles present |
| `not_contains(*needles)` | None of them appear |
| `regex_match(pattern)` | A pattern matches |
| `json_equals` | Two JSON payloads match structurally |
| `json_subset` | Fraction of the expected keys present with matching values |
| `numeric_close(tolerance)` | Numbers within a tolerance |
| `length_between(min, max)` | Output length is in range |
| `token_f1` | Word-overlap F1 — a cheap stand-in for similarity |
| `used_tool(name)` | The agent reached for a particular tool |
| `llm_judge(agent, rubric=...)` | Another agent scores against a rubric |

Declaring several is the point: when a run regresses, you can see *which
criterion* broke.

```python
scorers = {
    "correct": exact_match,
    "concise": length_between(maximum=400),
    "no_hedging": not_contains("as an AI", "I cannot"),
}
```

### Scoring behaviour, not just text

`used_tool` receives the whole `AgentRun` rather than its text, so an eval can
assert on what the agent *did*:

```python
scorers = {"searched": used_tool("search_docs")}
```

Any scorer can opt in by setting `fn.wants_run = True`.

### An LLM judge

```python
from mlango.evals import llm_judge

from support.agents import Judge

scorers = {
    "helpfulness": llm_judge(
        Judge(),
        rubric="Score 1 if the answer resolves the question, 0 if it does not.",
    )
}
```

The judge is asked for a bare number, and anything unparseable scores 0 rather
than raising — one confused judgement should not abort a sweep.

### Writing your own

```python
def mentions_price(output, expected) -> bool:
    """True when the answer quotes a figure in the reference."""
    return any(token in str(output) for token in str(expected).split() if token.startswith("$"))
```

## Custom logic

Override the hooks when the declaration is not enough:

```python
class AnswerQuality(Eval):
    class Meta:
        dataset = SupportCases
        target = Support
        input_field = "question"

    def cases(self):
        """Only the hard ones."""
        return SupportCases.objects.filter(difficulty="hard")

    def predict(self, case):
        """Run the target however you like."""
        return Support().run(case.question, session_id=f"eval-{case.id}")

    def score(self, case, output) -> dict:
        return {"cited": case.source in output.output}

    def decide(self, scores) -> bool | None:
        """Turn per-scorer values into pass or fail."""
        return scores["cited"]
```

## Reading the report

```python
report = AnswerQuality.evaluate()

report.total
report.passed
report.pass_rate
report.mean_scores()      # {'overlap': 0.71, 'cited': 0.5}
report.failures()         # every case that did not pass
report.run.uuid
```

Each case is persisted with its inputs, output, expected value, per-scorer
scores and — for agent targets — the trace it produced. So a failure in the
admin links straight to the step-by-step trace that caused it.

## Errors are recorded, not fatal

A case that raises is recorded with its traceback and the suite continues. One
broken case should not throw away the results of the other ninety-nine. Set
`fail_fast = True` when you would rather stop.

## In continuous integration

```bash
python manage.py evaluate support.AnswerQuality --min-pass-rate 0.9
```

Exits non-zero below the threshold, which is enough to fail a pipeline. For
model training, the equivalent guardrail is a callback:

```python
from mlango.training import MetricThreshold

model.train(callbacks=[MetricThreshold("accuracy", minimum=0.85)])
```
