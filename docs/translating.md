# Translating the documentation

mlango aims to be approachable, and "approachable" depends on the language you
think in. Translations are welcome **one page at a time** — there is no need to
translate everything before opening a pull request.

## How it is wired

English is the source of truth. Translations sit beside each page with a locale
suffix:

```
docs/
├── index.md          ← English (source)
├── index.ru.md       ← Russian
├── index.es.md       ← Spanish, once someone writes it
├── tutorial.md
└── tutorial.ru.md
```

A page with no translation for the reader's language **falls back to English
automatically**, so a partial translation never leaves a broken navigation tree
or a dead link. That is what makes incremental contribution safe.

## Adding a page in an existing language

1. Copy the English page: `cp docs/models.md docs/models.ru.md`
2. Translate the prose. Leave code, identifiers, settings names, field names and
   command names in English — they are API, not text.
3. Build and look at it: `mkdocs serve` then open the language switcher.
4. Open a pull request titled `docs(i18n): translate models.md to ru`.

!!! warning "Give linked headings an explicit anchor"
    A heading written in a non-Latin script slugifies to a positional anchor like
    `_3`, which shifts as soon as a section is added above it. If anything links
    to the heading, pin the English page's anchor onto it:

    ```markdown
    ### Свои данные { #bringing-your-own-data }
    ```

    Then `[text](cli.md#bringing-your-own-data)` lands correctly in every
    language.

## Adding a new language

Add a locale block to `mkdocs.yml` under the `i18n` plugin:

```yaml
- locale: es
  name: Español
  build: true
  site_description: >-
    Un framework con baterías incluidas para machine learning,
    analítica y agentes LLM.
  nav_translations:
    Getting started: Primeros pasos
    Introduction: Introducción
    Tutorial: Tutorial
    Concepts: Conceptos
    # ... the rest of the navigation labels
```

Then translate at least `index.md` in the same pull request, so the language has
a landing page rather than an English one behind a Spanish switcher.

Current languages: **English** (source), **Русский**.

## What to translate, and what to leave

| Translate | Leave in English |
|---|---|
| Prose, headings, table headers | Code blocks and inline `code` |
| Explanations and rationale | Setting names (`METASTORE`, `DEFAULT_PROVIDER`) |
| Admonition text | Field and class names (`LabelField`, `Dataset`) |
| Image alt text | Command names (`manage.py train`) |
| Docstring prose *in examples*, if it aids understanding | Method names, `Meta` option names |

Translating an identifier makes the example stop working, and a reader who
copies it will be confused rather than helped.

## Terminology

Some terms have no good local equivalent and are better left in English, spelled
in the local script if that reads more naturally. Suggested handling:

| English | Guidance |
|---|---|
| dataset | Translate if your language has an established word; otherwise transliterate. |
| run | Translate. It means one execution, not a jog. |
| trace / span | Keep — these are observability terms of art. |
| fingerprint | Translate the meaning ("hash of the declaration"), not the metaphor. |
| queryset | Keep. It names a specific class. |
| batteries-included | Translate the idea, not the idiom. |

If a term resists translation, open a
[translation issue](https://github.com/DrobyshevDev/mlango/issues/new?template=translation.yml)
and discuss it before committing to a choice — consistency across pages matters
more than any single word.

## Keeping translations honest

When an English page changes materially, its translations go stale silently.
Two habits keep that manageable:

- A pull request that changes English prose should note which translations it
  invalidates, so a follow-up can be opened.
- A translator picking up a page should diff the English file's history since the
  translation's last update.

A slightly stale translation is better than none. A translation that documents
behaviour the framework no longer has is worse than none — if you find one,
please open an issue rather than leaving it.

## Building the docs locally

```bash
pip install mkdocs-material mkdocs-static-i18n
mkdocs serve            # http://127.0.0.1:8000
mkdocs build --strict   # what CI runs
```

`--strict` turns warnings into errors, so a broken link fails the build rather
than shipping.
