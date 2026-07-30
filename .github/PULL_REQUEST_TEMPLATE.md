## What this changes

<!-- One or two sentences. What is different for a user after this lands? -->

## Why

<!-- The problem it solves. Link an issue if there is one: Fixes #123 -->

## Checklist

- [ ] `ruff check mlango tests` passes
- [ ] `ruff format mlango tests` leaves nothing to change
- [ ] `pytest -q` passes
- [ ] Tests cover the new behaviour, named after the guarantee they protect
- [ ] `CHANGELOG.md` updated under `## Unreleased`
- [ ] Docs updated if this changes an API or a default
- [ ] No new required dependency in the core (optional ones go behind an extra)

If this touches the scaffold, settings, or any command, confirm the four-command
path still works:

- [ ] `mlango startproject demo && cd demo && python manage.py migrate && python manage.py train demo.Sentiment && python manage.py runserver`

## Notes for the reviewer

<!-- Anything non-obvious: a trade-off you made, an alternative you rejected,
     a place you would like a second opinion. -->
