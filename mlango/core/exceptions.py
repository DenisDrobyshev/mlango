"""Exception hierarchy for mlango.

Mirrors the shape of ``django.core.exceptions``: a small set of well-named
errors that callers can catch precisely instead of parsing messages.
"""

from __future__ import annotations


class MlangoError(Exception):
    """Base class for every error raised by the framework."""


class ImproperlyConfigured(MlangoError):
    """mlango is somehow improperly configured (settings, apps, backends)."""


class AppRegistryNotReady(MlangoError):
    """The app registry was queried before ``mlango.setup()`` ran."""


class ValidationError(MlangoError):
    """A value failed field validation.

    ``errors`` maps a field name to the list of messages for that field. A
    non-field error is stored under the ``__all__`` key, as in Django.
    """

    NON_FIELD = "__all__"

    def __init__(self, message: str | dict[str, list[str]], field: str | None = None):
        if isinstance(message, dict):
            self.errors = {k: list(v) for k, v in message.items()}
            summary = "; ".join(f"{k}: {', '.join(v)}" for k, v in self.errors.items())
        else:
            key = field or self.NON_FIELD
            self.errors = {key: [message]}
            summary = message if field is None else f"{field}: {message}"
        super().__init__(summary)

    def merge(self, other: ValidationError) -> ValidationError:
        for key, messages in other.errors.items():
            self.errors.setdefault(key, []).extend(messages)
        return self


class FieldError(MlangoError):
    """A field was declared or referenced incorrectly."""


class DoesNotExist(MlangoError):
    """A lookup returned nothing where exactly one object was expected."""


class MultipleObjectsReturned(MlangoError):
    """A lookup returned several objects where exactly one was expected."""


class BackendNotAvailable(ImproperlyConfigured):
    """A trainer or provider backend is registered but its dependency is missing."""


class MigrationError(MlangoError):
    """A migration could not be built or applied."""


class RunError(MlangoError):
    """A training, evaluation or agent run failed."""


class ProviderError(MlangoError):
    """An LLM provider call failed."""
