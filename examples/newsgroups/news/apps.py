"""App configuration for the news app."""

from mlango.core import AppConfig


class NewsConfig(AppConfig):
    name = "news"
    verbose_name = "News"

    def ready(self) -> None:
        """Runs once every app is loaded — wire signal receivers here."""
