"""Engine and session management for the metastore.

The engine is created lazily from ``settings.METASTORE`` and cached per URL, so
importing this module has no side effects and tests can swap the database by
reconfiguring settings.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from mlango.metastore.models import Base

_engines: dict[str, Engine] = {}
_sessionmakers: dict[str, sessionmaker[Session]] = {}
_ensured: set[str] = set()


def metastore_url() -> str:
    from mlango.conf import settings

    url = str(settings.METASTORE.get("URL", "sqlite:///mlango.db"))
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        # Resolve relative SQLite paths against BASE_DIR so the database does
        # not follow the shell's working directory around.
        rel = url[len("sqlite:///") :]
        if rel and rel != ":memory:" and not os.path.isabs(rel):
            url = "sqlite:///" + os.path.join(str(settings.BASE_DIR), rel).replace("\\", "/")
    return url


def get_engine(url: str | None = None) -> Engine:
    from mlango.conf import settings

    url = url or metastore_url()
    if url in _engines:
        return _engines[url]

    options: dict[str, Any] = {
        "echo": bool(settings.METASTORE.get("ECHO", False)),
        "future": True,
    }
    if url.startswith("sqlite"):
        # A training loop writing metrics while the admin reads them is the
        # normal case, so allow cross-thread use and turn on WAL below.
        options["connect_args"] = {"check_same_thread": False}
    else:
        options["pool_pre_ping"] = bool(settings.METASTORE.get("POOL_PRE_PING", True))

    engine = create_engine(url, **options)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - driver hook
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    _engines[url] = engine
    _sessionmakers[url] = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return engine


def get_sessionmaker(url: str | None = None) -> sessionmaker[Session]:
    url = url or metastore_url()
    if url not in _sessionmakers:
        get_engine(url)
    return _sessionmakers[url]


def ensure_schema(url: str | None = None) -> None:
    """Create the metastore tables once per process.

    These tables are framework-owned and never change shape at a user's
    request, so there is nothing to be gained by making people run ``migrate``
    before their first ``materialize()``. Declarative migrations are a separate
    concern and still explicit.
    """
    url = url or metastore_url()
    if url in _ensured:
        return
    create_all(url)
    _ensured.add(url)


def new_session(url: str | None = None) -> Session:
    """A session the caller owns and must close."""
    ensure_schema(url)
    return get_sessionmaker(url)()


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on error."""
    session = new_session(url)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all(url: str | None = None) -> None:
    """Create every metastore table that does not exist yet."""
    Base.metadata.create_all(get_engine(url))


def drop_all(url: str | None = None) -> None:
    """Drop every metastore table. Destructive; used by ``flush`` and tests."""
    Base.metadata.drop_all(get_engine(url))


def dispose_all() -> None:
    """Close every pooled connection — call before deleting a SQLite file."""
    for engine in _engines.values():
        engine.dispose()
    _engines.clear()
    _sessionmakers.clear()
    _ensured.clear()


def table_names(url: str | None = None) -> list[str]:
    from sqlalchemy import inspect

    return sorted(inspect(get_engine(url)).get_table_names())


def metastore_ready(url: str | None = None) -> bool:
    """True when the core tables exist — i.e. ``migrate`` has been run."""
    try:
        return "mlango_runs" in table_names(url)
    except Exception:
        return False
