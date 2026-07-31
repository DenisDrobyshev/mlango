"""The metastore's own schema: creation, and upgrading it in place."""

from __future__ import annotations

import logging

import sqlalchemy as sa

from mlango.metastore.models import Base, ModelVersion
from mlango.metastore.session import (
    align_schema,
    create_all,
    ensure_schema,
    get_engine,
    metastore_ready,
    session_scope,
    table_names,
)


def _rebuild_without(engine: sa.Engine, table: sa.Table, column: str) -> None:
    """Replace ``table`` with the shape it had before ``column`` was added.

    Copying the columns rather than hardcoding a CREATE TABLE keeps this test
    honest about whichever column the suite points it at, so it still means
    something after the next release adds another one.
    """
    staging = sa.MetaData()
    sa.Table(
        table.name,
        staging,
        *[
            sa.Column(c.name, c.type, primary_key=c.primary_key, nullable=c.nullable)
            for c in table.columns
            if c.name != column
        ],
    )
    with engine.begin() as connection:
        connection.execute(sa.text(f"DROP TABLE {table.name}"))
    staging.create_all(engine)


class TestSchema:
    def test_tables_are_created_on_demand(self, project):
        ensure_schema()
        assert "mlango_runs" in table_names()
        assert metastore_ready()

    def test_an_up_to_date_database_is_left_alone(self, project):
        ensure_schema()
        assert align_schema() == []


class TestUpgrades:
    def test_a_column_added_by_a_newer_release_is_backfilled(self, project):
        """The upgrade path: an old database, a new mlango, no data loss."""
        ensure_schema()
        table = Base.metadata.tables["mlango_model_versions"]
        _rebuild_without(get_engine(), table, "importances")

        with session_scope() as session:
            session.add(ModelVersion(label="old.Model", version=1, fingerprint="abc"))

        assert align_schema() == ["mlango_model_versions.importances"]

        with session_scope() as session:
            row = session.execute(sa.select(ModelVersion)).scalar_one()
            assert row.label == "old.Model", "the existing row survived the upgrade"
            assert row.importances is None
            row.importances = {"text": 1.0}

        with session_scope() as session:
            assert session.execute(sa.select(ModelVersion)).scalar_one().importances == {
                "text": 1.0
            }

    def test_running_it_twice_adds_nothing(self, project):
        ensure_schema()
        _rebuild_without(get_engine(), Base.metadata.tables["mlango_model_versions"], "notes")
        assert align_schema() == ["mlango_model_versions.notes"]
        assert align_schema() == []

    def test_a_not_null_column_backfills_its_declared_default(self, project):
        """``notes`` is NOT NULL with a default of "" — old rows must get it."""
        ensure_schema()
        _rebuild_without(get_engine(), Base.metadata.tables["mlango_model_versions"], "notes")
        with get_engine().begin() as connection:
            connection.execute(
                sa.text(
                    "INSERT INTO mlango_model_versions "
                    "(label, version, fingerprint, params, metrics, stage, created_at) "
                    "VALUES ('old.Model', 1, 'abc', '{}', '{}', 'none', '2026-01-01 00:00:00')"
                )
            )

        align_schema()

        with session_scope() as session:
            assert session.execute(sa.select(ModelVersion)).scalar_one().notes == ""

    def test_a_missing_table_is_left_to_create_all(self, project):
        """Aligning columns must not trip over a table that is not there yet."""
        ensure_schema()
        with get_engine().begin() as connection:
            connection.execute(sa.text("DROP TABLE mlango_metrics"))

        assert align_schema() == []

        create_all()
        assert "mlango_metrics" in table_names()

    def test_a_not_null_column_is_reported_rather_than_guessed(self, project, caplog):
        """Inventing a value for a NOT NULL column is a migration, not a startup step."""
        ensure_schema()
        table = Base.metadata.tables["mlango_model_versions"]
        _rebuild_without(get_engine(), table, "fingerprint")

        with caplog.at_level(logging.WARNING, logger="mlango.metastore"):
            added = align_schema()

        assert added == []
        assert "fingerprint" in caplog.text
        assert "NOT NULL" in caplog.text
