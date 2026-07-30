"""Schema migrations for declared datasets, models, agents and evals.

Generated migration files import from this module, so it re-exports both the
``Migration`` base class and every operation under one name::

    from mlango import migrations

    class Migration(migrations.Migration):
        operations = [migrations.AddField("Reviews", "language", fields.CharField())]
"""

from mlango.migrations.autodetector import MigrationAutodetector
from mlango.migrations.executor import MigrationExecutor
from mlango.migrations.loader import MigrationLoader
from mlango.migrations.migration import Migration
from mlango.migrations.operations import (
    AddField,
    AlterField,
    AlterOptions,
    CreateObject,
    DeleteObject,
    Operation,
    RemoveField,
    RenameObject,
    RunPython,
)
from mlango.migrations.state import ObjectState, ProjectState
from mlango.migrations.writer import MigrationWriter, build_filename, next_migration_number

__all__ = [
    "Migration",
    "Operation",
    "CreateObject",
    "DeleteObject",
    "RenameObject",
    "AddField",
    "RemoveField",
    "AlterField",
    "AlterOptions",
    "RunPython",
    "MigrationAutodetector",
    "MigrationExecutor",
    "MigrationLoader",
    "MigrationWriter",
    "ObjectState",
    "ProjectState",
    "build_filename",
    "next_migration_number",
]
