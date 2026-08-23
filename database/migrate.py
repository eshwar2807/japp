"""Additive schema migration for SQLite.

`Base.metadata.create_all()` creates tables that do not exist. It does not
touch tables that do. On a fresh database that is invisible; against a volume
holding an earlier schema it means every column added since the last deploy is
missing, and the app dies on the first query that selects it.

This closes that gap for the only kind of change this project makes: adding
columns. It compares the models to the live schema and issues `ALTER TABLE ADD
COLUMN` for anything absent.

Deliberately additive only. Dropping or retyping a column in SQLite means
rebuilding the table, which risks data on a change that has never been needed
here. If that day comes, this is the point to bring in Alembic rather than to
extend this.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.schema import CreateColumn

log = logging.getLogger(__name__)


def _column_clause(engine: Any, column: Any) -> str:
    """The `name TYPE [NOT NULL DEFAULT x]` fragment for ADD COLUMN."""
    clause = str(CreateColumn(column).compile(engine))

    # SQLite refuses a NOT NULL column with no default on an existing table,
    # since existing rows would have nothing to put in it.
    if not column.nullable and "DEFAULT" not in clause.upper():
        default = getattr(column.default, "arg", None)
        if default is None or callable(default):
            # Fall back to a type-appropriate zero value; the application's own
            # default takes over for every row written afterwards.
            python_type = getattr(column.type, "python_type", str)
            try:
                sample = python_type()
            except (TypeError, NotImplementedError):
                sample = ""
            default = sample
        literal = (
            "1" if default is True else
            "0" if default is False else
            str(default) if isinstance(default, (int, float)) else
            f"'{str(default)}'"
        )
        clause = f"{clause} DEFAULT {literal}"
    return clause


def add_missing_columns(engine: Any, metadata: Any) -> list[str]:
    """Bring existing tables up to the models. Returns what was added."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    added: list[str] = []

    for table_name, table in metadata.tables.items():
        if table_name not in existing_tables:
            continue  # create_all handles brand-new tables
        present = {col["name"] for col in inspector.get_columns(table_name)}

        for column in table.columns:
            if column.name in present:
                continue
            clause = _column_clause(engine, column)
            statement = f"ALTER TABLE {table_name} ADD COLUMN {clause}"
            try:
                with engine.begin() as connection:
                    connection.execute(text(statement))
            except Exception:
                # One unmigratable column must not stop the rest, and must not
                # take the process down on boot.
                log.exception("Could not add %s.%s", table_name, column.name)
                continue
            added.append(f"{table_name}.{column.name}")
            log.info("Schema: added column %s.%s", table_name, column.name)

    return added
