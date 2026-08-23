"""Schema evolution against a database written by an earlier version.

This reproduces the failure that took the deployed app down: create_all()
leaves existing tables alone, so every column added since the last deploy was
missing and the first query selecting one killed the process on boot.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text

from database.migrate import add_missing_columns
from database.models import Base


def _old_database(path) -> str:
    """A database at the current schema, then rolled back a few columns."""
    url = f"sqlite:///{path}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)

    # SQLite cannot drop columns on older versions, so rebuild run_jobs without
    # `priority` to imitate a volume written before that column existed.
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE run_jobs RENAME TO run_jobs_old"))
        connection.execute(text("""
            CREATE TABLE run_jobs (
                id INTEGER NOT NULL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                application_id INTEGER,
                batch_id VARCHAR(32),
                kind VARCHAR(16),
                job_url VARCHAR(1024),
                job_description TEXT,
                status VARCHAR(9),
                block_mode VARCHAR(13),
                blocking_action_id INTEGER,
                holds_browser BOOLEAN,
                message TEXT,
                attempts INTEGER,
                position INTEGER,
                created_at DATETIME,
                started_at DATETIME,
                blocked_at DATETIME,
                finished_at DATETIME
            )"""))
        connection.execute(text("DROP TABLE run_jobs_old"))
        connection.execute(text("ALTER TABLE users DROP COLUMN pending_profile_json"))
        connection.execute(text("ALTER TABLE users DROP COLUMN notify_mode"))
    engine.dispose()
    return url


def test_an_old_database_is_missing_the_new_columns(tmp_path):
    """Confirms the fixture really does reproduce the deployed state."""
    url = _old_database(tmp_path / "old.db")
    engine = create_engine(url)
    columns = {c["name"] for c in inspect(engine).get_columns("run_jobs")}
    assert "priority" not in columns


def test_missing_columns_are_added(tmp_path):
    url = _old_database(tmp_path / "old.db")
    engine = create_engine(url)

    added = add_missing_columns(engine, Base.metadata)

    assert "run_jobs.priority" in added
    assert "users.pending_profile_json" in added
    assert "users.notify_mode" in added

    inspector = inspect(engine)
    assert "priority" in {c["name"] for c in inspector.get_columns("run_jobs")}
    assert "notify_mode" in {c["name"] for c in inspector.get_columns("users")}


def test_migration_is_idempotent(tmp_path):
    url = _old_database(tmp_path / "old.db")
    engine = create_engine(url)

    assert add_missing_columns(engine, Base.metadata)
    assert add_missing_columns(engine, Base.metadata) == []


def test_existing_rows_survive_and_get_the_default(tmp_path):
    """A migration that dropped data would be worse than the bug."""
    url = _old_database(tmp_path / "old.db")
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO run_jobs (id, user_id, kind, job_url, status, message, "
            "attempts, position) VALUES (1, 1, 'tailor', 'https://x.com/1', "
            "'Queued', 'existing row', 0, 1)"))

    add_missing_columns(engine, Base.metadata)

    with engine.begin() as connection:
        row = connection.execute(text(
            "SELECT job_url, message, priority FROM run_jobs WHERE id = 1")).one()
    assert row[0] == "https://x.com/1"
    assert row[1] == "existing row"
    assert row[2] in (0, False)          # NOT NULL column got its default


def test_dbmanager_opens_an_old_database_without_crashing(tmp_path):
    """The end-to-end case: boot against a volume from an earlier version."""
    from cryptography.fernet import Fernet

    from database.db_manager import DBManager

    url = _old_database(tmp_path / "old.db")
    db = DBManager(db_url=url, key=Fernet.generate_key())

    # The query that killed the deployed app on startup.
    assert db.reset_stale_running_jobs() == 0
    user = db.create_user("ada@example.com", "$argon2id$fake")
    job = db.enqueue_job(user.id, kind="tailor", job_url="https://x.com/1")
    assert job.priority is False


def test_a_fresh_database_needs_no_migration(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'new.db'}")
    Base.metadata.create_all(engine)
    assert add_missing_columns(engine, Base.metadata) == []
