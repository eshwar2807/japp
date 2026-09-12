"""Put the package root on sys.path and isolate every test run from real data."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Must be set before `config.settings` is imported anywhere.
_TMP = Path(tempfile.mkdtemp(prefix="jp_test_"))
os.environ.setdefault("JP_DATA_DIR", str(_TMP / "data"))
os.environ.setdefault("JP_OUTPUT_DIR", str(_TMP / "output"))
os.environ.setdefault("JP_DB_PATH", str(_TMP / "data" / "test.db"))
os.environ.setdefault("JP_DB_URL", f"sqlite:///{_TMP / 'data' / 'test.db'}")
os.environ.setdefault("JP_KEY_PATH", str(_TMP / "data" / "vault.key"))
# No test may make a real API call, and none should change behaviour based on
# whether a key is present in the developer's environment.
os.environ.pop("ANTHROPIC_API_KEY", None)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from config import settings  # noqa: E402
from database.db_manager import DBManager  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    """A throwaway DBManager with its own database file and vault key."""
    from cryptography.fernet import Fernet

    from database.db_manager import DBManager

    return DBManager(db_url=f"sqlite:///{tmp_path/'t.db'}", key=Fernet.generate_key())


@pytest.fixture()
def profile():
    from engine.ats_optimizer import load_master_profile

    return load_master_profile()


@pytest.fixture()
def prof_fixture():
    """A fully-populated profile used by cross-module tests."""
    from engine.schemas import MasterProfile

    return MasterProfile.model_validate(
        {
            "contact": {
                "full_name": "Ada Lovelace",
                "preferred_name": "Ada",
                "email": "ada@example.com",
                "phone": "+1-555-010-0100",
                "location": {
                    "city": "Austin",
                    "state": "TX",
                    "country": "United States",
                    "postal_code": "78701",
                    "willing_to_relocate": True,
                },
                "links": {"linkedin": "https://linkedin.com/in/ada"},
            },
            "skills": {"hard": ["Python"], "tooling": ["PostgreSQL"], "soft": ["Mentoring"]},
            "experience": [
                {
                    "company": "Analytical Engines",
                    "title": "Principal Engineer",
                    "start_date": "2015-01",
                    "is_current": True,
                    "bullets": ["Cut p99 latency 60%."],
                }
            ],
            "legal": {
                "work_authorization_us": "Yes",
                "requires_sponsorship_now_or_future": "No",
                "desired_salary": "185000",
            },
            "voluntary_disclosures": {"gender": "Decline to self-identify"},
        }
    )


#: Meets the signup policy: length, mixed case, digit, symbol.
GOOD_PASSWORD = "Correct-Horse-9x!"

# The dashboard fixture lives here rather than in test_web.py so other
# suites can drive the app without importing a test module for its fixtures.
@pytest.fixture()
def web(tmp_path, monkeypatch):
    """A dashboard wired to a throwaway database and vault."""
    monkeypatch.setattr(settings, "DB_URL", f"sqlite:///{tmp_path/'web.db'}")
    monkeypatch.setattr(settings, "KEY_PATH", tmp_path / "vault.key")
    monkeypatch.setattr(settings, "SECRET_KEY", "test-secret-key-not-for-real-use")
    monkeypatch.setattr(settings, "COOKIE_SECURE", False)
    monkeypatch.setattr(settings, "ALLOW_SIGNUP", True)
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path / "out")
    # Tests must not behave differently because a developer happens to have a
    # key in their environment: the per-user vault is the only source here.
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", None)

    # Same key resolution as the app, so both share one vault.
    db = DBManager(db_url=settings.DB_URL)

    import web.deps as deps

    deps.get_db.cache_clear()
    deps.get_sessions.cache_clear()
    deps.get_limiter.cache_clear()
    # The worker is cached too. Without clearing it, every web test reuses one
    # dispatcher bound to a database from the first test that ran, polling
    # several times a second for the rest of the suite and competing with the
    # concurrency tests.
    deps.get_worker.cache_clear()
    monkeypatch.setattr(deps, "get_db", lambda: db)

    from web.app import create_app

    app = create_app()
    app.dependency_overrides[deps.get_db] = lambda: db
    client = TestClient(app, follow_redirects=False)
    client.db = db
    yield client

    # Stop whatever the app's lifespan started, so no dispatcher outlives the
    # test that created it.
    worker = deps.get_worker()
    worker.stop(timeout=2)
    deps.get_worker.cache_clear()

def signup(client, email: str, password: str = GOOD_PASSWORD):
    client.get("/login")  # obtain a CSRF cookie
    token = client.cookies.get("jp_csrf")
    return client.post(
        "/signup",
        data={"email": email, "password": password, "confirm": password, "csrf_token": token},
    )
