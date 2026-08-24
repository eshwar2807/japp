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
