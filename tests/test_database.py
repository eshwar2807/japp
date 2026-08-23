"""Module 1 validation: models, CRUD, Fernet vault, feedback mining."""

import string

import pytest

from database.db_manager import DBManager, generate_password, normalize_domain
from database.models import ApplicationStatus


# ---------------- password generation ----------------


def test_generated_password_length_and_classes():
    pw = generate_password(16)
    assert len(pw) == 16
    assert any(c.isupper() for c in pw)
    assert any(c.islower() for c in pw)
    assert any(c.isdigit() for c in pw)
    assert any(c in "!@#$%^&*-_=+?" for c in pw)
    assert all(c in string.printable for c in pw)


def test_generated_passwords_are_unique():
    assert len({generate_password() for _ in range(200)}) == 200


def test_short_password_rejected():
    with pytest.raises(ValueError):
        generate_password(4)


# ---------------- domain normalization ----------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://boards.greenhouse.io/acme/jobs/123", "boards.greenhouse.io"),
        ("https://www.myworkdayjobs.com/en-US/acme", "myworkdayjobs.com"),
        ("jobs.lever.co/foo", "jobs.lever.co"),
        ("", ""),
    ],
)
def test_normalize_domain(raw, expected):
    assert normalize_domain(raw) == expected


# ---------------- credential vault ----------------


def test_password_roundtrips_and_is_never_stored_plaintext(db: DBManager):
    cred = db.upsert_credential("https://acme.wd1.myworkdayjobs.com/x", "me@example.com", "hunter2!A")
    assert b"hunter2" not in cred.encrypted_password
    assert db.get_password("acme.wd1.myworkdayjobs.com", "me@example.com") == "hunter2!A"


def test_get_or_create_credential_is_idempotent(db: DBManager):
    _, pw1, created1 = db.get_or_create_credential("greenhouse.io", "me@example.com")
    _, pw2, created2 = db.get_or_create_credential("greenhouse.io", "me@example.com")
    assert created1 is True and created2 is False
    assert pw1 == pw2 and len(pw1) == 16


def test_wrong_key_cannot_decrypt(db: DBManager, tmp_path):
    from cryptography.fernet import Fernet

    db.upsert_credential("acme.com", "me@example.com", "secret")
    other = DBManager(db_url=db.db_url, key=Fernet.generate_key())
    with pytest.raises(RuntimeError, match="vault key"):
        other.get_password("acme.com", "me@example.com")


# ---------------- applications ----------------


def _make_app(db, **kw):
    defaults = dict(
        company="Acme",
        role_title="Backend Engineer",
        job_url="https://boards.greenhouse.io/acme/jobs/1",
        job_description="Python, PostgreSQL, AWS",
        match_score=87.5,
    )
    return db.create_application(**{**defaults, **kw})


def test_create_and_fetch_application(db: DBManager):
    app = _make_app(db)
    assert app.id is not None
    assert app.status is ApplicationStatus.DRAFT
    assert app.portal_domain == "boards.greenhouse.io"
    assert db.get_application(app.id).company == "Acme"


def test_mark_submitted_sets_timestamp(db: DBManager):
    app = _make_app(db)
    updated = db.mark_submitted(app.id)
    assert updated.status is ApplicationStatus.APPLIED
    assert updated.submitted_at is not None


def test_update_rejects_unknown_field(db: DBManager):
    app = _make_app(db)
    with pytest.raises(ValueError, match="no field"):
        db.update_application(app.id, nonsense=1)


def test_status_from_str_is_case_insensitive():
    assert ApplicationStatus.from_str("interview") is ApplicationStatus.INTERVIEW
    assert ApplicationStatus.from_str("REJECTED") is ApplicationStatus.REJECTED
    with pytest.raises(ValueError):
        ApplicationStatus.from_str("Pending")


# ---------------- feedback loop ----------------


def test_feedback_updates_status_and_appends_note(db: DBManager):
    app = _make_app(db)
    db.record_feedback(app.id, ApplicationStatus.INTERVIEW, "Recruiter screen scheduled")
    refreshed = db.get_application(app.id)
    assert refreshed.status is ApplicationStatus.INTERVIEW
    assert "Recruiter screen scheduled" in refreshed.notes
    assert len(refreshed.feedback) == 1


def test_successful_examples_prefers_matching_titles(db: DBManager):
    payload = {
        "summary": "Backend engineer",
        "tailored_experience": [{"company": "Acme", "title": "SWE", "bullets": ["Did a thing"]}],
    }
    win = _make_app(db, role_title="Senior Backend Engineer", tailored_payload=payload)
    other = _make_app(
        db,
        role_title="Marketing Analyst",
        job_url="https://x.com/2",
        tailored_payload=payload,
    )
    loss = _make_app(
        db, role_title="Backend Engineer II", job_url="https://x.com/3", tailored_payload=payload
    )
    db.record_feedback(win.id, ApplicationStatus.INTERVIEW)
    db.record_feedback(other.id, ApplicationStatus.OFFER)
    db.record_feedback(loss.id, ApplicationStatus.REJECTED)

    examples = db.successful_examples("Backend Engineer", limit=3)
    titles = [e["role_title"] for e in examples]
    assert "Senior Backend Engineer" in titles      # positive + title match
    assert "Backend Engineer II" not in titles      # rejected -> excluded
    assert titles[0] == "Senior Backend Engineer"   # title match outranks


def test_no_examples_when_history_empty(db: DBManager):
    assert db.successful_examples("Backend Engineer") == []


def test_stats(db: DBManager):
    a = _make_app(db)
    b = _make_app(db, job_url="https://x.com/9")
    db.mark_submitted(a.id)
    db.mark_submitted(b.id)
    db.record_feedback(a.id, ApplicationStatus.INTERVIEW)
    s = db.stats()
    assert s["total"] == 2 and s["submitted"] == 2
    assert s["positive_outcomes"] == 1 and s["response_rate"] == 50.0


def test_response_rate_counts_manually_submitted_applications(db: DBManager):
    """Regression: an application marked Interview without going through
    mark_submitted was excluded from the denominator, reporting 0% response
    rate while an interview was on screen."""
    app = _make_app(db)
    db.record_feedback(app.id, ApplicationStatus.INTERVIEW, "Recruiter screen")

    stats = db.stats()
    assert stats["submitted"] == 1
    assert stats["positive_outcomes"] == 1
    assert stats["response_rate"] == 100.0


def test_drafts_are_excluded_from_the_response_rate(db: DBManager):
    _make_app(db)                                        # stays Draft
    applied = _make_app(db, job_url="https://x.com/2")
    db.mark_submitted(applied.id)
    stats = db.stats()
    assert stats["submitted"] == 1
    assert stats["response_rate"] == 0.0
