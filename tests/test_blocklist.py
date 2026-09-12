"""Employers the user will not apply to, enforced rather than requested.

The dashboard already had a "Never show me (one per line)" box. Its contents
went into the model's prompt as a sentence asking it not to return those
companies, and nowhere else - the same shape of bug as the "never queue a
company already applied to" comment that sat above code doing no such thing. A
blocklist that depends on a model choosing to honour it is not a blocklist.

Matching is on whole words and on the board slug, because a posting can reach
the pipeline as job-boards.greenhouse.io/palantir with a company field saying
something else entirely.
"""

from __future__ import annotations

import pytest

from engine.boards import is_blocked, normalize_company


# ---------------- matching ----------------


@pytest.mark.parametrize("company", [
    "Deloitte",
    "Deloitte Consulting LLP",
    "deloitte",
    "  Deloitte  ",
    "Deloitte Touche Tohmatsu Limited",
])
def test_a_blocked_employer_is_caught_however_it_is_written(company):
    assert is_blocked(company, "", ["Deloitte"])


def test_a_blocked_employer_is_caught_by_its_board_slug():
    """The company field is whatever the board says; the slug is the identity."""
    assert is_blocked("", "https://job-boards.greenhouse.io/palantir/jobs/1",
                      ["Palantir"])


def test_a_legal_suffix_does_not_defeat_the_match():
    assert normalize_company("Block, Inc.") == "block"
    assert is_blocked("Block, Inc.", "", ["Block"])


def test_a_longer_name_containing_the_term_is_not_blocked():
    """Blocking "Block" must not also rule out "Blockchain Labs" - substring
    matching would, which is why this is word-based."""
    assert not is_blocked("Blockchain Labs", "", ["Block"])


def test_an_unrelated_employer_passes():
    assert not is_blocked("Stripe", "https://boards.greenhouse.io/stripe/jobs/1",
                          ["Deloitte", "Palantir"])


def test_a_multi_word_entry_matches_the_whole_phrase():
    assert is_blocked("Modern Treasury", "", ["Modern Treasury"])
    assert not is_blocked("Treasury Wine Estates", "", ["Modern Treasury"])


@pytest.mark.parametrize("blocked", [[], None, ["", "   "]])
def test_an_empty_blocklist_blocks_nothing(blocked):
    assert not is_blocked("Deloitte", "", blocked)


def test_the_reason_returned_is_the_entry_that_matched():
    """Logged back to the user, so it has to say which rule fired."""
    assert is_blocked("Deloitte Consulting", "", ["Palantir", "Deloitte"]) == "deloitte"


# ---------------- storage ----------------


@pytest.fixture()
def user(db):
    return db.create_user("block@example.com", "$argon2id$fake")


def test_blocking_an_employer_persists_it(db, user):
    db.block_company("Deloitte", user_id=user.id)
    assert db.blocked_companies(user.id) == ["Deloitte"]


def test_blocking_the_same_employer_twice_does_not_duplicate(db, user):
    db.block_company("Deloitte", user_id=user.id)
    db.block_company("deloitte inc", user_id=user.id)
    assert len(db.blocked_companies(user.id)) == 1


def test_unblocking_removes_it(db, user):
    db.block_company("Deloitte", user_id=user.id)
    db.block_company("Palantir", user_id=user.id)
    db.unblock_company("Deloitte", user_id=user.id)
    assert db.blocked_companies(user.id) == ["Palantir"]


def test_a_blank_entry_is_ignored(db, user):
    db.block_company("   ", user_id=user.id)
    assert db.blocked_companies(user.id) == []


def test_one_users_blocklist_does_not_affect_another(db, user):
    db.block_company("Deloitte", user_id=user.id)
    other = db.create_user("other@example.com", "$argon2id$fake")
    assert db.blocked_companies(other.id) == []


# ---------------- enforcement ----------------


def test_blocked_postings_are_dropped_before_queueing(db, user):
    from engine.boards import Posting
    from web.runner import _drop_blocked

    db.block_company("Palantir", user_id=user.id)
    postings = [
        Posting(company="Palantir", title="Engineer", url="https://x.com/1"),
        Posting(company="Stripe", title="Engineer", url="https://x.com/2"),
    ]

    kept = _drop_blocked(db, user.id, postings)

    assert [p.company for p in kept] == ["Stripe"]


def test_dropping_a_blocked_posting_is_logged(db, user):
    from engine.boards import Posting
    from web.runner import _drop_blocked

    db.block_company("Palantir", user_id=user.id)
    _drop_blocked(db, user.id, [Posting(company="Palantir", title="E",
                                        url="https://x.com/1")])

    events = [e for e in db.list_logs(user_id=user.id, limit=20)
              if e.event == "blocked_company"]
    assert events and "Palantir" in events[0].message


def test_nothing_is_dropped_when_the_list_is_empty(db, user):
    from engine.boards import Posting
    from web.runner import _drop_blocked

    postings = [Posting(company="Palantir", title="E", url="https://x.com/1")]
    assert _drop_blocked(db, user.id, postings) == postings


def test_a_manually_queued_blocked_posting_is_refused(db, user):
    """Discovery is not the only way a URL arrives. Pasting one into the
    dashboard bypasses every filter that runs at discovery time."""
    from web.runner import run_tailor_job

    db.set_anthropic_key(user.id, "sk-ant-test")
    db.block_company("Palantir", user_id=user.id)
    job = db.enqueue_job(
        user.id, kind="tailor",
        job_url="https://job-boards.greenhouse.io/palantir/jobs/999",
        job_description="Senior Java Engineer. Spring Boot. 4+ years.")

    run_tailor_job(db, job.id, user.id, gatekeeper=None)

    assert "blocklist" in db.get_job(job.id).message.lower()
    assert db.applications_today(user.id) == 0


def test_palantir_is_no_longer_screened_by_default():
    """It was added to the daily list earlier on the strength of a screening
    run; the user has since said they will not apply there."""
    from config import settings

    names = {c.strip().lower() for c in settings.TOPUP_COMPANIES}
    assert "palantir" not in names
