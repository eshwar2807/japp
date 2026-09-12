"""A transient upstream failure is not a failed application.

Six tailoring jobs were lost in a single burst to

    503 overloaded_error: API key validation is temporarily unavailable.
                          Please retry.

The worker treated any exception as final, so each of those was a posting the
user never got to see - killed by a few seconds of upstream weather, on an
error whose own text asked for a retry.

Two things changed: the API client now rides out an overload rather than giving
up after the SDK's two quick retries, and the worker requeues a job whose
failure said nothing about the job itself.
"""

from __future__ import annotations

import pytest

from web.queue_worker import MAX_JOB_ATTEMPTS, is_transient


# ---------------- what deserves another go ----------------


@pytest.mark.parametrize("message", [
    "Error code: 503 - {'type': 'overloaded_error', 'message': 'API key "
    "validation is temporarily unavailable. Please retry.'}",
    "Error code: 429 - rate_limit_error",
    "Error code: 500 - internal server error",
    "Error code: 502 - bad gateway",
    "Request timed out",
    "Connection reset by peer",
    "The remote end closed the connection without response",
    "Service Unavailable",
])
def test_upstream_weather_is_retried(message):
    assert is_transient(RuntimeError(message))


@pytest.mark.parametrize("message", [
    "Error code: 401 - {'type': 'authentication_error', "
    "'message': 'API key is invalid.'}",
    "Error code: 400 - {'type': 'invalid_request_error', "
    "'message': 'Schema is too complex.'}",
    "Error code: 400 - invalid_request_error: adaptive thinking is not "
    "supported on this model",
    "No such column: run_jobs.priority",
    "Fill in your profile before running the pipeline.",
])
def test_a_failure_that_will_repeat_is_not_retried(message):
    """Retrying these spends money to arrive at the same place."""
    assert not is_transient(RuntimeError(message))


def test_an_authentication_error_is_never_retried_despite_its_code():
    """A 401 carries a number in the same shape as a 500 but means the
    opposite: nothing about waiting will fix it."""
    assert not is_transient(RuntimeError(
        "Error code: 401 - authentication_error: API key is invalid."))


# ---------------- the client rides out an overload ----------------


def test_the_client_retries_more_than_the_sdk_default():
    """Two quick retries were not enough for a real overload."""
    from engine.llm import CLIENT_MAX_RETRIES

    assert CLIENT_MAX_RETRIES > 2


def test_every_engine_builds_its_client_through_the_factory():
    """Four call sites each constructing their own client is four retry
    policies that drift apart."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for name in ("discovery", "ats_optimizer", "answer_resolver", "resume_import"):
        text = (root / "engine" / f"{name}.py").read_text()
        if "anthropic.Anthropic(" in text:
            offenders.append(name)
    assert not offenders, f"these build a client directly: {offenders}"


# ---------------- the worker requeues instead of failing ----------------


@pytest.fixture()
def user(db):
    return db.create_user("ada@example.com", "$argon2id$fake")


def test_a_retried_job_goes_back_to_the_queue(db, user):
    from database.models import JobStatus

    job = db.enqueue_job(user.id, kind="tailor", job_url="https://x.com/1")
    db.finish_job(job.id, JobStatus.RUNNING)

    attempt = db.retry_job(job.id, "transient")

    assert attempt == 1
    assert db.get_job(job.id).status is JobStatus.QUEUED


def test_attempts_accumulate_so_a_retry_loop_terminates(db, user):
    job = db.enqueue_job(user.id, kind="tailor", job_url="https://x.com/1")

    for expected in range(1, MAX_JOB_ATTEMPTS + 1):
        assert db.retry_job(job.id, "transient") == expected

    assert db.get_job(job.id).attempts >= MAX_JOB_ATTEMPTS


def test_a_fresh_job_starts_with_no_attempts(db, user):
    job = db.enqueue_job(user.id, kind="tailor", job_url="https://x.com/1")
    assert (db.get_job(job.id).attempts or 0) == 0
