"""Batch queue: park/resume semantics, slot accounting, and notifications.

The central claim under test is that a blocked job releases its worker slot, so
a batch keeps moving instead of serialising behind the first challenge. If that
regresses, "batch the blocks" silently becomes "wait for the first block".
"""

from __future__ import annotations

import threading
import time

import pytest

from automation.notifier import Notice, Notifier, WebhookChannel, _shell_quote_applescript
from database.models import ActionKind, BlockMode, JobStatus
from web.queue_worker import MAX_HELD_BROWSERS, ParkRegistry, QueueGatekeeper, QueueWorker


@pytest.fixture()
def user(db):
    return db.create_user("ada@example.com", "$argon2id$fake")


class RecordingChannel:
    """Captures notices instead of alerting the machine."""

    name = "test"

    def __init__(self) -> None:
        self.sent: list[Notice] = []
        self.fail = False

    def send(self, notice: Notice) -> None:
        if self.fail:
            raise RuntimeError("channel unavailable")
        self.sent.append(notice)


@pytest.fixture()
def worker(db):
    channel = RecordingChannel()
    w = QueueWorker(db, slots=1, notifier=Notifier(db, channels=[channel]))
    w.channel = channel
    yield w
    w.stop(timeout=2)


#: Generous, because these tests wait on real threads and a 1s dispatcher poll.
#: They return as soon as the condition holds, so a healthy run pays nothing;
#: a tight bound only buys flakes under load.
WAIT = 20.0


def wait_until(predicate, timeout=WAIT, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------- queue mechanics ----------------


def test_jobs_are_claimed_in_order(db, user):
    first = db.enqueue_job(user.id, job_url="https://x.com/1")
    second = db.enqueue_job(user.id, job_url="https://x.com/2")

    assert db.claim_next_job().id == first.id
    assert db.claim_next_job().id == second.id
    assert db.claim_next_job() is None


def test_ready_jobs_are_served_before_queued_ones(db, user):
    queued = db.enqueue_job(user.id, job_url="https://x.com/new")
    blocked = db.enqueue_job(user.id, job_url="https://x.com/blocked")
    db.block_job(blocked.id, BlockMode.NEEDS_ANSWER, "needs an answer")
    db.release_job(blocked.id)

    # Clearing a block should drain the backlog, not queue behind fresh work.
    assert db.claim_next_job().id == blocked.id
    assert db.claim_next_job().id == queued.id


def test_claiming_marks_running_and_counts_attempts(db, user):
    job = db.enqueue_job(user.id, job_url="https://x.com/1")
    claimed = db.claim_next_job()
    assert claimed.status is JobStatus.RUNNING
    assert claimed.attempts == 1
    assert claimed.started_at is not None


def test_stale_running_jobs_are_requeued(db, user):
    db.enqueue_job(user.id, job_url="https://x.com/1")
    db.claim_next_job()                       # simulates a process that then died

    assert db.reset_stale_running_jobs() == 1
    assert db.claim_next_job() is not None    # runnable again


def test_terminal_jobs_are_never_reclaimed(db, user):
    job = db.enqueue_job(user.id, job_url="https://x.com/1")
    db.claim_next_job()
    db.finish_job(job.id, JobStatus.DONE, "done")
    assert db.claim_next_job() is None


def test_queue_summary_counts_by_status(db, user):
    a = db.enqueue_job(user.id, job_url="https://x.com/1")
    db.enqueue_job(user.id, job_url="https://x.com/2")
    db.claim_next_job()
    db.block_job(a.id, BlockMode.NEEDS_ANSWER, "blocked")

    summary = db.queue_summary(user.id)
    assert summary["Blocked"] == 1 and summary["Queued"] == 1
    assert summary["active"] == 1


# ---------------- the core claim ----------------


def test_a_parked_job_releases_its_slot_so_the_batch_continues(db, user, worker):
    """With one slot, a job parked on a live browser must not stall the queue."""
    parked = threading.Event()
    second_ran = threading.Event()

    def blocking_handler(db_, job_id, user_id, gatekeeper):
        parked.set()
        gatekeeper.confirm("submit the application", ["parked for the test"])

    def quick_handler(db_, job_id, user_id, gatekeeper):
        second_ran.set()
        db_.finish_job(job_id, JobStatus.DONE, "done")

    calls = {"n": 0}

    def dispatch(db_, job_id, user_id, gatekeeper):
        calls["n"] += 1
        (blocking_handler if calls["n"] == 1 else quick_handler)(db_, job_id, user_id, gatekeeper)

    worker.handlers["tailor"] = dispatch
    first = db.enqueue_job(user.id, job_url="https://x.com/blocks")
    second = db.enqueue_job(user.id, job_url="https://x.com/fast")
    worker.start()

    assert parked.wait(WAIT), "first job never reached its gate"
    assert second_ran.wait(WAIT), "second job never ran — the block stalled the queue"

    assert db.get_job(first.id).status is JobStatus.BLOCKED
    assert db.get_job(second.id).status is JobStatus.DONE


def test_answering_releases_a_parked_job(db, user, worker):
    parked = threading.Event()
    approved = {}

    def handler(db_, job_id, user_id, gatekeeper):
        parked.set()
        approved["value"] = gatekeeper.confirm("submit the application")
        db_.finish_job(job_id, JobStatus.DONE, "done")

    worker.handlers["tailor"] = handler
    job = db.enqueue_job(user.id, job_url="https://x.com/1")
    worker.start()

    assert parked.wait(WAIT)
    assert wait_until(lambda: db.get_job(job.id).status is JobStatus.BLOCKED)

    action_id = db.get_job(job.id).blocking_action_id
    db.answer_action(action_id, "Approve", user_id=user.id)
    assert worker.release_action(action_id, "Approve") == 1

    assert wait_until(lambda: db.get_job(job.id).status is JobStatus.DONE, timeout=WAIT)
    assert approved["value"] is True


def test_a_missing_value_unwinds_instead_of_holding_a_browser(db, user, worker):
    """NEEDS_ANSWER blocks must free the browser and be resumable from scratch."""
    asked = threading.Event()

    def handler(db_, job_id, user_id, gatekeeper):
        asked.set()
        gatekeeper.ask("How many years of Go?", "Not in your profile.")

    worker.handlers["tailor"] = handler
    job = db.enqueue_job(user.id, job_url="https://x.com/1")
    worker.start()

    assert asked.wait(WAIT)
    assert wait_until(lambda: db.get_job(job.id).status is JobStatus.BLOCKED)

    blocked = db.get_job(job.id)
    assert blocked.block_mode is BlockMode.NEEDS_ANSWER
    assert blocked.holds_browser is False, "an answerable block must not pin a browser"
    assert worker.registry.held_count() == 0


def test_answering_a_missing_value_requeues_the_job(db, user, worker):
    def handler(db_, job_id, user_id, gatekeeper):
        gatekeeper.ask("How many years of Go?", "Not in your profile.")

    worker.handlers["tailor"] = handler
    job = db.enqueue_job(user.id, job_url="https://x.com/1")
    worker.start()
    assert wait_until(lambda: db.get_job(job.id).status is JobStatus.BLOCKED)
    worker.stop(timeout=2)

    action_id = db.get_job(job.id).blocking_action_id
    db.answer_action(action_id, "4", remember=True, user_id=user.id)
    assert worker.release_action(action_id, "4") == 1

    assert db.get_job(job.id).status is JobStatus.READY
    # And the answer is now available to every later run.
    assert db.answered_action_map(user.id)["How many years of Go?"] == "4"


def test_cancelling_wakes_a_parked_job(db, user, worker):
    parked = threading.Event()

    def handler(db_, job_id, user_id, gatekeeper):
        parked.set()
        gatekeeper.confirm("submit the application")
        db_.finish_job(job_id, JobStatus.DONE, "done")

    worker.handlers["tailor"] = handler
    job = db.enqueue_job(user.id, job_url="https://x.com/1")
    worker.start()
    assert parked.wait(WAIT)
    assert wait_until(lambda: db.get_job(job.id).status is JobStatus.BLOCKED)

    assert worker.registry.cancel(job.id) is True
    assert wait_until(lambda: worker.registry.held_count() == 0, timeout=WAIT)


def test_held_browsers_are_capped(db, user):
    """Past the cap a gate declines rather than exhausting the machine."""
    registry = ParkRegistry()
    for i in range(MAX_HELD_BROWSERS):
        registry.park(1000 + i, user.id, "holding")

    keeper = QueueGatekeeper(db, user.id, 1, registry, Notifier(db, channels=[]),
                             lambda: None, lambda: None)
    assert keeper.confirm("submit the application") is False
    assert registry.held_count() == MAX_HELD_BROWSERS


def test_a_failing_job_is_recorded_not_swallowed(db, user, worker):
    def handler(db_, job_id, user_id, gatekeeper):
        raise RuntimeError("driver exploded")

    worker.handlers["tailor"] = handler
    job = db.enqueue_job(user.id, job_url="https://x.com/1")
    worker.start()

    assert wait_until(lambda: db.get_job(job.id).status is JobStatus.FAILED, timeout=WAIT)
    assert "driver exploded" in db.get_job(job.id).message


# ---------------- notifications ----------------


def test_parking_sends_a_notification(db, user, worker):
    def handler(db_, job_id, user_id, gatekeeper):
        gatekeeper.ask("How many years of Go?", "Not in your profile.")

    worker.handlers["tailor"] = handler
    db.enqueue_job(user.id, job_url="https://x.com/1")
    worker.start()

    assert wait_until(lambda: worker.channel.sent), "no notification sent"
    notice = worker.channel.sent[0]
    assert "Job pipeline" in notice.title
    assert "/actions" in notice.url


def test_the_quiet_window_suppresses_repeat_alerts(db, user):
    channel = RecordingChannel()
    notifier = Notifier(db, channels=[channel])
    db.update_user(user.id, notify_quiet_seconds=600)

    notifier.notify(db.get_user(user.id), Notice("first", "body"))
    assert db.recently_notified(user.id, 600) is True
    assert db.recently_notified(user.id, 0) is False


def test_notification_failures_do_not_propagate(db, user):
    channel = RecordingChannel()
    channel.fail = True
    delivered = Notifier(db, channels=[channel]).notify(user, Notice("t", "b"))

    assert delivered == []
    recorded = db.list_notifications(user.id)[0]
    assert recorded.delivered is False and "unavailable" in recorded.error


def test_notifications_are_recorded_for_the_ui(db, user):
    Notifier(db, channels=[RecordingChannel()]).notify(user, Notice("Blocked", "Acme"))
    row = db.list_notifications(user.id)[0]
    assert row.delivered is True and row.title == "Blocked"


# ---------------- webhook privacy ----------------


def test_webhook_rejects_a_non_http_url():
    assert WebhookChannel("file:///etc/passwd").available() is False
    assert WebhookChannel("").available() is False
    with pytest.raises(ValueError):
        WebhookChannel("nope").send(Notice("t", "b"))


def test_webhook_payload_carries_no_sensitive_content(monkeypatch):
    """Only the reason and a link leave the machine."""
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout=None):
        captured["body"] = request.data.decode()
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    WebhookChannel("https://ntfy.sh/topic").send(
        Notice("Job pipeline: human verification needed", "Acme — Backend Engineer", "/actions")
    )

    body = captured["body"]
    assert "Acme" in body                      # enough context to act
    for secret in ("sk-ant-", "password", "resume", "ssn", "@example.com"):
        assert secret not in body.lower()


def test_applescript_injection_is_escaped():
    """Company names come from job postings, i.e. untrusted input."""
    escaped = _shell_quote_applescript('Acme" & do shell script "rm -rf ~')
    assert '" & do shell script "' not in escaped
    assert escaped.count('\\"') == 2


# ---------------- isolation ----------------


def test_jobs_are_scoped_to_their_owner(db, user):
    other = db.create_user("eve@example.com", "$argon2id$fake")
    job = db.enqueue_job(user.id, job_url="https://x.com/1")

    assert db.get_job(job.id, user_id=other.id) is None
    assert db.get_job(job.id, user_id=user.id) is not None
    assert db.list_jobs(user_id=other.id) == []


def test_cancelling_another_users_job_is_refused(db, user):
    other = db.create_user("eve@example.com", "$argon2id$fake")
    job = db.enqueue_job(user.id, job_url="https://x.com/1")
    with pytest.raises(PermissionError):
        db.cancel_job(job.id, user_id=other.id)


def test_a_job_is_wakeable_the_moment_it_reads_as_blocked(db, user, worker):
    """Regression: the job was marked BLOCKED before it was registered as
    wakeable, so an answer arriving in that window was dropped and the run
    waited out its full timeout."""
    seen_blocked = threading.Event()

    def handler(db_, job_id, user_id, gatekeeper):
        gatekeeper.confirm("submit the application")
        db_.finish_job(job_id, JobStatus.DONE, "done")

    worker.handlers["tailor"] = handler
    job = db.enqueue_job(user.id, job_url="https://x.com/1")

    def watch():
        while not seen_blocked.is_set():
            current = db.get_job(job.id)
            if current and current.status is JobStatus.BLOCKED:
                # The instant the status is visible, the session must be wakeable.
                assert worker.registry.release(job.id, "Approve") is True
                seen_blocked.set()
                return
            time.sleep(0.005)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    worker.start()

    assert seen_blocked.wait(WAIT), "job never reported as blocked"
    watcher.join(timeout=2)
    assert wait_until(lambda: db.get_job(job.id).status is JobStatus.DONE, timeout=WAIT)


def test_releasing_a_job_with_no_live_session_requeues_it(db, user, worker):
    """A restart drops in-memory sessions; the job must not be stranded."""
    job = db.enqueue_job(user.id, job_url="https://x.com/1")
    item = db.create_action(user.id, ActionKind.SUBMIT_CONFIRMATION, "Approve?")
    db.block_job(job.id, BlockMode.NEEDS_BROWSER, "waiting",
                 action_id=item.id, holds_browser=True)

    assert worker.registry.held_count() == 0        # nothing parked in memory
    assert worker.release_action(item.id, "Approve") == 1
    assert db.get_job(job.id).status is JobStatus.READY


def test_an_application_with_no_resume_fails_fast(db, user, worker):
    """Regression: `Path("")` is `Path(".")`, which exists — so a missing
    resume path passed the guard and the driver tried to upload a directory."""
    from web.runner import run_apply_job

    db.save_profile(user.id, {"contact": {"full_name": "Ada", "email": "a@b.com"}})
    application = db.create_application(
        company="Acme", role_title="Engineer", job_url="https://x.com/1",
        user_id=user.id, resume_pdf_path=None,
    )
    job = db.enqueue_job(user.id, "apply", "https://x.com/1",
                         application_id=application.id)

    with pytest.raises(RuntimeError, match="no resume"):
        run_apply_job(db, job.id, user.id, gatekeeper=None)


def test_a_resume_path_pointing_nowhere_fails_fast(db, user, tmp_path):
    from web.runner import run_apply_job

    db.save_profile(user.id, {"contact": {"full_name": "Ada", "email": "a@b.com"}})
    application = db.create_application(
        company="Acme", role_title="Engineer", job_url="https://x.com/1",
        user_id=user.id, resume_pdf_path=str(tmp_path / "gone.pdf"),
    )
    job = db.enqueue_job(user.id, "apply", "https://x.com/1",
                         application_id=application.id)

    with pytest.raises(RuntimeError, match="missing"):
        run_apply_job(db, job.id, user.id, gatekeeper=None)


def test_a_worker_only_claims_the_kinds_it_can_run(db, user):
    """A hosted dashboard must not start browser work nobody can watch."""
    db.enqueue_job(user.id, kind="apply", job_url="https://x.com/apply")
    tailor = db.enqueue_job(user.id, kind="tailor", job_url="https://x.com/tailor")

    claimed = db.claim_next_job(kinds=("tailor",))
    assert claimed.id == tailor.id
    # The apply job is left alone for the local agent.
    assert db.claim_next_job(kinds=("tailor",)) is None
    assert db.claim_next_job(kinds=("apply",)) is not None


def test_an_unfiltered_worker_claims_everything(db, user):
    db.enqueue_job(user.id, kind="apply", job_url="https://x.com/apply")
    assert db.claim_next_job() is not None
