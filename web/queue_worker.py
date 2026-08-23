"""Batch queue: run applications until each one needs a human, then move on.

The point of this module is that a blocked job must not hold up the queue. A
naive worker pool blocks on the first challenge and the "batch" degenerates
into a serial wait. Here, parking a job *releases its slot*, so the next
application starts immediately and the blocks accumulate for you to clear in
one sitting.

Two kinds of block, because they cost very different things:

  NEEDS_ANSWER   The pipeline hit a field it will not guess. The browser is
                 closed and the job re-runs from the top once answered - the
                 persistent profile keeps you logged in and re-filling a form
                 is cheap. Survives a server restart.

  NEEDS_BROWSER  A verification challenge or a final submit approval. These can
                 only be resolved in the live window, so the session is held
                 open and the job's thread waits in place. Held sessions are
                 capped (`MAX_HELD_BROWSERS`); past the cap a job fails rather
                 than exhausting the machine.
"""

from __future__ import annotations

import logging
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from automation.gatekeeper import Gatekeeper
from automation.notifier import Notifier, block_notice
from config import settings
from database.models import ActionKind, BlockMode, JobStatus, LogLevel

log = logging.getLogger(__name__)

#: Concurrent actively-running jobs. Parked jobs do not count against this.
DEFAULT_SLOTS = 2
#: Live browser sessions that may be held open waiting for a human.
MAX_HELD_BROWSERS = 3
#: How long a held session waits before giving up and closing.
HELD_BROWSER_TIMEOUT = 3600
#: Dispatcher idle poll. Short, because it also bounds how long a freed slot
#: sits unused after a job parks — the delay before the next batch item starts.
POLL_SECONDS = 0.25


class QueueStopped(Exception):
    """Raised inside a parked job when the worker is shutting down."""


# --------------------------------------------------------------------------
# Park registry
# --------------------------------------------------------------------------


@dataclass
class ParkedSession:
    """A job waiting in place while holding a live browser."""

    job_id: int
    user_id: int
    reason: str
    event: threading.Event = field(default_factory=threading.Event)
    answer: str = ""
    cancelled: bool = False


class ParkRegistry:
    """Tracks jobs parked on a live browser, so routes can wake them."""

    def __init__(self) -> None:
        self._sessions: dict[int, ParkedSession] = {}
        self._lock = threading.Lock()

    def park(self, job_id: int, user_id: int, reason: str) -> ParkedSession:
        session = ParkedSession(job_id=job_id, user_id=user_id, reason=reason)
        with self._lock:
            self._sessions[job_id] = session
        return session

    def release(self, job_id: int, answer: str = "") -> bool:
        """Wake a parked job. Returns False if it was not parked."""
        with self._lock:
            session = self._sessions.get(job_id)
        if session is None:
            return False
        session.answer = answer
        session.event.set()
        return True

    def cancel(self, job_id: int) -> bool:
        with self._lock:
            session = self._sessions.get(job_id)
        if session is None:
            return False
        session.cancelled = True
        session.event.set()
        return True

    def forget(self, job_id: int) -> None:
        with self._lock:
            self._sessions.pop(job_id, None)

    def held_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def cancel_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.cancelled = True
            session.event.set()


# --------------------------------------------------------------------------
# Gatekeeper
# --------------------------------------------------------------------------


class QueueGatekeeper(Gatekeeper):
    """Turns every stop into a queue block plus a notification.

    Whether the job parks in place or unwinds depends on what is blocking:
    anything that must be resolved in the live window parks; anything that is
    just a missing value raises so the job can release its browser and re-run
    later with the answer known.
    """

    #: Raised to unwind a job that can be resumed from scratch.
    class NeedsAnswer(Exception):
        def __init__(self, question: str, action_id: int | None) -> None:
            super().__init__(question)
            self.question = question
            self.action_id = action_id

    def __init__(
        self,
        db,
        user_id: int,
        job_id: int,
        registry: ParkRegistry,
        notifier: Notifier,
        release_slot: Callable[[], None],
        reacquire_slot: Callable[[], None],
        application_id: int | None = None,
    ) -> None:
        self.db = db
        self.user_id = user_id
        self.job_id = job_id
        self.registry = registry
        self.notifier = notifier
        self.release_slot = release_slot
        self.reacquire_slot = reacquire_slot
        self.application_id = application_id

    # -- helpers --

    def _notify(self, reason: str, kind: str) -> None:
        user = self.db.get_user(self.user_id)
        if user is None:
            return
        # One notification per quiet window, so a ten-item batch does not fire
        # ten alerts in a row.
        if self.db.recently_notified(self.user_id, user.notify_quiet_seconds or 0):
            log.debug("Within the quiet window; not notifying for job %s", self.job_id)
            return
        job = self.db.get_job(self.job_id)
        self.notifier.notify(user, block_notice(job, reason, kind), job_id=self.job_id)

    def _create_action(self, kind: ActionKind, question: str, reason: str,
                       options: list[str] | None = None, field_type: str = "text"):
        return self.db.create_action(
            user_id=self.user_id, kind=kind, question=question, reason=reason,
            application_id=self.application_id, options=options,
            field_type=field_type, required=True,
        )

    def _park_on_browser(self, action_id: int, reason: str, kind: str) -> str | None:
        """Hold the live session and wait for the human. Returns their answer."""
        if self.registry.held_count() >= MAX_HELD_BROWSERS:
            raise RuntimeError(
                f"Already holding {MAX_HELD_BROWSERS} browser sessions for other "
                "blocked jobs. Clear some in the dashboard, then retry this one."
            )

        # Register as wakeable BEFORE publishing the blocked status. The
        # dashboard reacts to that status, so the reverse order leaves a window
        # where an answer finds nothing to wake and the run waits out its whole
        # timeout.
        session = self.registry.park(self.job_id, self.user_id, reason)
        self.db.block_job(self.job_id, BlockMode.NEEDS_BROWSER, reason,
                          action_id=action_id, holds_browser=True)
        self.db.log_event(self.user_id, "job_parked",
                          f"Job #{self.job_id} parked holding a browser: {reason}",
                          level=LogLevel.WARNING, application_id=self.application_id)
        self._notify(reason, kind)
        # Free the slot so the rest of the batch keeps moving while this waits.
        self.release_slot()
        try:
            if not session.event.wait(timeout=HELD_BROWSER_TIMEOUT):
                raise TimeoutError(
                    f"No response within {HELD_BROWSER_TIMEOUT // 60} minutes."
                )
            if session.cancelled:
                raise QueueStopped(f"Job #{self.job_id} cancelled while parked.")
            return session.answer
        finally:
            self.registry.forget(self.job_id)
            # Re-enter the pool before doing more work, so the cap still holds.
            self.reacquire_slot()

    # -- interface --

    def alert(self, title: str, lines: list[str] | None = None) -> None:
        self.db.log_event(self.user_id, "alert", f"{title}: {'; '.join(lines or [])}",
                          level=LogLevel.WARNING, application_id=self.application_id)

    def confirm(self, action: str, details: list[str] | None = None) -> bool:
        lowered = action.lower()
        if "verification" in lowered or "captcha" in lowered:
            kind, action_kind = "human verification needed", ActionKind.CAPTCHA
        elif "account" in lowered:
            kind, action_kind = "account creation needs you", ActionKind.ACCOUNT_CREATION
        elif "submit" in lowered:
            kind, action_kind = "ready to submit", ActionKind.SUBMIT_CONFIRMATION
        else:
            kind, action_kind = "needs your approval", ActionKind.UNMAPPED_FIELD

        item = self._create_action(
            action_kind, f"Approve: {action}?", "\n".join(details or []),
            options=["Approve", "Cancel"], field_type="confirm",
        )
        try:
            answer = self._park_on_browser(item.id, action, kind)
        except (TimeoutError, RuntimeError) as exc:
            log.warning("Gate could not be held for job %s: %s", self.job_id, exc)
            return False
        return (answer or "").strip().lower() in ("approve", "yes", "y", "true")

    def ask(self, question: str, reason: str = "", kind: str = "unmapped_field",
            options: list[str] | None = None, required: bool = False) -> str | None:
        if kind == "captcha":
            item = self._create_action(ActionKind.CAPTCHA, question, reason)
            try:
                return self._park_on_browser(item.id, question, "human verification needed")
            except (TimeoutError, RuntimeError):
                return None

        # A plain missing value does not need the live window. Unwind so the
        # browser closes and the slot frees; the job re-runs once answered.
        item = self._create_action(
            ActionKind.UNMAPPED_FIELD, question, reason,
            options=options, field_type="select" if options else "text",
        )
        raise QueueGatekeeper.NeedsAnswer(question, item.id)


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------


class QueueWorker:
    """Dispatcher that drains the job queue, honouring the active-slot cap."""

    def __init__(
        self,
        db,
        slots: int = DEFAULT_SLOTS,
        notifier: Notifier | None = None,
        kinds: tuple[str, ...] = ("tailor", "apply"),
    ) -> None:
        self.db = db
        #: Which job kinds this worker will claim. A hosted instance runs
        #: ("tailor",) only; `apply` needs a browser a human can watch, so it is
        #: left for the local agent to claim over the API.
        self.kinds = kinds
        self.registry = ParkRegistry()
        self.notifier = notifier or Notifier(db)
        self._slots = threading.BoundedSemaphore(slots)
        self._slot_count = slots
        self._stop = threading.Event()
        self._dispatcher: threading.Thread | None = None
        self._threads: set[threading.Thread] = set()
        self._lock = threading.Lock()
        #: Overridden in tests to avoid launching a real browser.
        self.handlers: dict[str, Callable[..., Any]] = {}

    # ---------------- lifecycle ----------------

    def start(self) -> None:
        if self._dispatcher and self._dispatcher.is_alive():
            return
        requeued = self.db.reset_stale_running_jobs()
        if requeued:
            log.info("Requeued %d job(s) left running by a previous process.", requeued)
        self._stop.clear()
        self._dispatcher = threading.Thread(target=self._loop, name="jp-dispatch", daemon=True)
        self._dispatcher.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self.registry.cancel_all()
        if self._dispatcher:
            self._dispatcher.join(timeout=timeout)
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout=timeout)

    # ---------------- dispatch ----------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._slots.acquire(timeout=POLL_SECONDS):
                continue
            if self._stop.is_set():
                self._release()
                return
            try:
                job = self.db.claim_next_job(kinds=self.kinds)
            except Exception:
                log.exception("Could not claim a job")
                self._release()
                continue
            if job is None:
                self._release()
                self._stop.wait(POLL_SECONDS)
                continue
            self._spawn(job)

    def _release(self) -> None:
        try:
            self._slots.release()
        except ValueError:      # already at full capacity
            pass

    def _reacquire(self) -> None:
        while not self._stop.is_set():
            if self._slots.acquire(timeout=POLL_SECONDS):
                return

    def _spawn(self, job) -> None:
        thread = threading.Thread(target=self._run_job, args=(job.id, job.user_id,
                                                              job.kind),
                                  name=f"jp-job-{job.id}", daemon=True)
        with self._lock:
            self._threads.add(thread)
        thread.start()

    # ---------------- execution ----------------

    def _run_job(self, job_id: int, user_id: int, kind: str) -> None:
        released = False

        def release_slot() -> None:
            nonlocal released
            if not released:
                released = True
                self._release()

        def reacquire_slot() -> None:
            nonlocal released
            if released:
                self._reacquire()
                released = False

        try:
            gatekeeper = QueueGatekeeper(
                self.db, user_id, job_id, self.registry, self.notifier,
                release_slot, reacquire_slot,
            )
            handler = self.handlers.get(kind) or _default_handlers()[kind]
            handler(self.db, job_id, user_id, gatekeeper)

        except QueueGatekeeper.NeedsAnswer as need:
            # Recoverable: unwind, free the browser, resume once answered.
            self.db.block_job(job_id, BlockMode.NEEDS_ANSWER, need.question,
                              action_id=need.action_id, holds_browser=False)
            self.db.log_event(user_id, "job_blocked",
                              f"Job #{job_id} needs an answer: {need.question}",
                              level=LogLevel.WARNING)
            self._notify_block(user_id, job_id, need.question, "needs an answer")

        except QueueStopped as exc:
            self.db.finish_job(job_id, JobStatus.CANCELLED, str(exc))

        except Exception as exc:
            log.exception("Job %s failed", job_id)
            self.db.finish_job(job_id, JobStatus.FAILED, str(exc)[:500])
            self.db.log_event(user_id, "job_failed", str(exc), level=LogLevel.ERROR)
            self._notify_block(user_id, job_id, str(exc)[:200], "run failed")

        finally:
            self.registry.forget(job_id)
            release_slot()
            with self._lock:
                self._threads.discard(threading.current_thread())

    def _notify_block(self, user_id: int, job_id: int, reason: str, kind: str) -> None:
        user = self.db.get_user(user_id)
        if user is None:
            return
        if self.db.recently_notified(user_id, user.notify_quiet_seconds or 0):
            return
        job = self.db.get_job(job_id)
        try:
            self.notifier.notify(user, block_notice(job, reason, kind), job_id=job_id)
        except Exception:
            log.exception("Notification failed for job %s", job_id)

    # ---------------- external release ----------------

    def release_action(self, action_id: int, answer: str = "") -> int:
        """Wake every job blocked on this action. Returns how many moved."""
        moved = 0
        for job in self.db.jobs_blocked_on_action(action_id):
            if job.block_mode is BlockMode.NEEDS_BROWSER:
                if self.registry.release(job.id, answer):
                    moved += 1
                else:
                    # The session is gone (timed out, or the process restarted).
                    # Requeue rather than strand the job as permanently blocked.
                    log.info("Job %s was not parked in memory; requeueing.", job.id)
                    self.db.release_job(job.id)
                    moved += 1
            else:
                self.db.release_job(job.id)     # re-runs with the answer known
                moved += 1
        return moved


def _default_handlers() -> dict[str, Callable[..., Any]]:
    from web import runner

    return {"tailor": runner.run_tailor_job, "apply": runner.run_apply_job}
