"""Background execution of pipeline jobs launched from the dashboard.

Jobs run on a thread pool rather than in the request, so the HTTP response
returns immediately and the browser polls for progress. Playwright's sync API
is used from these worker threads, which is safe because they have no asyncio
loop of their own.

Every job writes structured events to ``run_logs`` and token spend to
``llm_usage``, which is what the Logs and Costs pages read.
"""

from __future__ import annotations

import json
import logging
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import settings
from database.models import ActionKind, ApplicationStatus, LogLevel
from engine.cost_tracker import TokenUsage, compute_cost
from engine.schemas import MasterProfile, TailoredResumeSchema

log = logging.getLogger(__name__)

MAX_WORKERS = 4


@dataclass
class RunState:
    """Live status of one job, for the progress indicator."""

    run_id: str
    user_id: int
    kind: str                       # "tailor" | "apply"
    application_id: int | None = None
    status: str = "running"         # running | done | failed | blocked
    message: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "status": self.status,
            "message": self.message,
            "application_id": self.application_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class RunManager:
    """Owns the worker pool and the in-memory status of active runs."""

    def __init__(self, max_workers: int = MAX_WORKERS) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="jp-run")
        self._runs: dict[str, RunState] = {}
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    # ---------------- bookkeeping ----------------

    def _new_run(self, user_id: int, kind: str, application_id: int | None = None) -> RunState:
        import secrets

        state = RunState(run_id=secrets.token_urlsafe(8), user_id=user_id, kind=kind,
                         application_id=application_id)
        with self._lock:
            self._runs[state.run_id] = state
        return state

    def get(self, run_id: str, user_id: int) -> RunState | None:
        with self._lock:
            state = self._runs.get(run_id)
        # Never expose another user's run.
        return state if state and state.user_id == user_id else None

    def active_for_user(self, user_id: int) -> list[RunState]:
        with self._lock:
            return [s for s in self._runs.values()
                    if s.user_id == user_id and s.status == "running"]

    def _finish(self, state: RunState, status: str, message: str) -> None:
        state.status = status
        state.message = message
        state.finished_at = datetime.now(timezone.utc)

    # ---------------- submission ----------------

    def submit_tailor(self, db, user_id: int, url: str, jd_text: str | None) -> RunState:
        state = self._new_run(user_id, "tailor")
        self._futures[state.run_id] = self._pool.submit(
            self._guard, state, db, lambda: _do_tailor(db, user_id, url, jd_text, state)
        )
        return state

    def submit_apply(self, db, user_id: int, application_id: int) -> RunState:
        state = self._new_run(user_id, "apply", application_id)
        self._futures[state.run_id] = self._pool.submit(
            self._guard, state, db, lambda: _do_apply(db, user_id, application_id, state)
        )
        return state

    def _guard(self, state: RunState, db, work) -> None:
        """Run a job, converting any failure into a log entry and action item."""
        try:
            work()
        except Exception as exc:
            log.exception("Run %s failed", state.run_id)
            self._finish(state, "failed", str(exc))
            db.log_event(
                state.user_id, f"{state.kind}_failed",
                f"{exc}\n{traceback.format_exc(limit=3)}",
                level=LogLevel.ERROR, application_id=state.application_id,
            )
            db.create_action(
                user_id=state.user_id,
                kind=ActionKind.ERROR,
                question=f"The {state.kind} run failed. Review and retry?",
                reason=str(exc),
                application_id=state.application_id,
            )


runs = RunManager()


# --------------------------------------------------------------------------
# Job bodies
# --------------------------------------------------------------------------


def _load_profile(db, user_id: int) -> MasterProfile:
    raw = db.get_profile(user_id)
    if not raw:
        raise RuntimeError("Fill in your profile before running the pipeline.")
    return MasterProfile.model_validate(raw)


def _usage_recorder(db, user_id: int, application_id: list[int | None], model: str):
    """Build an on_usage callback that prices each call and stores it.

    ``application_id`` is a one-element list so the id can be filled in after
    the application row is created mid-run.
    """

    def record(phase: str, response: Any) -> None:
        usage = TokenUsage.from_response(response)
        cost = compute_cost(model, usage)
        db.record_usage(
            user_id=user_id,
            model=model,
            phase=phase,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cost_usd=cost,
            application_id=application_id[0],
        )
        db.log_event(
            user_id, f"llm_{phase}",
            f"{model}: {usage.input_tokens} in / {usage.output_tokens} out = ${cost:.4f}",
            application_id=application_id[0],
        )

    return record


def _do_tailor(db, user_id: int, url: str, jd_text: str | None, state: RunState) -> None:
    from engine.ats_optimizer import ATSOptimizer
    from engine.pdf_generator import PDFGenerator

    profile = _load_profile(db, user_id)
    api_key = db.get_anthropic_key(user_id)
    if not api_key and not settings.ANTHROPIC_API_KEY:
        raise RuntimeError("Add your Anthropic API key in Settings first.")

    db.log_event(user_id, "tailor_start", f"Tailoring for {url}")

    if not jd_text:
        db.log_event(user_id, "fetch_jd", "Fetching the posting")
        jd_text = fetch_job_description(url)

    app_ref: list[int | None] = [None]
    optimizer = ATSOptimizer(
        profile=profile,
        api_key=api_key,
        on_usage=_usage_recorder(db, user_id, app_ref, settings.LLM_MODEL),
    )

    resume, keywords = optimizer.run(
        jd_text,
        few_shot=lambda kw: db.successful_examples(kw.role_title, user_id=user_id),
    )
    db.log_event(
        user_id, "tailored",
        f"{keywords.role_title} @ {keywords.company or 'unknown'} - "
        f"{resume.ats_match_percentage:.1f}% match",
    )

    pdf_path = PDFGenerator().generate(
        resume, profile,
        company=keywords.company or "Company",
        role_title=keywords.role_title,
    )

    app = db.create_application(
        company=keywords.company or "Unknown",
        role_title=keywords.role_title,
        job_url=url,
        job_description=jd_text,
        resume_pdf_path=str(pdf_path),
        match_score=resume.ats_match_percentage,
        tailored_payload=resume.model_dump(),
        user_id=user_id,
    )
    app_ref[0] = app.id
    state.application_id = app.id

    # Genuine gaps become action items so the user can decide what to do.
    for missing in resume.keywords_missing[:10]:
        db.create_action(
            user_id=user_id,
            kind=ActionKind.UNMAPPED_FIELD,
            question=f"The posting asks for '{missing}'. Do you have relevant experience?",
            reason="Not found in your profile. If you do have it, adding it here lets "
                   "future applications use it.",
            application_id=app.id,
        )

    db.log_event(user_id, "resume_built", f"Resume PDF written: {pdf_path.name}",
                 application_id=app.id)
    runs._finish(state, "done", f"Application #{app.id} ready at "
                                f"{resume.ats_match_percentage:.1f}% match")


def _do_apply(db, user_id: int, application_id: int, state: RunState) -> None:
    from automation.ats_drivers import get_driver_class
    from automation.gatekeeper import DashboardGatekeeper
    from automation.stealth_browser import (
        HumanBrowser,
        HumanDeclined,
        ManualInterventionRequired,
    )
    from engine.screener_mapper import ScreenerMapper

    app = db.get_application(application_id, user_id=user_id)
    if app is None:
        raise RuntimeError(f"No application #{application_id}.")
    pdf_path = Path(app.resume_pdf_path or "")
    if not pdf_path.exists():
        raise RuntimeError("The tailored resume PDF is missing. Re-run tailoring.")

    profile = _load_profile(db, user_id)
    resume = TailoredResumeSchema.model_validate(json.loads(app.tailored_payload or "{}"))

    # Questions the user has already answered in the dashboard are reused, so
    # the same question is never asked twice.
    answers = {**db.answered_action_map(user_id), **resume.screener_answers}
    mapper = ScreenerMapper(profile, answers)

    gatekeeper = DashboardGatekeeper(db, user_id, application_id)
    driver_class = get_driver_class(app.job_url)
    db.log_event(user_id, "apply_start",
                 f"Opening {app.job_url} with the {driver_class.NAME} driver",
                 application_id=application_id)

    try:
        with HumanBrowser(gatekeeper=gatekeeper) as browser:
            driver = driver_class(browser, profile, mapper, db)
            outcome = driver.apply(app.job_url, pdf_path)
            shot_dir = settings.OUTPUT_DIR / "screenshots"
            outcome.screenshot_path = str(
                browser.screenshot(shot_dir / f"app_{application_id}.png")
            )
    except (ManualInterventionRequired, HumanDeclined) as exc:
        db.log_event(user_id, "apply_blocked", str(exc), level=LogLevel.WARNING,
                     application_id=application_id)
        runs._finish(state, "blocked", str(exc))
        return

    for escalation in outcome.escalations:
        db.create_action(
            user_id=user_id,
            kind=ActionKind.UNMAPPED_FIELD,
            question=escalation.question,
            reason=escalation.reason,
            application_id=application_id,
            required=True,
        )

    if outcome.submitted:
        db.mark_submitted(application_id)
        db.log_event(user_id, "submitted",
                     f"Submitted; {outcome.fields_filled} fields filled",
                     application_id=application_id)
        runs._finish(state, "done", "Application submitted")
    else:
        db.log_event(user_id, "apply_incomplete", outcome.message or "Not submitted",
                     level=LogLevel.WARNING, application_id=application_id)
        runs._finish(state, "blocked",
                     f"Not submitted. {len(outcome.escalations)} item(s) need you.")


def fetch_job_description(url: str) -> str:
    """Scrape the visible text of a posting, headless."""
    from automation.gatekeeper import AutoDeclineGatekeeper
    from automation.stealth_browser import HumanBrowser

    with HumanBrowser(headless=True, gatekeeper=AutoDeclineGatekeeper()) as browser:
        browser.goto(url)
        text = browser.page.evaluate(
            """() => {
                const main = document.querySelector(
                    'main, article, [class*="job-description" i], [class*="description" i], #content'
                );
                return (main || document.body).innerText;
            }"""
        )
    return "\n".join(line.rstrip() for line in (text or "").splitlines() if line.strip())
