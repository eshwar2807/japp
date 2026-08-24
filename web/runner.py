"""Job bodies executed by the queue worker.

Each function runs on a worker thread and takes a gatekeeper, so the same code
serves an attended CLI run and a batched dashboard run. Playwright's sync API
is used from these threads, which is safe because they have no asyncio loop.

Blocking is the gatekeeper's business, not this module's: `ask()` either parks
the job in place (live browser needed) or raises `NeedsAnswer` to unwind it.
Neither case is caught here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import settings
from database.models import ActionKind, JobStatus, LogLevel
from engine.cost_tracker import TokenUsage, compute_cost
from engine.schemas import MasterProfile, TailoredResumeSchema

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def load_profile(db, user_id: int) -> MasterProfile:
    raw = db.get_profile(user_id)
    if not raw:
        raise RuntimeError("Fill in your profile before running the pipeline.")
    return MasterProfile.model_validate(raw)


def usage_recorder(db, user_id: int, application_ref: list[int | None], model: str):
    """on_usage callback that prices each call and records it.

    `application_ref` is a one-element list so the id can be filled in after the
    application row is created part-way through the run.
    """

    def record(phase: str, response: Any) -> None:
        usage = TokenUsage.from_response(response)
        cost = compute_cost(model, usage)
        db.record_usage(
            user_id=user_id, model=model, phase=phase,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cost_usd=cost, application_id=application_ref[0],
        )
        db.log_event(
            user_id, f"llm_{phase}",
            f"{model}: {usage.input_tokens} in / {usage.output_tokens} out = ${cost:.4f}",
            application_id=application_ref[0],
        )

    return record


def fetch_job_description(url: str) -> str:
    """Scrape the visible text of a posting, headless and unattended."""
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


# --------------------------------------------------------------------------
# Tailor
# --------------------------------------------------------------------------


def run_tailor_job(db, job_id: int, user_id: int, gatekeeper) -> None:
    from engine.ats_optimizer import ATSOptimizer
    from engine.pdf_generator import PDFGenerator

    job = db.get_job(job_id)
    if job is None:
        raise RuntimeError(f"No job #{job_id}.")

    profile = load_profile(db, user_id)
    api_key = db.get_anthropic_key(user_id)
    if not api_key and not settings.ANTHROPIC_API_KEY:
        raise RuntimeError("Add your Anthropic API key in Settings first.")

    # Bulk discovery work runs on the cheap tier; roles flagged as priority get
    # the expensive one. Resolved before it is logged.
    model = db.model_for_job(user_id, bool(job.priority))
    db.log_event(user_id, "tailor_start",
                 f"Tailoring for {job.job_url} on {model}"
                 f"{' (priority)' if job.priority else ''}")

    jd_text = job.job_description
    if not jd_text:
        db.log_event(user_id, "fetch_jd", "Fetching the posting")
        jd_text = fetch_job_description(job.job_url)

    application_ref: list[int | None] = [job.application_id]
    optimizer = ATSOptimizer(
        profile=profile, model=model, api_key=api_key,
        on_usage=usage_recorder(db, user_id, application_ref, model),
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
        company=keywords.company or "Company", role_title=keywords.role_title,
    )
    application = db.create_application(
        company=keywords.company or "Unknown",
        role_title=keywords.role_title,
        job_url=job.job_url,
        job_description=jd_text,
        resume_pdf_path=str(pdf_path),
        match_score=resume.ats_match_percentage,
        tailored_payload=resume.model_dump(),
        salary_min=keywords.salary_min,
        salary_max=keywords.salary_max,
        user_id=user_id,
    )
    application_ref[0] = application.id

    # Genuine gaps become queue items, but must not block the job: they are
    # information about your profile, not a prerequisite for this application.
    for missing in resume.keywords_missing[:10]:
        db.create_action(
            user_id=user_id, kind=ActionKind.UNMAPPED_FIELD,
            question=f"The posting asks for '{missing}'. Do you have relevant experience?",
            reason="Not found in your profile. Adding it lets future applications use it.",
            application_id=application.id,
        )

    db.log_event(user_id, "resume_built", f"Resume PDF written: {pdf_path.name}",
                 application_id=application.id)

    # Queue the browser step for anything that cleared the bar. This does not
    # submit anything: the driver still stops at the submit gate for approval.
    # It only saves opening each tailored application and clicking through.
    threshold = db.auto_apply_threshold(user_id)
    queued_apply = False
    if threshold is not None and resume.ats_match_percentage >= threshold:
        db.enqueue_job(user_id, kind="apply", job_url=job.job_url,
                       application_id=application.id, batch_id=job.batch_id)
        queued_apply = True
        db.log_event(
            user_id, "auto_queued",
            f"{resume.ats_match_percentage:.1f}% >= {threshold:.0f}% threshold; "
            "queued for the agent. Submission still needs your approval.",
            application_id=application.id,
        )
    elif threshold is not None:
        db.log_event(
            user_id, "below_threshold",
            f"{resume.ats_match_percentage:.1f}% is below the {threshold:.0f}% "
            "threshold; left as a draft for you to review.",
            application_id=application.id,
        )

    db.finish_job(
        job_id, JobStatus.DONE,
        f"Tailored at {resume.ats_match_percentage:.1f}% match"
        + (" - queued to apply" if queued_apply else ""),
        application_id=application.id,
    )


# --------------------------------------------------------------------------
# Apply
# --------------------------------------------------------------------------


def run_apply_job(db, job_id: int, user_id: int, gatekeeper) -> None:
    from automation.ats_drivers import get_driver_class
    from automation.stealth_browser import (
        HumanBrowser,
        HumanDeclined,
        ManualInterventionRequired,
    )
    from engine.screener_mapper import ScreenerMapper

    job = db.get_job(job_id)
    if job is None or job.application_id is None:
        raise RuntimeError(f"Job #{job_id} has no application to apply with.")

    application = db.get_application(job.application_id, user_id=user_id)
    if application is None:
        raise RuntimeError(f"No application #{job.application_id}.")

    # `Path("")` resolves to `Path(".")`, which exists — so an empty path must
    # be rejected explicitly and the check must be is_file(), not exists().
    if not application.resume_pdf_path:
        raise RuntimeError("This application has no resume yet. Re-run tailoring.")
    pdf_path = Path(application.resume_pdf_path)
    if not pdf_path.is_file():
        raise RuntimeError(f"Resume PDF is missing at {pdf_path}. Re-run tailoring.")

    gatekeeper.application_id = application.id
    profile = load_profile(db, user_id)
    resume = TailoredResumeSchema.model_validate(json.loads(application.tailored_payload or "{}"))

    # The resolution ladder: profile rules, then answers you have already
    # given, then a single batched inference attempt, then you.
    from engine.answer_resolver import LLMAnswerResolver

    resolver = LLMAnswerResolver(
        api_key=db.get_anthropic_key(user_id),
        on_usage=usage_recorder(db, user_id, [application.id], settings.LLM_MODEL_BULK),
    )
    mapper = ScreenerMapper(
        profile,
        screener_answers=resume.screener_answers,
        remembered=db.answered_action_map(user_id),
        resolver=resolver,
        salary_min=application.salary_min,
        salary_max=application.salary_max,
    )

    driver_class = get_driver_class(application.job_url)
    db.log_event(user_id, "apply_start",
                 f"Opening {application.job_url} with the {driver_class.NAME} driver",
                 application_id=application.id)

    try:
        with HumanBrowser(gatekeeper=gatekeeper) as browser:
            driver = driver_class(browser, profile, mapper, db)
            outcome = driver.apply(application.job_url, pdf_path)
            outcome.screenshot_path = str(
                browser.screenshot(settings.OUTPUT_DIR / "screenshots" / f"app_{application.id}.png")
            )
    except (ManualInterventionRequired, HumanDeclined) as exc:
        db.log_event(user_id, "apply_stopped", str(exc), level=LogLevel.WARNING,
                     application_id=application.id)
        db.finish_job(job_id, JobStatus.FAILED, str(exc)[:500],
                      application_id=application.id)
        return

    for escalation in outcome.escalations:
        db.create_action(
            user_id=user_id, kind=ActionKind.UNMAPPED_FIELD,
            question=escalation.question, reason=escalation.reason,
            application_id=application.id, required=True,
        )

    if outcome.submitted:
        db.mark_submitted(application.id)
        db.log_event(user_id, "submitted",
                     f"Submitted; {outcome.fields_filled} fields filled",
                     application_id=application.id)
        db.finish_job(job_id, JobStatus.DONE, "Submitted",
                      application_id=application.id)
    else:
        db.log_event(user_id, "apply_incomplete", outcome.message or "Not submitted",
                     level=LogLevel.WARNING, application_id=application.id)
        db.finish_job(job_id, JobStatus.FAILED,
                      f"Not submitted. {len(outcome.escalations)} item(s) need you.",
                      application_id=application.id)


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def run_discovery_job(db, job_id: int, user_id: int, gatekeeper) -> None:
    """Find companies that are hiring, then queue their real postings.

    Board-sourced postings arrive with their full description attached, so the
    tailor jobs this queues need no browser fetch at all.
    """
    import json as _json

    from engine.discovery import DiscoveryCriteria, DiscoveryEngine

    job = db.get_job(job_id)
    if job is None:
        raise RuntimeError(f"No job #{job_id}.")

    api_key = db.get_anthropic_key(user_id)
    if not api_key and not settings.ANTHROPIC_API_KEY:
        raise RuntimeError("Add your Anthropic API key in Settings first.")

    try:
        criteria = DiscoveryCriteria.model_validate(_json.loads(job.job_description or "{}"))
    except (ValueError, _json.JSONDecodeError) as exc:
        raise RuntimeError(f"Discovery criteria are not valid: {exc}") from exc

    # Never queue a company already applied to, and never re-queue a posting.
    existing = db.list_applications(limit=1000, user_id=user_id)
    already_applied = sorted({a.company for a in existing if a.company})
    seen_urls = {a.job_url for a in existing if a.job_url}

    application_ref: list[int | None] = [None]
    engine = DiscoveryEngine(
        api_key=api_key,
        on_usage=usage_recorder(db, user_id, application_ref, settings.LLM_MODEL_DISCOVERY),
    )
    db.log_event(user_id, "discovery_start", f"Searching for: {criteria.describe()}")

    result = engine.run(criteria, already_applied=already_applied)
    postings = [p for p in result["postings"] if p.url not in seen_urls]

    # Respect the remaining daily allowance rather than dumping 500 jobs in.
    remaining = max(settings.DAILY_APPLICATION_CAP - db.applications_today(user_id), 0)
    if remaining <= 0:
        db.finish_job(job_id, JobStatus.DONE,
                      "Daily application cap already reached; queued nothing.")
        return
    postings = postings[:remaining]

    batch_id = f"disc{job_id}"
    for posting in postings:
        db.enqueue_job(
            user_id, kind="tailor", job_url=posting.url,
            # The board already gave us the description, so no page fetch later.
            job_description=posting.description or None,
            batch_id=batch_id,
        )

    for problem in result["problems"][:10]:
        db.log_event(user_id, "discovery_problem", problem, level=LogLevel.WARNING)

    message = (f"{len(result['companies'])} companies searched, "
               f"{len(postings)} postings queued")
    db.log_event(user_id, "discovery_done", message)
    db.finish_job(job_id, JobStatus.DONE, message)
