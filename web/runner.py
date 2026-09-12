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


def _drop_blocked(db, user_id: int, postings: list) -> list:
    """Remove postings at employers the user refuses to apply to."""
    from engine.boards import is_blocked

    blocked = db.blocked_companies(user_id)
    if not blocked:
        return postings
    kept, dropped = [], []
    for posting in postings:
        hit = is_blocked(posting.company, posting.url, blocked)
        (dropped if hit else kept).append(posting)
    if dropped:
        db.log_event(
            user_id, "blocked_company",
            f"Skipped {len(dropped)} posting(s) at blocked employers: "
            + ", ".join(sorted({p.company for p in dropped})[:8]),
        )
    return kept


def _company_has_room(counts: dict[str, int], company: str) -> bool:
    """Whether another application at this employer is within the cap.

    Mutates `counts` so a single run cannot queue four roles at one company by
    checking each against the same starting number - which is exactly what
    happened: four Databricks postings, two Stripe and two Tebra all reached
    the queue, each destined for its own separately tailored resume.
    """
    key = (company or "").strip().lower()
    if not key or settings.MAX_APPLICATIONS_PER_COMPANY <= 0:
        return True
    if counts.get(key, 0) >= settings.MAX_APPLICATIONS_PER_COMPANY:
        return False
    counts[key] = counts.get(key, 0) + 1
    return True


def _within_company_cap(db, user_id: int, postings: list) -> list:
    """Drop postings that would put a second resume in front of one recruiter."""
    counts = db.applications_per_company(user_id)
    kept, dropped = [], []
    for posting in postings:
        (kept if _company_has_room(counts, posting.company) else dropped).append(posting)
    if dropped:
        db.log_event(
            user_id, "company_cap",
            f"Skipped {len(dropped)} posting(s) at employers already applied to "
            f"(cap {settings.MAX_APPLICATIONS_PER_COMPANY} per company): "
            + ", ".join(sorted({p.company for p in dropped})[:8]),
        )
    return kept


def _reuse_company_resume(db, user_id: int, job, jd_text: str, api_key: str,
                          model: str, application_ref: list) -> str | None:
    """Score a repeat employer's posting against the resume already sent them.

    Returns a status message when this posting was handled by reuse, or None to
    fall through to ordinary tailoring. Only the keyword-extraction call is
    spent here - the expensive tailoring passes are skipped entirely, because
    the resume is already decided.
    """
    from engine.ats_optimizer import ATSOptimizer, score_match
    from engine.schemas import TailoredResumeSchema

    prior = db.resume_for_company(_company_of(job), user_id=user_id)
    if prior is None:
        return None

    optimizer = ATSOptimizer(
        profile=load_profile(db, user_id), model=model, api_key=api_key,
        on_usage=usage_recorder(db, user_id, application_ref, model),
    )
    keywords = optimizer.extract_keywords(jd_text)
    resume = TailoredResumeSchema.model_validate(json.loads(prior.tailored_payload))
    score, detail = score_match(keywords, resume)

    bar = settings.ELIGIBLE_MATCH_THRESHOLD
    if score < bar:
        db.log_event(
            user_id, "reuse_below_bar",
            f"The resume already sent to {prior.company} scores {score:.1f}% on "
            f"{keywords.role_title} (bar {bar:.0f}%). Not applying rather than "
            "sending that employer a second, different resume.",
        )
        return (f"Skipped: the resume used at {prior.company} scores "
                f"{score:.1f}% here (bar {bar:.0f}%)")

    application = db.create_application(
        company=prior.company,
        role_title=keywords.role_title,
        job_url=job.job_url,
        job_description=jd_text,
        match_score=score,
        tailored_payload=json.loads(prior.tailored_payload),
        resume_pdf_path=prior.resume_pdf_path,
        user_id=user_id,
        salary_min=getattr(prior, "salary_min", None),
        salary_max=getattr(prior, "salary_max", None),
    )
    application_ref[0] = application.id
    db.log_event(
        user_id, "resume_reused",
        f"Reusing the resume already sent to {prior.company} (application "
        f"#{prior.id}); it scores {score:.1f}% on {keywords.role_title}.",
        application_id=application.id,
    )
    db.enqueue_job(user_id, kind="apply", job_url=job.job_url,
                   application_id=application.id)
    return f"Reused the {prior.company} resume - {score:.1f}% match, queued to apply"


def _company_of(job) -> str:
    """Best guess at the employer for a queued posting, before tailoring names it."""
    from engine.boards import detect_board

    detected = detect_board(job.job_url or "")
    return detected[1] if detected else ""


# --------------------------------------------------------------------------
# Tailor
# --------------------------------------------------------------------------


def run_tailor_job(db, job_id: int, user_id: int, gatekeeper) -> None:
    from engine.ats_optimizer import ATSOptimizer
    from engine.pdf_generator import PDFGenerator

    job = db.get_job(job_id)
    if job is None:
        raise RuntimeError(f"No job #{job_id}.")

    # Checked before anything is loaded or fetched. A posting queued by hand
    # never passed the discovery filters, and "I am not comfortable applying
    # here" has to hold however the URL arrived - without paying for a profile
    # load and a browser fetch of a description nobody will use.
    from engine.boards import is_blocked

    blocked_hit = is_blocked(_company_of(job), job.job_url,
                             db.blocked_companies(user_id))
    if blocked_hit:
        db.log_event(user_id, "blocked_company",
                     f"{job.job_url} is at a blocked employer ({blocked_hit}).")
        db.finish_job(job_id, JobStatus.DONE,
                      f"Skipped: {blocked_hit} is on your blocklist")
        return

    profile = load_profile(db, user_id)
    api_key = db.get_anthropic_key(user_id)
    if not api_key and not settings.ANTHROPIC_API_KEY:
        raise RuntimeError("Add your Anthropic API key in Settings first.")

    # Bulk discovery work runs on the cheap tier; roles flagged as priority get
    # the expensive one. Resolved before it is logged.
    model = db.model_for_job(user_id, bool(job.priority))
    # The cap counts applications worth sending, not drafts. Stopping at twenty
    # drafts when only one qualifies would leave the user with one usable role.
    if db.eligible_today(user_id) >= settings.DAILY_APPLICATION_CAP:
        db.log_event(
            user_id, "application_cap",
            f"{settings.DAILY_APPLICATION_CAP} applications at or above "
            f"{settings.ELIGIBLE_MATCH_THRESHOLD:.0f}% found today; "
            "leaving the rest of the queue for tomorrow.",
        )
        db.finish_job(job_id, JobStatus.DONE, "Skipped: daily target already met")
        return

    db.log_event(user_id, "tailor_start",
                 f"Tailoring for {job.job_url} on {model}"
                 f"{' (priority)' if job.priority else ''}")

    jd_text = job.job_description
    if not jd_text:
        db.log_event(user_id, "fetch_jd", "Fetching the posting")
        jd_text = fetch_job_description(job.job_url)

    # Location, re-checked here and not only at discovery. A row queued under
    # an older, looser filter is still in the queue after the filter is fixed,
    # and a Vietnam-only posting reached the browser that way. Costs nothing.
    from engine.boards import location_allowed, location_for_url

    wanted_locations = (db.get_discovery_criteria(user_id) or {}).get("locations") or []
    # Rows queued before the location was recorded have none, and skipping the
    # check for them is exactly how the Vietnam posting would still get through.
    where = job.job_location or location_for_url(job.job_url)
    if where and not location_allowed(where, wanted_locations):
        db.log_event(user_id, "ineligible",
                     f"{job.job_url}: {where} is outside "
                     + ", ".join(wanted_locations))
        db.finish_job(job_id, JobStatus.DONE,
                      f"Skipped: {where} is not a requested location")
        return

    # Hard constraints, checked before anything is spent. These are not
    # preferences: a clearance an H-1B holder cannot obtain, or an employer who
    # states they will not sponsor, closes the role however well it fits.
    from engine.eligibility import assess

    verdict = assess(
        job.job_url, jd_text,
        require_java=settings.REQUIRE_JAVA,
        max_years=settings.MAX_YEARS_REQUIRED,
        needs_sponsorship=str(
            profile.legal.get("requires_sponsorship_now_or_future", "")
        ).strip().lower().startswith("y"),
        can_obtain_clearance=settings.CAN_OBTAIN_CLEARANCE,
        exclude_above_level=settings.EXCLUDE_ABOVE_LEVEL,
    )
    if not verdict.eligible:
        db.log_event(user_id, "ineligible",
                     f"{job.job_url}: " + "; ".join(verdict.reasons))
        db.finish_job(job_id, JobStatus.DONE,
                      "Skipped: " + "; ".join(verdict.reasons))
        return

    # A company already applied to keeps the resume it was sent. Tailoring a
    # fresh variant would put a second version of the same person in front of
    # one recruiter, and would also score the posting against a resume built
    # for it rather than the one that will actually go out.
    application_ref: list[int | None] = [job.application_id]
    reused = _reuse_company_resume(db, user_id, job, jd_text, api_key, model,
                                   application_ref)
    if reused is not None:
        db.finish_job(job_id, JobStatus.DONE, reused, application_id=application_ref[0])
        return

    optimizer = ATSOptimizer(
        profile=profile, model=model, api_key=api_key,
        on_usage=usage_recorder(db, user_id, application_ref, model),
    )
    from engine.ats_optimizer import NotViable

    try:
        resume, keywords = optimizer.run(
            jd_text,
            few_shot=lambda kw: db.successful_examples(kw.role_title, user_id=user_id),
        )
    except NotViable as verdict:
        # Rejected after one cheap call instead of four. Nothing is created:
        # a posting the profile cannot meet is not an application.
        db.log_event(
            user_id, "not_viable",
            f"Best achievable {verdict.ceiling:.1f}% against a "
            f"{verdict.target:.0f}% target. Out of reach: "
            + ", ".join(verdict.unreachable[:8]),
        )
        db.finish_job(job_id, JobStatus.DONE,
                      f"Skipped: best achievable {verdict.ceiling:.1f}% "
                      f"(target {verdict.target:.0f}%)")
        return
    if resume.removed_unsupported:
        db.log_event(
            user_id, "fabrication_blocked",
            "Removed skills the profile does not evidence: "
            + ", ".join(resume.removed_unsupported),
            level=LogLevel.WARNING,
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
        # Recorded rather than re-derived. An ineligible posting is normally
        # rejected before reaching here, so this stays True in practice - but
        # the count must never again include rows that cannot be applied to.
        eligible=verdict.eligible,
        ineligible_reason="; ".join(verdict.reasons),
    )
    application_ref[0] = application.id

    # Gaps are NOT queue items. One item per missing keyword produced 171
    # dismissible rows across eight applications - noise that buried the items
    # that actually block a run. They are already listed on the application
    # page and in the log below, which is where information belongs.
    if resume.keywords_missing:
        db.log_event(
            user_id, "gaps",
            "Requirements your profile does not cover: "
            + ", ".join(resume.keywords_missing[:12]),
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
        # A tailor job requeued after a restart would otherwise queue a second
        # apply job for an application that already has one.
        if db.has_pending_apply_job(application.id):
            log.info("Application %s already has a pending apply job", application.id)
        else:
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

    # A stored link is often the employer's careers page, which carries no form
    # at all - that is what produced "no resume input" and "submit button not
    # found" on Stripe. Find the page that really holds the form before opening
    # a browser, and if there is none, say so instead of asking the user to
    # explain a missing submit button.
    from engine.boards import BoardError, find_application_form

    try:
        target = find_application_form(application.job_url)
    except BoardError as exc:
        # Could not tell. Carry on with the stored link rather than writing off
        # a real application because a board was slow.
        db.log_event(user_id, "form_lookup_failed", str(exc),
                     level=LogLevel.WARNING, application_id=application.id)
        target = application.job_url
    if target is None:
        db.log_event(
            user_id, "no_application_form",
            f"{application.job_url} serves no application form on any known ATS "
            "address; this employer takes applications only through their own "
            "site. Apply to this one by hand.",
            level=LogLevel.WARNING, application_id=application.id,
        )
        db.finish_job(job_id, JobStatus.DONE,
                      "Skipped: no reachable application form - apply by hand")
        return

    driver_class = get_driver_class(target)
    if target != application.job_url:
        db.log_event(user_id, "apply_url_repaired",
                     f"{application.job_url} carries no form; using {target}",
                     application_id=application.id)
    db.log_event(user_id, "apply_start",
                 f"Opening {target} with the {driver_class.NAME} driver",
                 application_id=application.id)

    try:
        with HumanBrowser(gatekeeper=gatekeeper) as browser:
            driver = driver_class(browser, profile, mapper, db)
            outcome = driver.apply(target, pdf_path)
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

    # Ranked by estimated fit so the queue gets plausible postings rather than
    # every opening these companies happen to list. The estimate is free; the
    # tailoring it avoids is not.
    result = engine.run(
        criteria,
        already_applied=already_applied,
        profile=load_profile(db, user_id),
        min_estimated_fit=settings.DISCOVERY_MIN_FIT,
    )
    # Both forms are checked: rows queued before apply_url existed hold the
    # employer's own link, and re-queueing all of them would be a fresh start.
    # Remember every board that answered. A company found by web search is
    # useful once; its board is useful every day after, and this is what lets
    # the screened list grow instead of staying whatever was hand-written.
    from engine.boards import is_blocked as _is_blocked

    blocked_names = db.blocked_companies(user_id)
    learned = 0
    for name, provider, slug in result.get("resolved_boards") or []:
        if _is_blocked(name, "", blocked_names):
            continue
        before = set(db.learned_boards(user_id))
        db.learn_board(name, provider, slug, user_id=user_id)
        if name not in before:
            learned += 1
    if learned:
        db.log_event(user_id, "boards_learned",
                     f"Added {learned} new job board(s) to the daily screen; "
                     f"{len(db.learned_boards(user_id))} learned in total.")

    postings = [p for p in result["postings"]
                if p.url not in seen_urls and p.apply_url not in seen_urls]
    postings = _drop_blocked(db, user_id, postings)
    postings = _within_company_cap(db, user_id, postings)

    # Screening budget, not application budget. Most of these will be rejected
    # by the viability check for one cheap call each, and twenty matches means
    # looking at far more than twenty postings.
    remaining = max(settings.DAILY_SCREEN_CAP - db.screened_today(user_id), 0)
    if remaining <= 0:
        db.finish_job(job_id, JobStatus.DONE,
                      f"Daily screening cap of {settings.DAILY_SCREEN_CAP} reached; "
                      "queued nothing.")
        return
    postings = postings[:remaining]

    batch_id = f"disc{job_id}"
    for posting in postings:
        db.enqueue_job(
            user_id, kind="tailor", job_url=posting.apply_url,
            # The board already gave us the description, so no page fetch later.
            job_description=posting.description or None,
            job_location=posting.location or None,
            batch_id=batch_id,
        )

    if result.get("scored"):
        best = result["scored"][:5]
        db.log_event(
            user_id, "discovery_ranked",
            "Best estimated fits: "
            + ", ".join(f"{p.title[:34]} {score:.0f}%" for p, score in best),
        )

    for problem in result["problems"][:10]:
        db.log_event(user_id, "discovery_problem", problem, level=LogLevel.WARNING)

    message = (f"{len(result['companies'])} companies searched, "
               f"{len(postings)} postings queued")
    db.log_event(user_id, "discovery_done", message)
    db.finish_job(job_id, JobStatus.DONE, message)


# --------------------------------------------------------------------------
# Daily top-up
# --------------------------------------------------------------------------


def run_topup_job(db, job_id: int, user_id: int, gatekeeper) -> None:
    """Screen known boards until the ready target is met again.

    Reads boards directly rather than asking a model to find companies. Board
    reads are free and reliable; a model guessing company board slugs produced
    eight consecutive runs that queued nothing at all.
    """
    from engine.ats_optimizer import estimate_ceiling
    from engine.boards import ensure_description, matches, resolve_board
    from engine.eligibility import assess

    profile = load_profile(db, user_id)
    needs_sponsorship = str(
        profile.legal.get("requires_sponsorship_now_or_future", "")
    ).strip().lower().startswith("y")

    ready = db.eligible_today(user_id)
    target = settings.DAILY_APPLICATION_CAP
    if ready >= target:
        db.finish_job(job_id, JobStatus.DONE,
                      f"{ready} already ready; nothing to top up.")
        return

    # The hand-written list plus everything discovery has learned since.
    companies = list(settings.TOPUP_COMPANIES)
    seen_names = {c.strip().lower() for c in companies}
    for name in db.learned_boards(user_id):
        if name.strip().lower() not in seen_names:
            companies.append(name)
            seen_names.add(name.strip().lower())

    db.log_event(user_id, "topup_start",
                 f"{ready}/{target} ready; screening {len(companies)} boards "
                 f"({len(companies) - len(settings.TOPUP_COMPANIES)} learned)")

    seen = {a.job_url for a in db.list_applications(limit=1000, user_id=user_id)}
    seen |= {j.job_url for j in db.list_jobs(user_id=user_id, limit=1000)}
    per_company = db.applications_per_company(user_id)
    capped: list[str] = []

    candidates: list[tuple[float, Any]] = []
    reachable = 0
    from engine.boards import is_blocked

    blocked = db.blocked_companies(user_id)
    for company in companies:
        if is_blocked(company, "", blocked):
            continue
        resolved = resolve_board(company, "")
        if not resolved:
            continue
        reachable += 1
        for posting in resolved[2]:
            if not _company_has_room(per_company, posting.company):
                capped.append(posting.company)
                continue
            if posting.url in seen or posting.apply_url in seen:
                continue
            if not matches(posting,
                           ["java", "software engineer", "backend", "developer"],
                           ["remote", "united states"],
                           ["intern", "manager", "director"]):
                continue
            posting = ensure_description(posting)
            verdict = assess(
                posting.title, posting.description,
                require_java=settings.REQUIRE_JAVA,
                max_years=settings.MAX_YEARS_REQUIRED,
                needs_sponsorship=needs_sponsorship,
                can_obtain_clearance=settings.CAN_OBTAIN_CLEARANCE,
                exclude_above_level=settings.EXCLUDE_ABOVE_LEVEL,
            )
            if not verdict.eligible:
                continue
            posting.company = company
            candidates.append((estimate_ceiling(posting.description, profile), posting))

    candidates.sort(key=lambda pair: -pair[0])
    room = max(settings.DAILY_SCREEN_CAP - db.screened_today(user_id), 0)
    queued = 0
    for _, posting in candidates[:room]:
        db.enqueue_job(user_id, kind="tailor", job_url=posting.apply_url,
                       job_location=posting.location or None,
                       job_description=posting.description or None, batch_id="topup")
        queued += 1

    # "0 eligible postings found" said nothing about why, and the why was that
    # every candidate had already been applied to or was at a capped employer.
    # A run that finds nothing has to say what it discarded.
    message = (f"{reachable} boards read, {len(candidates)} eligible postings found, "
               f"{queued} queued for screening")
    if capped:
        from collections import Counter

        top = ", ".join(f"{name} x{n}" for name, n in Counter(capped).most_common(5))
        message += f"; {len(capped)} skipped at capped employers ({top})"
    db.log_event(user_id, "topup_done", message)

    # The boards we know are exhausted and the day is still short. Rather than
    # stopping there - which is what left 19/20 sitting until someone asked -
    # go and find boards we do not know yet. Discovery searches the web for
    # companies hiring, and every board it resolves is remembered for tomorrow.
    if queued == 0 and db.eligible_today(user_id) < target:
        _widen_the_search(db, user_id, target, this_job=job_id)

    db.finish_job(job_id, JobStatus.DONE, message)


def _widen_the_search(db, user_id: int, target: int,
                      this_job: int | None = None) -> None:
    """Queue a discovery run to find employers we have never screened.

    Bounded by the same daily allowance as the top-up itself: a market with
    nothing in it must not be searched all day.
    """
    import json as _json

    if db.has_screening_in_flight(user_id, exclude_job_id=this_job):
        return
    runs = db.topup_runs_today(user_id)
    if runs >= settings.TOPUP_MAX_RUNS_PER_DAY:
        return

    criteria = dict(db.get_discovery_criteria(user_id) or {})
    criteria["max_companies"] = criteria.get("max_companies") or 20
    job = db.enqueue_job(user_id, kind="discover",
                         job_description=_json.dumps(criteria))
    db.log_event(
        user_id, "search_widened",
        f"Known boards are exhausted and the day is short "
        f"({db.eligible_today(user_id)}/{target}). Searching for employers we "
        f"have not screened before; queued as job #{job.id}.",
    )

