"""HTML pages: overview, profile, applications, actions, logs, costs, settings."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

from fastapi import APIRouter, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from config import settings
from database.models import (
    ActionStatus,
    ApplicationStatus,
    BlockMode,
    JobStatus,
    LogLevel,
)
from engine.cost_tracker import COST_WINDOWS, PRICING_AS_OF, daily_series, summarize
from web.deps import AdminUser, CSRFProtected, CurrentUser, Database, get_worker
from web.profile_form import (
    DISCLOSURE_FIELDS,
    LEGAL_FIELDS,
    blank_profile,
    completeness,
    parse_profile_form,
)
from web.security import CSRF_COOKIE, mask_secret

router = APIRouter(tags=["dashboard"])

#: Most postings accepted in one paste.
MAX_BATCH = 25



def render(request: Request, template: str, user, db, **context) -> HTMLResponse:
    from web.app import templates

    base = {
        "user": user,
        "csrf_token": getattr(request.state, "csrf_token", ""),
        "open_actions": db.count_open_actions(user.id) if user else 0,
        "queue": db.queue_summary(user.id) if user else {},
        "path": request.url.path,
    }
    return templates.TemplateResponse(request, template, {**base, **context})


def _redirect(path: str, ok: str = "", error: str = "") -> RedirectResponse:
    if ok:
        path += ("&" if "?" in path else "?") + f"ok={quote_plus(ok)}"
    if error:
        path += ("&" if "?" in path else "?") + f"error={quote_plus(error)}"
    return RedirectResponse(path, status_code=303)


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


@router.get("/healthz", include_in_schema=False)
def healthz(db: Database):
    """Liveness probe for the platform. Unauthenticated by necessity, so it
    reveals nothing beyond whether the process can reach its database."""
    from fastapi.responses import JSONResponse

    try:
        db.count_users()
    except Exception:
        return JSONResponse({"status": "degraded"}, status_code=503)
    return JSONResponse({"status": "ok"})


# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def overview(request: Request, user: CurrentUser, db: Database):
    profile = db.get_profile(user.id) or blank_profile()
    readiness = completeness(profile)
    stats = db.stats(user_id=user.id)

    today = daily_series(db.usage_rows(user.id, days=1), days=1)
    month = daily_series(db.usage_rows(user.id, days=30), days=30)

    return render(
        request, "overview.html", user, db,
        readiness=readiness,
        stats=stats,
        applications=db.list_applications(limit=8, user_id=user.id),
        submitted_today=db.submitted_today(user.id),
        awaiting_agent=db.awaiting_agent(user.id),
        actions=db.list_actions(user_id=user.id, limit=5),
        logs=db.list_logs(user_id=user.id, limit=8),
        cost_today=today[-1]["cost"] if today else 0.0,
        cost_month=summarize(month)["total_cost"],
        has_api_key=bool(user.encrypted_anthropic_key),
    )


# --------------------------------------------------------------------------
# Profile
# --------------------------------------------------------------------------


@router.get("/profile", response_class=HTMLResponse)
def profile_page(request: Request, user: CurrentUser, db: Database,
                 welcome: int = 0, ok: str = "", error: str = ""):
    pending = db.get_pending_profile(user.id)
    if pending:
        # Show the imported draft, clearly flagged. It is not profile data
        # until the user reviews it and presses Save.
        profile = pending.get("profile") or blank_profile()
        uncertain = pending.get("uncertain") or []
    else:
        profile = db.get_profile(user.id)
        uncertain = []
        if profile is None:
            profile = blank_profile()
            profile["contact"]["email"] = user.email

    return render(
        request, "profile.html", user, db,
        profile=profile,
        readiness=completeness(profile),
        legal_fields=LEGAL_FIELDS,
        disclosure_fields=DISCLOSURE_FIELDS,
        pending_import=bool(pending),
        uncertain=uncertain,
        max_upload_mb=_max_upload_mb(),
        welcome=bool(welcome), ok=ok, error=error,
    )


def _max_upload_mb() -> int:
    from engine.resume_import import MAX_UPLOAD_BYTES

    return MAX_UPLOAD_BYTES // 1_048_576


@router.post("/profile/import")
async def import_resume(request: Request, user: CurrentUser, db: Database,
                        _csrf: None = CSRFProtected):
    """Read an uploaded resume into a draft profile for review."""
    from engine.resume_import import (
        MAX_UPLOAD_BYTES,
        ResumeImportError,
        ResumeImporter,
        detect_kind,
        extract_text,
        to_profile,
    )

    form = await request.form()
    upload = form.get("resume")
    if upload is None or not getattr(upload, "filename", ""):
        return _redirect("/profile", error="Choose a resume file to import.")

    api_key = db.get_anthropic_key(user.id)
    if not api_key and not settings.ANTHROPIC_API_KEY:
        return _redirect("/settings", error="Add your Anthropic API key before importing.")

    try:
        data = await upload.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise ResumeImportError(
                f"File is too large; the limit is {MAX_UPLOAD_BYTES // 1_048_576}MB.")
        kind = detect_kind(upload.filename, getattr(upload, "content_type", "") or "")
        text = extract_text(data, kind)
    except ResumeImportError as exc:
        return _redirect("/profile", error=str(exc)[:250])
    finally:
        # The document itself is never stored: only what was read out of it.
        data = b""

    importer = ResumeImporter(api_key=api_key)
    try:
        imported = importer.parse(text)
    except Exception as exc:
        db.log_event(user.id, "resume_import_failed", str(exc)[:300], level=LogLevel.ERROR)
        return _redirect("/profile", error=f"Could not read that resume: {exc}"[:250])

    draft = to_profile(imported, db.get_profile(user.id))
    db.save_pending_profile(user.id, draft, imported.uncertain)
    db.log_event(user.id, "resume_imported",
                 f"Imported {upload.filename}: {len(imported.experience)} role(s), "
                 f"{len(imported.uncertain)} item(s) to check")

    return _redirect("/profile",
                     ok=f"Read {len(imported.experience)} role(s) from {upload.filename}. "
                        "Check everything below, then Save.")


@router.post("/profile/import/discard")
def discard_import(request: Request, user: CurrentUser, db: Database,
                   _csrf: None = CSRFProtected):
    db.clear_pending_profile(user.id)
    return _redirect("/profile", ok="Import discarded.")


@router.post("/profile")
async def save_profile(request: Request, user: CurrentUser, db: Database,
                       _csrf: None = CSRFProtected):
    form = await request.form()
    profile = parse_profile_form(form)

    # Validate against the canonical schema before storing, so a malformed
    # profile is rejected here rather than mid-run.
    try:
        from engine.schemas import MasterProfile

        MasterProfile.model_validate(profile)
    except Exception as exc:
        return _redirect("/profile", error=f"Profile is not valid: {exc}"[:300])

    db.save_profile(user.id, profile)
    db.clear_pending_profile(user.id)
    db.log_event(user.id, "profile_saved",
                 f"Profile saved ({completeness(profile)['percent']}% complete)")
    return _redirect("/profile", ok="Profile saved")


# --------------------------------------------------------------------------
# Applications
# --------------------------------------------------------------------------


@router.get("/applications", response_class=HTMLResponse)
def applications_page(request: Request, user: CurrentUser, db: Database,
                      status: str = "", ok: str = "", error: str = ""):
    status_filter = None
    if status:
        try:
            status_filter = ApplicationStatus.from_str(status)
        except ValueError:
            status_filter = None
    return render(
        request, "applications.html", user, db,
        applications=db.list_applications(status=status_filter, limit=100, user_id=user.id),
        statuses=list(ApplicationStatus),
        current_status=status, ok=ok, error=error,
    )


@router.post("/applications/new")
async def new_application(request: Request, user: CurrentUser, db: Database,
                          _csrf: None = CSRFProtected):
    """Enqueue one posting, or a whole batch pasted one URL per line."""
    import secrets

    form = await request.form()
    raw = str(form.get("job_url", ""))
    jd_text = str(form.get("job_description", "")).strip() or None

    urls, invalid = [], []
    for line in raw.replace(",", "\n").splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        (urls if candidate.startswith(("http://", "https://")) else invalid).append(candidate)

    if not urls:
        return _redirect("/applications", error="Enter at least one job posting URL.")
    if len(urls) > MAX_BATCH:
        return _redirect("/applications",
                         error=f"Queue at most {MAX_BATCH} postings at a time.")

    profile = db.get_profile(user.id)
    if not profile or not completeness(profile)["ready"]:
        return _redirect("/profile", error="Complete your profile before applying.")
    if not db.get_anthropic_key(user.id) and not settings.ANTHROPIC_API_KEY:
        return _redirect("/settings", error="Add your Anthropic API key first.")

    # A pasted job description only makes sense for a single posting.
    batch_id = secrets.token_urlsafe(8) if len(urls) > 1 else None
    for url in urls:
        db.enqueue_job(user.id, kind="tailor", job_url=url,
                       job_description=jd_text if len(urls) == 1 else None,
                       batch_id=batch_id)

    db.log_event(user.id, "queued", f"Queued {len(urls)} posting(s) for tailoring")
    note = f"Queued {len(urls)} posting(s)."
    if invalid:
        note += f" Skipped {len(invalid)} line(s) that were not http(s) URLs."
    return _redirect("/queue", ok=note)


@router.get("/applications/{app_id}", response_class=HTMLResponse)
def application_detail(request: Request, user: CurrentUser, db: Database, app_id: int,
                       ok: str = "", error: str = ""):
    app = db.get_application(app_id, user_id=user.id)
    if app is None:
        return _redirect("/applications", error="Application not found.")

    tailored = None
    if app.tailored_payload:
        try:
            tailored = json.loads(app.tailored_payload)
        except json.JSONDecodeError:
            tailored = None

    credential = db.get_credential(app.job_url, user_id=user.id) if app.job_url else None

    return render(
        request, "application_detail.html", user, db,
        app=app,
        tailored=tailored,
        credential=credential,
        actions=db.list_actions(user_id=user.id, status=None, application_id=app_id),
        logs=db.list_logs(user_id=user.id, application_id=app_id, limit=100),
        statuses=list(ApplicationStatus),
        resume_exists=bool(app.resume_pdf_path and Path(app.resume_pdf_path).is_file()),
        ok=ok, error=error,
    )


@router.post("/applications/{app_id}/feedback")
def submit_feedback(request: Request, user: CurrentUser, db: Database, app_id: int,
                    status: str = Form(...), notes: str = Form(""),
                    _csrf: None = CSRFProtected):
    try:
        new_status = ApplicationStatus.from_str(status)
    except ValueError:
        return _redirect(f"/applications/{app_id}", error="Unknown status.")

    db.record_feedback(app_id, new_status, notes, user_id=user.id)
    db.log_event(user.id, "feedback", f"{new_status.value}: {notes[:200]}",
                 application_id=app_id)
    return _redirect(f"/applications/{app_id}", ok=f"Recorded as {new_status.value}")


@router.post("/applications/{app_id}/apply")
def run_apply(request: Request, user: CurrentUser, db: Database, app_id: int,
              _csrf: None = CSRFProtected):
    app = db.get_application(app_id, user_id=user.id)
    if app is None:
        return _redirect("/applications", error="Application not found.")
    db.enqueue_job(user.id, kind="apply", job_url=app.job_url, application_id=app_id)
    return _redirect("/queue",
                     ok="Queued. Anything needing you will appear under Needs you.")


@router.get("/applications/{app_id}/resume")
def download_resume(request: Request, user: CurrentUser, db: Database, app_id: int):
    app = db.get_application(app_id, user_id=user.id)
    if app is None or not app.resume_pdf_path:
        return _redirect("/applications", error="Resume not found.")
    path = Path(app.resume_pdf_path)
    if not path.is_file():
        return _redirect(f"/applications/{app_id}", error="Resume file is missing on disk.")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


@router.get("/discover", response_class=HTMLResponse)
def discover_page(request: Request, user: CurrentUser, db: Database,
                  ok: str = "", error: str = ""):
    from engine.discovery import DiscoveryCriteria

    profile = db.get_profile(user.id) or {}
    saved = db.get_discovery_criteria(user.id)
    criteria = DiscoveryCriteria.model_validate(saved) if saved else DiscoveryCriteria(
        titles=profile.get("target_titles") or [],
        locations=[str((profile.get("contact") or {}).get("location", {}).get("city", ""))]
        if profile else [],
    )
    runs = db.list_jobs(user_id=user.id, limit=50)
    return render(
        request, "discover.html", user, db,
        criteria=criteria,
        recent=[j for j in runs if j.kind == "discover"][:10],
        tailored_today=db.applications_today(user.id),
        daily_cap=settings.DAILY_APPLICATION_CAP,
        submitted_today=db.submitted_today(user.id),
        awaiting_agent=db.awaiting_agent(user.id),
        spend_today=db.spend_today(user.id),
        spend_cap=db.daily_cap_for(user.id),
        discovery_model=settings.LLM_MODEL_DISCOVERY,
        bulk_model=db.model_for_job(user.id, priority=False),
        ok=ok, error=error,
    )


@router.post("/discover")
async def start_discovery(request: Request, user: CurrentUser, db: Database,
                          _csrf: None = CSRFProtected):
    from engine.discovery import DiscoveryCriteria

    form = await request.form()

    def lines(key: str) -> list[str]:
        raw = str(form.get(key, "") or "")
        return [part.strip() for part in raw.replace("\n", ",").split(",") if part.strip()]

    try:
        criteria = DiscoveryCriteria(
            titles=lines("titles"),
            locations=lines("locations"),
            seniority=str(form.get("seniority", "")).strip(),
            industries=lines("industries"),
            company_size=str(form.get("company_size", "")).strip(),
            remote_only=str(form.get("remote_only", "")).lower() in ("on", "true", "1"),
            exclude_companies=lines("exclude_companies"),
            max_companies=int(form.get("max_companies") or 15),
            max_postings=int(form.get("max_postings") or settings.DAILY_APPLICATION_CAP),
        )
    except (ValueError, TypeError) as exc:
        return _redirect("/discover", error=f"Those criteria are not valid: {exc}"[:200])

    if not criteria.titles:
        return _redirect("/discover", error="Give at least one job title to search for.")

    profile = db.get_profile(user.id)
    if not profile or not completeness(profile)["ready"]:
        return _redirect("/profile", error="Complete your profile before discovering roles.")
    if not db.get_anthropic_key(user.id) and not settings.ANTHROPIC_API_KEY:
        return _redirect("/settings", error="Add your Anthropic API key first.")

    db.save_discovery_criteria(user.id, criteria.model_dump())
    db.enqueue_job(user.id, kind="discover",
                   job_description=criteria.model_dump_json())
    db.log_event(user.id, "discovery_queued", f"Searching for: {criteria.describe()}")
    return _redirect("/queue", ok="Searching for roles. Postings will appear in the queue.")


# --------------------------------------------------------------------------
# Batch queue
# --------------------------------------------------------------------------


@router.get("/queue", response_class=HTMLResponse)
def queue_page(request: Request, user: CurrentUser, db: Database,
               ok: str = "", error: str = ""):
    jobs = db.list_jobs(user_id=user.id, limit=200)
    summary = db.queue_summary(user.id)
    blocked = [j for j in jobs if j.status is JobStatus.BLOCKED]
    return render(
        request, "queue.html", user, db,
        jobs=jobs, summary=summary, blocked=blocked,
        held=sum(1 for j in blocked if j.holds_browser),
        max_held=_max_held(),
        ok=ok, error=error,
    )


def _desktop_available() -> bool:
    from automation.notifier import DesktopChannel

    return DesktopChannel().available()


def _max_held() -> int:
    from web.queue_worker import MAX_HELD_BROWSERS

    return MAX_HELD_BROWSERS


@router.post("/queue/{job_id}/cancel")
def cancel_job(request: Request, user: CurrentUser, db: Database, job_id: int,
               _csrf: None = CSRFProtected):
    if db.get_job(job_id, user_id=user.id) is None:
        return _redirect("/queue", error="Job not found.")
    get_worker().registry.cancel(job_id)
    db.cancel_job(job_id, user_id=user.id)
    return _redirect("/queue", ok=f"Job #{job_id} cancelled.")


@router.post("/queue/clear-finished")
def clear_finished(request: Request, user: CurrentUser, db: Database,
                   _csrf: None = CSRFProtected):
    removed = db.purge_finished_jobs(user.id)
    return _redirect("/queue", ok=f"Cleared {removed} finished job(s).")


# --------------------------------------------------------------------------
# Action queue
# --------------------------------------------------------------------------


@router.get("/actions", response_class=HTMLResponse)
def actions_page(request: Request, user: CurrentUser, db: Database,
                 show: str = "open", ok: str = "", error: str = ""):
    status = {"open": ActionStatus.OPEN, "answered": ActionStatus.ANSWERED,
              "all": None}.get(show, ActionStatus.OPEN)
    items = db.list_actions(user_id=user.id, status=status, limit=200)
    blocked_jobs = {
        job.blocking_action_id: job
        for job in db.list_jobs(user_id=user.id, statuses=(JobStatus.BLOCKED,))
    }
    for item in items:
        item.options = json.loads(item.options_json) if item.options_json else []
        job = blocked_jobs.get(item.id)
        # Surfaced so you can tell what is merely informational from what is
        # actually holding a run (and a browser window) open.
        item.blocking_job = job
        item.holds_browser = bool(job and job.holds_browser)

    open_count = db.count_open_actions(user.id)
    return render(request, "actions.html", user, db, items=items, show=show,
                  open_count=open_count, blocked_count=len(blocked_jobs),
                  held_count=sum(1 for j in blocked_jobs.values() if j.holds_browser),
                  ok=ok, error=error)


@router.post("/actions/{action_id}/answer")
def answer_action(request: Request, user: CurrentUser, db: Database, action_id: int,
                  answer: str = Form(...), remember: str = Form(""),
                  _csrf: None = CSRFProtected):
    db.answer_action(action_id, answer.strip(),
                     remember=str(remember).lower() in ("on", "true", "1"),
                     user_id=user.id)
    resumed = get_worker().release_action(action_id, answer.strip())
    db.log_event(user.id, "action_answered",
                 f"#{action_id}: {answer[:120]} (resumed {resumed} job(s))")

    remaining = db.count_open_actions(user.id)
    note = "Answer saved."
    if resumed:
        note += f" Resumed {resumed} job(s)."
    if remaining:
        note += f" {remaining} item(s) still waiting."
    return _redirect("/actions", ok=note)


@router.post("/actions/{action_id}/dismiss")
def dismiss_action(request: Request, user: CurrentUser, db: Database, action_id: int,
                   _csrf: None = CSRFProtected):
    db.dismiss_action(action_id, user_id=user.id)
    # A job parked on this action would otherwise wait out its whole timeout.
    for job in db.jobs_blocked_on_action(action_id):
        get_worker().registry.cancel(job.id)
        db.cancel_job(job.id, user_id=user.id)
    return _redirect("/actions", ok="Dismissed; any job waiting on it was cancelled.")


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------


@router.get("/logs", response_class=HTMLResponse)
def logs_page(request: Request, user: CurrentUser, db: Database, level: str = "", limit: int = 200):
    level_filter = None
    if level:
        try:
            level_filter = LogLevel(level.upper())
        except ValueError:
            level_filter = None
    return render(
        request, "logs.html", user, db,
        logs=db.list_logs(user_id=user.id, level=level_filter, limit=min(limit, 1000)),
        levels=list(LogLevel), current_level=level,
        retention_days=settings.LOG_RETENTION_DAYS,
    )


# --------------------------------------------------------------------------
# Costs
# --------------------------------------------------------------------------


@router.get("/costs", response_class=HTMLResponse)
def costs_page(request: Request, user: CurrentUser, db: Database, days: int = 30):
    if days not in COST_WINDOWS:
        days = 30
    series = daily_series(db.usage_rows(user.id, days=days), days=days)
    summary = summarize(series)

    # `height` is a whole percentage of the chart's drawing area, resolved to a
    # `.hNN` class in the stylesheet. It cannot be an inline style: the CSP sets
    # `style-src 'self'`, so the browser discards inline style attributes and
    # every bar would render at zero height.
    peak = max((e["cost"] for e in series), default=0.0) or 1.0
    chart = [
        {**entry,
         "height": max(round(100 * entry["cost"] / peak), 1) if entry["cost"] else 0,
         "label": entry["date"].strftime("%b %d")}
        for entry in series
    ]

    return render(
        request, "costs.html", user, db,
        days=days, windows=COST_WINDOWS, series=chart, summary=summary,
        by_model=db.usage_by_model(user.id, days),
        by_phase=db.usage_by_phase(user.id, days),
        per_application=db.cost_per_application(user.id, days),
        pricing_as_of=PRICING_AS_OF.isoformat(),
    )


# --------------------------------------------------------------------------
# Admin
# --------------------------------------------------------------------------


@router.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request, user: AdminUser, db: Database,
               ok: str = "", error: str = ""):
    users = db.user_overview()
    return render(
        request, "admin.html", user, db,
        users=users,
        totals={
            "users": len(users),
            "applications": sum(u["applications"] for u in users),
            "spend": round(sum(u["llm_spend"] for u in users), 4),
            "with_keys": sum(1 for u in users if u["has_anthropic_key"]),
        },
        invite_required=bool(settings.INVITE_CODE),
        ok=ok, error=error,
    )


@router.post("/admin/users/{target_id}/active")
def set_user_active(request: Request, user: AdminUser, db: Database, target_id: int,
                    active: str = Form("false"), _csrf: None = CSRFProtected):
    if target_id == user.id:
        return _redirect("/admin", error="You cannot suspend your own account.")
    enable = str(active).lower() in ("true", "on", "1")
    db.set_user_active(target_id, enable)
    db.log_event(user.id, "admin_user_active",
                 f"{'Restored' if enable else 'Suspended'} user #{target_id}",
                 level=LogLevel.WARNING)
    return _redirect("/admin", ok=f"User #{target_id} "
                                  f"{'restored' if enable else 'suspended'}.")


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, user: CurrentUser, db: Database,
                  ok: str = "", error: str = "", new_key: str = "", reveal: int = 0):
    stored = db.get_anthropic_key(user.id)
    credentials = db.list_credentials(user_id=user.id)
    # Only the one credential the user explicitly asked to see is decrypted.
    revealed = ""
    if reveal:
        match = next((c for c in credentials if c.id == reveal), None)
        if match:
            revealed = db.decrypt(match.encrypted_password)
    return render(
        request, "settings.html", user, db,
        reveal_id=reveal, revealed_password=revealed,
        anthropic_key_preview=mask_secret(stored) if stored else "",
        has_anthropic_key=bool(stored),
        api_key_prefix=user.api_key_prefix,
        notifications=db.list_notifications(user.id, limit=8),
        desktop_available=_desktop_available(),
        auto_apply_threshold=db.auto_apply_threshold(user.id),
        spend_cap=db.daily_cap_for(user.id),
        spend_today=db.spend_today(user.id),
        api_key_created=user.api_key_created_at,
        new_key=new_key,
        credentials=credentials,
        model=settings.LLM_MODEL,
        ok=ok, error=error,
    )


@router.post("/settings/automation")
def save_automation(request: Request, user: CurrentUser, db: Database,
                    auto_apply_enabled: str = Form(""),
                    auto_apply_threshold: str = Form("85"),
                    daily_spend_cap_usd: str = Form(""),
                    _csrf: None = CSRFProtected):
    enabled = str(auto_apply_enabled).lower() in ("on", "true", "1")
    threshold = None
    if enabled:
        try:
            threshold = float(auto_apply_threshold)
        except (TypeError, ValueError):
            return _redirect("/settings", error="The threshold must be a number.")
        if not 0 < threshold <= 100:
            return _redirect("/settings", error="The threshold must be between 1 and 100.")

    cap = None
    if str(daily_spend_cap_usd).strip():
        try:
            cap = max(0.0, float(daily_spend_cap_usd))
        except (TypeError, ValueError):
            return _redirect("/settings", error="The spend cap must be a number.")

    db.update_user(user.id, auto_apply_threshold=threshold, daily_spend_cap_usd=cap)
    db.log_event(user.id, "automation_settings",
                 f"Auto-apply {'at ' + str(int(threshold)) + '%' if threshold else 'off'}"
                 f"; spend cap {'$' + str(cap) if cap is not None else 'default'}")
    return _redirect("/settings", ok="Automation settings saved.")


@router.post("/settings/anthropic-key")
def save_anthropic_key(request: Request, user: CurrentUser, db: Database,
                       anthropic_key: str = Form(""), _csrf: None = CSRFProtected):
    key = anthropic_key.strip()
    if key and not key.startswith("sk-ant-"):
        return _redirect("/settings", error="That does not look like an Anthropic API key.")
    db.set_anthropic_key(user.id, key or None)
    db.log_event(user.id, "anthropic_key",
                 "API key stored (encrypted)" if key else "API key removed")
    return _redirect("/settings", ok="API key saved." if key else "API key removed.")


@router.post("/settings/notifications")
def save_notifications(request: Request, user: CurrentUser, db: Database,
                       notify_desktop: str = Form(""),
                       notify_webhook_url: str = Form(""),
                       notify_quiet_seconds: str = Form("120"),
                       _csrf: None = CSRFProtected):
    url = notify_webhook_url.strip()
    if url and not url.startswith(("http://", "https://")):
        return _redirect("/settings", error="Webhook URL must start with http:// or https://")
    try:
        quiet = max(0, min(int(notify_quiet_seconds or 0), 3600))
    except ValueError:
        quiet = 120

    db.update_user(
        user.id,
        notify_desktop=str(notify_desktop).lower() in ("on", "true", "1"),
        notify_webhook_url=url or None,
        notify_quiet_seconds=quiet,
    )
    db.log_event(user.id, "notify_settings", "Notification preferences updated")
    return _redirect("/settings", ok="Notification settings saved.")


@router.post("/settings/notifications/test")
def test_notification(request: Request, user: CurrentUser, db: Database,
                      _csrf: None = CSRFProtected):
    from automation.notifier import Notice, Notifier

    delivered = Notifier(db).notify(
        db.get_user(user.id),
        Notice(title="Job pipeline: test",
               body="If you can see this, notifications are working.",
               url="/queue"),
    )
    if delivered:
        return _redirect("/settings", ok=f"Test sent via {', '.join(delivered)}.")
    return _redirect("/settings",
                     error="Nothing was delivered. Check the channels below and the log.")


@router.post("/settings/api-key")
def rotate_api_key(request: Request, user: CurrentUser, db: Database,
                   _csrf: None = CSRFProtected):
    from web.security import generate_api_key

    raw, key_hash, prefix = generate_api_key()
    db.update_user(user.id, api_key_hash=key_hash, api_key_prefix=prefix,
                   api_key_created_at=datetime.now(timezone.utc))
    db.log_event(user.id, "api_key_rotated", "Dashboard API key rotated")
    # Shown exactly once; only the hash is stored.
    return _redirect(f"/settings?new_key={quote_plus(raw)}", ok="New API key generated.")


@router.post("/settings/api-key/revoke")
def revoke_api_key(request: Request, user: CurrentUser, db: Database,
                   _csrf: None = CSRFProtected):
    db.update_user(user.id, api_key_hash=None, api_key_prefix=None, api_key_created_at=None)
    db.log_event(user.id, "api_key_revoked", "Dashboard API key revoked")
    return _redirect("/settings", ok="API key revoked.")


@router.post("/settings/credentials/{cred_id}/reveal")
def reveal_credential(request: Request, user: CurrentUser, db: Database, cred_id: int,
                      _csrf: None = CSRFProtected):
    creds = {c.id: c for c in db.list_credentials(user_id=user.id)}
    cred = creds.get(cred_id)
    if cred is None:
        return _redirect("/settings", error="Credential not found.")
    db.log_event(user.id, "credential_revealed",
                 f"Password shown for {cred.portal_domain}", level=LogLevel.WARNING)
    return _redirect(f"/settings?reveal={cred_id}")
