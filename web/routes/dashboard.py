"""HTML pages: overview, profile, applications, actions, logs, costs, settings."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

from fastapi import APIRouter, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from config import settings
from database.models import ActionStatus, ApplicationStatus, LogLevel
from engine.cost_tracker import COST_WINDOWS, PRICING_AS_OF, daily_series, summarize
from web.deps import CSRFProtected, CurrentUser, Database
from web.profile_form import (
    DISCLOSURE_FIELDS,
    LEGAL_FIELDS,
    blank_profile,
    completeness,
    parse_profile_form,
)
from web.security import CSRF_COOKIE, mask_secret
from web.runner import runs

router = APIRouter(tags=["dashboard"])



def render(request: Request, template: str, user, db, **context) -> HTMLResponse:
    from web.app import templates

    base = {
        "user": user,
        "csrf_token": getattr(request.state, "csrf_token", ""),
        "open_actions": db.count_open_actions(user.id) if user else 0,
        "active_runs": [r.as_dict() for r in runs.active_for_user(user.id)] if user else [],
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
    profile = db.get_profile(user.id)
    if profile is None:
        profile = blank_profile()
        profile["contact"]["email"] = user.email
    return render(
        request, "profile.html", user, db,
        profile=profile,
        readiness=completeness(profile),
        legal_fields=LEGAL_FIELDS,
        disclosure_fields=DISCLOSURE_FIELDS,
        welcome=bool(welcome), ok=ok, error=error,
    )


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
    form = await request.form()
    url = str(form.get("job_url", "")).strip()
    jd_text = str(form.get("job_description", "")).strip() or None

    if not url.startswith(("http://", "https://")):
        return _redirect("/applications", error="Enter a valid job posting URL.")

    profile = db.get_profile(user.id)
    if not profile or not completeness(profile)["ready"]:
        return _redirect("/profile", error="Complete your profile before applying.")
    if not db.get_anthropic_key(user.id) and not settings.ANTHROPIC_API_KEY:
        return _redirect("/settings", error="Add your Anthropic API key first.")

    runs.submit_tailor(db, user.id, url, jd_text)
    return _redirect("/applications", ok="Tailoring started. Watch the logs for progress.")


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
        resume_exists=bool(app.resume_pdf_path and Path(app.resume_pdf_path).exists()),
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
    runs.submit_apply(db, user.id, app_id)
    return _redirect(f"/applications/{app_id}",
                     ok="Browser session starting. Anything needing you appears in Actions.")


@router.get("/applications/{app_id}/resume")
def download_resume(request: Request, user: CurrentUser, db: Database, app_id: int):
    app = db.get_application(app_id, user_id=user.id)
    if app is None or not app.resume_pdf_path:
        return _redirect("/applications", error="Resume not found.")
    path = Path(app.resume_pdf_path)
    if not path.exists():
        return _redirect(f"/applications/{app_id}", error="Resume file is missing on disk.")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


# --------------------------------------------------------------------------
# Action queue
# --------------------------------------------------------------------------


@router.get("/actions", response_class=HTMLResponse)
def actions_page(request: Request, user: CurrentUser, db: Database,
                 show: str = "open", ok: str = "", error: str = ""):
    status = {"open": ActionStatus.OPEN, "answered": ActionStatus.ANSWERED,
              "all": None}.get(show, ActionStatus.OPEN)
    items = db.list_actions(user_id=user.id, status=status, limit=200)
    for item in items:
        item.options = json.loads(item.options_json) if item.options_json else []
    return render(request, "actions.html", user, db, items=items, show=show,
                  ok=ok, error=error)


@router.post("/actions/{action_id}/answer")
def answer_action(request: Request, user: CurrentUser, db: Database, action_id: int,
                  answer: str = Form(...), remember: str = Form(""),
                  _csrf: None = CSRFProtected):
    db.answer_action(action_id, answer.strip(),
                     remember=str(remember).lower() in ("on", "true", "1"),
                     user_id=user.id)
    db.log_event(user.id, "action_answered", f"#{action_id}: {answer[:120]}")
    return _redirect("/actions", ok="Answer saved. A paused run will pick it up.")


@router.post("/actions/{action_id}/dismiss")
def dismiss_action(request: Request, user: CurrentUser, db: Database, action_id: int,
                   _csrf: None = CSRFProtected):
    db.dismiss_action(action_id, user_id=user.id)
    return _redirect("/actions", ok="Dismissed.")


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
        api_key_created=user.api_key_created_at,
        new_key=new_key,
        credentials=credentials,
        model=settings.LLM_MODEL,
        ok=ok, error=error,
    )


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
