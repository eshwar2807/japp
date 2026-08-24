"""JSON API, authenticated with a bearer API key.

Every endpoint is scoped to the authenticated user; there is no cross-tenant
read path. Rate limiting is applied in the `api_user` dependency.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

from database.models import ActionKind, ActionStatus, ApplicationStatus, JobStatus, LogLevel
from engine.cost_tracker import COST_WINDOWS, daily_series, summarize
from web.deps import APIUser, Database, get_worker

router = APIRouter(prefix="/api/v1", tags=["api"])


def _application_json(app: Any, include_detail: bool = False) -> dict[str, Any]:
    data = {
        "id": app.id,
        "company": app.company,
        "role_title": app.role_title,
        "job_url": app.job_url,
        "match_score": app.match_score,
        "status": app.status.value,
        "portal_domain": app.portal_domain,
        "created_at": app.created_at.isoformat(),
        "submitted_at": app.submitted_at.isoformat() if app.submitted_at else None,
    }
    if include_detail:
        data["job_description"] = app.job_description
        data["notes"] = app.notes
        data["resume_pdf_path"] = app.resume_pdf_path
        try:
            data["tailored"] = json.loads(app.tailored_payload or "{}")
        except json.JSONDecodeError:
            data["tailored"] = {}
        data["feedback"] = [
            {"status": f.status.value, "notes": f.notes, "created_at": f.created_at.isoformat()}
            for f in app.feedback
        ]
    return data


# --------------------------------------------------------------------------


@router.get("/me")
def me(user: APIUser, db: Database):
    profile = db.get_profile(user.id)
    from web.profile_form import completeness

    return {
        "id": user.id,
        "email": user.email,
        "profile_complete": completeness(profile)["ready"] if profile else False,
        "has_anthropic_key": bool(user.encrypted_anthropic_key),
        "stats": db.stats(user_id=user.id),
    }


@router.get("/applications")
def list_applications(user: APIUser, db: Database,
                      status: str = Query(default=""), limit: int = Query(default=50, le=200)):
    status_filter = None
    if status:
        try:
            status_filter = ApplicationStatus.from_str(status)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    apps = db.list_applications(status=status_filter, limit=limit, user_id=user.id)
    return {"count": len(apps), "applications": [_application_json(a) for a in apps]}


@router.get("/applications/{app_id}")
def get_application(user: APIUser, db: Database, app_id: int):
    app = db.get_application(app_id, user_id=user.id)
    if app is None:
        raise HTTPException(404, "Application not found.")
    return _application_json(app, include_detail=True)


@router.post("/applications", status_code=202)
def create_application(request: Request, user: APIUser, db: Database,
                       payload: dict = Body(...)):
    url = str(payload.get("job_url", "")).strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "job_url must be an absolute http(s) URL.")
    if not db.get_profile(user.id):
        raise HTTPException(409, "Complete your profile first.")

    from web.deps import enforce_rate_limit

    enforce_rate_limit(request, "run", str(user.id))
    job = db.enqueue_job(
        user.id, kind="tailor", job_url=url,
        job_description=str(payload.get("job_description", "")).strip() or None,
    )
    return {"job_id": job.id, "status": job.status.value, "job_url": job.job_url}


@router.post("/applications/{app_id}/feedback")
def post_feedback(user: APIUser, db: Database, app_id: int, payload: dict = Body(...)):
    try:
        status = ApplicationStatus.from_str(str(payload.get("status", "")))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if db.get_application(app_id, user_id=user.id) is None:
        raise HTTPException(404, "Application not found.")
    db.record_feedback(app_id, status, str(payload.get("notes", "")), user_id=user.id)
    return {"ok": True, "application_id": app_id, "status": status.value}


@router.get("/actions")
def list_actions(user: APIUser, db: Database, show: str = Query(default="open")):
    status = {"open": ActionStatus.OPEN, "answered": ActionStatus.ANSWERED,
              "all": None}.get(show, ActionStatus.OPEN)
    items = db.list_actions(user_id=user.id, status=status)
    return {
        "count": len(items),
        "actions": [
            {
                "id": i.id,
                "kind": i.kind.value,
                "status": i.status.value,
                "question": i.question,
                "reason": i.reason,
                "options": json.loads(i.options_json) if i.options_json else [],
                "required": i.required,
                "application_id": i.application_id,
                "created_at": i.created_at.isoformat(),
            }
            for i in items
        ],
    }


@router.post("/actions/{action_id}/answer")
def answer_action(user: APIUser, db: Database, action_id: int, payload: dict = Body(...)):
    answer = str(payload.get("answer", "")).strip()
    if not answer:
        raise HTTPException(400, "answer is required.")
    item = db.answer_action(action_id, answer, bool(payload.get("remember")), user_id=user.id)
    resumed = get_worker().release_action(action_id, answer)
    return {"ok": True, "id": item.id, "status": item.status.value, "jobs_resumed": resumed}


@router.get("/costs")
def costs(user: APIUser, db: Database, days: int = Query(default=30)):
    if days not in COST_WINDOWS:
        raise HTTPException(400, f"days must be one of {list(COST_WINDOWS)}.")
    series = daily_series(db.usage_rows(user.id, days=days), days=days)
    return {
        "days": days,
        "summary": summarize(series),
        "by_model": db.usage_by_model(user.id, days),
        "by_phase": db.usage_by_phase(user.id, days),
        "cost_per_application": db.cost_per_application(user.id, days),
        "series": [
            {"date": e["date"].isoformat(), "cost": e["cost"],
             "tokens": e["tokens"], "calls": e["calls"]}
            for e in series
        ],
    }


@router.get("/logs")
def logs(user: APIUser, db: Database, limit: int = Query(default=100, le=1000),
         level: str = Query(default=""), since_id: int | None = Query(default=None)):
    level_filter = None
    if level:
        try:
            level_filter = LogLevel(level.upper())
        except ValueError as exc:
            raise HTTPException(400, "Unknown log level.") from exc
    rows = db.list_logs(user_id=user.id, level=level_filter, limit=limit, since_id=since_id)
    return {
        "count": len(rows),
        "logs": [
            {"id": r.id, "level": r.level.value, "event": r.event, "message": r.message,
             "application_id": r.application_id, "created_at": r.created_at.isoformat()}
            for r in rows
        ],
    }


def _job_json(job) -> dict:
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status.value,
        "block_mode": job.block_mode.value,
        "blocking_action_id": job.blocking_action_id,
        "holds_browser": job.holds_browser,
        "message": job.message,
        "job_url": job.job_url,
        "application_id": job.application_id,
        "batch_id": job.batch_id,
        "created_at": job.created_at.isoformat(),
        "blocked_at": job.blocked_at.isoformat() if job.blocked_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


@router.get("/queue")
def list_queue(user: APIUser, db: Database, status: str = Query(default="")):
    statuses = None
    if status:
        try:
            statuses = (JobStatus(status),)
        except ValueError as exc:
            valid = ", ".join(s.value for s in JobStatus)
            raise HTTPException(400, f"status must be one of: {valid}") from exc
    jobs = db.list_jobs(user_id=user.id, statuses=statuses)
    return {"summary": db.queue_summary(user.id), "jobs": [_job_json(j) for j in jobs]}


@router.get("/queue/{job_id}")
def get_job(user: APIUser, db: Database, job_id: int):
    job = db.get_job(job_id, user_id=user.id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return _job_json(job)


@router.post("/queue/{job_id}/cancel")
def cancel_job(user: APIUser, db: Database, job_id: int):
    if db.get_job(job_id, user_id=user.id) is None:
        raise HTTPException(404, "Job not found.")
    get_worker().registry.cancel(job_id)
    job = db.cancel_job(job_id, user_id=user.id)
    return _job_json(job)


# --------------------------------------------------------------------------
# Local agent
# --------------------------------------------------------------------------
#
# The hosted dashboard cannot run application forms: a verification challenge or
# a submit review needs a browser window a human can see, and there is none in a
# datacenter. So `apply` jobs are claimed over this API by an agent running on
# the user's own machine, which drives the browser locally and reports back.
#
# Everything here is scoped to the authenticated key's user, exactly like the
# rest of the API.


@router.post("/agent/claim")
def agent_claim(user: APIUser, db: Database):
    """Claim the next apply job for this user, with everything needed to run it.

    Returns 204 when there is nothing to do, so an idle agent costs one small
    request per poll.
    """
    from fastapi.responses import Response

    job = db.claim_next_job(kinds=("apply",), user_id=user.id)
    if job is None:
        return Response(status_code=204)

    application = db.get_application(job.application_id, user_id=user.id) if job.application_id else None
    if application is None:
        db.finish_job(job.id, JobStatus.FAILED, "Job has no application.")
        raise HTTPException(409, "Job has no application attached.")

    profile = db.get_profile(user.id)
    if not profile:
        db.finish_job(job.id, JobStatus.FAILED, "No profile configured.")
        raise HTTPException(409, "Fill in your profile first.")

    try:
        tailored = json.loads(application.tailored_payload or "{}")
    except json.JSONDecodeError:
        tailored = {}

    return {
        "job": _job_json(job),
        "application": {
            "id": application.id,
            "company": application.company,
            "role_title": application.role_title,
            "job_url": application.job_url,
            "salary_min": application.salary_min,
            "salary_max": application.salary_max,
        },
        "profile": profile,
        "tailored": tailored,
        # Answers the user has already given, so the agent never re-asks.
        "known_answers": {**db.answered_action_map(user.id),
                          **(tailored.get("screener_answers") or {})},
        "resume_url": f"/api/v1/applications/{application.id}/resume",
    }


@router.get("/applications/{app_id}/resume")
def agent_download_resume(user: APIUser, db: Database, app_id: int):
    """The tailored PDF, so the agent can attach it locally."""
    from pathlib import Path

    from fastapi.responses import FileResponse

    application = db.get_application(app_id, user_id=user.id)
    if application is None or not application.resume_pdf_path:
        raise HTTPException(404, "Resume not found.")
    path = Path(application.resume_pdf_path)
    if not path.is_file():
        raise HTTPException(404, "Resume file is missing on the server.")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@router.post("/agent/jobs/{job_id}/block")
def agent_block(user: APIUser, db: Database, job_id: int, payload: dict = Body(...)):
    """Report that the agent's run needs a human, and open an action item."""
    from database.models import ActionKind, BlockMode

    job = db.get_job(job_id, user_id=user.id)
    if job is None:
        raise HTTPException(404, "Job not found.")

    kind_map = {
        "captcha": ActionKind.CAPTCHA,
        "account_creation": ActionKind.ACCOUNT_CREATION,
        "submit": ActionKind.SUBMIT_CONFIRMATION,
        "login": ActionKind.LOGIN_REQUIRED,
        "unmapped_field": ActionKind.UNMAPPED_FIELD,
    }
    kind = kind_map.get(str(payload.get("kind", "unmapped_field")), ActionKind.UNMAPPED_FIELD)
    question = str(payload.get("question", "")).strip() or "Needs your input"
    holds_browser = bool(payload.get("holds_browser", True))

    item = db.create_action(
        user_id=user.id, kind=kind, question=question,
        reason=str(payload.get("reason", "")), application_id=job.application_id,
        options=payload.get("options") or None,
        field_type=str(payload.get("field_type", "text")), required=True,
    )
    db.block_job(
        job_id,
        BlockMode.NEEDS_BROWSER if holds_browser else BlockMode.NEEDS_ANSWER,
        question, action_id=item.id, holds_browser=holds_browser,
    )

    # Same notification path the hosted worker uses, so a block reaches you
    # whether it happened here or on your laptop.
    from automation.notifier import Notifier, block_notice

    account = db.get_user(user.id)
    if account and not db.recently_notified(user.id, account.notify_quiet_seconds or 0):
        Notifier(db).notify(account, block_notice(db.get_job(job_id), question, kind.value),
                            job_id=job_id)

    return {"action_id": item.id, "status": "Blocked"}


@router.get("/agent/actions/{action_id}")
def agent_poll_action(user: APIUser, db: Database, action_id: int):
    """Poll one action item; the agent waits on this while parked."""
    items = db.list_actions(user_id=user.id, status=None, limit=500)
    item = next((i for i in items if i.id == action_id), None)
    if item is None:
        raise HTTPException(404, "Action not found.")
    return {"id": item.id, "status": item.status.value, "answer": item.answer}


@router.post("/agent/jobs/{job_id}/finish")
def agent_finish(user: APIUser, db: Database, job_id: int, payload: dict = Body(...)):
    """Report the outcome of an agent-run application."""
    job = db.get_job(job_id, user_id=user.id)
    if job is None:
        raise HTTPException(404, "Job not found.")

    submitted = bool(payload.get("submitted"))
    message = str(payload.get("message", ""))[:500]

    if submitted and job.application_id:
        db.mark_submitted(job.application_id)
    for escalation in payload.get("escalations") or []:
        db.create_action(
            user_id=user.id, kind=ActionKind.UNMAPPED_FIELD,
            question=str(escalation.get("question", ""))[:500],
            reason=str(escalation.get("reason", ""))[:500],
            application_id=job.application_id, required=True,
        )

    db.finish_job(job_id, JobStatus.DONE if submitted else JobStatus.FAILED, message,
                  application_id=job.application_id)
    return {"ok": True, "status": "Done" if submitted else "Failed"}


@router.post("/agent/jobs/{job_id}/log")
def agent_log(user: APIUser, db: Database, job_id: int, payload: dict = Body(...)):
    """Stream a log line from the agent into the dashboard's run log."""
    job = db.get_job(job_id, user_id=user.id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    level = str(payload.get("level", "INFO")).upper()
    try:
        parsed = LogLevel(level)
    except ValueError:
        parsed = LogLevel.INFO
    db.log_event(user.id, f"agent_{str(payload.get('event', 'event'))[:48]}",
                 str(payload.get("message", ""))[:2000], level=parsed,
                 application_id=job.application_id)
    return {"ok": True}
