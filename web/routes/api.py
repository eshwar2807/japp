"""JSON API, authenticated with a bearer API key.

Every endpoint is scoped to the authenticated user; there is no cross-tenant
read path. Rate limiting is applied in the `api_user` dependency.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

from database.models import ActionStatus, ApplicationStatus, JobStatus, LogLevel
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
