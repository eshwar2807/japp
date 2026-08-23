"""Signup, login, logout, password change."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse

from config import settings
from database.models import LogLevel
from web.deps import (
    CSRFProtected,
    CurrentUser,
    Database,
    client_ip,
    enforce_rate_limit,
    get_sessions,
    optional_user,
)
from web.security import (
    SESSION_COOKIE,
    SESSION_MAX_AGE,
    hash_password,
    is_locked,
    lockout_until,
    needs_rehash,
    password_problems,
    valid_email,
    verify_password,
)

router = APIRouter(tags=["auth"])


def _render(request: Request, template: str, **context) -> HTMLResponse:
    from web.app import templates

    context.setdefault("csrf_token", getattr(request.state, "csrf_token", ""))
    return templates.TemplateResponse(request, template, context)


def _set_session(response, user) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        get_sessions().issue(user.id, int(user.session_epoch or 1)),
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite="lax",
        max_age=SESSION_MAX_AGE,
        path="/",
    )


def _safe_next(raw: str | None) -> str:
    """Only allow same-site relative redirects, never an absolute URL."""
    if not raw or not raw.startswith("/") or raw.startswith("//"):
        return "/"
    return raw


# --------------------------------------------------------------------------


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request, next: str = "/"):
    if optional_user(request):
        return RedirectResponse(_safe_next(next), status_code=303)
    return _render(request, "login.html", next=_safe_next(next),
                   allow_signup=settings.ALLOW_SIGNUP)


@router.post("/login")
def login(
    request: Request,
    db: Database,
    email: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
    _csrf: None = CSRFProtected,
):
    enforce_rate_limit(request, "login", client_ip(request))
    enforce_rate_limit(request, "login", email.strip().lower())

    generic = "Email or password is incorrect."
    user = db.get_user_by_email(email)

    if user and is_locked(user.locked_until):
        return _render(request, "login.html", next=_safe_next(next),
                       allow_signup=settings.ALLOW_SIGNUP,
                       error="Account temporarily locked after repeated failed logins.")

    if user is None or not verify_password(password, user.password_hash):
        if user is not None:
            failed = (user.failed_logins or 0) + 1
            db.update_user(user.id, failed_logins=failed, locked_until=lockout_until(failed))
            db.log_event(user.id, "login_failed", f"Failed login from {client_ip(request)}",
                         level=LogLevel.WARNING)
        # Same message and code whether or not the account exists.
        return _render(request, "login.html", next=_safe_next(next),
                       allow_signup=settings.ALLOW_SIGNUP, error=generic)

    updates = {
        "failed_logins": 0,
        "locked_until": None,
        "last_login_at": datetime.now(timezone.utc),
    }
    if needs_rehash(user.password_hash):
        updates["password_hash"] = hash_password(password)
    db.update_user(user.id, **updates)
    db.log_event(user.id, "login", f"Signed in from {client_ip(request)}")

    response = RedirectResponse(_safe_next(next), status_code=303)
    _set_session(response, db.get_user(user.id))
    return response


@router.get("/signup", response_class=HTMLResponse)
def signup_form(request: Request):
    if not settings.ALLOW_SIGNUP:
        return _render(request, "login.html", next="/", allow_signup=False,
                       error="Signup is disabled on this instance.")
    return _render(request, "signup.html")


@router.post("/signup")
def signup(
    request: Request,
    db: Database,
    email: str = Form(...),
    password: str = Form(...),
    confirm: str = Form(...),
    _csrf: None = CSRFProtected,
):
    if not settings.ALLOW_SIGNUP:
        return _render(request, "signup.html", error="Signup is disabled on this instance.")

    enforce_rate_limit(request, "signup", client_ip(request))
    email = email.strip().lower()

    if not valid_email(email):
        return _render(request, "signup.html", email=email, error="Enter a valid email address.")
    if password != confirm:
        return _render(request, "signup.html", email=email, error="Passwords do not match.")

    problems = password_problems(password, email)
    if problems:
        return _render(request, "signup.html", email=email, error=" ".join(problems))

    try:
        user = db.create_user(email, hash_password(password))
    except ValueError:
        # Do not confirm whether an address is already registered.
        return _render(request, "signup.html", email=email,
                       error="Could not create that account. Try logging in instead.")

    db.log_event(user.id, "signup", f"Account created from {client_ip(request)}")
    response = RedirectResponse("/profile?welcome=1", status_code=303)
    _set_session(response, user)
    return response


@router.post("/logout")
def logout(request: Request, db: Database, user: CurrentUser, _csrf: None = CSRFProtected):
    db.log_event(user.id, "logout", "Signed out")
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@router.post("/settings/password")
def change_password(
    request: Request,
    db: Database,
    user: CurrentUser,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm: str = Form(...),
    _csrf: None = CSRFProtected,
):
    if not verify_password(current_password, user.password_hash):
        return RedirectResponse("/settings?error=Current+password+is+incorrect", status_code=303)
    if new_password != confirm:
        return RedirectResponse("/settings?error=New+passwords+do+not+match", status_code=303)

    problems = password_problems(new_password, user.email)
    if problems:
        from urllib.parse import quote_plus

        return RedirectResponse(f"/settings?error={quote_plus(' '.join(problems))}",
                                status_code=303)

    db.update_user(user.id, password_hash=hash_password(new_password))
    db.bump_session_epoch(user.id)   # retires every other session
    db.log_event(user.id, "password_changed", "Password changed; other sessions signed out")

    response = RedirectResponse("/settings?ok=Password+updated", status_code=303)
    _set_session(response, db.get_user(user.id))
    return response
