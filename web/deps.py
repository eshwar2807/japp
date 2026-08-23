"""Shared FastAPI dependencies: database, authentication, CSRF, rate limits."""

from __future__ import annotations

import functools
from typing import Annotated

from fastapi import Depends, Form, HTTPException, Request, status
from fastapi.responses import RedirectResponse

from config import settings
from database.db_manager import DBManager
from database.models import User
from web.security import (
    CSRF_COOKIE,
    RATE_LIMITS,
    SESSION_COOKIE,
    RateLimiter,
    SessionManager,
    api_keys_match,
    csrf_valid,
    hash_api_key,
)


class RedirectToLogin(Exception):
    """Raised for unauthenticated HTML requests; converted to a 302 upstream."""

    def __init__(self, next_url: str = "/") -> None:
        self.next_url = next_url


@functools.lru_cache(maxsize=1)
def get_db() -> DBManager:
    return DBManager()


@functools.lru_cache(maxsize=1)
def get_sessions() -> SessionManager:
    return SessionManager(settings.load_or_create_secret_key())


@functools.lru_cache(maxsize=1)
def get_limiter() -> RateLimiter:
    return RateLimiter()


def client_ip(request: Request) -> str:
    """Best-effort client address.

    X-Forwarded-For is only trusted when a proxy is declared, because a
    spoofable header would otherwise let an attacker evade rate limiting.
    """
    if settings.PROXY_SERVER:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request, bucket: str, identity: str) -> None:
    limit, window = RATE_LIMITS[bucket]
    allowed, retry = get_limiter().check(f"{bucket}:{identity}", limit, window)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many requests. Try again in {retry}s.",
            headers={"Retry-After": str(retry)},
        )


# --------------------------------------------------------------------------
# Session authentication (HTML)
# --------------------------------------------------------------------------


def optional_user(request: Request) -> User | None:
    """Resolve the signed-in user, or None. Never raises."""
    data = get_sessions().read(request.cookies.get(SESSION_COOKIE, ""))
    if not data:
        return None
    user = get_db().get_user(int(data.get("uid", 0)))
    if user is None or not user.is_active:
        return None
    # A password change bumps the epoch, retiring every older cookie.
    if int(data.get("epoch", 0)) != int(user.session_epoch or 1):
        return None
    return user


def require_user(request: Request) -> User:
    user = optional_user(request)
    if user is None:
        raise RedirectToLogin(request.url.path)
    return user


CurrentUser = Annotated[User, Depends(require_user)]
OptionalUser = Annotated["User | None", Depends(optional_user)]
Database = Annotated[DBManager, Depends(get_db)]


# --------------------------------------------------------------------------
# CSRF
# --------------------------------------------------------------------------


def verify_csrf(request: Request, csrf_token: str = Form(default="")) -> None:
    """Double-submit cookie check on every state-changing form post."""
    if not csrf_valid(request.cookies.get(CSRF_COOKIE), csrf_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing CSRF token. Reload the page and try again.",
        )


CSRFProtected = Depends(verify_csrf)


# --------------------------------------------------------------------------
# API-key authentication (JSON)
# --------------------------------------------------------------------------


def api_user(request: Request) -> User:
    """Authenticate a programmatic caller via `Authorization: Bearer <key>`."""
    header = request.headers.get("authorization", "")
    scheme, _, raw = header.partition(" ")
    ip = client_ip(request)

    if scheme.lower() != "bearer" or not raw:
        enforce_rate_limit(request, "api_auth_fail", ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Look up by hash; a constant-time compare then confirms the match, so a
    # near-miss key cannot be distinguished by response timing.
    user = get_db().get_user_by_api_key_hash(hash_api_key(raw.strip()))
    if user is None or not api_keys_match(raw.strip(), user.api_key_hash or ""):
        enforce_rate_limit(request, "api_auth_fail", ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    enforce_rate_limit(request, "api", str(user.id))
    return user


APIUser = Annotated[User, Depends(api_user)]
