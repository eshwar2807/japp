"""FastAPI application factory for the dashboard."""

from __future__ import annotations

import logging
import secrets
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import settings
from web.deps import RedirectToLogin, get_db
from web.security import CSRF_COOKIE, SECURITY_HEADERS, generate_csrf_token

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))


def _fmt_money(value: float | None) -> str:
    value = value or 0.0
    if value and value < 0.01:
        return f"${value:.4f}"
    return f"${value:,.2f}"


def _fmt_tokens(value: int | None) -> str:
    value = int(value or 0)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


templates.env.filters["money"] = _fmt_money
templates.env.filters["tokens"] = _fmt_tokens


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Housekeeping on boot: drop log rows past the retention window."""
    removed = get_db().prune_logs(settings.LOG_RETENTION_DAYS)
    if removed:
        log.info("Pruned %d log rows older than %d days.",
                 removed, settings.LOG_RETENTION_DAYS)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Job Application Pipeline",
        lifespan=lifespan,
        docs_url=None,      # no public schema browser on a credential-holding app
        redoc_url=None,
        openapi_url=None,
    )

    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

    # ---------------- middleware ----------------

    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        # Resolve the CSRF token up front so the value rendered into the form
        # is the same one the cookie will carry. Minting it on the response
        # instead leaves a first-time visitor with an empty token in the form.
        existing = request.cookies.get(CSRF_COOKIE)
        token = existing or generate_csrf_token()
        request.state.csrf_token = token

        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        if settings.COOKIE_SECURE:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )

        if existing is None:
            response.set_cookie(
                CSRF_COOKIE,
                token,
                httponly=False,   # a double-submit token, not a secret
                secure=settings.COOKIE_SECURE,
                samesite="lax",
                max_age=60 * 60 * 12,
            )
        return response

    # ---------------- error handling ----------------

    @app.exception_handler(RedirectToLogin)
    async def redirect_to_login(request: Request, exc: RedirectToLogin):
        return RedirectResponse(f"/login?next={exc.next_url}", status_code=303)

    @app.exception_handler(PermissionError)
    async def permission_denied(request: Request, exc: PermissionError):
        # Cross-tenant access attempts are logged and answered as 404, so the
        # response does not confirm that the resource exists.
        log.warning("Permission denied on %s: %s", request.url.path, exc)
        return JSONResponse({"detail": "Not found."}, status_code=404)

    # ---------------- routes ----------------

    from web.routes import api, auth, dashboard

    app.include_router(auth.router)
    app.include_router(dashboard.router)
    app.include_router(api.router)

    return app


app = create_app()
