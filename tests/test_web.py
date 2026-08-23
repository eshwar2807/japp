"""Dashboard tests: auth, tenant isolation, CSRF, rate limits, API, costs.

The security assertions here are the point of the file. A dashboard holding a
credential vault and a provider API key has to fail closed, so each control is
tested by trying to defeat it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from config import settings
from database.db_manager import DBManager
from database.models import ActionKind, ActionStatus, ApplicationStatus

GOOD_PASSWORD = "Correct-Horse-9x!"


def _ready_profile() -> dict:
    """The minimum profile that passes the completeness gate."""
    from web.profile_form import blank_profile

    profile = blank_profile()
    profile["contact"].update(full_name="Ada Lovelace", email="ada@example.com",
                              phone="+1-555-010-0100")
    profile["contact"]["location"]["city"] = "Austin"
    profile["summary"] = "Backend engineer."
    profile["skills"]["hard"] = ["Python"]
    profile["experience"] = [{"company": "Analytical Engines", "title": "Engineer",
                              "start_date": "2020-01", "is_current": True,
                              "bullets": ["Did a thing."]}]
    profile["education"] = [{"institution": "University of London"}]
    profile["legal"].update(work_authorization_us="Yes",
                            requires_sponsorship_now_or_future="No",
                            earliest_start_date="2026-09-01", desired_salary="185000")
    return profile
OTHER_PASSWORD = "Battery-Staple-7z!"


@pytest.fixture()
def web(tmp_path, monkeypatch):
    """A dashboard wired to a throwaway database and vault."""
    monkeypatch.setattr(settings, "DB_URL", f"sqlite:///{tmp_path/'web.db'}")
    monkeypatch.setattr(settings, "KEY_PATH", tmp_path / "vault.key")
    monkeypatch.setattr(settings, "SECRET_KEY", "test-secret-key-not-for-real-use")
    monkeypatch.setattr(settings, "COOKIE_SECURE", False)
    monkeypatch.setattr(settings, "ALLOW_SIGNUP", True)
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path / "out")

    # Same key resolution as the app, so both share one vault.
    db = DBManager(db_url=settings.DB_URL)

    import web.deps as deps

    deps.get_db.cache_clear()
    deps.get_sessions.cache_clear()
    deps.get_limiter.cache_clear()
    monkeypatch.setattr(deps, "get_db", lambda: db)

    from web.app import create_app

    app = create_app()
    app.dependency_overrides[deps.get_db] = lambda: db
    client = TestClient(app, follow_redirects=False)
    client.db = db
    return client


def signup(client, email: str, password: str = GOOD_PASSWORD):
    client.get("/login")  # obtain a CSRF cookie
    token = client.cookies.get("jp_csrf")
    return client.post(
        "/signup",
        data={"email": email, "password": password, "confirm": password, "csrf_token": token},
    )


def _form_token(client, path: str) -> str:
    """The CSRF token rendered into a form on a cold first visit."""
    import re

    return re.search(r'name="csrf_token" value="([^"]*)"', client.get(path).text).group(1)


def csrf(client) -> str:
    client.get("/")
    return client.cookies.get("jp_csrf") or ""


# ---------------- signup / login ----------------


def test_signup_creates_account_and_signs_in(web):
    response = signup(web, "ada@example.com")
    assert response.status_code == 303
    assert response.headers["location"].startswith("/profile")
    assert web.db.get_user_by_email("ada@example.com") is not None


def test_password_is_never_stored_in_plaintext(web):
    signup(web, "ada@example.com")
    user = web.db.get_user_by_email("ada@example.com")
    assert GOOD_PASSWORD not in user.password_hash
    assert user.password_hash.startswith("$argon2")


def test_weak_password_rejected(web):
    web.get("/login")
    token = web.cookies.get("jp_csrf")
    response = web.post("/signup", data={"email": "a@b.com", "password": "short",
                                         "confirm": "short", "csrf_token": token})
    assert response.status_code == 200
    assert "at least 12 characters" in response.text


def test_duplicate_signup_does_not_confirm_the_address_exists(web):
    signup(web, "ada@example.com")
    web.cookies.clear()
    response = signup(web, "ada@example.com")
    # The response must not distinguish "taken" from any other failure.
    assert "already exists" not in response.text.lower()
    assert "already registered" not in response.text.lower()
    assert "Could not create that account" in response.text


def test_wrong_password_gives_the_same_message_as_unknown_user(web):
    signup(web, "ada@example.com")
    web.cookies.clear()

    web.get("/login")
    token = web.cookies.get("jp_csrf")
    known = web.post("/login", data={"email": "ada@example.com", "password": "Wrong-Pass-123!",
                                     "csrf_token": token, "next": "/"})
    unknown = web.post("/login", data={"email": "nobody@example.com", "password": "Wrong-Pass-123!",
                                       "csrf_token": token, "next": "/"})
    assert known.status_code == unknown.status_code == 200
    assert "Email or password is incorrect." in known.text
    assert "Email or password is incorrect." in unknown.text


def test_login_open_redirect_is_blocked(web):
    signup(web, "ada@example.com")
    web.cookies.clear()
    web.get("/login")
    token = web.cookies.get("jp_csrf")
    response = web.post("/login", data={"email": "ada@example.com", "password": GOOD_PASSWORD,
                                        "csrf_token": token, "next": "https://evil.example.com"})
    assert response.headers["location"] == "/"


# ---------------- session security ----------------


def test_anonymous_pages_redirect_to_login(web):
    for path in ("/", "/profile", "/applications", "/actions", "/logs", "/costs", "/settings"):
        response = web.get(path)
        assert response.status_code == 303, path
        assert response.headers["location"].startswith("/login"), path


def test_password_change_invalidates_other_sessions(web):
    signup(web, "ada@example.com")
    assert web.get("/").status_code == 200

    web.post("/settings/password", data={
        "current_password": GOOD_PASSWORD, "new_password": OTHER_PASSWORD,
        "confirm": OTHER_PASSWORD, "csrf_token": csrf(web)})

    # A second client holding the pre-change cookie is now signed out.
    stale = TestClient(web.app, follow_redirects=False)
    stale.cookies.set("jp_session", web.cookies.get("jp_session") or "")
    user = web.db.get_user_by_email("ada@example.com")
    assert user.session_epoch == 2


def test_tampered_session_cookie_is_rejected(web):
    signup(web, "ada@example.com")
    web.cookies.set("jp_session", "forged.session.value")
    assert web.get("/").status_code == 303


# ---------------- CSRF ----------------


def test_post_without_csrf_token_is_rejected(web):
    signup(web, "ada@example.com")
    response = web.post("/applications/new", data={"job_url": "https://example.com/j/1"})
    assert response.status_code == 403


def test_post_with_wrong_csrf_token_is_rejected(web):
    signup(web, "ada@example.com")
    response = web.post("/applications/new",
                        data={"job_url": "https://example.com/j/1", "csrf_token": "nope"})
    assert response.status_code == 403


# ---------------- tenant isolation ----------------


@pytest.fixture()
def two_users(web):
    signup(web, "ada@example.com")
    ada = web.db.get_user_by_email("ada@example.com")
    app_a = web.db.create_application(
        company="Acme", role_title="Engineer", job_url="https://x.com/1", user_id=ada.id)

    other = TestClient(web.app, follow_redirects=False)
    signup(other, "eve@example.com")
    eve = web.db.get_user_by_email("eve@example.com")
    return web, other, ada, eve, app_a


def test_user_cannot_open_another_users_application(two_users):
    _, other, _, _, app_a = two_users
    response = other.get(f"/applications/{app_a.id}")
    assert response.status_code == 303
    assert "not+found" in response.headers["location"].lower().replace("%20", "+")


def test_user_cannot_download_another_users_resume(two_users):
    _, other, _, _, app_a = two_users
    response = other.get(f"/applications/{app_a.id}/resume")
    assert response.status_code == 303


def test_user_cannot_submit_feedback_on_another_users_application(two_users):
    _, other, _, _, app_a = two_users
    other.get("/")
    response = other.post(f"/applications/{app_a.id}/feedback",
                          data={"status": "Interview", "notes": "mine now",
                                "csrf_token": other.cookies.get("jp_csrf")})
    assert response.status_code in (303, 404)
    assert app_a.status is ApplicationStatus.DRAFT


def test_application_list_is_scoped_to_the_user(two_users):
    web, other, _, _, _ = two_users
    assert "Acme" in web.get("/applications").text
    assert "Acme" not in other.get("/applications").text


def test_user_cannot_answer_another_users_action(two_users):
    web, other, ada, _, app_a = two_users
    item = web.db.create_action(ada.id, ActionKind.UNMAPPED_FIELD, "Secret question?",
                                application_id=app_a.id)
    other.get("/")
    response = other.post(f"/actions/{item.id}/answer",
                          data={"answer": "hijacked", "csrf_token": other.cookies.get("jp_csrf")})
    assert response.status_code == 404
    refreshed = web.db.list_actions(user_id=ada.id, status=None)[0]
    assert refreshed.status is ActionStatus.OPEN


# ---------------- secrets handling ----------------


def test_anthropic_key_is_encrypted_and_never_echoed(web):
    signup(web, "ada@example.com")
    secret = "sk-ant-api03-SECRETVALUE1234567890"
    web.post("/settings/anthropic-key",
             data={"anthropic_key": secret, "csrf_token": csrf(web)})

    user = web.db.get_user_by_email("ada@example.com")
    assert user.encrypted_anthropic_key is not None
    assert secret.encode() not in user.encrypted_anthropic_key
    assert web.db.get_anthropic_key(user.id) == secret

    page = web.get("/settings").text
    assert secret not in page          # only a masked preview is rendered
    assert "sk-a" in page


def test_api_key_is_shown_once_and_stored_only_as_a_hash(web):
    signup(web, "ada@example.com")
    response = web.post("/settings/api-key", data={"csrf_token": csrf(web)})
    raw = response.headers["location"].split("new_key=")[1].split("&")[0]
    from urllib.parse import unquote_plus

    raw = unquote_plus(raw)

    user = web.db.get_user_by_email("ada@example.com")
    assert user.api_key_hash and raw not in user.api_key_hash
    assert len(user.api_key_hash) == 64
    # Reloading settings without the one-time parameter must not reveal it.
    assert raw not in web.get("/settings").text


# ---------------- JSON API ----------------


@pytest.fixture()
def api_client(web):
    signup(web, "ada@example.com")
    response = web.post("/settings/api-key", data={"csrf_token": csrf(web)})
    from urllib.parse import unquote_plus

    raw = unquote_plus(response.headers["location"].split("new_key=")[1].split("&")[0])
    client = TestClient(web.app)
    client.db = web.db
    client.raw_key = raw
    client.user = web.db.get_user_by_email("ada@example.com")
    return client


def test_api_requires_a_bearer_token(api_client):
    assert api_client.get("/api/v1/me").status_code == 401


def test_api_rejects_an_invalid_key(api_client):
    response = api_client.get("/api/v1/me",
                              headers={"Authorization": "Bearer jp_live_totallywrong"})
    assert response.status_code == 401


def test_api_accepts_a_valid_key(api_client):
    response = api_client.get("/api/v1/me",
                              headers={"Authorization": f"Bearer {api_client.raw_key}"})
    assert response.status_code == 200
    assert response.json()["email"] == "ada@example.com"


def test_api_key_scopes_data_to_its_owner(api_client):
    other = api_client.db.create_user("eve@example.com", "$argon2id$fake")
    api_client.db.create_application(company="EveCo", role_title="X",
                                     job_url="https://y.com/1", user_id=other.id)
    api_client.db.create_application(company="AdaCo", role_title="Y",
                                     job_url="https://y.com/2", user_id=api_client.user.id)

    body = api_client.get("/api/v1/applications",
                          headers={"Authorization": f"Bearer {api_client.raw_key}"}).json()
    companies = {a["company"] for a in body["applications"]}
    assert companies == {"AdaCo"}


def test_revoked_api_key_stops_working(api_client, web):
    web.post("/settings/api-key/revoke", data={"csrf_token": csrf(web)})
    response = api_client.get("/api/v1/me",
                              headers={"Authorization": f"Bearer {api_client.raw_key}"})
    assert response.status_code == 401


def test_api_costs_endpoint_rejects_unsupported_windows(api_client):
    auth = {"Authorization": f"Bearer {api_client.raw_key}"}
    assert api_client.get("/api/v1/costs?days=7", headers=auth).status_code == 400
    assert api_client.get("/api/v1/costs?days=90", headers=auth).status_code == 200


# ---------------- costs ----------------


def test_cost_dashboard_aggregates_recorded_usage(web):
    signup(web, "ada@example.com")
    user = web.db.get_user_by_email("ada@example.com")
    for _ in range(3):
        web.db.record_usage(user.id, "claude-opus-5", "tailor",
                            input_tokens=10_000, output_tokens=2_000, cost_usd=0.10)

    page = web.get("/costs?days=30").text
    assert "$0.30" in page
    assert "claude-opus-5" in page

    for window in (30, 90, 120, 360):
        assert web.get(f"/costs?days={window}").status_code == 200


def test_cost_windows_outside_the_allowed_set_fall_back(web):
    signup(web, "ada@example.com")
    assert web.get("/costs?days=7").status_code == 200   # silently uses 30


# ---------------- security headers ----------------


def test_security_headers_present(web):
    response = web.get("/login")
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert "'unsafe-inline'" not in response.headers["Content-Security-Policy"]


def test_openapi_schema_is_not_exposed(web):
    for path in ("/openapi.json", "/docs", "/redoc"):
        assert web.get(path).status_code == 404


# ---------------- CSRF token minting ----------------


def test_first_visit_form_carries_a_usable_csrf_token(web):
    """Regression: the token was minted on the response, so a first-time
    visitor's form rendered an empty value and every signup 403'd."""
    import re

    response = web.get("/signup")
    rendered = re.search(r'name="csrf_token" value="([^"]*)"', response.text)
    assert rendered and rendered.group(1), "signup form rendered without a CSRF token"

    cookie = response.cookies.get("jp_csrf")
    assert cookie == rendered.group(1), "rendered token must match the cookie"


def test_signup_succeeds_on_a_cold_first_visit(web):
    """No warm-up request: exactly what a real browser does."""
    import re

    fresh = TestClient(web.app, follow_redirects=False)
    page = fresh.get("/signup")
    token = re.search(r'name="csrf_token" value="([^"]*)"', page.text).group(1)

    response = fresh.post("/signup", data={
        "email": "cold@example.com", "password": GOOD_PASSWORD,
        "confirm": GOOD_PASSWORD, "csrf_token": token})
    assert response.status_code == 303
    assert web.db.get_user_by_email("cold@example.com") is not None


def test_login_form_also_carries_a_token_on_first_visit(web):
    import re

    page = TestClient(web.app, follow_redirects=False).get("/login")
    token = re.search(r'name="csrf_token" value="([^"]*)"', page.text)
    assert token and token.group(1)

# ---------------- CSP compliance ----------------


@pytest.mark.parametrize("path", ["/", "/profile", "/applications", "/actions",
                                  "/logs", "/costs", "/settings"])
def test_pages_contain_no_inline_styles(web, path):
    """The CSP sets style-src 'self', so the browser discards inline style
    attributes entirely. Any that slip in are silently dead styling — which is
    how the cost chart once rendered as a flat line."""
    import re

    signup(web, "ada@example.com")
    html = web.get(path).text
    assert re.search(r'style="[^"]*"', html) is None, f"{path} emits inline styles"


def test_login_and_signup_contain_no_inline_styles(web):
    import re

    for path in ("/login", "/signup"):
        assert re.search(r'style="[^"]*"', web.get(path).text) is None, path


def test_chart_bar_classes_exist_in_the_stylesheet(web):
    """Every .hNN class the template can emit must be defined, or bars vanish."""
    from pathlib import Path

    css = (Path(__file__).resolve().parent.parent / "web/static/app.css").read_text()
    for pct in (0, 1, 37, 99, 100):
        assert f".chart .col i.h{pct} {{" in css
        assert f".bar > i.w{pct} {{" in css


def test_cost_chart_scales_bars_to_the_peak(web):
    """Regression: bars collapsed to nothing, drawing a flat line regardless
    of actual spend."""
    import re

    signup(web, "ada@example.com")
    user = web.db.get_user_by_email("ada@example.com")
    web.db.record_usage(user.id, "claude-opus-5", "tailor", cost_usd=1.00)

    classes = re.findall(r'<i class="h(\d+)"></i>', web.get("/costs?days=30").text)
    assert classes, "chart emitted no bars"
    assert max(int(c) for c in classes) == 100, "peak day must fill the chart"


# ---------------- batch queue routes ----------------


def test_batch_paste_enqueues_one_job_per_url(web):
    signup(web, "ada@example.com")
    user = web.db.get_user_by_email("ada@example.com")
    web.db.save_profile(user.id, _ready_profile())
    web.db.set_anthropic_key(user.id, "sk-ant-test")

    response = web.post("/applications/new", data={
        "job_url": "https://boards.greenhouse.io/a/jobs/1\n"
                   "https://jobs.lever.co/b/xyz\n"
                   "   \n"
                   "not-a-url",
        "csrf_token": csrf(web)})
    assert response.status_code == 303
    assert response.headers["location"].startswith("/queue")

    jobs = web.db.list_jobs(user_id=user.id)
    assert len(jobs) == 2, "only the two valid URLs should be queued"
    assert len({j.batch_id for j in jobs}) == 1, "a paste is one batch"
    assert all(j.status.value == "Queued" for j in jobs)


def test_single_url_keeps_the_pasted_job_description(web):
    signup(web, "ada@example.com")
    user = web.db.get_user_by_email("ada@example.com")
    web.db.save_profile(user.id, _ready_profile())
    web.db.set_anthropic_key(user.id, "sk-ant-test")

    web.post("/applications/new", data={
        "job_url": "https://boards.greenhouse.io/a/jobs/1",
        "job_description": "Python, Kubernetes, REST APIs.",
        "csrf_token": csrf(web)})

    job = web.db.list_jobs(user_id=user.id)[0]
    assert job.job_description == "Python, Kubernetes, REST APIs."
    assert job.batch_id is None


def test_batch_size_is_capped(web):
    signup(web, "ada@example.com")
    user = web.db.get_user_by_email("ada@example.com")
    web.db.save_profile(user.id, _ready_profile())
    web.db.set_anthropic_key(user.id, "sk-ant-test")

    urls = "\n".join(f"https://x.com/j/{i}" for i in range(40))
    response = web.post("/applications/new", data={"job_url": urls, "csrf_token": csrf(web)})
    assert "at+most" in response.headers["location"].replace("%20", "+")
    assert web.db.list_jobs(user_id=user.id) == []


def test_queueing_requires_a_complete_profile(web):
    signup(web, "ada@example.com")
    response = web.post("/applications/new", data={
        "job_url": "https://x.com/j/1", "csrf_token": csrf(web)})
    assert response.headers["location"].startswith("/profile")


def test_queue_page_lists_jobs_and_blocks(web):
    from database.models import BlockMode

    signup(web, "ada@example.com")
    user = web.db.get_user_by_email("ada@example.com")
    job = web.db.enqueue_job(user.id, job_url="https://boards.greenhouse.io/a/jobs/1")
    web.db.block_job(job.id, BlockMode.NEEDS_BROWSER, "verification challenge",
                     holds_browser=True)

    page = web.get("/queue").text
    assert "Needs you" in page and "holding browser" in page


def test_queue_is_scoped_to_its_owner(web):
    signup(web, "ada@example.com")
    ada = web.db.get_user_by_email("ada@example.com")
    web.db.enqueue_job(ada.id, job_url="https://secret.example.com/job/1")

    other = TestClient(web.app, follow_redirects=False)
    signup(other, "eve@example.com")
    assert "secret.example.com" not in other.get("/queue").text


def test_cancelling_another_users_job_is_refused(web):
    signup(web, "ada@example.com")
    ada = web.db.get_user_by_email("ada@example.com")
    job = web.db.enqueue_job(ada.id, job_url="https://x.com/1")

    other = TestClient(web.app, follow_redirects=False)
    signup(other, "eve@example.com")
    other.get("/")
    response = other.post(f"/queue/{job.id}/cancel",
                          data={"csrf_token": other.cookies.get("jp_csrf")})
    assert response.headers["location"].startswith("/queue")
    assert web.db.get_job(job.id).status.value == "Queued"


# ---------------- notification settings ----------------


def test_notification_preferences_round_trip(web):
    signup(web, "ada@example.com")
    web.post("/settings/notifications", data={
        "notify_desktop": "on",
        "notify_webhook_url": "https://ntfy.sh/my-topic",
        "notify_quiet_seconds": "300",
        "csrf_token": csrf(web)})

    user = web.db.get_user_by_email("ada@example.com")
    assert user.notify_desktop is True
    assert user.notify_webhook_url == "https://ntfy.sh/my-topic"
    assert user.notify_quiet_seconds == 300


def test_webhook_url_must_be_http(web):
    signup(web, "ada@example.com")
    response = web.post("/settings/notifications", data={
        "notify_webhook_url": "file:///etc/passwd", "csrf_token": csrf(web)})
    assert "error=" in response.headers["location"]
    assert web.db.get_user_by_email("ada@example.com").notify_webhook_url is None


def test_quiet_window_is_clamped(web):
    signup(web, "ada@example.com")
    web.post("/settings/notifications", data={
        "notify_quiet_seconds": "999999", "csrf_token": csrf(web)})
    assert web.db.get_user_by_email("ada@example.com").notify_quiet_seconds == 3600


# ---------------- queue API ----------------


def test_api_exposes_the_queue(api_client):
    auth = {"Authorization": f"Bearer {api_client.raw_key}"}
    api_client.db.enqueue_job(api_client.user.id, job_url="https://x.com/1")

    body = api_client.get("/api/v1/queue", headers=auth).json()
    assert body["summary"]["Queued"] == 1
    assert body["jobs"][0]["job_url"] == "https://x.com/1"


def test_api_queue_rejects_an_unknown_status(api_client):
    auth = {"Authorization": f"Bearer {api_client.raw_key}"}
    assert api_client.get("/api/v1/queue?status=Nonsense", headers=auth).status_code == 400


def test_api_cannot_read_another_users_job(api_client):
    other = api_client.db.create_user("eve@example.com", "$argon2id$fake")
    job = api_client.db.enqueue_job(other.id, job_url="https://x.com/1")
    auth = {"Authorization": f"Bearer {api_client.raw_key}"}
    assert api_client.get(f"/api/v1/queue/{job.id}", headers=auth).status_code == 404


# ---------------- invite code ----------------


def test_signup_requires_the_invite_code_when_one_is_set(web, monkeypatch):
    monkeypatch.setattr(settings, "INVITE_CODE", "letmein-1234")
    fresh = TestClient(web.app, follow_redirects=False)
    token = _form_token(fresh, "/signup")

    response = fresh.post("/signup", data={
        "email": "eve@example.com", "password": GOOD_PASSWORD,
        "confirm": GOOD_PASSWORD, "invite_code": "wrong", "csrf_token": token})

    assert response.status_code == 200
    assert "invite code is not valid" in response.text
    assert web.db.get_user_by_email("eve@example.com") is None


def test_signup_succeeds_with_the_right_invite_code(web, monkeypatch):
    monkeypatch.setattr(settings, "INVITE_CODE", "letmein-1234")
    fresh = TestClient(web.app, follow_redirects=False)
    token = _form_token(fresh, "/signup")

    response = fresh.post("/signup", data={
        "email": "eve@example.com", "password": GOOD_PASSWORD,
        "confirm": GOOD_PASSWORD, "invite_code": "letmein-1234", "csrf_token": token})

    assert response.status_code == 303
    assert web.db.get_user_by_email("eve@example.com") is not None


def test_a_bad_invite_code_does_not_reveal_whether_the_email_exists(web, monkeypatch):
    """The code is checked first, so the response is identical either way."""
    monkeypatch.setattr(settings, "INVITE_CODE", "letmein-1234")
    signup_ok = TestClient(web.app, follow_redirects=False)
    token = _form_token(signup_ok, "/signup")
    signup_ok.post("/signup", data={"email": "taken@example.com", "password": GOOD_PASSWORD,
                                    "confirm": GOOD_PASSWORD, "invite_code": "letmein-1234",
                                    "csrf_token": token})

    fresh = TestClient(web.app, follow_redirects=False)
    token = _form_token(fresh, "/signup")
    existing = fresh.post("/signup", data={"email": "taken@example.com", "password": GOOD_PASSWORD,
                                           "confirm": GOOD_PASSWORD, "invite_code": "no",
                                           "csrf_token": token})
    fresh2 = TestClient(web.app, follow_redirects=False)
    token2 = _form_token(fresh2, "/signup")
    unknown = fresh2.post("/signup", data={"email": "new@example.com", "password": GOOD_PASSWORD,
                                           "confirm": GOOD_PASSWORD, "invite_code": "no",
                                           "csrf_token": token2})
    assert existing.status_code == unknown.status_code
    assert "invite code is not valid" in existing.text
    assert "invite code is not valid" in unknown.text


def test_signup_works_when_no_invite_code_is_configured(web, monkeypatch):
    monkeypatch.setattr(settings, "INVITE_CODE", None)
    assert signup(web, "ada@example.com").status_code == 303


# ---------------- admin ----------------


def test_the_first_account_becomes_admin(web):
    signup(web, "first@example.com")
    assert web.db.get_user_by_email("first@example.com").is_admin is True


def test_later_accounts_are_not_admin(web):
    signup(web, "first@example.com")
    second = TestClient(web.app, follow_redirects=False)
    signup(second, "second@example.com")
    assert web.db.get_user_by_email("second@example.com").is_admin is False


def test_nominated_admin_email_is_promoted(web, monkeypatch):
    signup(web, "first@example.com")
    monkeypatch.setattr(settings, "ADMIN_EMAIL", "boss@example.com")
    other = TestClient(web.app, follow_redirects=False)
    signup(other, "boss@example.com")
    assert web.db.get_user_by_email("boss@example.com").is_admin is True


def test_admin_page_is_invisible_to_normal_users(web):
    signup(web, "first@example.com")            # admin
    second = TestClient(web.app, follow_redirects=False)
    signup(second, "second@example.com")

    assert web.get("/admin").status_code == 200
    # 404, not 403 — a 403 would confirm the page exists.
    assert second.get("/admin").status_code == 404
    assert "/admin" not in second.get("/").text


def test_admin_view_never_exposes_other_users_secrets(web):
    signup(web, "first@example.com")
    second = TestClient(web.app, follow_redirects=False)
    signup(second, "second@example.com")

    victim = web.db.get_user_by_email("second@example.com")
    web.db.set_anthropic_key(victim.id, "sk-ant-api03-VICTIMSECRET0000")
    web.db.upsert_credential("acme.com", "second@example.com", "PortalPassw0rd!",
                             user_id=victim.id)
    web.db.save_profile(victim.id, _ready_profile())

    page = web.get("/admin").text
    assert "second@example.com" in page              # the account is listed
    assert "sk-ant-api03-VICTIMSECRET0000" not in page
    assert "PortalPassw0rd!" not in page
    assert "Ada Lovelace" not in page                # nor their profile contents


def test_admin_can_suspend_and_restore_an_account(web):
    signup(web, "first@example.com")
    second = TestClient(web.app, follow_redirects=False)
    signup(second, "second@example.com")
    target = web.db.get_user_by_email("second@example.com")

    assert second.get("/").status_code == 200
    web.post(f"/admin/users/{target.id}/active",
             data={"active": "false", "csrf_token": csrf(web)})

    assert web.db.get_user_by_email("second@example.com").is_active is False
    # Suspension retires existing sessions immediately.
    assert second.get("/").status_code == 303

    web.post(f"/admin/users/{target.id}/active",
             data={"active": "true", "csrf_token": csrf(web)})
    assert web.db.get_user_by_email("second@example.com").is_active is True


def test_admin_cannot_suspend_themselves(web):
    signup(web, "first@example.com")
    me = web.db.get_user_by_email("first@example.com")
    response = web.post(f"/admin/users/{me.id}/active",
                        data={"active": "false", "csrf_token": csrf(web)})
    assert "error=" in response.headers["location"]
    assert web.db.get_user_by_email("first@example.com").is_active is True


def test_normal_user_cannot_suspend_anyone(web):
    signup(web, "first@example.com")
    admin = web.db.get_user_by_email("first@example.com")
    second = TestClient(web.app, follow_redirects=False)
    signup(second, "second@example.com")
    second.get("/")

    response = second.post(f"/admin/users/{admin.id}/active",
                           data={"active": "false", "csrf_token": second.cookies.get("jp_csrf")})
    assert response.status_code == 404
    assert web.db.get_user_by_email("first@example.com").is_active is True


def test_healthz_is_public_and_leaks_nothing(web):
    response = web.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    # No account, version or configuration detail in an unauthenticated probe.
    assert "email" not in response.text and "version" not in response.text
