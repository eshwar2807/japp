"""One switch that decides whether the pipeline spends money.

The search reads job boards for free, but screening what passes them costs an
LLM call per posting, and it ran every day whether or not anyone was applying
that day. The control for that existed - `topup_enabled` - but it was buried in
a scheduling form on the settings page and said nothing about cost, so it read
as a scheduling preference rather than the spending decision it is.

Off is a legitimate resting state. The tests below care about two things: that
off really does stop the spending, and that off is never mistaken for broken.
"""

from __future__ import annotations

import pytest

from config import settings
from tests.conftest import signup


@pytest.fixture()
def account(web):
    signup(web, "switch@example.com")
    return web.db.get_user_by_email("switch@example.com")


# ---------------- what the switch actually gates ----------------


def test_the_search_is_off_for_a_new_account(db):
    """Nobody should start paying for a daily search they did not ask for."""
    user = db.create_user("fresh@example.com", "$argon2id$fake")
    assert user.topup_enabled is False


def test_nothing_is_searched_while_the_switch_is_off(db, monkeypatch):
    from datetime import datetime, timedelta, timezone as tz

    monkeypatch.setattr(settings, "DAILY_APPLICATION_CAP", 20)
    user = db.create_user("off@example.com", "$argon2id$fake")
    db.update_user(user.id, topup_enabled=False, timezone="UTC",
                   notify_utc_offset_minutes=0, topup_hour=0)
    db.update_user(user.id, last_topup_at=datetime.now(tz.utc).replace(tzinfo=None)
                   - timedelta(hours=3))

    assert db.users_due_for_topup() == []


def test_the_same_account_is_searched_once_the_switch_is_on(db, monkeypatch):
    """The only difference between this test and the one above is the switch."""
    from datetime import datetime, timedelta, timezone as tz

    monkeypatch.setattr(settings, "DAILY_APPLICATION_CAP", 20)
    user = db.create_user("on@example.com", "$argon2id$fake")
    db.update_user(user.id, topup_enabled=True, timezone="UTC",
                   notify_utc_offset_minutes=0, topup_hour=0)
    db.update_user(user.id, last_topup_at=datetime.now(tz.utc).replace(tzinfo=None)
                   - timedelta(hours=3))

    assert db.users_due_for_topup() == [user.id]


def test_the_switch_also_stops_the_shortfall_retries(db, monkeypatch):
    """The retry loop and the automatic widening both run through the same
    check, so turning the search off has to stop those too - they are where the
    repeated spending actually came from."""
    from datetime import datetime, timedelta, timezone as tz

    monkeypatch.setattr(settings, "DAILY_APPLICATION_CAP", 20)
    user = db.create_user("retry@example.com", "$argon2id$fake")
    db.update_user(user.id, topup_enabled=True, timezone="UTC",
                   notify_utc_offset_minutes=0, topup_hour=0)
    db.update_user(user.id, last_topup_at=datetime.now(tz.utc).replace(tzinfo=None)
                   - timedelta(hours=3))
    assert db.users_due_for_topup() == [user.id]      # would retry

    db.update_user(user.id, topup_enabled=False)
    assert db.users_due_for_topup() == []             # and now does not


# ---------------- the control itself ----------------


def test_the_switch_can_be_turned_on_from_any_page(web, account):
    web.post("/search/toggle",
             data={"on": "1", "back": "/applications",
                   "csrf_token": web.cookies.get("jp_csrf")})

    assert web.db.get_user(account.id).topup_enabled is True


def test_the_switch_can_be_turned_off_again(web, account):
    web.db.update_user(account.id, topup_enabled=True)

    web.post("/search/toggle",
             data={"on": "0", "back": "/", "csrf_token": web.cookies.get("jp_csrf")})

    assert web.db.get_user(account.id).topup_enabled is False


def test_toggling_is_recorded_with_what_it_had_spent(web, account):
    """So a later "why did this cost that" has a timeline to read."""
    web.post("/search/toggle",
             data={"on": "1", "back": "/", "csrf_token": web.cookies.get("jp_csrf")})

    events = [e for e in web.db.list_logs(user_id=account.id, limit=20)
              if e.event == "search_toggled"]
    assert events and "on" in events[0].message


def test_the_toggle_requires_a_csrf_token(web, account):
    response = web.post("/search/toggle", data={"on": "1", "back": "/"})
    assert response.status_code == 403
    assert web.db.get_user(account.id).topup_enabled is False


def test_the_toggle_will_not_bounce_you_off_site(web, account):
    """`back` comes from the form, so it is attacker-controllable."""
    response = web.post(
        "/search/toggle",
        data={"on": "1", "back": "https://evil.example.com/x",
              "csrf_token": web.cookies.get("jp_csrf")})

    assert response.headers["location"] == "/"


# ---------------- off must not look broken ----------------


def test_every_page_says_whether_the_search_is_on(web, account):
    """0/20 with no explanation is what makes someone think it is broken."""
    web.db.update_user(account.id, topup_enabled=False)

    for path in ("/", "/applications", "/discover"):
        page = web.get(path).text
        assert "Job search is off" in page, path
        assert "costing nothing" in page, path


def test_the_banner_shows_the_spend_when_the_search_is_on(web, account):
    web.db.update_user(account.id, topup_enabled=True)

    page = web.get("/").text
    assert "Job search is on" in page
    assert "today" in page


def test_the_banner_uses_no_inline_styles(web, account):
    """The CSP has no unsafe-inline, so a style attribute is silently dropped
    and the control renders unstyled."""
    assert 'style="' not in web.get("/").text
