"""Discovery, board clients, spend caps and the daily digest.

The property that matters most here: the model supplies companies, the
employer's own API supplies postings. A hallucinated job cannot survive that,
because it simply will not appear in the board response.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from config import settings
from engine.boards import (
    BoardError,
    Posting,
    dedupe,
    detect_board,
    location_allowed,
    matches,
)
from engine.discovery import (
    CompanySearchResult,
    CompanySuggestion,
    DiscoveryCriteria,
    DiscoveryEngine,
)


@pytest.fixture()
def user(db):
    user = db.create_user("ada@example.com", "$argon2id$fake")
    # Discovery ranks postings against the profile, so a run needs one.
    db.save_profile(user.id, {
        "contact": {"full_name": "Ada", "email": "a@b.com",
                    "location": {"city": "Austin"}},
        "summary": "Backend engineer.",
        "skills": {"hard": ["Microservices Architecture"],
                   "tooling": ["Java", "Spring Boot", "Kubernetes"]},
        "experience": [{"company": "Acme", "title": "Engineer",
                        "start_date": "2020-01", "is_current": True,
                        "bullets": ["Did a thing."]}],
        "education": [{"institution": "UoL"}],
        "legal": {"work_authorization_us": "Yes"},
    })
    return db.get_user(user.id)


def posting(**kw):
    base = dict(company="Acme", title="Senior Backend Engineer",
                url="https://boards.greenhouse.io/acme/jobs/1",
                location="Austin, TX", board="greenhouse", external_id="1")
    return Posting(**{**base, **kw})


# ---------------- the URL the browser is actually sent to ----------------
#
# A board returns whatever link the employer configured, and a large employer
# configures their own careers site. Stripe's Greenhouse board returns
# stripe.com/jobs/search?gh_jid=7217048: a marketing page listing every open
# role, with no form anywhere on it. Opening that produced "no resume input",
# "1 field discovered" and "submit button not found" - none of them
# form-filling faults, all of them the wrong page.


def test_an_employer_careers_link_is_replaced_by_the_ats_form():
    p = posting(company="stripe", slug="stripe", external_id="7217048",
                url="https://stripe.com/jobs/search?gh_jid=7217048")
    assert p.apply_url == ("https://job-boards.greenhouse.io/embed/job_app"
                           "?for=stripe&token=7217048")


def test_the_employer_link_is_still_kept_for_a_human_to_read():
    """The marketing page is the better thing to show a person; it is only
    unusable as an automation target."""
    p = posting(url="https://stripe.com/jobs/search?gh_jid=7217048")
    assert p.url == "https://stripe.com/jobs/search?gh_jid=7217048"


@pytest.mark.parametrize("board,slug,job_id,expected", [
    ("greenhouse", "flexport", "7975365",
     "https://job-boards.greenhouse.io/embed/job_app?for=flexport&token=7975365"),
    ("lever", "acme", "abc-123", "https://jobs.lever.co/acme/abc-123/apply"),
    ("ashby", "acme", "uuid-1", "https://jobs.ashbyhq.com/acme/uuid-1/application"),
    ("smartrecruiters", "acme", "744000", "https://jobs.smartrecruiters.com/acme/744000"),
    ("workable", "acme", "A1B2C3", "https://apply.workable.com/acme/j/A1B2C3/apply/"),
])
def test_every_board_resolves_to_its_own_apply_host(board, slug, job_id, expected):
    assert posting(board=board, slug=slug, external_id=job_id,
                   url="https://careers.acme.com/roles/1").apply_url == expected


def test_the_display_name_is_not_used_as_the_board_slug():
    """Ashby and Workable report a company display name, which is not a valid
    path segment. Using it would build a 404."""
    p = posting(company="Acme Corporation, Inc.", slug="acme",
                board="ashby", external_id="u1")
    assert p.apply_url == "https://jobs.ashbyhq.com/acme/u1/application"


@pytest.mark.parametrize("kw", [
    {"board": ""},                    # found by web search, not a board
    {"external_id": ""},              # nothing to address
    {"board": "workday"},             # no public pattern to build
    {"slug": ""},                     # never guess one from the company name
])
def test_an_unbuildable_url_falls_back_rather_than_guessing(kw):
    p = posting(url="https://careers.acme.com/roles/1", **kw)
    assert p.apply_url == "https://careers.acme.com/roles/1"


@pytest.mark.parametrize("stored", [
    "https://stripe.com/jobs/search?gh_jid=7217048",
    "https://stripe.com/careers/search?gh_jid=7217048",
    "https://boards.eu.example.com/careers?foo=bar&gh_jid=7217048&utm_source=x",
])
def test_a_stored_careers_link_is_repaired_from_its_job_id(stored):
    """Applications queued before apply_url existed still hold the employer's
    page. A Greenhouse job id is globally unique, so the form is recoverable
    without knowing which board it came from."""
    from engine.boards import canonicalize_apply_url

    assert canonicalize_apply_url(stored) == (
        "https://boards.greenhouse.io/embed/job_app?token=7217048")


@pytest.mark.parametrize("stored", [
    "https://job-boards.greenhouse.io/flexport/jobs/7975365",
    "https://jobs.lever.co/acme/abc-123/apply",
    "https://acme.wd1.myworkdayjobs.com/en-US/careers/job/x",
    "",
])
def test_a_url_with_nothing_to_repair_is_left_alone(stored):
    from engine.boards import canonicalize_apply_url

    assert canonicalize_apply_url(stored) == stored


# ---------------- board detection ----------------


@pytest.mark.parametrize("url,expected", [
    ("https://boards.greenhouse.io/stripe/jobs/123", ("greenhouse", "stripe")),
    ("https://job-boards.greenhouse.io/acme", ("greenhouse", "acme")),
    ("https://jobs.lever.co/spotify/abc-123", ("lever", "spotify")),
    ("https://jobs.eu.lever.co/acme/x", ("lever", "acme")),
    ("https://jobs.ashbyhq.com/linear/xyz", ("ashby", "linear")),
    ("https://acme.wd1.myworkdayjobs.com/careers", None),
    ("https://acme.com/careers", None),
    ("", None),
])
def test_detect_board(url, expected):
    assert detect_board(url) == expected


# ---------------- filtering ----------------


def test_title_filter_keeps_only_matching_roles():
    assert matches(posting(title="Senior Backend Engineer"), titles=["backend"])
    assert not matches(posting(title="Account Executive"), titles=["backend"])


def test_excluded_terms_win_over_title_match():
    """An internship matching the title must still be dropped."""
    assert not matches(posting(title="Backend Engineer Intern"),
                       titles=["backend"], exclude=["intern"])


def test_remote_locations_are_accepted_for_any_city():
    assert matches(posting(location="Remote - US"), locations=["austin"])


def test_no_filters_keeps_everything():
    assert matches(posting(title="Anything At All"))


def test_dedupe_drops_repeats_and_previously_seen():
    a, b = posting(external_id="1"), posting(external_id="2")
    assert len(dedupe([a, b, posting(external_id="1")])) == 2
    assert dedupe([a, b], seen_keys={a.key}) == [b]


def test_dedupe_drops_postings_with_no_url():
    assert dedupe([posting(url="")]) == []


# ---------------- discovery ----------------


class StubMessages:
    """Returns a fixed company list without touching the network."""

    def __init__(self, result, stop_reason="end_turn"):
        self.result = result
        self.stop_reason = stop_reason
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed_output=self.result, stop_reason=self.stop_reason,
                               content=[])


def test_discovery_asks_for_companies_with_web_search():
    stub = StubMessages(CompanySearchResult(companies=[
        CompanySuggestion(name="Stripe", careers_url="https://boards.greenhouse.io/stripe")]))
    engine = DiscoveryEngine(client=stub)

    engine.find_companies(DiscoveryCriteria(titles=["Backend Engineer"]))

    tools = stub.calls[0]["tools"]
    assert tools[0]["type"].startswith("web_search_")
    assert stub.calls[0]["output_format"] is CompanySearchResult


def test_discovery_passes_exclusions_into_the_prompt():
    stub = StubMessages(CompanySearchResult(companies=[]))
    DiscoveryEngine(client=stub).find_companies(
        DiscoveryCriteria(titles=["Backend"]), already_applied=["Acme", "Globex"])

    prompt = stub.calls[0]["messages"][0]["content"]
    assert "Acme" in prompt and "Globex" in prompt
    assert "do not return these" in prompt.lower()


def test_discovery_uses_an_opus_tier_model_not_haiku():
    """Web search is unavailable on Haiku, so discovery must not use it."""
    assert "haiku" not in settings.LLM_MODEL_DISCOVERY.lower()
    assert DiscoveryEngine(client=StubMessages(CompanySearchResult())).model \
        == settings.LLM_MODEL_DISCOVERY


def test_postings_come_from_the_board_not_the_model(monkeypatch):
    """The model names a company; the employer's API supplies the jobs."""
    fetched = {}

    def fake_fetch(board, slug):
        fetched["called"] = (board, slug)
        return [posting(title="Senior Backend Engineer"),
                posting(title="Office Manager", external_id="2")]

    monkeypatch.setattr("engine.discovery.resolve_board",
                        lambda company, url="", slug="": ("greenhouse", "acme",
                                                          fake_fetch("greenhouse", "acme")))
    engine = DiscoveryEngine(client=StubMessages(CompanySearchResult()))

    postings, problems = engine.collect_postings(
        [CompanySuggestion(name="Acme", careers_url="https://boards.greenhouse.io/acme")],
        DiscoveryCriteria(titles=["backend"]),
    )
    assert fetched["called"] == ("greenhouse", "acme")
    assert [p.title for p in postings] == ["Senior Backend Engineer"]
    assert problems == []


def test_a_company_with_no_readable_board_is_reported_not_dropped_silently(monkeypatch):
    engine = DiscoveryEngine(client=StubMessages(CompanySearchResult()))
    monkeypatch.setattr("engine.discovery.resolve_board",
                        lambda company, url="", slug="": None)
    postings, problems = engine.collect_postings(
        [CompanySuggestion(name="Mystery Co", careers_url="https://mystery.com/jobs")],
        DiscoveryCriteria(),
    )
    assert postings == []
    assert "Mystery Co" in problems[0]


def test_a_failing_board_does_not_abort_the_whole_run(monkeypatch):
    from engine.boards import BoardError

    def flaky(board, slug):
        if slug == "broken":
            raise BoardError("HTTP 404")
        return [posting(company=slug)]

    def fake_resolve(company, url="", slug=""):
        detected = detect_board(url)
        target = detected[1] if detected else ""
        try:
            return ("greenhouse", target, flaky("greenhouse", target))
        except BoardError:
            return None

    monkeypatch.setattr("engine.discovery.resolve_board", fake_resolve)
    engine = DiscoveryEngine(client=StubMessages(CompanySearchResult()))
    postings, problems = engine.collect_postings([
        CompanySuggestion(name="Broken", careers_url="https://boards.greenhouse.io/broken"),
        CompanySuggestion(name="Fine", careers_url="https://boards.greenhouse.io/fine"),
    ], DiscoveryCriteria())

    assert len(postings) == 1 and len(problems) == 1


def test_max_postings_is_enforced(monkeypatch):
    monkeypatch.setattr(
        "engine.discovery.resolve_board",
        lambda company, url="", slug="": (
            "greenhouse", "acme", [posting(external_id=str(i)) for i in range(50)]))
    engine = DiscoveryEngine(client=StubMessages(CompanySearchResult()))
    postings, _ = engine.collect_postings(
        [CompanySuggestion(name="Acme", careers_url="https://boards.greenhouse.io/acme")],
        DiscoveryCriteria(max_postings=10))
    assert len(postings) == 10


# ---------------- spend cap ----------------


def test_spend_today_only_counts_today(db, user):
    db.record_usage(user.id, "claude-haiku-4-5", "tailor", cost_usd=0.50)
    assert db.spend_today(user.id) == pytest.approx(0.50)

    # Backdate a row: yesterday's spend must not count against today's cap.
    from database.models import LLMUsage

    with db.session() as sess:
        sess.add(LLMUsage(user_id=user.id, model="claude-opus-5", phase="tailor",
                          cost_usd=99.0,
                          created_at=datetime.now(timezone.utc) - timedelta(days=1)))
    assert db.spend_today(user.id) == pytest.approx(0.50)


def test_cap_blocks_once_reached(db, user, monkeypatch):
    monkeypatch.setattr(settings, "DAILY_SPEND_CAP_USD", 1.0)
    assert db.is_over_daily_cap(user.id) is False
    db.record_usage(user.id, "claude-opus-5", "tailor", cost_usd=1.0)
    assert db.is_over_daily_cap(user.id) is True
    assert user.id in db.users_over_daily_cap()


def test_a_per_user_cap_overrides_the_default(db, user, monkeypatch):
    monkeypatch.setattr(settings, "DAILY_SPEND_CAP_USD", 1.0)
    db.update_user(user.id, daily_spend_cap_usd=50.0)
    db.record_usage(user.id, "claude-opus-5", "tailor", cost_usd=10.0)
    assert db.is_over_daily_cap(user.id) is False


def test_a_zero_cap_disables_the_limit(db, user, monkeypatch):
    monkeypatch.setattr(settings, "DAILY_SPEND_CAP_USD", 0.0)
    db.record_usage(user.id, "claude-opus-5", "tailor", cost_usd=999.0)
    assert db.is_over_daily_cap(user.id) is False
    assert db.users_over_daily_cap() == set()


def test_capped_users_are_skipped_when_claiming(db, user, monkeypatch):
    monkeypatch.setattr(settings, "DAILY_SPEND_CAP_USD", 1.0)
    other = db.create_user("eve@example.com", "$argon2id$fake")
    db.enqueue_job(user.id, job_url="https://x.com/1")
    db.enqueue_job(other.id, job_url="https://x.com/2")
    db.record_usage(user.id, "claude-opus-5", "tailor", cost_usd=5.0)

    claimed = db.claim_next_job(exclude_users=db.users_over_daily_cap())
    assert claimed.user_id == other.id


# ---------------- model tiering ----------------


def test_bulk_and_priority_use_different_models(db, user):
    assert db.model_for_job(user.id, priority=False) == settings.LLM_MODEL_BULK
    assert db.model_for_job(user.id, priority=True) == settings.LLM_MODEL_PRIORITY
    assert "haiku" in settings.LLM_MODEL_BULK.lower()


def test_per_user_model_overrides_apply(db, user):
    db.update_user(user.id, model_bulk="claude-sonnet-5")
    assert db.model_for_job(user.id, priority=False) == "claude-sonnet-5"


# ---------------- digest ----------------


def _digest_user(db, due=True):
    """A digest user whose local time is deliberately past (or before) its hour.

    Anchored to the current UTC hour rather than a fixed one, so these tests do
    not pass or fail depending on the time of day.
    """
    from datetime import datetime, timezone

    user = db.create_user("digest@example.com", "$argon2id$fake")
    utc_hour = datetime.now(timezone.utc).hour
    # Shift local time to 12:00, then set the hour either side of it.
    offset = (12 - utc_hour) * 60
    db.update_user(user.id, notify_mode="digest",
                   notify_digest_hour=9 if due else 20,
                   notify_utc_offset_minutes=offset)
    return db.get_user(user.id)


def test_digest_is_due_only_when_something_is_blocked(db):
    from database.models import ActionKind

    user = _digest_user(db)
    assert db.users_due_for_digest() == []          # nothing to report

    db.create_action(user.id, ActionKind.UNMAPPED_FIELD, "Years of Go?")
    assert db.users_due_for_digest() == [(user.id, 1)]


def test_digest_is_not_due_before_the_chosen_hour(db):
    from database.models import ActionKind

    user = _digest_user(db, due=False)
    db.create_action(user.id, ActionKind.UNMAPPED_FIELD, "Years of Go?")
    assert db.users_due_for_digest() == []


def test_digest_is_sent_once_per_day(db):
    from database.models import ActionKind

    user = _digest_user(db)
    db.create_action(user.id, ActionKind.UNMAPPED_FIELD, "Years of Go?")
    assert db.users_due_for_digest() == [(user.id, 1)]

    db.mark_digest_sent(user.id)
    assert db.users_due_for_digest() == []


def test_immediate_mode_users_never_appear_in_the_digest(db):
    from database.models import ActionKind

    user = db.create_user("now@example.com", "$argon2id$fake")
    db.create_action(user.id, ActionKind.UNMAPPED_FIELD, "Years of Go?")
    assert db.users_due_for_digest() == []


def test_digest_mode_suppresses_per_block_alerts(db):
    """The whole point: a hundred blocks make no noise until the summary."""
    from automation.notifier import Notifier
    from web.queue_worker import ParkRegistry, QueueGatekeeper

    class Recorder:
        name = "test"

        def __init__(self):
            self.sent = []

        def send(self, notice):
            self.sent.append(notice)

    channel = Recorder()
    user = _digest_user(db)
    job = db.enqueue_job(user.id, job_url="https://x.com/1")

    keeper = QueueGatekeeper(db, user.id, job.id, ParkRegistry(),
                             Notifier(db, channels=[channel]), lambda: None, lambda: None)
    keeper._notify("something blocked", "needs you")
    assert channel.sent == []

    db.update_user(user.id, notify_mode="immediate")
    keeper._notify("something blocked", "needs you")
    assert len(channel.sent) == 1


# ---------------- discovery as a queued job ----------------


def test_discovery_queues_tailor_jobs_with_descriptions_attached(db, user, monkeypatch):
    """Board postings carry their own description, so no page fetch is needed."""
    from web.runner import run_discovery_job

    db.set_anthropic_key(user.id, "sk-ant-test")
    criteria = DiscoveryCriteria(titles=["backend"], max_postings=5)
    job = db.enqueue_job(user.id, kind="discover",
                         job_description=criteria.model_dump_json())

    class FakeEngine:
        def __init__(self, **kw):
            pass

        def run(self, criteria, already_applied=(), seen_keys=None, **kw):
            return {
                "companies": [CompanySuggestion(name="Acme")],
                # Different employers: one application per company is the cap,
                # so same-company postings would collapse to one and this test
                # is about descriptions, not that rule.
                "postings": [posting(external_id="1", company="Acme",
                                     description="Python and Kubernetes."),
                             posting(external_id="2", company="Globex",
                                     url="https://boards.greenhouse.io/globex/jobs/2",
                                     description="Go and gRPC.")],
                "problems": [],
                "notes": "",
            }

    monkeypatch.setattr("engine.discovery.DiscoveryEngine", FakeEngine)
    run_discovery_job(db, job.id, user.id, gatekeeper=None)

    queued = [j for j in db.list_jobs(user_id=user.id) if j.kind == "tailor"]
    assert len(queued) == 2
    assert all(j.job_description for j in queued), "descriptions must come from the board"
    assert db.get_job(job.id).status.value == "Done"


def test_discovery_skips_postings_already_applied_to(db, user, monkeypatch):
    from web.runner import run_discovery_job

    db.set_anthropic_key(user.id, "sk-ant-test")
    db.create_application(company="Acme", role_title="Senior Backend Engineer",
                          job_url="https://boards.greenhouse.io/acme/jobs/1",
                          user_id=user.id)
    job = db.enqueue_job(user.id, kind="discover",
                         job_description=DiscoveryCriteria().model_dump_json())

    class FakeEngine:
        def __init__(self, **kw):
            pass

        def run(self, criteria, already_applied=(), seen_keys=None, **kw):
            assert "Acme" in already_applied, "applied companies must be excluded upstream"
            return {"companies": [], "postings": [posting(external_id="1")],
                    "problems": [], "notes": ""}

    monkeypatch.setattr("engine.discovery.DiscoveryEngine", FakeEngine)
    run_discovery_job(db, job.id, user.id, gatekeeper=None)

    assert [j for j in db.list_jobs(user_id=user.id) if j.kind == "tailor"] == []


def test_discovery_respects_the_daily_screening_cap(db, user, monkeypatch):
    """Screening is budgeted separately from applications: most postings are
    rejected for one cheap call, so the screening allowance is much larger."""
    from web.runner import run_discovery_job

    monkeypatch.setattr(settings, "DAILY_SCREEN_CAP", 3)
    db.set_anthropic_key(user.id, "sk-ant-test")
    job = db.enqueue_job(user.id, kind="discover",
                         job_description=DiscoveryCriteria().model_dump_json())

    class FakeEngine:
        def __init__(self, **kw):
            pass

        def run(self, criteria, already_applied=(), seen_keys=None, **kw):
            return {"companies": [],
                    # One per employer, so the screening cap is what limits
                    # this and not the per-company cap.
                    "postings": [posting(external_id=str(i), company=f"Co{i}",
                                         url=f"https://boards.greenhouse.io/co{i}/jobs/{i}")
                                 for i in range(10)],
                    "problems": [], "notes": ""}

    monkeypatch.setattr("engine.discovery.DiscoveryEngine", FakeEngine)
    run_discovery_job(db, job.id, user.id, gatekeeper=None)

    assert len([j for j in db.list_jobs(user_id=user.id) if j.kind == "tailor"]) == 3


def test_discovery_stops_when_the_screening_cap_is_already_spent(db, user, monkeypatch):
    from web.runner import run_discovery_job

    monkeypatch.setattr(settings, "DAILY_SCREEN_CAP", 1)
    db.set_anthropic_key(user.id, "sk-ant-test")
    db.enqueue_job(user.id, kind="tailor", job_url="https://x.com/already")
    job = db.enqueue_job(user.id, kind="discover",
                         job_description=DiscoveryCriteria().model_dump_json())

    class FakeEngine:
        def __init__(self, **kw):
            pass

        def run(self, *a, **k):
            return {"companies": [], "postings": [posting()], "problems": [], "notes": ""}

    monkeypatch.setattr("engine.discovery.DiscoveryEngine", FakeEngine)
    run_discovery_job(db, job.id, user.id, gatekeeper=None)

    assert "reached" in db.get_job(job.id).message
    # Only the job that consumed the budget; discovery added none.
    tailor_jobs = [j for j in db.list_jobs(user_id=user.id) if j.kind == "tailor"]
    assert [j.job_url for j in tailor_jobs] == ["https://x.com/already"]


def test_the_spend_cap_check_is_a_fixed_number_of_queries(db, monkeypatch):
    """Regression: this ran once per user per dispatcher tick — four times a
    second — and the resulting query storm hung the whole test suite."""
    monkeypatch.setattr(settings, "DAILY_SPEND_CAP_USD", 1.0)
    for i in range(25):
        u = db.create_user(f"u{i}@example.com", "$argon2id$fake")
        db.record_usage(u.id, "claude-opus-5", "tailor", cost_usd=2.0)

    calls = {"n": 0}
    original = db.get_user

    def counting_get_user(user_id):
        calls["n"] += 1
        return original(user_id)

    monkeypatch.setattr(db, "get_user", counting_get_user)
    assert len(db.users_over_daily_cap()) == 25
    assert calls["n"] == 0, "must not fetch each user individually"


def test_capped_set_is_cached_between_dispatcher_ticks(db, monkeypatch):
    from automation.notifier import Notifier
    from web.queue_worker import QueueWorker

    monkeypatch.setattr(settings, "DAILY_SPEND_CAP_USD", 1.0)
    worker = QueueWorker(db, slots=1, notifier=Notifier(db, channels=[]))

    calls = {"n": 0}
    original = db.users_over_daily_cap

    def counting():
        calls["n"] += 1
        return original()

    monkeypatch.setattr(db, "users_over_daily_cap", counting)
    for _ in range(50):
        worker._capped_users()

    assert calls["n"] == 1, "the ceiling must be recomputed on a timer, not per tick"


def test_midnight_is_a_valid_digest_hour(db):
    """Regression: `hour or 18` treated 0 as unset, so anyone choosing
    midnight silently got 18:00. Only visible between 00:00 and 18:00 UTC."""
    from datetime import datetime, timezone

    from database.models import ActionKind

    user = db.create_user("midnight@example.com", "$argon2id$fake")
    # Offset chosen so the user's local hour is exactly their digest hour.
    local_hour = datetime.now(timezone.utc).hour
    db.update_user(user.id, notify_mode="digest", notify_digest_hour=0,
                   notify_utc_offset_minutes=-local_hour * 60)
    db.create_action(user.id, ActionKind.UNMAPPED_FIELD, "Years of Go?")

    assert db.users_due_for_digest() == [(user.id, 1)]


def test_a_digest_hour_of_zero_is_not_confused_with_unset(db):
    from database.models import ActionKind

    user = db.create_user("late@example.com", "$argon2id$fake")
    # Local time is 01:00, digest hour 0 -> due; digest hour 18 -> not due.
    from datetime import datetime, timezone

    offset = (1 - datetime.now(timezone.utc).hour) * 60
    db.update_user(user.id, notify_mode="digest", notify_digest_hour=0,
                   notify_utc_offset_minutes=offset)
    db.create_action(user.id, ActionKind.UNMAPPED_FIELD, "Years of Go?")
    assert db.users_due_for_digest() == [(user.id, 1)]

    db.update_user(user.id, notify_digest_hour=18)
    assert db.users_due_for_digest() == []


def test_discovery_only_queues_postings_that_could_plausibly_fit(db, user, monkeypatch):
    """Ranking is free local string work on text the board already gave us, so
    a hopeless posting should never reach the queue and cost a tailoring call."""
    from web.runner import run_discovery_job

    db.set_anthropic_key(user.id, "sk-ant-test")
    job = db.enqueue_job(user.id, kind="discover",
                         job_description=DiscoveryCriteria().model_dump_json())

    good = posting(external_id="1", url="https://boards.greenhouse.io/a/jobs/1",
                   description="We need Java, Spring Boot and Kubernetes experience.")
    bad = posting(external_id="2", url="https://boards.greenhouse.io/a/jobs/2",
                  description="We need Haskell, OCaml, Erlang, Prolog and Fortran.")

    captured = {}
    real_rank = DiscoveryEngine.rank_by_fit

    class FakeEngine:
        def __init__(self, **kw):
            pass

        def run(self, criteria, already_applied=(), seen_keys=None,
                profile=None, min_estimated_fit=0.0):
            captured["profile_given"] = profile is not None
            # The real ranking, captured before the patch replaced the class.
            kept, scored = real_rank(self, [good, bad], profile, min_estimated_fit)
            return {"companies": [], "postings": kept, "problems": [],
                    "notes": "", "scored": scored}

    monkeypatch.setattr("engine.discovery.DiscoveryEngine", FakeEngine)
    run_discovery_job(db, job.id, user.id, gatekeeper=None)

    assert captured["profile_given"] is True
    queued = [j.job_url for j in db.list_jobs(user_id=user.id) if j.kind == "tailor"]
    assert good.url in queued
    assert bad.url not in queued, "a posting with no overlap should not be queued"


# ---------------- location filtering ----------------
#
# A discovery run targeting the US queued ClickHouse roles in Canada. The
# filter let any posting through whose location contained the word "remote",
# so "Remote Canada" satisfied a US-only search. For a candidate whose work
# authorisation is US-only, that is not a near miss.

US_SEARCH = ["remote", "united states", "ohio"]


def _at(location):
    return Posting(company="X", title="Senior Backend Engineer",
                   url="https://boards.greenhouse.io/x/jobs/1", location=location)


@pytest.mark.parametrize("location", [
    "Remote Canada", "Toronto, Canada", "Remote India", "Bengaluru, India",
    "Berlin, Germany", "Paris, France", "London, UK", "Belgrade, Serbia",
    "Remote EMEA", "Sydney, Australia", "Sao Paulo, Brazil",
])
def test_a_us_search_rejects_other_countries_even_when_remote(location):
    assert matches(_at(location), ["backend"], US_SEARCH, []) is False


@pytest.mark.parametrize("location", [
    "Remote US", "Remote - United States", "United States", "Columbus, Ohio",
    "New York, NY", "San Francisco, California", "Austin, TX",
])
def test_a_us_search_accepts_us_locations(location):
    assert matches(_at(location), ["backend"], US_SEARCH, []) is True


def test_a_city_outside_the_requested_state_is_still_accepted():
    """Country is the filter that matters. Rejecting a New York role because
    the search said Ohio would discard viable work, and relocation is a
    question the application itself asks."""
    assert matches(_at("New York, NY"), ["backend"], ["ohio"], []) is True


@pytest.mark.parametrize("location", ["Remote", "Hybrid", "", "Flexible"])
def test_a_posting_naming_no_country_is_not_excluded(location):
    """Ambiguous is not disqualifying: better to tailor one extra than to miss
    a real role because the board was vague."""
    assert matches(_at(location), ["backend"], US_SEARCH, []) is True


def test_a_multi_country_posting_including_the_us_is_accepted():
    assert matches(_at("US or Canada"), ["backend"], US_SEARCH, []) is True


def test_the_filter_is_not_us_specific():
    canada = ["remote", "canada"]
    assert matches(_at("Toronto, Canada"), ["backend"], canada, []) is True
    assert matches(_at("Remote Canada"), ["backend"], canada, []) is True
    assert matches(_at("Remote US"), ["backend"], canada, []) is False


def test_no_location_filter_accepts_anything():
    assert matches(_at("Remote Canada"), ["backend"], [], []) is True


def test_screening_is_budgeted_separately_from_applications(db, user, monkeypatch):
    """An unviable posting costs one extraction call and creates nothing, so
    the screening allowance has to be far larger than the application cap or
    twenty matches can never be found."""
    monkeypatch.setattr(settings, "DAILY_SCREEN_CAP", 250)
    monkeypatch.setattr(settings, "DAILY_APPLICATION_CAP", 20)

    for i in range(30):
        db.enqueue_job(user.id, kind="tailor", job_url=f"https://x.com/{i}")
    for i in range(5):
        db.create_application(company=f"Co{i}", role_title="Engineer",
                              job_url=f"https://y.com/{i}", user_id=user.id)

    assert db.screened_today(user.id) == 30
    assert db.applications_today(user.id) == 5
    assert settings.DAILY_SCREEN_CAP > settings.DAILY_APPLICATION_CAP


def test_a_foreign_posting_already_queued_is_dropped_before_it_costs_anything(db, user):
    """Discovery filters location, but rows queued under an older filter are
    still in the queue after it is fixed. A Vietnam-only Java role reached the
    browser that way, so the check is repeated here where the spending starts.
    """
    from web.runner import run_tailor_job

    db.set_anthropic_key(user.id, "sk-ant-test")
    db.save_discovery_criteria(user.id, {"titles": ["Java"],
                                         "locations": ["United States"]})
    job = db.enqueue_job(
        user.id, kind="tailor",
        job_url="https://job-boards.greenhouse.io/aperiasolutions/jobs/5206595007",
        job_location="Danang, Danang, Vietnam; Ho Chi Minh City, Vietnam",
        job_description="Senior Fullstack Java Developer. Spring Boot. 4+ years.")

    run_tailor_job(db, job.id, user.id, gatekeeper=None)

    assert "not a requested location" in db.get_job(job.id).message.lower()
    assert db.applications_today(user.id) == 0


def test_a_us_posting_already_queued_is_not_dropped(db, user, monkeypatch):
    """The backstop must not start rejecting the roles we want. Tailoring is
    stubbed out: what is under test is that the run reaches it at all."""
    from web.runner import run_tailor_job

    reached = []

    class Reached(RuntimeError):
        pass

    def stop_here(self, *a, **kw):
        reached.append(True)
        raise Reached("got past the location gate")

    monkeypatch.setattr("engine.ats_optimizer.ATSOptimizer.run", stop_here)

    db.set_anthropic_key(user.id, "sk-ant-test")
    db.save_discovery_criteria(user.id, {"titles": ["Java"],
                                         "locations": ["United States"]})
    job = db.enqueue_job(
        user.id, kind="tailor", job_url="https://x.com/1",
        job_location="Austin, TX",
        job_description="Senior Java Engineer. Spring Boot, REST APIs. 4+ years.")

    with pytest.raises(Reached):
        run_tailor_job(db, job.id, user.id, gatekeeper=None)

    assert reached, "the location backstop rejected a US posting"


def test_tailoring_stops_once_the_application_cap_is_reached(db, user, monkeypatch):
    """The queue keeps its remaining postings for tomorrow rather than
    tailoring past the cap."""
    from web.runner import run_tailor_job

    monkeypatch.setattr(settings, "DAILY_APPLICATION_CAP", 2)
    db.set_anthropic_key(user.id, "sk-ant-test")
    # The cap counts applications worth sending, so these must clear the bar.
    for i in range(2):
        db.create_application(company=f"Co{i}", role_title="Engineer",
                              job_url=f"https://y.com/{i}", match_score=85.0,
                              user_id=user.id)

    job = db.enqueue_job(
        user.id, kind="tailor", job_url="https://x.com/1",
        job_description="Senior Java Engineer. Spring Boot, REST APIs. 4+ years.")
    run_tailor_job(db, job.id, user.id, gatekeeper=None)

    assert "target already met" in db.get_job(job.id).message.lower()
    # No third application was created.
    assert db.applications_today(user.id) == 2


# ---------------- board resolution ----------------
#
# Eight discovery runs each reported "0 postings queued". The model guesses a
# company's board slug and provider, and gets the provider wrong often: whole
# live boards were discarded as 404s for being offered as Greenhouse when they
# were on Ashby.


def test_slug_candidates_handle_real_company_names():
    from engine.boards import slug_candidates

    assert "globalhealthcareexchange" in slug_candidates("Global Healthcare Exchange")
    assert "doordash" in slug_candidates("DoorDash")
    # Legal suffixes are sometimes kept in the slug and sometimes dropped.
    interactive = slug_candidates("Interactive Brokers, Inc.")
    assert any("interactivebrokers" in c for c in interactive)


def test_resolution_tries_the_same_slug_on_every_provider(monkeypatch):
    """The slug is usually right and the provider usually wrong."""
    from engine import boards

    tried = []

    def fake_fetch(provider, slug):
        tried.append((provider, slug))
        if provider == "ashby" and slug == "confluent":
            return [Posting(company="Confluent", title="Java Engineer",
                            url="https://jobs.ashbyhq.com/confluent/1")]
        raise boards.BoardError("HTTP 404")

    monkeypatch.setattr(boards, "fetch_board", fake_fetch)
    result = boards.resolve_board("Confluent", "https://boards.greenhouse.io/confluent")

    assert result is not None
    provider, slug, postings = result
    assert (provider, slug) == ("ashby", "confluent")
    assert ("greenhouse", "confluent") in tried      # the hint was tried first
    assert len(postings) == 1


def test_resolution_falls_back_to_names_derived_from_the_company(monkeypatch):
    from engine import boards

    def fake_fetch(provider, slug):
        if slug == "globalhealthcareexchange":
            return [Posting(company="GHX", title="Java Engineer", url="u")]
        raise boards.BoardError("HTTP 404")

    monkeypatch.setattr(boards, "fetch_board", fake_fetch)
    result = boards.resolve_board("Global Healthcare Exchange", "https://ghx.com/careers")

    assert result is not None and result[1] == "globalhealthcareexchange"


def test_resolution_gives_up_rather_than_looping(monkeypatch):
    from engine import boards

    calls = []

    def always_404(provider, slug):
        calls.append((provider, slug))
        raise boards.BoardError("HTTP 404")

    monkeypatch.setattr(boards, "fetch_board", always_404)
    assert boards.resolve_board("Nonexistent Co", "https://x.com/careers") is None
    # Each provider/slug pair is attempted at most once.
    assert len(calls) == len(set(calls))


def test_an_empty_board_is_not_treated_as_resolved(monkeypatch):
    """A board that answers with no postings is not the company's board."""
    from engine import boards

    monkeypatch.setattr(boards, "fetch_board", lambda p, s: [])
    assert boards.resolve_board("Ghost Co", "https://boards.greenhouse.io/ghost") is None


# ---------------- queueing what is ready ----------------


def test_lowering_the_threshold_queues_already_tailored_applications(db, user):
    """An application is auto-queued only when tailored. Lowering the bar
    afterwards left three eligible and two queued."""
    from config import settings as cfg

    db.create_application(company="Toast", role_title="Senior Java Engineer",
                          job_url="https://x.com/1", match_score=73.1, user_id=user.id)
    db.create_application(company="Low", role_title="Java Engineer",
                          job_url="https://x.com/2", match_score=52.0, user_id=user.id)

    queued = db.queue_ready_applications(user.id, threshold=70.0)
    assert queued == 1

    apply_jobs = [j for j in db.list_jobs(user_id=user.id) if j.kind == "apply"]
    assert len(apply_jobs) == 1


def test_syncing_twice_does_not_duplicate(db, user):
    db.create_application(company="Toast", role_title="Senior Java Engineer",
                          job_url="https://x.com/1", match_score=73.1, user_id=user.id)
    assert db.queue_ready_applications(user.id, threshold=70.0) == 1
    assert db.queue_ready_applications(user.id, threshold=70.0) == 0


def test_ineligible_applications_are_never_queued_by_sync(db, user):
    db.create_application(company="Branch", role_title="Senior Engineer",
                          job_url="https://x.com/1", match_score=90.0, user_id=user.id,
                          eligible=False, ineligible_reason="will not sponsor")
    assert db.queue_ready_applications(user.id, threshold=70.0) == 0


# ---------------- daily top-up ----------------


def _topup_user(db, hour_offset=0):
    """A user whose local clock has just passed their top-up hour."""
    from datetime import datetime, timezone as tz

    user = db.create_user("topup@example.com", "$argon2id$fake")
    utc_hour = datetime.now(tz.utc).hour
    # Shift local time to 12:00 so the hour can be set either side of it.
    db.update_user(user.id, topup_enabled=True, timezone="UTC",
                   notify_utc_offset_minutes=0, topup_hour=(utc_hour + hour_offset) % 24)
    return db.get_user(user.id)


def test_a_topup_is_due_once_its_hour_has_passed(db):
    user = _topup_user(db, hour_offset=0)
    assert db.users_due_for_topup() == [user.id]


def test_a_topup_is_not_due_before_its_hour(db):
    from datetime import datetime, timezone as tz

    user = db.create_user("later@example.com", "$argon2id$fake")
    db.update_user(user.id, topup_enabled=True, timezone="UTC",
                   topup_hour=(datetime.now(tz.utc).hour + 2) % 24)
    if datetime.now(tz.utc).hour + 2 <= 23:      # skip when the shift wraps midnight
        assert db.users_due_for_topup() == []


def test_a_topup_runs_once_per_day(db):
    user = _topup_user(db)
    assert db.users_due_for_topup() == [user.id]
    db.mark_topup_run(user.id)
    assert db.users_due_for_topup() == []


def test_a_disabled_topup_never_fires(db):
    user = _topup_user(db)
    db.update_user(user.id, topup_enabled=False)
    assert db.users_due_for_topup() == []


def test_the_topup_hour_is_read_in_the_users_timezone(db):
    """3am means 3am where the user is, not on the server."""
    from datetime import datetime, timezone as tz

    user = db.create_user("pacific@example.com", "$argon2id$fake")
    utc_hour = datetime.now(tz.utc).hour
    pacific_hour = (utc_hour - 7) % 24
    db.update_user(user.id, topup_enabled=True, timezone="America/Los_Angeles",
                   topup_hour=pacific_hour)

    # Due in Pacific; the same hour set against UTC would usually not be.
    assert user.id in db.users_due_for_topup()


def test_the_topup_does_nothing_when_the_target_is_already_met(db, user, monkeypatch):
    from web.runner import run_topup_job

    monkeypatch.setattr(settings, "DAILY_APPLICATION_CAP", 2)
    db.set_anthropic_key(user.id, "sk-ant-test")
    for i in range(2):
        db.create_application(company=f"Co{i}", role_title="Java Engineer",
                              job_url=f"https://x.com/{i}", match_score=85.0,
                              user_id=user.id)

    job = db.enqueue_job(user.id, kind="topup")
    run_topup_job(db, job.id, user.id, gatekeeper=None)

    assert "nothing to top up" in db.get_job(job.id).message
    assert [j for j in db.list_jobs(user_id=user.id) if j.kind == "tailor"] == []


def test_a_midnight_topup_hour_is_honoured(db):
    """Hour 0 is falsy, so `topup_hour or 3` quietly moved a midnight top-up to
    three in the morning. The same mistake was fixed once already for the
    digest hour; this is the one that was left."""
    from datetime import datetime, timezone as tz

    user = db.create_user("midnight@example.com", "$argon2id$fake")
    db.update_user(user.id, topup_enabled=True, timezone="UTC",
                   notify_utc_offset_minutes=0, topup_hour=0)

    # At any hour of the day, a midnight schedule has already come round.
    assert datetime.now(tz.utc).hour >= 0
    assert db.users_due_for_topup() == [user.id]


# ---------------- one resume per recruiter ----------------
#
# Every application is tailored to its own posting, so two roles at one company
# means two differently-emphasised resumes in front of the same recruiter - an
# ATS keys candidates by email and shows them together. The comment in the
# discovery path claimed "never queue a company already applied to", but the
# exclusion list was only pasted into the prompt as a request to the model.
# Four Databricks postings, two Stripe and two Tebra reached the queue.


def test_a_second_role_at_the_same_company_is_not_queued(db, user, monkeypatch):
    from web.runner import _within_company_cap

    monkeypatch.setattr(settings, "MAX_APPLICATIONS_PER_COMPANY", 1)
    db.create_application(company="Databricks", role_title="Backend Engineer",
                          job_url="https://x.com/1", user_id=user.id)

    kept = _within_company_cap(db, user.id, [posting(company="Databricks")])

    assert kept == []


def test_one_run_cannot_queue_several_roles_at_one_company(db, user, monkeypatch):
    """The count has to move as the run goes, or every posting is checked
    against the same starting number and all of them pass."""
    from web.runner import _within_company_cap

    monkeypatch.setattr(settings, "MAX_APPLICATIONS_PER_COMPANY", 1)
    postings = [posting(company="Databricks", external_id=str(i)) for i in range(4)]

    kept = _within_company_cap(db, user.id, postings)

    assert len(kept) == 1


def test_other_companies_are_unaffected(db, user, monkeypatch):
    from web.runner import _within_company_cap

    monkeypatch.setattr(settings, "MAX_APPLICATIONS_PER_COMPANY", 1)
    db.create_application(company="Databricks", role_title="Backend Engineer",
                          job_url="https://x.com/1", user_id=user.id)

    kept = _within_company_cap(db, user.id, [
        posting(company="Databricks"), posting(company="Stripe"),
        posting(company="Toast"),
    ])

    assert sorted(p.company for p in kept) == ["Stripe", "Toast"]


def test_the_company_name_is_matched_case_and_space_insensitively(db, user, monkeypatch):
    """Boards spell the same employer several ways."""
    from web.runner import _within_company_cap

    monkeypatch.setattr(settings, "MAX_APPLICATIONS_PER_COMPANY", 1)
    db.create_application(company="databricks", role_title="Engineer",
                          job_url="https://x.com/1", user_id=user.id)

    assert _within_company_cap(db, user.id, [posting(company="  Databricks ")]) == []


def test_the_cap_is_configurable(db, user, monkeypatch):
    """Someone may decide two at a large employer is fine."""
    from web.runner import _within_company_cap

    monkeypatch.setattr(settings, "MAX_APPLICATIONS_PER_COMPANY", 2)
    postings = [posting(company="Databricks", external_id=str(i)) for i in range(4)]

    assert len(_within_company_cap(db, user.id, postings)) == 2


def test_a_posting_with_no_company_name_is_not_dropped(db, user, monkeypatch):
    """An unnamed employer cannot be shown to have been applied to already."""
    from web.runner import _within_company_cap

    monkeypatch.setattr(settings, "MAX_APPLICATIONS_PER_COMPANY", 1)
    assert len(_within_company_cap(db, user.id, [posting(company="")])) == 1


def test_dropping_a_posting_is_logged_not_silent(db, user, monkeypatch):
    """A posting vanishing from the queue with no explanation is the thing that
    made the location bug so hard to see."""
    from web.runner import _within_company_cap

    monkeypatch.setattr(settings, "MAX_APPLICATIONS_PER_COMPANY", 1)
    _within_company_cap(db, user.id, [posting(company="Databricks", external_id=str(i))
                                      for i in range(3)])

    events = [e for e in db.list_logs(user_id=user.id, limit=20)
              if e.event == "company_cap"]
    assert events and "Databricks" in events[0].message


def test_drafts_count_towards_the_cap(db, user, monkeypatch):
    """A queued draft is just as much a second resume as a submitted one."""
    from web.runner import _within_company_cap

    monkeypatch.setattr(settings, "MAX_APPLICATIONS_PER_COMPANY", 1)
    db.create_application(company="Stripe", role_title="Engineer",
                          job_url="https://x.com/1", user_id=user.id,
                          match_score=20.0)

    assert _within_company_cap(db, user.id, [posting(company="Stripe")]) == []


# ---------------- one resume per employer, reused ----------------
#
# A second role at a company already applied to goes out on the resume that
# company already has - not a freshly tailored variant. Two differently
# emphasised resumes are two versions of one person to a recruiter who sees
# both. It follows that the 70% bar must be measured against the resume that
# will actually be sent, not against one built for this posting.


def _stored_resume(db, user, company="Databricks", skills=("Java", "Spring Boot")):
    payload = {
        "summary": "Backend engineer building Java services.",
        "highlighted_skills": list(skills),
        "tailored_experience": [{
            "title": "Senior Engineer", "company": "Acme",
            "start_date": "2020-01",
            "bullets": ["Built Java and Spring Boot services on Kubernetes."],
        }],
        "ats_match_percentage": 88.0,
    }
    return db.create_application(
        company=company, role_title="Backend Engineer",
        job_url="https://boards.greenhouse.io/databricks/jobs/1",
        match_score=88.0, tailored_payload=payload,
        resume_pdf_path="/tmp/prior.pdf", user_id=user.id)


def test_the_resume_already_sent_is_the_one_looked_up(db, user):
    prior = _stored_resume(db, user)
    found = db.resume_for_company("Databricks", user_id=user.id)
    assert found is not None and found.id == prior.id


def test_the_lookup_ignores_case_and_padding(db, user):
    _stored_resume(db, user, company="databricks")
    assert db.resume_for_company("  Databricks ", user_id=user.id) is not None


def test_an_employer_never_applied_to_has_no_stored_resume(db, user):
    _stored_resume(db, user)
    assert db.resume_for_company("Stripe", user_id=user.id) is None


def test_one_users_resume_is_not_offered_to_another(db, user):
    _stored_resume(db, user)
    other = db.create_user("eve@example.com", "$argon2id$fake")
    assert db.resume_for_company("Databricks", user_id=other.id) is None


def test_an_application_with_no_resume_is_not_offered_for_reuse(db, user):
    """A row that only got as far as being queued has nothing to reuse."""
    db.create_application(company="Stripe", role_title="Engineer",
                          job_url="https://x.com/1", user_id=user.id)
    assert db.resume_for_company("Stripe", user_id=user.id) is None


def test_the_stored_resume_is_scored_against_the_new_posting(db, user):
    """The bar applies to the resume that will be sent. A stored resume full of
    Java scores well on a Java posting and badly on a Go one, and that
    difference is the whole point of checking."""
    from engine.ats_optimizer import score_match
    from engine.schemas import JDKeywords, TailoredResumeSchema
    import json as _json

    prior = _stored_resume(db, user)
    resume = TailoredResumeSchema.model_validate(_json.loads(prior.tailored_payload))

    java = JDKeywords(role_title="Java Engineer", company="Databricks",
                      hard_skills=[], tooling=["Java", "Spring Boot"], soft_skills=[])
    rust = JDKeywords(role_title="Rust Engineer", company="Databricks",
                      hard_skills=[], tooling=["Rust", "WebAssembly"], soft_skills=[])

    assert score_match(java, resume)[0] > score_match(rust, resume)[0]


def test_the_company_cap_is_off_by_default():
    """Capping at one rejected every posting worth screening, because almost
    every board had already been applied to once."""
    assert settings.MAX_APPLICATIONS_PER_COMPANY == 0


def test_a_cap_of_zero_lets_every_posting_through(db, user, monkeypatch):
    from web.runner import _within_company_cap

    monkeypatch.setattr(settings, "MAX_APPLICATIONS_PER_COMPANY", 0)
    db.create_application(company="Databricks", role_title="Engineer",
                          job_url="https://x.com/1", user_id=user.id)

    postings = [posting(company="Databricks", external_id=str(i)) for i in range(4)]
    assert len(_within_company_cap(db, user.id, postings)) == 4


# ---------------- the target is a target, not one attempt ----------------
#
# A top-up ran once a day. When it fell short - 11/20, then 19/20 - the gap sat
# there until the next morning, and the only thing that ever closed it was the
# user noticing and asking. That is the whole complaint: it should be
# automatic. So a run that leaves the day short schedules another one.


def _topup_ready_user(db, ran_minutes_ago=60):
    from datetime import datetime, timedelta, timezone as tz

    user = db.create_user("topup2@example.com", "$argon2id$fake")
    db.update_user(user.id, topup_enabled=True, timezone="UTC",
                   notify_utc_offset_minutes=0, topup_hour=0)
    db.update_user(user.id, last_topup_at=datetime.now(tz.utc).replace(tzinfo=None)
                   - timedelta(minutes=ran_minutes_ago))
    return db.get_user(user.id)


def test_a_short_day_schedules_another_run(db, monkeypatch):
    """The bug in one test: already ran today, still short, must run again."""
    monkeypatch.setattr(settings, "DAILY_APPLICATION_CAP", 20)
    user = _topup_ready_user(db)

    assert db.users_due_for_topup() == [user.id]


def test_a_met_target_does_not_schedule_more(db, monkeypatch):
    monkeypatch.setattr(settings, "DAILY_APPLICATION_CAP", 2)
    user = _topup_ready_user(db)
    for i in range(2):
        db.create_application(company=f"Co{i}", role_title="Java Engineer",
                              job_url=f"https://x.com/{i}", match_score=88.0,
                              user_id=user.id)

    assert db.users_due_for_topup() == []


def test_it_waits_rather_than_piling_on_work_already_running(db, monkeypatch):
    """Retrying while the previous run's postings are still being screened
    would queue the same shortfall twice."""
    monkeypatch.setattr(settings, "DAILY_APPLICATION_CAP", 20)
    user = _topup_ready_user(db)
    db.enqueue_job(user.id, kind="tailor", job_url="https://x.com/1")

    assert db.users_due_for_topup() == []


def test_it_waits_out_the_cooldown_before_trying_again(db, monkeypatch):
    """Nothing changes in the space of a minute; retrying that fast just
    re-reads the same boards."""
    monkeypatch.setattr(settings, "DAILY_APPLICATION_CAP", 20)
    monkeypatch.setattr(settings, "TOPUP_RETRY_MINUTES", 20)
    user = _topup_ready_user(db, ran_minutes_ago=2)

    assert db.users_due_for_topup() == []


def test_a_dry_day_stops_after_its_allowance(db, monkeypatch):
    """A market with nothing in it must not spin all day."""
    monkeypatch.setattr(settings, "DAILY_APPLICATION_CAP", 20)
    monkeypatch.setattr(settings, "TOPUP_MAX_RUNS_PER_DAY", 3)
    user = _topup_ready_user(db)
    for _ in range(3):
        db.enqueue_job(user.id, kind="topup")
    # Those count as runs, but must not also count as work in flight.
    for job in db.list_jobs(user_id=user.id, limit=10):
        db.finish_job(job.id, __import__("database.models", fromlist=["JobStatus"]).JobStatus.DONE, "")

    assert db.topup_runs_today(user.id) == 3
    assert db.users_due_for_topup() == []


def test_a_disabled_topup_never_retries(db, monkeypatch):
    monkeypatch.setattr(settings, "DAILY_APPLICATION_CAP", 20)
    user = _topup_ready_user(db)
    db.update_user(user.id, topup_enabled=False)

    assert db.users_due_for_topup() == []


# ---------------- the screened list has to grow by itself ----------------
#
# The pipeline could only ever read a hand-written list of companies. When that
# list ran dry - 53 boards read, 0 eligible postings, 19/20 ready - the day
# stopped short, and the only way the list ever grew was someone editing
# settings.py. Discovery already finds employers by web search and resolves
# their boards; nothing kept the result.


def test_a_discovered_board_is_remembered(db, user):
    db.learn_board("Braze", "greenhouse", "braze", user_id=user.id)
    assert "Braze" in db.learned_boards(user.id)


def test_learning_the_same_board_twice_does_not_duplicate_it(db, user):
    """Discovery turns up the same employers run after run."""
    db.learn_board("Braze", "greenhouse", "braze", user_id=user.id)
    db.learn_board("braze", "greenhouse", "braze", user_id=user.id)
    assert db.learned_boards(user.id).count("Braze") == 1
    assert len(db.learned_boards(user.id)) == 1


def test_a_later_sighting_refreshes_the_slug(db, user):
    """A board that moved provider should not stay wrong forever."""
    db.learn_board("Braze", "greenhouse", "braze-old", user_id=user.id)
    db.learn_board("Braze", "ashby", "braze", user_id=user.id)

    from database.models import KnownBoard

    with db.session() as sess:
        row = sess.query(KnownBoard).filter_by(company="Braze").one()
        assert (row.provider, row.slug) == ("ashby", "braze")


def test_one_users_boards_are_not_offered_to_another(db, user):
    db.learn_board("Braze", "greenhouse", "braze", user_id=user.id)
    other = db.create_user("eve2@example.com", "$argon2id$fake")
    assert db.learned_boards(other.id) == []


def test_a_nameless_board_is_ignored(db, user):
    db.learn_board("   ", "greenhouse", "x", user_id=user.id)
    assert db.learned_boards(user.id) == []


def test_a_productive_board_records_what_it_yielded(db, user):
    """Which learned boards actually work is worth knowing."""
    from database.models import KnownBoard

    db.learn_board("Braze", "greenhouse", "braze", user_id=user.id)
    db.record_board_yield("Braze", 6, user_id=user.id)

    with db.session() as sess:
        row = sess.query(KnownBoard).filter_by(company="Braze").one()
        assert row.eligible_seen == 6
        assert row.last_yield_at is not None


def test_recording_a_yield_for_an_unknown_board_is_harmless(db, user):
    db.record_board_yield("NeverSeen", 3, user_id=user.id)
    assert db.learned_boards(user.id) == []


def test_a_zero_yield_is_not_recorded_as_a_hit(db, user):
    from database.models import KnownBoard

    db.learn_board("Quiet", "greenhouse", "quiet", user_id=user.id)
    db.record_board_yield("Quiet", 0, user_id=user.id)

    with db.session() as sess:
        assert sess.query(KnownBoard).filter_by(company="Quiet").one().eligible_seen == 0


def test_a_running_topup_does_not_block_its_own_decision(db, user):
    """The first version of this asked "is anything running?" from inside a
    running top-up, so the answer was always yes and the search never widened -
    the retry fired, found nothing, and stopped."""
    job = db.enqueue_job(user.id, kind="topup")
    from database.models import JobStatus

    db.start_job(job.id) if hasattr(db, "start_job") else None

    assert db.has_screening_in_flight(user.id) is True
    assert db.has_screening_in_flight(user.id, exclude_job_id=job.id) is False


def test_other_work_still_blocks_widening(db, user):
    """Excluding your own job must not blind you to everything else."""
    mine = db.enqueue_job(user.id, kind="topup")
    db.enqueue_job(user.id, kind="tailor", job_url="https://x.com/1")

    assert db.has_screening_in_flight(user.id, exclude_job_id=mine.id) is True
