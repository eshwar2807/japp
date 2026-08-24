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
from engine.boards import Posting, dedupe, detect_board, location_allowed, matches
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

    monkeypatch.setattr("engine.discovery.fetch_board", fake_fetch)
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

    monkeypatch.setattr("engine.discovery.fetch_board", flaky)
    engine = DiscoveryEngine(client=StubMessages(CompanySearchResult()))
    postings, problems = engine.collect_postings([
        CompanySuggestion(name="Broken", careers_url="https://boards.greenhouse.io/broken"),
        CompanySuggestion(name="Fine", careers_url="https://boards.greenhouse.io/fine"),
    ], DiscoveryCriteria())

    assert len(postings) == 1 and len(problems) == 1


def test_max_postings_is_enforced(monkeypatch):
    monkeypatch.setattr("engine.discovery.fetch_board",
                        lambda b, s: [posting(external_id=str(i)) for i in range(50)])
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
                "postings": [posting(external_id="1", description="Python and Kubernetes."),
                             posting(external_id="2", url="https://boards.greenhouse.io/acme/jobs/2",
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
                    "postings": [posting(external_id=str(i),
                                         url=f"https://boards.greenhouse.io/acme/jobs/{i}")
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
