"""Finding the page that actually holds the application form.

The agent kept stopping with "no resume input", "1 field discovered" and
"submit button not found". None of those were form-filling faults: it was being
sent to the employer's careers listing, which has no form on it at all.

Two routes exist for a Greenhouse posting and neither works everywhere, so both
are tried and the answer is decided by what the page contains:

    Stripe      hosted page redirects to their site   embed works
    Flexport    hosted page works                     embed works

A third outcome matters as much as the other two: "could not tell". A timeout
or a rate-limit is not evidence that a posting has no form, and reporting it as
one would quietly turn a real application into manual work.
"""

from __future__ import annotations

import urllib.error

import pytest

from engine.boards import (
    BoardError,
    apply_url_candidates,
    find_application_form,
    looks_like_an_application_form,
)

FORM_PAGE = """
<html><title>Job Application for Backend Engineer at Stripe</title>
<input type="file" name="resume"><button>Submit Application</button></html>
"""

LISTING_PAGE = """
<html><title>Stripe Careers | Open Roles</title>
<input type="text" name="q" placeholder="Search for a role">
<a href="/jobs/1">AEO and GEO Marketing Manager</a></html>
"""


# ---------------- telling the two pages apart ----------------


def test_an_application_page_is_recognised():
    assert looks_like_an_application_form(FORM_PAGE)


def test_a_careers_listing_is_not_mistaken_for_a_form():
    """This page answers 200 and looks perfectly healthy."""
    assert not looks_like_an_application_form(LISTING_PAGE)


@pytest.mark.parametrize("html", ["", None])
def test_an_empty_response_is_not_a_form(html):
    assert not looks_like_an_application_form(html)


# ---------------- which addresses get tried ----------------


def test_both_greenhouse_routes_are_offered():
    """Neither the hosted page nor the embed works everywhere, so a posting is
    not written off until both have been tried."""
    urls = apply_url_candidates("https://stripe.com/jobs/search?gh_jid=7217048")
    assert "https://job-boards.greenhouse.io/stripe/jobs/7217048" not in urls, (
        "the slug is not knowable from a stripe.com URL")
    assert "https://boards.greenhouse.io/embed/job_app?token=7217048" in urls


def test_the_slug_is_used_when_it_is_known():
    urls = apply_url_candidates(
        "https://boards.greenhouse.io/flexport/jobs/7975365", job_id="7975365")
    assert "https://job-boards.greenhouse.io/flexport/jobs/7975365" in urls
    assert ("https://job-boards.greenhouse.io/embed/job_app"
            "?for=flexport&token=7975365") in urls


def test_the_original_url_is_tried_first():
    """It is frequently correct, and trying it first avoids a needless fetch."""
    urls = apply_url_candidates("https://jobs.lever.co/acme/abc/apply")
    assert urls[0] == "https://jobs.lever.co/acme/abc/apply"


def test_candidates_are_not_repeated():
    urls = apply_url_candidates(
        "https://boards.greenhouse.io/embed/job_app?token=7217048")
    assert len(urls) == len(set(urls))


# ---------------- choosing between them ----------------


@pytest.fixture()
def fetches(monkeypatch):
    """Serve canned pages per URL and record what was requested."""
    asked: list[str] = []

    def fake(pages):
        def urlopen(request, timeout=None):
            url = request.full_url
            asked.append(url)
            if url not in pages:
                # A real "not here", which settles that candidate. A bare
                # OSError would mean "could not tell", which is different.
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

            class Response:
                def read(self_inner):
                    return pages[url].encode()

                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *a):
                    return False

            return Response()

        monkeypatch.setattr("engine.boards.urllib.request.urlopen", urlopen)
        return asked

    return fake


def test_the_first_page_holding_a_form_wins(fetches):
    embed = "https://boards.greenhouse.io/embed/job_app?token=7217048"
    listing = "https://stripe.com/jobs/search?gh_jid=7217048"
    fetches({listing: LISTING_PAGE, embed: FORM_PAGE})

    assert find_application_form(listing) == embed


def test_a_reachable_page_with_no_form_is_rejected_not_accepted(fetches):
    """The Stripe failure in one line: 200 is not the same as usable."""
    listing = "https://stripe.com/jobs/search?gh_jid=7217048"
    fetches({listing: LISTING_PAGE})

    assert find_application_form(listing) is None


def test_an_employer_with_no_automatable_form_returns_none(fetches):
    """Some employers really do only accept applications through their own
    scripted site, and None is the honest answer for those.

    No real employer is named here on purpose. Databricks was named originally,
    on the strength of URLs a diagnostic had clipped to 74 characters - every
    one 404'd, which looked exactly like "no form exists". With the whole id
    their postings serve a form perfectly well.
    """
    listing = "https://careers.example.com/roles/1?gh_jid=999999"
    fetches({listing: LISTING_PAGE})

    assert find_application_form(listing) is None


def test_an_unreachable_candidate_does_not_stop_the_search(fetches):
    """A 404 on one route says nothing about the other."""
    embed = "https://boards.greenhouse.io/embed/job_app?token=7217048"
    asked = fetches({embed: FORM_PAGE})

    assert find_application_form(
        "https://stripe.com/jobs/search?gh_jid=7217048") == embed
    assert len(asked) > 1, "gave up before trying the embed route"


# ---------------- "no form" is not the same as "could not tell" ----------------


def test_a_timeout_does_not_get_reported_as_no_form(monkeypatch):
    """Answering None to a transient failure would write off a real
    application as manual work. The caller has to be able to tell them apart."""
    def always_times_out(request, timeout=None):
        raise TimeoutError("too slow")

    monkeypatch.setattr("engine.boards.urllib.request.urlopen", always_times_out)

    with pytest.raises(BoardError):
        find_application_form("https://stripe.com/jobs/search?gh_jid=7217048")


def test_a_rate_limit_is_inconclusive_too(monkeypatch):
    def rate_limited(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many", {}, None)

    monkeypatch.setattr("engine.boards.urllib.request.urlopen", rate_limited)

    with pytest.raises(BoardError):
        find_application_form("https://stripe.com/jobs/search?gh_jid=7217048")


def test_a_404_settles_the_question(monkeypatch):
    """Every candidate answering "not here" is a real answer, not a failure."""
    def missing(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr("engine.boards.urllib.request.urlopen", missing)

    assert find_application_form("https://stripe.com/jobs/search?gh_jid=7217048") is None


def test_a_form_found_after_a_failure_still_wins(monkeypatch):
    """One flaky candidate must not hide a working one."""
    embed = "https://boards.greenhouse.io/embed/job_app?token=7217048"

    def flaky(request, timeout=None):
        if request.full_url != embed:
            raise TimeoutError("too slow")

        class Response:
            def read(self):
                return FORM_PAGE.encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return Response()

    monkeypatch.setattr("engine.boards.urllib.request.urlopen", flaky)

    assert find_application_form(
        "https://stripe.com/jobs/search?gh_jid=7217048") == embed
