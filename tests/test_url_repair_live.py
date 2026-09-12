"""Form and location resolution, checked against the real postings in the queue.

Marked `live` because it reaches real employer sites. The hermetic coverage is
in test_apply_url.py and test_location_filter.py; this exists to confirm the
resolvers' verdicts still match what those sites actually serve.

    pytest tests/test_url_repair_live.py -m live

The URLs below are copied verbatim from the deployed queue. That matters: an
earlier version of this file held URLs clipped to 74 characters by a diagnostic
print, which dropped the last digit of every Databricks job id. Every one of
them 404'd, and the conclusion drawn from that - "Databricks has no automatable
form" - was wrong. Copy whole URLs.
"""

from __future__ import annotations

import time
import urllib.request

import pytest

from engine.boards import (
    BoardError,
    find_application_form,
    location_allowed,
    location_for_url,
    looks_like_an_application_form,
)

US_SEARCH = ["Remote", "United States"]

QUEUED = [
    "https://careers.toasttab.com/jobs?gh_jid=7590046",
    "https://stripe.com/jobs/search?gh_jid=7217048",
    "https://stripe.com/jobs/search?gh_jid=8107302",
    "https://stripe.com/jobs/search?gh_jid=8069941",
    "https://databricks.com/company/careers/open-positions/job?gh_jid=7643201002",
    "https://databricks.com/company/careers/open-positions/job?gh_jid=6779084002",
    "https://databricks.com/company/careers/open-positions/job?gh_jid=6544435002",
    "https://databricks.com/company/careers/open-positions/job?gh_jid=4799387002",
]


# ---------------- the page that holds the form ----------------


@pytest.mark.live
@pytest.mark.parametrize("stored", QUEUED)
def test_a_queued_link_resolves_to_a_page_that_can_be_filled(stored):
    """The failure this fixes reported "no resume input" and "submit button not
    found". Both must be answerable on the page the resolver picks."""
    # These all hit one host in quick succession, and Greenhouse throttles.
    # A throttled read says nothing about whether the code is right, so it is
    # reported as a skip rather than a failure - a live test that cries wolf
    # gets ignored, which is worse than one that occasionally abstains.
    time.sleep(1.0)
    try:
        resolved = find_application_form(stored)
    except BoardError as exc:
        pytest.skip(f"board did not answer conclusively: {exc}")
    assert resolved is not None, f"no form found for {stored}"

    request = urllib.request.Request(resolved, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8", "replace")

    assert looks_like_an_application_form(body)
    assert 'type="file"' in body, "no resume upload on the resolved page"


# ---------------- where the posting is ----------------


@pytest.mark.live
def test_the_vietnam_posting_reports_its_real_location():
    """The posting that got through: a row with no stored location must still
    be identifiable, or the backstop silently skips it."""
    where = location_for_url(
        "https://job-boards.greenhouse.io/aperiasolutions/jobs/5206595007")

    assert "vietnam" in where.lower(), where
    assert not location_allowed(where, US_SEARCH)


@pytest.mark.live
def test_a_foreign_posting_in_the_queue_is_caught():
    """One of the queued Databricks roles is in Denmark. It reads as an
    ordinary US-company posting until the board is asked where it is."""
    where = location_for_url(
        "https://databricks.com/company/careers/open-positions/job?gh_jid=7643201002")

    assert "denmark" in where.lower(), where
    assert not location_allowed(where, US_SEARCH)


@pytest.mark.live
@pytest.mark.parametrize("url", [
    "https://stripe.com/jobs/search?gh_jid=7217048",
    "https://databricks.com/company/careers/open-positions/job?gh_jid=6544435002",
])
def test_an_employer_careers_link_still_yields_a_location(url):
    """These name no board, so the account is guessed from the hostname and
    confirmed against the board. Without this the filter has nothing to check
    and the posting passes unexamined."""
    assert location_for_url(url), f"no location resolved for {url}"


@pytest.mark.live
def test_a_hostname_that_is_not_the_board_account_yields_nothing():
    """The guess is confirmed against the board, so a wrong one must not return
    some other company's location."""
    assert location_for_url(
        "https://not-a-real-company-xyz.com/jobs?gh_jid=7217048") == ""


@pytest.mark.live
def test_an_unidentifiable_url_reports_nothing_rather_than_guessing():
    assert location_for_url("https://careers.acme.com/roles/1") == ""


@pytest.mark.live
def test_an_ashby_posting_reports_its_location():
    """Ashby and Lever postings were coming back "unknown", which the backstop
    treats as unexamined rather than as allowed."""
    where = location_for_url(
        "https://jobs.ashbyhq.com/plaid/ecc50d24-e303-480d-85d7-3041c1508cfe")

    assert where, "no location resolved for the queued Ashby posting"


# ---------------- Workable, against a live account ----------------
#
# The client shipped unverified because every account tried at the time
# returned an empty list - which turned out to be true of those accounts, not
# a fault in the client. Innovatrics is one that is actually hiring.


@pytest.mark.live
def test_the_workable_client_reads_a_live_board():
    from engine.boards import workable

    postings = workable("innovatrics")

    assert postings, "no postings parsed from a board that has them"
    assert all(p.board == "workable" for p in postings)
    assert all(p.slug == "innovatrics" for p in postings)
    assert all(p.title for p in postings), "a posting came back with no title"
    assert all(p.external_id for p in postings), "no shortcode, so no apply URL"
    assert any(p.location for p in postings), "no location on any posting"
    assert any(p.description for p in postings), "no description on any posting"


@pytest.mark.live
def test_a_workable_apply_url_reaches_a_real_form():
    """The URL is built from the shortcode rather than taken from the board, so
    it is worth confirming it actually lands somewhere."""
    from engine.boards import workable

    posting = next(p for p in workable("innovatrics") if p.external_id)
    request = urllib.request.Request(
        posting.apply_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.status == 200


@pytest.mark.live
def test_workable_locations_are_filtered_like_any_other_board():
    """Innovatrics is in Slovakia, so a US search must reject the lot."""
    from engine.boards import location_allowed, workable

    postings = [p for p in workable("innovatrics") if p.location]
    assert postings
    assert not any(location_allowed(p.location, US_SEARCH) for p in postings), (
        "a Slovakian posting passed a US-only search")
