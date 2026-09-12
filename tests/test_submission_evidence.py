"""Reporting an application as sent only when the employer says it was.

`submit` used to return True as soon as the click landed and the page settled,
so "Submitted (6 fields filled)" meant no more than "a button was pressed".
There was no receipt, no screenshot on the agent path, and nothing a person
could check afterwards.

Three outcomes are now distinguishable, and they are not the same thing:

    submitted     the page showed a receipt
    unconfirmed   the click landed and nothing said it worked
    not submitted the run stopped before clicking
"""

from __future__ import annotations

import pytest

from automation.ats_drivers.base_driver import CONFIRMATION_MARKERS, BaseATSDriver


class FakePage:
    """A page whose text and URL can be scripted per read."""

    def __init__(self, texts, url="https://jobs.example.com/apply"):
        self._texts = list(texts)
        self.url = url
        self.waits = 0

    def inner_text(self, selector, timeout=None):
        return self._texts.pop(0) if self._texts else ""

    def wait_for_timeout(self, ms):
        self.waits += 1


class FakeBrowser:
    def __init__(self, page):
        self.page = page


def driver_with(page):
    driver = BaseATSDriver.__new__(BaseATSDriver)
    driver.browser = FakeBrowser(page)
    return driver


# ---------------- reading the receipt ----------------


@pytest.mark.parametrize("marker", CONFIRMATION_MARKERS)
def test_every_confirmation_wording_is_recognised(marker):
    page = FakePage([f"Some heading. {marker.title()}. Back to jobs."])
    assert driver_with(page).confirmation_evidence("https://jobs.example.com/apply")


def test_a_page_that_still_shows_the_form_is_not_a_confirmation():
    """The exact failure: the click landed, the page settled, and nothing on it
    said the application was taken."""
    form = "Name Email Phone Submit application"
    page = FakePage([form, form, form])
    assert driver_with(page).confirmation_evidence("https://jobs.example.com/apply") == ""


def test_a_confirmation_that_appears_late_is_still_found():
    """Some portals render the receipt a beat after the page settles."""
    page = FakePage(["still working...", "still working...",
                     "Thank you for applying to Plaid"])
    assert driver_with(page).confirmation_evidence("https://jobs.example.com/apply")
    assert page.waits >= 1


def test_a_redirect_to_a_confirmation_page_counts():
    """Not every ATS words it; some just navigate."""
    page = FakePage(["", "", ""], url="https://jobs.example.com/application/confirmation")
    evidence = driver_with(page).confirmation_evidence("https://jobs.example.com/apply")
    assert "confirmation" in evidence


def test_a_redirect_to_an_unrelated_page_does_not_count():
    """Bouncing back to the job list is not a receipt."""
    page = FakePage(["", "", ""], url="https://jobs.example.com/jobs")
    assert driver_with(page).confirmation_evidence("https://jobs.example.com/apply") == ""


def test_an_unreadable_page_is_not_treated_as_confirmation():
    class Exploding(FakePage):
        def inner_text(self, selector, timeout=None):
            raise RuntimeError("detached")

    page = Exploding([], url="https://jobs.example.com/apply")
    assert driver_with(page).confirmation_evidence("https://jobs.example.com/apply") == ""
