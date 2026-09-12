"""A browser that will not start must not take the rest of the queue with it.

One stale Chrome held the persistent profile, so the first launch failed with
"Opening in existing browser session". The thirteen jobs behind it then failed
with something entirely different:

    It looks like you are using Playwright Sync API inside the asyncio loop.
    Please use the Async API instead.

The cause was ours. `start()` began the Playwright driver and then launched the
browser; when the launch raised, the exception came out of `__enter__`, and
Python does not call `__exit__` when `__enter__` raises. So `close()` never
ran, the driver stayed alive, and every later `sync_playwright()` in that
process hit a leftover event loop.

One failed launch, fourteen failed applications, thirteen of them for a reason
that had nothing to do with them.
"""

from __future__ import annotations

import pytest

from automation.stealth_browser import BrowserUnavailable, HumanBrowser


class FakeDriver:
    """Stands in for the Playwright driver, recording whether it was stopped."""

    def __init__(self, error: Exception):
        self.error = error
        self.stopped = False
        self.chromium = self

    def launch_persistent_context(self, **kwargs):
        raise self.error

    def stop(self):
        self.stopped = True


@pytest.fixture()
def failing_launch(monkeypatch, tmp_path):
    """A HumanBrowser whose launch fails, with the driver observable."""

    def build(error: Exception):
        driver = FakeDriver(error)
        monkeypatch.setattr(
            "playwright.sync_api.sync_playwright",
            lambda: type("P", (), {"start": staticmethod(lambda: driver)})(),
        )
        return HumanBrowser(user_data_dir=tmp_path / "profile"), driver

    return build


PROFILE_CLASH = RuntimeError(
    "BrowserType.launch_persistent_context: Opening in existing browser session. "
    "This usually means that the profile is already in use by another instance "
    "of Chromium."
)


def test_a_failed_launch_stops_the_driver_it_started(failing_launch):
    """The whole bug in one assertion: leaving this running is what poisoned
    every later job in the process."""
    browser, driver = failing_launch(PROFILE_CLASH)

    with pytest.raises(BrowserUnavailable):
        browser.start()

    assert driver.stopped, "the Playwright driver was left running"


def test_the_same_holds_when_used_as_a_context_manager(failing_launch):
    """__exit__ never runs if __enter__ raises, which is how this went
    unnoticed."""
    browser, driver = failing_launch(PROFILE_CLASH)

    with pytest.raises(BrowserUnavailable):
        with browser:
            pass

    assert driver.stopped


def test_a_profile_clash_says_what_to_do_about_it(failing_launch):
    """Playwright's own message is forty lines of launch flags."""
    browser, _ = failing_launch(PROFILE_CLASH)

    with pytest.raises(BrowserUnavailable) as caught:
        browser.start()

    message = str(caught.value)
    assert "already using the browser profile" in message
    assert "profile" in message
    assert "--disable-field-trial-config" not in message, "leaked the launch flags"


def test_an_unrecognised_failure_still_reports_its_first_line(failing_launch):
    browser, _ = failing_launch(RuntimeError("Executable doesn't exist at /nope\nmore detail"))

    with pytest.raises(BrowserUnavailable) as caught:
        browser.start()

    assert "Executable doesn't exist" in str(caught.value)
    assert "more detail" not in str(caught.value)


def test_the_browser_is_left_in_a_clean_state_after_a_failed_launch(failing_launch):
    """A half-built browser must not look usable to anything that kept a
    reference to it."""
    browser, _ = failing_launch(PROFILE_CLASH)

    with pytest.raises(BrowserUnavailable):
        browser.start()

    assert browser._playwright is None
    assert browser.context is None
    assert browser.page is None
