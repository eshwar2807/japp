"""Module C validation: timing, guardrails, driver routing, live form scanning.

The form-scan tests drive a real headless Chromium against a fixture page that
reproduces the label patterns Greenhouse and Workday actually use. They are
skipped automatically when the browser binary is not installed.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from automation.ats_drivers import (
    BaseATSDriver,
    GreenhouseDriver,
    WorkdayDriver,
    get_driver_class,
)
from automation.stealth_browser import (
    CAPTCHA_SIGNATURES,
    HumanBrowser,
    bezier_path,
    confirm,
    gaussian_delay_ms,
    think_pause,
)
from engine.screener_mapper import FieldType

FIXTURE = Path(__file__).parent / "fixtures" / "sample_application_form.html"


# ---------------- timing ----------------


def test_keystroke_delays_stay_in_the_specified_band():
    samples = [gaussian_delay_ms() for _ in range(4000)]
    assert min(samples) >= 40.0
    assert max(samples) <= 120.0


def test_keystroke_delays_are_actually_distributed():
    """Clamping must not collapse the distribution into a constant."""
    samples = [gaussian_delay_ms() for _ in range(4000)]
    assert 60 < statistics.mean(samples) < 100
    assert statistics.pstdev(samples) > 8
    assert len(set(round(s, 3) for s in samples)) > 1000


def test_think_pause_range():
    assert all(0.35 <= think_pause() <= 1.4 for _ in range(200))


# ---------------- mouse path ----------------


def test_bezier_path_hits_both_endpoints():
    path = bezier_path((0, 0), (300, 200), steps=20)
    assert len(path) == 21
    assert path[0] == pytest.approx((0, 0), abs=1e-6)
    assert path[-1] == pytest.approx((300, 200), abs=1e-6)


def test_bezier_path_is_not_a_straight_line():
    """A perfectly collinear path is the tell-tale of scripted movement."""
    path = bezier_path((0, 0), (400, 0), steps=30)
    max_deviation = max(abs(y) for _, y in path)
    assert max_deviation > 1.0


def test_bezier_path_handles_zero_length():
    path = bezier_path((50, 50), (50, 50), steps=5)
    assert all(p == pytest.approx((50, 50), abs=1e-6) for p in path)


# ---------------- guardrails ----------------


def test_confirm_declines_without_a_tty(monkeypatch):
    """No human present means no consent - never proceed by default."""
    monkeypatch.setattr("sys.stdin", None)
    assert confirm("Submit this application?") is False


def test_confirm_respects_explicit_default(monkeypatch):
    monkeypatch.setattr("sys.stdin", None)
    assert confirm("Continue reading?", default=True) is True


def test_captcha_signatures_cover_the_major_vendors():
    joined = " ".join(CAPTCHA_SIGNATURES)
    for vendor in ("recaptcha", "hcaptcha", "turnstile"):
        assert vendor in joined


# ---------------- driver routing ----------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://boards.greenhouse.io/acme/jobs/123", GreenhouseDriver),
        ("https://job-boards.greenhouse.io/acme/jobs/9", GreenhouseDriver),
        ("https://jobs.lever.co/acme/abc-123", GreenhouseDriver),
        ("https://acme.wd1.myworkdayjobs.com/en-US/careers/job/x", WorkdayDriver),
        ("https://acme.myworkdaysite.com/en-US/careers", WorkdayDriver),
        ("https://careers.acme.com/apply/1", BaseATSDriver),
        ("", BaseATSDriver),
    ],
)
def test_driver_routing(url, expected):
    assert get_driver_class(url) is expected


def test_workday_is_checked_before_greenhouse():
    """A tenant URL containing both names must route to Workday."""
    assert get_driver_class("https://greenhouse.myworkdayjobs.com/x") is WorkdayDriver


# ---------------- live form scanning ----------------


def _browser_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            return Path(p.chromium.executable_path).exists()
    except Exception:
        return False


requires_browser = pytest.mark.skipif(
    not _browser_available(), reason="Chromium not installed (run: playwright install chromium)"
)


@pytest.fixture(scope="module")
def scanned_fields():
    """Run the real field scanner against the fixture form."""
    from playwright.sync_api import sync_playwright

    from automation.ats_drivers.base_driver import FIELD_SCAN_JS

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(FIXTURE.as_uri())
        raw = page.evaluate(FIELD_SCAN_JS)
        browser.close()
    return raw


@requires_browser
def test_scanner_resolves_labels_through_every_fallback(scanned_fields):
    labels = {f["name"]: f["label"] for f in scanned_fields}
    assert labels["job_application[first_name]"] == "First Name *"      # label[for]
    assert labels["email"] == "Email Address"                            # aria-label
    assert labels["phone"] == "Phone Number"                             # placeholder
    assert "LinkedIn" in labels["urls[LinkedIn]"]                        # wrapping label
    assert "legally authorized" in labels["work_auth"]                   # aria-labelledby
    assert labels["gender"] == "Gender"                                  # fieldset legend


@requires_browser
def test_scanner_skips_hidden_disabled_and_non_input_controls(scanned_fields):
    names = {f["name"] for f in scanned_fields}
    assert "csrf" not in names           # type=hidden
    assert "disabled_field" not in names  # disabled
    assert "hidden_field" not in names    # display:none
    assert all(f["field_type"] != "skip" for f in scanned_fields)


@requires_browser
def test_scanner_reads_select_options_and_drops_placeholders(scanned_fields):
    work_auth = next(f for f in scanned_fields if f["name"] == "work_auth")
    assert work_auth["field_type"] == "select"
    assert work_auth["options"] == ["Yes", "No"]        # "Please select" removed

    sponsorship = next(f for f in scanned_fields if f["name"] == "sponsorship")
    assert sponsorship["options"] == ["Yes", "No"]      # "-- Select --" removed


@requires_browser
def test_scanner_collapses_radio_groups_with_their_options(scanned_fields):
    genders = [f for f in scanned_fields if f["name"] == "gender"]
    assert len(genders) == 1, "radio group must be emitted once, not per button"
    assert "I decline to self identify" in genders[0]["options"]


@requires_browser
def test_scanner_detects_required_fields(scanned_fields):
    by_name = {f["name"]: f for f in scanned_fields}
    assert by_name["job_application[first_name]"]["required"] is True
    assert by_name["why"]["required"] is True
    assert by_name["salary"]["required"] is False


@requires_browser
def test_scanner_finds_the_file_input(scanned_fields):
    resume = next(f for f in scanned_fields if f["name"] == "resume")
    assert resume["field_type"] == "file"


@requires_browser
def test_end_to_end_mapping_of_a_real_form(scanned_fields, prof_fixture):
    """The full path: scan a real page -> map every field -> split escalations."""
    from engine.screener_mapper import FieldSpec, ScreenerMapper

    fields = [
        FieldSpec(
            label=f["label"],
            name=f["name"],
            field_type=FieldType(f["field_type"]),
            options=f["options"],
            required=f["required"],
            selector=f["selector"],
        )
        for f in scanned_fields
    ]
    mapper = ScreenerMapper(
        prof_fixture, {"Why do you want to work at Acme?": "I admire the robotics work."}
    )
    autofill, escalations = mapper.map_form(fields)

    answers = {a.question: a.value for a in autofill.values()}
    assert answers["First Name *"] == "Ada"
    assert answers["Email Address"] == "ada@example.com"
    assert "legally authorized" in " ".join(answers)
    assert [v for k, v in answers.items() if "legally authorized" in k] == ["Yes"]
    assert [v for k, v in answers.items() if "sponsorship" in k] == ["No"]
    assert [v for k, v in answers.items() if k == "Gender"] == ["I decline to self identify"]

    # The unanswerable required question must escalate rather than be invented.
    escalated = " ".join(e.question for e in escalations)
    assert "kernel scheduler" in escalated


# ---------------- browser lifecycle ----------------


def test_browser_close_is_safe_before_start():
    HumanBrowser().close()  # must not raise


# ---------------- iframe scoping ----------------


def test_dom_defaults_to_the_page_and_can_be_scoped():
    browser = HumanBrowser()
    browser.page = "PAGE"          # stand-ins; only the indirection is under test
    assert browser.dom == "PAGE"
    browser.use_frame("FRAME")
    assert browser.dom == "FRAME"
    browser.use_frame(None)
    assert browser.dom == "PAGE"


@requires_browser
def test_form_inside_an_iframe_is_scannable():
    """Greenhouse embedded on a careers site: the form lives in an iframe."""
    from playwright.sync_api import sync_playwright

    from automation.ats_drivers.base_driver import FIELD_SCAN_JS

    embedded = FIXTURE.parent / "embedded_form.html"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(embedded.as_uri())
        page.wait_for_selector("iframe#grnhse_iframe")

        # The top-level document has no form fields at all.
        assert page.evaluate(FIELD_SCAN_JS) == []

        frame = page.locator("iframe#grnhse_iframe").first.element_handle().content_frame()
        fields = frame.evaluate(FIELD_SCAN_JS)
        browser.close()

    names = {f["name"] for f in fields}
    assert "job_application[first_name]" in names
    assert "resume" in names
