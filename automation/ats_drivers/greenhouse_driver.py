"""Greenhouse and Lever: single-page application forms, no account required.

Both render the whole application as one form, so the flow is: open the form,
attach the resume, fill everything, submit once. Lever differs mainly in
selectors, which is why it shares this driver.
"""

from __future__ import annotations

import logging

from automation.ats_drivers.base_driver import ApplicationOutcome, BaseATSDriver

log = logging.getLogger(__name__)


class GreenhouseDriver(BaseATSDriver):
    NAME = "greenhouse/lever"
    DOMAINS = (
        "greenhouse.io",
        "boards.greenhouse.io",
        "job-boards.greenhouse.io",
        "lever.co",
        "jobs.lever.co",
    )

    RESUME_INPUT_SELECTORS = (
        "input[type='file'][name='resume']",              # Lever
        "#resume_fieldset input[type='file']",            # Greenhouse classic
        "input[type='file'][id*='resume' i]",
        "input[type='file'][accept*='pdf']",
        "input[type='file']",
    )

    SUBMIT_SELECTORS = (
        "#submit_app",                                    # Greenhouse classic
        "button:has-text('Submit Application')",
        "button:has-text('Submit application')",
        "div.postings-btn-wrapper button",                # Lever
        "button[type='submit']",
        "input[type='submit']",
    )

    APPLY_SELECTORS = (
        "a:has-text('Apply for this job')",
        "a.postings-btn",                                 # Lever
        "button:has-text('Apply')",
        "a:has-text('Apply')",
    )

    def open_application_form(self) -> None:
        """Greenhouse embeds the form inline; Lever links to /apply."""
        page = self.browser.page

        # Already on the form?
        for marker in ("#application_form", "form.application-form", "#application-form"):
            if self.browser.dom.locator(marker).count():
                log.info("Application form already present.")
                return

        for selector in self.APPLY_SELECTORS:
            locator = page.locator(selector).first
            try:
                if locator.count() and locator.is_visible():
                    self.browser.human_click(locator)
                    page.wait_for_load_state("networkidle", timeout=20000)
                    break
            except Exception:
                continue

        # Greenhouse embedded on a company careers site renders the form in an
        # iframe. Scope locators to that frame; mouse and keyboard stay on the
        # page, which is what makes the human-paced clicking still work.
        iframe = page.locator("iframe#grnhse_iframe").first
        if iframe.count():
            handle = iframe.element_handle()
            frame = handle.content_frame() if handle else None
            if frame is not None:
                log.info("Greenhouse iframe detected; scoping locators to it.")
                self.browser.use_frame(frame)

        self.browser.human_scroll(400, 3)

    def apply(self, job_url, resume_pdf) -> ApplicationOutcome:
        """Greenhouse/Lever need no account, so skip the credential step."""
        outcome = ApplicationOutcome()

        self.browser.goto(job_url)
        self.open_application_form()

        outcome.resume_uploaded = self.upload_resume(resume_pdf)
        # Parsed-resume autofill takes a moment and overwrites fields; let it
        # finish before we type, or our values get clobbered.
        self.browser.page.wait_for_timeout(2500)
        self.browser.human_scroll(700, 4)

        filled, escalations = self.fill_form()
        outcome.fields_filled = filled
        outcome.escalations = escalations

        outcome.submitted = self.submit(escalations)
        outcome.message = "Submitted" if outcome.submitted else "Not submitted"
        return outcome
