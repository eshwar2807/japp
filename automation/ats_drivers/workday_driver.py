"""Workday: multi-step wizard behind an account.

Workday tenants are all *.myworkdayjobs.com and share `data-automation-id`
attributes, which are far more stable than class names. The flow is:
sign in (or register), then page through My Information -> My Experience ->
Application Questions -> Voluntary Disclosures -> Review -> Submit.

Account creation and final submission are both human-gated.
"""

from __future__ import annotations

import logging

from automation.ats_drivers.base_driver import ApplicationOutcome, BaseATSDriver
from automation.stealth_browser import ManualInterventionRequired
from config import settings

log = logging.getLogger(__name__)


class WorkdayDriver(BaseATSDriver):
    NAME = "workday"
    DOMAINS = ("myworkdayjobs.com", "myworkdaysite.com", "wd1.", "wd3.", "wd5.")

    #: Workday's automation ids are the reliable hooks.
    APPLY_SELECTORS = (
        "a[data-automation-id='adventureButton']",
        "button[data-automation-id='adventureButton']",
        "a:has-text('Apply')",
    )
    AUTOFILL_WITH_RESUME = "a[data-automation-id='autofillWithResume']"
    APPLY_MANUALLY = "a[data-automation-id='applyManually']"

    SIGN_IN_LINK = "button[data-automation-id='signInLink'], a:has-text('Sign In')"
    CREATE_ACCOUNT_LINK = (
        "button[data-automation-id='createAccountLink'], a:has-text('Create Account')"
    )
    EMAIL_INPUT = "input[data-automation-id='email']"
    PASSWORD_INPUT = "input[data-automation-id='password']"
    VERIFY_PASSWORD_INPUT = "input[data-automation-id='verifyPassword']"
    SIGN_IN_SUBMIT = "button[data-automation-id='signInSubmitButton']"
    CREATE_ACCOUNT_SUBMIT = "button[data-automation-id='createAccountSubmitButton']"

    RESUME_INPUT_SELECTORS = (
        "input[data-automation-id='file-upload-input-ref']",
        "input[type='file']",
    )
    NEXT_SELECTORS = (
        "button[data-automation-id='bottom-navigation-next-button']",
        "button:has-text('Save and Continue')",
        "button:has-text('Continue')",
        "button:has-text('Next')",
    )
    SUBMIT_SELECTORS = (
        "button[data-automation-id='bottom-navigation-next-button']:has-text('Submit')",
        "button:has-text('Submit')",
    )

    MAX_WIZARD_STEPS = 8

    # ---------------- auth ----------------

    def _is_signed_in(self) -> bool:
        page = self.browser.page
        return bool(
            page.locator("button[data-automation-id='utilityButtonAccount']").count()
            or page.locator("[data-automation-id='applicationSection']").count()
        )

    def sign_in_or_register(self, job_url: str) -> bool:
        """Returns True when a new account was created during this run."""
        if self._is_signed_in():
            log.info("Already signed in on this Workday tenant.")
            return False
        if self.db is None:
            raise ManualInterventionRequired("No credential vault available for Workday sign-in.")

        page = self.browser.page
        email = self.profile.contact.email
        _, password, created = self.db.get_or_create_credential(job_url, email)

        if created:
            # New account: fill the registration form, then hand over for the
            # actual account creation. We never create the account unattended.
            for selector in (self.CREATE_ACCOUNT_LINK,):
                locator = page.locator(selector).first
                if locator.count() and locator.is_visible():
                    self.browser.human_click(locator)
                    break

            if page.locator(self.EMAIL_INPUT).count():
                self.browser.human_type(self.EMAIL_INPUT, email)
            if page.locator(self.PASSWORD_INPUT).count():
                self.browser.human_type(self.PASSWORD_INPUT, password)
            if page.locator(self.VERIFY_PASSWORD_INPUT).count():
                self.browser.human_type(self.VERIFY_PASSWORD_INPUT, password)

            self.browser.raise_on_captcha()
            self.browser.hand_off(
                "create this Workday account",
                [
                    f"Tenant: {page.url}",
                    f"Email: {email}",
                    "The generated password is filled in and stored encrypted.",
                    "Accept the tenant's terms yourself, then click Create Account.",
                ],
            )
            return True

        # Existing credential: sign in.
        locator = page.locator(self.SIGN_IN_LINK).first
        if locator.count() and locator.is_visible():
            self.browser.human_click(locator)
        if page.locator(self.EMAIL_INPUT).count():
            self.browser.human_type(self.EMAIL_INPUT, email)
            self.browser.human_type(self.PASSWORD_INPUT, password)
            self.browser.raise_on_captcha()
            submit = page.locator(self.SIGN_IN_SUBMIT).first
            if submit.count():
                self.browser.human_click(submit)
                page.wait_for_load_state("networkidle", timeout=30000)
        return False

    # ---------------- flow ----------------

    def open_application_form(self) -> None:
        page = self.browser.page
        for selector in self.APPLY_SELECTORS:
            locator = page.locator(selector).first
            try:
                if locator.count() and locator.is_visible():
                    self.browser.human_click(locator)
                    page.wait_for_load_state("networkidle", timeout=25000)
                    break
            except Exception:
                continue

        # Workday offers "Autofill with Resume" vs "Apply Manually".
        # Autofill parses the PDF, which both saves typing and confirms the
        # resume is machine-readable - exactly what we want.
        autofill = page.locator(self.AUTOFILL_WITH_RESUME).first
        if autofill.count() and autofill.is_visible():
            self.browser.human_click(autofill)
            page.wait_for_load_state("networkidle", timeout=25000)

    def apply(self, job_url, resume_pdf) -> ApplicationOutcome:
        outcome = ApplicationOutcome()

        self.browser.goto(job_url)
        outcome.account_created = self.sign_in_or_register(job_url)
        self.open_application_form()

        outcome.resume_uploaded = self.upload_resume(resume_pdf)
        self.browser.page.wait_for_timeout(3000)  # let Workday parse the PDF

        all_escalations = []
        total_filled = 0

        for step in range(1, self.MAX_WIZARD_STEPS + 1):
            self.browser.raise_on_captcha()
            filled, escalations = self.fill_form()
            total_filled += filled
            all_escalations.extend(escalations)
            log.info("Wizard step %d: filled %d field(s)", step, filled)

            if escalations:
                self.browser.hand_off(
                    f"step {step}: {len(escalations)} field(s) need you",
                    [f"{e.question[:60]} - {e.reason}" for e in escalations],
                )

            # A "Submit" on the nav button means we have reached Review.
            nav = self.browser.page.locator(
                "button[data-automation-id='bottom-navigation-next-button']"
            ).first
            if nav.count() and "submit" in (nav.inner_text() or "").strip().lower():
                break
            if not self.click_next():
                log.info("No further wizard step found; stopping at step %d.", step)
                break

        outcome.fields_filled = total_filled
        outcome.escalations = all_escalations
        outcome.submitted = self.submit(all_escalations)
        outcome.message = "Submitted" if outcome.submitted else "Not submitted"
        return outcome
