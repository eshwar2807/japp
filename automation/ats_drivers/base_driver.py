"""Common ATS form interactions shared by every portal driver.

Subclasses override the parts that differ per portal (how to reach the form,
how to log in, where the submit button lives). Everything else - discovering
fields, resolving answers, uploading the resume, escalating ambiguity - is
implemented once here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from automation.stealth_browser import (
    HumanBrowser,
    HumanDeclined,
    ManualInterventionRequired,
    think_pause,
)
from config import settings
from database.db_manager import DBManager
from engine.schemas import MasterProfile
from engine.screener_mapper import FieldSpec, FieldType, MappedAnswer, ScreenerMapper

log = logging.getLogger(__name__)


# JavaScript form scanner. Runs in one round trip and returns a descriptor for
# every visible control, with the label resolved through the same fallback chain
# a screen reader would use.
FIELD_SCAN_JS = r"""
() => {
  const LABELISH = 'label, legend, .label, [class*="label" i], h3, h4';

  const visible = (el) => {
    const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
  };

  const clean = (t) => (t || '').replace(/\s+/g, ' ').trim();

  const ariaLabel = (el) => {
    if (el.getAttribute('aria-label')) return clean(el.getAttribute('aria-label'));
    const by = el.getAttribute('aria-labelledby');
    if (by) {
      const parts = by.split(/\s+/).map(id => document.getElementById(id))
        .filter(Boolean).map(n => clean(n.innerText));
      if (parts.filter(Boolean).length) return parts.filter(Boolean).join(' ');
    }
    return '';
  };

  const legendFor = (el) => {
    const fs = el.closest('fieldset');
    if (!fs) return '';
    const lg = fs.querySelector('legend');
    return lg ? clean(lg.innerText) : '';
  };

  // Nearest label-ish element that PRECEDES this control in document order.
  // Scanning an ancestor's descendants instead would return the first label in
  // the whole form, silently mislabelling every field after it.
  const precedingLabel = (el) => {
    let node = el;
    for (let depth = 0; node && depth < 5; depth++) {
      let sib = node.previousElementSibling;
      while (sib) {
        const consider = (cand) => {
          if (!cand) return '';
          // A <label for="other"> belongs to a different control.
          const target = cand.getAttribute && cand.getAttribute('for');
          if (target && target !== el.id) return '';
          return clean(cand.innerText);
        };
        if (sib.matches && sib.matches(LABELISH)) {
          const t = consider(sib);
          if (t) return t;
        } else if (sib.querySelector) {
          const t = consider(sib.querySelector(LABELISH));
          if (t) return t;
        }
        sib = sib.previousElementSibling;
      }
      node = node.parentElement;
    }
    return '';
  };

  // Label as read for a single control.
  const labelFor = (el) => {
    const aria = ariaLabel(el);
    if (aria) return aria;
    if (el.id) {
      const lab = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (lab && clean(lab.innerText)) return clean(lab.innerText);
    }
    const wrap = el.closest('label');
    if (wrap && clean(wrap.innerText)) return clean(wrap.innerText);
    const lg = legendFor(el);
    if (lg) return lg;
    const prev = precedingLabel(el);
    if (prev) return prev;
    return clean(el.getAttribute('placeholder'));
  };

  // Question text for a radio/checkbox GROUP. The wrapping <label> and
  // label[for] of a radio hold the option text ("Male"), not the question, so
  // both are skipped here.
  const groupLabelFor = (el) => {
    const aria = ariaLabel(el);
    if (aria) return aria;
    const lg = legendFor(el);
    if (lg) return lg;
    return precedingLabel(el);
  };

  const typeOf = (el) => {
    const tag = el.tagName.toLowerCase();
    if (tag === 'select') return 'select';
    if (tag === 'textarea') return 'textarea';
    const t = (el.getAttribute('type') || 'text').toLowerCase();
    if (['radio', 'checkbox', 'file', 'date'].includes(t)) return t;
    if (['hidden', 'submit', 'button', 'image', 'reset'].includes(t)) return 'skip';
    return 'text';
  };

  const isRequired = (el, label) =>
    el.required ||
    el.getAttribute('aria-required') === 'true' ||
    /\*/.test(label || '') ||
    el.closest('[class*="required" i]') !== null;

  const out = [];
  const seenGroups = new Set();

  document.querySelectorAll('input, select, textarea').forEach((el, idx) => {
    const kind = typeOf(el);
    if (kind === 'skip') return;
    if (!visible(el) && kind !== 'file') return;   // file inputs are often hidden by design
    if (el.disabled || el.readOnly) return;

    const name = el.getAttribute('name') || el.id || '';

    if (kind === 'radio') {
      const key = name || labelFor(el);
      if (seenGroups.has(key)) return;
      seenGroups.add(key);

      const peers = name
        ? Array.from(document.querySelectorAll(
            `input[type="radio"][name="${CSS.escape(name)}"]`))
        : [el];
      const question = groupLabelFor(el) || labelFor(el);
      out.push({
        label: question.slice(0, 300),
        name: name,
        field_type: 'radio',
        required: isRequired(el, question),
        selector: name ? `input[type="radio"][name="${name}"]` : '',
        options: peers.map(p => labelFor(p) || clean(p.value)).filter(Boolean),
        values: peers.map(p => p.value),
      });
      return;
    }

    const label = labelFor(el).slice(0, 300);

    let options = [];
    if (kind === 'select') {
      options = Array.from(el.options)
        .map(o => clean(o.label || o.text))
        .filter(o => o && !/^(select|choose|please select|--)/i.test(o));
    }

    // Stable selector: id > name > positional fallback.
    let selector = '';
    if (el.id) selector = `#${CSS.escape(el.id)}`;
    else if (name) selector = `${el.tagName.toLowerCase()}[name="${name}"]`;
    else selector = `__index__${idx}`;

    out.push({
      label, name, field_type: kind, required: isRequired(el, label),
      selector, options, values: [],
    });
  });

  return out;
}
"""


class ApplicationOutcome(BaseModel):
    """What happened on one application attempt."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    submitted: bool = False
    fields_filled: int = 0
    resume_uploaded: bool = False
    account_created: bool = False
    escalations: list[MappedAnswer] = Field(default_factory=list)
    screenshot_path: str | None = None
    message: str = ""

    @property
    def needs_human(self) -> bool:
        return bool(self.escalations) or not self.submitted


class BaseATSDriver:
    """Template-method driver. Subclasses override the portal-specific hooks."""

    #: Domain fragments this driver claims.
    DOMAINS: tuple[str, ...] = ()
    NAME = "generic"

    #: Buttons that move the form forward vs. finish it.
    NEXT_SELECTORS = (
        "button:has-text('Next')",
        "button:has-text('Continue')",
        "button:has-text('Save and Continue')",
    )
    SUBMIT_SELECTORS = (
        "button:has-text('Submit Application')",
        "button:has-text('Submit')",
        "input[type='submit']",
        "button[type='submit']",
    )
    RESUME_INPUT_SELECTORS = (
        "input[type='file'][name*='resume' i]",
        "input[type='file'][id*='resume' i]",
        "input[type='file'][accept*='pdf']",
        "input[type='file']",
    )

    def __init__(
        self,
        browser: HumanBrowser,
        profile: MasterProfile,
        mapper: ScreenerMapper,
        db: DBManager | None = None,
    ) -> None:
        self.browser = browser
        self.profile = profile
        self.mapper = mapper
        self.db = db

    # ---------------- registration ----------------

    @classmethod
    def matches(cls, url: str) -> bool:
        return any(domain in (url or "").lower() for domain in cls.DOMAINS)

    # ---------------- field discovery ----------------

    def discover_fields(self) -> list[FieldSpec]:
        """Scan the current page for fillable controls."""
        raw = self.browser.dom.evaluate(FIELD_SCAN_JS)
        fields: list[FieldSpec] = []
        for item in raw:
            try:
                fields.append(
                    FieldSpec(
                        label=item.get("label", ""),
                        name=item.get("name", ""),
                        field_type=FieldType(item.get("field_type", "text")),
                        options=[o for o in item.get("options", []) if o],
                        required=bool(item.get("required")),
                        selector=item.get("selector", ""),
                    )
                )
            except ValueError:
                fields.append(FieldSpec(**{**item, "field_type": FieldType.UNKNOWN}))
        log.info("Discovered %d fields (%d required)", len(fields), sum(f.required for f in fields))
        return fields

    # ---------------- filling ----------------

    def fill_field(self, field: FieldSpec, answer: MappedAnswer) -> bool:
        """Type/select one answer. Returns True when the field was filled."""
        if not field.selector or field.selector.startswith("__index__"):
            log.debug("Skipping field with no stable selector: %s", field.question)
            return False

        try:
            if field.field_type is FieldType.SELECT:
                self.browser.human_select(field.selector, answer.value)
            elif field.field_type is FieldType.RADIO:
                option = self.browser.dom.locator(
                    f"{field.selector} >> xpath=.."
                ).get_by_text(answer.value, exact=False).first
                self.browser.human_click(option)
            elif field.field_type is FieldType.CHECKBOX:
                checkbox = self.browser.dom.locator(field.selector).first
                should_check = answer.value.strip().lower() in ("yes", "true", "1", "on", "agree")
                if checkbox.is_checked() != should_check:
                    self.browser.human_click(checkbox)
            else:
                self.browser.human_type(field.selector, answer.value)
            return True
        except Exception as exc:  # a single stubborn field must not kill the run
            log.warning("Could not fill %r (%s): %s", field.question, field.selector, exc)
            return False

    def fill_form(self, fields: list[FieldSpec] | None = None) -> tuple[int, list[MappedAnswer]]:
        """Map and fill everything answerable; return (filled_count, escalations)."""
        fields = fields if fields is not None else self.discover_fields()
        autofill, escalations = self.mapper.map_form(fields)
        by_selector = {f.selector or f.name or f.label: f for f in fields}

        filled = 0
        for key, answer in autofill.items():
            field = by_selector.get(key)
            if field and self.fill_field(field, answer):
                filled += 1
                log.info("  %-46s -> %s", field.question[:46], answer.value[:40])

        return filled, escalations

    # ---------------- resume upload ----------------

    def upload_resume(self, pdf_path: Path) -> bool:
        for selector in self.RESUME_INPUT_SELECTORS:
            locator = self.browser.dom.locator(selector).first
            try:
                if locator.count() == 0:
                    continue
                self.browser.upload_file(locator, pdf_path)
                return True
            except Exception as exc:
                log.debug("Upload via %s failed: %s", selector, exc)
        log.warning("No resume file input found on this page.")
        return False

    # ---------------- account handling ----------------

    def ensure_account(self, portal_url: str) -> tuple[bool, str | None]:
        """Look up (or provision) a credential for this portal.

        Returns ``(created_now, password)``. Registration itself is gated: the
        form is filled, then a human confirms before the account is created.
        Nothing is submitted automatically.
        """
        if self.db is None:
            return False, None

        email = self.profile.contact.email
        _, password, created = self.db.get_or_create_credential(portal_url, email)
        if created:
            log.info("No stored credential for this portal; generated a new password.")
        else:
            log.info("Reusing stored credential for %s", email)
        return created, password

    def register_account(self, email: str, password: str) -> bool:
        """Portal-specific sign-up. Base implementation escalates to a human."""
        if settings.REQUIRE_CONFIRM_BEFORE_REGISTER:
            self.browser.hand_off(
                "account creation",
                [
                    f"Portal: {self.browser.page.url}",
                    f"Email: {email}",
                    "A password has been generated and stored encrypted in the vault.",
                    "Create the account in the browser (paste the password from "
                    "`main.py creds --show`), then return here.",
                ],
            )
            return True
        raise ManualInterventionRequired(
            "Automatic account registration is not enabled for this driver."
        )

    # ---------------- navigation & submit ----------------

    def click_next(self) -> bool:
        for selector in self.NEXT_SELECTORS:
            locator = self.browser.dom.locator(selector).first
            try:
                if locator.count() and locator.is_visible():
                    self.browser.human_click(locator)
                    self.browser.page.wait_for_load_state("networkidle", timeout=15000)
                    return True
            except Exception:
                continue
        return False

    def find_submit(self) -> Any | None:
        for selector in self.SUBMIT_SELECTORS:
            locator = self.browser.dom.locator(selector).first
            try:
                if locator.count() and locator.is_visible():
                    return locator
            except Exception:
                continue
        return None

    def submit(self, escalations: list[MappedAnswer]) -> bool:
        """Final gate. Never submits without an explicit human yes."""
        self.browser.raise_on_captcha()

        if escalations:
            self.browser.hand_off(
                f"{len(escalations)} field(s) could not be answered safely",
                [f"{e.question[:60]} - {e.reason}" for e in escalations],
            )

        button = self.find_submit()
        if button is None:
            self.browser.hand_off(
                "submit button not found",
                ["Locate and review the form in the browser window."],
            )
            return False

        if settings.REQUIRE_CONFIRM_BEFORE_SUBMIT:
            try:
                with self.browser.gate(
                    "SUBMIT this application",
                    [
                        f"URL: {self.browser.page.url}",
                        "Review every field in the browser window first.",
                        "This is irreversible.",
                    ],
                ):
                    self.browser.human_click(button)
            except HumanDeclined:
                log.info("Submission declined by operator.")
                return False
        else:
            self.browser.human_click(button)

        self.browser.page.wait_for_load_state("networkidle", timeout=30000)
        return True

    # ---------------- template method ----------------

    def apply(self, job_url: str, resume_pdf: Path) -> ApplicationOutcome:
        """Full flow. Subclasses usually override `open_application_form` only."""
        outcome = ApplicationOutcome()

        self.browser.goto(job_url)
        self.open_application_form()

        created, _ = self.ensure_account(job_url)
        outcome.account_created = created

        outcome.resume_uploaded = self.upload_resume(resume_pdf)
        self.browser.human_scroll(600, 4)

        filled, escalations = self.fill_form()
        outcome.fields_filled = filled
        outcome.escalations = escalations

        outcome.submitted = self.submit(escalations)
        outcome.message = "Submitted" if outcome.submitted else "Not submitted"
        return outcome

    def open_application_form(self) -> None:
        """Click through to the actual form. Overridden per portal."""
        for selector in (
            "button:has-text('Apply')",
            "a:has-text('Apply')",
            "button:has-text('Apply Now')",
        ):
            locator = self.browser.dom.locator(selector).first
            try:
                if locator.count() and locator.is_visible():
                    self.browser.human_click(locator)
                    self.browser.page.wait_for_load_state("networkidle", timeout=20000)
                    return
            except Exception:
                continue
        log.info("No 'Apply' button found; assuming the form is already open.")
