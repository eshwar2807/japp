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
from engine.screener_mapper import (
    AnswerSource,
    FieldSpec,
    FieldType,
    MappedAnswer,
    ScreenerMapper,
)

log = logging.getLogger(__name__)

#: What an ATS says once it has actually taken the application. Clicking the
#: button proves nothing on its own, so one of these has to appear before a run
#: is reported as submitted.
CONFIRMATION_MARKERS = (
    "thank you for applying",
    "thanks for applying",
    "thank you for your application",
    "thank you for your interest",
    "application received",
    "application submitted",
    "application complete",
    "we have received your application",
    "we've received your application",
    "your application has been submitted",
    "successfully submitted",
    "submission received",
)


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
    // A search box is never an application field, wherever it sits.
    if (['hidden', 'submit', 'button', 'image', 'reset', 'search'].includes(t)) return 'skip';
    return 'text';
  };

  const isRequired = (el, label) =>
    el.required ||
    el.getAttribute('aria-required') === 'true' ||
    /\*/.test(label || '') ||
    el.closest('[class*="required" i]') !== null;

  // Site chrome. A careers page wraps the posting in a header search box, a
  // footer "check out other roles" widget and a cookie bar, all of which hold
  // real text inputs. Scanning them alongside the application is how a city
  // ended up typed into a job-search field instead of the form.
  const CHROME = 'header,footer,nav,[role="search"],[role="banner"],' +
                 '[role="contentinfo"],[id*="cookie" i],[class*="cookie" i],' +
                 '[id*="search" i],[class*="search" i]';
  const inChrome = (el) => el.closest(CHROME) !== null;

  // The application form, not the page. Pick the form holding the most
  // fillable controls; fall back to the document when a page uses no <form>
  // at all, which some single-page ATS front-ends do.
  const scope = (() => {
    let best = null, most = 0;
    for (const f of document.querySelectorAll('form')) {
      if (inChrome(f)) continue;
      const n = Array.from(f.querySelectorAll('input, select, textarea'))
        .filter(e => typeOf(e) !== 'skip' && !e.disabled && !e.readOnly).length;
      if (n > most) { best = f; most = n; }
    }
    return most >= 2 ? best : document;
  })();

  // Hints that sit alongside a group's question without being it.
  const GROUP_HINT = /^(select all that apply|choose all that apply|check all that apply|optional|required|\*)\.?$/i;

  //: The question a group of controls is asking, as opposed to the text of any
  //: one option. A <legend> when there is one; otherwise the fieldset's own
  //: text with the option labels taken out, which is how Ashby writes it.
  const groupQuestion = (fs, optionLabels) => {
    const lg = fs.querySelector('legend');
    if (lg && clean(lg.innerText)) return clean(lg.innerText);
    const aria = ariaLabel(fs);
    if (aria) return aria;
    let text = fs.innerText || '';
    for (const option of optionLabels) text = text.split(option).join('\n');
    const lines = text.split('\n').map(s => s.trim())
      .filter(s => s && !GROUP_HINT.test(s));
    // Plaid writes the hint on the same line as the question:
    // "Why are you interested in working at Plaid? Select all that apply."
    const TRAILING_HINT = /\s*(select|choose|check)\s+all\s+that\s+apply\.?\s*$/i;
    return clean((lines[0] || precedingLabel(fs) || '').replace(TRAILING_HINT, ''));
  };

  const out = [];
  const seenGroups = new Set();
  let groupSeq = 0;

  scope.querySelectorAll('input, select, textarea').forEach((el, idx) => {
    const kind = typeOf(el);
    if (kind === 'skip') return;
    if (scope === document && inChrome(el)) return;
    if (!visible(el) && kind !== 'file') return;   // file inputs are often hidden by design
    if (el.disabled || el.readOnly) return;

    const name = el.getAttribute('name') || el.id || '';

    // A fieldset holding several checkboxes is one multi-select question, not
    // one field per box. Ashby names each box after its own option, so the
    // name-based grouping that works for radios leaves three fields labelled
    // "San Francisco HQ", "New York City Office", "Seattle Office" - each with
    // no options recorded. A location rule then matched the word "City" and
    // typed the candidate's home city into a checkbox. Grouped, the question
    // reads "Preferred Work Location" and carries its options, so an answer
    // that is not one of them is escalated instead of forced in.
    if (kind === 'checkbox') {
      const fs = el.closest('fieldset');
      const boxes = fs
        ? Array.from(fs.querySelectorAll('input[type="checkbox"]'))
            .filter(b => visible(b) && !b.disabled && !b.readOnly)
        : [];
      if (fs && boxes.length > 1) {
        if (seenGroups.has(fs)) return;
        seenGroups.add(fs);
        if (!fs.dataset.jpGroup) fs.dataset.jpGroup = String(++groupSeq);

        const optionLabels = boxes.map(b => labelFor(b) || clean(b.value)).filter(Boolean);
        const question = groupQuestion(fs, optionLabels);
        out.push({
          label: question.slice(0, 300),
          name: question,
          field_type: 'radio',       // one question, answered from its options
          required: isRequired(el, question),
          selector: `fieldset[data-jp-group="${fs.dataset.jpGroup}"] input[type="checkbox"]`,
          options: optionLabels,
          values: boxes.map(b => b.value),
        });
        return;
      }
      // A lone checkbox - "I agree" - keeps the ordinary treatment below.
    }

    if (kind === 'radio') {
      const key = name || labelFor(el);
      if (seenGroups.has(key)) return;
      seenGroups.add(key);

      const peers = name
        ? Array.from(scope.querySelectorAll(
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
    #: What the page said afterwards. Empty means the click landed but
    #: nothing confirmed it, which is not the same as success.
    confirmation: str = ""
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

        if escalations:
            filled += self.ask_for_unanswered(escalations, by_selector)
            escalations = [e for e in escalations if not e.value]
        return filled, escalations

    def ask_for_unanswered(
        self, escalations: list[MappedAnswer], by_selector: dict[str, FieldSpec]
    ) -> int:
        """Put each unanswered field to the supervisor as an actual question.

        Previously these were rolled into one approve-or-decline gate reading
        "1 field(s) could not be answered safely", which told the user a
        question existed without telling them what it was or letting them
        answer it. The gatekeeper can carry a real question, its options and
        the reason, so it does - and a value that comes back is filled in and
        remembered for the next application.
        """
        filled = 0
        for answer in escalations:
            field = by_selector.get(answer.question) or next(
                (f for f in by_selector.values() if f.question == answer.question), None
            )
            reply = self.browser.gatekeeper.ask(
                answer.question,
                reason=answer.reason,
                kind="unmapped_field",
                options=list(field.options) if field and field.options else None,
                required=bool(field and field.required),
            )
            if not reply:
                continue
            answered = answer.model_copy(update={
                "value": reply,
                "source": AnswerSource.HUMAN,
                "confidence": 1.0,
                "reason": "you answered this",
            })
            if field and self.fill_field(field, answered):
                filled += 1
                # Nothing is stored here on purpose: ask() records the
                # question as an action item, and answering one is what
                # answered_action_map replays into the next application.
                log.info("  %-46s -> %s (from you)",
                         field.question[:46], reply[:40])
        return filled

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

        before_url = self.browser.page.url
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
        self.confirmation = self.confirmation_evidence(before_url)
        return True

    def confirmation_evidence(self, before_url: str) -> str:
        """What the page says after submitting, or "" if it says nothing.

        Clicking a button is not evidence that an employer received anything.
        `submit` used to return True the moment the click landed and the page
        settled, so "Submitted" meant only "we clicked". This looks for what an
        ATS actually shows on success, and reports plainly when it finds none.
        """
        page = self.browser.page
        for _ in range(3):
            try:
                text = (page.inner_text("body", timeout=5000) or "").lower()
            except Exception:
                text = ""
            for marker in CONFIRMATION_MARKERS:
                if marker in text:
                    return marker
            # Some portals confirm by navigating rather than by wording.
            if page.url != before_url and any(
                token in page.url.lower()
                for token in ("confirm", "thank", "success", "submitted", "complete")
            ):
                return f"redirected to {page.url}"
            page.wait_for_timeout(2000)
        return ""

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

        self.confirmation = ""
        clicked = self.submit(escalations)
        outcome.confirmation = self.confirmation
        # Submitted means the employer said so. A click with nothing to show
        # for it is reported as exactly that, so a run is never recorded as a
        # sent application on the strength of a button press.
        outcome.submitted = bool(clicked and outcome.confirmation)
        if outcome.submitted:
            outcome.message = f"Submitted - confirmed by: {outcome.confirmation[:80]}"
        elif clicked:
            outcome.message = ("Clicked submit but the page showed no confirmation; "
                               "check this one yourself")
        else:
            outcome.message = "Not submitted"
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
