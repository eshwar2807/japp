"""Deterministic mapping from application-form fields to answers.

Resolution order, highest authority first:

  1. RULE     - a regex rule pulls the answer straight from master_profile.
                Legal and identity answers ONLY ever come from here. The LLM is
                never allowed to invent a work-authorisation answer.
  2. LLM      - a screener answer produced during tailoring, for posting-specific
                questions the rule table cannot know about.
  3. UNMAPPED - nothing matched. The field is escalated to a human.

Every answer carries a confidence. Anything below
``MIN_AUTOFILL_CONFIDENCE`` is flagged ``needs_human`` and the driver stops
rather than guessing.
"""

from __future__ import annotations

import enum
import re
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process

from engine.schemas import MasterProfile

#: Below this, a driver must ask a human instead of typing something.
MIN_AUTOFILL_CONFIDENCE = 0.75

#: How close a resolved value must be to an available <option> to pick it.
OPTION_MATCH_THRESHOLD = 80


class AnswerSource(str, enum.Enum):
    RULE = "rule"          # master_profile, deterministic
    LLM = "llm"            # tailoring pass, posting-specific
    MEMORY = "memory"      # something you answered before, reused
    INFERRED = "inferred"  # derived from profile facts, with grounding
    UNMAPPED = "unmapped"  # escalate to a human


class FieldType(str, enum.Enum):
    TEXT = "text"
    TEXTAREA = "textarea"
    SELECT = "select"
    RADIO = "radio"
    CHECKBOX = "checkbox"
    FILE = "file"
    DATE = "date"
    UNKNOWN = "unknown"


class FieldSpec(BaseModel):
    """A form control discovered on the page."""

    model_config = ConfigDict(extra="ignore")

    label: str = ""
    name: str = ""
    field_type: FieldType = FieldType.TEXT
    options: list[str] = Field(default_factory=list)
    required: bool = False
    selector: str = ""

    @property
    def question(self) -> str:
        """Best available human-readable description of the field."""
        return (self.label or self.name or self.selector).strip()


class MappedAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str
    value: str = ""
    source: AnswerSource = AnswerSource.UNMAPPED
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""

    @property
    def needs_human(self) -> bool:
        return (
            self.source is AnswerSource.UNMAPPED
            or self.confidence < MIN_AUTOFILL_CONFIDENCE
            or not self.value
        )


# --------------------------------------------------------------------------
# Rule table
# --------------------------------------------------------------------------

Resolver = Callable[[MasterProfile], Any]

#: (name, pattern, resolver). Ordered most-specific first; the first hit wins.
#: Sponsorship rules MUST precede generic work-authorisation rules, since
#: "authorized to work without sponsorship" contains both phrasings.
RULES: list[tuple[str, str, Resolver]] = [
    # --- combined auth + sponsorship phrasing ---
    (
        "auth_without_sponsorship",
        r"(authori[sz]ed|eligible|legally).{0,40}(work|employ).{0,60}without.{0,20}sponsor",
        lambda p: _combined_auth(p),
    ),
    # --- sponsorship ---
    (
        "sponsorship",
        r"(require|need|request).{0,40}sponsor|sponsorship.{0,30}(now|future|require)|visa\s+sponsor",
        lambda p: p.legal.get("requires_sponsorship_now_or_future"),
    ),
    # --- work authorization ---
    (
        "work_authorization",
        r"(authori[sz]ed|eligible|legally\s+(able|entitled)|right)\s.{0,40}(work|employ)",
        lambda p: p.legal.get("work_authorization_us"),
    ),
    ("visa_status", r"visa\s*(status|type)|immigration\s+status|work\s+status", lambda p: p.legal.get("visa_status")),
    # --- clearance & checks ---
    ("clearance", r"security\s+clearance|clearance\s+level", lambda p: p.legal.get("security_clearance")),
    ("background_check", r"background\s+(check|screen)", lambda p: p.legal.get("willing_to_complete_background_check")),
    ("drug_test", r"drug\s+(test|screen)", lambda p: p.legal.get("willing_to_complete_drug_test")),
    ("age_18", r"(18\s+years|over\s+18|at\s+least\s+18|age\s+of\s+majority)", lambda p: p.legal.get("age_over_18")),
    (
        "previously_employed",
        r"(previously|ever\s+been|formerly).{0,30}(employ|work).{0,30}(here|us|company|organi)",
        lambda p: p.legal.get("previously_employed_here"),
    ),
    ("non_compete", r"non[-\s]?compete|restrictive\s+covenant", lambda p: p.legal.get("non_compete_restrictions")),
    # --- logistics ---
    ("notice_period", r"notice\s+period", lambda p: p.legal.get("notice_period")),
    (
        "start_date",
        r"(earliest|available|availability).{0,25}start|start\s+date|when\s+can\s+you\s+start",
        lambda p: p.legal.get("earliest_start_date"),
    ),
    (
        "salary",
        r"(salary|compensation|pay)\s*(expectation|requirement|desired|range)|desired\s+(salary|pay)|expected\s+(salary|compensation)",
        lambda p: p.legal.get("desired_salary"),
    ),
    ("relocate", r"(willing|open).{0,20}relocat|relocation", lambda p: _yes_no(p.contact.location.willing_to_relocate)),
    ("remote_pref", r"(remote|hybrid|on[-\s]?site)\s*(preference|work)", lambda p: p.contact.location.remote_preference),
    ("references", r"reference(s)?\s+(available|upon|on\s+request)", lambda p: p.legal.get("reference_available_on_request")),
    ("how_heard", r"how\s+did\s+you\s+(hear|find|learn)", lambda p: p.preferences.get("how_did_you_hear_about_us")),
    # --- voluntary disclosures ---
    ("hispanic", r"hispanic|latino", lambda p: p.voluntary_disclosures.get("hispanic_or_latino")),
    ("race", r"race|ethnicity|ethnic\s+group", lambda p: p.voluntary_disclosures.get("race_ethnicity")),
    ("gender", r"\bgender\b|\bsex\b", lambda p: p.voluntary_disclosures.get("gender")),
    ("veteran", r"veteran|military\s+service|protected\s+veteran", lambda p: p.voluntary_disclosures.get("veteran_status")),
    ("disability", r"disab(ility|led)", lambda p: p.voluntary_disclosures.get("disability_status")),
    # --- identity & contact (specific before generic) ---
    ("preferred_name", r"preferred\s+(name|first)", lambda p: p.contact.preferred_name or p.contact.first_name),
    ("first_name", r"\b(first|given)\s*name\b|\bfname\b", lambda p: p.contact.first_name),
    ("last_name", r"\b(last|family|sur)\s*name\b|\blname\b", lambda p: p.contact.last_name),
    ("full_name", r"\b(full|legal)?\s*name\b", lambda p: p.contact.full_name),
    ("email", r"e[-\s]?mail", lambda p: p.contact.email),
    ("phone", r"phone|mobile|telephone|cell", lambda p: p.contact.phone),
    # --- links ---
    ("linkedin", r"linked\s?in", lambda p: p.contact.links.get("linkedin")),
    ("github", r"git\s?hub", lambda p: p.contact.links.get("github")),
    (
        "portfolio",
        r"portfolio|personal\s+(site|website)|\bwebsite\b|\burl\b",
        lambda p: p.contact.links.get("portfolio") or p.contact.links.get("website"),
    ),
    # --- address ---
    ("postal_code", r"(zip|postal)\s*code|\bzip\b", lambda p: p.contact.location.postal_code),
    ("city", r"\bcity\b|\btown\b", lambda p: p.contact.location.city),
    ("state", r"\bstate\b|\bprovince\b|\bregion\b", lambda p: p.contact.location.state),
    ("country", r"\bcountry\b", lambda p: p.contact.location.country),
    ("location", r"\blocation\b|current\s+(location|address)|where.{0,20}(based|located)", lambda p: str(p.contact.location)),
    # --- current role ---
    ("current_company", r"current\s+(employer|company|organi)", lambda p: _current(p, "company")),
    ("current_title", r"current\s+(title|role|position|job)", lambda p: _current(p, "title")),
    ("years_experience",
     r"years\s+of\s+(relevant\s+)?experience|how\s+many\s+years|years?\s+of\s+experience",
     lambda p, q="": _years_experience(p, q)),
]

_COMPILED: list[tuple[str, re.Pattern[str], Resolver]] = [
    (name, re.compile(pattern, re.IGNORECASE), fn) for name, pattern, fn in RULES
]


def _yes_no(value: Any) -> str:
    return "Yes" if value else "No"


def _combined_auth(profile: MasterProfile) -> str | None:
    """'Authorised to work WITHOUT sponsorship?' needs both facts to answer.

    Yes only when authorised AND no sponsorship needed. If authorised but
    sponsorship IS required, the honest answer is No. Anything else is
    ambiguous and gets escalated.
    """
    authorized = str(profile.legal.get("work_authorization_us", "")).strip().lower()
    sponsorship = str(profile.legal.get("requires_sponsorship_now_or_future", "")).strip().lower()
    if authorized.startswith("y") and sponsorship.startswith("n"):
        return "Yes"
    if authorized.startswith("y") and sponsorship.startswith("y"):
        return "No"
    if authorized.startswith("n"):
        return "No"
    return None


def _current(profile: MasterProfile, attr: str) -> str | None:
    for exp in profile.experience:
        if exp.is_current:
            return getattr(exp, attr, None)
    return getattr(profile.experience[0], attr, None) if profile.experience else None


#: Words that keep a "years of experience" question generic. Anything else
#: after "years of" names a specific skill, and total career length is not the
#: answer to that.
_GENERIC_EXPERIENCE_WORDS = {
    "relevant", "professional", "work", "working", "total", "overall",
    "industry", "experience", "engineering", "software", "full", "time",
    "fulltime", "post", "graduate", "your", "of", "do", "you", "have", "the",
    "in", "a", "an", "and", "with", "how", "many", "years", "year",
}


def _asks_about_a_specific_skill(question: str) -> bool:
    """True when a years-of-experience question names a particular technology.

    "How many years of experience do you have?" is answerable from employment
    dates. "How many years of Go?" is not - answering it with total career
    length is simply a wrong number on a job application.
    """
    match = re.search(r"years?\s+(?:of\s+)?(.{0,40})", question, re.IGNORECASE)
    if not match:
        return False
    tail = re.sub(r"[^a-z0-9+#. ]+", " ", match.group(1).lower())
    return any(
        word and word not in _GENERIC_EXPERIENCE_WORDS
        for word in tail.split()
    )


def _years_experience(profile: MasterProfile, question: str = "") -> str | None:
    """Total professional years, derived from the earliest start date."""
    if question and _asks_about_a_specific_skill(question):
        return None
    starts = [e.start_date for e in profile.experience if e.start_date]
    if not starts:
        return None
    from datetime import date

    try:
        earliest = min(starts)
        year, _, month = earliest.partition("-")
        today = date.today()
        years = today.year - int(year) + (today.month - int(month or 1)) / 12
    except (ValueError, TypeError):
        return None
    return str(max(int(years), 0))


# --------------------------------------------------------------------------
# Mapper
# --------------------------------------------------------------------------


class ScreenerMapper:
    def __init__(
        self,
        profile: MasterProfile,
        screener_answers: dict[str, str] | None = None,
        remembered: dict[str, str] | None = None,
        resolver: Any | None = None,
    ) -> None:
        self.profile = profile
        self.screener_answers = screener_answers or {}
        #: Answers the user gave to earlier applications, reused verbatim.
        self.remembered = remembered or {}
        #: Optional LLM fallback, tried once per form before escalating.
        self.resolver = resolver

    # ---------------- resolution ----------------

    def _from_rules(self, question: str) -> tuple[str, str] | None:
        """Return (value, rule_name) for the first matching rule."""
        for name, pattern, resolver in _COMPILED:
            if not pattern.search(question):
                continue
            try:
                # A few resolvers need the question text to tell a generic
                # question from one about a specific technology.
                try:
                    value = resolver(self.profile, question)
                except TypeError:
                    value = resolver(self.profile)
            except (AttributeError, KeyError, IndexError, TypeError):
                value = None
            if value is None:
                continue
            text = str(value).strip()
            # Unfilled template placeholders are not answers.
            if text and not text.startswith("<"):
                return text, name
        return None

    def _from_answers(
        self, question: str, pool: dict[str, str]
    ) -> tuple[str, float] | None:
        """Fuzzy-match a field label against a pool of known answers."""
        if not pool:
            return None
        match = process.extractOne(
            question, list(pool), scorer=fuzz.token_set_ratio, processor=default_process
        )
        if match and match[1] >= 75:
            return pool[match[0]], min(0.70 + (match[1] - 75) / 100, 0.9)
        return None

    def _from_llm(self, question: str) -> tuple[str, float] | None:
        return self._from_answers(question, self.screener_answers)

    def _from_memory(self, question: str) -> tuple[str, float] | None:
        """Something answered on a previous application.

        Ranked above the tailoring pass's own answers: a value the user typed
        themselves outranks one the model composed.
        """
        return self._from_answers(question, self.remembered)

    def map_field(self, field: FieldSpec) -> MappedAnswer:
        question = field.question
        if not question:
            return MappedAnswer(
                question="", source=AnswerSource.UNMAPPED, reason="Field has no label or name."
            )

        rule = self._from_rules(question)
        if rule:
            value, rule_name = rule
            answer = MappedAnswer(
                question=question, value=value, source=AnswerSource.RULE,
                confidence=1.0, reason=f"master_profile rule '{rule_name}'",
            )
        else:
            remembered = self._from_memory(question)
            llm = self._from_llm(question) if remembered is None else None
            if remembered:
                value, confidence = remembered
                answer = MappedAnswer(
                    question=question, value=value, source=AnswerSource.MEMORY,
                    confidence=confidence, reason="you answered this before",
                )
            elif llm:
                value, confidence = llm
                answer = MappedAnswer(
                    question=question, value=value, source=AnswerSource.LLM,
                    confidence=confidence, reason="tailored screener answer",
                )
            else:
                return MappedAnswer(
                    question=question,
                    source=AnswerSource.UNMAPPED,
                    reason="No profile rule, past answer or tailored answer matched.",
                )

        # Constrained controls: the value must correspond to a real option.
        if field.options:
            answer = self._match_option(answer, field)
        return answer

    def _match_option(self, answer: MappedAnswer, field: FieldSpec) -> MappedAnswer:
        """Snap a free-text answer onto one of the control's actual options."""
        options = field.options
        exact = next((o for o in options if o.strip().lower() == answer.value.strip().lower()), None)
        if exact:
            return answer.model_copy(update={"value": exact})

        match = process.extractOne(
            answer.value, options, scorer=fuzz.token_set_ratio, processor=default_process
        )
        if match and match[1] >= OPTION_MATCH_THRESHOLD:
            return answer.model_copy(
                update={
                    "value": match[0],
                    "confidence": answer.confidence * (match[1] / 100),
                    "reason": f"{answer.reason} -> option '{match[0]}' ({match[1]:.0f}% match)",
                }
            )

        return answer.model_copy(
            update={
                "value": "",
                "source": AnswerSource.UNMAPPED,
                "confidence": 0.0,
                "reason": (
                    f"Answer {answer.value!r} does not correspond to any available option "
                    f"({', '.join(options[:6])}{'...' if len(options) > 6 else ''})."
                ),
            }
        )

    # ---------------- batch ----------------

    def map_form(self, fields: list[FieldSpec]) -> tuple[dict[str, MappedAnswer], list[MappedAnswer]]:
        """Map a whole form.

        Returns ``(autofillable, escalations)``. Anything the deterministic
        steps cannot answer is offered to the resolver in a single batched call
        before being escalated, so a human is asked only about fields that
        genuinely have no honest answer available.
        """
        autofill: dict[str, MappedAnswer] = {}
        escalations: list[MappedAnswer] = []
        pending: list[tuple[FieldSpec, MappedAnswer]] = []

        for field in fields:
            if field.field_type is FieldType.FILE:
                continue  # handled by the resume-upload step
            answer = self.map_field(field)
            if answer.needs_human:
                if field.required or answer.source is not AnswerSource.UNMAPPED:
                    pending.append((field, answer))
                continue
            autofill[field.selector or field.name or field.label] = answer

        for field, answer in self._try_resolver(pending):
            if answer.needs_human:
                escalations.append(answer)
            else:
                autofill[field.selector or field.name or field.label] = answer

        return autofill, escalations

    def _try_resolver(
        self, pending: list[tuple[FieldSpec, MappedAnswer]]
    ) -> list[tuple[FieldSpec, MappedAnswer]]:
        """One batched attempt to answer what the deterministic steps missed."""
        if not pending or self.resolver is None:
            return pending

        questions = [
            {
                "question": field.question,
                "field_type": field.field_type.value,
                "options": field.options,
                "required": field.required,
            }
            for field, _ in pending
        ]
        known = {**self.remembered, **self.screener_answers}
        resolved = self.resolver.resolve(questions, self.profile, known)

        out: list[tuple[FieldSpec, MappedAnswer]] = []
        for field, answer in pending:
            candidate = resolved.get(field.question)
            if candidate is None or not candidate.usable:
                out.append((field, answer))
                continue

            inferred = MappedAnswer(
                question=field.question,
                value=candidate.answer,
                source=AnswerSource.INFERRED,
                confidence=candidate.confidence,
                reason=f"derived from your profile: {candidate.grounding[:160]}",
            )
            if field.options:
                inferred = self._match_option(inferred, field)
            out.append((field, inferred))
        return out
