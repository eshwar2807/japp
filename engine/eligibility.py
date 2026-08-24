"""Deterministic pre-screen: is this posting worth spending anything on?

Every check here runs on raw posting text with no LLM call, so an ineligible
posting costs nothing at all rather than an extraction call. The rules encode
hard constraints - things no amount of resume tailoring can change:

  * a role that is not a Java role
  * a seniority bar above the candidate's actual experience
  * a security clearance an H-1B holder cannot obtain
  * an employer who states they will not sponsor

The last two matter most. Clearance and sponsorship are not preferences; a
posting requiring US citizenship is closed regardless of fit, and applying
anyway wastes the candidate's time and the employer's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Eligibility:
    """Why a posting is or is not worth pursuing."""

    eligible: bool = True
    reasons: list[str] = field(default_factory=list)
    years_required: int | None = None
    clearance: str = ""

    def reject(self, reason: str) -> "Eligibility":
        self.eligible = False
        self.reasons.append(reason)
        return self


# --------------------------------------------------------------------------
# Java
# --------------------------------------------------------------------------

_JAVA = re.compile(
    r"\b(java|spring\s?boot|spring\s+framework|j2ee|jakarta\s?ee|jvm|hibernate|"
    r"micronaut|quarkus|jakarta|servlet|maven|gradle)\b",
    re.IGNORECASE,
)
#: "JavaScript" contains "Java" but is a different language entirely.
_JAVASCRIPT_ONLY = re.compile(r"\bjavascript\b|\bnode\.?js\b|\btypescript\b", re.IGNORECASE)


def is_java_role(title: str, description: str) -> bool:
    """Does this posting actually involve Java?

    Checked against the title first, then the description, with JavaScript
    stripped out - "JavaScript" contains the substring "Java" and would
    otherwise make every front-end role look like a match.
    """
    haystack = f"{title}\n{description}"
    without_js = _JAVASCRIPT_ONLY.sub(" ", haystack)
    return bool(_JAVA.search(without_js))


# --------------------------------------------------------------------------
# Experience
# --------------------------------------------------------------------------

_YEARS = re.compile(
    r"(\d{1,2})\s*(?:\+|plus)?\s*(?:-|–|to)?\s*(\d{1,2})?\s*\+?\s*years?"
    r"(?:\s+of)?(?:\s+\w+){0,3}?\s+experience",
    re.IGNORECASE,
)


def years_required(description: str) -> int | None:
    """The lowest number of years the posting asks for, if it says.

    Takes the minimum across all mentions: a posting saying "5+ years backend,
    8+ years leadership" is gated at five for an individual contributor, and
    treating it as eight would discard a viable role.
    """
    values = []
    for match in _YEARS.finditer(description or ""):
        low = int(match.group(1))
        if 0 < low <= 30:
            values.append(low)
    return min(values) if values else None


# --------------------------------------------------------------------------
# Clearance
# --------------------------------------------------------------------------

#: Clearances that require US citizenship, so an H-1B holder cannot obtain them.
_CITIZEN_ONLY_CLEARANCE = re.compile(
    r"\b(top\s?secret|ts/sci|ts-sci|\bsci\b|secret\s+clearance|security\s+clearance|"
    r"dod\s+clearance|q\s+clearance|l\s+clearance|polygraph|full\s+scope|"
    r"public\s+trust|suitability\s+clearance|active\s+clearance|"
    r"clearance\s+(?:is\s+)?required|must\s+(?:have|hold|possess)\s+.{0,30}clearance)\b",
    re.IGNORECASE,
)
#: Screenings anyone may undergo; not a clearance in the citizenship sense.
_ORDINARY_SCREENING = re.compile(
    r"\b(background\s+(?:check|screen)|drug\s+(?:test|screen)|credit\s+check|"
    r"fingerprint)\b",
    re.IGNORECASE,
)


def clearance_required(description: str) -> str:
    """The clearance a posting demands, or "" when it demands none.

    Ordinary background checks are not clearances: everyone goes through those,
    and treating them as disqualifying would reject most of the market.
    """
    text = description or ""
    match = _CITIZEN_ONLY_CLEARANCE.search(text)
    if not match:
        return ""
    phrase = match.group(0)
    if _ORDINARY_SCREENING.fullmatch(phrase.strip()):
        return ""
    return phrase.strip()


# --------------------------------------------------------------------------
# Citizenship and sponsorship
# --------------------------------------------------------------------------

_CITIZENSHIP_ONLY = re.compile(
    r"\b(u\.?s\.?\s+citizens?(?:hip)?\s+(?:is\s+)?(?:required|only)|"
    r"must\s+be\s+a\s+u\.?s\.?\s+citizen|citizenship\s+required|"
    r"us\s+citizens?\s+only|green\s+card\s+holders?\s+or\s+u\.?s\.?\s+citizens?\s+only|"
    r"(?:usc|gc)\s*(?:/|or)\s*(?:usc|gc)\s+only)\b",
    re.IGNORECASE,
)

_NO_SPONSORSHIP = re.compile(
    r"(not?\s+(?:able\s+to\s+)?(?:provide|offer|consider)\s+(?:visa\s+)?sponsorship|"
    r"(?:unable|will\s+not|cannot|can\s?not)\s+(?:to\s+)?sponsor|"
    r"no\s+(?:visa\s+)?sponsorship|without\s+(?:the\s+)?need\s+(?:for|of)\s+sponsorship|"
    r"do(?:es)?\s+not\s+sponsor|"
    r"authoriz\w+\s+to\s+work\s+.{0,40}without\s+sponsorship|"
    r"not\s+require\s+sponsorship)",
    re.IGNORECASE,
)


def citizenship_required(description: str) -> bool:
    return bool(_CITIZENSHIP_ONLY.search(description or ""))


def refuses_sponsorship(description: str) -> bool:
    """Does the employer state they will not sponsor?

    Relevant for anyone on a visa: an H-1B transfer is sponsorship, so a
    posting that rules it out is closed however well the resume matches.
    """
    return bool(_NO_SPONSORSHIP.search(description or ""))


# --------------------------------------------------------------------------
# Reserved postings
# --------------------------------------------------------------------------

#: Postings open only to a specific group. Common in India and Brazil, and
#: appearing in US diversity programmes: the role is genuinely restricted, so
#: applying wastes everyone's time.
_RESERVED_FOR = re.compile(
    r"("
    r"(?:exclusively|only|reserved|restricted|earmarked)\s+(?:open\s+)?"
    r"(?:for|to)\s+(?:persons?|people|candidates?|applicants?|individuals?)\s+"
    r"with\s+(?:a\s+)?disabilit\w+"
    r"|(?:persons?|people|candidates?|applicants?)\s+with\s+disabilit\w+\s+"
    r"(?:only|are\s+encouraged\s+to\s+apply\s+exclusively)"
    r"|\bpwd\s+(?:only|candidates?\s+only|position|vacancy|role)\b"
    r"|\bonly\s+pwd\b"
    r"|this\s+(?:position|role|vacancy|opening)\s+is\s+(?:reserved|exclusive|"
    r"restricted|only\s+open)\s+(?:for|to)\s+"
    r"(?:persons?|people|candidates?)?\s*with\s+disabilit\w+"
    r"|vaga\s+exclusiva\s+para\s+pcd"
    r"|posi..o\s+exclusiva\s+para\s+pessoas\s+com\s+defici.ncia"
    r")",
    re.IGNORECASE,
)

#: Equal-opportunity boilerplate. Appears on almost every posting and says the
#: opposite: the role is open to everyone. Matching this would reject the
#: entire market, so it is checked first and wins.
_EEO_BOILERPLATE = re.compile(
    r"(equal\s+opportunity|do\s+not\s+discriminate|without\s+regard\s+to|"
    r"regardless\s+of\s+.{0,60}disabilit|reasonable\s+accommodation|"
    r"encourage\s+(?:applications|candidates)\s+from\s+all|"
    r"affirmative\s+action|protected\s+veteran)",
    re.IGNORECASE,
)


def reserved_for_other_group(description: str) -> str:
    """Is this posting restricted to a group the candidate is not in?

    Deliberately narrow. Nearly every posting mentions disability in its equal
    opportunity statement, and treating that as a restriction would reject the
    whole market. Only explicit reservation language counts, and only when the
    surrounding sentence is not boilerplate.
    """
    text = description or ""
    match = _RESERVED_FOR.search(text)
    if not match:
        return ""

    # The sentence around the match decides it: "we do not discriminate on the
    # basis of disability" must never read as "reserved for candidates with
    # disabilities".
    start = max(match.start() - 160, 0)
    end = min(match.end() + 160, len(text))
    if _EEO_BOILERPLATE.search(text[start:end]):
        return ""
    return match.group(0).strip()[:80]


# --------------------------------------------------------------------------
# Combined
# --------------------------------------------------------------------------


def assess(
    title: str,
    description: str,
    *,
    require_java: bool = True,
    max_years: int | None = 6,
    needs_sponsorship: bool = True,
    can_obtain_clearance: bool = False,
) -> Eligibility:
    """Screen one posting against the candidate's hard constraints."""
    result = Eligibility()

    if require_java and not is_java_role(title, description):
        result.reject("not a Java role")

    years = years_required(description)
    result.years_required = years
    if max_years is not None and years is not None and years > max_years:
        result.reject(f"asks for {years}+ years, above the {max_years}-year limit")

    clearance = clearance_required(description)
    result.clearance = clearance
    if clearance and not can_obtain_clearance:
        result.reject(f"requires a clearance ({clearance})")

    if citizenship_required(description):
        result.reject("requires US citizenship")

    if needs_sponsorship and refuses_sponsorship(description):
        result.reject("employer states they will not sponsor")

    reserved = reserved_for_other_group(description)
    if reserved:
        result.reject(f"reserved for a specific group ({reserved})")

    return result
