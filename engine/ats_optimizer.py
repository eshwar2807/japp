"""Two-pass ATS tailoring engine.

Pass 1  extract what the posting actually asks for            -> JDKeywords
Pass 2  map verified profile facts onto those keywords        -> TailoredResumeDraft

The model's self-reported ``ats_match_percentage`` is treated as advisory only.
The authoritative score is computed locally by :func:`score_match`, which is
deterministic and auditable. If the local score misses the target, the engine
re-prompts with the specific uncovered keywords, up to
``settings.MAX_TAILOR_ITERATIONS`` times.

Determinism note: current Claude models reject ``temperature`` (HTTP 400), so
reproducibility comes from strict structured outputs plus a fixed prompt rather
than from sampling parameters. See config/settings.py.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from rapidfuzz import fuzz

from config import settings
from engine.schemas import (
    JDKeywords,
    MasterProfile,
    TailoredResumeDraft,
    TailoredResumeSchema,
)

log = logging.getLogger(__name__)


class LLMClientProtocol(Protocol):
    """Minimal surface used from the Anthropic SDK, so tests can substitute a stub."""

    def parse(self, **kwargs: Any) -> Any: ...


# --------------------------------------------------------------------------
# Profile loading
# --------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"<[^<>]{2,200}>")


def load_master_profile(path: Path | None = None) -> MasterProfile:
    raw = json.loads(Path(path or settings.MASTER_PROFILE_PATH).read_text())
    return MasterProfile.model_validate(raw)


def find_placeholders(profile: MasterProfile) -> list[str]:
    """Return dotted paths still holding ``<PLACEHOLDER>`` text.

    Applying with placeholders in your resume is worse than not applying, so
    the CLI refuses to submit until this list is empty.
    """
    found: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str) and _PLACEHOLDER_RE.search(node):
            found.append(path)

    walk(profile.model_dump(mode="json"), "")
    return found


# --------------------------------------------------------------------------
# Deterministic local scoring
# --------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9+#. ]+", " ", text.lower())


def keyword_present(keyword: str, haystack: str, threshold: int | None = None) -> bool:
    """True when `keyword` appears in `haystack`, allowing minor variation.

    Exact substring first (cheap, no false positives), then a token-set ratio
    to catch "CI/CD pipelines" vs "CI/CD" and "Postgres" vs "PostgreSQL".
    """
    threshold = threshold or settings.FUZZY_MATCH_THRESHOLD
    kw, hay = _normalize(keyword).strip(), _normalize(haystack)
    if not kw:
        return False
    if kw in hay:
        return True

    # Slide a window of the keyword's own token length across the document and
    # compare like with like. Scoring the keyword against the whole document
    # would dilute the ratio to nothing; scoring against an equal-length window
    # catches real variants ("Postgres"/"PostgreSQL", "system"/"systems")
    # without matching unrelated text.
    kw_tokens = kw.split()
    hay_tokens = hay.split()
    window = len(kw_tokens)
    for i in range(max(len(hay_tokens) - window + 1, 1)):
        chunk = " ".join(hay_tokens[i : i + window])
        if fuzz.token_sort_ratio(kw, chunk) >= threshold:
            return True
    return False


def profile_corpus(profile: MasterProfile) -> str:
    """Every fact the profile actually contains, as one searchable string."""
    parts = [
        " ".join(profile.skills.flat),
        profile.summary or "",
    ]
    for exp in profile.experience:
        parts.extend([exp.company, exp.title, " ".join(exp.bullets),
                      " ".join(exp.tech_used)])
    for edu in profile.education:
        parts.extend([edu.institution, edu.degree, edu.field_of_study])
    for cert in profile.certifications:
        parts.extend([cert.name, cert.issuer])
    for project in profile.projects:
        parts.append(str(project.get("name", "")))
        parts.append(str(project.get("description", "")))
    return "\n".join(p for p in parts if p)


#: Words that describe *what kind* of skill something is rather than naming it.
#: "Kubernetes orchestration" is a fair rephrasing of "Kubernetes"; "Python
#: programming" is not a rephrasing of anything if the profile lacks Python.
#: So the check is on the distinctive tokens, not the whole phrase.
_GENERIC_SKILL_WORDS = {
    "and", "or", "the", "of", "in", "for", "with", "a", "an",
    "architecture", "architectures", "design", "designing", "development",
    "developing", "engineering", "engineer", "management", "managing",
    "orchestration", "automation", "automated", "systems", "system",
    "programming", "pipeline", "pipelines", "cloud", "native", "quality",
    "standards", "support", "practices", "principles", "based", "driven",
    "expertise", "experience", "skills", "advanced", "modern", "scale",
    "scalable", "distributed", "high", "availability", "performance",
    "integration", "deployment", "testing", "migration", "modernization",
    "leadership", "mentorship", "collaboration", "communication", "ownership",
}


def _distinctive_tokens(skill: str) -> list[str]:
    tokens = re.split(r"[^a-z0-9+#.]+", (skill or "").lower())
    return [t for t in tokens if len(t) > 1 and t not in _GENERIC_SKILL_WORDS]


def strip_unsupported_skills(
    draft: TailoredResumeDraft, profile: MasterProfile
) -> tuple[TailoredResumeDraft, list[str]]:
    """Remove highlighted skills the profile does not actually support.

    The prompt forbids claiming a skill the candidate lacks, and a model on the
    cheap tier still added "Python" and "MySQL" to a Java engineer's resume
    because the posting asked for them. A prompt is a request; this is a check.

    Returns the cleaned draft and whatever was removed, so the removal is
    reported rather than silently swallowed.
    """
    corpus = profile_corpus(profile)
    kept, removed = [], []
    for skill in draft.highlighted_skills:
        distinctive = _distinctive_tokens(skill)
        if not distinctive:
            # A phrase made entirely of generic words ("code quality and
            # standards") names no technology, so match the whole phrase.
            supported = keyword_present(skill, corpus)
        else:
            # The first distinctive token is the head of the phrase and carries
            # the claim: "Docker containerization" claims Docker, "Python
            # microservices" claims Python. Requiring every token instead would
            # strip fair rephrasings like "production support and incident
            # management"; requiring any would let "Python microservices"
            # through on the strength of "microservices".
            supported = keyword_present(distinctive[0], corpus)

        (kept if supported else removed).append(skill)

    if not removed:
        return draft, []
    return draft.model_copy(update={"highlighted_skills": kept}), removed


def resume_text(tailored: TailoredResumeSchema | TailoredResumeDraft) -> str:
    """Flatten a tailored resume into the text an ATS parser would read."""
    parts = [tailored.summary, " ".join(tailored.highlighted_skills)]
    for block in tailored.tailored_experience:
        parts.extend([block.title, block.company, " ".join(block.bullets)])
    return "\n".join(parts)


def score_match(
    keywords: JDKeywords,
    tailored: TailoredResumeSchema | TailoredResumeDraft,
) -> tuple[float, dict[str, list[str]]]:
    """Authoritative ATS match score.

    Returns ``(score_0_to_100, {"covered": [...], "missing": [...]})``.
    Categories are weighted per ``settings.SCORE_WEIGHTS``; an absent category
    (e.g. a posting listing no soft skills) has its weight redistributed rather
    than counting as a free 100%.
    """
    text = resume_text(tailored)
    buckets = {
        "hard_skills": keywords.hard_skills,
        "tooling": keywords.tooling,
        "soft_skills": keywords.soft_skills,
    }

    covered: list[str] = []
    missing: list[str] = []
    earned = 0.0
    available = 0.0

    for name, terms in buckets.items():
        weight = settings.SCORE_WEIGHTS.get(name, 0.0)
        if not terms:
            continue
        available += weight
        hits = 0
        for term in terms:
            if keyword_present(term, text):
                covered.append(term)
                hits += 1
            else:
                missing.append(term)
        earned += weight * (hits / len(terms))

    # Title alignment: does the resume echo the posted role?
    title_weight = settings.SCORE_WEIGHTS.get("title", 0.0)
    if keywords.role_title:
        available += title_weight
        titles = " ".join(b.title for b in tailored.tailored_experience)
        ratio = max(
            fuzz.token_set_ratio(_normalize(keywords.role_title), _normalize(titles)),
            fuzz.partial_ratio(_normalize(keywords.role_title), _normalize(text)),
        )
        earned += title_weight * (ratio / 100.0)

    score = 100.0 * earned / available if available else 0.0
    return round(min(score, 100.0), 1), {"covered": covered, "missing": missing}


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

EXTRACT_SYSTEM = """\
You are an ATS keyword extraction engine. Read a job description and return \
structured data about what the posting requires.

Rules:
- Extract only what the posting states. Do not infer or embellish.
- `hard_skills`: technical competencies (e.g. "distributed systems", "REST API design").
- `tooling`: named products, frameworks, languages, platforms (e.g. "Kubernetes", "Python").
- Order every list by how prominently the posting emphasises the item.
- `screener_questions`: application-form questions this employer is likely to ask, \
phrased as the form would phrase them (work authorisation, sponsorship, start date, \
salary expectation, location, and any posting-specific requirement).
- Use the posting's own vocabulary verbatim. If it says "GCP", do not write "Google Cloud"."""

TAILOR_SYSTEM = """\
You are a resume tailoring engine operating under a strict no-fabrication policy.

You receive VERIFIED_FACTS (the candidate's canonical profile) and JD_KEYWORDS.
Your job is to re-express the verified facts using the job description's vocabulary.

ABSOLUTE RULES:
1. Never invent an employer, title, date, degree, certification, metric or technology.
   Every claim must trace to VERIFIED_FACTS.
2. Never claim a skill the candidate does not have. If the posting requires something
   absent from VERIFIED_FACTS, list it in `keywords_missing` and move on. An honest gap
   is required output, not a failure.
3. You MAY: reorder bullets by relevance, adopt the posting's terminology for a
   technology the candidate genuinely used, surface a real accomplishment that the
   original phrasing buried, and tighten wording.
4. Preserve `company`, `title`, `start_date` and `end_date` byte-for-byte from
   VERIFIED_FACTS. These are legal statements.
5. Bullets: start with a strong past-tense verb, one accomplishment each, keep any real
   metric, 1-2 lines. Return every role from VERIFIED_FACTS, most recent first.
6. `highlighted_skills`: only skills present in VERIFIED_FACTS, ordered so JD-relevant
   ones come first.
7. `screener_answers`: answer each question using master profile legal/contact values
   verbatim where one applies (source="profile"). Compose from verified facts only
   otherwise (source="derived"). Never guess a legal answer.
8. `ats_match_percentage`: your honest estimate. It is recomputed independently, so an
   inflated number will simply be overwritten.

Single-column, ATS-parseable prose. No tables, columns, graphics, or special characters
beyond standard punctuation."""


def _build_tailor_prompt(
    job_description: str,
    keywords: JDKeywords,
    profile: MasterProfile,
    few_shot: Sequence[dict] = (),
    missing_hint: Sequence[str] = (),
) -> str:
    sections = [
        "<JOB_DESCRIPTION>\n" + job_description.strip() + "\n</JOB_DESCRIPTION>",
        "<JD_KEYWORDS>\n"
        + json.dumps(keywords.model_dump(), indent=2)
        + "\n</JD_KEYWORDS>",
        "<VERIFIED_FACTS>\n"
        + json.dumps(profile.verified_facts, indent=2, default=str)
        + "\n</VERIFIED_FACTS>",
        "<LEGAL_ANSWERS>\n"
        + json.dumps(
            {
                **{k: v for k, v in profile.legal.items() if not k.startswith("_")},
                **{k: v for k, v in profile.voluntary_disclosures.items() if not k.startswith("_")},
                "full_name": profile.contact.full_name,
                "email": profile.contact.email,
                "phone": profile.contact.phone,
                "location": str(profile.contact.location),
            },
            indent=2,
            default=str,
        )
        + "\n</LEGAL_ANSWERS>",
    ]

    if few_shot:
        sections.append(
            "<PROVEN_EXAMPLES>\n"
            "Bullets below came from this candidate's own past applications that "
            "advanced to a screen, interview or offer. Mirror their structure and "
            "specificity. Do NOT copy their content unless it is also in "
            "VERIFIED_FACTS.\n"
            + json.dumps(list(few_shot), indent=2, default=str)
            + "\n</PROVEN_EXAMPLES>"
        )

    if missing_hint:
        sections.append(
            "<REVISION_REQUEST>\n"
            "A previous attempt scored below target. These JD keywords were not "
            "detected in the output:\n"
            + json.dumps(list(missing_hint), indent=2)
            + "\nFor each: if VERIFIED_FACTS genuinely supports it, surface it "
            "explicitly using the posting's exact wording. If it does not, leave it "
            "in `keywords_missing`. Do not fabricate to close the gap.\n"
            "</REVISION_REQUEST>"
        )

    sections.append(
        "Return the tailored resume for this posting as structured output."
    )
    return "\n\n".join(sections)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


class ATSOptimizer:
    def __init__(
        self,
        client: LLMClientProtocol | None = None,
        model: str = settings.LLM_MODEL,
        profile: MasterProfile | None = None,
        api_key: str | None = None,
        on_usage: Callable[[str, Any], None] | None = None,
    ) -> None:
        self.model = model
        self.profile = profile or load_master_profile()
        self._client = client
        #: Per-user key from the dashboard vault; falls back to the environment.
        self.api_key = api_key
        #: Called as on_usage(phase, response) after every completed call, so the
        #: caller can record token spend without this module knowing about the DB.
        self.on_usage = on_usage

    @property
    def client(self) -> LLMClientProtocol:
        """Lazily build the Anthropic client so importing this module needs no key."""
        if self._client is None:
            import anthropic

            key = self.api_key or settings.ANTHROPIC_API_KEY
            kwargs = {"api_key": key} if key else {}
            self._client = anthropic.Anthropic(**kwargs).messages
        return self._client

    def _call(self, system: str, prompt: str, output_format: type, phase: str = "") -> Any:
        from engine.llm import request_params

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_format": output_format,
            # Thinking and effort follow the model: the bulk tier rejects both.
            # The SDK merges `format` into output_config for us.
            **request_params(self.model, settings.LLM_EFFORT),
        }
        # Only legacy models accept sampling params; current ones return 400.
        if settings.LLM_TEMPERATURE is not None and settings.uses_sampling_params(self.model):
            kwargs["temperature"] = settings.LLM_TEMPERATURE

        response = self.client.parse(**kwargs)
        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError(
                "Model declined this request "
                f"({getattr(response, 'stop_details', None)}). Check the job description input."
            )
        if self.on_usage is not None:
            try:
                self.on_usage(phase, response)
            except Exception:  # accounting must never break a run
                log.exception("Usage callback failed for phase %s", phase)

        parsed = response.parsed_output
        if parsed is None:
            raise RuntimeError("Model returned no parseable structured output.")
        return parsed

    # ---------------- Pass 1 ----------------

    def extract_keywords(self, job_description: str) -> JDKeywords:
        if not job_description.strip():
            raise ValueError("Job description is empty.")
        prompt = (
            "<JOB_DESCRIPTION>\n"
            + job_description.strip()
            + "\n</JOB_DESCRIPTION>\n\nExtract the structured requirements."
        )
        return self._call(EXTRACT_SYSTEM, prompt, JDKeywords, phase="extract_keywords")

    # ---------------- Pass 2 ----------------

    def tailor(
        self,
        job_description: str,
        keywords: JDKeywords,
        few_shot: Sequence[dict] = (),
        missing_hint: Sequence[str] = (),
    ) -> TailoredResumeDraft:
        prompt = _build_tailor_prompt(
            job_description, keywords, self.profile, few_shot, missing_hint
        )
        return self._call(TAILOR_SYSTEM, prompt, TailoredResumeDraft, phase="tailor")

    # ---------------- Orchestration ----------------

    def run(
        self,
        job_description: str,
        few_shot: Sequence[dict] | Callable[[JDKeywords], Sequence[dict]] = (),
        target: float | None = None,
        max_iterations: int | None = None,
    ) -> tuple[TailoredResumeSchema, JDKeywords]:
        """Full two-pass tailoring with score-driven retries.

        ``few_shot`` may be a callable taking the extracted keywords, since the
        feedback loop can only look up comparable past applications once the
        role title is known.

        Returns the best-scoring result even if the target was never reached,
        with the true local score written into ``ats_match_percentage``.
        """
        target = target if target is not None else settings.TARGET_MATCH_SCORE
        max_iterations = max_iterations or settings.MAX_TAILOR_ITERATIONS

        keywords = self.extract_keywords(job_description)
        log.info(
            "Pass 1: %s @ %s | %d hard skills, %d tools",
            keywords.role_title,
            keywords.company or "unknown company",
            len(keywords.hard_skills),
            len(keywords.tooling),
        )

        if callable(few_shot):
            few_shot = few_shot(keywords)
            if few_shot:
                log.info("Feeding %d proven example(s) from past interviews.", len(few_shot))

        best: TailoredResumeSchema | None = None
        best_score = -1.0
        missing: list[str] = []

        for attempt in range(1, max_iterations + 1):
            draft = self.tailor(job_description, keywords, few_shot, missing)

            # Enforce the no-fabrication rule rather than trusting it. Anything
            # the profile does not support is removed before scoring, so an
            # invented skill cannot inflate the match either.
            draft, invented = strip_unsupported_skills(draft, self.profile)
            if invented:
                log.warning(
                    "Removed %d unsupported skill(s) from the tailored resume: %s",
                    len(invented), ", ".join(invented),
                )

            score, detail = score_match(keywords, draft)
            log.info(
                "Pass 2 attempt %d/%d: local score %.1f%% (model claimed %.1f%%)",
                attempt,
                max_iterations,
                score,
                draft.ats_match_percentage,
            )

            result = draft.to_schema()
            result.ats_match_percentage = score       # local score is authoritative
            result.keywords_covered = detail["covered"]
            # Union of what the model admitted to and what the scorer could not
            # find, minus anything actually covered - a keyword must never be
            # reported as both present and missing.
            covered_lower = {c.lower() for c in detail["covered"]}
            result.keywords_missing = sorted(
                {k for k in set(draft.keywords_missing) | set(detail["missing"])
                 if k.lower() not in covered_lower}
            )
            result.removed_unsupported = invented

            if score > best_score:
                best, best_score = result, score
            if score >= target:
                break
            missing = detail["missing"]
            if not missing:
                break  # nothing actionable left; another attempt would be identical

        assert best is not None
        if best_score < target:
            log.warning(
                "Best score %.1f%% is below the %.1f%% target. Genuine gaps: %s",
                best_score,
                target,
                ", ".join(best.keywords_missing[:8]) or "none identified",
            )
        return best, keywords
