"""Pydantic contracts shared by the tailoring engine, PDF renderer and drivers.

Two shapes exist for the tailored resume:

  * ``TailoredResumeDraft`` - what the LLM is asked to emit. Uses a *list* of
    question/answer pairs because strict JSON schema requires
    ``additionalProperties: false``, which forbids free-form dict keys.
  * ``TailoredResumeSchema`` - the spec's public shape, with
    ``screener_answers: dict[str, str]``. Built from the draft via
    :meth:`TailoredResumeDraft.to_schema`.
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_STRICT = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Pass 1 - job description analysis
# --------------------------------------------------------------------------


class JDKeywords(BaseModel):
    """Structured extraction of what the posting actually asks for."""

    model_config = _STRICT

    role_title: str = Field(description="Normalized job title as posted.")
    company: str = Field(default="", description="Hiring company, '' if not stated.")
    seniority: str = Field(default="", description="e.g. Junior, Mid, Senior, Staff.")
    hard_skills: list[str] = Field(
        default_factory=list, description="Technical competencies, most important first."
    )
    soft_skills: list[str] = Field(default_factory=list)
    tooling: list[str] = Field(
        default_factory=list, description="Named products/frameworks/platforms."
    )
    responsibilities: list[str] = Field(default_factory=list)
    required_years_experience: float | None = Field(
        default=None, description="Minimum years stated, null if unstated."
    )
    salary_min: float | None = Field(
        default=None,
        description="Lower bound of any stated salary range, as a yearly number "
        "in the posting's currency. Null if the posting states none.",
    )
    salary_max: float | None = Field(
        default=None, description="Upper bound of any stated salary range. Null if unstated."
    )
    education_requirement: str = Field(default="")
    screener_questions: list[str] = Field(
        default_factory=list,
        description="Application questions implied or stated by the posting.",
    )

    @field_validator("hard_skills", "soft_skills", "tooling", mode="after")
    @classmethod
    def _dedupe(cls, values: list[str]) -> list[str]:
        seen: dict[str, str] = {}
        for v in values:
            cleaned = v.strip()
            if cleaned and cleaned.lower() not in seen:
                seen[cleaned.lower()] = cleaned
        return list(seen.values())

    @property
    def all_keywords(self) -> list[str]:
        return [*self.hard_skills, *self.tooling, *self.soft_skills]


# --------------------------------------------------------------------------
# Pass 2 - tailored resume
# --------------------------------------------------------------------------


class ExperienceBlock(BaseModel):
    """One employment entry, rephrased but factually unchanged."""

    model_config = _STRICT

    company: str
    title: str
    location: str = ""
    start_date: str = Field(description="YYYY-MM, copied verbatim from the master profile.")
    end_date: str = Field(default="Present", description="YYYY-MM or 'Present'.")
    bullets: list[str] = Field(default_factory=list)

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _coerce_date(cls, v: Any) -> str:
        if v in (None, "", "null"):
            return "Present"
        return str(v)

    @field_validator("bullets", mode="after")
    @classmethod
    def _clean_bullets(cls, bullets: list[str]) -> list[str]:
        out = []
        for b in bullets:
            b = re.sub(r"\s+", " ", b).strip().lstrip("•-*").strip()
            if b:
                out.append(b)
        return out


class ScreenerAnswer(BaseModel):
    """A single application-form answer."""

    model_config = _STRICT

    question: str
    answer: str
    source: str = Field(
        default="profile",
        description="'profile' when copied from master_profile.legal/contact, "
        "'derived' when composed from verified facts.",
    )


class TailoredResumeDraft(BaseModel):
    """Exactly what the model returns. Kept strict-schema friendly."""

    model_config = _STRICT

    summary: str
    highlighted_skills: list[str]
    tailored_experience: list[ExperienceBlock]
    ats_match_percentage: float = Field(ge=0.0, le=100.0)
    screener_answers: list[ScreenerAnswer] = Field(default_factory=list)
    keywords_covered: list[str] = Field(default_factory=list)
    keywords_missing: list[str] = Field(
        default_factory=list,
        description="JD keywords with no honest support in the master profile. "
        "Never fabricate these - they are reported as genuine gaps.",
    )

    def to_schema(self) -> "TailoredResumeSchema":
        return TailoredResumeSchema(
            summary=self.summary,
            highlighted_skills=self.highlighted_skills,
            tailored_experience=self.tailored_experience,
            ats_match_percentage=self.ats_match_percentage,
            screener_answers={a.question: a.answer for a in self.screener_answers},
            keywords_covered=self.keywords_covered,
            keywords_missing=self.keywords_missing,
        )


class TailoredResumeSchema(BaseModel):
    """Public shape used by the PDF renderer, the drivers and the DB."""

    model_config = ConfigDict(extra="ignore")

    summary: str
    highlighted_skills: list[str]
    tailored_experience: list[ExperienceBlock]
    ats_match_percentage: float = Field(ge=0.0, le=100.0)
    screener_answers: dict[str, str] = Field(default_factory=dict)
    keywords_covered: list[str] = Field(default_factory=list)
    keywords_missing: list[str] = Field(default_factory=list)

    @field_validator("screener_answers", mode="before")
    @classmethod
    def _normalize_answers(cls, v: Any) -> dict[str, str]:
        """Accept either the dict form or the list-of-pairs form."""
        if isinstance(v, dict):
            return {str(k): str(val) for k, val in v.items()}
        if isinstance(v, list):
            out: dict[str, str] = {}
            for item in v:
                if isinstance(item, ScreenerAnswer):
                    out[item.question] = item.answer
                elif isinstance(item, dict) and "question" in item:
                    out[str(item["question"])] = str(item.get("answer", ""))
            return out
        return {}


# --------------------------------------------------------------------------
# Master profile
# --------------------------------------------------------------------------


class ProfileExperience(BaseModel):
    model_config = ConfigDict(extra="allow")

    company: str
    title: str
    location: str = ""
    start_date: str
    end_date: str | None = None
    is_current: bool = False
    employment_type: str = ""
    bullets: list[str] = Field(default_factory=list)
    tech_used: list[str] = Field(default_factory=list)

    @property
    def display_end(self) -> str:
        return "Present" if self.is_current or not self.end_date else self.end_date


class ProfileEducation(BaseModel):
    model_config = ConfigDict(extra="allow")

    institution: str
    degree: str = ""
    field_of_study: str = ""
    location: str = ""
    start_date: str = ""
    end_date: str = ""
    gpa: float | None = None


class ProfileCertification(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    issuer: str = ""
    issue_date: str = ""
    expiry_date: str | None = None
    credential_id: str = ""


class ProfileSkills(BaseModel):
    model_config = ConfigDict(extra="allow")

    hard: list[str] = Field(default_factory=list)
    tooling: list[str] = Field(default_factory=list)
    soft: list[str] = Field(default_factory=list)

    @property
    def flat(self) -> list[str]:
        return [*self.hard, *self.tooling, *self.soft]


class ProfileLocation(BaseModel):
    model_config = ConfigDict(extra="allow")

    city: str = ""
    state: str = ""
    country: str = ""
    postal_code: str = ""
    willing_to_relocate: bool = False
    remote_preference: str = ""

    def __str__(self) -> str:
        return ", ".join(p for p in (self.city, self.state) if p)


class ProfileContact(BaseModel):
    model_config = ConfigDict(extra="allow")

    full_name: str
    preferred_name: str = ""
    email: str
    phone: str = ""
    location: ProfileLocation = Field(default_factory=ProfileLocation)
    links: dict[str, str] = Field(default_factory=dict)

    @property
    def first_name(self) -> str:
        return (self.preferred_name or self.full_name).split()[0] if self.full_name else ""

    @property
    def last_name(self) -> str:
        parts = self.full_name.split()
        return parts[-1] if len(parts) > 1 else ""


class MasterProfile(BaseModel):
    """Validated view of ``config/master_profile.json``."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    schema_version: str = "1.0.0"
    contact: ProfileContact
    target_titles: list[str] = Field(default_factory=list)
    summary: str = ""
    skills: ProfileSkills = Field(default_factory=ProfileSkills)
    experience: list[ProfileExperience] = Field(default_factory=list)
    education: list[ProfileEducation] = Field(default_factory=list)
    certifications: list[ProfileCertification] = Field(default_factory=list)
    projects: list[dict[str, Any]] = Field(default_factory=list)
    legal: dict[str, Any] = Field(default_factory=dict)
    voluntary_disclosures: dict[str, Any] = Field(default_factory=dict)
    preferences: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _drop_comments(cls, data: Any) -> Any:
        """Strip the ``_comment`` documentation keys used throughout the file."""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if not k.startswith("_")}
        return data

    @property
    def verified_facts(self) -> dict[str, Any]:
        """The only facts the LLM is allowed to draw on. Nothing else exists."""
        return {
            "skills": self.skills.flat,
            "experience": [
                {
                    "company": e.company,
                    "title": e.title,
                    "location": e.location,
                    "start_date": e.start_date,
                    "end_date": e.display_end,
                    "bullets": e.bullets,
                    "tech_used": e.tech_used,
                }
                for e in self.experience
            ],
            "education": [e.model_dump() for e in self.education],
            "certifications": [c.model_dump() for c in self.certifications],
            "summary": self.summary,
        }


PlaceholderPattern = Annotated[str, Field(pattern=r"^(?!<).*")]
