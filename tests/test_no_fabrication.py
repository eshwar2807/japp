"""Skills claimed on a resume must be evidenced by the profile.

This exists because of a live failure. Tailoring a Java engineer's profile
against a posting that wanted Python and Spark produced a resume listing
"Python" and "MySQL" in its skills. The profile mentioned neither. The prompt
forbids exactly that, and the model did it anyway on the cheap tier.

A prompt is a request. This is a check.
"""

from __future__ import annotations

import pytest

from engine.ats_optimizer import (
    profile_corpus,
    strip_unsupported_skills,
)
from engine.schemas import ExperienceBlock, MasterProfile, TailoredResumeDraft


@pytest.fixture()
def java_profile():
    """A Java/Spring engineer. No Python, no MySQL, no Spark anywhere."""
    return MasterProfile.model_validate({
        "contact": {"full_name": "E N", "email": "a@b.com"},
        "skills": {
            "hard": ["Microservices Architecture", "CI/CD Pipeline Automation",
                     "Production Support"],
            "tooling": ["Java", "Spring Boot", "Apigee Hybrid", "Kubernetes",
                        "SQL", "Docker"],
        },
        "experience": [{
            "company": "Acme", "title": "Senior Software Engineer",
            "start_date": "2021-01", "is_current": True,
            "bullets": ["Led API gateway migration to Apigee Hybrid.",
                        "Migrated CI/CD from Jenkins to GitHub Actions."],
            "tech_used": ["Jenkins"],
        }],
        "education": [{"institution": "University", "degree": "BSc"}],
    })


def draft_with(skills):
    return TailoredResumeDraft(
        summary="Engineer.", highlighted_skills=skills,
        tailored_experience=[ExperienceBlock(company="Acme", title="SSE",
                                             start_date="2021-01")],
        ats_match_percentage=61.0)


# ---------------- the live failure ----------------


@pytest.mark.parametrize("invented", ["Python", "MySQL", "Spark", "Kotlin", "Airflow"])
def test_a_skill_absent_from_the_profile_is_removed(java_profile, invented):
    clean, removed = strip_unsupported_skills(draft_with(["Java", invented]), java_profile)
    assert invented in removed
    assert invented not in clean.highlighted_skills
    assert "Java" in clean.highlighted_skills


def test_the_exact_live_fabrication_is_caught(java_profile):
    """Python and MySQL on a Java engineer's resume."""
    clean, removed = strip_unsupported_skills(
        draft_with(["Distributed systems design", "Java", "Python", "SQL", "MySQL"]),
        java_profile)
    assert set(removed) == {"Python", "MySQL", "Distributed systems design"}
    assert clean.highlighted_skills == ["Java", "SQL"]


def test_a_fabrication_hidden_behind_a_real_word_is_still_caught(java_profile):
    """"Python microservices" claims Python, however familiar the second word."""
    _, removed = strip_unsupported_skills(
        draft_with(["Python microservices", "Advanced Python"]), java_profile)
    assert removed == ["Python microservices", "Advanced Python"]


# ---------------- fair rephrasings survive ----------------


@pytest.mark.parametrize("rephrasing", [
    "Kubernetes orchestration",
    "Docker containerization",
    "Microservices architecture",
    "Production support and incident management",
    "Java development",
])
def test_a_rephrasing_of_a_real_skill_is_kept(java_profile, rephrasing):
    """Stripping these would weaken the resume for no honesty gain."""
    clean, removed = strip_unsupported_skills(draft_with([rephrasing]), java_profile)
    assert removed == []
    assert clean.highlighted_skills == [rephrasing]


def test_a_skill_evidenced_only_in_a_bullet_counts(java_profile):
    """Jenkins appears in a bullet and in tech_used, not the skills list."""
    clean, removed = strip_unsupported_skills(draft_with(["Jenkins"]), java_profile)
    assert removed == [] and clean.highlighted_skills == ["Jenkins"]


def test_nothing_is_removed_when_everything_is_supported(java_profile):
    skills = ["Java", "Spring Boot", "Kubernetes", "Docker", "SQL"]
    clean, removed = strip_unsupported_skills(draft_with(skills), java_profile)
    assert removed == []
    assert clean.highlighted_skills == skills


def test_the_corpus_includes_bullets_and_tech_used(java_profile):
    corpus = profile_corpus(java_profile).lower()
    assert "apigee" in corpus and "jenkins" in corpus and "github actions" in corpus
    assert "python" not in corpus


# ---------------- integration with scoring ----------------


def test_an_invented_skill_cannot_inflate_the_match_score(java_profile):
    """The fabrication also raised the score, because the scorer saw the word
    in the resume text. Stripping happens before scoring for that reason."""
    from types import SimpleNamespace

    from engine.ats_optimizer import ATSOptimizer
    from engine.schemas import JDKeywords

    keywords = JDKeywords(role_title="Backend Engineer",
                          hard_skills=["Python", "Java"], tooling=["Spark", "Kubernetes"])
    inflated = draft_with(["Java", "Python", "Spark", "Kubernetes"])

    class Stub:
        def parse(self, **kwargs):
            payload = keywords if kwargs["output_format"] is JDKeywords else inflated
            return SimpleNamespace(parsed_output=payload, stop_reason="end_turn")

    optimizer = ATSOptimizer(client=Stub(), profile=java_profile)
    resume, _ = optimizer.run("A job needing Python, Java, Spark and Kubernetes.",
                              max_iterations=1)

    assert "Python" not in resume.highlighted_skills
    assert "Spark" not in resume.highlighted_skills
    assert set(resume.removed_unsupported) == {"Python", "Spark"}
    assert resume.ats_match_percentage < 100


def test_a_keyword_is_never_both_covered_and_missing(java_profile):
    """The live run reported Python as covered and missing at once."""
    from types import SimpleNamespace

    from engine.ats_optimizer import ATSOptimizer
    from engine.schemas import JDKeywords

    keywords = JDKeywords(role_title="Backend Engineer",
                          hard_skills=["Java"], tooling=["Kubernetes"])
    draft = draft_with(["Java", "Kubernetes"])
    draft = draft.model_copy(update={"keywords_missing": ["Java"]})

    class Stub:
        def parse(self, **kwargs):
            payload = keywords if kwargs["output_format"] is JDKeywords else draft
            return SimpleNamespace(parsed_output=payload, stop_reason="end_turn")

    resume, _ = ATSOptimizer(client=Stub(), profile=java_profile).run("x", max_iterations=1)

    overlap = {c.lower() for c in resume.keywords_covered} & {
        m.lower() for m in resume.keywords_missing}
    assert overlap == set()
