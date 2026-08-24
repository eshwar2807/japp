"""Module B validation: keyword scoring, two-pass orchestration, PDF rendering.

The Anthropic call is exercised through a stub client so the engine's control
flow (retries, score override, gap reporting) is tested without network access.
The scorer and the PDF pipeline are tested for real.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from engine.ats_optimizer import (
    ATSOptimizer,
    find_placeholders,
    keyword_present,
    load_master_profile,
    score_match,
)
from engine.pdf_generator import PDFGenerator, build_filename, slugify
from engine.schemas import (
    ExperienceBlock,
    JDKeywords,
    MasterProfile,
    ScreenerAnswer,
    TailoredResumeDraft,
    TailoredResumeSchema,
)

JD = """
Senior Backend Engineer at Acme Robotics.
We need strong Python and distributed systems experience. You will design REST APIs,
work with PostgreSQL and Kubernetes, and own CI/CD. 5+ years required.
Strong communication and mentoring skills.
"""


# ---------------- fixtures ----------------


@pytest.fixture()
def keywords():
    return JDKeywords(
        role_title="Senior Backend Engineer",
        company="Acme Robotics",
        seniority="Senior",
        hard_skills=["Python", "distributed systems", "REST API design", "CI/CD"],
        soft_skills=["communication", "mentoring"],
        tooling=["PostgreSQL", "Kubernetes"],
        required_years_experience=5,
        screener_questions=["Are you authorized to work in the US?"],
    )


def make_draft(bullets, skills, pct=91.0, missing=()):
    return TailoredResumeDraft(
        summary="Backend engineer focused on distributed systems.",
        highlighted_skills=list(skills),
        tailored_experience=[
            ExperienceBlock(
                company="Acme",
                title="Senior Backend Engineer",
                start_date="2022-06",
                end_date="Present",
                bullets=list(bullets),
            )
        ],
        ats_match_percentage=pct,
        screener_answers=[
            ScreenerAnswer(question="Are you authorized to work in the US?", answer="Yes")
        ],
        keywords_missing=list(missing),
    )


class StubClient:
    """Returns queued responses in order; records the prompts it received."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.responses.pop(0)
        return SimpleNamespace(parsed_output=payload, stop_reason="end_turn")


# ---------------- keyword matching ----------------


@pytest.mark.parametrize(
    "kw,text,expected",
    [
        ("Python", "Built services in Python 3.11", True),
        ("CI/CD", "Owned the CI/CD pipeline end to end", True),
        ("PostgreSQL", "Migrated Postgres schemas", True),          # fuzzy variant
        ("distributed systems", "Designed distributed system nodes", True),
        ("Kubernetes", "Deployed with Docker Compose", False),
        ("Rust", "Wrote Python and Go services", False),
        ("", "anything", False),
    ],
)
def test_keyword_present(kw, text, expected):
    assert keyword_present(kw, text) is expected


def test_score_rewards_coverage(keywords):
    strong = make_draft(
        bullets=[
            "Designed REST API design patterns for distributed systems in Python",
            "Ran PostgreSQL migrations and Kubernetes rollouts through CI/CD",
            "Led communication with stakeholders and mentoring of two engineers",
        ],
        skills=["Python", "PostgreSQL", "Kubernetes"],
    )
    weak = make_draft(bullets=["Wrote some code"], skills=["Excel"])

    strong_score, strong_detail = score_match(keywords, strong)
    weak_score, weak_detail = score_match(keywords, weak)

    assert strong_score > 80.0
    assert weak_score < 30.0
    assert not strong_detail["missing"]
    assert "Kubernetes" in weak_detail["missing"]


def test_score_ignores_absent_categories():
    """A posting with no soft skills listed must not be scored on soft skills."""
    kw = JDKeywords(role_title="Data Engineer", hard_skills=["Python"], tooling=["Airflow"])
    draft = make_draft(bullets=["Built Airflow DAGs in Python"], skills=["Python"])
    draft.tailored_experience[0].title = "Data Engineer"
    score, _ = score_match(kw, draft)
    assert score > 95.0


def test_score_is_bounded(keywords):
    score, _ = score_match(keywords, make_draft(bullets=["Python " * 200], skills=[]))
    assert 0.0 <= score <= 100.0


# ---------------- schema behaviour ----------------


def test_screener_answers_normalize_from_list():
    draft = make_draft(bullets=["x"], skills=["Python"])
    schema = draft.to_schema()
    assert schema.screener_answers == {"Are you authorized to work in the US?": "Yes"}


def test_screener_answers_accept_dict_form():
    schema = TailoredResumeSchema(
        summary="s",
        highlighted_skills=[],
        tailored_experience=[],
        ats_match_percentage=50,
        screener_answers={"Q": "A"},
    )
    assert schema.screener_answers == {"Q": "A"}


def test_bullets_are_cleaned():
    block = ExperienceBlock(
        company="A", title="B", start_date="2020-01", bullets=["• Did   a thing ", "", "- Other"]
    )
    assert block.bullets == ["Did a thing", "Other"]


def test_percentage_out_of_range_rejected():
    with pytest.raises(Exception):
        make_draft(bullets=["x"], skills=[], pct=140.0)


# ---------------- orchestration ----------------


def test_run_overrides_model_claimed_score(keywords, profile):
    """A model claiming 99% on a thin resume must not survive."""
    inflated = make_draft(bullets=["Wrote code"], skills=["Excel"], pct=99.0)
    client = StubClient(keywords, inflated, inflated, inflated)
    opt = ATSOptimizer(client=client, profile=profile)

    result, kw = opt.run(JD)

    assert result.ats_match_percentage < 50.0     # local score wins
    assert kw.role_title == "Senior Backend Engineer"
    assert "Kubernetes" in result.keywords_missing


def test_run_stops_early_when_target_met(keywords, profile):
    good = make_draft(
        bullets=[
            "Built distributed systems in Python with REST API design",
            "Owned CI/CD, PostgreSQL and Kubernetes; mentoring and communication",
        ],
        skills=["Python"],
    )
    client = StubClient(keywords, good, good, good)
    opt = ATSOptimizer(client=client, profile=profile)
    result, _ = opt.run(JD)

    assert result.ats_match_percentage >= 80.0
    assert len(client.calls) == 2  # pass 1 + a single pass 2


def test_retry_feeds_missing_keywords_back_into_prompt(keywords, profile):
    weak = make_draft(bullets=["Wrote code"], skills=[], pct=40.0)
    client = StubClient(keywords, weak, weak, weak)
    ATSOptimizer(client=client, profile=profile).run(JD)

    assert len(client.calls) == 4  # pass 1 + 3 tailoring attempts
    retry_prompt = client.calls[2]["messages"][0]["content"]
    assert "REVISION_REQUEST" in retry_prompt
    assert "Kubernetes" in retry_prompt


def test_few_shot_examples_reach_the_prompt(keywords, profile):
    good = make_draft(bullets=["Python distributed systems REST API design CI/CD"], skills=[])
    client = StubClient(keywords, good, good, good)
    opt = ATSOptimizer(client=client, profile=profile)
    opt.run(JD, few_shot=[{"role_title": "Backend Engineer", "outcome": "Interview"}])

    prompt = client.calls[1]["messages"][0]["content"]
    assert "PROVEN_EXAMPLES" in prompt and "Interview" in prompt


def test_prompt_never_leaks_unverified_facts(keywords, profile):
    """Only master-profile facts may enter the tailoring prompt."""
    good = make_draft(bullets=["x"], skills=[])
    client = StubClient(keywords, good, good, good)
    ATSOptimizer(client=client, profile=profile).run(JD)

    prompt = client.calls[1]["messages"][0]["content"]
    assert "VERIFIED_FACTS" in prompt and "LEGAL_ANSWERS" in prompt
    for company in (e.company for e in profile.experience):
        assert company in prompt


def test_temperature_not_sent_to_current_models(keywords, profile):
    """Sampling params return HTTP 400 on current models - never send them."""
    good = make_draft(bullets=["x"], skills=[])
    client = StubClient(keywords, good, good, good)
    ATSOptimizer(client=client, profile=profile, model="claude-opus-5").run(JD)
    assert all("temperature" not in call for call in client.calls)


def test_empty_job_description_rejected(profile):
    with pytest.raises(ValueError):
        ATSOptimizer(client=StubClient(), profile=profile).extract_keywords("   ")


def test_refusal_stop_reason_surfaces(profile):
    class Refusing:
        def parse(self, **kw):
            return SimpleNamespace(parsed_output=None, stop_reason="refusal", stop_details=None)

    with pytest.raises(RuntimeError, match="declined"):
        ATSOptimizer(client=Refusing(), profile=profile).extract_keywords(JD)


# ---------------- profile loading ----------------


def test_master_profile_loads_and_strips_comments(profile):
    assert isinstance(profile, MasterProfile)
    assert profile.contact.email
    assert "_comment" not in profile.model_dump()
    assert profile.skills.hard


def test_placeholder_detection_flags_template(profile):
    missing = find_placeholders(profile)
    assert any("contact.full_name" in m for m in missing)


def test_placeholder_detection_clean_profile():
    clean = MasterProfile.model_validate(
        {"contact": {"full_name": "Ada Lovelace", "email": "ada@example.com"}}
    )
    assert find_placeholders(clean) == []


# ---------------- PDF ----------------


@pytest.fixture()
def real_profile():
    """A fully-populated profile, since the shipped one is a placeholder template."""
    return MasterProfile.model_validate(
        {
            "contact": {
                "full_name": "Ada Lovelace",
                "email": "ada@example.com",
                "phone": "+1-555-010-0100",
                "location": {"city": "Austin", "state": "TX"},
                "links": {"linkedin": "https://linkedin.com/in/ada"},
            },
            "skills": {
                "hard": ["Python", "Distributed Systems"],
                "tooling": ["PostgreSQL", "Kubernetes"],
                "soft": ["Mentoring"],
            },
            "experience": [
                {
                    "company": "Analytical Engines",
                    "title": "Principal Engineer",
                    "start_date": "2021-03",
                    "is_current": True,
                    "bullets": ["Cut p99 latency 60% across 14 services."],
                }
            ],
            "education": [
                {
                    "institution": "University of London",
                    "degree": "BSc",
                    "field_of_study": "Mathematics",
                    "end_date": "1842-05",
                }
            ],
            "certifications": [{"name": "CKA", "issuer": "CNCF", "issue_date": "2023-04"}],
        }
    )


@pytest.fixture()
def tailored():
    return TailoredResumeSchema(
        summary="Principal engineer with deep distributed systems experience.",
        highlighted_skills=["Python", "Kubernetes", "Distributed Systems", "Mentoring"],
        tailored_experience=[
            ExperienceBlock(
                company="Analytical Engines",
                title="Principal Engineer",
                location="Austin, TX",
                start_date="2021-03",
                end_date="Present",
                bullets=[
                    "Cut p99 latency 60% across 14 services using Python and Kubernetes.",
                    "Mentored four engineers through distributed systems design reviews.",
                ],
            )
        ],
        ats_match_percentage=88.0,
        screener_answers={"Are you authorized to work in the US?": "Yes"},
    )


def test_html_renders_all_sections(tailored, real_profile):
    html = PDFGenerator().render_html(tailored, real_profile)
    for expected in (
        "Ada Lovelace",
        "ada@example.com",
        "Principal Engineer",
        "Analytical Engines",
        "Cut p99 latency 60%",
        "University of London",
        "CKA",
        "SUMMARY".title(),
    ):
        assert expected in html


def test_html_escapes_injected_markup(real_profile, tailored):
    tailored.summary = "<script>alert(1)</script> & more"
    html = PDFGenerator().render_html(tailored, real_profile)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_pdf_generates_and_verifies(tailored, real_profile, tmp_path):
    out = tmp_path / "resume.pdf"
    path = PDFGenerator().generate(
        tailored, real_profile, "Acme Robotics", "Senior Backend Engineer", output_path=out
    )
    assert path.exists() and path.stat().st_size > 1024
    assert path.read_bytes()[:5] == b"%PDF-"

    report = PDFGenerator.verify(path, tailored)
    assert report["pages"] == 1
    assert report["text_extracted"] > 200


def test_pdf_text_is_extractable_by_an_ats(tailored, real_profile, tmp_path):
    """The whole point: an ATS must be able to read the words back out."""
    from pypdf import PdfReader

    out = tmp_path / "r.pdf"
    PDFGenerator().generate(tailored, real_profile, "Acme", "SBE", output_path=out)
    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(out)).pages)

    for expected in ("Ada Lovelace", "Analytical Engines", "Kubernetes", "p99 latency"):
        assert expected in text, f"{expected!r} not extractable from the PDF"


def test_verify_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf" * 500)
    with pytest.raises(ValueError, match="bad header"):
        PDFGenerator.verify(bad)


def test_verify_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        PDFGenerator.verify(tmp_path / "nope.pdf")


@pytest.mark.parametrize(
    "raw,expected",
    [("Acme Corp. (US)", "Acme_Corp_US"), ("Sr. Engineer/Lead", "Sr_Engineer_Lead"), ("", "unknown")],
)
def test_slugify(raw, expected):
    assert slugify(raw) == expected


def test_build_filename_shape():
    name = build_filename("Ada Lovelace", "Acme Robotics", "Senior Backend Engineer")
    assert name.startswith("Ada_Lovelace_Acme_Robotics_Senior_Backend_Engineer_")
    assert name.endswith(".pdf")


def test_contact_line_does_not_run_into_the_name(tailored, real_profile, tmp_path):
    """Regression: the name once extracted as 'Ada LovelaceAustin, TX'.

    An ATS that reads the name glued to the city gets the candidate's name
    wrong on the very first field, so the heading must extract as its own line
    and every separator must carry surrounding whitespace.
    """
    pypdf = pytest.importorskip("pypdf")

    path = PDFGenerator().generate(
        tailored, real_profile, "Acme", "Engineer", output_path=tmp_path / "r.pdf"
    )
    text = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(str(path)).pages)

    assert "Ada LovelaceAustin" not in text
    assert "Ada Lovelace" in text
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    assert lines[0] == "Ada Lovelace"
    # Contact details survive on their own line, separated readably.
    assert any("ada@example.com" in ln and "Austin" in ln for ln in lines)


def test_extracted_text_preserves_every_bullet(tailored, real_profile, tmp_path):
    """Bullets are the payload; none may be dropped or truncated by rendering."""
    pypdf = pytest.importorskip("pypdf")

    path = PDFGenerator().generate(
        tailored, real_profile, "Acme", "Engineer", output_path=tmp_path / "r.pdf"
    )
    text = " ".join(
        (p.extract_text() or "").replace("\n", " ") for p in pypdf.PdfReader(str(path)).pages
    )
    normalized = " ".join(text.split())

    for block in tailored.tailored_experience:
        for bullet in block.bullets:
            assert " ".join(bullet.split()) in normalized, f"bullet lost in render: {bullet}"


# ---------------- matcher calibration ----------------
#
# Hand-checked cases against a Java/Spring enterprise profile. The matcher was
# wrong in both directions: it missed real coverage on paraphrased multi-word
# requirements, and credited short technology names that appeared inside longer
# words. Both distorted every score in the system.


@pytest.fixture()
def java_corpus():
    from engine.ats_optimizer import profile_corpus

    profile = MasterProfile.model_validate({
        "contact": {"full_name": "E N", "email": "a@b.com"},
        "skills": {
            "hard": ["Microservices Architecture", "RESTful API Development",
                     "Cloud-Native Deployment", "CI/CD Pipeline Automation",
                     "Docker Containerization & Orchestration",
                     "Scalable Distributed Systems", "High-Availability System Design",
                     "Application Security & Vulnerability Remediation"],
            "tooling": ["Java", "Spring Boot", "Spring Data JPA", "Git", "GitHub",
                        "Docker", "Kubernetes", "Jenkins", "SQL", "Hibernate"],
        },
        "experience": [{"company": "Acme", "title": "Senior Software Engineer",
                        "start_date": "2021-01", "is_current": True,
                        "bullets": ["Led API gateway migration."]}],
    })
    return profile_corpus(profile)


@pytest.mark.parametrize("requirement", [
    "cloud-native architectures",      # profile says Cloud-Native Deployment
    "container orchestration",         # Docker Containerization & Orchestration
    "RESTful APIs",                    # RESTful API Development (plural)
    "Git version control",             # Git
    "API security patterns",           # RESTful API + Application Security
    "microservices architecture",
    "distributed systems",
    "CI/CD pipelines",
    "Java",
    "Spring Boot",
])
def test_paraphrased_requirements_are_recognised(java_corpus, requirement):
    """These were all scored as missing, which capped every ceiling far below
    what the profile actually supports."""
    from engine.ats_optimizer import keyword_present

    assert keyword_present(requirement, java_corpus) is True


@pytest.mark.parametrize("requirement", [
    "Go",                    # matched fuzzily against an unrelated word
    "Scala",                 # matched inside "Scalable"
    "Python", "Kotlin", "Rust", "Spark", "Terraform", "GraphQL", "AWS Lambda",
    "machine learning",
    "data modeling",         # matched on "data" from Spring Data JPA
    "Python microservices",  # claimed Python behind a familiar second word
    "Hexagonal architecture",
])
def test_absent_technologies_are_not_credited(java_corpus, requirement):
    """A false positive here inflates the score and, because the fabrication
    check shares this function, lets an invented skill look supported."""
    from engine.ats_optimizer import keyword_present

    assert keyword_present(requirement, java_corpus) is False


def test_short_names_require_an_exact_word(java_corpus):
    """Two- and three-letter languages must not ride on a longer word."""
    from engine.ats_optimizer import keyword_present

    assert keyword_present("Java", java_corpus) is True     # present exactly
    assert keyword_present("Go", java_corpus) is False
    assert keyword_present("R", java_corpus) is False


def test_ceiling_reflects_what_the_profile_can_reach(java_corpus):
    """The ceiling is what a resume containing everything would score, so a
    posting the profile cannot meet is identifiable before any tailoring."""
    from engine.ats_optimizer import ceiling_score
    from engine.schemas import JDKeywords, MasterProfile

    profile = MasterProfile.model_validate({
        "contact": {"full_name": "E", "email": "a@b.com"},
        "skills": {"hard": ["Microservices Architecture"],
                   "tooling": ["Java", "Spring Boot", "Docker"]},
    })
    reachable = JDKeywords(role_title="Java Engineer",
                           hard_skills=["microservices architecture"],
                           tooling=["Java", "Spring Boot", "Docker"])
    unreachable = JDKeywords(role_title="ML Engineer",
                             hard_skills=["machine learning"],
                             tooling=["Python", "PyTorch", "Spark"])

    assert ceiling_score(reachable, profile)[0] == 100.0
    assert ceiling_score(unreachable, profile)[0] < 20.0
