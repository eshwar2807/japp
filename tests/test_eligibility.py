"""Hard constraints screened before anything is spent.

These encode things no amount of tailoring can change. Two matter most for a
visa holder: a US security clearance cannot be obtained on an H-1B, and an
H-1B transfer is sponsorship, so an employer who rules either out has closed
the role regardless of how well the resume matches.
"""

from __future__ import annotations

import pytest

from engine.eligibility import (
    assess,
    citizenship_required,
    clearance_required,
    is_java_role,
    refuses_sponsorship,
    years_required,
)


# ---------------- Java ----------------


@pytest.mark.parametrize("title,description", [
    ("Senior Java Developer", "Build services."),
    ("Backend Engineer", "We use Java 17, Spring Boot and Hibernate."),
    ("Software Engineer", "Experience with J2EE and Maven required."),
    ("Platform Engineer", "JVM tuning and Quarkus microservices."),
])
def test_java_roles_are_recognised(title, description):
    assert is_java_role(title, description) is True


@pytest.mark.parametrize("title,description", [
    ("Frontend Engineer", "React, JavaScript and TypeScript."),
    ("Full Stack Engineer", "Node.js and JavaScript across the stack."),
    ("Data Scientist", "Python, pandas, PyTorch."),
    ("SRE", "Go, Terraform, Kubernetes."),
])
def test_non_java_roles_are_rejected(title, description):
    """"JavaScript" contains "Java"; without stripping it every front-end role
    would look like a match."""
    assert is_java_role(title, description) is False


# ---------------- experience ----------------


@pytest.mark.parametrize("text,expected", [
    ("5+ years of experience", 5),
    ("Minimum 8 years of software engineering experience", 8),
    ("3-5 years experience required", 3),
    ("10+ years of relevant experience", 10),
    ("At least 6 years working with Java", 6),
    ("No number given here", None),
])
def test_years_required_is_read_from_the_posting(text, expected):
    assert years_required(text) == expected


def test_a_requirement_written_before_the_number_is_found():
    """A Staff role at 8+ years passed a six-year ceiling because the pattern
    required the word experience to follow the number, and this posting put it
    first: 'Extensive experience (typically 8+ years) building...'
    """
    assert years_required(
        "Extensive experience (typically 8+ years) building and operating "
        "backend distributed systems at scale"
    ) == 8


def test_the_highest_stated_requirement_wins():
    """Reversed from taking the minimum. A posting wanting "8+ years backend"
    and "2+ years Kubernetes" is an eight-year role with a secondary skill;
    reading it as a two-year role let Staff openings through."""
    assert years_required("8+ years of backend experience. "
                          "2+ years of Kubernetes experience.") == 8


@pytest.mark.parametrize("text", [
    "We believe the way people interact with their finances will improve in "
    "the next few years",
    "Founded 12 years ago, we now serve millions of customers",
    "Over the past 5 years we have grown considerably",
])
def test_narrative_year_counts_are_not_requirements(text):
    """Company history and forward-looking statements are not job criteria."""
    assert years_required(text) is None


def test_a_role_above_the_years_limit_is_rejected():
    verdict = assess("Senior Java Engineer", "Java Spring Boot. 10+ years of experience.",
                     max_years=6)
    assert verdict.eligible is False
    assert "10+ years" in verdict.reasons[0]


def test_a_role_within_the_years_limit_is_accepted():
    assert assess("Senior Java Engineer",
                  "Java Spring Boot. 4+ years of experience.", max_years=6).eligible


def test_a_role_stating_no_years_is_not_rejected_for_it():
    """Silence is not a bar; most postings do not state a number."""
    assert assess("Senior Java Engineer", "Java and Spring Boot.", max_years=6).eligible


# ---------------- clearance ----------------


@pytest.mark.parametrize("text", [
    "Active TS/SCI clearance required",
    "Must have Secret clearance",
    "Top Secret clearance with polygraph",
    "Public Trust clearance required",
    "DoD clearance required",
])
def test_clearance_requirements_are_detected(text):
    assert clearance_required(text) != ""


@pytest.mark.parametrize("text", [
    "Background check required",
    "Drug screen prior to start",
    "Employment is contingent on a background check",
])
def test_ordinary_screening_is_not_a_clearance(text):
    """Everyone goes through these; treating them as disqualifying would
    reject most of the market."""
    assert clearance_required(text) == ""


def test_a_clearance_role_is_rejected_for_an_h1b_holder():
    verdict = assess("Senior Java Engineer",
                     "Java Spring Boot. Active Secret clearance required.",
                     can_obtain_clearance=False)
    assert verdict.eligible is False
    assert "clearance" in verdict.reasons[0]


def test_a_clearance_role_is_allowed_when_the_candidate_can_obtain_one():
    assert assess("Senior Java Engineer",
                  "Java Spring Boot. Secret clearance required.",
                  can_obtain_clearance=True).eligible


# ---------------- citizenship and sponsorship ----------------


@pytest.mark.parametrize("text", [
    "US citizenship is required",
    "Must be a U.S. citizen",
    "US citizens only",
    "Citizenship required for this role",
])
def test_citizenship_requirements_are_detected(text):
    assert citizenship_required(text) is True


@pytest.mark.parametrize("text", [
    "We are unable to sponsor visas at this time",
    "No visa sponsorship available",
    "Candidates must be authorized to work in the US without sponsorship",
    "We do not sponsor employment visas",
    "Applicants must not require sponsorship now or in the future",
])
def test_sponsorship_refusals_are_detected(text):
    assert refuses_sponsorship(text) is True


@pytest.mark.parametrize("text", [
    "We welcome applicants of all backgrounds",
    "We sponsor visas for exceptional candidates",
    "Visa sponsorship available",
])
def test_ordinary_postings_are_not_read_as_refusals(text):
    assert refuses_sponsorship(text) is False


def test_a_no_sponsorship_posting_is_rejected_for_a_visa_holder():
    verdict = assess("Senior Java Engineer",
                     "Java Spring Boot. We are unable to sponsor visas.",
                     needs_sponsorship=True)
    assert verdict.eligible is False
    assert "sponsor" in verdict.reasons[0]


def test_the_same_posting_is_fine_for_someone_not_needing_sponsorship():
    assert assess("Senior Java Engineer",
                  "Java Spring Boot. We are unable to sponsor visas.",
                  needs_sponsorship=False).eligible


# ---------------- combined ----------------


def test_a_good_posting_passes_every_check():
    verdict = assess(
        "Senior Java Developer",
        "Spring Boot microservices, REST APIs, Kubernetes. 4+ years of "
        "experience. Remote US. Background check required.",
    )
    assert verdict.eligible is True
    assert verdict.reasons == []
    assert verdict.years_required == 4


def test_every_failing_reason_is_reported_not_just_the_first():
    """A posting can fail several ways, and the log should say so."""
    verdict = assess(
        "Data Scientist",
        "Python and PyTorch. 12+ years of experience. TS/SCI clearance "
        "required. US citizens only. We cannot sponsor visas.",
    )
    assert verdict.eligible is False
    assert len(verdict.reasons) == 5


# ---------------- postings reserved for another group ----------------
#
# Some employers post roles open only to persons with disabilities. Those are
# genuinely restricted and applying wastes everyone's time. The difficulty is
# that nearly every posting mentions disability in its equal-opportunity
# statement, and reading that as a restriction would reject the whole market.


@pytest.mark.parametrize("text", [
    "This position is reserved for persons with disabilities.",
    "PWD only. Candidates must hold a valid disability certificate.",
    "This role is exclusively for candidates with disabilities.",
    "Open only to persons with disabilities.",
    "This vacancy is restricted to people with disabilities.",
    "Vaga exclusiva para PcD",
])
def test_genuinely_reserved_postings_are_detected(text):
    from engine.eligibility import reserved_for_other_group

    assert reserved_for_other_group(text) != ""


@pytest.mark.parametrize("text", [
    "We are an equal opportunity employer and do not discriminate on the basis "
    "of race, religion, gender, or disability.",
    "Qualified applicants receive consideration without regard to disability status.",
    "We provide reasonable accommodation to individuals with disabilities during "
    "the interview process.",
    "We encourage applications from all backgrounds, including people with disabilities.",
    "EEO/AA employer: minorities, women, protected veterans, individuals with disabilities.",
    "Affirmative action employer. All applicants with disabilities are welcome to apply.",
])
def test_equal_opportunity_boilerplate_is_not_a_restriction(text):
    """This appears on almost every posting and says the opposite."""
    from engine.eligibility import reserved_for_other_group

    assert reserved_for_other_group(text) == ""


def test_a_reserved_posting_is_rejected_by_assess():
    verdict = assess(
        "Senior Java Developer",
        "Java Spring Boot microservices. 4+ years. This position is reserved "
        "for persons with disabilities.",
    )
    assert verdict.eligible is False
    assert "reserved" in verdict.reasons[0]


def test_an_ordinary_posting_with_an_eeo_statement_still_passes():
    """The common case: a normal role whose footer mentions disability."""
    verdict = assess(
        "Senior Java Developer",
        "Java Spring Boot microservices. 4+ years of experience. Remote US. "
        "We are an equal opportunity employer and do not discriminate on the "
        "basis of race, gender, age, or disability status.",
    )
    assert verdict.eligible is True


# ---------------- seniority by title ----------------
#
# Two Staff roles cleared a six-year ceiling because their requirement was
# phrased in a way the year parser did not recognise. The title is a more
# reliable signal than years text, which postings write inconsistently.


@pytest.mark.parametrize("title", [
    "Staff Software Engineer",
    "Senior Staff Software Engineer",
    "Principal Engineer",
    "Distinguished Engineer",
    "Software Architect",
    "Engineering Manager",
    "Director of Engineering",
    "Head of Platform",
])
def test_titles_above_the_target_level_are_rejected(title):
    from engine.eligibility import above_target_level

    assert above_target_level(title) != ""


@pytest.mark.parametrize("title", [
    "Senior Software Engineer",
    "Java Developer",
    "Senior Backend Engineer",
    "Software Engineer II",
    "Senior Full Stack Developer",
    "SDE1 Back-End Engineer",
    "Backend Engineer, Payments",
])
def test_target_level_titles_are_kept(title):
    from engine.eligibility import above_target_level

    assert above_target_level(title) == ""


def test_staff_in_the_description_does_not_reject_a_senior_role():
    """"staffing" and "staff members" appear in descriptions and mean
    something else entirely; only the title is checked."""
    verdict = assess(
        "Senior Java Engineer",
        "Java Spring Boot. 4+ years. You will work with our staff across teams "
        "and partner with staffing to grow the group.",
    )
    assert verdict.eligible is True


def test_a_staff_role_is_rejected_by_title_even_without_a_years_statement():
    verdict = assess("Staff Software Engineer", "Java Spring Boot microservices.")
    assert verdict.eligible is False
    assert "above the target level" in verdict.reasons[0]


def test_the_seniority_filter_can_be_turned_off():
    assert assess("Staff Software Engineer", "Java Spring Boot.",
                  exclude_above_level=False).eligible is True
