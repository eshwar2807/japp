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
    ("No number given here", None),
])
def test_years_required_is_read_from_the_posting(text, expected):
    assert years_required(text) == expected


def test_the_lowest_stated_requirement_wins():
    """"5+ years backend, 8+ years leadership" gates an IC at five; reading it
    as eight would discard a viable role."""
    assert years_required("5+ years of backend experience and "
                          "8+ years of leadership experience") == 5


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
