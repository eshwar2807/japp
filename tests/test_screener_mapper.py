"""Module C validation: deterministic form-field mapping.

The critical property under test: legal answers always come from the master
profile, never from the LLM, and anything ambiguous escalates to a human
instead of being guessed.
"""

from __future__ import annotations

import pytest

from engine.schemas import MasterProfile
from engine.screener_mapper import (
    MIN_AUTOFILL_CONFIDENCE,
    AnswerSource,
    FieldSpec,
    FieldType,
    ScreenerMapper,
)


@pytest.fixture()
def prof():
    return MasterProfile.model_validate(
        {
            "contact": {
                "full_name": "Ada Lovelace",
                "preferred_name": "Ada",
                "email": "ada@example.com",
                "phone": "+1-555-010-0100",
                "location": {
                    "city": "Austin",
                    "state": "TX",
                    "country": "United States",
                    "postal_code": "78701",
                    "willing_to_relocate": True,
                    "remote_preference": "hybrid",
                },
                "links": {"linkedin": "https://linkedin.com/in/ada", "github": "https://github.com/ada"},
            },
            "experience": [
                {
                    "company": "Analytical Engines",
                    "title": "Principal Engineer",
                    "start_date": "2015-01",
                    "is_current": True,
                }
            ],
            "legal": {
                "work_authorization_us": "Yes",
                "requires_sponsorship_now_or_future": "No",
                "security_clearance": "None",
                "willing_to_complete_background_check": "Yes",
                "age_over_18": "Yes",
                "notice_period": "2 weeks",
                "earliest_start_date": "2026-09-01",
                "desired_salary": "185000",
                "previously_employed_here": "No",
            },
            "voluntary_disclosures": {
                "gender": "Decline to self-identify",
                "veteran_status": "Decline to self-identify",
                "disability_status": "Decline to self-identify",
                "race_ethnicity": "Decline to self-identify",
            },
            "preferences": {"how_did_you_hear_about_us": "Company careers page"},
        }
    )


@pytest.fixture()
def mapper(prof):
    return ScreenerMapper(prof, {"Why do you want to work at Acme?": "I admire the robotics work."})


def f(label, **kw):
    return FieldSpec(label=label, selector=f"#{label[:12]}", **kw)


# ---------------- identity & contact ----------------


@pytest.mark.parametrize(
    "label,expected",
    [
        ("First Name", "Ada"),
        ("Last name", "Lovelace"),
        ("Full legal name", "Ada Lovelace"),
        ("Email Address", "ada@example.com"),
        ("Phone Number", "+1-555-010-0100"),
        ("LinkedIn Profile", "https://linkedin.com/in/ada"),
        ("GitHub URL", "https://github.com/ada"),
        ("City", "Austin"),
        ("State/Province", "TX"),
        ("Zip Code", "78701"),
        ("Current Company", "Analytical Engines"),
        ("Current Title", "Principal Engineer"),
    ],
)
def test_contact_fields_resolve_from_profile(mapper, label, expected):
    answer = mapper.map_field(f(label))
    assert answer.value == expected
    assert answer.source is AnswerSource.RULE
    assert not answer.needs_human


# ---------------- legal answers ----------------


@pytest.mark.parametrize(
    "label,expected",
    [
        ("Are you legally authorized to work in the United States?", "Yes"),
        ("Will you now or in the future require visa sponsorship?", "No"),
        ("Do you have an active security clearance?", "None"),
        ("Are you willing to complete a background check?", "Yes"),
        ("Are you over 18 years of age?", "Yes"),
        ("What is your notice period?", "2 weeks"),
        ("Earliest start date", "2026-09-01"),
        ("Desired salary", "185000"),
        ("Have you previously been employed by this company?", "No"),
        ("Are you willing to relocate?", "Yes"),
        ("How did you hear about us?", "Company careers page"),
    ],
)
def test_legal_fields_resolve_from_profile(mapper, label, expected):
    answer = mapper.map_field(f(label))
    assert answer.value == expected
    assert answer.source is AnswerSource.RULE
    assert answer.confidence == 1.0


def test_combined_auth_and_sponsorship_phrasing(mapper):
    """'Authorized to work without sponsorship' must not match the plain auth rule."""
    answer = mapper.map_field(
        f("Are you legally authorized to work in the US without sponsorship now or in the future?")
    )
    assert answer.value == "Yes"
    assert "auth_without_sponsorship" in answer.reason


def test_combined_auth_is_no_when_sponsorship_needed(prof):
    prof.legal["requires_sponsorship_now_or_future"] = "Yes"
    answer = ScreenerMapper(prof).map_field(
        f("Are you authorized to work in the United States without sponsorship?")
    )
    assert answer.value == "No"


def test_sponsorship_rule_precedes_work_authorization(mapper):
    answer = mapper.map_field(f("Do you require sponsorship to work in the US?"))
    assert answer.value == "No"


@pytest.mark.parametrize(
    "label", ["Gender", "Veteran Status", "Disability Status", "Race / Ethnicity"]
)
def test_voluntary_disclosures_decline_by_default(mapper, label):
    answer = mapper.map_field(f(label))
    assert answer.value == "Decline to self-identify"


def test_legal_answer_never_comes_from_the_llm(prof):
    """An LLM answer must not be able to override a work-authorisation question."""
    mapper = ScreenerMapper(
        prof, {"Are you legally authorized to work in the United States?": "No"}
    )
    answer = mapper.map_field(f("Are you legally authorized to work in the United States?"))
    assert answer.source is AnswerSource.RULE
    assert answer.value == "Yes"  # profile wins


def test_placeholder_values_are_not_treated_as_answers(prof):
    prof.legal["visa_status"] = "<e.g. US Citizen>"
    answer = ScreenerMapper(prof).map_field(f("What is your visa status?"))
    assert answer.needs_human


# ---------------- LLM-sourced answers ----------------


def test_posting_specific_question_uses_llm_answer(mapper):
    answer = mapper.map_field(f("Why do you want to work at Acme?", field_type=FieldType.TEXTAREA))
    assert answer.source is AnswerSource.LLM
    assert "robotics" in answer.value
    assert answer.confidence >= MIN_AUTOFILL_CONFIDENCE


def test_unrelated_question_is_escalated(mapper):
    answer = mapper.map_field(f("Describe your favourite kernel scheduler", required=True))
    assert answer.source is AnswerSource.UNMAPPED
    assert answer.needs_human
    assert "No profile rule" in answer.reason


def test_unlabelled_field_is_escalated(mapper):
    assert ScreenerMapper.map_field(mapper, FieldSpec()).needs_human


# ---------------- option snapping ----------------


def test_select_snaps_to_exact_option(mapper):
    answer = mapper.map_field(
        f("Are you legally authorized to work in the US?", field_type=FieldType.SELECT,
          options=["Yes", "No"])
    )
    assert answer.value == "Yes"
    assert not answer.needs_human


def test_select_snaps_to_fuzzy_option(mapper):
    answer = mapper.map_field(
        f("Gender", field_type=FieldType.SELECT,
          options=["Male", "Female", "I decline to self identify", "Other"])
    )
    assert answer.value == "I decline to self identify"
    assert not answer.needs_human


def test_select_escalates_when_no_option_fits(mapper):
    answer = mapper.map_field(
        f("Desired salary", field_type=FieldType.SELECT,
          options=["Under $100k", "$100k-$150k", "Over $250k"], required=True)
    )
    assert answer.needs_human
    assert "does not correspond" in answer.reason


# ---------------- whole-form behaviour ----------------


def test_map_form_splits_autofill_from_escalations(mapper):
    fields = [
        f("First Name", required=True),
        f("Email", required=True),
        f("Resume", field_type=FieldType.FILE, required=True),
        f("Describe a time you rewrote a compiler", field_type=FieldType.TEXTAREA, required=True),
        f("Optional nickname for the team wiki"),
    ]
    autofill, escalations = mapper.map_form(fields)

    assert len(autofill) == 2                              # first name + email
    assert len(escalations) == 1                           # the required unmapped one
    assert "compiler" in escalations[0].question
    assert all(a.field_type is not FieldType.FILE for a in fields if a.selector in autofill)


def test_file_fields_are_never_mapped_as_text(mapper):
    autofill, escalations = mapper.map_form([f("Resume/CV", field_type=FieldType.FILE, required=True)])
    assert autofill == {} and escalations == []
