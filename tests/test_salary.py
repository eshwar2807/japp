"""Desired salary is answered per posting, not from one static number.

The rule: midpoint of whatever range the posting states, else the figure in
your profile, else a configured default.
"""

from __future__ import annotations

import pytest

from engine.schemas import MasterProfile
from engine.screener_mapper import (
    DEFAULT_SALARY_FALLBACK,
    AnswerSource,
    FieldSpec,
    FieldType,
    ScreenerMapper,
    salary_for_posting,
    wants_salary,
)


@pytest.fixture()
def prof():
    return MasterProfile.model_validate({
        "contact": {"full_name": "Ada", "email": "a@b.com"},
        "legal": {"work_authorization_us": "Yes", "desired_salary": ""},
    })


def f(label, **kw):
    return FieldSpec(label=label, selector="#salary", **kw)


# ---------------- question detection ----------------


@pytest.mark.parametrize("question", [
    "Desired salary", "What are your salary expectations?",
    "Expected compensation", "Salary requirement",
    "What is your target compensation?", "Desired pay",
])
def test_salary_questions_are_recognised(question):
    assert wants_salary(question) is True


@pytest.mark.parametrize("question", [
    "Why do you want to work here?", "Years of experience",
    "Are you authorized to work in the US?",
])
def test_other_questions_are_not_salary_questions(question):
    assert wants_salary(question) is False


# ---------------- the rule ----------------


def test_the_midpoint_of_a_posted_range_is_used():
    value, reason = salary_for_posting(160_000, 200_000)
    assert value == "180000"
    assert "midpoint" in reason


def test_a_single_posted_figure_is_used_as_is():
    assert salary_for_posting(180_000, None)[0] == "180000"
    assert salary_for_posting(None, 180_000)[0] == "180000"


def test_the_profile_figure_is_used_when_the_posting_states_nothing():
    value, reason = salary_for_posting(None, None, "185000")
    assert value == "185000"
    assert "profile" in reason


def test_a_formatted_profile_figure_is_cleaned():
    assert salary_for_posting(None, None, "$185,000")[0] == "185000"


def test_the_default_applies_when_nothing_else_is_known():
    value, reason = salary_for_posting(None, None, "")
    assert value == str(DEFAULT_SALARY_FALLBACK)
    assert "no range" in reason


def test_an_unparseable_profile_figure_falls_back():
    assert salary_for_posting(None, None, "Negotiable")[0] == str(DEFAULT_SALARY_FALLBACK)


def test_a_transposed_range_is_normalised_not_discarded():
    """min and max the wrong way round have the same midpoint either way, so
    using the posting's real numbers beats falling back to a default."""
    assert salary_for_posting(200_000, 160_000, "")[0] == "180000"
    assert salary_for_posting(160_000, 200_000, "")[0] == "180000"


def test_the_answer_is_always_a_bare_number():
    """Forms parse this field; currency symbols and commas break them."""
    for args in [(160_000, 200_000), (None, None, "$185,000"), (None, None, "")]:
        assert salary_for_posting(*args)[0].isdigit()


# ---------------- through the mapper ----------------


def test_the_posting_range_beats_the_profile_figure(prof):
    """The whole point: the same profile answers differently per posting."""
    prof.legal["desired_salary"] = "185000"
    mapper = ScreenerMapper(prof, salary_min=200_000, salary_max=240_000)

    answer = mapper.map_field(f("Desired salary"))
    assert answer.value == "220000"
    assert answer.source is AnswerSource.RULE
    assert "midpoint" in answer.reason


def test_the_same_profile_answers_differently_for_two_postings(prof):
    low = ScreenerMapper(prof, salary_min=120_000, salary_max=140_000)
    high = ScreenerMapper(prof, salary_min=220_000, salary_max=280_000)

    assert low.map_field(f("Desired salary")).value == "130000"
    assert high.map_field(f("Desired salary")).value == "250000"


def test_a_posting_with_no_range_uses_the_default(prof):
    answer = ScreenerMapper(prof).map_field(f("Expected compensation"))
    assert answer.value == str(DEFAULT_SALARY_FALLBACK)
    assert not answer.needs_human


def test_a_per_user_fallback_overrides_the_default(prof):
    mapper = ScreenerMapper(prof, salary_fallback=175_000)
    assert mapper.map_field(f("Desired salary")).value == "175000"


def test_a_profile_preference_sets_the_fallback(prof):
    prof.preferences["desired_salary_fallback"] = 165_000
    assert ScreenerMapper(prof).map_field(f("Desired salary")).value == "165000"


def test_a_salary_dropdown_still_has_to_match_an_option(prof):
    """A band select cannot take a raw number."""
    answer = ScreenerMapper(prof, salary_min=160_000, salary_max=200_000).map_field(
        f("Desired salary", field_type=FieldType.SELECT,
          options=["Under $100k", "$100k-$150k", "$150k-$200k", "Over $200k"],
          required=True))
    assert answer.value in ("$150k-$200k", "")     # snapped, or escalated


def test_salary_is_still_never_inferred_by_the_model(prof):
    """Computing a midpoint is arithmetic on a stated range. Guessing what you
    would accept is not, and stays blocked."""
    from engine.answer_resolver import must_not_infer

    assert must_not_infer("What is your desired salary?") is True
    assert must_not_infer("Expected compensation") is True
