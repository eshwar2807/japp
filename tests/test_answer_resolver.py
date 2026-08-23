"""The resolution ladder: rules -> memory -> inference -> ask a human.

The safety property under test: legal and identity questions never reach the
inference step, and an ungrounded answer is discarded rather than typed into a
job application.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.answer_resolver import (
    MIN_RESOLVE_CONFIDENCE,
    LLMAnswerResolver,
    ResolvedAnswer,
    ResolverBatch,
    must_not_infer,
)
from engine.schemas import MasterProfile
from engine.screener_mapper import AnswerSource, FieldSpec, FieldType, ScreenerMapper


@pytest.fixture()
def prof():
    return MasterProfile.model_validate({
        "contact": {"full_name": "Ada Lovelace", "email": "ada@example.com",
                    "phone": "+1-555-010-0100",
                    "location": {"city": "Austin", "state": "TX"}},
        "skills": {"hard": ["Python", "Distributed Systems"], "tooling": ["Kubernetes"]},
        "experience": [{"company": "Analytical Engines", "title": "Principal Engineer",
                        "start_date": "2015-01", "is_current": True,
                        "bullets": ["Cut p99 latency 60% across 14 services."]}],
        "legal": {"work_authorization_us": "Yes",
                  "requires_sponsorship_now_or_future": "No"},
    })


class StubResolver:
    """Returns canned answers without an API call."""

    def __init__(self, answers: dict[str, ResolvedAnswer] | None = None):
        self.answers = answers or {}
        self.seen: list[list[dict]] = []

    def resolve(self, questions, profile, previous_answers=None):
        self.seen.append(questions)
        return self.answers


def field(label, **kw):
    return FieldSpec(label=label, selector=f"#{abs(hash(label)) % 9999}", **kw)


# ---------------- the never-infer guard ----------------


@pytest.mark.parametrize("question", [
    "Are you legally authorized to work in the United States?",
    "Will you now or in the future require visa sponsorship?",
    "Do you hold an active security clearance?",
    "What is your desired salary?",
    "Gender",
    "Veteran status",
    "Do you have a disability?",
    "Have you ever been convicted of a felony?",
    "Date of birth",
])
def test_legal_and_identity_questions_are_never_inferred(question):
    assert must_not_infer(question) is True


@pytest.mark.parametrize("question", [
    "How many years of Go experience do you have?",
    "Describe a time you scaled a system under load",
    "Which frameworks have you used in production?",
    "Why do you want to work here?",
])
def test_ordinary_questions_may_be_inferred(question):
    assert must_not_infer(question) is False


def test_blocked_questions_are_not_even_sent_to_the_model(prof):
    """They must not appear in the prompt, let alone be answered."""
    sent = {}

    class Recorder:
        def parse(self, **kwargs):
            sent["prompt"] = kwargs["messages"][0]["content"]
            return SimpleNamespace(parsed_output=ResolverBatch(answers=[]),
                                   stop_reason="end_turn")

    LLMAnswerResolver(client=Recorder()).resolve(
        [{"question": "Are you legally authorized to work in the US?"},
         {"question": "How many years of Python?"}],
        prof, {})

    assert "legally authorized" not in sent["prompt"]
    assert "years of Python" in sent["prompt"]


def test_an_inferred_legal_answer_is_discarded_even_if_returned(prof):
    """Defence in depth: a model that answers anyway does not get used."""

    class Rogue:
        def parse(self, **kwargs):
            return SimpleNamespace(
                parsed_output=ResolverBatch(answers=[
                    ResolvedAnswer(question="Do you require visa sponsorship?",
                                   can_answer=True, answer="No",
                                   grounding="seems likely", confidence=1.0)]),
                stop_reason="end_turn")

    resolved = LLMAnswerResolver(client=Rogue()).resolve(
        [{"question": "How many years of Python?"}], prof, {})
    assert resolved == {}


# ---------------- usability rules ----------------


def test_an_ungrounded_answer_is_not_usable():
    answer = ResolvedAnswer(question="q", can_answer=True, answer="4",
                            grounding="", confidence=1.0)
    assert answer.usable is False


def test_a_low_confidence_answer_is_not_usable():
    answer = ResolvedAnswer(question="q", can_answer=True, answer="4",
                            grounding="Started 2015", confidence=MIN_RESOLVE_CONFIDENCE - 0.01)
    assert answer.usable is False


def test_cannot_answer_is_respected():
    assert ResolvedAnswer(question="q", can_answer=False, answer="maybe",
                          grounding="x", confidence=1.0).usable is False


def test_a_resolver_crash_falls_back_to_escalation(prof):
    class Broken:
        def parse(self, **kwargs):
            raise RuntimeError("API down")

    assert LLMAnswerResolver(client=Broken()).resolve(
        [{"question": "How many years of Python?"}], prof, {}) == {}


# ---------------- the ladder ----------------


def test_rules_beat_memory_and_inference(prof):
    """A profile fact is authoritative; nothing downstream may override it."""
    mapper = ScreenerMapper(
        prof,
        remembered={"Are you legally authorized to work in the US?": "No"},
        resolver=StubResolver(),
    )
    answer = mapper.map_field(field("Are you legally authorized to work in the US?"))
    assert answer.source is AnswerSource.RULE
    assert answer.value == "Yes"


def test_memory_beats_a_tailored_answer(prof):
    """What you typed yourself outranks what the model composed."""
    mapper = ScreenerMapper(
        prof,
        screener_answers={"How many years of Go?": "2"},
        remembered={"How many years of Go?": "4"},
    )
    answer = mapper.map_field(field("How many years of Go?"))
    assert answer.source is AnswerSource.MEMORY
    assert answer.value == "4"


def test_a_remembered_answer_is_reused_on_the_next_application(prof):
    mapper = ScreenerMapper(prof, remembered={"How many years of Go experience?": "4"})
    answer = mapper.map_field(field("Years of experience with Go?"))
    assert answer.source is AnswerSource.MEMORY
    assert not answer.needs_human


def test_inference_catches_what_lexical_matching_misses(prof):
    """The gap this module exists for: same question, unrecognisable wording."""
    plain = ScreenerMapper(prof, remembered={"How many years of Go experience?": "4"})
    assert plain.map_field(field("Describe your Go background", required=True)).needs_human

    resolver = StubResolver({"Describe your Go background": ResolvedAnswer(
        question="Describe your Go background", can_answer=True,
        answer="Four years of Go in production.",
        grounding="previous answer: 4 years of Go", confidence=0.9)})
    smart = ScreenerMapper(prof, remembered={"How many years of Go experience?": "4"},
                           resolver=resolver)

    autofill, escalations = smart.map_form([field("Describe your Go background", required=True)])
    assert escalations == []
    answer = list(autofill.values())[0]
    assert answer.source is AnswerSource.INFERRED
    assert "previous answer" in answer.reason


def test_unanswerable_fields_still_reach_a_human(prof):
    resolver = StubResolver({"Describe your Rust background": ResolvedAnswer(
        question="Describe your Rust background", can_answer=False, confidence=0.0)})
    mapper = ScreenerMapper(prof, resolver=resolver)

    autofill, escalations = mapper.map_form([field("Describe your Rust background", required=True)])
    assert autofill == {}
    assert len(escalations) == 1


def test_the_resolver_is_called_once_per_form_not_once_per_field(prof):
    """Cost control: a hundred applications must not mean a thousand calls."""
    resolver = StubResolver()
    mapper = ScreenerMapper(prof, resolver=resolver)
    mapper.map_form([field(f"Obscure question {i}", required=True) for i in range(6)])

    assert len(resolver.seen) == 1
    assert len(resolver.seen[0]) == 6


def test_an_inferred_answer_still_has_to_match_the_available_options(prof):
    resolver = StubResolver({"Preferred working style": ResolvedAnswer(
        question="Preferred working style", can_answer=True, answer="Somewhere quiet",
        grounding="profile", confidence=0.95)})
    mapper = ScreenerMapper(prof, resolver=resolver)

    autofill, escalations = mapper.map_form([
        field("Preferred working style", field_type=FieldType.SELECT,
              options=["Remote", "Hybrid", "On-site"], required=True)])
    assert autofill == {}
    assert "does not correspond" in escalations[0].reason


def test_no_resolver_configured_behaves_as_before(prof):
    mapper = ScreenerMapper(prof)
    autofill, escalations = mapper.map_form([field("Obscure question", required=True)])
    assert autofill == {} and len(escalations) == 1


# ---------------- years-of-experience precision ----------------


def test_generic_years_question_uses_total_career_length(prof):
    """Answerable from employment dates."""
    answer = ScreenerMapper(prof).map_field(field("How many years of experience do you have?"))
    assert answer.source is AnswerSource.RULE
    assert answer.value.isdigit() and int(answer.value) >= 10


@pytest.mark.parametrize("question", [
    "How many years of Go experience do you have?",
    "Years of experience with Kubernetes",
    "How many years of Python have you used?",
])
def test_technology_specific_years_questions_do_not_get_career_length(prof, question):
    """Regression: these answered with total career length — a wrong number on
    a real application. They must fall through to memory or inference."""
    answer = ScreenerMapper(prof).map_field(field(question, required=True))
    assert answer.source is not AnswerSource.RULE
    assert answer.needs_human


def test_a_technology_specific_years_question_uses_a_remembered_answer(prof):
    mapper = ScreenerMapper(prof, remembered={"How many years of Go experience?": "4"})
    answer = mapper.map_field(field("Years of experience with Go?"))
    assert answer.source is AnswerSource.MEMORY
    assert answer.value == "4"
