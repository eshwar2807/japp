"""Request parameters must follow the model, not the code path.

This reproduces a live failure: resume import ran on the bulk tier but sent
`thinking={"type": "adaptive"}`, which Haiku 4.5 rejects outright —

    400 invalid_request_error: adaptive thinking is not supported on this model

Three of the four call sites had the same bug, so the entire bulk pipeline was
broken. These tests exercise each engine with a bulk-tier model and assert the
outgoing kwargs are actually acceptable to it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from config import settings
from engine.llm import (
    request_params,
    normalize_effort,
    supports_adaptive_thinking,
    supports_effort,
)

BULK = "claude-haiku-4-5"
TIER1 = "claude-opus-5"


class Recorder:
    """Captures the kwargs an engine would send."""

    def __init__(self, parsed):
        self.parsed = parsed
        self.kwargs = {}

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(parsed_output=self.parsed, stop_reason="end_turn",
                               content=[], usage=None)


# ---------------- capability table ----------------


@pytest.mark.parametrize("model", [
    "claude-opus-5", "claude-sonnet-5", "claude-opus-4-8",
    "claude-opus-4-6", "claude-fable-5",
])
def test_current_models_take_adaptive_thinking(model):
    assert supports_adaptive_thinking(model) is True
    assert request_params(model)["thinking"] == {"type": "adaptive"}


@pytest.mark.parametrize("model", [
    "claude-haiku-4-5", "claude-sonnet-4-5", "claude-3-5-sonnet-20241022",
])
def test_older_models_get_no_thinking_parameter(model):
    """Sending one is a 400, so it must be absent — not disabled, absent."""
    assert supports_adaptive_thinking(model) is False
    assert "thinking" not in request_params(model)


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-sonnet-4-5"])
def test_models_without_effort_get_no_output_config(model):
    assert supports_effort(model) is False
    assert "output_config" not in request_params(model, "high")


def test_xhigh_is_clamped_on_models_that_predate_it():
    assert normalize_effort("claude-opus-4-6", "xhigh") == "high"
    assert normalize_effort("claude-opus-5", "xhigh") == "xhigh"


def test_an_unknown_effort_is_dropped_rather_than_sent():
    assert normalize_effort("claude-opus-5", "turbo") is None


def test_an_unknown_model_is_treated_conservatively():
    """Better a default-mode request than a 400 on an unrecognised id."""
    assert request_params("claude-something-unreleased") == {}


# ---------------- every call site ----------------


def test_resume_import_on_the_bulk_tier_sends_no_adaptive_thinking():
    from engine.resume_import import (
        ImportedBasics,
        ImportedCredentials,
        ImportedHistory,
        ResumeImporter,
    )

    class BySchema(Recorder):
        def parse(self, **kwargs):
            schema = kwargs["output_format"]
            self.parsed = {ImportedBasics: ImportedBasics(full_name="Ada"),
                           ImportedHistory: ImportedHistory(),
                           ImportedCredentials: ImportedCredentials()}[schema]
            return super().parse(**kwargs)

    recorder = BySchema(None)
    ResumeImporter(client=recorder, model=BULK).parse("x" * 400)

    assert "thinking" not in recorder.kwargs
    assert "output_config" not in recorder.kwargs
    assert recorder.kwargs["model"] == BULK


def test_answer_resolver_on_the_bulk_tier_sends_no_adaptive_thinking(monkeypatch):
    from engine.answer_resolver import LLMAnswerResolver, ResolverBatch
    from engine.schemas import MasterProfile

    profile = MasterProfile.model_validate({
        "contact": {"full_name": "Ada", "email": "a@b.com"}})
    recorder = Recorder(ResolverBatch(answers=[]))
    LLMAnswerResolver(client=recorder, model=BULK).resolve(
        [{"question": "How many years of Go?"}], profile, {})

    assert "thinking" not in recorder.kwargs


def test_tailoring_on_the_bulk_tier_sends_no_adaptive_thinking():
    from engine.ats_optimizer import ATSOptimizer
    from engine.schemas import JDKeywords, MasterProfile

    profile = MasterProfile.model_validate({
        "contact": {"full_name": "Ada", "email": "a@b.com"}})
    recorder = Recorder(JDKeywords(role_title="Backend Engineer"))
    ATSOptimizer(client=recorder, model=BULK, profile=profile).extract_keywords("A job.")

    assert "thinking" not in recorder.kwargs
    assert "output_config" not in recorder.kwargs


def test_tailoring_on_the_priority_tier_still_gets_thinking_and_effort():
    from engine.ats_optimizer import ATSOptimizer
    from engine.schemas import JDKeywords, MasterProfile

    profile = MasterProfile.model_validate({
        "contact": {"full_name": "Ada", "email": "a@b.com"}})
    recorder = Recorder(JDKeywords(role_title="Backend Engineer"))
    ATSOptimizer(client=recorder, model=TIER1, profile=profile).extract_keywords("A job.")

    assert recorder.kwargs["thinking"] == {"type": "adaptive"}
    assert recorder.kwargs["output_config"]["effort"] == settings.LLM_EFFORT


def test_discovery_gets_thinking_because_it_runs_on_the_opus_tier():
    from engine.discovery import CompanySearchResult, DiscoveryCriteria, DiscoveryEngine

    recorder = Recorder(CompanySearchResult(companies=[]))
    DiscoveryEngine(client=recorder).find_companies(DiscoveryCriteria(titles=["Backend"]))

    assert recorder.kwargs["thinking"] == {"type": "adaptive"}
    assert recorder.kwargs["tools"][0]["type"].startswith("web_search_")


def test_no_engine_hardcodes_adaptive_thinking():
    """The parameter must come from the helper, or the next tier change
    reintroduces exactly this bug."""
    import pathlib

    engine_dir = pathlib.Path(__file__).resolve().parent.parent / "engine"
    offenders = [
        path.name for path in engine_dir.glob("*.py")
        if path.name != "llm.py" and '"adaptive"' in path.read_text()
    ]
    assert offenders == [], f"hardcoded adaptive thinking in: {offenders}"


def test_the_configured_bulk_model_accepts_what_the_pipeline_sends():
    """Whatever the bulk tier is set to, the kwargs must be valid for it."""
    params = request_params(settings.LLM_MODEL_BULK, settings.LLM_EFFORT)
    if not supports_adaptive_thinking(settings.LLM_MODEL_BULK):
        assert "thinking" not in params
    if not supports_effort(settings.LLM_MODEL_BULK):
        assert "output_config" not in params
