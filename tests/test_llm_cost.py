"""Keeping the discovery bill in proportion to what discovery is worth.

Measured over one day of real use:

    opus-5   / discovery         22 calls   avg 91,537 in   $14.28   (84%)
    haiku    / tailor           182 calls   avg  6,015 in    $2.26   (13%)
    haiku    / extract_keywords  68 calls   avg  3,032 in    $0.36    (2%)

Twenty-two discovery calls cost five times what two hundred and fifty
tailoring calls did, and `cache_read_tokens` was zero on all 272. The reason is
the pause-turn loop: a server-side search pauses the turn, and continuing it
resends every search result collected so far, so the same tokens are paid for
on every continuation.

Caching that prefix is the fix. Note where it does *not* apply: the minimum
cacheable prefix is 4,096 tokens on Haiku 4.5, and TAILOR_SYSTEM is about 426,
so marking the tailoring prompts would pay write premiums and never read.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from config import settings
from engine.discovery import DiscoveryCriteria, DiscoveryEngine


class Block:
    """A content block shaped like the SDK's, including a None field."""

    def __init__(self, text: str):
        self.text = text

    def model_dump(self, mode=None, exclude_none=False):
        data = {"type": "text", "text": self.text, "citations": None}
        return {k: v for k, v in data.items() if not (exclude_none and v is None)}


class PausingClient:
    """Pauses the turn twice, then answers - the shape that costs the money."""

    def __init__(self, pauses: int = 2):
        self.pauses = pauses
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        paused = len(self.calls) <= self.pauses
        from engine.discovery import CompanySearchResult

        return SimpleNamespace(
            parsed_output=None if paused else CompanySearchResult(companies=[]),
            stop_reason="pause_turn" if paused else "end_turn",
            content=[Block("a" * 200), Block("search results " * 50)],
        )


# ---------------- the breakpoint ----------------


def test_each_continued_turn_carries_a_cache_breakpoint():
    """Without this the accumulated search results are re-billed at full price
    on every continuation."""
    client = PausingClient(pauses=2)
    DiscoveryEngine(client=client).find_companies(DiscoveryCriteria(titles=["Java"]))

    assert len(client.calls) == 3, "expected two continuations"
    continued = client.calls[-1]["messages"]
    assistant_turns = [m for m in continued if m["role"] == "assistant"]
    assert assistant_turns, "nothing was appended to continue the turn"
    for turn in assistant_turns:
        assert turn["content"][-1].get("cache_control") == {"type": "ephemeral"}


def test_only_the_last_block_of_a_turn_is_marked():
    """A breakpoint per block would blow the four-breakpoint limit."""
    client = PausingClient(pauses=2)
    DiscoveryEngine(client=client).find_companies(DiscoveryCriteria(titles=["Java"]))

    for turn in (m for m in client.calls[-1]["messages"] if m["role"] == "assistant"):
        marked = [b for b in turn["content"] if "cache_control" in b]
        assert len(marked) == 1


def test_breakpoints_stay_within_the_documented_limit():
    """At most four `cache_control` blocks may appear in one request."""
    client = PausingClient(pauses=5)
    # More pauses than MAX_PAUSE_TURNS allows, so the run gives up - which is
    # fine here: what is under test is how many breakpoints accumulated.
    with pytest.raises(RuntimeError):
        DiscoveryEngine(client=client).find_companies(DiscoveryCriteria(titles=["Java"]))

    for call in client.calls:
        marked = sum(
            1
            for message in call["messages"]
            for block in (message["content"] if isinstance(message["content"], list) else [])
            if isinstance(block, dict) and "cache_control" in block
        )
        assert marked <= 4, f"{marked} breakpoints in one request"


def test_a_turn_that_never_pauses_adds_no_breakpoint():
    """One-shot discovery has no repeated prefix, so caching would only cost."""
    client = PausingClient(pauses=0)
    DiscoveryEngine(client=client).find_companies(DiscoveryCriteria(titles=["Java"]))

    assert len(client.calls) == 1
    assert all(m["role"] != "assistant" for m in client.calls[0]["messages"])


def test_serialising_a_turn_drops_empty_fields():
    """`None` fields round-tripped back into the request are rejected."""
    blocks = DiscoveryEngine._cacheable([Block("hello")])

    assert "citations" not in blocks[0]
    assert blocks[0]["text"] == "hello"


def test_an_empty_turn_does_not_raise():
    assert DiscoveryEngine._cacheable([]) == []


# ---------------- effort ----------------


def test_discovery_has_its_own_effort_setting():
    """Output tokens are the part of the bill caching cannot touch, and finding
    which companies are hiring is retrieval rather than reasoning."""
    assert settings.LLM_EFFORT_DISCOVERY != settings.LLM_EFFORT
    assert settings.LLM_EFFORT_DISCOVERY in ("low", "medium", "high", "xhigh", "max")


def test_discovery_sends_its_own_effort_not_the_global_one(monkeypatch):
    from engine.llm import request_params

    monkeypatch.setattr(settings, "LLM_EFFORT", "max")
    monkeypatch.setattr(settings, "LLM_EFFORT_DISCOVERY", "low")

    client = PausingClient(pauses=0)
    engine = DiscoveryEngine(client=client)
    engine.find_companies(DiscoveryCriteria(titles=["Java"]))

    expected = request_params(engine.model, "low")
    if "output_config" in expected:
        assert client.calls[0]["output_config"] == expected["output_config"]


def test_tailoring_effort_is_left_alone(monkeypatch):
    """Cheapening discovery must not cheapen the resumes."""
    monkeypatch.setattr(settings, "LLM_EFFORT_DISCOVERY", "low")
    assert settings.LLM_EFFORT == "high"
