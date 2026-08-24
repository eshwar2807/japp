"""Find companies that are hiring, then read their real postings.

Two stages, deliberately separated:

  1. Claude searches the web for companies matching your criteria and returns
     their ATS board URLs. This is judgement work - "who is hiring senior
     backend engineers in Austin" - and the web is the only place to get it.

  2. Every actual posting is then fetched from that company's own board API.

The split matters: the model never supplies a job. It supplies a *company*,
and the employer's own API supplies the openings. So a hallucinated posting is
structurally impossible - if the board does not return it, it does not exist.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import settings
from engine.llm import request_params
from engine.boards import (
    BoardError,
    Posting,
    dedupe,
    detect_board,
    fetch_board,
    matches,
    resolve_board,
)

log = logging.getLogger(__name__)

#: Server-side web search. Requires Opus/Sonnet tier - Haiku cannot run it,
#: which is why discovery does not use the cheap bulk model.
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 8}

MAX_PAUSE_TURNS = 4


class DiscoveryCriteria(BaseModel):
    """What you are looking for."""

    model_config = ConfigDict(extra="ignore")

    titles: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    seniority: str = ""
    industries: list[str] = Field(default_factory=list)
    company_size: str = ""
    remote_only: bool = False
    exclude_companies: list[str] = Field(default_factory=list)
    exclude_title_terms: list[str] = Field(
        default_factory=lambda: ["intern", "internship", "contract", "unpaid"]
    )
    max_companies: int = Field(default=15, ge=1, le=40)
    max_postings: int = Field(default=100, ge=1, le=500)

    def describe(self) -> str:
        parts = []
        if self.seniority:
            parts.append(self.seniority)
        parts.append(" / ".join(self.titles) if self.titles else "software roles")
        if self.locations:
            parts.append("in " + ", ".join(self.locations))
        if self.remote_only:
            parts.append("(remote only)")
        if self.industries:
            parts.append("industries: " + ", ".join(self.industries))
        if self.company_size:
            parts.append(f"company size: {self.company_size}")
        return " ".join(parts)


class CompanySuggestion(BaseModel):
    """A company Claude believes is hiring, with its board."""

    model_config = ConfigDict(extra="ignore")

    name: str
    careers_url: str = Field(
        default="", description="Public job-board URL, e.g. boards.greenhouse.io/acme"
    )
    board: str = Field(default="", description="greenhouse | lever | ashby | unknown")
    board_slug: str = Field(default="", description="Company identifier on that board.")
    why: str = Field(default="", description="One line on why this fits the criteria.")

    @field_validator("board", mode="after")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return (value or "").strip().lower()

    def resolve(self) -> tuple[str, str] | None:
        """Best available (board, slug), preferring what the URL actually says."""
        detected = detect_board(self.careers_url)
        if detected:
            return detected
        if self.board in ("greenhouse", "lever", "ashby") and self.board_slug:
            return self.board, self.board_slug.strip()
        return None


class CompanySearchResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    companies: list[CompanySuggestion] = Field(default_factory=list)
    notes: str = ""


DISCOVERY_SYSTEM = """\
You find companies that are currently hiring, for a job seeker's automated \
pipeline.

Use web search to identify real companies matching the criteria, then return \
each one's public job-board URL.

Rules:
- Return COMPANIES and their board URLs. Do not return individual job postings; \
those are fetched from each board's API afterwards, and anything you invent \
would simply fail to appear.
- Strongly prefer companies on Greenhouse (boards.greenhouse.io/NAME), Lever \
(jobs.lever.co/NAME) or Ashby (jobs.ashbyhq.com/NAME). Those have public APIs \
this pipeline can read. A company whose board cannot be identified is of no use.
- `board_slug` is the identifier in the URL, not the display name: \
boards.greenhouse.io/stripe -> "stripe".
- Only companies you have evidence are actively hiring. Do not pad the list.
- Do not repeat any company in the exclusion list.
- If you cannot find the board URL for a company, leave it out rather than \
guessing at a slug."""


class DiscoveryEngine:
    def __init__(
        self,
        client: Any | None = None,
        model: str = "",
        api_key: str | None = None,
        on_usage: Any = None,
    ) -> None:
        self.model = model or settings.LLM_MODEL_DISCOVERY
        self.api_key = api_key
        self.on_usage = on_usage
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            key = self.api_key or settings.ANTHROPIC_API_KEY
            self._client = anthropic.Anthropic(**({"api_key": key} if key else {})).messages
        return self._client

    # ---------------- stage 1: companies ----------------

    def find_companies(
        self, criteria: DiscoveryCriteria, already_applied: Sequence[str] = ()
    ) -> CompanySearchResult:
        exclusions = sorted({*criteria.exclude_companies, *already_applied})
        prompt = (
            f"Criteria: {criteria.describe()}\n\n"
            f"Find up to {criteria.max_companies} companies currently hiring for this.\n"
        )
        if exclusions:
            prompt += (
                "\nAlready applied to or explicitly excluded - do not return these:\n"
                + ", ".join(exclusions[:120])
                + "\n"
            )
        prompt += "\nReturn the companies and their job-board URLs."

        response = self._call(prompt)
        result = response.parsed_output
        if result is None:
            raise RuntimeError("Discovery returned no structured output.")
        return result

    def _call(self, prompt: str) -> Any:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        # A server-side search can pause the turn; continue it rather than
        # treating the partial response as the answer.
        for _ in range(MAX_PAUSE_TURNS):
            response = self.client.parse(
                model=self.model,
                max_tokens=settings.LLM_MAX_TOKENS,
                system=DISCOVERY_SYSTEM,
                messages=messages,
                tools=[WEB_SEARCH_TOOL],
                output_format=CompanySearchResult,
                **request_params(self.model, settings.LLM_EFFORT),
            )
            if self.on_usage is not None:
                try:
                    self.on_usage("discovery", response)
                except Exception:
                    log.exception("Usage callback failed during discovery")

            stop = getattr(response, "stop_reason", None)
            if stop == "refusal":
                raise RuntimeError(
                    f"Model declined the discovery request "
                    f"({getattr(response, 'stop_details', None)})."
                )
            if stop != "pause_turn":
                return response
            messages.append({"role": "assistant", "content": response.content})

        return response

    # ---------------- stage 2: real postings ----------------

    def rank_by_fit(
        self, postings: list[Posting], profile: Any, target: float = 0.0
    ) -> tuple[list[Posting], list[tuple[Posting, float]]]:
        """Order postings by how much of each one the profile already covers.

        Board postings arrive with their full text, so this costs nothing: no
        LLM call, no network. Postings whose stated requirements the profile
        largely lacks are separated out rather than queued, because tailoring
        cannot invent the missing half.

        Returns (viable, rejected_with_scores).
        """
        from engine.ats_optimizer import estimate_ceiling
        from engine.eligibility import assess

        from config import settings as _settings

        needs_sponsorship = str(
            (profile.legal or {}).get("requires_sponsorship_now_or_future", "")
        ).strip().lower().startswith("y")

        scored = []
        for posting in postings:
            verdict = assess(
                posting.title, posting.description,
                require_java=_settings.REQUIRE_JAVA,
                max_years=_settings.MAX_YEARS_REQUIRED,
                needs_sponsorship=needs_sponsorship,
                can_obtain_clearance=_settings.CAN_OBTAIN_CLEARANCE,
            )
            if not verdict.eligible:
                log.info("Skipping %s: %s", posting.title[:40], "; ".join(verdict.reasons))
                continue
            scored.append((posting, estimate_ceiling(posting.description, profile)))

        scored.sort(key=lambda pair: -pair[1])
        viable = [p for p, score in scored if score >= target]
        return viable, scored

    def collect_postings(
        self,
        companies: Sequence[CompanySuggestion],
        criteria: DiscoveryCriteria,
        seen_keys: set[str] | None = None,
    ) -> tuple[list[Posting], list[str]]:
        """Fetch each company's board. Returns (postings, problems)."""
        collected: list[Posting] = []
        problems: list[str] = []

        for company in companies:
            # The model guesses both slug and provider, and gets the provider
            # wrong often enough to matter: whole boards were discarded as 404s
            # for being offered as Greenhouse when they are on Ashby.
            resolved = resolve_board(company.name, company.careers_url,
                                     company.board_slug)
            if resolved is None:
                problems.append(f"{company.name}: no board found on any provider")
                continue
            board, slug, postings = resolved

            kept = [
                p for p in postings
                if matches(p, criteria.titles, criteria.locations,
                           criteria.exclude_title_terms)
                and not (criteria.remote_only and "remote" not in (p.location or "").lower())
            ]
            for posting in kept:
                posting.company = company.name or posting.company
            collected.extend(kept)
            log.info("%s (%s/%s): %d of %d postings match",
                     company.name, board, slug, len(kept), len(postings))

        unique = dedupe(collected, seen_keys)
        return unique[: criteria.max_postings], problems

    # ---------------- orchestration ----------------

    def run(
        self,
        criteria: DiscoveryCriteria,
        already_applied: Sequence[str] = (),
        seen_keys: set[str] | None = None,
        profile: Any | None = None,
        min_estimated_fit: float = 0.0,
    ) -> dict[str, Any]:
        """Find companies, read their boards, and keep what could clear the bar.

        Ranking by estimated fit costs nothing - board postings arrive with
        their full text, so it is pure local string work. Doing it here means
        the postings that reach the queue are the plausible ones, rather than
        every opening the boards happen to list.
        """
        search = self.find_companies(criteria, already_applied)
        postings, problems = self.collect_postings(search.companies, criteria, seen_keys)

        scored: list[tuple[Posting, float]] = []
        if profile is not None and postings:
            postings, scored = self.rank_by_fit(postings, profile, min_estimated_fit)
            log.info("%d of %d postings estimated at or above %.0f%% fit",
                     len(postings), len(scored), min_estimated_fit)

        return {
            "companies": search.companies,
            "postings": postings,
            "problems": problems,
            "notes": search.notes,
            "scored": scored,
        }
