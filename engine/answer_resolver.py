"""Last resort before asking a human: can this field be answered honestly?

The screener mapper resolves fields in a ladder:

  1. Rule table       - a fact from master_profile. Deterministic, authoritative.
  2. Remembered answer - something you already answered, matched lexically.
  3. This module       - the model reads your profile plus everything you have
                         previously answered, and answers only if the facts
                         genuinely support it.
  4. Escalate          - ask you.

Step 3 exists because steps 1 and 2 are literal. "Describe your Go background"
scores 37% against "How many years of Go experience?" and escalates, even
though you answered it last week. A model reading both sees the connection.

Two hard constraints:

* **Legal and identity questions never reach this module.** Work authorisation,
  sponsorship, clearance and EEO answers come from the profile or from you,
  never from inference. A plausible guess there is a false statement on an
  application.
* **No fabrication.** The model must cite the fact it relied on. An answer with
  no grounding is discarded and the field escalates instead.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from config import settings

log = logging.getLogger(__name__)

#: Below this the answer is not used, and the field goes to a human.
MIN_RESOLVE_CONFIDENCE = 0.75

#: Questions that must never be answered by inference, whatever the profile
#: seems to imply. These are legal statements, and the rule table owns them.
NEVER_INFER = re.compile(
    r"(authori[sz]ed|eligible|sponsor|visa|work\s+permit|clearance|citizen|"
    r"felony|conviction|background\s+check|drug\s+(test|screen)|"
    r"gender|race|ethnicity|hispanic|latino|veteran|disab(ility|led)|"
    r"date\s+of\s+birth|age|ssn|social\s+security|salary|compensation)",
    re.IGNORECASE,
)


def must_not_infer(question: str) -> bool:
    """True when a question is off-limits to inference."""
    return bool(NEVER_INFER.search(question or ""))


class ResolvedAnswer(BaseModel):
    """One field the model attempted to answer."""

    model_config = ConfigDict(extra="ignore")

    question: str
    can_answer: bool = Field(
        description="True only when the profile or a previous answer genuinely supports this."
    )
    answer: str = Field(default="", description="Empty when can_answer is false.")
    grounding: str = Field(
        default="",
        description="The specific fact relied on, quoted. Empty means ungrounded.",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def usable(self) -> bool:
        return (
            self.can_answer
            and bool(self.answer.strip())
            and bool(self.grounding.strip())
            and self.confidence >= MIN_RESOLVE_CONFIDENCE
        )


class ResolverBatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answers: list[ResolvedAnswer] = Field(default_factory=list)


RESOLVER_SYSTEM = """\
You answer job-application form fields on a candidate's behalf, using only \
facts they have already given you.

You receive the candidate's PROFILE, their PREVIOUS_ANSWERS to earlier \
application questions, and a list of QUESTIONS from the form in front of them.

For each question:
- Set `can_answer` true ONLY if the profile or a previous answer genuinely \
supports an answer. Quote the supporting fact in `grounding`.
- If you are inferring, extrapolating, or would be making a reasonable-sounding \
guess, set `can_answer` false. A human will be asked instead, which is the \
correct outcome. Leaving a question to a human costs a moment; inventing an \
answer puts a false statement on a job application.
- Never invent employers, dates, degrees, metrics, tools or years of experience.
- Years of experience may be computed from employment dates in the profile, or \
reused from a previous answer to the same question in different words.
- Open-ended prompts ("describe a time when...", "why this company") may be \
composed ONLY from accomplishments already in the profile. If the profile has \
nothing relevant, say you cannot answer.
- Match the form's expected format: a number for a years field, Yes/No for a \
yes/no question, one of the listed options when options are given.
- `confidence` reflects how directly the facts support the answer: 1.0 for a \
value copied straight from the profile, lower when composed from several facts."""


class LLMAnswerResolver:
    """Batched resolver. One call per form, not one per field."""

    def __init__(
        self,
        client: Any | None = None,
        model: str = "",
        api_key: str | None = None,
        on_usage: Any = None,
    ) -> None:
        # Runs on the cheap tier: it is called once per application, and the
        # task is extraction rather than composition.
        self.model = model or settings.LLM_MODEL_BULK
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

    def resolve(
        self,
        questions: Sequence[dict[str, Any]],
        profile: Any,
        previous_answers: dict[str, str] | None = None,
    ) -> dict[str, ResolvedAnswer]:
        """Answer what can honestly be answered. Returns {question: answer}.

        Questions that must not be inferred are dropped before the call, so
        they are never even shown to the model.
        """
        import json

        eligible = [q for q in questions if not must_not_infer(q.get("question", ""))]
        skipped = len(questions) - len(eligible)
        if skipped:
            log.info("%d field(s) held back from inference as legal/identity", skipped)
        if not eligible:
            return {}

        prompt = (
            "<PROFILE>\n"
            + json.dumps(
                profile.verified_facts if hasattr(profile, "verified_facts") else profile,
                indent=2, default=str,
            )
            + "\n</PROFILE>\n\n<PREVIOUS_ANSWERS>\n"
            + json.dumps(previous_answers or {}, indent=2)
            + "\n</PREVIOUS_ANSWERS>\n\n<QUESTIONS>\n"
            + json.dumps(eligible, indent=2)
            + "\n</QUESTIONS>\n\nAnswer only what the facts support."
        )

        try:
            response = self.client.parse(
                model=self.model,
                max_tokens=settings.LLM_MAX_TOKENS,
                system=RESOLVER_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                output_format=ResolverBatch,
                thinking={"type": "adaptive"},
            )
        except Exception:
            # A resolver failure must never fail the application; the fields
            # simply escalate as they would have before this existed.
            log.exception("Answer resolver failed; falling back to escalation")
            return {}

        if self.on_usage is not None:
            try:
                self.on_usage("resolve_answers", response)
            except Exception:
                log.exception("Usage callback failed in the resolver")

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            return {}

        resolved: dict[str, ResolvedAnswer] = {}
        for answer in parsed.answers:
            # Guard again on the way out: a model that answered a legal
            # question anyway does not get to have that answer used.
            if must_not_infer(answer.question):
                log.warning("Discarding inferred answer to a legal question: %s",
                            answer.question[:60])
                continue
            resolved[answer.question] = answer
        return resolved
