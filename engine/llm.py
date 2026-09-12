"""Model-aware request parameters.

Tiered models mean a request built for one model gets sent to another. Two
parameters are not portable across tiers:

* **Adaptive thinking** (``thinking={"type": "adaptive"}``) exists on the 4.6
  generation and later. Older models take ``{"type": "enabled",
  "budget_tokens": N}`` instead and reject ``adaptive`` outright:

      400 invalid_request_error: adaptive thinking is not supported on this model

* **Effort** (``output_config.effort``) is supported from Opus 4.5 upward and
  errors on Sonnet 4.5 and Haiku 4.5.

So the parameters have to follow the model, not the code path. Everything that
calls the API builds its kwargs through :func:`request_params`.

Thinking is simply omitted on older models rather than translated to a token
budget. The work sent to the cheap tier is extraction and transcription, where
thinking buys little and costs tokens on every one of a hundred applications.
"""

from __future__ import annotations

from typing import Any

#: Model families that accept `thinking={"type": "adaptive"}`.
ADAPTIVE_THINKING_MODELS = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
)

#: Model families that accept `output_config.effort`.
EFFORT_MODELS = ADAPTIVE_THINKING_MODELS + ("claude-opus-4-5",)

#: `xhigh` arrived later than the rest; older effort-capable models reject it.
XHIGH_MODELS = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
)

VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _matches(model: str, families: tuple[str, ...]) -> bool:
    model = (model or "").strip()
    return any(model.startswith(family) for family in families)


def supports_adaptive_thinking(model: str) -> bool:
    return _matches(model, ADAPTIVE_THINKING_MODELS)


def supports_effort(model: str) -> bool:
    return _matches(model, EFFORT_MODELS)


def normalize_effort(model: str, effort: str | None) -> str | None:
    """Clamp an effort level to what this model actually accepts."""
    if not effort or not supports_effort(model):
        return None
    effort = effort.strip().lower()
    if effort not in VALID_EFFORTS:
        return None
    if effort == "xhigh" and not _matches(model, XHIGH_MODELS):
        return "high"          # nearest supported level, rather than a 400
    return effort


def request_params(model: str, effort: str | None = None) -> dict[str, Any]:
    """The thinking/effort kwargs this model will accept.

    Returns an empty dict for models that support neither, which is correct:
    omitting them runs the model in its default mode rather than erroring.
    """
    params: dict[str, Any] = {}
    if supports_adaptive_thinking(model):
        params["thinking"] = {"type": "adaptive"}

    resolved = normalize_effort(model, effort)
    if resolved:
        params["output_config"] = {"effort": resolved}
    return params


# --------------------------------------------------------------------------
# Client construction
# --------------------------------------------------------------------------

#: How many times to ride out a transient failure before giving up.
#:
#: The SDK defaults to two quick retries, which was not enough for a real
#: overload: six tailoring jobs died in one burst on
#:
#:     503 overloaded_error: API key validation is temporarily unavailable.
#:                           Please retry.
#:
#: an error whose own text asks for a retry. Each of those was a posting the
#: user never saw, lost to a few seconds of upstream weather.
CLIENT_MAX_RETRIES = 6

#: Generous, because a tailoring call with thinking enabled is not quick, and a
#: timeout here costs the whole job.
CLIENT_TIMEOUT_SECONDS = 600.0


def make_client(api_key: str = "", **overrides: Any):
    """An Anthropic client configured to survive a transient overload.

    Every call site builds its client through this, so the retry policy is one
    decision rather than four that drift apart.
    """
    import anthropic

    options: dict[str, Any] = {
        "max_retries": CLIENT_MAX_RETRIES,
        "timeout": CLIENT_TIMEOUT_SECONDS,
    }
    if api_key:
        options["api_key"] = api_key
    options.update(overrides)
    return anthropic.Anthropic(**options)
