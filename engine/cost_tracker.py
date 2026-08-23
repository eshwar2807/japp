"""Token accounting and USD cost for every Claude call.

Prices are per million tokens, Anthropic first-party API rates. Cache
multipliers follow standard Anthropic pricing: a 5-minute cache write costs
1.25x base input, a 1-hour write 2x, and a cache read 0.1x.

Rates change. ``PRICING`` is the single place to update them, and
``PRICING_AS_OF`` is surfaced in the dashboard so a stale table is visible
rather than silently wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

PRICING_AS_OF = date(2026, 6, 24)

CACHE_WRITE_MULTIPLIER_5M = 1.25
CACHE_WRITE_MULTIPLIER_1H = 2.0
CACHE_READ_MULTIPLIER = 0.1


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    #: Optional promotional rate, applied through ``intro_until`` inclusive.
    intro_input_per_mtok: float | None = None
    intro_output_per_mtok: float | None = None
    intro_until: date | None = None

    def rates(self, on: date) -> tuple[float, float]:
        if (
            self.intro_until
            and on <= self.intro_until
            and self.intro_input_per_mtok is not None
            and self.intro_output_per_mtok is not None
        ):
            return self.intro_input_per_mtok, self.intro_output_per_mtok
        return self.input_per_mtok, self.output_per_mtok


PRICING: dict[str, ModelPrice] = {
    "claude-fable-5": ModelPrice(10.00, 50.00),
    "claude-mythos-5": ModelPrice(10.00, 50.00),
    "claude-opus-5": ModelPrice(5.00, 25.00),
    "claude-opus-4-8": ModelPrice(5.00, 25.00),
    "claude-opus-4-7": ModelPrice(5.00, 25.00),
    "claude-opus-4-6": ModelPrice(5.00, 25.00),
    "claude-sonnet-5": ModelPrice(
        3.00, 15.00,
        intro_input_per_mtok=2.00, intro_output_per_mtok=10.00,
        intro_until=date(2026, 8, 31),
    ),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
}

#: Used when a model id is not in the table, so an unknown model still shows a
#: non-zero, clearly-flagged cost rather than reading as free.
FALLBACK_PRICE = ModelPrice(5.00, 25.00)


def price_for(model: str) -> tuple[ModelPrice, bool]:
    """Return (price, is_known). Unknown ids fall back to Opus-tier rates."""
    if model in PRICING:
        return PRICING[model], True
    for known, price in PRICING.items():  # tolerate dated suffixes
        if model.startswith(known):
            return price, True
    return FALLBACK_PRICE, False


@dataclass
class TokenUsage:
    """Normalised view of ``response.usage`` across SDK versions."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @classmethod
    def from_response(cls, response: Any) -> "TokenUsage":
        usage = getattr(response, "usage", None)
        if usage is None:
            return cls()

        def field(*names: str) -> int:
            for name in names:
                value = getattr(usage, name, None)
                if value is None and isinstance(usage, dict):
                    value = usage.get(name)
                if isinstance(value, int):
                    return value
            return 0

        return cls(
            input_tokens=field("input_tokens"),
            output_tokens=field("output_tokens"),
            cache_read_tokens=field("cache_read_input_tokens", "cache_read_tokens"),
            cache_write_tokens=field("cache_creation_input_tokens", "cache_write_tokens"),
        )

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


def compute_cost(
    model: str,
    usage: TokenUsage,
    on: date | None = None,
    long_cache: bool = False,
) -> float:
    """USD cost of one call, rounded to 6 decimal places."""
    price, _ = price_for(model)
    in_rate, out_rate = price.rates(on or datetime.now(timezone.utc).date())
    write_multiplier = CACHE_WRITE_MULTIPLIER_1H if long_cache else CACHE_WRITE_MULTIPLIER_5M

    cost = (
        usage.input_tokens * in_rate
        + usage.output_tokens * out_rate
        + usage.cache_read_tokens * in_rate * CACHE_READ_MULTIPLIER
        + usage.cache_write_tokens * in_rate * write_multiplier
    ) / 1_000_000
    return round(cost, 6)


# --------------------------------------------------------------------------
# Aggregation for the dashboard
# --------------------------------------------------------------------------

#: Windows offered in the cost view, in days.
COST_WINDOWS = (30, 90, 120, 360)


def daily_series(
    rows: list[tuple[datetime, float, int]], days: int, today: date | None = None
) -> list[dict[str, Any]]:
    """Bucket usage rows into one entry per day, including days with no spend.

    ``rows`` is ``(created_at, cost_usd, total_tokens)``. Gap-filling matters:
    a chart that silently skips empty days misrepresents the trend.
    """
    today = today or datetime.now(timezone.utc).date()
    start = today - timedelta(days=days - 1)

    buckets: dict[date, dict[str, Any]] = {
        start + timedelta(days=i): {"date": start + timedelta(days=i), "cost": 0.0,
                                    "tokens": 0, "calls": 0}
        for i in range(days)
    }

    for created_at, cost, tokens in rows:
        day = created_at.date() if isinstance(created_at, datetime) else created_at
        bucket = buckets.get(day)
        if bucket is None:
            continue
        bucket["cost"] += cost or 0.0
        bucket["tokens"] += tokens or 0
        bucket["calls"] += 1

    series = [buckets[k] for k in sorted(buckets)]
    for entry in series:
        entry["cost"] = round(entry["cost"], 6)
    return series


def summarize(series: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(e["cost"] for e in series)
    active = [e for e in series if e["calls"]]
    peak = max(series, key=lambda e: e["cost"]) if series else None
    return {
        "total_cost": round(total, 4),
        "total_tokens": sum(e["tokens"] for e in series),
        "total_calls": sum(e["calls"] for e in series),
        "days": len(series),
        "active_days": len(active),
        "avg_per_day": round(total / len(series), 4) if series else 0.0,
        "avg_per_active_day": round(total / len(active), 4) if active else 0.0,
        "peak_day": peak["date"].isoformat() if peak and peak["cost"] else None,
        "peak_cost": round(peak["cost"], 4) if peak else 0.0,
    }
