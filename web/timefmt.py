"""Rendering timestamps in the viewer's timezone.

Everything is stored in UTC, which is correct, and displayed in UTC, which is
not: a log read at 9pm in California showing 04:00 the next day is a log nobody
can reconcile with what they just did.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

log = logging.getLogger(__name__)

#: Offered in the settings dropdown. Any IANA name works; these are shortcuts.
COMMON_ZONES = [
    ("America/Los_Angeles", "Pacific (PST/PDT)"),
    ("America/Denver", "Mountain (MST/MDT)"),
    ("America/Chicago", "Central (CST/CDT)"),
    ("America/New_York", "Eastern (EST/EDT)"),
    ("Europe/London", "UK (GMT/BST)"),
    ("Europe/Dublin", "Ireland (GMT/IST)"),
    ("Europe/Berlin", "Central Europe (CET/CEST)"),
    ("Asia/Kolkata", "India (IST)"),
    ("Asia/Singapore", "Singapore (SGT)"),
    ("Australia/Sydney", "Sydney (AEST/AEDT)"),
    ("UTC", "UTC"),
]


def resolve(name: str | None) -> ZoneInfo:
    """A usable zone for `name`, falling back to UTC rather than raising."""
    if not name:
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        log.warning("Unknown timezone %r; showing UTC", name)
        return ZoneInfo("UTC")


def valid_zone(name: str) -> bool:
    try:
        ZoneInfo(name)
        return True
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        return False


def to_zone(value: datetime | None, zone_name: str | None) -> datetime | None:
    """Convert a stored timestamp into the viewer's zone.

    Rows written before timezone-aware columns existed come back naive; those
    are UTC by construction, so they are labelled rather than guessed at.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(resolve(zone_name))


def make_filters(zone_name: str | None) -> dict:
    """Jinja filters bound to one viewer's zone."""

    def when(value, fmt="%b %d %H:%M"):
        converted = to_zone(value, zone_name)
        return converted.strftime(fmt) if converted else ""

    def when_precise(value):
        return when(value, "%b %d %H:%M:%S")

    def day(value):
        return when(value, "%b %d")

    def full(value):
        return when(value, "%Y-%m-%d %H:%M %Z")

    return {"when": when, "when_precise": when_precise, "day": day, "full": full}


def abbreviation(zone_name: str | None) -> str:
    """Current short name for the zone, e.g. PST or PDT."""
    return datetime.now(resolve(zone_name)).strftime("%Z")
