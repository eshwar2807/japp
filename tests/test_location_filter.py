"""Keeping a US-only search inside the US.

This filter has now been wrong twice, both times in the same direction: it let
a posting through because nothing in it looked forbidden.

    "Remote Canada"                     passed on the word "remote"
    "Danang, Danang, Vietnam; ..."      passed because Vietnam was not listed

The second is the instructive one. The table of countries can always be one
country short, so the rule no longer ends in "allow what I do not recognise".
A search that asks for the US requires evidence of the US, and a location
naming some place we cannot identify is a place that was not asked for.
"""

from __future__ import annotations

import pytest

from engine.boards import location_allowed, names_a_place

US = ["United States"]


# ---------------- the postings that got through ----------------


def test_a_vietnam_posting_is_refused_by_a_us_search():
    """Verbatim from the queue: a Senior Fullstack Java Developer at Aperia."""
    assert not location_allowed(
        "Danang, Danang, Vietnam; Ho Chi Minh City, Ho Chi Minh City, Vietnam", US)


def test_a_remote_canada_posting_is_refused_by_a_us_search():
    assert not location_allowed("Remote Canada", US)


@pytest.mark.parametrize("where", [
    "Jakarta, Indonesia",
    "Bangkok, Thailand",
    "Istanbul, Turkey",
    "Seoul, South Korea",
    "Dubai, United Arab Emirates",
    "Lagos, Nigeria",
    "Prague, Czech Republic",
    "Santiago, Chile",
])
def test_other_countries_are_refused_too(where):
    """The point of the rewrite: this must not depend on the table being
    complete on the day the posting appears."""
    assert not location_allowed(where, US)


def test_a_country_absent_from_the_table_is_still_refused():
    """The exact failure mode, with a country deliberately not listed."""
    assert not location_allowed("Ulaanbaatar, Mongolia", US)


# ---------------- and the ones that must still get through ----------------


@pytest.mark.parametrize("where", [
    "Austin, TX",
    "San Francisco, CA",
    "Remote, United States",
    "Remote - US",
    "New York, NY",
    "Boston",
    "Seattle, Washington",
    "Chicago, IL; Austin, TX",
    "Dublin, CA",
])
def test_us_postings_are_kept(where):
    assert location_allowed(where, US)


def test_a_us_city_sharing_a_name_with_a_foreign_capital_is_kept():
    """The user lives in Dublin, California. Ireland is a different Dublin."""
    assert location_allowed("Dublin, CA", US)
    assert not location_allowed("Dublin, Ireland", US)


def test_a_multi_country_posting_including_the_us_is_kept():
    assert location_allowed("US or Canada, Remote", US)


def test_a_two_letter_code_does_not_override_a_named_country():
    """In "Toronto, CA" the CA is Canada. Reading it as California is how a
    Canadian role reaches someone who needs US authorisation."""
    assert not location_allowed("Toronto, CA", US)


@pytest.mark.parametrize("where", ["Remote", "Hybrid", "Anywhere", "", "Flexible"])
def test_a_posting_naming_no_place_is_open_to_anywhere(where):
    assert location_allowed(where, US)


def test_an_unrestricted_search_keeps_everything():
    assert location_allowed("Danang, Vietnam", [])


# ---------------- the distinction the rule turns on ----------------


@pytest.mark.parametrize("where", ["Remote", "Hybrid", "remote - flexible", "", "N/A"])
def test_these_name_no_place(where):
    assert not names_a_place(where)


@pytest.mark.parametrize("where", ["Danang, Vietnam", "Remote - Poland", "Berlin"])
def test_these_do_name_a_place(where):
    assert names_a_place(where)
