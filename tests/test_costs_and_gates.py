"""Cost accounting and the human-in-the-loop gate mechanism."""

from __future__ import annotations

import threading
import time
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from automation.gatekeeper import (
    AutoDeclineGatekeeper,
    DashboardGatekeeper,
    GateTimeout,
    TerminalGatekeeper,
)
from database.models import ActionKind, ActionStatus
from engine.cost_tracker import (
    FALLBACK_PRICE,
    PRICING,
    TokenUsage,
    compute_cost,
    daily_series,
    price_for,
    summarize,
)

# ---------------- pricing ----------------


def test_cost_of_a_plain_call():
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    # Opus 5: $5 in + $25 out per MTok.
    assert compute_cost("claude-opus-5", usage) == pytest.approx(30.0)


def test_cache_reads_are_a_tenth_of_input():
    cached = TokenUsage(cache_read_tokens=1_000_000)
    plain = TokenUsage(input_tokens=1_000_000)
    assert compute_cost("claude-opus-5", cached) == pytest.approx(
        compute_cost("claude-opus-5", plain) * 0.1
    )


def test_cache_writes_cost_more_than_input():
    write = TokenUsage(cache_write_tokens=1_000_000)
    plain = TokenUsage(input_tokens=1_000_000)
    assert compute_cost("claude-opus-5", write) == pytest.approx(
        compute_cost("claude-opus-5", plain) * 1.25
    )
    assert compute_cost("claude-opus-5", write, long_cache=True) == pytest.approx(
        compute_cost("claude-opus-5", plain) * 2.0
    )


def test_sonnet_intro_pricing_applies_only_within_its_window():
    usage = TokenUsage(input_tokens=1_000_000)
    during = compute_cost("claude-sonnet-5", usage, on=date(2026, 8, 22))
    after = compute_cost("claude-sonnet-5", usage, on=date(2026, 9, 1))
    assert during == pytest.approx(2.0)
    assert after == pytest.approx(3.0)


def test_unknown_model_is_priced_not_free():
    price, known = price_for("claude-something-unreleased")
    assert known is False and price is FALLBACK_PRICE
    assert compute_cost("claude-something-unreleased", TokenUsage(input_tokens=1_000_000)) > 0


def test_every_priced_model_has_output_above_input():
    for model, price in PRICING.items():
        assert price.output_per_mtok > price.input_per_mtok, model


def test_zero_usage_costs_nothing():
    assert compute_cost("claude-opus-5", TokenUsage()) == 0.0


# ---------------- usage parsing ----------------


def test_token_usage_reads_an_sdk_response():
    response = SimpleNamespace(usage=SimpleNamespace(
        input_tokens=120, output_tokens=45,
        cache_read_input_tokens=900, cache_creation_input_tokens=30))
    usage = TokenUsage.from_response(response)
    assert (usage.input_tokens, usage.output_tokens) == (120, 45)
    assert usage.cache_read_tokens == 900 and usage.cache_write_tokens == 30
    assert usage.total == 1095


def test_token_usage_tolerates_a_missing_usage_block():
    assert TokenUsage.from_response(SimpleNamespace()).total == 0
    assert TokenUsage.from_response(SimpleNamespace(usage=None)).total == 0


# ---------------- aggregation ----------------


def test_daily_series_fills_days_with_no_spend():
    now = datetime.now(timezone.utc)
    rows = [(now, 0.5, 1000), (now - timedelta(days=3), 0.25, 500)]
    series = daily_series(rows, days=7)
    assert len(series) == 7
    assert sum(e["cost"] for e in series) == pytest.approx(0.75)
    assert sum(1 for e in series if e["calls"] == 0) == 5


def test_daily_series_ignores_rows_outside_the_window():
    old = datetime.now(timezone.utc) - timedelta(days=400)
    series = daily_series([(old, 99.0, 1)], days=30)
    assert sum(e["cost"] for e in series) == 0.0


def test_summary_separates_average_per_day_from_per_active_day():
    now = datetime.now(timezone.utc)
    series = daily_series([(now, 3.0, 100)], days=30)
    summary = summarize(series)
    assert summary["total_cost"] == pytest.approx(3.0)
    assert summary["active_days"] == 1
    assert summary["avg_per_day"] == pytest.approx(0.1)
    assert summary["avg_per_active_day"] == pytest.approx(3.0)


# ---------------- gatekeepers ----------------


def test_terminal_gatekeeper_declines_without_a_tty(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", None)
    assert TerminalGatekeeper().confirm("submit the application") is False


def test_auto_decline_gatekeeper_refuses_everything():
    keeper = AutoDeclineGatekeeper()
    assert keeper.confirm("submit the application") is False
    assert keeper.ask("What is your salary expectation?") is None


def test_dashboard_gatekeeper_creates_an_action_and_waits(db):
    """A gate becomes a queue item; answering it releases the run."""
    user = db.create_user("ada@example.com", "$argon2id$fake")
    keeper = DashboardGatekeeper(db, user.id, wait_seconds=10, poll_seconds=0.05)

    def answer_soon():
        time.sleep(0.2)
        item = db.list_actions(user_id=user.id)[0]
        db.answer_action(item.id, "Approve", user_id=user.id)

    threading.Thread(target=answer_soon, daemon=True).start()
    assert keeper.confirm("submit the application") is True

    items = db.list_actions(user_id=user.id, status=None)
    assert len(items) == 1
    assert items[0].kind is ActionKind.SUBMIT_CONFIRMATION


def test_dashboard_gate_treats_cancel_as_declined(db):
    user = db.create_user("ada@example.com", "$argon2id$fake")
    keeper = DashboardGatekeeper(db, user.id, wait_seconds=10, poll_seconds=0.05)

    def cancel_soon():
        time.sleep(0.2)
        item = db.list_actions(user_id=user.id)[0]
        db.answer_action(item.id, "Cancel", user_id=user.id)

    threading.Thread(target=cancel_soon, daemon=True).start()
    assert keeper.confirm("submit the application") is False


def test_dashboard_gate_times_out_as_declined(db):
    """No answer must never mean yes."""
    user = db.create_user("ada@example.com", "$argon2id$fake")
    keeper = DashboardGatekeeper(db, user.id, wait_seconds=0.3, poll_seconds=0.05)
    assert keeper.confirm("submit the application") is False


def test_dismissing_an_action_releases_the_gate_as_declined(db):
    user = db.create_user("ada@example.com", "$argon2id$fake")
    keeper = DashboardGatekeeper(db, user.id, wait_seconds=10, poll_seconds=0.05)

    def dismiss_soon():
        time.sleep(0.2)
        db.dismiss_action(db.list_actions(user_id=user.id)[0].id, user_id=user.id)

    threading.Thread(target=dismiss_soon, daemon=True).start()
    assert keeper.confirm("submit the application") is False


def test_captcha_gate_is_recorded_as_human_verification(db):
    """A verification challenge is queued for a human, never solved."""
    user = db.create_user("ada@example.com", "$argon2id$fake")
    keeper = DashboardGatekeeper(db, user.id, wait_seconds=0.2, poll_seconds=0.05)
    keeper.ask("Clear the verification challenge", kind="captcha")

    item = db.list_actions(user_id=user.id, status=None)[0]
    assert item.kind is ActionKind.CAPTCHA
    assert item.status is ActionStatus.OPEN


def test_remembered_answers_are_reused_across_applications(db):
    """Answer a question once in the dashboard; never be asked again."""
    user = db.create_user("ada@example.com", "$argon2id$fake")
    item = db.create_action(user.id, ActionKind.UNMAPPED_FIELD,
                            "How many years of Kubernetes experience?")
    db.answer_action(item.id, "4", remember=True, user_id=user.id)

    assert db.answered_action_map(user.id) == {"How many years of Kubernetes experience?": "4"}

    # An answer the user chose not to remember stays out of the reuse map.
    other = db.create_action(user.id, ActionKind.UNMAPPED_FIELD, "One-off question?")
    db.answer_action(other.id, "no", remember=False, user_id=user.id)
    assert "One-off question?" not in db.answered_action_map(user.id)


def test_answer_map_is_scoped_per_user(db):
    ada = db.create_user("ada@example.com", "$argon2id$fake")
    eve = db.create_user("eve@example.com", "$argon2id$fake")
    item = db.create_action(ada.id, ActionKind.UNMAPPED_FIELD, "Salary?")
    db.answer_action(item.id, "185000", remember=True, user_id=ada.id)

    assert db.answered_action_map(ada.id) == {"Salary?": "185000"}
    assert db.answered_action_map(eve.id) == {}
