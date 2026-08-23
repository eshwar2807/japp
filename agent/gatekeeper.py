"""Gatekeeper that reports to the hosted dashboard and waits for an answer.

Same contract as the local and dashboard gatekeepers, so the driver code runs
unchanged. The difference is only where the question goes and where the answer
comes from: over HTTP to your dashboard, which you can answer from your phone
while the browser sits open on your desk.
"""

from __future__ import annotations

import logging
import time

from automation.gatekeeper import Gatekeeper

log = logging.getLogger(__name__)

POLL_SECONDS = 3
DEFAULT_WAIT_SECONDS = 1800


class AgentGatekeeper(Gatekeeper):
    def __init__(self, client, job_id: int, wait_seconds: int = DEFAULT_WAIT_SECONDS,
                 poll_seconds: int = POLL_SECONDS) -> None:
        self.client = client
        self.job_id = job_id
        self.wait_seconds = wait_seconds
        self.poll_seconds = poll_seconds

    def _wait(self, action_id: int) -> tuple[str, str]:
        deadline = time.monotonic() + self.wait_seconds
        while time.monotonic() < deadline:
            state = self.client.poll_action(action_id)
            if state["status"] != "Open":
                return state["status"], state.get("answer") or ""
            time.sleep(self.poll_seconds)
        return "Timeout", ""

    def alert(self, title: str, lines: list[str] | None = None) -> None:
        self.client.log(self.job_id, "alert", f"{title}: {'; '.join(lines or [])}",
                        level="WARNING")

    def confirm(self, action: str, details: list[str] | None = None) -> bool:
        lowered = action.lower()
        if "verification" in lowered or "captcha" in lowered:
            kind = "captcha"
        elif "account" in lowered:
            kind = "account_creation"
        elif "submit" in lowered:
            kind = "submit"
        else:
            kind = "unmapped_field"

        print(f"\n  Waiting on you: {action}")
        print("  Answer it in the dashboard; this window stays open.")

        action_id = self.client.block(
            self.job_id, kind, f"Approve: {action}?", "\n".join(details or []),
            options=["Approve", "Cancel"], holds_browser=True,
        )
        status, answer = self._wait(action_id)
        if status == "Timeout":
            print("  No answer in time; treating as declined.")
            return False
        return status == "Answered" and answer.strip().lower() in (
            "approve", "yes", "y", "true"
        )

    def ask(self, question: str, reason: str = "", kind: str = "unmapped_field",
            options: list[str] | None = None, required: bool = False) -> str | None:
        print(f"\n  Needs an answer: {question}")
        action_id = self.client.block(
            self.job_id, kind, question, reason, options=options, holds_browser=True,
        )
        status, answer = self._wait(action_id)
        return answer or None if status == "Answered" else None
