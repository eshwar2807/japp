"""Where the pipeline asks a human for something.

The pipeline stops for a human in three situations: an irreversible action
needs approval, a form field cannot be answered safely, or a human-verification
challenge appears. Who gets asked, and how, depends on where the run started:

  * ``TerminalGatekeeper``  - CLI runs. Blocking y/N prompts on stdin.
  * ``DashboardGatekeeper`` - web runs. Writes an ActionItem, then waits for the
    user to answer it in the dashboard.
  * ``AutoDeclineGatekeeper`` - unattended runs. Declines everything, so nothing
    irreversible can happen with nobody watching.

Keeping this behind one interface is what lets the same driver code serve a
terminal run and a web run without branching on context.
"""

from __future__ import annotations

import logging
import sys
import time
from abc import ABC, abstractmethod
from typing import Any

log = logging.getLogger(__name__)

#: How long a dashboard run waits for a human before giving up.
DEFAULT_WAIT_SECONDS = 900
POLL_INTERVAL_SECONDS = 3


class GateTimeout(Exception):
    """No human answered within the wait window."""


class Gatekeeper(ABC):
    """Interface between automation and whoever is supervising it."""

    @abstractmethod
    def confirm(self, action: str, details: list[str] | None = None) -> bool:
        """Approve an irreversible action. False means do not proceed."""

    @abstractmethod
    def ask(
        self,
        question: str,
        reason: str = "",
        kind: str = "unmapped_field",
        options: list[str] | None = None,
        required: bool = False,
    ) -> str | None:
        """Get a value the pipeline could not determine. None means unanswered."""

    def alert(self, title: str, lines: list[str] | None = None) -> None:
        """Surface something noteworthy that does not block."""
        log.warning("%s | %s", title, "; ".join(lines or []))


# --------------------------------------------------------------------------
# Terminal
# --------------------------------------------------------------------------


class TerminalGatekeeper(Gatekeeper):
    def __init__(self, default: bool = False) -> None:
        #: Answer used when no TTY is attached. False for every destructive gate.
        self.default = default

    def alert(self, title: str, lines: list[str] | None = None) -> None:
        width = 74
        print("\n" + "!" * width)
        print(f"  {title}")
        for line in lines or []:
            print(f"    - {line}")
        print("!" * width)

    def _prompt(self, text: str, default: bool) -> bool:
        if not sys.stdin or not sys.stdin.isatty():
            log.warning("No TTY available; declining gate: %s", text)
            return default
        suffix = " [Y/n] " if default else " [y/N] "
        try:
            reply = input("\n" + text + suffix).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return default if not reply else reply in ("y", "yes")

    def confirm(self, action: str, details: list[str] | None = None) -> bool:
        self.alert(f"About to {action}", details)
        return self._prompt(f"Proceed with: {action}?", self.default)

    def ask(
        self,
        question: str,
        reason: str = "",
        kind: str = "unmapped_field",
        options: list[str] | None = None,
        required: bool = False,
    ) -> str | None:
        self.alert(
            "Input needed",
            [question, reason] + ([f"Options: {', '.join(options)}"] if options else []),
        )
        if not sys.stdin or not sys.stdin.isatty():
            return None
        try:
            answer = input("Answer (blank to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return answer or None


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------


class DashboardGatekeeper(Gatekeeper):
    """Turns every stop into a dashboard action item and waits for the answer.

    This is what makes a human-verification challenge recoverable rather than
    fatal: the browser session stays open and parked on the challenge, the user
    is told exactly what is blocking, and clearing it in the real browser lets
    the run continue from where it stopped.
    """

    def __init__(
        self,
        db: Any,
        user_id: int,
        application_id: int | None = None,
        wait_seconds: int = DEFAULT_WAIT_SECONDS,
        poll_seconds: int = POLL_INTERVAL_SECONDS,
    ) -> None:
        self.db = db
        self.user_id = user_id
        self.application_id = application_id
        self.wait_seconds = wait_seconds
        self.poll_seconds = poll_seconds

    # -- helpers --

    def _kind(self, kind: str):
        from database.models import ActionKind

        mapping = {
            "unmapped_field": ActionKind.UNMAPPED_FIELD,
            "captcha": ActionKind.CAPTCHA,
            "account_creation": ActionKind.ACCOUNT_CREATION,
            "submit": ActionKind.SUBMIT_CONFIRMATION,
            "login": ActionKind.LOGIN_REQUIRED,
            "error": ActionKind.ERROR,
        }
        return mapping.get(kind, ActionKind.UNMAPPED_FIELD)

    def _wait_for(self, action_id: int) -> tuple[str, str]:
        """Block until answered or dismissed. Returns (status, answer)."""
        from database.models import ActionStatus

        deadline = time.monotonic() + self.wait_seconds
        while time.monotonic() < deadline:
            items = self.db.list_actions(
                user_id=self.user_id, status=None, application_id=self.application_id
            )
            item = next((i for i in items if i.id == action_id), None)
            if item and item.status is not ActionStatus.OPEN:
                return item.status.value, item.answer or ""
            time.sleep(self.poll_seconds)
        raise GateTimeout(
            f"No response within {self.wait_seconds}s for action #{action_id}."
        )

    # -- interface --

    def alert(self, title: str, lines: list[str] | None = None) -> None:
        from database.models import LogLevel

        self.db.log_event(
            self.user_id,
            "alert",
            f"{title}: {'; '.join(lines or [])}",
            level=LogLevel.WARNING,
            application_id=self.application_id,
        )

    def confirm(self, action: str, details: list[str] | None = None) -> bool:
        kind = "submit" if "submit" in action.lower() else (
            "account_creation" if "account" in action.lower() else "unmapped_field"
        )
        item = self.db.create_action(
            user_id=self.user_id,
            kind=self._kind(kind),
            question=f"Approve: {action}?",
            reason="\n".join(details or []),
            application_id=self.application_id,
            field_type="confirm",
            options=["Approve", "Cancel"],
            required=True,
        )
        try:
            status, answer = self._wait_for(item.id)
        except GateTimeout:
            self.alert("Gate timed out; treating as declined", [action])
            return False
        return status == "Answered" and answer.strip().lower() in ("approve", "yes", "y", "true")

    def ask(
        self,
        question: str,
        reason: str = "",
        kind: str = "unmapped_field",
        options: list[str] | None = None,
        required: bool = False,
    ) -> str | None:
        item = self.db.create_action(
            user_id=self.user_id,
            kind=self._kind(kind),
            question=question,
            reason=reason,
            application_id=self.application_id,
            field_type="select" if options else "text",
            options=options,
            required=required,
        )
        try:
            status, answer = self._wait_for(item.id)
        except GateTimeout:
            return None
        return answer or None if status == "Answered" else None


# --------------------------------------------------------------------------
# Unattended
# --------------------------------------------------------------------------


class AutoDeclineGatekeeper(Gatekeeper):
    """Declines everything. The safe default when nobody is supervising."""

    def confirm(self, action: str, details: list[str] | None = None) -> bool:
        log.warning("Unattended run; declining: %s", action)
        return False

    def ask(self, question: str, reason: str = "", kind: str = "unmapped_field",
            options: list[str] | None = None, required: bool = False) -> str | None:
        log.warning("Unattended run; cannot answer: %s", question)
        return None
