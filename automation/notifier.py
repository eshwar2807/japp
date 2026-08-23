"""Telling the human that a run needs them.

Two channels:

  * **desktop** - a local OS notification. Nothing leaves the machine.
  * **webhook** - an opt-in POST to a URL the user supplies (ntfy, Slack,
    Discord, whatever). This sends data to a third party, so the payload is
    deliberately thin: what is blocking, and which job. Never the job
    description, the resume, form answers, credentials or profile data. If you
    want detail, open the dashboard - the notification's job is only to get you
    there.

Delivery failures are logged and swallowed. A notification that cannot be sent
must never take down the run it was reporting on.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger(__name__)

WEBHOOK_TIMEOUT_SECONDS = 6
MAX_TITLE = 120
MAX_BODY = 400


@dataclass
class Notice:
    """One thing worth interrupting the user for."""

    title: str
    body: str
    url: str = ""

    def clipped(self) -> "Notice":
        return Notice(self.title[:MAX_TITLE], self.body[:MAX_BODY], self.url)


class Channel(Protocol):
    name: str

    def send(self, notice: Notice) -> None:
        """Deliver, or raise on failure."""


# --------------------------------------------------------------------------
# Desktop
# --------------------------------------------------------------------------


def _shell_quote_applescript(value: str) -> str:
    """AppleScript string literal escaping.

    The notification text contains company and role names from job postings,
    i.e. untrusted input. Escaping it prevents that text from terminating the
    string and being interpreted as script.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


class DesktopChannel:
    """Local OS notification. macOS, Linux (notify-send) and Windows."""

    name = "desktop"

    def available(self) -> bool:
        system = platform.system()
        if system == "Darwin":
            return bool(shutil.which("osascript"))
        if system == "Linux":
            return bool(shutil.which("notify-send"))
        if system == "Windows":
            return bool(shutil.which("powershell"))
        return False

    def send(self, notice: Notice) -> None:
        notice = notice.clipped()
        system = platform.system()

        if system == "Darwin":
            script = (
                f'display notification "{_shell_quote_applescript(notice.body)}" '
                f'with title "{_shell_quote_applescript(notice.title)}" '
                f'sound name "Ping"'
            )
            subprocess.run(["osascript", "-e", script], check=True,
                           capture_output=True, timeout=10)
        elif system == "Linux":
            subprocess.run(["notify-send", notice.title, notice.body],
                           check=True, capture_output=True, timeout=10)
        elif system == "Windows":
            script = (
                "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications,"
                " ContentType=WindowsRuntime] > $null; "
                f"Write-Output {json.dumps(notice.title + ': ' + notice.body)}"
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", script],
                           check=True, capture_output=True, timeout=10)
        else:
            raise RuntimeError(f"No desktop notifier for {system}.")


# --------------------------------------------------------------------------
# Webhook
# --------------------------------------------------------------------------


class WebhookChannel:
    """POST a small JSON payload to a user-supplied URL.

    The payload carries no job description, resume content, form answers or
    credentials - only what is blocking and where to go. Anything richer would
    be handing a third party the contents of your applications.
    """

    name = "webhook"

    def __init__(self, url: str) -> None:
        self.url = (url or "").strip()

    def available(self) -> bool:
        return self.url.startswith(("http://", "https://"))

    def send(self, notice: Notice) -> None:
        if not self.available():
            raise ValueError("Webhook URL must be an absolute http(s) URL.")
        notice = notice.clipped()
        payload = json.dumps({
            "title": notice.title,
            "message": notice.body,
            # Slack and Discord both read `text`/`content`; including both makes
            # the common destinations work without per-vendor formatting.
            "text": f"{notice.title}\n{notice.body}",
            "content": f"{notice.title}\n{notice.body}",
            "url": notice.url,
        }).encode()

        request = urllib.request.Request(
            self.url, data=payload,
            headers={"Content-Type": "application/json",
                     "User-Agent": "job-pipeline/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=WEBHOOK_TIMEOUT_SECONDS) as response:
            if response.status >= 400:
                raise RuntimeError(f"Webhook returned HTTP {response.status}.")


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


class Notifier:
    """Fans a notice out to the channels a user has enabled, and records it."""

    def __init__(self, db, channels: list[Channel] | None = None) -> None:
        self.db = db
        self._channels = channels   # injected in tests; resolved per-user otherwise

    def channels_for(self, user) -> list[Channel]:
        if self._channels is not None:
            return self._channels
        channels: list[Channel] = []
        if getattr(user, "notify_desktop", False):
            desktop = DesktopChannel()
            if desktop.available():
                channels.append(desktop)
            else:
                log.debug("Desktop notifications unavailable on this platform.")
        webhook_url = getattr(user, "notify_webhook_url", None)
        if webhook_url:
            channels.append(WebhookChannel(webhook_url))
        return channels

    def notify(self, user, notice: Notice, job_id: int | None = None) -> list[str]:
        """Send on every enabled channel. Returns the names that succeeded."""
        delivered: list[str] = []
        for channel in self.channels_for(user):
            error = ""
            try:
                channel.send(notice)
                delivered.append(channel.name)
            except (subprocess.SubprocessError, OSError, urllib.error.URLError,
                    RuntimeError, ValueError) as exc:
                error = str(exc)[:500]
                log.warning("Notification via %s failed: %s", channel.name, error)

            try:
                self.db.record_notification(
                    user_id=user.id, job_id=job_id, channel=channel.name,
                    title=notice.title, body=notice.body,
                    delivered=not error, error=error,
                )
            except Exception:      # accounting must not break the run
                log.exception("Could not record notification")
        return delivered


def block_notice(job, reason: str, kind: str, base_url: str = "") -> Notice:
    """Build the notice for a job that has just parked."""
    app = getattr(job, "application", None)
    where = f"{app.company} — {app.role_title}" if app else (job.job_url or "a posting")
    return Notice(
        title=f"Job pipeline: {kind}",
        body=f"{where}\n{reason}",
        url=f"{base_url}/actions" if base_url else "/actions",
    )
