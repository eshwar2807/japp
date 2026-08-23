"""HTTP client for the hosted dashboard's agent API."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30


class AgentError(Exception):
    """The server rejected a request, or could not be reached."""


class DashboardClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("Server URL must start with http:// or https://")
        if self.base_url.startswith("http://") and "localhost" not in self.base_url \
                and "127.0.0.1" not in self.base_url:
            # The API key travels on every request; plain HTTP would leak it.
            raise ValueError("Refusing to send an API key over plain HTTP to a remote host.")
        self.api_key = api_key
        self.timeout = timeout

    # ---------------- transport ----------------

    def _request(self, method: str, path: str, payload: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            url, data=data, method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "job-pipeline-agent/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status == 204:
                    return None
                body = response.read()
                return json.loads(body) if body else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code == 401:
                raise AgentError("API key rejected. Generate one in Settings.") from exc
            raise AgentError(f"HTTP {exc.code} on {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise AgentError(f"Cannot reach {self.base_url}: {exc.reason}") from exc

    # ---------------- endpoints ----------------

    def whoami(self) -> dict:
        return self._request("GET", "/api/v1/me")

    def claim(self) -> dict | None:
        """Claim the next apply job, or None when the queue is empty."""
        return self._request("POST", "/api/v1/agent/claim")

    def download_resume(self, path: str, destination: Path) -> Path:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                destination.write_bytes(response.read())
        except urllib.error.URLError as exc:
            raise AgentError(f"Could not download the resume: {exc}") from exc

        if destination.stat().st_size < 1024 or destination.read_bytes()[:5] != b"%PDF-":
            raise AgentError("Downloaded resume is not a valid PDF.")
        return destination

    def block(self, job_id: int, kind: str, question: str, reason: str = "",
              options: list[str] | None = None, holds_browser: bool = True) -> int:
        result = self._request("POST", f"/api/v1/agent/jobs/{job_id}/block", {
            "kind": kind, "question": question, "reason": reason,
            "options": options, "holds_browser": holds_browser,
        })
        return int(result["action_id"])

    def poll_action(self, action_id: int) -> dict:
        return self._request("GET", f"/api/v1/agent/actions/{action_id}")

    def finish(self, job_id: int, submitted: bool, message: str = "",
               escalations: list[dict] | None = None) -> None:
        self._request("POST", f"/api/v1/agent/jobs/{job_id}/finish", {
            "submitted": submitted, "message": message,
            "escalations": escalations or [],
        })

    def log(self, job_id: int, event: str, message: str = "", level: str = "INFO") -> None:
        try:
            self._request("POST", f"/api/v1/agent/jobs/{job_id}/log", {
                "event": event, "message": message, "level": level,
            })
        except AgentError:
            # Log shipping must never take down the run it is reporting on.
            log.debug("Could not ship log line for job %s", job_id)
