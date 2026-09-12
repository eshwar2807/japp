#!/usr/bin/env python3
"""Local apply agent.

Runs on your own machine and does the one thing the hosted dashboard cannot:
drive an application form in a browser you can actually see. It claims only
`apply` jobs, so tailoring stays on the server.

    python -m agent --server https://japp.fly.dev

The API key is read from JP_AGENT_KEY, or from --key-file. It is never passed
on the command line by default, because arguments are visible to every process
on the machine via `ps`.

Nothing here bypasses a verification challenge. When one appears, the browser
window stops on it and the dashboard asks you to clear it — from your phone if
you like — and the run resumes once you do.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.client import AgentError, DashboardClient  # noqa: E402
from agent.gatekeeper import AgentGatekeeper  # noqa: E402

log = logging.getLogger("agent")

IDLE_POLL_SECONDS = 10
_stop = False


def _handle_signal(signum, frame):  # pragma: no cover - signal path
    global _stop
    _stop = True
    print("\nFinishing the current job, then stopping...")


def resolve_key(args) -> str:
    if args.key_file:
        return Path(args.key_file).expanduser().read_text().strip()
    key = os.getenv("JP_AGENT_KEY", "").strip()
    if key:
        return key
    raise SystemExit(
        "No API key. Set JP_AGENT_KEY, or pass --key-file with a file containing it.\n"
        "Generate one under Settings in the dashboard."
    )


def run_one(client: DashboardClient, payload: dict) -> None:
    """Run a single claimed apply job against a local browser."""
    from automation.ats_drivers import get_driver_class
    from automation.stealth_browser import (
        BrowserUnavailable,
        HumanBrowser,
        HumanDeclined,
        ManualInterventionRequired,
    )
    from engine.schemas import MasterProfile, TailoredResumeSchema
    from engine.screener_mapper import ScreenerMapper

    job = payload["job"]
    application = payload["application"]
    job_id = job["id"]

    print(f"\n  {application['company']} — {application['role_title']}")
    print(f"  {application['job_url']}")

    profile = MasterProfile.model_validate(payload["profile"])
    tailored = TailoredResumeSchema.model_validate(payload.get("tailored") or {})

    # Same resolution ladder the server uses: rules, then what you have
    # answered before, then one batched inference attempt, then you.
    resolver = None
    if payload.get("anthropic_key"):
        from engine.answer_resolver import LLMAnswerResolver

        resolver = LLMAnswerResolver(api_key=payload["anthropic_key"])

    mapper = ScreenerMapper(
        profile,
        screener_answers=tailored.screener_answers,
        remembered=payload.get("known_answers") or {},
        resolver=resolver,
        salary_min=application.get("salary_min"),
        salary_max=application.get("salary_max"),
    )

    # The stored link is often the employer's careers listing, which has no
    # form on it - that is what produced "no resume input" and "submit button
    # not found". Settle on the page that really holds the form before starting
    # a browser, and if no such page exists, hand this one back rather than
    # asking about a submit button that was never going to be there.
    from engine.boards import BoardError, find_application_form

    try:
        target = find_application_form(application["job_url"])
    except BoardError as exc:
        # Could not tell. Carry on with the stored link rather than writing off
        # a real application because a board was slow.
        client.log(job_id, "form_lookup_failed", str(exc), level="WARNING")
        target = application["job_url"]
    if target is None:
        message = ("No application form at this URL on any known ATS address; "
                   "this employer only accepts applications through their own "
                   "site. Apply to this one by hand.")
        client.log(job_id, "no_application_form", message, level="WARNING")
        client.finish(job_id, submitted=False, message=message)
        print(f"  Skipped: no reachable form — apply by hand at {application['job_url']}")
        return
    if target != application["job_url"]:
        print(f"  Form is at: {target}")

    with tempfile.TemporaryDirectory(prefix="jp_agent_") as tmp:
        resume = client.download_resume(payload["resume_url"], Path(tmp) / "resume.pdf")
        client.log(job_id, "started", f"Agent picked up {target}")

        gatekeeper = AgentGatekeeper(client, job_id)
        driver_class = get_driver_class(target)

        try:
            with HumanBrowser(gatekeeper=gatekeeper) as browser:
                driver = driver_class(browser, profile, mapper, db=None)
                outcome = driver.apply(target, resume)
                # Proof of what the page looked like when the run ended. The
                # server-side path has always kept one; the agent kept nothing,
                # so a run reporting "Submitted" left no record behind at all.
                shots = Path.home() / ".japp" / "screenshots"
                shots.mkdir(parents=True, exist_ok=True)
                try:
                    outcome.screenshot_path = str(
                        browser.screenshot(shots / f"app_{application['id']}.png"))
                    print(f"  Screenshot: {outcome.screenshot_path}")
                except Exception as exc:
                    log.debug("Could not capture a screenshot: %s", exc)
        except BrowserUnavailable as exc:
            # Nothing to do with this application: the machine cannot run any
            # of them. Hand the job back untouched rather than burning through
            # the whole queue marking each one failed.
            client.log(job_id, "browser_unavailable", str(exc), level="ERROR")
            client.release(job_id, str(exc)[:400])
            raise
        except (ManualInterventionRequired, HumanDeclined) as exc:
            client.log(job_id, "stopped", str(exc), level="WARNING")
            client.finish(job_id, submitted=False, message=str(exc)[:400])
            print(f"  Stopped: {exc}")
            return
        except Exception as exc:
            client.log(job_id, "error", str(exc), level="ERROR")
            client.finish(job_id, submitted=False, message=str(exc)[:400])
            print(f"  Failed: {exc}")
            return

    client.finish(
        job_id,
        submitted=outcome.submitted,
        message=outcome.message or ("Submitted" if outcome.submitted else "Not submitted"),
        escalations=[{"question": e.question, "reason": e.reason} for e in outcome.escalations],
    )
    # Say which of the two happened. "Submitted" on the strength of a click
    # alone was the thing that could not be trusted.
    if outcome.submitted:
        print(f"  Submitted — confirmed by: {outcome.confirmation[:70]} "
              f"({outcome.fields_filled} fields filled)")
    elif outcome.confirmation == "" and "no confirmation" in (outcome.message or ""):
        print(f"  UNCONFIRMED — the click landed but the page showed no receipt. "
              f"Check it yourself ({outcome.fields_filled} fields filled)")
    else:
        print(f"  Not submitted ({outcome.fields_filled} fields filled)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent", description="Run application jobs on this machine.")
    parser.add_argument("--server", required=True, help="Dashboard URL, e.g. https://japp.fly.dev")
    parser.add_argument("--key-file", help="File containing the API key")
    parser.add_argument("--once", action="store_true", help="Run one job, then exit")
    parser.add_argument("--poll", type=int, default=IDLE_POLL_SECONDS,
                        help="Seconds between polls when idle")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(name)-20s %(message)s")
    for noisy in ("weasyprint", "fontTools", "PIL"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    try:
        client = DashboardClient(args.server, resolve_key(args))
        identity = client.whoami()
    except (AgentError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Connected to {args.server} as {identity['email']}")
    print("Claiming application jobs. A browser opens for each one; "
          "anything needing you appears in the dashboard.")
    print("Ctrl-C to stop.\n")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    idle_notice = True
    while not _stop:
        try:
            payload = client.claim()
        except AgentError as exc:
            print(f"  {exc}; retrying in {args.poll}s")
            time.sleep(args.poll)
            continue

        if payload is None:
            if idle_notice:
                print(f"  Nothing queued. Polling every {args.poll}s...")
                idle_notice = False
            if args.once:
                return 0
            time.sleep(args.poll)
            continue

        idle_notice = True
        try:
            run_one(client, payload)
        except BrowserUnavailable as exc:
            print(f"\n  Stopping: {exc}")
            print("  The job was put back in the queue. Start the agent again "
                  "once that is sorted.")
            return 1
        except Exception:
            log.exception("Job failed unexpectedly")
        if args.once:
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
