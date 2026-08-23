#!/usr/bin/env python3
"""CLI orchestrator for the job application pipeline.

    python main.py doctor                     check the setup and master profile
    python main.py tailor <url|--jd-file>     tailor + build the PDF, no browser
    python main.py apply <url>                full flow, human-gated at submit
    python main.py review                     application history and stats
    python main.py feedback --app-id 3 --status Interview --notes "..."
    python main.py creds                      list stored portal credentials
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import settings  # noqa: E402
from database.db_manager import DBManager  # noqa: E402
from database.models import ApplicationStatus  # noqa: E402
from engine.ats_optimizer import ATSOptimizer, find_placeholders, load_master_profile  # noqa: E402
from engine.pdf_generator import PDFGenerator  # noqa: E402
from engine.schemas import TailoredResumeSchema  # noqa: E402

try:
    from rich.console import Console
    from rich.table import Table

    console = Console()
except ImportError:  # rich is optional
    console = None
    Table = None  # type: ignore


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------


def say(message: str, style: str = "") -> None:
    if console:
        console.print(message, style=style)
    else:
        print(message)


def rule(title: str) -> None:
    if console:
        console.rule(f"[bold]{title}")
    else:
        print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)-24s %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    for noisy in ("weasyprint", "fontTools", "PIL"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


# --------------------------------------------------------------------------
# Shared steps
# --------------------------------------------------------------------------


def load_profile_or_exit(allow_placeholders: bool = False):
    """Load the master profile, refusing to continue on unfilled placeholders."""
    try:
        profile = load_master_profile()
    except FileNotFoundError:
        say(f"[red]Master profile not found:[/] {settings.MASTER_PROFILE_PATH}")
        sys.exit(2)
    except Exception as exc:
        say(f"[red]Master profile is invalid:[/] {exc}")
        sys.exit(2)

    placeholders = find_placeholders(profile)
    if placeholders and not allow_placeholders:
        say("[red]Master profile still contains placeholder values.[/]")
        say("Applying with these would send a broken resume. Fill them in first:\n")
        for path in placeholders[:25]:
            say(f"  - {path}")
        if len(placeholders) > 25:
            say(f"  ... and {len(placeholders) - 25} more")
        say(f"\nEdit: {settings.MASTER_PROFILE_PATH}")
        sys.exit(2)
    return profile


def fetch_job_description(url: str, jd_file: Path | None) -> str:
    """Read the posting from a file, or scrape the visible text of the page."""
    if jd_file:
        return Path(jd_file).read_text(encoding="utf-8")

    say(f"Fetching job description from {url} ...")
    from automation.stealth_browser import HumanBrowser

    with HumanBrowser(headless=True) as browser:
        browser.goto(url)
        text = browser.page.evaluate(
            """() => {
                const main = document.querySelector(
                    'main, article, [class*="job-description" i], [class*="description" i], #content'
                );
                return (main || document.body).innerText;
            }"""
        )
    text = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
    if len(text) < 200:
        say("[yellow]Scraped very little text. Pass --jd-file with the posting instead.[/]")
    return text


def tailor_and_build(
    job_description: str, profile, db: DBManager, target: float | None = None
) -> tuple[TailoredResumeSchema, object, Path]:
    """Two-pass tailoring, few-shot from past wins, then PDF."""
    optimizer = ATSOptimizer(profile=profile)

    resume, keywords = optimizer.run(
        job_description,
        few_shot=lambda kw: db.successful_examples(kw.role_title),
        target=target,
    )

    pdf_path = PDFGenerator().generate(
        resume,
        profile,
        company=keywords.company or "Company",
        role_title=keywords.role_title,
        keep_html=True,
    )
    return resume, keywords, pdf_path


def print_tailoring_report(resume: TailoredResumeSchema, keywords, pdf_path: Path) -> None:
    rule("Tailoring result")
    target = settings.TARGET_MATCH_SCORE
    colour = "green" if resume.ats_match_percentage >= target else "yellow"
    say(f"Role         : {keywords.role_title} @ {keywords.company or 'unknown'}")
    say(f"ATS match    : [{colour}]{resume.ats_match_percentage:.1f}%[/] (target {target:.0f}%)")
    say(f"Resume PDF   : {pdf_path}")

    if resume.keywords_covered:
        say(f"\nCovered ({len(resume.keywords_covered)}): " + ", ".join(resume.keywords_covered[:18]))
    if resume.keywords_missing:
        say(f"\n[yellow]Genuine gaps ({len(resume.keywords_missing)}):[/] "
            + ", ".join(resume.keywords_missing[:18]))
        say("[dim]These are requirements your profile does not support. They were "
            "reported rather than invented.[/]")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_doctor(args) -> int:
    rule("Setup check")
    ok = True

    say(f"Python           : {sys.version.split()[0]}")
    if sys.version_info < (3, 11):
        say("  [red]Python 3.11+ required.[/]")
        ok = False

    for name, module in (
        ("playwright", "playwright"),
        ("weasyprint", "weasyprint"),
        ("anthropic", "anthropic"),
        ("sqlalchemy", "sqlalchemy"),
        ("cryptography", "cryptography"),
    ):
        try:
            __import__(module)
            say(f"{name:<17}: [green]installed[/]")
        except ImportError:
            say(f"{name:<17}: [red]missing[/] (pip install -r requirements.txt)")
            ok = False

    key_status = "env JP_ENCRYPTION_KEY" if settings.ENCRYPTION_KEY else str(settings.KEY_PATH)
    say(f"Vault key        : {key_status}")
    say(f"Database         : {settings.DB_URL}")
    say(f"LLM model        : {settings.LLM_MODEL} (effort={settings.LLM_EFFORT})")

    if settings.ANTHROPIC_API_KEY:
        say("Anthropic auth   : [green]ANTHROPIC_API_KEY set[/]")
    else:
        say("Anthropic auth   : [yellow]no ANTHROPIC_API_KEY[/] "
            "(the SDK will try an `ant auth login` profile)")

    say(f"Submit gate      : {'ON' if settings.REQUIRE_CONFIRM_BEFORE_SUBMIT else '[red]OFF[/]'}")
    say(f"Register gate    : {'ON' if settings.REQUIRE_CONFIRM_BEFORE_REGISTER else '[red]OFF[/]'}")

    rule("Master profile")
    try:
        profile = load_master_profile()
    except Exception as exc:
        say(f"[red]Invalid:[/] {exc}")
        return 1

    placeholders = find_placeholders(profile)
    if placeholders:
        say(f"[yellow]{len(placeholders)} placeholder value(s) still to fill:[/]")
        for path in placeholders[:20]:
            say(f"  - {path}")
        if len(placeholders) > 20:
            say(f"  ... and {len(placeholders) - 20} more")
        ok = False
    else:
        say("[green]All fields populated.[/]")
        say(f"  {len(profile.experience)} role(s), {len(profile.skills.flat)} skill(s), "
            f"{len(profile.education)} education entr(ies)")

    say("\n[green]Ready.[/]" if ok else "\n[yellow]Fix the items above before applying.[/]")
    return 0 if ok else 1


def cmd_tailor(args) -> int:
    profile = load_profile_or_exit(args.force)
    db = DBManager()
    jd = fetch_job_description(args.url, args.jd_file)
    resume, keywords, pdf_path = tailor_and_build(jd, profile, db, args.target)

    app = db.create_application(
        company=keywords.company or "Unknown",
        role_title=keywords.role_title,
        job_url=args.url,
        job_description=jd,
        resume_pdf_path=str(pdf_path),
        match_score=resume.ats_match_percentage,
        tailored_payload=resume.model_dump(),
    )
    print_tailoring_report(resume, keywords, pdf_path)
    say(f"\nSaved as application [bold]#{app.id}[/] (status: Draft)")
    say(f"Apply with: python main.py apply {args.url} --app-id {app.id}")
    return 0


def cmd_apply(args) -> int:
    from automation.ats_drivers import get_driver_class
    from automation.stealth_browser import (
        HumanBrowser,
        HumanDeclined,
        ManualInterventionRequired,
    )
    from engine.screener_mapper import ScreenerMapper

    profile = load_profile_or_exit(args.force)
    db = DBManager()

    # Reuse a prepared draft, or tailor now.
    if args.app_id:
        app = db.get_application(args.app_id)
        if app is None:
            say(f"[red]No application #{args.app_id}.[/]")
            return 2
        import json

        resume = TailoredResumeSchema.model_validate(json.loads(app.tailored_payload or "{}"))
        pdf_path = Path(app.resume_pdf_path or "")
        if not pdf_path.exists():
            say(f"[red]Resume PDF missing:[/] {pdf_path}. Re-run `tailor`.")
            return 2
        say(f"Using prepared application #{app.id}: {app.role_title} @ {app.company}")
    else:
        jd = fetch_job_description(args.url, args.jd_file)
        resume, keywords, pdf_path = tailor_and_build(jd, profile, db, args.target)
        print_tailoring_report(resume, keywords, pdf_path)
        app = db.create_application(
            company=keywords.company or "Unknown",
            role_title=keywords.role_title,
            job_url=args.url,
            job_description=jd,
            resume_pdf_path=str(pdf_path),
            match_score=resume.ats_match_percentage,
            tailored_payload=resume.model_dump(),
        )

    if resume.ats_match_percentage < settings.TARGET_MATCH_SCORE and not args.force:
        say(f"\n[yellow]Match score {resume.ats_match_percentage:.1f}% is below the "
            f"{settings.TARGET_MATCH_SCORE:.0f}% target.[/]")
        say("Re-run with --force to apply anyway, or strengthen your master profile.")
        return 1

    driver_class = get_driver_class(args.url)
    rule(f"Applying via {driver_class.NAME} driver")
    say("[dim]A browser window will open. Every irreversible step waits for you.[/]")

    mapper = ScreenerMapper(profile, resume.screener_answers)
    try:
        with HumanBrowser() as browser:
            driver = driver_class(browser, profile, mapper, db)
            outcome = driver.apply(args.url, pdf_path)

            shot = browser.screenshot(
                settings.OUTPUT_DIR / "screenshots" / f"app_{app.id}.png"
            )
            outcome.screenshot_path = str(shot)
    except (ManualInterventionRequired, HumanDeclined) as exc:
        say(f"\n[yellow]Stopped: {exc}[/]")
        db.update_application(app.id, notes=f"{app.notes}\nHalted: {exc}".strip())
        return 1
    except Exception as exc:
        say(f"\n[red]Automation error: {exc}[/]")
        db.update_application(app.id, notes=f"{app.notes}\nError: {exc}".strip())
        return 1

    rule("Outcome")
    say(f"Fields filled  : {outcome.fields_filled}")
    say(f"Resume uploaded: {'yes' if outcome.resume_uploaded else 'no'}")
    say(f"Account created: {'yes' if outcome.account_created else 'no'}")
    say(f"Screenshot     : {outcome.screenshot_path}")

    if outcome.escalations:
        say(f"\n[yellow]{len(outcome.escalations)} field(s) needed you:[/]")
        for esc in outcome.escalations:
            say(f"  - {esc.question[:66]}\n      {esc.reason}")

    if outcome.submitted:
        db.mark_submitted(app.id)
        say(f"\n[green]Submitted.[/] Application #{app.id} marked Applied.")
    else:
        say(f"\n[yellow]Not submitted.[/] Application #{app.id} stays Draft.")

    say(f"\nWhen you hear back: python main.py feedback --app-id {app.id} "
        "--status Interview --notes '...'")
    return 0


def cmd_review(args) -> int:
    db = DBManager()
    stats = db.stats()

    rule("Pipeline")
    say(f"Applications : {stats['total']}  (submitted {stats['submitted']})")
    say(f"Positive     : {stats['positive_outcomes']}  "
        f"(response rate {stats['response_rate']}%)")
    say(f"Avg match    : {stats['avg_match_score']}%")

    status = ApplicationStatus.from_str(args.status) if args.status else None
    apps = db.list_applications(status=status, limit=args.limit)
    if not apps:
        say("\n[dim]No applications yet. Start with: python main.py tailor <url>[/]")
        return 0

    if console and Table:
        table = Table(show_lines=False)
        for col in ("ID", "Company", "Role", "Match", "Status", "Created"):
            table.add_column(col)
        for app in apps:
            colour = "green" if app.match_score >= settings.TARGET_MATCH_SCORE else "yellow"
            table.add_row(
                str(app.id),
                app.company[:24],
                app.role_title[:34],
                f"[{colour}]{app.match_score:.0f}%[/]",
                app.status.value,
                app.created_at.strftime("%Y-%m-%d"),
            )
        console.print(table)
    else:
        for app in apps:
            print(f"#{app.id:<4} {app.company[:22]:<22} {app.role_title[:30]:<30} "
                  f"{app.match_score:>5.0f}%  {app.status.value}")

    if args.app_id:
        app = db.get_application(args.app_id)
        if app:
            rule(f"Application #{app.id}")
            say(f"URL   : {app.job_url}")
            say(f"Resume: {app.resume_pdf_path}")
            say(f"Notes :\n{app.notes or '(none)'}")
            for entry in app.feedback:
                say(f"  [{entry.created_at:%Y-%m-%d}] {entry.status.value}: {entry.notes}")
    return 0


def cmd_feedback(args) -> int:
    db = DBManager()
    try:
        status = ApplicationStatus.from_str(args.status)
    except ValueError as exc:
        say(f"[red]{exc}[/]")
        return 2

    app = db.get_application(args.app_id)
    if app is None:
        say(f"[red]No application #{args.app_id}.[/]")
        return 2

    db.record_feedback(args.app_id, status, args.notes or "")
    say(f"Recorded [bold]{status.value}[/] for #{args.app_id} "
        f"({app.role_title} @ {app.company}).")

    from database.models import POSITIVE_STATUSES

    if status in POSITIVE_STATUSES:
        examples = db.successful_examples(app.role_title)
        say(f"[green]This application now seeds the feedback loop.[/] "
            f"{len(examples)} proven example(s) available for similar roles.")
    return 0


def cmd_creds(args) -> int:
    db = DBManager()
    creds = db.list_credentials()
    if not creds:
        say("[dim]No stored credentials yet.[/]")
        return 0

    rule("Credential vault")
    for cred in creds:
        line = f"{cred.portal_domain:<38} {cred.username_email}"
        if args.show and (not args.portal or args.portal in cred.portal_domain):
            line += f"   {db.decrypt(cred.encrypted_password)}"
        say(line)

    if not args.show:
        say("\n[dim]Passwords hidden. Use --show (optionally --portal <domain>) to reveal.[/]")
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Autonomous ATS job application pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check dependencies and the master profile").set_defaults(
        func=cmd_doctor
    )

    for name, help_text, func in (
        ("tailor", "tailor a resume and build the PDF (no browser)", cmd_tailor),
        ("apply", "tailor if needed, then drive the application form", cmd_apply),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("url", help="job posting URL")
        p.add_argument("--jd-file", type=Path, help="read the posting from a file instead")
        p.add_argument("--target", type=float, help="override the target match score")
        p.add_argument("--force", action="store_true",
                       help="proceed despite placeholders or a low match score")
        if name == "apply":
            p.add_argument("--app-id", type=int, help="reuse a draft prepared by `tailor`")
        p.set_defaults(func=func)

    p = sub.add_parser("review", help="list applications and pipeline stats")
    p.add_argument("--status", help="filter by status (Applied, Interview, Rejected, ...)")
    p.add_argument("--app-id", type=int, help="show full detail for one application")
    p.add_argument("--limit", type=int, default=25)
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("feedback", help="record an outcome and feed the iteration engine")
    p.add_argument("--app-id", type=int, required=True)
    p.add_argument("--status", required=True,
                   help="Applied | Screening | Interview | Offer | Rejected | Ghosted | Withdrawn")
    p.add_argument("--notes", default="")
    p.set_defaults(func=cmd_feedback)

    p = sub.add_parser("creds", help="list stored portal credentials")
    p.add_argument("--show", action="store_true", help="reveal decrypted passwords")
    p.add_argument("--portal", help="only reveal this domain")
    p.set_defaults(func=cmd_creds)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        return args.func(args)
    except SystemExit as exc:
        # Keep main() callable in-process (tests, embedding) while behaving
        # identically at the command line.
        return int(exc.code or 0)
    except KeyboardInterrupt:
        say("\n[yellow]Interrupted.[/]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
