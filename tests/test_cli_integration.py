"""Module D validation: the full CLI loop with a stubbed LLM.

Proves the closed loop actually closes - an application marked Interview shows
up as a few-shot example inside the next tailoring prompt.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import main as cli
from config import settings
from database.db_manager import DBManager
from database.models import ApplicationStatus
from engine.ats_optimizer import ATSOptimizer
from engine.schemas import ExperienceBlock, JDKeywords, ScreenerAnswer, TailoredResumeDraft

JD_TEXT = """
Senior Backend Engineer - Acme Robotics
Python, distributed systems, REST API design, CI/CD. PostgreSQL and Kubernetes.
Strong communication and mentoring.
"""

FILLED_PROFILE = {
    "contact": {
        "full_name": "Ada Lovelace",
        "preferred_name": "Ada",
        "email": "ada@example.com",
        "phone": "+1-555-010-0100",
        "location": {"city": "Austin", "state": "TX", "country": "United States",
                     "postal_code": "78701", "willing_to_relocate": True},
        "links": {"linkedin": "https://linkedin.com/in/ada"},
    },
    "summary": "Backend engineer.",
    "skills": {"hard": ["Python", "Distributed Systems"], "tooling": ["PostgreSQL", "Kubernetes"],
               "soft": ["Mentoring"]},
    "experience": [{
        "company": "Analytical Engines", "title": "Principal Engineer",
        "location": "Austin, TX", "start_date": "2015-01", "is_current": True,
        "bullets": ["Cut p99 latency 60% across 14 services."],
        "tech_used": ["Python", "Kubernetes"],
    }],
    "education": [{"institution": "University of London", "degree": "BSc",
                   "field_of_study": "Mathematics", "end_date": "1842-05"}],
    "certifications": [],
    "projects": [],
    "legal": {"work_authorization_us": "Yes", "requires_sponsorship_now_or_future": "No",
              "desired_salary": "185000", "earliest_start_date": "2026-09-01"},
    "voluntary_disclosures": {"gender": "Decline to self-identify"},
    "preferences": {"how_did_you_hear_about_us": "Company careers page"},
}

KEYWORDS = JDKeywords(
    role_title="Senior Backend Engineer",
    company="Acme Robotics",
    hard_skills=["Python", "distributed systems", "REST API design", "CI/CD"],
    soft_skills=["communication", "mentoring"],
    tooling=["PostgreSQL", "Kubernetes"],
)

STRONG_DRAFT = TailoredResumeDraft(
    summary="Principal engineer in distributed systems and REST API design.",
    highlighted_skills=["Python", "Kubernetes", "PostgreSQL", "Distributed Systems", "Mentoring"],
    tailored_experience=[
        ExperienceBlock(
            company="Analytical Engines", title="Senior Backend Engineer",
            location="Austin, TX", start_date="2015-01", end_date="Present",
            bullets=[
                "Cut p99 latency 60% across 14 Python services on Kubernetes.",
                "Owned CI/CD and PostgreSQL migrations for distributed systems.",
                "Drove REST API design reviews; mentoring and communication across four teams.",
            ],
        )
    ],
    ats_match_percentage=92.0,
    screener_answers=[ScreenerAnswer(question="Why Acme?", answer="The robotics work.")],
)


@pytest.fixture()
def cli_env(tmp_path, monkeypatch):
    """Isolated profile, database, vault and output directory for the CLI."""
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(FILLED_PROFILE))

    monkeypatch.setattr(settings, "MASTER_PROFILE_PATH", profile_path)
    monkeypatch.setattr(settings, "RESUME_DIR", tmp_path / "resumes")
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(settings, "DB_URL", f"sqlite:///{tmp_path/'cli.db'}")
    monkeypatch.setattr(settings, "KEY_PATH", tmp_path / "vault.key")
    monkeypatch.setattr(settings, "ENCRYPTION_KEY", None)

    jd_file = tmp_path / "jd.txt"
    jd_file.write_text(JD_TEXT)

    prompts: list[str] = []

    class StubMessages:
        def parse(self, **kwargs):
            prompts.append(kwargs["messages"][0]["content"])
            payload = KEYWORDS if kwargs["output_format"] is JDKeywords else STRONG_DRAFT
            return SimpleNamespace(parsed_output=payload, stop_reason="end_turn")

    monkeypatch.setattr(
        ATSOptimizer, "client", property(lambda self: StubMessages())
    )
    return SimpleNamespace(tmp=tmp_path, jd_file=jd_file, prompts=prompts)


# ---------------- tailor ----------------


def test_tailor_builds_pdf_and_logs_the_application(cli_env, capsys):
    code = cli.main(
        ["tailor", "https://boards.greenhouse.io/acme/jobs/1", "--jd-file", str(cli_env.jd_file)]
    )
    assert code == 0

    out = capsys.readouterr().out
    assert "Senior Backend Engineer" in out
    assert "ATS match" in out

    apps = DBManager(db_url=settings.DB_URL).list_applications()
    assert len(apps) == 1
    app = apps[0]
    assert app.company == "Acme Robotics"
    assert app.status is ApplicationStatus.DRAFT
    assert app.match_score >= 80.0
    assert app.resume_pdf_path and app.resume_pdf_path.endswith(".pdf")

    from pathlib import Path

    pdf = Path(app.resume_pdf_path)
    assert pdf.exists() and pdf.read_bytes()[:5] == b"%PDF-"


def test_tailor_refuses_a_profile_with_placeholders(cli_env, monkeypatch, capsys):
    bad = cli_env.tmp / "bad.json"
    bad.write_text(json.dumps({**FILLED_PROFILE,
                               "contact": {**FILLED_PROFILE["contact"],
                                           "full_name": "<YOUR NAME>"}}))
    monkeypatch.setattr(settings, "MASTER_PROFILE_PATH", bad)

    code = cli.main(["tailor", "https://x.com/1", "--jd-file", str(cli_env.jd_file)])
    assert code == 2
    assert "placeholder" in capsys.readouterr().out.lower()


# ---------------- review & feedback ----------------


def test_review_reports_stats(cli_env, capsys):
    cli.main(["tailor", "https://boards.greenhouse.io/acme/jobs/1",
              "--jd-file", str(cli_env.jd_file)])
    capsys.readouterr()

    assert cli.main(["review"]) == 0
    out = capsys.readouterr().out
    assert "Applications" in out and "Acme Robotics" in out


def test_feedback_updates_status(cli_env, capsys):
    cli.main(["tailor", "https://boards.greenhouse.io/acme/jobs/1",
              "--jd-file", str(cli_env.jd_file)])
    capsys.readouterr()

    assert cli.main(["feedback", "--app-id", "1", "--status", "Interview",
                     "--notes", "Recruiter screen Tuesday"]) == 0
    out = capsys.readouterr().out
    assert "Interview" in out

    app = DBManager(db_url=settings.DB_URL).get_application(1)
    assert app.status is ApplicationStatus.INTERVIEW
    assert "Recruiter screen Tuesday" in app.notes


def test_feedback_rejects_unknown_status(cli_env, capsys):
    cli.main(["tailor", "https://x.com/1", "--jd-file", str(cli_env.jd_file)])
    capsys.readouterr()
    assert cli.main(["feedback", "--app-id", "1", "--status", "Pending"]) == 2


def test_feedback_rejects_unknown_application(cli_env):
    assert cli.main(["feedback", "--app-id", "999", "--status", "Interview"]) == 2


# ---------------- the closed loop ----------------


def test_interview_outcome_feeds_the_next_tailoring_prompt(cli_env):
    """The whole point of the feedback engine."""
    cli.main(["tailor", "https://boards.greenhouse.io/acme/jobs/1",
              "--jd-file", str(cli_env.jd_file)])
    cli.main(["feedback", "--app-id", "1", "--status", "Interview", "--notes", "Screen booked"])

    cli_env.prompts.clear()
    cli.main(["tailor", "https://boards.greenhouse.io/acme/jobs/2",
              "--jd-file", str(cli_env.jd_file)])

    tailor_prompt = cli_env.prompts[1]  # [0] is the extraction pass
    assert "PROVEN_EXAMPLES" in tailor_prompt
    assert "Interview" in tailor_prompt
    assert "Cut p99 latency 60%" in tailor_prompt


def test_rejected_outcome_does_not_feed_the_next_prompt(cli_env):
    cli.main(["tailor", "https://boards.greenhouse.io/acme/jobs/1",
              "--jd-file", str(cli_env.jd_file)])
    cli.main(["feedback", "--app-id", "1", "--status", "Rejected"])

    cli_env.prompts.clear()
    cli.main(["tailor", "https://boards.greenhouse.io/acme/jobs/2",
              "--jd-file", str(cli_env.jd_file)])

    assert "PROVEN_EXAMPLES" not in cli_env.prompts[1]


# ---------------- credentials ----------------


def test_creds_hides_passwords_by_default(cli_env, capsys):
    db = DBManager(db_url=settings.DB_URL)
    db.upsert_credential("acme.wd1.myworkdayjobs.com", "ada@example.com", "S3cret!Pass123")

    cli.main(["creds"])
    out = capsys.readouterr().out
    assert "acme.wd1.myworkdayjobs.com" in out
    assert "S3cret!Pass123" not in out

    cli.main(["creds", "--show"])
    assert "S3cret!Pass123" in capsys.readouterr().out
