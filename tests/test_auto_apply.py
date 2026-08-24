"""Auto-apply: queue the browser step above a match threshold.

The property that must never break: auto-queueing decides only whether a
browser *opens*. Submission still requires explicit approval on every single
application, so a threshold set too low costs wasted browser time, never an
unreviewed application reaching an employer.
"""

from __future__ import annotations

import pytest

from config import settings
from database.models import JobStatus


#: Passes the hard pre-screen: Java, under the years limit, no clearance, no
#: sponsorship refusal. Fixtures need one or tailoring is skipped before it runs.
JAVA_JD = (
    "Senior Java Engineer. Spring Boot microservices, REST APIs, Kubernetes. "
    "4+ years of experience. Remote US."
)


@pytest.fixture()
def user(db):
    user = db.create_user("ada@example.com", "$argon2id$fake")
    db.set_anthropic_key(user.id, "sk-ant-test")
    db.save_profile(user.id, {
        "contact": {"full_name": "Ada", "email": "a@b.com",
                    "location": {"city": "Austin"}},
        "summary": "Engineer.", "skills": {"hard": ["Python"]},
        "experience": [{"company": "Acme", "title": "Engineer",
                        "start_date": "2020-01", "is_current": True,
                        "bullets": ["Did a thing."]}],
        "education": [{"institution": "UoL"}],
        "legal": {"work_authorization_us": "Yes"},
    })
    return db.get_user(user.id)


def _tailor(db, user, score, monkeypatch, tmp_path, url="https://x.com/j/1"):
    """Run a tailor job whose result scores `score`."""
    from engine.schemas import ExperienceBlock, JDKeywords, TailoredResumeSchema
    from web import runner

    # Distinct per URL so the unique-posting constraint is not tripped.
    keywords = JDKeywords(role_title=f"Backend Engineer {url[-1]}", company="Acme")
    resume = TailoredResumeSchema(
        summary="Engineer.", highlighted_skills=["Python"],
        tailored_experience=[ExperienceBlock(company="Acme", title="Engineer",
                                             start_date="2020-01", end_date="Present",
                                             bullets=["Did a thing."])],
        ats_match_percentage=score)

    class FakeOptimizer:
        def __init__(self, **kw):
            pass

        def run(self, jd, few_shot=(), target=None, max_iterations=None):
            return resume, keywords

    class FakePDF:
        def generate(self, *a, **kw):
            path = tmp_path / "r.pdf"
            path.write_bytes(b"%PDF-1.4\n" + b"x" * 2000)
            return path

    monkeypatch.setattr("engine.ats_optimizer.ATSOptimizer", FakeOptimizer)
    monkeypatch.setattr("engine.pdf_generator.PDFGenerator", FakePDF)

    job = db.enqueue_job(user.id, kind="tailor", job_url=url,
                         job_description=JAVA_JD)
    runner.run_tailor_job(db, job.id, user.id, gatekeeper=None)
    return job


def _apply_jobs(db, user):
    return [j for j in db.list_jobs(user_id=user.id) if j.kind == "apply"]


# ---------------- the threshold ----------------


def test_a_high_scoring_application_queues_itself(db, user, monkeypatch, tmp_path):
    db.update_user(user.id, auto_apply_threshold=85.0)
    _tailor(db, user, 91.4, monkeypatch, tmp_path)

    jobs = _apply_jobs(db, user)
    assert len(jobs) == 1
    assert jobs[0].status is JobStatus.QUEUED
    assert jobs[0].application_id is not None


def test_a_low_scoring_application_stays_a_draft(db, user, monkeypatch, tmp_path):
    db.update_user(user.id, auto_apply_threshold=85.0)
    _tailor(db, user, 62.0, monkeypatch, tmp_path)

    assert _apply_jobs(db, user) == []
    application = db.list_applications(user_id=user.id)[0]
    assert application.status.value == "Draft"


def test_a_score_exactly_on_the_threshold_queues(db, user, monkeypatch, tmp_path):
    db.update_user(user.id, auto_apply_threshold=85.0)
    _tailor(db, user, 85.0, monkeypatch, tmp_path)
    assert len(_apply_jobs(db, user)) == 1


def test_nothing_queues_when_auto_apply_is_off(db, user, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "AUTO_APPLY_THRESHOLD", 0.0)
    db.update_user(user.id, auto_apply_threshold=None)
    _tailor(db, user, 99.0, monkeypatch, tmp_path)

    assert _apply_jobs(db, user) == []


def test_a_threshold_of_zero_means_off_not_always(db, user, monkeypatch, tmp_path):
    """0 must disable the feature, not queue every application."""
    monkeypatch.setattr(settings, "AUTO_APPLY_THRESHOLD", 0.0)
    db.update_user(user.id, auto_apply_threshold=0.0)
    assert db.auto_apply_threshold(user.id) is None

    _tailor(db, user, 10.0, monkeypatch, tmp_path)
    assert _apply_jobs(db, user) == []


def test_the_decision_is_recorded_either_way(db, user, monkeypatch, tmp_path):
    db.update_user(user.id, auto_apply_threshold=85.0)
    _tailor(db, user, 91.4, monkeypatch, tmp_path)
    events = {log.event for log in db.list_logs(user_id=user.id, limit=50)}
    assert "auto_queued" in events

    _tailor(db, user, 40.0, monkeypatch, tmp_path, url="https://x.com/j/2")
    events = {log.event for log in db.list_logs(user_id=user.id, limit=50)}
    assert "below_threshold" in events


def test_a_per_user_threshold_overrides_the_instance_default(db, user, monkeypatch):
    monkeypatch.setattr(settings, "AUTO_APPLY_THRESHOLD", 90.0)
    db.update_user(user.id, auto_apply_threshold=70.0)
    assert db.auto_apply_threshold(user.id) == 70.0


# ---------------- the guarantee ----------------


def test_auto_queued_work_is_still_gated_before_submission(db, user, monkeypatch, tmp_path):
    """Auto-apply opens a browser. It does not submit.

    The submit gate is what stands between an automated run and an employer,
    so it must be unaffected by this setting.
    """
    from automation.gatekeeper import AutoDeclineGatekeeper
    from web.queue_worker import ParkRegistry, QueueGatekeeper
    from automation.notifier import Notifier

    db.update_user(user.id, auto_apply_threshold=85.0)
    _tailor(db, user, 99.0, monkeypatch, tmp_path)
    job = _apply_jobs(db, user)[0]

    # Unattended, the gate declines rather than submitting.
    assert AutoDeclineGatekeeper().confirm("SUBMIT this application") is False

    # Supervised, it blocks and waits for a human instead of proceeding.
    keeper = QueueGatekeeper(db, user.id, job.id, ParkRegistry(),
                             Notifier(db, channels=[]), lambda: None, lambda: None,
                             application_id=job.application_id,
                             wait_seconds=0.3, poll_seconds=0.05)
    assert keeper.confirm("SUBMIT this application") is False   # times out unanswered
    blocked = db.get_job(job.id)
    assert blocked.status is JobStatus.BLOCKED


def test_auto_queued_jobs_are_apply_kind_so_the_server_will_not_run_them(db, user,
                                                                        monkeypatch, tmp_path):
    """A hosted worker must leave these for the local agent."""
    db.update_user(user.id, auto_apply_threshold=85.0)
    _tailor(db, user, 95.0, monkeypatch, tmp_path)

    assert db.claim_next_job(kinds=("tailor", "discover")) is None
    assert db.claim_next_job(kinds=("apply",)) is not None
