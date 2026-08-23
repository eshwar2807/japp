"""SQLAlchemy 2.0 schemas for the application pipeline.

  * users         - dashboard accounts; owns everything below
  * applications  - one row per submitted (or attempted) application
  * credentials   - encrypted per-portal login vault
  * feedback      - outcome events, used as few-shot fuel for future tailoring
  * action_items  - the human-in-the-loop queue (unmapped fields, CAPTCHAs, ...)
  * llm_usage     - per-call token spend, for the cost dashboard
  * run_logs      - structured run history surfaced in the UI

Every user-owned table carries a nullable ``user_id``. Nullable because the CLI
predates the dashboard and writes rows with no user attached; the web layer
always scopes its queries by user and never returns another user's rows.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ApplicationStatus(str, enum.Enum):
    """Lifecycle of a single application."""

    DRAFT = "Draft"            # tailored, PDF built, not yet submitted
    APPLIED = "Applied"
    SCREENING = "Screening"
    INTERVIEW = "Interview"
    OFFER = "Offer"
    REJECTED = "Rejected"
    GHOSTED = "Ghosted"
    WITHDRAWN = "Withdrawn"

    @classmethod
    def from_str(cls, value: str) -> "ApplicationStatus":
        for member in cls:
            if member.value.lower() == str(value).strip().lower():
                return member
        raise ValueError(
            f"Unknown status {value!r}. Valid: {', '.join(m.value for m in cls)}"
        )


#: Statuses that count as a positive signal when mining few-shot examples.
POSITIVE_STATUSES = (
    ApplicationStatus.SCREENING,
    ApplicationStatus.INTERVIEW,
    ApplicationStatus.OFFER,
)


class Application(Base):
    __tablename__ = "applications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )
    company: Mapped[str] = mapped_column(String(255), index=True)
    role_title: Mapped[str] = mapped_column(String(255), index=True)
    job_url: Mapped[str] = mapped_column(String(1024))
    job_description: Mapped[str] = mapped_column(Text, default="")

    resume_pdf_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # Full TailoredResumeSchema JSON, so past bullets can be mined verbatim.
    tailored_payload: Mapped[str | None] = mapped_column(Text, nullable=True)

    match_score: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[ApplicationStatus] = mapped_column(
        SAEnum(ApplicationStatus, values_callable=lambda e: [m.value for m in e]),
        default=ApplicationStatus.DRAFT,
        index=True,
    )
    portal_domain: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    notes: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    feedback: Mapped[list["Feedback"]] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
        order_by="Feedback.created_at",
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id", "company", "role_title", "job_url", name="uq_application_posting"
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Application #{self.id} {self.company} / {self.role_title} "
            f"{self.status.value} {self.match_score:.1f}%>"
        )


class Credential(Base):
    """Encrypted portal logins. `encrypted_password` is a Fernet token.

    The plaintext password is never written to this table, to logs, or to disk.
    """

    __tablename__ = "credentials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )
    portal_domain: Mapped[str] = mapped_column(String(255), index=True)
    username_email: Mapped[str] = mapped_column(String(320))
    encrypted_password: Mapped[bytes] = mapped_column()
    notes: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id", "portal_domain", "username_email", name="uq_credential_identity"
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Credential {self.username_email}@{self.portal_domain}>"


class Feedback(Base):
    """An outcome event on an application. Drives the iteration engine."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )
    application_id: Mapped[int] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[ApplicationStatus] = mapped_column(
        SAEnum(ApplicationStatus, values_callable=lambda e: [m.value for m in e])
    )
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    application: Mapped[Application] = relationship(back_populates="feedback")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Feedback app={self.application_id} {self.status.value}>"


# ==========================================================================
# Dashboard tables
# ==========================================================================


class User(Base):
    """A dashboard account.

    ``password_hash`` is Argon2id. ``api_key_hash`` is a SHA-256 of the raw API
    key - the raw key is shown exactly once at creation and is not recoverable.
    ``encrypted_anthropic_key`` is a Fernet token; the plaintext provider key is
    never stored, logged, or rendered back to the browser.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))

    api_key_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    api_key_prefix: Mapped[str | None] = mapped_column(String(16), nullable=True)
    api_key_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    encrypted_anthropic_key: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    #: The user's master profile, as JSON. Same schema as master_profile.json.
    profile_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: A resume import awaiting review. Never treated as profile data until the
    #: user saves the form, because extraction can misread a date or employer.
    pending_profile_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    #: Grants the /admin view. The first account created becomes admin, or set
    #: JP_ADMIN_EMAIL to nominate one.
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    failed_logins: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    #: Bumped on password change / "log out everywhere"; invalidates old sessions.
    session_epoch: Mapped[int] = mapped_column(Integer, default=1)

    #: Notification preferences. The webhook is opt-in and user-supplied; only
    #: a job reference and the reason are ever sent, never form or profile data.
    notify_desktop: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_webhook_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    #: Suppress repeat notifications for the same batch within this many seconds.
    notify_quiet_seconds: Mapped[int] = mapped_column(Integer, default=120)
    #: "immediate" alerts on every block; "digest" stays silent all day and
    #: sends one summary, which is what makes a 100-a-day run bearable.
    notify_mode: Mapped[str] = mapped_column(String(16), default="immediate")
    notify_digest_hour: Mapped[int] = mapped_column(Integer, default=18)
    notify_utc_offset_minutes: Mapped[int] = mapped_column(Integer, default=0)
    last_digest_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    #: Per-user overrides for the tiered models and the daily spend ceiling.
    model_bulk: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_priority: Mapped[str | None] = mapped_column(String(64), nullable=True)
    daily_spend_cap_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User #{self.id} {self.email}>"


class ActionKind(str, enum.Enum):
    """Why a run stopped and needs a human."""

    UNMAPPED_FIELD = "Unmapped field"
    CAPTCHA = "Human verification"
    ACCOUNT_CREATION = "Account creation"
    SUBMIT_CONFIRMATION = "Submit confirmation"
    LOGIN_REQUIRED = "Login required"
    ERROR = "Error"


class ActionStatus(str, enum.Enum):
    OPEN = "Open"
    ANSWERED = "Answered"
    DISMISSED = "Dismissed"


class ActionItem(Base):
    """One thing the pipeline could not do safely on its own.

    This is the queue the dashboard surfaces: unanswerable screener questions,
    human-verification challenges, account-creation steps. Answers submitted
    here are written back into the user's profile knowledge so the same question
    is never asked twice.
    """

    __tablename__ = "action_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )
    application_id: Mapped[int | None] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"), nullable=True, index=True
    )

    kind: Mapped[ActionKind] = mapped_column(
        SAEnum(ActionKind, values_callable=lambda e: [m.value for m in e]),
        default=ActionKind.UNMAPPED_FIELD,
        index=True,
    )
    status: Mapped[ActionStatus] = mapped_column(
        SAEnum(ActionStatus, values_callable=lambda e: [m.value for m in e]),
        default=ActionStatus.OPEN,
        index=True,
    )

    question: Mapped[str] = mapped_column(Text, default="")
    reason: Mapped[str] = mapped_column(Text, default="")
    field_type: Mapped[str] = mapped_column(String(32), default="text")
    #: JSON list of permitted values for selects/radios.
    options_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    required: Mapped[bool] = mapped_column(Boolean, default=False)

    answer: Mapped[str] = mapped_column(Text, default="")
    #: When true, the answer is folded back into the master profile.
    remember: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    application: Mapped["Application | None"] = relationship()

    __table_args__ = (Index("ix_action_open", "user_id", "status"),)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ActionItem #{self.id} {self.kind.value} {self.status.value}>"


class LLMUsage(Base):
    """One Claude API call. Powers the cost dashboard."""

    __tablename__ = "llm_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )
    application_id: Mapped[int | None] = mapped_column(
        ForeignKey("applications.id", ondelete="SET NULL"), nullable=True, index=True
    )

    model: Mapped[str] = mapped_column(String(64), index=True)
    #: Which pipeline step: 'extract_keywords', 'tailor', ...
    phase: Mapped[str] = mapped_column(String(32), default="")

    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0)

    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    __table_args__ = (Index("ix_usage_user_day", "user_id", "created_at"),)

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<LLMUsage {self.model} {self.phase} ${self.cost_usd:.4f}>"


class LogLevel(str, enum.Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class RunLog(Base):
    """Structured pipeline events, surfaced as the live log in the dashboard."""

    __tablename__ = "run_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )
    application_id: Mapped[int | None] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"), nullable=True, index=True
    )

    level: Mapped[LogLevel] = mapped_column(
        SAEnum(LogLevel, values_callable=lambda e: [m.value for m in e]),
        default=LogLevel.INFO,
        index=True,
    )
    event: Mapped[str] = mapped_column(String(64), default="")
    message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    __table_args__ = (Index("ix_log_user_time", "user_id", "created_at"),)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RunLog {self.level.value} {self.event}>"


# ==========================================================================
# Batch queue
# ==========================================================================


class JobStatus(str, enum.Enum):
    """Lifecycle of one queued pipeline job.

    BLOCKED is the state that makes batching work: the job has hit something
    only a human can resolve, has released its worker slot, and is waiting.
    READY means the human has answered and the job may resume.
    """

    QUEUED = "Queued"
    RUNNING = "Running"
    BLOCKED = "Blocked"
    READY = "Ready"
    DONE = "Done"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


#: Terminal states; a job in one of these is never picked up again.
TERMINAL_JOB_STATUSES = (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)


class BlockMode(str, enum.Enum):
    """What a blocked job needs in order to continue.

    NEEDS_ANSWER  - a value the pipeline could not determine. The browser is
                    closed and the job resumes from scratch once answered,
                    which is cheap and survives a restart.
    NEEDS_BROWSER - a human-verification or login challenge that must be
                    cleared in the live window. The session is held open, so
                    these are capped.
    """

    NONE = "None"
    NEEDS_ANSWER = "Needs answer"
    NEEDS_BROWSER = "Needs browser"


class RunJob(Base):
    """One unit of queued work: tailor a posting, or apply to one."""

    __tablename__ = "run_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    application_id: Mapped[int | None] = mapped_column(
        ForeignKey("applications.id", ondelete="SET NULL"), nullable=True, index=True
    )
    #: Groups jobs enqueued together, so a batch can be tracked as a unit.
    batch_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    #: Priority jobs use the expensive model; bulk discovery jobs the cheap one.
    priority: Mapped[bool] = mapped_column(Boolean, default=False)

    kind: Mapped[str] = mapped_column(String(16), default="tailor")   # tailor | apply
    job_url: Mapped[str] = mapped_column(String(1024), default="")
    job_description: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus, values_callable=lambda e: [m.value for m in e]),
        default=JobStatus.QUEUED,
        index=True,
    )
    block_mode: Mapped[BlockMode] = mapped_column(
        SAEnum(BlockMode, values_callable=lambda e: [m.value for m in e]),
        default=BlockMode.NONE,
    )
    #: The action item the human must resolve before this job can continue.
    blocking_action_id: Mapped[int | None] = mapped_column(
        ForeignKey("action_items.id", ondelete="SET NULL"), nullable=True
    )
    #: True while a live browser context is held open for this job.
    holds_browser: Mapped[bool] = mapped_column(Boolean, default=False)

    message: Mapped[str] = mapped_column(Text, default="")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    position: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    blocked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    application: Mapped["Application | None"] = relationship()
    blocking_action: Mapped["ActionItem | None"] = relationship()

    __table_args__ = (Index("ix_job_user_status", "user_id", "status"),)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_JOB_STATUSES

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RunJob #{self.id} {self.kind} {self.status.value}>"


class Notification(Base):
    """A dispatched notification, kept so the UI can show what was sent."""

    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    job_id: Mapped[int | None] = mapped_column(
        ForeignKey("run_jobs.id", ondelete="SET NULL"), nullable=True
    )
    channel: Mapped[str] = mapped_column(String(24), default="desktop")
    title: Mapped[str] = mapped_column(String(255), default="")
    body: Mapped[str] = mapped_column(Text, default="")
    delivered: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Notification {self.channel} {'ok' if self.delivered else 'failed'}>"
