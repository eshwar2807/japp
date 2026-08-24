"""CRUD layer + encrypted credential vault.

The vault key is a Fernet key resolved in this order:
  1. settings.ENCRYPTION_KEY  (env JP_ENCRYPTION_KEY)
  2. settings.KEY_PATH        (file, chmod 600)
  3. generated and written to settings.KEY_PATH on first use

Losing the key means the stored passwords are unrecoverable, which is the point.
"""

from __future__ import annotations

import json
import os
import secrets
import string
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Sequence
from urllib.parse import urlparse

import logging

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from config import settings
from database.models import (
    POSITIVE_STATUSES,
    TERMINAL_JOB_STATUSES,
    ActionItem,
    BlockMode,
    ActionKind,
    ActionStatus,
    Application,
    ApplicationStatus,
    Base,
    Credential,
    Feedback,
    JobStatus,
    LLMUsage,
    LogLevel,
    Notification,
    RunJob,
    RunLog,
    User,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Key management
# --------------------------------------------------------------------------


def load_or_create_key(key_path: Path | None = None) -> bytes:
    """Return the Fernet key, generating and persisting one if needed."""
    if settings.ENCRYPTION_KEY:
        return settings.ENCRYPTION_KEY.encode()

    path = Path(key_path or settings.KEY_PATH)
    if path.exists():
        return path.read_bytes().strip()

    key = Fernet.generate_key()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create with 0600 from the start; never widen it afterwards.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(key)
    return key


def generate_password(length: int | None = None) -> str:
    """Cryptographically secure password with guaranteed character-class coverage.

    Portals commonly demand upper + lower + digit + symbol; we guarantee one of
    each and shuffle, so generated passwords never fail a complexity check.
    """
    length = length or settings.GENERATED_PASSWORD_LENGTH
    if length < 8:
        raise ValueError("Password length must be at least 8.")

    # Symbols restricted to ones ATS portals reliably accept.
    symbols = "!@#$%^&*-_=+?"
    pools = (string.ascii_uppercase, string.ascii_lowercase, string.digits, symbols)
    alphabet = "".join(pools)

    chars = [secrets.choice(pool) for pool in pools]
    chars += [secrets.choice(alphabet) for _ in range(length - len(pools))]
    # Fisher-Yates with a CSPRNG; random.shuffle is not cryptographically secure.
    for i in range(len(chars) - 1, 0, -1):
        j = secrets.randbelow(i + 1)
        chars[i], chars[j] = chars[j], chars[i]
    return "".join(chars)


def normalize_domain(url_or_domain: str) -> str:
    """`https://boards.greenhouse.io/acme/jobs/123` -> `boards.greenhouse.io`."""
    value = (url_or_domain or "").strip()
    if not value:
        return ""
    if "://" not in value:
        value = "https://" + value
    host = (urlparse(value).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


# --------------------------------------------------------------------------
# Manager
# --------------------------------------------------------------------------


class DBManager:
    """Single entry point for persistence. Safe to construct per-process."""

    def __init__(self, db_url: str | None = None, key: bytes | None = None) -> None:
        self.db_url = db_url or settings.DB_URL
        if self.db_url.startswith("sqlite:///"):
            Path(self.db_url.removeprefix("sqlite:///")).parent.mkdir(
                parents=True, exist_ok=True
            )
        self.engine = create_engine(self.db_url, future=True)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._fernet = Fernet(key or load_or_create_key())
        Base.metadata.create_all(self.engine)
        # create_all does not alter existing tables, so a database written by an
        # earlier version is missing every column added since. Close that gap
        # before anything queries it.
        from database.migrate import add_missing_columns

        added = add_missing_columns(self.engine, Base.metadata)
        if added:
            log.info("Schema updated: added %s", ", ".join(added))

    @contextmanager
    def session(self) -> Iterator[Session]:
        sess = self._session_factory()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    # ---------------- Credentials ----------------

    def encrypt(self, plaintext: str) -> bytes:
        return self._fernet.encrypt(plaintext.encode())

    def decrypt(self, token: bytes) -> str:
        try:
            return self._fernet.decrypt(token).decode()
        except InvalidToken as exc:
            raise RuntimeError(
                "Could not decrypt credential: wrong or missing vault key "
                f"(expected at {settings.KEY_PATH})."
            ) from exc

    def get_credential(
        self, portal: str, username: str | None = None, user_id: int | None = None
    ) -> Credential | None:
        domain = normalize_domain(portal) or portal
        with self.session() as sess:
            stmt = select(Credential).where(Credential.portal_domain == domain)
            if username:
                stmt = stmt.where(Credential.username_email == username)
            if user_id is not None:
                stmt = stmt.where(Credential.user_id == user_id)
            return sess.scalars(stmt.order_by(Credential.created_at.desc())).first()

    def get_password(self, portal: str, username: str | None = None) -> str | None:
        cred = self.get_credential(portal, username)
        return self.decrypt(cred.encrypted_password) if cred else None

    def upsert_credential(
        self,
        portal: str,
        username: str,
        password: str,
        notes: str = "",
        user_id: int | None = None,
    ) -> Credential:
        """Store or rotate a credential. Returns the persisted row."""
        domain = normalize_domain(portal) or portal
        token = self.encrypt(password)
        with self.session() as sess:
            stmt = select(Credential).where(
                Credential.portal_domain == domain,
                Credential.username_email == username,
            )
            if user_id is not None:
                stmt = stmt.where(Credential.user_id == user_id)
            existing = sess.scalars(stmt).first()
            if existing:
                existing.encrypted_password = token
                if notes:
                    existing.notes = notes
                sess.flush()
                return existing
            cred = Credential(
                portal_domain=domain,
                username_email=username,
                encrypted_password=token,
                notes=notes,
                user_id=user_id,
            )
            sess.add(cred)
            sess.flush()
            return cred

    def get_or_create_credential(
        self,
        portal: str,
        username: str,
        length: int | None = None,
        user_id: int | None = None,
    ) -> tuple[Credential, str, bool]:
        """Return (row, plaintext_password, created_now).

        `created_now=True` means no account existed for this portal, so the
        caller must register one using the returned password.
        """
        existing = self.get_credential(portal, username, user_id=user_id)
        if existing:
            return existing, self.decrypt(existing.encrypted_password), False
        password = generate_password(length)
        return (
            self.upsert_credential(portal, username, password, user_id=user_id),
            password,
            True,
        )

    def list_credentials(self, user_id: int | None = None) -> Sequence[Credential]:
        with self.session() as sess:
            stmt = select(Credential).order_by(Credential.portal_domain)
            if user_id is not None:
                stmt = stmt.where(Credential.user_id == user_id)
            return sess.scalars(stmt).all()

    # ---------------- Applications ----------------

    def create_application(
        self,
        company: str,
        role_title: str,
        job_url: str,
        job_description: str = "",
        resume_pdf_path: str | None = None,
        match_score: float = 0.0,
        status: ApplicationStatus = ApplicationStatus.DRAFT,
        tailored_payload: dict | None = None,
        notes: str = "",
        user_id: int | None = None,
        salary_min: float | None = None,
        salary_max: float | None = None,
    ) -> Application:
        with self.session() as sess:
            app = Application(
                company=company,
                role_title=role_title,
                job_url=job_url,
                job_description=job_description,
                resume_pdf_path=resume_pdf_path,
                match_score=match_score,
                status=status,
                portal_domain=normalize_domain(job_url),
                tailored_payload=json.dumps(tailored_payload) if tailored_payload else None,
                notes=notes,
                user_id=user_id,
                salary_min=salary_min,
                salary_max=salary_max,
            )
            sess.add(app)
            sess.flush()
            return app

    def get_application(self, app_id: int, user_id: int | None = None) -> Application | None:
        # Eager-load feedback: callers use the object after the session closes,
        # and a lazy load on a detached instance raises DetachedInstanceError.
        with self.session() as sess:
            stmt = (
                select(Application)
                .where(Application.id == app_id)
                .options(selectinload(Application.feedback))
            )
            if user_id is not None:
                stmt = stmt.where(Application.user_id == user_id)
            return sess.scalars(stmt).first()

    def list_applications(
        self,
        status: ApplicationStatus | None = None,
        limit: int = 100,
        user_id: int | None = None,
    ) -> Sequence[Application]:
        with self.session() as sess:
            stmt = (
                select(Application)
                .options(selectinload(Application.feedback))
                .order_by(Application.created_at.desc())
                .limit(limit)
            )
            if status:
                stmt = stmt.where(Application.status == status)
            if user_id is not None:
                stmt = stmt.where(Application.user_id == user_id)
            return sess.scalars(stmt).all()

    def update_application(self, app_id: int, user_id: int | None = None, **fields) -> Application:
        with self.session() as sess:
            app = sess.get(Application, app_id)
            if app is None:
                raise ValueError(f"No application with id {app_id}.")
            if user_id is not None and app.user_id not in (None, user_id):
                raise PermissionError(f"Application {app_id} does not belong to this user.")
            if "tailored_payload" in fields and isinstance(fields["tailored_payload"], dict):
                fields["tailored_payload"] = json.dumps(fields["tailored_payload"])
            for key, value in fields.items():
                if not hasattr(app, key):
                    raise ValueError(f"Application has no field {key!r}.")
                setattr(app, key, value)
            sess.flush()
            return app

    def mark_submitted(self, app_id: int) -> Application:
        return self.update_application(
            app_id,
            status=ApplicationStatus.APPLIED,
            submitted_at=datetime.now(timezone.utc),
        )

    # ---------------- Feedback / iteration ----------------

    def record_feedback(
        self,
        app_id: int,
        status: ApplicationStatus,
        notes: str = "",
        user_id: int | None = None,
    ) -> Feedback:
        """Log an outcome and roll it up onto the application row."""
        with self.session() as sess:
            app = sess.get(Application, app_id)
            if app is None:
                raise ValueError(f"No application with id {app_id}.")
            if user_id is not None and app.user_id not in (None, user_id):
                raise PermissionError(f"Application {app_id} does not belong to this user.")
            entry = Feedback(
                application_id=app_id, status=status, notes=notes,
                user_id=user_id if user_id is not None else app.user_id,
            )
            sess.add(entry)
            app.status = status
            if notes:
                stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                app.notes = f"{app.notes}\n[{stamp}] {status.value}: {notes}".strip()
            sess.flush()
            return entry

    def successful_examples(
        self, role_title: str, limit: int = 3, user_id: int | None = None
    ) -> list[dict]:
        """Few-shot fuel: tailored payloads from applications that got traction.

        Matching is loose on purpose - a "Senior Backend Engineer" interview is
        a useful example for a "Backend Engineer" application. Falls back to any
        positive-outcome application when nothing matches the title.
        """
        tokens = {t for t in role_title.lower().replace("/", " ").split() if len(t) > 3}

        with self.session() as sess:
            stmt = (
                select(Application)
                .where(
                    Application.status.in_(POSITIVE_STATUSES),
                    Application.tailored_payload.is_not(None),
                )
                .order_by(Application.match_score.desc(), Application.created_at.desc())
            )
            if user_id is not None:
                stmt = stmt.where(Application.user_id == user_id)
            rows = sess.scalars(stmt).all()

        def score(app: Application) -> int:
            title = app.role_title.lower()
            return sum(1 for t in tokens if t in title)

        ranked = sorted(rows, key=score, reverse=True)
        matched = [r for r in ranked if score(r) > 0] or ranked

        examples: list[dict] = []
        for app in matched[:limit]:
            try:
                payload = json.loads(app.tailored_payload or "{}")
            except json.JSONDecodeError:
                continue
            examples.append(
                {
                    "role_title": app.role_title,
                    "company": app.company,
                    "outcome": app.status.value,
                    "match_score": app.match_score,
                    "summary": payload.get("summary", ""),
                    "tailored_experience": payload.get("tailored_experience", []),
                }
            )
        return examples

    def stats(self, user_id: int | None = None) -> dict[str, int | float]:
        with self.session() as sess:
            stmt = select(Application)
            if user_id is not None:
                stmt = stmt.where(Application.user_id == user_id)
            apps = sess.scalars(stmt).all()
        total = len(apps)
        positive = sum(1 for a in apps if a.status in POSITIVE_STATUSES)
        # "Submitted" means the application left Draft, not merely that this tool
        # clicked the button. An application submitted by hand and then marked
        # Interview must still count in the response-rate denominator.
        submitted = sum(
            1 for a in apps
            if a.submitted_at is not None or a.status is not ApplicationStatus.DRAFT
        )
        avg = sum(a.match_score for a in apps) / total if total else 0.0
        return {
            "total": total,
            "submitted": submitted,
            "positive_outcomes": positive,
            "response_rate": round(100 * positive / submitted, 1) if submitted else 0.0,
            "avg_match_score": round(avg, 1),
        }

    # ======================================================================
    # Dashboard: users
    # ======================================================================

    def create_user(self, email: str, password_hash: str) -> User:
        """Persist a new account. Hashing happens in the web layer."""
        email = email.strip().lower()
        with self.session() as sess:
            if sess.scalars(select(User).where(User.email == email)).first():
                raise ValueError("An account with that email already exists.")
            user = User(email=email, password_hash=password_hash)
            sess.add(user)
            sess.flush()
            return user

    def get_user(self, user_id: int) -> User | None:
        with self.session() as sess:
            return sess.get(User, user_id)

    def get_user_by_email(self, email: str) -> User | None:
        with self.session() as sess:
            return sess.scalars(
                select(User).where(User.email == (email or "").strip().lower())
            ).first()

    def get_user_by_api_key_hash(self, key_hash: str) -> User | None:
        """Look the key up by hash. The raw key is never stored to compare against."""
        if not key_hash:
            return None
        with self.session() as sess:
            return sess.scalars(
                select(User).where(User.api_key_hash == key_hash, User.is_active.is_(True))
            ).first()

    def update_user(self, user_id: int, **fields) -> User:
        with self.session() as sess:
            user = sess.get(User, user_id)
            if user is None:
                raise ValueError(f"No user with id {user_id}.")
            for key, value in fields.items():
                if not hasattr(user, key):
                    raise ValueError(f"User has no field {key!r}.")
                setattr(user, key, value)
            sess.flush()
            return user

    def bump_session_epoch(self, user_id: int) -> None:
        """Invalidate every existing session for this user."""
        with self.session() as sess:
            user = sess.get(User, user_id)
            if user:
                user.session_epoch = (user.session_epoch or 1) + 1

    # ---------------- provider key ----------------

    def set_anthropic_key(self, user_id: int, raw_key: str | None) -> None:
        """Store the user's Claude API key encrypted, or clear it."""
        token = self.encrypt(raw_key) if raw_key else None
        with self.session() as sess:
            user = sess.get(User, user_id)
            if user is None:
                raise ValueError(f"No user with id {user_id}.")
            user.encrypted_anthropic_key = token

    def get_anthropic_key(self, user_id: int) -> str | None:
        """Decrypt on demand. Callers must not log or echo the result."""
        with self.session() as sess:
            user = sess.get(User, user_id)
            if user is None or not user.encrypted_anthropic_key:
                return None
            return self.decrypt(user.encrypted_anthropic_key)

    # ---------------- profile ----------------

    def save_profile(self, user_id: int, profile: dict) -> None:
        with self.session() as sess:
            user = sess.get(User, user_id)
            if user is None:
                raise ValueError(f"No user with id {user_id}.")
            user.profile_json = json.dumps(profile)

    def get_profile(self, user_id: int) -> dict | None:
        with self.session() as sess:
            user = sess.get(User, user_id)
            if user is None or not user.profile_json:
                return None
            try:
                return json.loads(user.profile_json)
            except json.JSONDecodeError:
                return None

    # ======================================================================
    # Dashboard: action queue
    # ======================================================================

    def create_action(
        self,
        user_id: int | None,
        kind: ActionKind,
        question: str,
        reason: str = "",
        application_id: int | None = None,
        field_type: str = "text",
        options: list[str] | None = None,
        required: bool = False,
    ) -> ActionItem:
        with self.session() as sess:
            item = ActionItem(
                user_id=user_id,
                application_id=application_id,
                kind=kind,
                question=question,
                reason=reason,
                field_type=field_type,
                options_json=json.dumps(options) if options else None,
                required=required,
            )
            sess.add(item)
            sess.flush()
            return item

    def list_actions(
        self,
        user_id: int | None = None,
        status: ActionStatus | None = ActionStatus.OPEN,
        application_id: int | None = None,
        limit: int = 200,
    ) -> Sequence[ActionItem]:
        with self.session() as sess:
            stmt = (
                select(ActionItem)
                .options(selectinload(ActionItem.application))
                .order_by(ActionItem.created_at.desc())
                .limit(limit)
            )
            if user_id is not None:
                stmt = stmt.where(ActionItem.user_id == user_id)
            if status is not None:
                stmt = stmt.where(ActionItem.status == status)
            if application_id is not None:
                stmt = stmt.where(ActionItem.application_id == application_id)
            return sess.scalars(stmt).all()

    def count_open_actions(self, user_id: int | None = None) -> int:
        with self.session() as sess:
            stmt = select(func.count(ActionItem.id)).where(
                ActionItem.status == ActionStatus.OPEN
            )
            if user_id is not None:
                stmt = stmt.where(ActionItem.user_id == user_id)
            return int(sess.scalar(stmt) or 0)

    def answer_action(
        self, action_id: int, answer: str, remember: bool = False, user_id: int | None = None
    ) -> ActionItem:
        with self.session() as sess:
            item = sess.get(ActionItem, action_id)
            if item is None:
                raise ValueError(f"No action item with id {action_id}.")
            if user_id is not None and item.user_id not in (None, user_id):
                raise PermissionError("That action item belongs to another user.")
            item.answer = answer
            item.remember = remember
            item.status = ActionStatus.ANSWERED
            item.answered_at = datetime.now(timezone.utc)
            sess.flush()
            return item

    def dismiss_action(self, action_id: int, user_id: int | None = None) -> ActionItem:
        with self.session() as sess:
            item = sess.get(ActionItem, action_id)
            if item is None:
                raise ValueError(f"No action item with id {action_id}.")
            if user_id is not None and item.user_id not in (None, user_id):
                raise PermissionError("That action item belongs to another user.")
            item.status = ActionStatus.DISMISSED
            item.answered_at = datetime.now(timezone.utc)
            sess.flush()
            return item

    def answered_action_map(self, user_id: int | None = None) -> dict[str, str]:
        """Every question this user has already answered, for reuse.

        Feeds straight into ScreenerMapper, so a question answered once in the
        dashboard is answered automatically on every later application.
        """
        with self.session() as sess:
            stmt = select(ActionItem).where(
                ActionItem.status == ActionStatus.ANSWERED,
                ActionItem.remember.is_(True),
                ActionItem.answer != "",
            )
            if user_id is not None:
                stmt = stmt.where(ActionItem.user_id == user_id)
            items = sess.scalars(stmt.order_by(ActionItem.answered_at.desc())).all()
        answers: dict[str, str] = {}
        for item in items:  # newest wins
            answers.setdefault(item.question, item.answer)
        return answers

    # ======================================================================
    # Dashboard: LLM usage
    # ======================================================================

    def record_usage(
        self,
        user_id: int | None,
        model: str,
        phase: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cost_usd: float = 0.0,
        application_id: int | None = None,
    ) -> LLMUsage:
        with self.session() as sess:
            row = LLMUsage(
                user_id=user_id,
                application_id=application_id,
                model=model,
                phase=phase,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                cost_usd=cost_usd,
            )
            sess.add(row)
            sess.flush()
            return row

    def usage_rows(
        self, user_id: int | None = None, days: int = 30
    ) -> list[tuple[datetime, float, int]]:
        """(created_at, cost_usd, total_tokens) within the window."""
        since = datetime.now(timezone.utc) - timedelta(days=days)
        with self.session() as sess:
            stmt = select(LLMUsage).where(LLMUsage.created_at >= since)
            if user_id is not None:
                stmt = stmt.where(LLMUsage.user_id == user_id)
            rows = sess.scalars(stmt).all()
        return [
            (
                r.created_at if r.created_at.tzinfo else r.created_at.replace(tzinfo=timezone.utc),
                r.cost_usd,
                r.total_tokens,
            )
            for r in rows
        ]

    def usage_by_model(self, user_id: int | None = None, days: int = 30) -> list[dict]:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        with self.session() as sess:
            stmt = (
                select(
                    LLMUsage.model,
                    func.count(LLMUsage.id),
                    func.sum(LLMUsage.cost_usd),
                    func.sum(LLMUsage.input_tokens + LLMUsage.output_tokens),
                )
                .where(LLMUsage.created_at >= since)
                .group_by(LLMUsage.model)
            )
            if user_id is not None:
                stmt = stmt.where(LLMUsage.user_id == user_id)
            rows = sess.execute(stmt).all()
        return [
            {"model": m, "calls": c, "cost": round(cost or 0.0, 4), "tokens": int(tok or 0)}
            for m, c, cost, tok in sorted(rows, key=lambda r: -(r[2] or 0))
        ]

    def usage_by_phase(self, user_id: int | None = None, days: int = 30) -> list[dict]:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        with self.session() as sess:
            stmt = (
                select(LLMUsage.phase, func.count(LLMUsage.id), func.sum(LLMUsage.cost_usd))
                .where(LLMUsage.created_at >= since)
                .group_by(LLMUsage.phase)
            )
            if user_id is not None:
                stmt = stmt.where(LLMUsage.user_id == user_id)
            rows = sess.execute(stmt).all()
        return [
            {"phase": p or "unknown", "calls": c, "cost": round(cost or 0.0, 4)}
            for p, c, cost in sorted(rows, key=lambda r: -(r[2] or 0))
        ]

    def cost_per_application(self, user_id: int | None = None, days: int = 30) -> float:
        rows = self.usage_rows(user_id, days)
        total = sum(cost for _, cost, _ in rows)
        since = datetime.now(timezone.utc) - timedelta(days=days)
        with self.session() as sess:
            stmt = select(func.count(Application.id)).where(Application.created_at >= since)
            if user_id is not None:
                stmt = stmt.where(Application.user_id == user_id)
            count = int(sess.scalar(stmt) or 0)
        return round(total / count, 4) if count else 0.0

    # ======================================================================
    # Dashboard: run logs
    # ======================================================================

    def log_event(
        self,
        user_id: int | None,
        event: str,
        message: str = "",
        level: LogLevel = LogLevel.INFO,
        application_id: int | None = None,
    ) -> RunLog:
        with self.session() as sess:
            row = RunLog(
                user_id=user_id,
                application_id=application_id,
                level=level,
                event=event[:64],
                message=message,
            )
            sess.add(row)
            sess.flush()
            return row

    def list_logs(
        self,
        user_id: int | None = None,
        application_id: int | None = None,
        level: LogLevel | None = None,
        limit: int = 200,
        since_id: int | None = None,
    ) -> Sequence[RunLog]:
        with self.session() as sess:
            stmt = select(RunLog).order_by(RunLog.id.desc()).limit(limit)
            if user_id is not None:
                stmt = stmt.where(RunLog.user_id == user_id)
            if application_id is not None:
                stmt = stmt.where(RunLog.application_id == application_id)
            if level is not None:
                stmt = stmt.where(RunLog.level == level)
            if since_id is not None:
                stmt = stmt.where(RunLog.id > since_id)
            return sess.scalars(stmt).all()

    def prune_logs(self, keep_days: int = 90) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
        with self.session() as sess:
            result = sess.execute(delete(RunLog).where(RunLog.created_at < cutoff))
            return int(result.rowcount or 0)

    # ======================================================================
    # Batch queue
    # ======================================================================

    def enqueue_job(
        self,
        user_id: int,
        kind: str = "tailor",
        job_url: str = "",
        job_description: str | None = None,
        application_id: int | None = None,
        batch_id: str | None = None,
    ) -> RunJob:
        with self.session() as sess:
            position = int(
                sess.scalar(
                    select(func.coalesce(func.max(RunJob.position), 0)).where(
                        RunJob.user_id == user_id
                    )
                )
                or 0
            )
            job = RunJob(
                user_id=user_id,
                kind=kind,
                job_url=job_url,
                job_description=job_description,
                application_id=application_id,
                batch_id=batch_id,
                position=position + 1,
            )
            sess.add(job)
            sess.flush()
            return job

    def get_job(self, job_id: int, user_id: int | None = None) -> RunJob | None:
        with self.session() as sess:
            stmt = (
                select(RunJob)
                .where(RunJob.id == job_id)
                .options(
                    selectinload(RunJob.application),
                    selectinload(RunJob.blocking_action),
                )
            )
            if user_id is not None:
                stmt = stmt.where(RunJob.user_id == user_id)
            return sess.scalars(stmt).first()

    def list_jobs(
        self,
        user_id: int | None = None,
        statuses: tuple[JobStatus, ...] | None = None,
        batch_id: str | None = None,
        limit: int = 200,
    ) -> Sequence[RunJob]:
        with self.session() as sess:
            stmt = (
                select(RunJob)
                .options(
                    selectinload(RunJob.application),
                    selectinload(RunJob.blocking_action),
                )
                .order_by(RunJob.position.asc(), RunJob.id.asc())
                .limit(limit)
            )
            if user_id is not None:
                stmt = stmt.where(RunJob.user_id == user_id)
            if statuses:
                stmt = stmt.where(RunJob.status.in_(statuses))
            if batch_id:
                stmt = stmt.where(RunJob.batch_id == batch_id)
            return sess.scalars(stmt).all()

    def claim_next_job(
        self,
        kinds: tuple[str, ...] | None = None,
        user_id: int | None = None,
        exclude_users: set[int] | None = None,
    ) -> RunJob | None:
        """Atomically move the oldest runnable job to RUNNING and return it.

        READY (a human answered) is served before QUEUED, so clearing blocks
        drains the backlog rather than competing with fresh work.

        `kinds` matters once the dashboard is hosted: the server can only run
        jobs that need no visible browser, so it claims `tailor` only and leaves
        `apply` for the local agent. Without that filter a hosted instance would
        happily start browser work nobody can see or clear.
        """
        with self.session() as sess:
            for status in (JobStatus.READY, JobStatus.QUEUED):
                stmt = (
                    select(RunJob)
                    .where(RunJob.status == status)
                    .order_by(RunJob.position.asc(), RunJob.id.asc())
                    .limit(1)
                    .with_for_update(nowait=False)
                )
                if kinds:
                    stmt = stmt.where(RunJob.kind.in_(kinds))
                if user_id is not None:
                    stmt = stmt.where(RunJob.user_id == user_id)
                if exclude_users:
                    stmt = stmt.where(RunJob.user_id.not_in(exclude_users))
                job = sess.scalars(stmt).first()
                if job is None:
                    continue
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now(timezone.utc)
                job.attempts = (job.attempts or 0) + 1
                job.block_mode = BlockMode.NONE
                sess.flush()
                sess.refresh(job)
                return job
            return None

    def block_job(
        self,
        job_id: int,
        mode: BlockMode,
        message: str,
        action_id: int | None = None,
        holds_browser: bool = False,
    ) -> RunJob:
        with self.session() as sess:
            job = sess.get(RunJob, job_id)
            if job is None:
                raise ValueError(f"No job with id {job_id}.")
            job.status = JobStatus.BLOCKED
            job.block_mode = mode
            job.message = message
            job.blocking_action_id = action_id
            job.holds_browser = holds_browser
            job.blocked_at = datetime.now(timezone.utc)
            sess.flush()
            return job

    def release_job(self, job_id: int) -> RunJob:
        """Mark a blocked job as answered and ready to resume."""
        with self.session() as sess:
            job = sess.get(RunJob, job_id)
            if job is None:
                raise ValueError(f"No job with id {job_id}.")
            if job.status is JobStatus.BLOCKED:
                job.status = JobStatus.READY
                job.block_mode = BlockMode.NONE
                job.message = ""
            sess.flush()
            return job

    def finish_job(
        self, job_id: int, status: JobStatus, message: str = "",
        application_id: int | None = None,
    ) -> RunJob:
        with self.session() as sess:
            job = sess.get(RunJob, job_id)
            if job is None:
                raise ValueError(f"No job with id {job_id}.")
            job.status = status
            job.message = message
            job.holds_browser = False
            job.block_mode = BlockMode.NONE
            job.finished_at = datetime.now(timezone.utc)
            if application_id is not None:
                job.application_id = application_id
            sess.flush()
            return job

    def cancel_job(self, job_id: int, user_id: int | None = None) -> RunJob:
        with self.session() as sess:
            job = sess.get(RunJob, job_id)
            if job is None:
                raise ValueError(f"No job with id {job_id}.")
            if user_id is not None and job.user_id != user_id:
                raise PermissionError("That job belongs to another user.")
            if job.status not in TERMINAL_JOB_STATUSES:
                job.status = JobStatus.CANCELLED
                job.finished_at = datetime.now(timezone.utc)
                job.holds_browser = False
            sess.flush()
            return job

    def jobs_blocked_on_action(self, action_id: int) -> Sequence[RunJob]:
        with self.session() as sess:
            return sess.scalars(
                select(RunJob).where(
                    RunJob.blocking_action_id == action_id,
                    RunJob.status == JobStatus.BLOCKED,
                )
            ).all()

    def queue_summary(self, user_id: int) -> dict[str, int]:
        with self.session() as sess:
            rows = sess.execute(
                select(RunJob.status, func.count(RunJob.id))
                .where(RunJob.user_id == user_id)
                .group_by(RunJob.status)
            ).all()
        counts = {status.value: 0 for status in JobStatus}
        for status, count in rows:
            counts[status.value] = int(count)
        counts["active"] = counts["Queued"] + counts["Running"] + counts["Ready"]
        return counts

    def reset_stale_running_jobs(self) -> int:
        """Requeue jobs left RUNNING by a crash or restart.

        Without this a killed process strands work in RUNNING forever, since no
        worker owns it any more.
        """
        with self.session() as sess:
            stale = sess.scalars(
                select(RunJob).where(RunJob.status == JobStatus.RUNNING)
            ).all()
            for job in stale:
                job.status = JobStatus.QUEUED
                job.started_at = None
                job.holds_browser = False
            return len(stale)

    # ======================================================================
    # Notifications
    # ======================================================================

    def record_notification(
        self, user_id: int, channel: str, title: str, body: str,
        delivered: bool = False, error: str = "", job_id: int | None = None,
    ) -> Notification:
        with self.session() as sess:
            row = Notification(
                user_id=user_id, job_id=job_id, channel=channel,
                title=title[:255], body=body, delivered=delivered, error=error,
            )
            sess.add(row)
            sess.flush()
            return row

    def list_notifications(self, user_id: int, limit: int = 50) -> Sequence[Notification]:
        with self.session() as sess:
            return sess.scalars(
                select(Notification)
                .where(Notification.user_id == user_id)
                .order_by(Notification.id.desc())
                .limit(limit)
            ).all()

    def recently_notified(self, user_id: int, within_seconds: int) -> bool:
        """True if this user was notified inside the quiet window.

        Stops a ten-application batch from firing ten notifications in a row.
        """
        if within_seconds <= 0:
            return False
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=within_seconds)
        with self.session() as sess:
            latest = sess.scalars(
                select(Notification)
                .where(Notification.user_id == user_id, Notification.delivered.is_(True))
                .order_by(Notification.id.desc())
                .limit(1)
            ).first()
        if latest is None:
            return False
        created = latest.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return created > cutoff

    def purge_finished_jobs(self, user_id: int) -> int:
        """Drop terminal jobs from the queue view. History lives in run_logs."""
        with self.session() as sess:
            result = sess.execute(
                delete(RunJob).where(
                    RunJob.user_id == user_id,
                    RunJob.status.in_(TERMINAL_JOB_STATUSES),
                )
            )
            return int(result.rowcount or 0)

    # ======================================================================
    # Admin
    # ======================================================================

    def count_users(self) -> int:
        with self.session() as sess:
            return int(sess.scalar(select(func.count(User.id))) or 0)

    def list_users(self, limit: int = 200) -> Sequence[User]:
        with self.session() as sess:
            return sess.scalars(
                select(User).order_by(User.created_at.desc()).limit(limit)
            ).all()

    def user_overview(self, limit: int = 200) -> list[dict]:
        """Account activity for the admin view.

        Deliberately excludes anything private: no decrypted keys, no portal
        passwords, no profile contents. Only whether a key is set, and activity
        counts. Holding that data is unavoidable; displaying it is not.
        """
        from web.profile_form import completeness

        rows: list[dict] = []
        for user in self.list_users(limit):
            profile = self.get_profile(user.id)
            usage = self.usage_rows(user.id, days=360)
            with self.session() as sess:
                applications = int(
                    sess.scalar(
                        select(func.count(Application.id)).where(
                            Application.user_id == user.id
                        )
                    )
                    or 0
                )
            rows.append({
                "id": user.id,
                "email": user.email,
                "is_admin": user.is_admin,
                "is_active": user.is_active,
                "created_at": user.created_at,
                "last_login_at": user.last_login_at,
                "failed_logins": user.failed_logins or 0,
                "locked": user.locked_until is not None,
                "has_anthropic_key": bool(user.encrypted_anthropic_key),
                "has_api_key": bool(user.api_key_hash),
                "profile_percent": completeness(profile)["percent"] if profile else 0,
                "applications": applications,
                "llm_calls": len(usage),
                "llm_spend": round(sum(cost for _, cost, _ in usage), 4),
                "open_actions": self.count_open_actions(user.id),
            })
        return rows

    def set_user_active(self, user_id: int, active: bool) -> User:
        """Suspend or restore an account, retiring its sessions either way."""
        with self.session() as sess:
            user = sess.get(User, user_id)
            if user is None:
                raise ValueError(f"No user with id {user_id}.")
            user.is_active = active
            user.session_epoch = (user.session_epoch or 1) + 1
            sess.flush()
            return user

    # ======================================================================
    # Spend control
    # ======================================================================

    def spend_today(self, user_id: int) -> float:
        """USD spent since midnight UTC. The cap is enforced against this."""
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        with self.session() as sess:
            total = sess.scalar(
                select(func.coalesce(func.sum(LLMUsage.cost_usd), 0.0)).where(
                    LLMUsage.user_id == user_id, LLMUsage.created_at >= start
                )
            )
        return round(float(total or 0.0), 6)

    def daily_cap_for(self, user_id: int) -> float:
        """The user's cap, falling back to the instance default. 0 disables."""
        user = self.get_user(user_id)
        if user is not None and user.daily_spend_cap_usd is not None:
            return float(user.daily_spend_cap_usd)
        from config import settings as _settings

        return float(_settings.DAILY_SPEND_CAP_USD)

    def is_over_daily_cap(self, user_id: int) -> bool:
        cap = self.daily_cap_for(user_id)
        return cap > 0 and self.spend_today(user_id) >= cap

    def users_over_daily_cap(self) -> set[int]:
        """Every user whose spend has hit their ceiling today.

        The dispatcher skips these rather than blocking each of their queued
        jobs one at a time, which would churn through a hundred rows to say the
        same thing a hundred times.
        """
        from config import settings as _settings

        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        default_cap = float(_settings.DAILY_SPEND_CAP_USD)

        # Spend per user and each user's cap in two queries, not one query per
        # user: the dispatcher calls this on a timer, so an N+1 here becomes a
        # steady drip of queries against the database.
        with self.session() as sess:
            spend = sess.execute(
                select(LLMUsage.user_id, func.sum(LLMUsage.cost_usd))
                .where(LLMUsage.created_at >= start, LLMUsage.user_id.is_not(None))
                .group_by(LLMUsage.user_id)
            ).all()
            caps = dict(
                sess.execute(select(User.id, User.daily_spend_cap_usd)).all()
            )

        over = set()
        for user_id, spent in spend:
            cap = caps.get(user_id)
            cap = float(cap) if cap is not None else default_cap
            if cap > 0 and float(spent or 0.0) >= cap:
                over.add(int(user_id))
        return over

    def applications_today(self, user_id: int) -> int:
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        with self.session() as sess:
            return int(
                sess.scalar(
                    select(func.count(Application.id)).where(
                        Application.user_id == user_id, Application.created_at >= start
                    )
                )
                or 0
            )

    def model_for_job(self, user_id: int, priority: bool) -> str:
        """Tiered model choice: cheap for bulk, expensive for flagged roles."""
        from config import settings as _settings

        user = self.get_user(user_id)
        if priority:
            return (user.model_priority if user and user.model_priority
                    else _settings.LLM_MODEL_PRIORITY)
        return (user.model_bulk if user and user.model_bulk
                else _settings.LLM_MODEL_BULK)

    # ======================================================================
    # Digest
    # ======================================================================

    def users_due_for_digest(self) -> list[tuple[int, int]]:
        """(user_id, open_block_count) for digests that should go out now.

        Due means: digest mode, the user's local hour has reached their chosen
        hour, nothing sent yet in their local day, and something to report.
        """
        due: list[tuple[int, int]] = []
        now = datetime.now(timezone.utc)

        for user in self.list_users(limit=1000):
            if (user.notify_mode or "immediate") != "digest" or not user.is_active:
                continue
            offset_minutes = user.notify_utc_offset_minutes
            offset = timedelta(minutes=int(offset_minutes if offset_minutes is not None else 0))
            local_now = now + offset
            # `or 18` would be wrong here: hour 0 is falsy, so anyone choosing
            # midnight would silently be given 18:00 instead.
            hour = user.notify_digest_hour
            if local_now.hour < int(hour if hour is not None else 18):
                continue
            if user.last_digest_at is not None:
                last = user.last_digest_at
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                if (last + offset).date() == local_now.date():
                    continue        # already sent today, local time
            blocked = self.count_open_actions(user.id)
            if blocked:
                due.append((user.id, blocked))
        return due

    def mark_digest_sent(self, user_id: int) -> None:
        with self.session() as sess:
            user = sess.get(User, user_id)
            if user:
                user.last_digest_at = datetime.now(timezone.utc)

    def save_discovery_criteria(self, user_id: int, criteria: dict) -> None:
        """Remember the last search so it does not have to be retyped."""
        profile = self.get_profile(user_id) or {}
        profile["_discovery"] = criteria
        self.save_profile(user_id, profile)

    def get_discovery_criteria(self, user_id: int) -> dict | None:
        return (self.get_profile(user_id) or {}).get("_discovery")

    def save_pending_profile(self, user_id: int, profile: dict, notes: list[str] | None = None) -> None:
        """Stash an imported profile for review. Not live until the user saves."""
        payload = {"profile": profile, "uncertain": notes or []}
        with self.session() as sess:
            user = sess.get(User, user_id)
            if user is None:
                raise ValueError(f"No user with id {user_id}.")
            user.pending_profile_json = json.dumps(payload)

    def get_pending_profile(self, user_id: int) -> dict | None:
        with self.session() as sess:
            user = sess.get(User, user_id)
            if user is None or not user.pending_profile_json:
                return None
        try:
            return json.loads(user.pending_profile_json)
        except json.JSONDecodeError:
            return None

    def clear_pending_profile(self, user_id: int) -> None:
        with self.session() as sess:
            user = sess.get(User, user_id)
            if user:
                user.pending_profile_json = None

    def auto_apply_threshold(self, user_id: int) -> float | None:
        """The score at or above which the browser step queues itself.

        None means off, and every tailored application waits for a click.
        """
        user = self.get_user(user_id)
        if user is not None and user.auto_apply_threshold is not None:
            return float(user.auto_apply_threshold) or None
        from config import settings as _settings

        return float(_settings.AUTO_APPLY_THRESHOLD) or None

    def submitted_today(self, user_id: int) -> int:
        """Applications actually sent to an employer since midnight UTC.

        Distinct from `applications_today`, which counts tailored drafts. The
        daily cap governs tailoring, because that is what costs money; this is
        what the user means by "applications".
        """
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        with self.session() as sess:
            return int(
                sess.scalar(
                    select(func.count(Application.id)).where(
                        Application.user_id == user_id,
                        Application.submitted_at.is_not(None),
                        Application.submitted_at >= start,
                    )
                )
                or 0
            )

    def awaiting_agent(self, user_id: int) -> int:
        """Applications tailored and queued, waiting for the local agent."""
        with self.session() as sess:
            return int(
                sess.scalar(
                    select(func.count(RunJob.id)).where(
                        RunJob.user_id == user_id,
                        RunJob.kind == "apply",
                        RunJob.status.in_((JobStatus.QUEUED, JobStatus.READY)),
                    )
                )
                or 0
            )

    def has_pending_apply_job(self, application_id: int) -> bool:
        """Is there already an unfinished apply job for this application?

        Guards against duplicates from three directions: a tailor job requeued
        after a restart re-running its auto-queue step, a user pressing "Run
        application" twice, and an API caller retrying.
        """
        with self.session() as sess:
            return sess.scalars(
                select(RunJob).where(
                    RunJob.application_id == application_id,
                    RunJob.kind == "apply",
                    RunJob.status.not_in(TERMINAL_JOB_STATUSES),
                ).limit(1)
            ).first() is not None

    def screened_today(self, user_id: int) -> int:
        """Postings queued for screening since midnight UTC.

        Counts tailor jobs rather than applications: most postings are rejected
        by the viability check and never become one, and the screening budget
        needs to reflect what was actually looked at.
        """
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        with self.session() as sess:
            return int(
                sess.scalar(
                    select(func.count(RunJob.id)).where(
                        RunJob.user_id == user_id,
                        RunJob.kind == "tailor",
                        RunJob.created_at >= start,
                    )
                )
                or 0
            )
