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

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from config import settings
from database.models import (
    POSITIVE_STATUSES,
    ActionItem,
    ActionKind,
    ActionStatus,
    Application,
    ApplicationStatus,
    Base,
    Credential,
    Feedback,
    LLMUsage,
    LogLevel,
    RunLog,
    User,
)

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
