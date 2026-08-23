"""Authentication, API keys, CSRF and rate limiting.

Threat model for a single-tenant-per-user dashboard holding a credential vault
and a provider API key:

  * Passwords are Argon2id. Never stored or logged in plaintext.
  * API keys are shown once and stored as SHA-256. A leaked database yields no
    usable key.
  * The Anthropic key is Fernet-encrypted at rest and never rendered back to
    the browser - only a masked preview.
  * Sessions are signed, expiring cookies carrying a `session_epoch`, so a
    password change invalidates every existing session.
  * Every state-changing form post requires a CSRF token bound to the session.
  * Login, signup and API auth are rate limited per identity and per IP.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

# --------------------------------------------------------------------------
# Passwords
# --------------------------------------------------------------------------

_hasher = PasswordHasher()

MIN_PASSWORD_LENGTH = 12
#: Rejected outright regardless of length or composition.
_COMMON_PASSWORDS = {
    "password", "password123", "123456789012", "qwertyuiop12",
    "letmein12345", "administrator", "changeme1234", "welcome12345",
}


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError, TypeError):
        return False


def needs_rehash(password_hash: str) -> bool:
    try:
        return _hasher.check_needs_rehash(password_hash)
    except (InvalidHashError, TypeError):
        return True


def password_problems(password: str, email: str = "") -> list[str]:
    """Return human-readable reasons the password is unacceptable."""
    problems: list[str] = []
    if len(password) < MIN_PASSWORD_LENGTH:
        problems.append(f"Must be at least {MIN_PASSWORD_LENGTH} characters.")
    if password.lower() in _COMMON_PASSWORDS:
        problems.append("That password is too common.")
    if email and email.split("@")[0].lower() in password.lower() and len(email.split("@")[0]) > 3:
        problems.append("Must not contain your email address.")
    classes = sum(
        bool(re.search(pattern, password))
        for pattern in (r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]")
    )
    if classes < 3:
        problems.append("Use at least three of: lowercase, uppercase, digits, symbols.")
    return problems


def invite_code_valid(supplied: str, expected: str | None) -> bool:
    """Constant-time invite-code check.

    Returns True when no code is configured, so an instance without one keeps
    working. A plain `==` would leak the code a character at a time to anyone
    who can time the response.
    """
    if not expected:
        return True
    return hmac.compare_digest((supplied or "").strip(), expected.strip())


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


def valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match((email or "").strip())) and len(email) <= 320


# --------------------------------------------------------------------------
# API keys
# --------------------------------------------------------------------------

API_KEY_PREFIX = "jp_live_"


def generate_api_key() -> tuple[str, str, str]:
    """Return (raw_key, sha256_hash, display_prefix).

    The raw key is returned to the caller exactly once. Only the hash is
    persisted, so the database alone never yields a working key.
    """
    raw = API_KEY_PREFIX + secrets.token_urlsafe(32)
    return raw, hash_api_key(raw), raw[: len(API_KEY_PREFIX) + 6]


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def api_keys_match(raw: str, stored_hash: str) -> bool:
    """Constant-time comparison, so timing cannot reveal a valid prefix."""
    if not raw or not stored_hash:
        return False
    return hmac.compare_digest(hash_api_key(raw), stored_hash)


def mask_secret(value: str, keep: int = 4) -> str:
    """`sk-ant-abc...wxyz` style preview. Never returns the full secret."""
    if not value:
        return ""
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}{'*' * 8}{value[-keep:]}"


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------

SESSION_COOKIE = "jp_session"
SESSION_MAX_AGE = 60 * 60 * 12  # 12 hours


class SessionManager:
    def __init__(self, secret_key: str, max_age: int = SESSION_MAX_AGE) -> None:
        self._serializer = URLSafeTimedSerializer(secret_key, salt="jp-session")
        self.max_age = max_age

    def issue(self, user_id: int, epoch: int) -> str:
        return self._serializer.dumps({"uid": user_id, "epoch": epoch})

    def read(self, token: str) -> dict[str, Any] | None:
        if not token:
            return None
        try:
            return self._serializer.loads(token, max_age=self.max_age)
        except (BadSignature, SignatureExpired):
            return None


# --------------------------------------------------------------------------
# CSRF
# --------------------------------------------------------------------------

CSRF_COOKIE = "jp_csrf"
CSRF_FIELD = "csrf_token"


def generate_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def csrf_valid(cookie_token: str | None, form_token: str | None) -> bool:
    if not cookie_token or not form_token:
        return False
    return hmac.compare_digest(cookie_token, form_token)


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


@dataclass
class _Bucket:
    hits: list[float] = field(default_factory=list)


class RateLimiter:
    """Fixed-window limiter keyed by an arbitrary identity string.

    In-process and therefore per-worker: adequate for a single-node dashboard,
    and the place to swap in Redis if this is ever run multi-process.
    """

    def __init__(self) -> None:
        self._buckets: dict[str, _Bucket] = {}
        self._lock = Lock()

    def check(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        """Return (allowed, seconds_until_retry)."""
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            bucket = self._buckets.setdefault(key, _Bucket())
            bucket.hits = [h for h in bucket.hits if h > cutoff]
            if len(bucket.hits) >= limit:
                retry = int(bucket.hits[0] + window_seconds - now) + 1
                return False, max(retry, 1)
            bucket.hits.append(now)
            return True, 0

    def reset(self, key: str) -> None:
        with self._lock:
            self._buckets.pop(key, None)

    def prune(self, older_than: int = 3600) -> None:
        cutoff = time.monotonic() - older_than
        with self._lock:
            for key in [k for k, b in self._buckets.items()
                        if not b.hits or max(b.hits) < cutoff]:
                self._buckets.pop(key, None)


#: (limit, window_seconds) per protected surface.
RATE_LIMITS = {
    "login": (8, 300),        # 8 attempts / 5 min
    "signup": (5, 3600),      # 5 accounts / hour / IP
    "api": (120, 60),         # 120 API calls / min / key
    "api_auth_fail": (20, 300),
    "run": (20, 3600),        # 20 pipeline runs / hour
}

#: Progressive lockout after repeated password failures on one account.
MAX_FAILED_LOGINS = 10
LOCKOUT_MINUTES = 15


def lockout_until(failed_logins: int) -> datetime | None:
    if failed_logins < MAX_FAILED_LOGINS:
        return None
    return datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_MINUTES)


def is_locked(locked_until: datetime | None) -> bool:
    if locked_until is None:
        return False
    if locked_until.tzinfo is None:
        locked_until = locked_until.replace(tzinfo=timezone.utc)
    return locked_until > datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Response hardening
# --------------------------------------------------------------------------

#: Strict CSP: no inline scripts, no third-party origins. The dashboard ships
#: its own CSS/JS, so nothing here needs relaxing.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; "
        "script-src 'self'; font-src 'self'; connect-src 'self'; "
        "form-action 'self'; frame-ancestors 'none'; base-uri 'self'; "
        "object-src 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=(), payment=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}
