"""Central configuration.

Everything tunable lives here so no module hardcodes a key, path, or model id.
Values resolve from environment variables (optionally via a local .env file),
falling back to sane defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_a, **_kw):  # type: ignore[misc]
        return False

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = Path(os.getenv("JP_DATA_DIR", BASE_DIR / "data"))
OUTPUT_DIR = Path(os.getenv("JP_OUTPUT_DIR", BASE_DIR / "output"))
RESUME_DIR = OUTPUT_DIR / "resumes"
TEMPLATE_DIR = BASE_DIR / "templates"

MASTER_PROFILE_PATH = Path(os.getenv("JP_MASTER_PROFILE", CONFIG_DIR / "master_profile.json"))
DB_PATH = Path(os.getenv("JP_DB_PATH", DATA_DIR / "pipeline.db"))
DB_URL = os.getenv("JP_DB_URL", f"sqlite:///{DB_PATH}")

# Playwright persistent profile (keeps you logged in between runs)
BROWSER_PROFILE_DIR = Path(os.getenv("JP_BROWSER_PROFILE", DATA_DIR / "browser_profile"))

for _d in (DATA_DIR, OUTPUT_DIR, RESUME_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------
# Fernet key for the credential vault. Generated on first run and written to
# JP_KEY_PATH (chmod 600) if not supplied via env.
ENCRYPTION_KEY = os.getenv("JP_ENCRYPTION_KEY")
KEY_PATH = Path(os.getenv("JP_KEY_PATH", DATA_DIR / "vault.key"))

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # None -> SDK resolves from `ant auth login`

# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------
# NOTE on the spec's `temperature=0.0`: sampling parameters (temperature/top_p/
# top_k) were REMOVED from current Claude models and now return HTTP 400.
# Determinism is achieved instead via strict structured outputs
# (output_config.format / messages.parse) plus a fixed, fully-specified prompt.
# LLM_TEMPERATURE is only sent when LLM_MODEL is a legacy model that accepts it.
LLM_MODEL = os.getenv("JP_LLM_MODEL", "claude-opus-5")
LLM_EFFORT = os.getenv("JP_LLM_EFFORT", "high")        # low|medium|high|xhigh|max
LLM_MAX_TOKENS = int(os.getenv("JP_LLM_MAX_TOKENS", "16000"))
LLM_TEMPERATURE: float | None = (
    float(os.environ["JP_LLM_TEMPERATURE"]) if "JP_LLM_TEMPERATURE" in os.environ else None
)
# Models that still accept sampling params (pre-4.6 generation).
LEGACY_SAMPLING_MODELS = ("claude-3", "claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-5")

# --------------------------------------------------------------------------
# ATS scoring
# --------------------------------------------------------------------------
TARGET_MATCH_SCORE = float(os.getenv("JP_TARGET_MATCH_SCORE", "80.0"))
MAX_TAILOR_ITERATIONS = int(os.getenv("JP_MAX_TAILOR_ITERATIONS", "3"))
FUZZY_MATCH_THRESHOLD = int(os.getenv("JP_FUZZY_THRESHOLD", "88"))  # rapidfuzz 0-100

# Weighting for the local (non-LLM) scorer.
SCORE_WEIGHTS = {"hard_skills": 0.55, "tooling": 0.25, "soft_skills": 0.10, "title": 0.10}

# --------------------------------------------------------------------------
# Browser automation
# --------------------------------------------------------------------------
HEADLESS = os.getenv("JP_HEADLESS", "false").lower() == "true"  # spec: headed mode
SLOW_MO_MS = int(os.getenv("JP_SLOW_MO_MS", "0"))
NAV_TIMEOUT_MS = int(os.getenv("JP_NAV_TIMEOUT_MS", "45000"))
PROXY_SERVER = os.getenv("JP_PROXY_SERVER")          # e.g. http://user:pass@host:port
VIEWPORT = {"width": 1440, "height": 900}

# Human-emulation keystroke timing (Gaussian, milliseconds), per spec.
KEYSTROKE_DELAY_MEAN_MS = float(os.getenv("JP_KEY_MEAN_MS", "80"))
KEYSTROKE_DELAY_STDEV_MS = float(os.getenv("JP_KEY_STDEV_MS", "18"))
KEYSTROKE_DELAY_MIN_MS = 40.0
KEYSTROKE_DELAY_MAX_MS = 120.0

# Human-in-the-loop safety. Leave these ON. The pipeline fills forms; a human
# confirms every irreversible action (account registration, final submit) and
# every CAPTCHA / unmapped field.
REQUIRE_CONFIRM_BEFORE_SUBMIT = os.getenv("JP_CONFIRM_SUBMIT", "true").lower() == "true"
REQUIRE_CONFIRM_BEFORE_REGISTER = os.getenv("JP_CONFIRM_REGISTER", "true").lower() == "true"

GENERATED_PASSWORD_LENGTH = int(os.getenv("JP_PASSWORD_LENGTH", "16"))

# --------------------------------------------------------------------------
# Web dashboard
# --------------------------------------------------------------------------
# Signing key for session cookies. Generated and persisted on first run if not
# supplied; rotating it logs everyone out, which is the intended behaviour.
SECRET_KEY = os.getenv("JP_SECRET_KEY")
SECRET_KEY_PATH = Path(os.getenv("JP_SECRET_KEY_PATH", DATA_DIR / "secret.key"))

WEB_HOST = os.getenv("JP_WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("JP_WEB_PORT", "8000"))
# Cookies are Secure by default; set JP_INSECURE_COOKIES=true only for local
# plain-HTTP development, never behind a real domain.
COOKIE_SECURE = os.getenv("JP_INSECURE_COOKIES", "false").lower() != "true"
# Open signup is convenient for a personal install and wrong for a shared one.
ALLOW_SIGNUP = os.getenv("JP_ALLOW_SIGNUP", "true").lower() == "true"
LOG_RETENTION_DAYS = int(os.getenv("JP_LOG_RETENTION_DAYS", "90"))


def load_or_create_secret_key() -> str:
    """Resolve the session signing key, generating one on first run."""
    if SECRET_KEY:
        return SECRET_KEY
    if SECRET_KEY_PATH.exists():
        return SECRET_KEY_PATH.read_text().strip()
    import secrets as _secrets

    key = _secrets.token_urlsafe(48)
    SECRET_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(SECRET_KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(key)
    return key


def uses_sampling_params(model: str = LLM_MODEL) -> bool:
    """True when `model` still accepts temperature/top_p/top_k."""
    return any(model.startswith(prefix) for prefix in LEGACY_SAMPLING_MODELS)
