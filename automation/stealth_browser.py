"""Playwright wrapper with human-paced interaction and hard human-in-the-loop gates.

Scope note
----------
This module deliberately does NOT implement browser-fingerprint spoofing or
anti-bot-detection evasion. It runs a real, visible Chromium with a persistent
profile and paces its interactions at human speed, which is what makes
automation reliable against JavaScript-heavy application forms (fields that
need real ``input``/``change`` events, debounced validation, lazy-rendered
sections). Every irreversible step - creating an account, submitting an
application - stops and waits for you.

If a CAPTCHA appears, the run pauses and hands the browser to you. Solving them
programmatically is out of scope by design.
"""

from __future__ import annotations

import logging
import math
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from automation.gatekeeper import Gatekeeper, TerminalGatekeeper
from config import settings

log = logging.getLogger(__name__)


class ManualInterventionRequired(Exception):
    """Raised when automation cannot safely continue without a human."""


class HumanDeclined(Exception):
    """Raised when the operator answers 'no' at a confirmation gate."""


# --------------------------------------------------------------------------
# Timing helpers
# --------------------------------------------------------------------------


def gaussian_delay_ms(
    mean: float | None = None,
    stdev: float | None = None,
    low: float = settings.KEYSTROKE_DELAY_MIN_MS,
    high: float = settings.KEYSTROKE_DELAY_MAX_MS,
) -> float:
    """A keystroke gap in ms, normally distributed and clamped to [low, high]."""
    mean = settings.KEYSTROKE_DELAY_MEAN_MS if mean is None else mean
    stdev = settings.KEYSTROKE_DELAY_STDEV_MS if stdev is None else stdev
    return min(max(random.gauss(mean, stdev), low), high)


def think_pause(low: float = 0.35, high: float = 1.4) -> float:
    """A between-actions pause, in seconds."""
    return random.uniform(low, high)


def bezier_path(
    start: tuple[float, float], end: tuple[float, float], steps: int = 24
) -> list[tuple[float, float]]:
    """Points along a quadratic Bezier from start to end.

    A straight interpolation produces perfectly collinear mouse samples; a
    single control point offset perpendicular to the path gives the gentle arc
    a hand actually makes. Also makes hover-dependent menus fire correctly.
    """
    (x0, y0), (x1, y1) = start, end
    dx, dy = x1 - x0, y1 - y0
    distance = math.hypot(dx, dy) or 1.0

    # Control point: midpoint pushed perpendicular to the path. The magnitude
    # has a floor as well as a ceiling - drawing it from a range straddling zero
    # occasionally produced a perfectly straight path, which no hand ever makes.
    offset = random.choice((-1, 1)) * random.uniform(0.04, 0.15) * distance
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    cx, cy = mx - dy / distance * offset, my + dx / distance * offset

    points = []
    for i in range(steps + 1):
        t = i / steps
        # Ease-in-out: start slow, accelerate, settle on the target.
        t = t * t * (3 - 2 * t)
        u = 1 - t
        points.append(
            (u * u * x0 + 2 * u * t * cx + t * t * x1, u * u * y0 + 2 * u * t * cy + t * t * y1)
        )
    return points


# --------------------------------------------------------------------------
# Confirmation gates
# --------------------------------------------------------------------------


def confirm(prompt: str, default: bool = False) -> bool:
    """Blocking yes/no gate on the terminal.

    In a non-interactive session there is no human to ask, so this returns
    ``default`` (False for every destructive gate) rather than proceeding.
    """
    if not sys.stdin or not sys.stdin.isatty():
        log.warning("No TTY available; declining gate: %s", prompt)
        return default
    suffix = " [y/N] " if not default else " [Y/n] "
    try:
        reply = input("\n" + prompt + suffix).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not reply:
        return default
    return reply in ("y", "yes")


def alert(title: str, lines: list[str] | None = None) -> None:
    """Loud terminal banner for anything a human must look at."""
    width = 74
    print("\n" + "!" * width)
    print(f"  {title}")
    for line in lines or []:
        print(f"    - {line}")
    print("!" * width)


# --------------------------------------------------------------------------
# Browser
# --------------------------------------------------------------------------

CAPTCHA_SIGNATURES = (
    "iframe[src*='recaptcha']",
    "iframe[src*='hcaptcha']",
    "iframe[title*='captcha' i]",
    "div.g-recaptcha",
    "div.h-captcha",
    "div.cf-turnstile",
    "#px-captcha",
    "[data-sitekey]",
)

CAPTCHA_TEXT = (
    "verify you are human",
    "i'm not a robot",
    "complete the security check",
    "unusual traffic",
)


class HumanBrowser:
    """Context-managed Playwright session with human-paced interactions."""

    def __init__(
        self,
        user_data_dir: Path | None = None,
        headless: bool | None = None,
        proxy: str | None = None,
        gatekeeper: Gatekeeper | None = None,
    ) -> None:
        self.user_data_dir = Path(user_data_dir or settings.BROWSER_PROFILE_DIR)
        self.headless = settings.HEADLESS if headless is None else headless
        self.proxy = proxy or settings.PROXY_SERVER
        #: Who gets asked when the run needs a human. Terminal by default; the
        #: web runner swaps in a DashboardGatekeeper.
        self.gatekeeper = gatekeeper or TerminalGatekeeper()
        self._playwright = None
        self.context = None
        self.page = None
        #: Where locators are resolved. Defaults to `page`, but a driver may
        #: point it at an iframe (Greenhouse embeds its form in one). Mouse and
        #: keyboard always stay on `page`; Playwright reports bounding boxes
        #: relative to the main frame, so coordinates remain valid either way.
        self.root = None
        self._mouse_pos = (400.0, 400.0)

    # ---------------- lifecycle ----------------

    def start(self) -> "HumanBrowser":
        from playwright.sync_api import sync_playwright

        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()

        launch: dict[str, Any] = {
            "user_data_dir": str(self.user_data_dir),
            "headless": self.headless,
            "viewport": settings.VIEWPORT,
            "slow_mo": settings.SLOW_MO_MS,
            "accept_downloads": True,
            # Persistent profile keeps portal logins between runs, so repeat
            # applications to the same ATS do not re-register an account.
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        if self.proxy:
            launch["proxy"] = {"server": self.proxy}

        self.context = self._playwright.chromium.launch_persistent_context(**launch)
        self.context.set_default_timeout(settings.NAV_TIMEOUT_MS)
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        log.info("Browser started (headless=%s, profile=%s)", self.headless, self.user_data_dir)
        return self

    def close(self) -> None:
        try:
            if self.context:
                self.context.close()
        finally:
            if self._playwright:
                self._playwright.stop()
            self._playwright = self.context = self.page = None

    def __enter__(self) -> "HumanBrowser":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------------- navigation ----------------

    def goto(self, url: str, wait_until: str = "domcontentloaded") -> None:
        log.info("Navigating to %s", url)
        self.root = None  # a new document invalidates any frame we were scoped to
        self.page.goto(url, wait_until=wait_until, timeout=settings.NAV_TIMEOUT_MS)
        time.sleep(think_pause(0.8, 2.0))
        self.raise_on_captcha()

    # ---------------- human-paced interactions ----------------

    def move_mouse_to(self, x: float, y: float) -> None:
        for px, py in bezier_path(self._mouse_pos, (x, y)):
            self.page.mouse.move(px, py)
            time.sleep(random.uniform(0.004, 0.016))
        self._mouse_pos = (x, y)

    def human_click(self, locator: Any, settle: bool = True) -> None:
        """Move the mouse along a curve to the element, then click it."""
        loc = self._resolve(locator)
        loc.scroll_into_view_if_needed()
        time.sleep(think_pause(0.15, 0.5))

        box = loc.bounding_box()
        if box:
            # Aim for a random point in the middle 60% of the element, not dead centre.
            target_x = box["x"] + box["width"] * random.uniform(0.2, 0.8)
            target_y = box["y"] + box["height"] * random.uniform(0.3, 0.7)
            self.move_mouse_to(target_x, target_y)
            self.page.mouse.click(target_x, target_y)
        else:
            loc.click()

        if settle:
            time.sleep(think_pause(0.3, 0.9))

    def human_type(self, locator: Any, text: str, clear: bool = True) -> None:
        """Type with Gaussian inter-key delays (40-120ms), plus word pauses."""
        loc = self._resolve(locator)
        self.human_click(loc, settle=False)
        if clear:
            loc.fill("")
            time.sleep(think_pause(0.1, 0.3))

        for char in str(text):
            self.page.keyboard.type(char, delay=gaussian_delay_ms())
            # People pause fractionally at word boundaries.
            if char == " " and random.random() < 0.18:
                time.sleep(random.uniform(0.05, 0.22))
        time.sleep(think_pause(0.2, 0.6))

    def human_select(self, locator: Any, value: str) -> None:
        loc = self._resolve(locator)
        loc.scroll_into_view_if_needed()
        time.sleep(think_pause(0.2, 0.6))
        loc.select_option(label=value)
        time.sleep(think_pause(0.2, 0.7))

    def human_scroll(self, total_px: int = 900, steps: int = 6) -> None:
        """Scroll in uneven increments with reading pauses."""
        remaining = total_px
        for _ in range(steps):
            if remaining <= 0:
                break
            delta = int(min(remaining, random.gauss(total_px / steps, total_px / (steps * 3))))
            delta = max(delta, 40)
            self.page.mouse.wheel(0, delta)
            remaining -= delta
            time.sleep(random.uniform(0.25, 1.1))

    def upload_file(self, locator: Any, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot upload missing file: {path}")
        self._resolve(locator).set_input_files(str(path))
        log.info("Uploaded %s", path.name)
        time.sleep(think_pause(0.8, 2.0))

    @property
    def dom(self) -> Any:
        """The Page or Frame that locators resolve against."""
        return self.root if self.root is not None else self.page

    def use_frame(self, frame: Any | None) -> None:
        """Point locator resolution at an iframe (None resets to the page)."""
        self.root = frame

    def _resolve(self, locator: Any) -> Any:
        return self.dom.locator(locator) if isinstance(locator, str) else locator

    # ---------------- guardrails ----------------

    def detect_captcha(self) -> str | None:
        """Return a description of the CAPTCHA on screen, or None."""
        for selector in CAPTCHA_SIGNATURES:
            try:
                if self.page.locator(selector).first.is_visible(timeout=350):
                    return f"CAPTCHA element matched: {selector}"
            except Exception:
                continue
        try:
            body = (self.page.inner_text("body", timeout=1500) or "").lower()
        except Exception:
            return None
        for phrase in CAPTCHA_TEXT:
            if phrase in body:
                return f"CAPTCHA phrase detected: {phrase!r}"
        return None

    def raise_on_captcha(self) -> None:
        """Park the run when a human-verification challenge appears.

        Nothing here tries to solve or evade the challenge. The browser stays
        open on it, the supervising human is told what is blocking, and the run
        resumes only once they have cleared it themselves.
        """
        found = self.detect_captcha()
        if not found:
            return

        approved = self.gatekeeper.confirm(
            "continue past a human-verification challenge",
            [
                found,
                f"URL: {self.page.url}",
                "The browser window is parked on the challenge. Clear it yourself, "
                "then approve to resume.",
            ],
        )
        if not approved:
            raise ManualInterventionRequired(found)

        time.sleep(think_pause(0.5, 1.2))
        if self.detect_captcha():
            raise ManualInterventionRequired(
                "Verification challenge still present after handoff; stopping."
            )

    def hand_off(self, reason: str, details: list[str] | None = None) -> None:
        """Give the browser to the supervisor and block until they hand it back."""
        if not self.gatekeeper.confirm(f"continue after handling: {reason}", details):
            raise ManualInterventionRequired(reason)

    @contextmanager
    def gate(self, action: str, details: list[str] | None = None) -> Iterator[None]:
        """Wrap an irreversible action in an explicit confirmation."""
        if not self.gatekeeper.confirm(action, details):
            raise HumanDeclined(action)
        yield

    # ---------------- diagnostics ----------------

    def screenshot(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(path), full_page=True)
        return path
