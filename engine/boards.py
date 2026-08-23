"""Clients for public ATS job-board APIs.

Greenhouse, Lever, Ashby and Workable all publish their boards as documented
JSON endpoints intended for consumption. Using them is both more reliable and
more honest than scraping: the data is structured, current, and served
deliberately. Nothing here scrapes a search page or logs into anything.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable

log = logging.getLogger(__name__)

TIMEOUT = 15
USER_AGENT = "job-pipeline/1.0 (+personal job search)"


@dataclass
class Posting:
    """One job posting from a board."""

    company: str
    title: str
    url: str
    location: str = ""
    board: str = ""
    external_id: str = ""
    description: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Stable identity for de-duplication across runs."""
        return f"{self.board}:{self.external_id}" if self.external_id else self.url


class BoardError(Exception):
    """The board could not be read."""


def _get_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise BoardError(f"HTTP {exc.code} from {url}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise BoardError(f"Could not reach {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BoardError(f"{url} did not return JSON") from exc


def _strip_html(text: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text or "", flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>|</p>|</div>|</li>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = (text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                .replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"'))
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", text)).strip()


# --------------------------------------------------------------------------
# Boards
# --------------------------------------------------------------------------


def greenhouse(token: str, with_content: bool = True) -> list[Posting]:
    """https://boards-api.greenhouse.io/v1/boards/{token}/jobs"""
    token = token.strip().strip("/")
    url = (f"https://boards-api.greenhouse.io/v1/boards/{urllib.parse.quote(token)}"
           f"/jobs{'?content=true' if with_content else ''}")
    payload = _get_json(url)
    postings = []
    for job in payload.get("jobs", []) or []:
        postings.append(Posting(
            company=token,
            title=job.get("title", ""),
            url=job.get("absolute_url", ""),
            location=(job.get("location") or {}).get("name", ""),
            board="greenhouse",
            external_id=str(job.get("id", "")),
            description=_strip_html(job.get("content", "")),
            updated_at=job.get("updated_at", ""),
        ))
    return postings


def lever(company: str) -> list[Posting]:
    """https://api.lever.co/v0/postings/{company}?mode=json"""
    company = company.strip().strip("/")
    payload = _get_json(
        f"https://api.lever.co/v0/postings/{urllib.parse.quote(company)}?mode=json"
    )
    postings = []
    for job in payload or []:
        categories = job.get("categories") or {}
        postings.append(Posting(
            company=company,
            title=job.get("text", ""),
            url=job.get("hostedUrl", ""),
            location=categories.get("location", ""),
            board="lever",
            external_id=str(job.get("id", "")),
            description=_strip_html(job.get("descriptionPlain") or job.get("description", "")),
            updated_at=str(job.get("createdAt", "")),
        ))
    return postings


def ashby(board: str) -> list[Posting]:
    """https://api.ashbyhq.com/posting-api/job-board/{board}"""
    board = board.strip().strip("/")
    payload = _get_json(
        f"https://api.ashbyhq.com/posting-api/job-board/{urllib.parse.quote(board)}"
        "?includeCompensation=true"
    )
    postings = []
    for job in payload.get("jobs", []) or []:
        postings.append(Posting(
            company=payload.get("name") or board,
            title=job.get("title", ""),
            url=job.get("jobUrl", ""),
            location=job.get("location", ""),
            board="ashby",
            external_id=str(job.get("id", "")),
            description=_strip_html(job.get("descriptionPlain")
                                    or job.get("descriptionHtml", "")),
            updated_at=job.get("publishedAt", ""),
        ))
    return postings


#: board name -> fetcher
FETCHERS = {"greenhouse": greenhouse, "lever": lever, "ashby": ashby}


# --------------------------------------------------------------------------
# Board detection
# --------------------------------------------------------------------------

_BOARD_PATTERNS = [
    ("greenhouse", re.compile(r"(?:job-)?boards\.greenhouse\.io/([A-Za-z0-9_-]+)")),
    ("greenhouse", re.compile(r"boards-api\.greenhouse\.io/v1/boards/([A-Za-z0-9_-]+)")),
    ("lever", re.compile(r"jobs\.(?:eu\.)?lever\.co/([A-Za-z0-9_-]+)")),
    ("ashby", re.compile(r"jobs\.ashbyhq\.com/([A-Za-z0-9_.-]+)")),
]


def detect_board(url: str) -> tuple[str, str] | None:
    """Return (board_name, slug) for a careers URL, or None."""
    for name, pattern in _BOARD_PATTERNS:
        match = pattern.search(url or "")
        if match:
            return name, match.group(1)
    return None


def fetch_board(board: str, slug: str) -> list[Posting]:
    fetcher = FETCHERS.get(board)
    if fetcher is None:
        raise BoardError(f"No client for board {board!r}.")
    return fetcher(slug)


def fetch_any(url_or_slug: str) -> list[Posting]:
    """Fetch from a careers URL, or try each board for a bare company slug."""
    detected = detect_board(url_or_slug)
    if detected:
        return fetch_board(*detected)

    slug = url_or_slug.strip().strip("/").lower()
    if not re.fullmatch(r"[a-z0-9_.-]+", slug):
        raise BoardError(f"{url_or_slug!r} is not a recognised board URL or company slug.")

    errors = []
    for name, fetcher in FETCHERS.items():
        try:
            postings = fetcher(slug)
            if postings:
                return postings
        except BoardError as exc:
            errors.append(f"{name}: {exc}")
    raise BoardError(f"No board found for {slug!r}. Tried: {'; '.join(errors)}")


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------


def matches(
    posting: Posting,
    titles: Iterable[str] = (),
    locations: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> bool:
    """Cheap local filter, applied before anything is sent to an LLM."""
    title = (posting.title or "").lower()
    where = (posting.location or "").lower()

    if any(term.lower() in title for term in exclude if term):
        return False

    title_terms = [t.lower() for t in titles if t]
    if title_terms and not any(term in title for term in title_terms):
        return False

    location_terms = [loc.lower() for loc in locations if loc]
    if location_terms and not any(
        term in where or term in title or "remote" in where for term in location_terms
    ):
        return False
    return True


def dedupe(postings: Iterable[Posting], seen_keys: set[str] | None = None) -> list[Posting]:
    """Drop repeats within the batch and anything already seen."""
    seen = set(seen_keys or ())
    unique: list[Posting] = []
    for posting in postings:
        if not posting.url or posting.key in seen:
            continue
        seen.add(posting.key)
        unique.append(posting)
    return unique
