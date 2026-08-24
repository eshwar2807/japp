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


def slug_candidates(company: str) -> list[str]:
    """Plausible board slugs for a company name, most likely first.

    Boards use a company's own shorthand, which is rarely its display name:
    "Global Healthcare Exchange" is `globalhealthcareexchangeinc`, and legal
    suffixes are sometimes kept and sometimes dropped.
    """
    name = (company or "").strip().lower()
    if not name:
        return []

    cleaned = re.sub(r"[^a-z0-9\s-]+", "", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    without_suffix = re.sub(
        r"\b(inc|llc|ltd|corp|corporation|company|co|technologies|technology|"
        r"labs|group|holdings|systems|software|solutions)\b", " ", cleaned
    ).strip()
    without_suffix = re.sub(r"\s+", " ", without_suffix)

    candidates = []
    for base in (cleaned, without_suffix):
        if not base:
            continue
        candidates.extend([
            base.replace(" ", ""),
            base.replace(" ", "-"),
            base.split(" ")[0],
            base.replace(" ", "") + "inc",
        ])
    # Preserve order, drop duplicates and anything implausibly short.
    seen, out = set(), []
    for candidate in candidates:
        if len(candidate) > 2 and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def resolve_board(company: str, hinted_url: str = "",
                  hinted_slug: str = "") -> tuple[str, str, list[Posting]] | None:
    """Find a company's real board by trying candidates until one answers.

    A model asked for a board URL guesses both the slug and the provider, and
    gets the provider wrong often: Confluent and Instructure were both offered
    as Greenhouse when they are on Ashby, so both were discarded as 404s
    despite having live boards. The slug is usually right, so the same slug is
    tried on every provider before the company is given up on.

    Returns (provider, slug, postings) or None.
    """
    attempts: list[tuple[str, str]] = []

    detected = detect_board(hinted_url)
    if detected:
        attempts.append(detected)
        # Same slug, other providers - this is the common failure.
        attempts.extend((name, detected[1]) for name in FETCHERS if name != detected[0])

    if hinted_slug:
        attempts.extend((name, hinted_slug.strip().lower()) for name in FETCHERS)

    for candidate in slug_candidates(company):
        attempts.extend((name, candidate) for name in FETCHERS)

    tried: set[tuple[str, str]] = set()
    for provider, slug in attempts:
        key = (provider, slug)
        if key in tried:
            continue
        tried.add(key)
        try:
            postings = fetch_board(provider, slug)
        except BoardError:
            continue
        if postings:
            log.info("Resolved %s to %s/%s (%d postings)",
                     company, provider, slug, len(postings))
            return provider, slug, postings
    return None


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


#: Countries other than the United States, as they appear in board locations.
_NON_US_COUNTRIES = {
    "canada": r"\b(canada|canadian|toronto|vancouver|montreal|ottawa|calgary)\b",
    "united kingdom": r"\b(united kingdom|\buk\b|england|scotland|wales|london|manchester)\b",
    "ireland": r"\b(ireland|dublin, ireland|irish)\b",
    "india": r"\b(india|bengaluru|bangalore|hyderabad|pune|mumbai|chennai|gurgaon|noida)\b",
    "germany": r"\b(germany|german|berlin|munich|hamburg|frankfurt)\b",
    "france": r"\b(france|french|paris|bordeaux|lyon)\b",
    "netherlands": r"\b(netherlands|amsterdam|dutch)\b",
    "spain": r"\b(spain|madrid|barcelona)\b",
    "portugal": r"\b(portugal|lisbon|porto)\b",
    "poland": r"\b(poland|warsaw|krakow)\b",
    "serbia": r"\b(serbia|belgrade)\b",
    "australia": r"\b(australia|sydney|melbourne)\b",
    "new zealand": r"\b(new zealand|auckland)\b",
    "singapore": r"\bsingapore\b",
    "japan": r"\b(japan|tokyo)\b",
    "china": r"\b(china|beijing|shanghai|shenzhen)\b",
    "hong kong": r"\bhong kong\b",
    "brazil": r"\b(brazil|brasil|sao paulo|s.o paulo)\b",
    "mexico": r"\b(mexico|guadalajara|monterrey)\b",
    "argentina": r"\b(argentina|buenos aires)\b",
    "colombia": r"\b(colombia|bogota|medellin)\b",
    "costa rica": r"\bcosta rica\b",
    "israel": r"\b(israel|tel aviv)\b",
    "switzerland": r"\b(switzerland|zurich|geneva)\b",
    "sweden": r"\b(sweden|stockholm)\b",
    "romania": r"\b(romania|bucharest)\b",
    "ukraine": r"\b(ukraine|kyiv|kiev)\b",
    "philippines": r"\b(philippines|manila)\b",
    "emea": r"\b(emea|apac|latam)\b",
}

_US_MARKER = re.compile(
    r"\b(us|usa|u\.s\.?a?\.?|united states|america|american|"
    r"alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|"
    r"florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|"
    r"louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|"
    r"missouri|montana|nebraska|nevada|hampshire|jersey|mexico\b(?! city)|"
    r"new york|carolina|dakota|ohio|oklahoma|oregon|pennsylvania|rhode island|"
    r"tennessee|texas|utah|vermont|virginia|washington|wisconsin|wyoming|"
    r"nyc|sf|bay area|silicon valley)\b",
    re.IGNORECASE,
)


def countries_named(location: str) -> set[str]:
    """Non-US countries this location string mentions."""
    text = (location or "").lower()
    return {name for name, pattern in _NON_US_COUNTRIES.items()
            if re.search(pattern, text, re.IGNORECASE)}


def location_allowed(posting_location: str, requested: Iterable[str]) -> bool:
    """Is this posting in a place the search asked for?

    "Remote" is not a country. A posting reading "Remote Canada" satisfied a
    US-only search purely because it contained the word remote, which is how a
    Canadian role reached a candidate who needs US authorisation.
    """
    terms = [t.strip().lower() for t in requested if t and t.strip()]
    if not terms:
        return True

    where = (posting_location or "").lower()
    wants_us = any(_US_MARKER.search(term) for term in terms)
    wanted_countries = set()
    for term in terms:
        wanted_countries |= countries_named(term)

    # A posting open to several countries including the requested one is
    # usable: "US or Canada" is fine for a US-authorised candidate.
    if wants_us and _US_MARKER.search(where):
        return True

    posting_countries = countries_named(where)
    if posting_countries:
        # Named somewhere specific: it must be somewhere that was asked for.
        return bool(posting_countries & wanted_countries)

    if _US_MARKER.search(where):
        return wants_us or not wanted_countries

    # No country named at all - a bare "Remote" or "Hybrid". Allow it only if
    # the search did not restrict to a country other than the US.
    return True


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

    # Country is the filter that matters: a role in the wrong country is
    # unusable regardless of how well it fits. City-level filtering is
    # deliberately not enforced - rejecting a New York role because the search
    # said Ohio would discard genuinely viable work, and relocation is a
    # question the application itself asks.
    if not location_allowed(posting.location, locations):
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
