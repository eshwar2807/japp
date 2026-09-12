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
    #: The board's own account name, which is not always the company name -
    #: Ashby and Workable report a display name. Needed to build apply_url.
    slug: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Stable identity for de-duplication across runs."""
        return f"{self.board}:{self.external_id}" if self.external_id else self.url

    @property
    def apply_url(self) -> str:
        """The ATS-hosted page that actually carries the application form.

        `url` is whatever the employer configured, and for a large employer that
        is usually their own careers site. Stripe's Greenhouse board returns
        `stripe.com/jobs/search?gh_jid=7217048` - a marketing page listing every
        open role, with no form on it at all. Sending the browser there produced
        "no resume input", "1 field discovered" and "submit button not found",
        none of which were form-filling faults.

        The board, the account and the job id are enough to address the ATS
        directly, so prefer that and keep `url` for the human-facing link.

        Greenhouse gets its embed endpoint rather than the hosted job page.
        An employer can point the hosted page back at their own site, and
        Stripe does: job-boards.greenhouse.io/stripe/jobs/7217048 redirects to
        stripe.com/careers/search, landing us on the same formless listing. The
        embed endpoint serves the bare application form and redirects nowhere -
        verified against both a redirecting board (Stripe) and a plain one
        (Flexport).

        The company name is never substituted for a missing slug: "Acme Corp"
        is a display name, not a path segment, and building on it yields a 404
        that looks like a real link.
        """
        slug = urllib.parse.quote(self.slug.strip("/"))
        job = urllib.parse.quote(str(self.external_id).strip("/"))
        if not slug or not job:
            return self.url
        canonical = {
            "greenhouse": ("https://job-boards.greenhouse.io/embed/job_app"
                           f"?for={slug}&token={job}"),
            "lever": f"https://jobs.lever.co/{slug}/{job}/apply",
            "ashby": f"https://jobs.ashbyhq.com/{slug}/{job}/application",
            "smartrecruiters": f"https://jobs.smartrecruiters.com/{slug}/{job}",
            "workable": f"https://apply.workable.com/{slug}/j/{job}/apply/",
        }.get(self.board, "")
        return canonical or self.url


#: An employer careers page that wraps a Greenhouse posting identifies it with
#: this parameter, whatever else the URL looks like.
_GH_JID = re.compile(r"[?&]gh_jid=(\d+)")


#: Text every Greenhouse application page carries and no listing page does.
_FORM_MARKERS = ("submit application", 'type="file"')


def looks_like_an_application_form(html: str) -> bool:
    """Whether a fetched page is a form to fill, rather than a list of jobs.

    The distinction is not decorative. A careers listing answers 200 and looks
    fine; opening one cost a browser session and a question to the user about a
    submit button that was never going to be there.
    """
    low = (html or "").lower()
    return all(marker in low for marker in _FORM_MARKERS)


def apply_url_candidates(url: str, slug: str = "", job_id: str = "") -> list[str]:
    """Every address a Greenhouse posting might serve its form from, in order.

    Neither route works everywhere, which is why both are tried:

    - the hosted job page is canonical, but an employer can point it back at
      their own careers site, and Stripe and Databricks both do;
    - the embed endpoint always serves the bare form where it is enabled, but
      it is not enabled on every board - Databricks answers 404.

    Verified against Stripe (embed only), Databricks (neither), and Flexport
    (both).
    """
    found = _GH_JID.search(url or "")
    job_id = job_id or (found.group(1) if found else "")
    if not slug:
        detected = detect_board(url)
        slug = detected[1] if detected and detected[0] == "greenhouse" else ""

    out = [url] if url else []
    if slug and job_id:
        out.append(f"https://job-boards.greenhouse.io/{slug}/jobs/{job_id}")
        out.append("https://job-boards.greenhouse.io/embed/job_app"
                   f"?for={slug}&token={job_id}")
    if job_id:
        out.append(f"https://boards.greenhouse.io/embed/job_app?token={job_id}")

    seen: set[str] = set()
    return [u for u in out if not (u in seen or seen.add(u))]


def find_application_form(url: str, slug: str = "", job_id: str = "") -> str | None:
    """The first candidate that actually serves a form, or None if none does.

    None is a real answer: some employers only accept applications through
    their own scripted careers site. Saying so plainly is more useful than
    opening the listing page and asking the user where the submit button went.

    "No form" and "could not tell" are deliberately different outcomes. A 404
    settles the question; a timeout or a rate-limit does not, and answering None
    to those would quietly write off a real application as manual work. When
    nothing was found and some candidate never answered, this raises BoardError
    so the caller can go on and let the browser decide.
    """
    inconclusive = False
    for candidate in apply_url_candidates(url, slug, job_id):
        try:
            request = urllib.request.Request(
                candidate, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                body = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            # A 4xx other than "slow down" means this address genuinely has
            # nothing on it. Anything else says only that the ask failed.
            if 400 <= exc.code < 500 and exc.code != 429:
                log.debug("%s: HTTP %s", candidate, exc.code)
            else:
                inconclusive = True
                log.debug("%s: HTTP %s, inconclusive", candidate, exc.code)
            continue
        except Exception as exc:
            inconclusive = True
            log.debug("%s did not answer: %s", candidate, exc)
            continue
        if looks_like_an_application_form(body):
            return candidate

    if inconclusive:
        raise BoardError(f"could not determine whether {url} has an application form")
    return None


#: How a Greenhouse job id appears across the URL shapes we store.
_GH_ID_IN_URL = re.compile(r"/jobs/(\d+)|[?&](?:token|gh_jid)=(\d+)")


def location_for_url(url: str) -> str:
    """Ask the board where a posting is, for a row that did not record it.

    Jobs queued before the location was stored on the row have nothing for the
    filter to check, and skipping the check for them is how a Vietnam-only role
    would still reach the browser. One cheap API call closes that gap and the
    answer comes from the employer rather than from a guess.

    An employer careers link carries the job id but names no board, so the
    board account is guessed from the hostname - stripe.com -> "stripe". That
    guess is never trusted: it is confirmed against the board, and a wrong one
    returns "" rather than another company's location.

    Returns "" when the posting cannot be identified; the caller treats that as
    "unknown", not as "allowed".
    """
    detected = detect_board(url or "")

    if detected and detected[0] == "lever":
        # Lever addresses a single posting directly.
        posting_id = urllib.parse.urlparse(url).path.rstrip("/").split("/")[-1]
        try:
            payload = _get_json(
                f"https://api.lever.co/v0/postings/"
                f"{urllib.parse.quote(detected[1])}/{urllib.parse.quote(posting_id)}"
            )
        except BoardError:
            return ""
        return (payload.get("categories") or {}).get("location", "") or ""

    if detected and detected[0] == "ashby":
        # Ashby publishes the board as a whole, so the posting is found in it.
        posting_id = urllib.parse.urlparse(url).path.rstrip("/").split("/")[-1]
        try:
            payload = _get_json(
                f"https://api.ashbyhq.com/posting-api/job-board/"
                f"{urllib.parse.quote(detected[1])}"
            )
        except BoardError:
            return ""
        for job in payload.get("jobs", []) or []:
            if str(job.get("id", "")) == posting_id:
                return job.get("location", "") or ""
        return ""

    found = _GH_ID_IN_URL.search(url or "")
    job_id = (found.group(1) or found.group(2)) if found else ""
    if not job_id:
        return ""

    if detected and detected[0] == "greenhouse":
        candidates = [detected[1]]
    else:
        host = urllib.parse.urlparse(url or "").netloc.lower()
        host = re.sub(r"^(www|jobs|careers|boards)\.", "", host)
        label = host.split(".")[0] if host else ""
        candidates = [label] if label else []

    for slug in candidates:
        try:
            payload = _get_json(
                f"https://boards-api.greenhouse.io/v1/boards/"
                f"{urllib.parse.quote(slug)}/jobs/{urllib.parse.quote(job_id)}"
            )
        except BoardError as exc:
            log.debug("could not read the location for %s: %s", url, exc)
            continue
        # The board answered for this exact id, so the guess was right.
        if str(payload.get("id", "")) == job_id:
            return (payload.get("location") or {}).get("name", "") or ""
    return ""


def canonicalize_apply_url(url: str) -> str:
    """Rewrite an employer careers link into the ATS form it wraps.

    `Posting.apply_url` does this at discovery time, when the board and account
    are still in hand. This is the same repair for a URL already stored without
    them - a queued row, or a link typed in by hand. A Greenhouse job id is
    globally unique, so its embed endpoint resolves from the id alone.

    Anything unrecognised is returned untouched rather than guessed at.
    """
    found = _GH_JID.search(url or "")
    if found:
        return f"https://boards.greenhouse.io/embed/job_app?token={found.group(1)}"
    return url


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
            slug=token,
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
            slug=company,
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
            slug=board,
            external_id=str(job.get("id", "")),
            description=_strip_html(job.get("descriptionPlain")
                                    or job.get("descriptionHtml", "")),
            updated_at=job.get("publishedAt", ""),
        ))
    return postings


def smartrecruiters(company: str, detail_limit: int = 60) -> list[Posting]:
    """https://api.smartrecruiters.com/v1/companies/{company}/postings

    The listing carries titles and locations but not descriptions, and one
    detail request per posting would be hundreds of calls for a large employer.
    So the listing is paged in full and details are fetched only for the
    postings a caller could plausibly want, newest first.
    """
    company = company.strip().strip("/")
    base = f"https://api.smartrecruiters.com/v1/companies/{urllib.parse.quote(company)}"

    raw: list[dict[str, Any]] = []
    offset, page = 0, 100
    while True:
        payload = _get_json(f"{base}/postings?limit={page}&offset={offset}")
        batch = payload.get("content") or []
        raw.extend(batch)
        offset += page
        if len(batch) < page or offset >= int(payload.get("totalFound") or 0) or offset >= 400:
            break

    postings = []
    for job in raw:
        location = job.get("location") or {}
        postings.append(Posting(
            company=(job.get("company") or {}).get("name") or company,
            title=job.get("name", ""),
            # The public apply page, not the API URL the listing returns.
            url=f"https://jobs.smartrecruiters.com/{company}/{job.get('id', '')}",
            location=location.get("fullLocation") or ", ".join(
                str(p) for p in (location.get("city"), location.get("region"),
                                 location.get("country")) if p),
            board="smartrecruiters",
            slug=company,
            external_id=str(job.get("id", "")),
            updated_at=str(job.get("releasedDate", "")),
            metadata={"detail_url": f"{base}/postings/{job.get('id', '')}"},
        ))
    return postings


def smartrecruiters_description(posting: Posting) -> str:
    """Fetch one SmartRecruiters posting's text, since the listing omits it."""
    detail_url = (posting.metadata or {}).get("detail_url")
    if not detail_url:
        return ""
    try:
        payload = _get_json(detail_url)
    except BoardError:
        return ""
    sections = ((payload.get("jobAd") or {}).get("sections") or {})
    parts = [
        (sections.get(key) or {}).get("text", "")
        for key in ("companyDescription", "jobDescription", "qualifications",
                    "additionalInformation")
    ]
    return _strip_html("\n\n".join(p for p in parts if p))


def workable_location(job: dict[str, Any]) -> str:
    """Where a Workable posting is.

    The client originally read a nested `location` object, which this board
    does not return, so every posting came back with no location at all - and a
    posting with no location is one the country filter cannot examine. The real
    shape is a `locations` list, with flat city/state/country as a fallback:

        "locations": [{"country": "Slovakia", "countryCode": "SK",
                       "city": "Bratislava", "region": "Bratislava Region"}]

    Every entry is kept, separated the way the other boards separate them, so a
    posting open in several countries is judged on all of them rather than on
    whichever happened to be first.
    """
    places = []
    for place in job.get("locations") or []:
        if not isinstance(place, dict) or place.get("hidden"):
            continue
        parts = [place.get("city"), place.get("region"), place.get("country")]
        joined = ", ".join(str(p) for p in parts if p)
        if joined:
            places.append(joined)

    if not places:
        parts = [job.get("city"), job.get("state"), job.get("country")]
        joined = ", ".join(str(p) for p in parts if p)
        if joined:
            places.append(joined)

    if not places and job.get("telecommuting"):
        return "Remote"
    where = "; ".join(dict.fromkeys(places))
    if where and job.get("telecommuting"):
        where = f"{where} (remote)"
    return where


def workable(account: str) -> list[Posting]:
    """https://apply.workable.com/api/v1/widget/accounts/{account}"""
    account = account.strip().strip("/").lower()
    payload = _get_json(
        f"https://apply.workable.com/api/v1/widget/accounts/"
        f"{urllib.parse.quote(account)}?details=true"
    )
    postings = []
    for job in payload.get("jobs") or []:
        postings.append(Posting(
            company=payload.get("name") or account,
            title=job.get("title", ""),
            # The board's own apply link when it gives one; it addresses the
            # posting by shortcode without the account, which apply_url cannot
            # know to do.
            url=job.get("application_url") or job.get("url")
                or job.get("shortlink", ""),
            location=workable_location(job),
            board="workable",
            slug=account,
            external_id=str(job.get("shortcode") or job.get("id", "")),
            description=_strip_html(job.get("description", "")),
            updated_at=str(job.get("published_on", "")),
        ))
    return postings


#: board name -> fetcher
FETCHERS = {
    "greenhouse": greenhouse,
    "lever": lever,
    "ashby": ashby,
    "smartrecruiters": smartrecruiters,
    "workable": workable,
}


# --------------------------------------------------------------------------
# Board detection
# --------------------------------------------------------------------------

_BOARD_PATTERNS = [
    ("greenhouse", re.compile(r"(?:job-)?boards\.greenhouse\.io/([A-Za-z0-9_-]+)")),
    ("greenhouse", re.compile(r"boards-api\.greenhouse\.io/v1/boards/([A-Za-z0-9_-]+)")),
    ("lever", re.compile(r"jobs\.(?:eu\.)?lever\.co/([A-Za-z0-9_-]+)")),
    ("ashby", re.compile(r"jobs\.ashbyhq\.com/([A-Za-z0-9_.-]+)")),
    ("smartrecruiters", re.compile(r"jobs\.smartrecruiters\.com/([A-Za-z0-9_.-]+)")),
    ("smartrecruiters", re.compile(r"api\.smartrecruiters\.com/v1/companies/([A-Za-z0-9_.-]+)")),
    ("workable", re.compile(r"apply\.workable\.com/([A-Za-z0-9_-]+)")),
    ("workable", re.compile(r"([A-Za-z0-9_-]+)\.workable\.com")),
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
    "vietnam": r"\b(vietnam|viet nam|hanoi|ha noi|danang|da nang|ho chi minh|saigon)\b",
    "indonesia": r"\b(indonesia|jakarta|bandung)\b",
    "thailand": r"\b(thailand|bangkok)\b",
    "malaysia": r"\b(malaysia|kuala lumpur)\b",
    "south korea": r"\b(south korea|korea|seoul)\b",
    "taiwan": r"\b(taiwan|taipei)\b",
    "pakistan": r"\b(pakistan|karachi|lahore|islamabad)\b",
    "bangladesh": r"\b(bangladesh|dhaka)\b",
    "sri lanka": r"\b(sri lanka|colombo)\b",
    "turkey": r"\b(turkey|turkiye|istanbul|ankara)\b",
    "egypt": r"\b(egypt|cairo)\b",
    "nigeria": r"\b(nigeria|lagos|abuja)\b",
    "kenya": r"\b(kenya|nairobi)\b",
    "south africa": r"\b(south africa|johannesburg|cape town|pretoria)\b",
    "uae": r"\b(uae|united arab emirates|dubai|abu dhabi)\b",
    "saudi arabia": r"\b(saudi arabia|riyadh|jeddah)\b",
    "czechia": r"\b(czechia|czech republic|prague|brno)\b",
    "hungary": r"\b(hungary|budapest)\b",
    "bulgaria": r"\b(bulgaria|sofia)\b",
    "greece": r"\b(greece|athens)\b",
    "italy": r"\b(italy|rome|milan)\b",
    "austria": r"\b(austria|vienna)\b",
    "belgium": r"\b(belgium|brussels)\b",
    "denmark": r"\b(denmark|copenhagen)\b",
    "norway": r"\b(norway|oslo)\b",
    "finland": r"\b(finland|helsinki)\b",
    "estonia": r"\b(estonia|tallinn)\b",
    "lithuania": r"\b(lithuania|vilnius)\b",
    "latvia": r"\b(latvia|riga)\b",
    "croatia": r"\b(croatia|zagreb)\b",
    "chile": r"\b(chile|santiago)\b",
    "peru": r"\b(peru|lima)\b",
    "uruguay": r"\b(uruguay|montevideo)\b",
    "guatemala": r"\bguatemala\b",
    "dominican republic": r"\b(dominican republic|santo domingo)\b",
    "armenia": r"\b(armenia|yerevan)\b",
    "georgia country": r"\b(tbilisi)\b",
}

#: Two-letter state codes, which is how a board most often says "United
#: States": "Austin, TX". Kept apart from _US_MARKER because these collide with
#: country codes - "Toronto, CA" is Canada, not California - so a non-US
#: country named in the same string always wins over one of these.
_US_STATE_CODE = re.compile(
    r",\s*(a[lkzr]|c[aot]|de|fl|ga|hi|i[dlan]|k[sy]|la|m[edaintso]|"
    r"n[cdehjmvy]|oh|ok|or|pa|ri|s[cd]|tn|tx|ut|v[at]|w[aivy]|dc)\b",
    re.IGNORECASE,
)

#: Words that describe how you work, not where. A posting saying only these
#: names no place at all and stays eligible for any country.
_PLACELESS = re.compile(
    r"\b(remote|hybrid|on-?site|in-?office|anywhere|global|worldwide|"
    r"distributed|flexible|various|multiple locations|work from home|wfh|"
    r"field|travel|headquarters|hq|office|home based|virtual|other|n/?a)\b",
    re.IGNORECASE,
)


def names_a_place(location: str) -> bool:
    """Whether a location string points at somewhere specific.

    "Remote" and "Hybrid - Flexible" do not; "Danang, Vietnam" does. The
    distinction decides what to do with a string holding no country this module
    recognises: a posting that names nowhere is open to anywhere, but one that
    names a place we could not identify is a place we did not ask for.
    """
    text = _PLACELESS.sub(" ", (location or "").lower())
    return bool(re.search(r"[a-z]{2,}", text))

_US_MARKER = re.compile(
    r"\b(us|usa|u\.s\.?a?\.?|united states|america|american|"
    r"alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|"
    r"florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|"
    r"louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|"
    r"missouri|montana|nebraska|nevada|hampshire|jersey|mexico\b(?! city)|"
    r"new york|carolina|dakota|ohio|oklahoma|oregon|pennsylvania|rhode island|"
    r"tennessee|texas|utah|vermont|virginia|washington|wisconsin|wyoming|"
    r"nyc|sf|bay area|silicon valley|"
    # Cities a board sometimes gives with no state after them. Without these a
    # bare "Boston" would be refused as an unidentified place now that an
    # unrecognised location no longer passes by default.
    r"austin|boston|chicago|seattle|denver|atlanta|dallas|houston|phoenix|"
    r"portland|miami|minneapolis|detroit|philadelphia|pittsburgh|charlotte|"
    r"raleigh|durham|nashville|columbus|indianapolis|milwaukee|baltimore|"
    r"kansas city|st\.? louis|salt lake|las vegas|orlando|tampa|sacramento|"
    r"san francisco|san jose|san diego|los angeles|oakland|berkeley|"
    r"palo alto|mountain view|sunnyvale|santa clara|santa monica|irvine|"
    r"brooklyn|cambridge|somerville|arlington|bellevue|redmond|plano|"
    r"boulder|chapel hill|ann arbor|madison|boise|omaha|tucson|albuquerque)\b",
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

    A US search now requires positive evidence of the US. The previous version
    ended in "allow anything not recognised", so a country missing from the
    table passed: a Vietnam-only posting was queued because Vietnam was not
    listed. Filling the table in does not fix that shape of bug - Indonesia,
    Turkey and fifty others would each wait their turn - so an unrecognised
    location that still names a place is now refused rather than allowed.
    """
    terms = [t.strip().lower() for t in requested if t and t.strip()]
    if not terms:
        return True

    where = (posting_location or "").lower()
    wants_us = any(_US_MARKER.search(t) or _US_STATE_CODE.search(t) for t in terms)
    wanted_countries = set()
    for term in terms:
        wanted_countries |= countries_named(term)

    posting_countries = countries_named(where)
    us_spelled_out = bool(_US_MARKER.search(where))
    us_named = us_spelled_out or bool(_US_STATE_CODE.search(where))

    if posting_countries:
        # Somewhere specific and recognised. A posting open to several
        # countries including the requested one is still usable - "US or
        # Canada" suits a US-authorised candidate - but only an unambiguous US
        # mention counts here, never a two-letter code: in "Toronto, CA" the
        # CA is Canada.
        if wants_us and us_spelled_out:
            return True
        return bool(posting_countries & wanted_countries)

    if us_named:
        return wants_us or not wanted_countries

    # Nothing recognised. A posting naming no place at all - a bare "Remote" -
    # is open to anywhere and stays. One naming a place we could not identify
    # is a place that was not asked for.
    return not names_a_place(where)


def ensure_description(posting: Posting) -> Posting:
    """Fill in a posting's text if its board omitted it from the listing.

    SmartRecruiters returns titles and locations in bulk but descriptions only
    per posting. Fetching every one would be hundreds of requests for a large
    employer, so this is called after the cheap title and location filters have
    cut the list down.
    """
    if posting.description or posting.board != "smartrecruiters":
        return posting
    posting.description = smartrecruiters_description(posting)
    return posting


def normalize_company(name: str) -> str:
    """Company name reduced for comparison: lowercase, no legal suffixes."""
    text = re.sub(r"[^a-z0-9 ]+", " ", (name or "").lower())
    text = re.sub(
        r"\b(inc|llc|llp|ltd|limited|corp|corporation|co|plc|gmbh|holdings|"
        r"group|technologies|technology|labs|software|systems)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_blocked(company: str, url: str, blocked: Iterable[str]) -> str:
    """Which blocklist entry rules this employer out, or "" if none.

    Matched on whole words rather than substrings, so blocking "Block" cannot
    also rule out "Blockchain Labs", and on the board slug as well as the name -
    a posting reaching us as job-boards.greenhouse.io/palantir has to be caught
    even when the company field says something else.

    This exists because the dashboard already had a "Never show me" box whose
    contents were only ever pasted into the model's prompt as a request. A
    blocklist that depends on the model choosing to honour it is not one.
    """
    terms = [normalize_company(t) for t in blocked or ()]
    terms = [t for t in terms if t]
    if not terms:
        return ""

    name_words = set(normalize_company(company).split())
    detected = detect_board(url or "")
    slug = normalize_company(detected[1]) if detected else ""
    slug_words = set(slug.split())

    for term in terms:
        parts = term.split()
        if parts and (set(parts) <= name_words or set(parts) <= slug_words):
            return term
    return ""


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
