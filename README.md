# Job Application Pipeline

Ingests a job posting, tailors your resume against it, renders an ATS-parseable
PDF, drives the application form in a real browser, and learns from outcomes.

```
web/        app.py               — FastAPI dashboard (auth, CSP, CSRF, rate limits)
            security.py          — Argon2 passwords, API keys, sessions, limiter
            queue_worker.py      — batch queue; parking frees a slot
            runner.py            — pipeline job bodies
            routes/              — HTML pages + JSON API
config/     master_profile.json  — canonical, immutable source of truth
            settings.py          — every tunable in one place
database/   models.py            — Applications / Credentials / Feedback
            db_manager.py        — CRUD + Fernet credential vault
engine/     schemas.py           — Pydantic contracts
            ats_optimizer.py     — two-pass tailoring + deterministic scoring
            pdf_generator.py     — Jinja2 → HTML → PDF, with verification
            screener_mapper.py   — deterministic form-field answering
automation/ stealth_browser.py   — Playwright wrapper, human-paced, human-gated
            gatekeeper.py        — where the pipeline asks a human
            notifier.py          — desktop + webhook alerts when a run parks
            ats_drivers/         — base + workday + greenhouse/lever
templates/  resume_template.html — single-column, parser-friendly
main.py     CLI orchestrator
```

## Hosted at https://japp.fly.dev

The dashboard runs on Fly.io; the browser work runs on your machine. That split
is deliberate: clearing a verification challenge means looking at a real browser
window, and a container in a datacenter has none. Hosting the whole pipeline
would make every such block unresolvable, and datacenter IPs draw more
bot-detection than a home connection.

| Runs on Fly | Runs on your Mac |
|---|---|
| Dashboard, profile, queue, action history, costs, logs, feedback, API | Application form-filling (`apply` jobs) |
| Resume tailoring, PDF rendering, job-description fetch | The browser window you clear challenges in |

`JP_WORKER_KINDS=tailor` enforces it: the hosted worker will not claim an
`apply` job, so a browser never starts where nobody can see it.

### The local agent

```bash
export JP_AGENT_KEY=$(cat ~/.japp-key)     # from Settings -> Dashboard API key
python -m agent --server https://japp.fly.dev
```

It claims apply jobs, drives the browser locally, and reports blocks back to the
dashboard — so you can answer from your phone while the window waits on your
desk. It refuses to send the API key over plain HTTP to a remote host, and reads
the key from the environment or a file rather than argv (arguments are visible
to every process via `ps`).

### Operating it

```bash
fly logs --app japp                       # live logs
fly status --app japp                     # machine health
fly ssh console --app japp                # shell in the container
fly secrets set JP_INVITE_CODE=... --app japp   # rotate the invite code
fly deploy --app japp                     # ship a change
```

State lives on a 1GB encrypted volume at `/data`: the database and generated
PDFs. The Fernet vault key and session key come from Fly secrets and are never
written to the volume, so a volume snapshot leaks neither.

Single machine on purpose — the SQLite database, the queue and any parked
sessions share one process. Two machines would mean two queues fighting over
one volume, so `auto_stop_machines` is off and `min_machines_running` is 1.

### Access

Signup requires an invite code (`JP_INVITE_CODE`). The code is compared with
`hmac.compare_digest` and checked before anything else, so a wrong code cannot
be used to probe which addresses are registered. The account matching
`JP_ADMIN_EMAIL`, or the first account created, gets `/admin`: accounts,
activity, spend, and suspend/restore. That view deliberately shows whether a
user has keys set, never their values, and never another user's profile.

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/playwright install chromium
```

WeasyPrint needs native libraries. On macOS: `brew install pango gdk-pixbuf libffi`.

Then fill in your details and check the setup:

```bash
cp .env.example .env          # add your Anthropic key
$EDITOR config/master_profile.json
.venv/bin/python main.py doctor
```

`doctor` refuses to pass while any `<PLACEHOLDER>` remains, because applying
with placeholder text in your resume is worse than not applying.

## Dashboard

```bash
python run_web.py            # http://127.0.0.1:8000
```

Sign up, fill in your profile, add your Anthropic key in Settings, and work from
there. The dashboard covers:

| Page | What it does |
|---|---|
| **Overview** | Readiness, response rate, spend today, what needs you |
| **Applications** | Every posting; open one for the job link, the tailored resume, the exact screener answers submitted, the portal account, and the run log |
| **Queue** | Batch progress: what is running, what is parked, what is holding a browser |
| **Needs you** | Unanswerable questions, verification challenges, approvals. Answering releases the parked run immediately; answers marked *remember* are reused on every later application |
| **Costs** | Daily spend over 30/90/120/360 days, split by model and pipeline step, plus cost per application |
| **Logs** | Every pipeline event, filterable by level |
| **Settings** | Anthropic key (encrypted), dashboard API key, stored portal credentials, password |

### Security

- **Passwords** — Argon2id. Login failures are rate limited per account and per
  IP, with lockout after repeated failures.
- **Sessions** — signed, expiring cookies carrying a `session_epoch`; changing
  your password invalidates every other session.
- **CSRF** — double-submit token on every state-changing form.
- **API keys** — `Authorization: Bearer …`, stored as SHA-256, shown once,
  compared in constant time, rate limited per key.
- **Isolation** — every query is scoped by user; cross-tenant access returns 404
  rather than confirming the row exists.
- **Secrets** — your Anthropic key and portal passwords are Fernet-encrypted;
  the key is never rendered back, only a masked preview.
- **Headers** — strict CSP with no `unsafe-inline`, `frame-ancestors 'none'`,
  nosniff, HSTS when cookies are Secure. No OpenAPI schema is exposed.

Bind to loopback unless you have HTTPS in front of it — this app holds a
credential vault.

## Batching

Paste several postings on the Applications page, one URL per line. Each runs
until it needs you, then **steps aside so the next one starts** — a block never
stalls the batch. You come back once and clear everything in one sitting.

Two kinds of block, because they cost different things:

| | What happens | Resume |
|---|---|---|
| **Needs an answer** | A field the pipeline will not guess. The browser closes and the slot frees. | Re-runs from the top with your answer. You stay logged in via the persistent profile, and the answer is reused on every later application. |
| **Needs the browser** | A verification challenge, login, or final submit approval. Only resolvable in the live window, so the session is held open. | Continues from exactly where it stopped. Capped at 3 held sessions. |

Answering in the dashboard releases the run straight away — no polling, no
restart. Dismissing an item cancels the run waiting on it.

### Notifications

Enable in Settings so you can walk away from a batch:

- **Desktop** — a local OS notification. Nothing leaves the machine.
- **Webhook** — optional, for ntfy/Slack/Discord. Sends only *what* is blocking
  and a link. Never the posting, your resume, form answers, or credentials.
- **Quiet window** — one alert per window (default 120s), so a ten-item batch
  does not fire ten times.

## JSON API

```bash
curl -H "Authorization: Bearer $JP_KEY" http://127.0.0.1:8000/api/v1/applications
curl -H "Authorization: Bearer $JP_KEY" http://127.0.0.1:8000/api/v1/costs?days=90
```

`/api/v1`: `me`, `applications` (GET/POST), `applications/{id}`,
`applications/{id}/feedback`, `actions`, `actions/{id}/answer`, `queue`,
`queue/{id}`, `queue/{id}/cancel`, `costs`, `logs`.

## CLI

```bash
# Tailor + build the PDF only. No browser, nothing submitted.
python main.py tailor https://boards.greenhouse.io/acme/jobs/123

# Full flow. Opens a visible browser; stops for you before anything irreversible.
python main.py apply https://boards.greenhouse.io/acme/jobs/123 --app-id 4

# History and stats
python main.py review
python main.py review --app-id 4

# Close the loop
python main.py feedback --app-id 4 --status Interview --notes "Recruiter screen Tuesday"

# Stored portal logins
python main.py creds --show --portal myworkdayjobs.com
```

`--jd-file posting.txt` reads the description from a file instead of scraping
the page — more reliable for postings behind JavaScript.

## How the pieces work

**Two-pass tailoring.** Pass 1 extracts what the posting asks for. Pass 2 maps
your verified facts onto that vocabulary under a strict no-fabrication policy:
the model may reorder, rephrase and adopt the posting's terminology, but
company names, titles and dates are copied byte-for-byte, and any requirement
your profile does not support is reported in `keywords_missing` rather than
invented.

**The score is computed locally.** The model's self-reported
`ats_match_percentage` is advisory. The authoritative number comes from
`score_match()`, a deterministic weighted keyword-coverage calculation. If it
lands below target, the engine re-prompts with the specific uncovered keywords,
up to `JP_MAX_TAILOR_ITERATIONS` times, and keeps the best result.

**Screener answers are deterministic.** Legal answers — work authorisation,
sponsorship, clearance, voluntary disclosures — come from `master_profile.json`
via a regex rule table, never from the LLM. The LLM only supplies answers to
posting-specific questions. Anything that matches no rule and no tailored
answer is escalated to you rather than guessed, and a select whose options
don't fit the answer escalates too.

**The feedback loop.** `feedback --status Interview` marks an application as a
positive signal. The next time you tailor for a similar title, those bullets are
passed into the prompt as proven examples. Rejections are excluded.

## Safety model

The pipeline fills forms; you approve anything irreversible.

- **Submission** always waits for an explicit `y` at the terminal.
- **Account creation** fills the form, then hands the browser to you.
- **Human-verification challenges** park the run and hand over. The browser
  window stays open on the challenge, an item appears in *Needs you* naming what
  is blocking, and approving it resumes the run from where it stopped. Nothing
  solves, evades, or outsources the check — no solver service is integrated, and
  a request to add one is the one thing this project will not do.
- **Unmapped fields** stop the run rather than receiving a guess.
- **No TTY, no consent** — in a non-interactive session every gate declines.

`JP_CONFIRM_SUBMIT` and `JP_CONFIRM_REGISTER` exist because they are wired
through the code, not as a suggestion to turn them off.

This module deliberately does not implement browser-fingerprint spoofing or
anti-bot-detection evasion. It runs a real, visible Chromium with a persistent
profile and human-paced interaction, which is what makes automation reliable
against JavaScript-heavy forms (real `input`/`change` events, debounced
validation, lazy-rendered sections).

## Credential vault

Generated passwords are 16 characters with guaranteed character-class coverage,
from `secrets`. They are encrypted with Fernet before they touch the database —
plaintext is never stored, logged, or written to disk. The key lives in
`data/vault.key` (mode 0600) or `JP_ENCRYPTION_KEY`. Lose the key and the stored
passwords are unrecoverable, which is the intent.

## Deviations from the original spec

| Spec | Shipped | Why |
|---|---|---|
| `temperature=0.0` | not sent | Sampling params were removed from current Claude models and return HTTP 400. Determinism comes from strict structured outputs plus a fixed prompt. Still sent for legacy models via `JP_LLM_TEMPERATURE`. |
| Claude 3.5 Sonnet | `claude-opus-5` | 3.5 Sonnet is retired. Configurable via `JP_LLM_MODEL`. |
| `screener_answers: dict[str, str]` | list of pairs on the wire | Strict JSON schema requires `additionalProperties: false`, which forbids free-form dict keys. `TailoredResumeSchema.screener_answers` is still a `dict[str, str]`. |
| `playwright-stealth` | not used | See the safety model above. |
| Automated account registration | form filled, human confirms | Same. |
| CAPTCHA solving / solver-service integration | not implemented | These checks exist to establish a human is present. The run parks and asks you instead — see Batching. |

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

234 tests, no API credit spent — the LLM is stubbed throughout.

The browser tests drive a real headless Chromium against fixture pages
reproducing Greenhouse and Workday label patterns, including an iframe-embedded
form; they skip automatically if Chromium isn't installed. The web tests attack
each control directly: cross-tenant reads, forged sessions, missing CSRF tokens,
revoked API keys, and inline styles that a strict CSP would silently discard.
The queue tests assert the property the batch depends on: with a single worker
slot, a parked job must not stop the next one from starting.
