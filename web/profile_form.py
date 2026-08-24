"""Translate between the master-profile JSON and the dashboard's HTML form."""

from __future__ import annotations

from typing import Any

#: Legal questions rendered as a fixed list, so nothing important is forgotten.
LEGAL_FIELDS = [
    ("work_authorization_us", "Are you legally authorized to work in the US?", ["Yes", "No"]),
    ("requires_sponsorship_now_or_future",
     "Will you now or in the future require visa sponsorship?", ["Yes", "No"]),
    ("visa_status", "Visa / work status", []),
    ("security_clearance", "Security clearance", []),
    ("willing_to_complete_background_check", "Willing to complete a background check?",
     ["Yes", "No"]),
    ("willing_to_complete_drug_test", "Willing to complete a drug test?", ["Yes", "No"]),
    ("age_over_18", "Are you over 18?", ["Yes", "No"]),
    ("previously_employed_here", "Previously employed by the companies you apply to?",
     ["Yes", "No"]),
    ("non_compete_restrictions", "Any non-compete restrictions?", ["Yes", "No"]),
    ("notice_period", "Notice period", []),
    ("earliest_start_date", "Earliest start date", []),
    ("desired_salary", "Desired salary", []),
    ("reference_available_on_request", "References available on request?", ["Yes", "No"]),
]

DISCLOSURE_FIELDS = [
    ("gender", "Gender"),
    ("race_ethnicity", "Race / ethnicity"),
    ("hispanic_or_latino", "Hispanic or Latino"),
    ("veteran_status", "Veteran status"),
    ("disability_status", "Disability status"),
]

DECLINE = "Decline to self-identify"


def blank_profile() -> dict[str, Any]:
    """An empty but structurally complete profile."""
    return {
        "schema_version": "1.0.0",
        "contact": {
            "full_name": "", "preferred_name": "", "email": "", "phone": "",
            "location": {"city": "", "state": "", "country": "United States",
                         "postal_code": "", "willing_to_relocate": False,
                         "remote_preference": "hybrid"},
            "links": {"linkedin": "", "github": "", "portfolio": "", "website": ""},
        },
        "target_titles": [],
        "summary": "",
        "skills": {"hard": [], "tooling": [], "soft": []},
        "experience": [],
        "education": [],
        "certifications": [],
        "projects": [],
        "legal": {key: "" for key, _, _ in LEGAL_FIELDS},
        "voluntary_disclosures": {key: DECLINE for key, _ in DISCLOSURE_FIELDS},
        "preferences": {"how_did_you_hear_about_us": "Company careers page"},
    }


def _lines(value: str | None) -> list[str]:
    return [line.strip() for line in (value or "").splitlines() if line.strip()]


def _text(form: Any, key: str, default: str = "") -> str:
    return str(form.get(key, default) or "").strip()


def _checked(form: Any, key: str) -> bool:
    return str(form.get(key, "")).lower() in ("on", "true", "1", "yes")


def _indexed(form: Any, prefix: str) -> list[int]:
    """Indices present for a repeating block, e.g. exp_0_company -> [0]."""
    found = set()
    for key in form.keys():
        if not key.startswith(f"{prefix}_"):
            continue
        part = key[len(prefix) + 1 :].split("_", 1)[0]
        if part.isdigit():
            found.add(int(part))
    return sorted(found)


def parse_profile_form(form: Any) -> dict[str, Any]:
    """Build a master-profile dict from submitted form data.

    Rows whose primary field is blank are dropped, so an empty "add another"
    block does not become a phantom job on the resume.
    """
    profile = blank_profile()

    profile["contact"].update(
        full_name=_text(form, "full_name"),
        preferred_name=_text(form, "preferred_name"),
        email=_text(form, "email"),
        phone=_text(form, "phone"),
    )
    profile["contact"]["location"].update(
        city=_text(form, "city"),
        state=_text(form, "state"),
        country=_text(form, "country", "United States"),
        postal_code=_text(form, "postal_code"),
        willing_to_relocate=_checked(form, "willing_to_relocate"),
        remote_preference=_text(form, "remote_preference", "hybrid"),
    )
    profile["contact"]["links"].update(
        linkedin=_text(form, "linkedin"),
        github=_text(form, "github"),
        portfolio=_text(form, "portfolio"),
        website=_text(form, "website"),
    )

    profile["summary"] = _text(form, "summary")
    profile["target_titles"] = _lines(form.get("target_titles"))
    profile["skills"] = {
        "hard": _lines(form.get("skills_hard")),
        "tooling": _lines(form.get("skills_tooling")),
        "soft": _lines(form.get("skills_soft")),
    }

    for i in _indexed(form, "exp"):
        company = _text(form, f"exp_{i}_company")
        title = _text(form, f"exp_{i}_title")
        if not company and not title:
            continue
        is_current = _checked(form, f"exp_{i}_is_current")
        profile["experience"].append({
            "company": company,
            "title": title,
            "location": _text(form, f"exp_{i}_location"),
            "start_date": _text(form, f"exp_{i}_start_date"),
            "end_date": None if is_current else (_text(form, f"exp_{i}_end_date") or None),
            "is_current": is_current,
            "employment_type": _text(form, f"exp_{i}_employment_type", "Full-time"),
            "bullets": _lines(form.get(f"exp_{i}_bullets")),
            "tech_used": [t.strip() for t in _text(form, f"exp_{i}_tech").split(",") if t.strip()],
        })

    for i in _indexed(form, "edu"):
        institution = _text(form, f"edu_{i}_institution")
        if not institution:
            continue
        gpa = _text(form, f"edu_{i}_gpa")
        profile["education"].append({
            "institution": institution,
            "degree": _text(form, f"edu_{i}_degree"),
            "field_of_study": _text(form, f"edu_{i}_field_of_study"),
            "location": _text(form, f"edu_{i}_location"),
            "start_date": _text(form, f"edu_{i}_start_date"),
            "end_date": _text(form, f"edu_{i}_end_date"),
            "gpa": float(gpa) if _is_number(gpa) else None,
        })

    for i in _indexed(form, "cert"):
        name = _text(form, f"cert_{i}_name")
        if not name:
            continue
        profile["certifications"].append({
            "name": name,
            "issuer": _text(form, f"cert_{i}_issuer"),
            "issue_date": _text(form, f"cert_{i}_issue_date"),
            "expiry_date": _text(form, f"cert_{i}_expiry_date") or None,
            "credential_id": _text(form, f"cert_{i}_credential_id"),
        })

    for i in _indexed(form, "proj"):
        name = _text(form, f"proj_{i}_name")
        if not name:
            continue
        profile["projects"].append({
            "name": name,
            "description": _text(form, f"proj_{i}_description"),
            "url": _text(form, f"proj_{i}_url"),
            "tech_used": [t.strip() for t in _text(form, f"proj_{i}_tech").split(",") if t.strip()],
        })

    profile["legal"] = {key: _text(form, f"legal_{key}") for key, _, _ in LEGAL_FIELDS}
    profile["voluntary_disclosures"] = {
        key: _text(form, f"vd_{key}", DECLINE) or DECLINE for key, _ in DISCLOSURE_FIELDS
    }
    profile["preferences"] = {
        "how_did_you_hear_about_us": _text(form, "how_did_you_hear_about_us",
                                           "Company careers page"),
        "min_salary": 0,
        "excluded_companies": _lines(form.get("excluded_companies")),
        "cover_letter_default": _text(form, "cover_letter_default"),
    }
    return profile


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def completeness(profile: dict[str, Any]) -> dict[str, Any]:
    """Score how ready the profile is, and name what is still missing.

    Drives the dashboard's readiness banner - the web equivalent of the CLI's
    placeholder check, but phrased as progress rather than a wall.
    """
    contact = profile.get("contact", {})
    location = contact.get("location", {})
    legal = profile.get("legal", {})
    skills = profile.get("skills", {})

    checks = [
        ("Full name", bool(contact.get("full_name"))),
        ("Email", bool(contact.get("email"))),
        ("Phone", bool(contact.get("phone"))),
        ("Location", bool(location.get("city"))),
        ("Professional summary", bool(profile.get("summary"))),
        ("Skills", bool(skills.get("hard") or skills.get("tooling"))),
        ("At least one role", bool(profile.get("experience"))),
        ("Role bullets", any(e.get("bullets") for e in profile.get("experience", []))),
        ("Education", bool(profile.get("education"))),
        ("Work authorization", bool(legal.get("work_authorization_us"))),
        ("Sponsorship answer", bool(legal.get("requires_sponsorship_now_or_future"))),
        ("Earliest start date", bool(legal.get("earliest_start_date"))),
        # Desired salary is deliberately not required. It is answered per
        # posting from whatever range the posting publishes, falling back to a
        # configured default, so demanding one figure up front blocks setup for
        # a value the pipeline no longer needs.
    ]
    missing = [name for name, ok in checks if not ok]
    done = len(checks) - len(missing)
    return {
        "percent": round(100 * done / len(checks)),
        "done": done,
        "total": len(checks),
        "missing": missing,
        "ready": not missing,
    }
