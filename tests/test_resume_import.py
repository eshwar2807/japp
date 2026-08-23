"""Resume import: extraction, the review gate, and what it must never touch."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from engine.resume_import import (
    MAX_UPLOAD_BYTES,
    ImportedEducation,
    ImportedExperience,
    ImportedProfile,
    ResumeImportError,
    ResumeImporter,
    detect_kind,
    extract_text,
    to_profile,
)

RESUME_TEXT = """Ada Lovelace
Austin, TX | +1 (512) 555-0142 | ada@example.com
linkedin.com/in/adalovelace

SUMMARY
Principal backend engineer with eight years in distributed systems.

EXPERIENCE
Principal Backend Engineer, Analytical Engines, Austin TX
March 2021 - Present
- Cut p99 API latency from 840ms to 210ms across 14 Python services.
- Rebuilt CI/CD on Kubernetes, taking deploys from weekly to 30+ per day.

Senior Software Engineer, Difference Systems, Remote
January 2018 - February 2021
- Migrated a 4TB PostgreSQL estate to partitioned tables with zero downtime.

EDUCATION
BSc Mathematics, University of London, 2017

SKILLS
Python, Go, Distributed Systems, Kubernetes, PostgreSQL, Terraform
"""


# ---------------- file handling ----------------


@pytest.mark.parametrize("filename,expected", [
    ("resume.pdf", "pdf"), ("cv.docx", "docx"),
    ("notes.txt", "txt"), ("readme.md", "txt"),
])
def test_supported_types_are_detected(filename, expected):
    assert detect_kind(filename) == expected


@pytest.mark.parametrize("filename", ["photo.png", "archive.zip", "resume.pages", ""])
def test_unsupported_types_are_rejected(filename):
    with pytest.raises(ResumeImportError, match="Unsupported"):
        detect_kind(filename)


def test_content_type_is_used_when_the_extension_is_missing():
    assert detect_kind("resume", "application/pdf") == "pdf"


def test_oversized_uploads_are_rejected():
    with pytest.raises(ResumeImportError, match="limit"):
        extract_text(b"x" * (MAX_UPLOAD_BYTES + 1), "txt")


def test_empty_uploads_are_rejected():
    with pytest.raises(ResumeImportError, match="empty"):
        extract_text(b"", "txt")


def test_a_file_with_no_text_layer_is_rejected_clearly():
    """A scanned PDF must say so, not silently import an empty profile."""
    with pytest.raises(ResumeImportError, match="scan or an image-only"):
        extract_text(b"tiny", "txt")


def test_plain_text_extraction_round_trips():
    text = extract_text(RESUME_TEXT.encode(), "txt")
    assert "Analytical Engines" in text and "840ms" in text


def test_pdf_extraction_reads_a_real_generated_pdf(tmp_path):
    """End to end against a PDF this project actually produced."""
    pytest.importorskip("weasyprint")
    from engine.pdf_generator import PDFGenerator
    from engine.schemas import ExperienceBlock, MasterProfile, TailoredResumeSchema

    profile = MasterProfile.model_validate({
        "contact": {"full_name": "Ada Lovelace", "email": "ada@example.com",
                    "phone": "+1-555-010-0100", "location": {"city": "Austin", "state": "TX"}},
        "skills": {"hard": ["Python"], "tooling": ["Kubernetes"]},
        "experience": [{"company": "Analytical Engines", "title": "Principal Engineer",
                        "start_date": "2021-03", "is_current": True, "bullets": ["x"]}],
        "education": [{"institution": "University of London", "degree": "BSc"}],
    })
    resume = TailoredResumeSchema(
        summary="Principal engineer with deep distributed systems experience.",
        highlighted_skills=["Python", "Kubernetes"],
        tailored_experience=[ExperienceBlock(
            company="Analytical Engines", title="Principal Engineer",
            start_date="2021-03", end_date="Present",
            bullets=["Cut p99 API latency from 840ms to 210ms across 14 services.",
                     "Rebuilt CI/CD on Kubernetes for 30+ deploys a day."])],
        ats_match_percentage=88.0)

    path = PDFGenerator().generate(resume, profile, "Acme", "Engineer",
                                   output_path=tmp_path / "r.pdf")
    text = extract_text(path.read_bytes(), "pdf")
    assert "Analytical Engines" in text
    assert "840ms" in text


def test_docx_extraction_reads_paragraphs_and_tables():
    docx = pytest.importorskip("docx")

    document = docx.Document()
    document.add_paragraph("Ada Lovelace")
    document.add_paragraph("Principal Backend Engineer, Analytical Engines")
    document.add_paragraph("Cut p99 API latency from 840ms to 210ms.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Skills"
    table.rows[0].cells[1].text = "Python, Go, Kubernetes"
    document.add_paragraph("x" * 250)      # clear the minimum-length guard

    buffer = io.BytesIO()
    document.save(buffer)

    text = extract_text(buffer.getvalue(), "docx")
    assert "Analytical Engines" in text
    assert "Python, Go, Kubernetes" in text      # table content survives


# ---------------- structured extraction ----------------


def test_importer_sends_the_resume_and_asks_for_transcription():
    sent = {}

    class Recorder:
        def parse(self, **kwargs):
            sent.update(kwargs)
            return SimpleNamespace(parsed_output=ImportedProfile(full_name="Ada Lovelace"),
                                   stop_reason="end_turn")

    result = ResumeImporter(client=Recorder()).parse(RESUME_TEXT)
    assert result.full_name == "Ada Lovelace"
    assert "Analytical Engines" in sent["messages"][0]["content"]
    assert "transcrib" in sent["system"].lower()
    assert sent["output_format"] is ImportedProfile


def test_importer_uses_the_cheap_tier():
    from config import settings

    assert ResumeImporter().model == settings.LLM_MODEL_BULK


def test_no_output_raises_rather_than_importing_nothing():
    class Empty:
        def parse(self, **kwargs):
            return SimpleNamespace(parsed_output=None, stop_reason="end_turn")

    with pytest.raises(ResumeImportError):
        ResumeImporter(client=Empty()).parse(RESUME_TEXT)


# ---------------- merging ----------------


def test_import_fills_an_empty_profile():
    imported = ImportedProfile(
        full_name="Ada Lovelace", email="ada@example.com", city="Austin",
        skills_tooling=["Go", "Kubernetes"],
        experience=[ImportedExperience(company="Analytical Engines",
                                       title="Principal Engineer",
                                       start_date="2021-03", is_current=True,
                                       bullets=["Cut p99 latency 60%."])],
        education=[ImportedEducation(institution="University of London", degree="BSc")])

    profile = to_profile(imported)
    assert profile["contact"]["full_name"] == "Ada Lovelace"
    assert profile["contact"]["location"]["city"] == "Austin"
    assert profile["skills"]["tooling"] == ["Go", "Kubernetes"]
    assert profile["experience"][0]["company"] == "Analytical Engines"
    assert profile["experience"][0]["end_date"] is None      # current role
    assert profile["education"][0]["institution"] == "University of London"


def test_import_never_overwrites_what_you_already_typed():
    existing = {"contact": {"full_name": "Ada King", "email": "", "phone": "",
                            "location": {"city": "London"}, "links": {}},
                "summary": "My own words.", "skills": {}, "experience": [],
                "education": [], "certifications": []}
    imported = ImportedProfile(full_name="ADA LOVELACE", email="ada@example.com",
                               city="Austin", summary="Generated summary.")

    profile = to_profile(imported, existing)
    assert profile["contact"]["full_name"] == "Ada King"        # kept
    assert profile["contact"]["location"]["city"] == "London"   # kept
    assert profile["summary"] == "My own words."                # kept
    assert profile["contact"]["email"] == "ada@example.com"     # filled the gap


def test_import_never_touches_legal_or_eeo_answers():
    """A resume does not contain these, so any value would be invented."""
    existing = {"legal": {"work_authorization_us": "Yes",
                          "requires_sponsorship_now_or_future": "No"},
                "voluntary_disclosures": {"gender": "Decline to self-identify"},
                "contact": {"location": {}, "links": {}},
                "skills": {}, "experience": [], "education": [], "certifications": []}

    profile = to_profile(ImportedProfile(full_name="Ada"), existing)
    assert profile["legal"]["work_authorization_us"] == "Yes"
    assert profile["legal"]["requires_sponsorship_now_or_future"] == "No"
    assert profile["voluntary_disclosures"]["gender"] == "Decline to self-identify"


def test_imported_profile_schema_has_no_legal_fields():
    """The model is never even given somewhere to put them."""
    fields = set(ImportedProfile.model_fields)
    for forbidden in ("work_authorization_us", "requires_sponsorship_now_or_future",
                      "visa_status", "gender", "race_ethnicity", "veteran_status",
                      "desired_salary", "security_clearance"):
        assert forbidden not in fields
