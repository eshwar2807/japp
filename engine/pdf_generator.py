"""Jinja2 -> HTML -> PDF resume compiler.

Optimised for machine parsing rather than visual flourish: single column, real
text, no images or tables, standard section headings. The generated PDF is also
verified after writing (non-empty, %PDF header, extractable text) so a silently
broken file never reaches an application form.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from config import settings
from engine.schemas import MasterProfile, TailoredResumeSchema

log = logging.getLogger(__name__)

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def slugify(value: str, max_length: int = 40) -> str:
    """Filesystem-safe token: 'Acme Corp. (US)' -> 'Acme_Corp_US'."""
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    cleaned = re.sub(r"_+", "_", _SAFE_CHARS.sub("_", normalized)).strip("_-")
    return cleaned[:max_length].strip("_-") or "unknown"


def build_filename(full_name: str, company: str, role: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d")
    return f"{slugify(full_name, 30)}_{slugify(company, 30)}_{slugify(role, 40)}_{stamp}.pdf"


class PDFGenerator:
    def __init__(self, template_dir: Path | None = None, template_name: str = "resume_template.html"):
        self.template_dir = Path(template_dir or settings.TEMPLATE_DIR)
        self.template_name = template_name
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True,          # resume content is model output; always escape
            undefined=StrictUndefined,  # fail loudly rather than render a blank field
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ---------------- context ----------------

    def build_context(
        self, resume: TailoredResumeSchema, profile: MasterProfile
    ) -> dict:
        """Merge tailored content with immutable profile facts.

        Contact details, education and certifications come from the master
        profile only - the LLM never gets to restate them.
        """
        # Group the highlighted skills back into labelled rows, preserving the
        # LLM's relevance ordering within each group.
        hard = {s.lower() for s in profile.skills.hard}
        tools = {s.lower() for s in profile.skills.tooling}
        soft = {s.lower() for s in profile.skills.soft}

        g_hard, g_tools, g_soft, g_other = [], [], [], []
        for skill in resume.highlighted_skills:
            key = skill.lower()
            if key in hard:
                g_hard.append(skill)
            elif key in tools:
                g_tools.append(skill)
            elif key in soft:
                g_soft.append(skill)
            else:
                g_other.append(skill)

        skill_groups = [
            ("Core", g_hard),
            ("Technologies", g_tools),
            ("Professional", g_soft),
        ]
        if g_other:
            skill_groups.append(("Additional", g_other))

        links = [
            (label.title(), url)
            for label, url in (profile.contact.links or {}).items()
            if url and not url.strip().startswith("<")
        ]

        return {
            "resume": resume,
            "contact": profile.contact,
            "links": links,
            "skill_groups": [(lbl, items) for lbl, items in skill_groups if items],
            "education": profile.education,
            "certifications": [
                c for c in profile.certifications if c.name and not c.name.startswith("<")
            ],
            "projects": [
                p for p in profile.projects
                if p.get("name") and not str(p["name"]).startswith("<")
            ],
        }

    # ---------------- render ----------------

    def render_html(self, resume: TailoredResumeSchema, profile: MasterProfile) -> str:
        template = self.env.get_template(self.template_name)
        return template.render(**self.build_context(resume, profile))

    def generate(
        self,
        resume: TailoredResumeSchema,
        profile: MasterProfile,
        company: str,
        role_title: str,
        output_path: Path | None = None,
        keep_html: bool = False,
    ) -> Path:
        """Render and write the PDF. Returns the path to the verified file."""
        from weasyprint import HTML  # imported lazily: pulls in native libs

        html = self.render_html(resume, profile)
        path = Path(
            output_path
            or settings.RESUME_DIR / build_filename(profile.contact.full_name, company, role_title)
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        HTML(string=html, base_url=str(self.template_dir)).write_pdf(str(path))

        if keep_html:
            path.with_suffix(".html").write_text(html, encoding="utf-8")

        self.verify(path, resume)
        log.info("Resume written: %s (%.1f KB)", path, path.stat().st_size / 1024)
        return path

    # ---------------- verification ----------------

    @staticmethod
    def verify(path: Path, resume: TailoredResumeSchema | None = None) -> dict:
        """Sanity-check the emitted PDF; raise if it is unusable.

        Catches the two failure modes that silently ruin an application: a
        zero-byte/corrupt file, and a PDF whose text cannot be extracted (which
        is exactly what an ATS would see).
        """
        if not path.exists():
            raise FileNotFoundError(f"PDF was not written: {path}")
        size = path.stat().st_size
        if size < 1024:
            raise ValueError(f"PDF is suspiciously small ({size} bytes): {path}")
        with path.open("rb") as fh:
            if fh.read(5) != b"%PDF-":
                raise ValueError(f"Not a valid PDF (bad header): {path}")

        report = {"path": str(path), "size_bytes": size, "text_extracted": None, "pages": None}

        try:  # optional: only if a text extractor is installed
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(path))
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
            report["pages"] = len(reader.pages)
            report["text_extracted"] = len(text)
            if len(text.strip()) < 100:
                raise ValueError(
                    f"PDF contains almost no extractable text ({len(text)} chars). "
                    "An ATS would read it as blank."
                )
            if resume is not None and resume.tailored_experience:
                first = resume.tailored_experience[0].company
                if first and first.lower() not in text.lower():
                    log.warning("Employer %r not found in extracted PDF text.", first)
        except ImportError:
            log.debug("pypdf not installed; skipping text-extraction check.")

        return report
