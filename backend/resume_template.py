"""Regenerates any resume into one clean, consistently aligned template.

Instead of patching the original file (which can't reflow, so edits land in
the wrong place), the resume text is parsed into sections/entries/bullets and
re-typeset from scratch, shrinking the type until it fits the page limit.
"""
import io
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from xml.sax.saxutils import escape

import pymupdf as fitz
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.platypus.flowables import HRFlowable

from resume_parser import BULLET_CHARS, clean_icons, join_wrapped

# Normalised (letters only, lowercase) section titles we recognise.
SECTION_TITLES = {
    "summary", "professionalsummary", "profile", "profilesummary", "objective", "careerobjective", "aboutme",
    "education", "academics", "academicdetails", "educationalqualification", "educationalqualifications",
    "skills", "technicalskills", "keyskills", "coreskills", "skillsandtools", "toolsandtechnologies",
    "skillsandinterests", "technologies", "corecompetencies",
    "experience", "workexperience", "professionalexperience", "employment", "employmenthistory",
    "internships", "internship", "internshipexperience", "relevantexperience",
    "projects", "academicprojects", "personalprojects", "keyprojects", "technicalprojects",
    "certifications", "certification", "certificates", "certificationsandcourses", "licensesandcertifications",
    "achievements", "awards", "awardsandachievements", "honors", "honorsandawards", "accomplishments",
    "publications", "research", "researchexperience", "activities", "extracurricularactivities",
    "extracurriculars", "leadership", "leadershipexperience", "positionsofresponsibility",
    "volunteer", "volunteering", "volunteerexperience", "courses", "coursework", "relevantcoursework",
    "languages", "interests", "hobbies", "workshops", "trainings", "training", "strengths",
    "references", "declaration", "featuredproject", "featuredprojects", "keyachievements",
    "areasofexpertise", "technicalexpertise", "softskills", "personaldetails", "personalinformation",
}
# Sections where a new bullet can sensibly be placed under an entry.
ENTRY_SECTION_HINTS = ("experience", "internship", "employment", "project", "research", "leadership", "volunteer")

BULLET_RE = re.compile(r"^\s*(?:[" + re.escape(BULLET_CHARS) + r"]|\d+[.)])\s+")
CONTACT_RE = re.compile(r"@|\+?\d[\d ()-]{7,}\d|https?://|www\.|linkedin|github|leetcode|portfolio|\|", re.I)
LABEL_RE =re.compile(r"^([A-Za-z][\w /&+.'’()-]{0,30}?)\s*:\s+(.+)$")


@dataclass
class Item:
    kind: str  # "header" | "sub" | "bullet" | "text"
    left: str
    right: str = ""


@dataclass
class Section:
    title: str
    items: List[Item] = field(default_factory=list)


@dataclass
class Resume:
    name: str
    contact: List[str]
    sections: List[Section]
    headline: List[str] = field(default_factory=list)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


# Words that make up compound headings like "Work Experience & Internships".
HEADING_WORDS = {
    "experience", "experiences", "internship", "internships", "education", "project", "projects", "skills",
    "skill", "certifications", "certification", "certificates", "achievements", "awards", "honors",
    "summary", "profile", "objective", "publications", "activities", "languages", "interests", "courses",
    "coursework", "leadership", "volunteering", "responsibility", "responsibilities", "training", "trainings",
}
HEADING_FILLER = {"and", "of", "work", "professional", "technical", "key", "relevant", "academic", "personal",
                  "other", "additional", "core", "selected", "positions", "extra", "curricular", "co", "featured"}


def _is_section_title(line: str) -> bool:
    if "\t" in line or len(line) > 45:
        return False
    if _norm(line) in SECTION_TITLES:
        return True
    words = re.findall(r"[a-z]+", line.lower())
    return bool(words) and any(w in HEADING_WORDS for w in words) and all(
        w in HEADING_WORDS or w in HEADING_FILLER for w in words)


def _is_skills(title: str) -> bool:
    t = _norm(title)
    return any(k in t for k in ("skill", "technolog", "competenc", "expertise"))


def _split_tab(line: str) -> Tuple[str, str]:
    left, _, right = line.partition("\t")
    return left.strip(), right.strip()


def parse_resume(text: str) -> Resume:
    raw_lines = [l for l in text.replace("\r\n", "\n").split("\n") if l.strip()]

    # Re-join wrapped continuation lines (pasted/DOCX text) onto the line
    # before: a line that's indented under a bullet, starts in lowercase,
    # follows a hyphenated break, or continues a sentence left dangling.
    joined: List[str] = []
    for raw in raw_lines:
        s = "\t".join(clean_icons(p) for p in raw.strip().split("\t")).strip("\t")
        if not s:
            continue
        if len(joined) > 1 and "\t" not in s and not BULLET_RE.match(s) and not _is_section_title(s):
            prev = joined[-1]
            indented_under_bullet = raw[:1] in "  " and BULLET_RE.match(prev)
            dangling = re.search(r"(,|\b(and|or|with|of|to|for|the|a|an|in|on|by|using))$", prev)
            hyphen_break = prev.endswith("-") and not prev.endswith(" -")
            if "\t" not in prev and len(prev) > 40 and (
                    indented_under_bullet or dangling or hyphen_break or s[:1].islower()):
                joined[-1] = join_wrapped(prev, s)
                continue
        joined.append(s)

    name = _split_tab(joined[0])[0] if joined else ""
    rest = joined[1:]
    # The block under the name: contact lines, plus a short headline such as
    # "Backend Developer · Java, Spring Boot". Both are centred under the name.
    contact: List[str] = []
    headline: List[str] = []
    while rest and not _is_section_title(rest[0]) and len(rest[0]) < 200:
        l = rest[0]
        if CONTACT_RE.search(l):
            contact += [p.strip(" |") for p in re.split(r"\s*[|\t•·]\s*", l) if p.strip(" |")]
        elif len(l.split()) <= 12 and not BULLET_RE.match(l) and "\t" not in l:
            headline.append(l)
        else:
            break
        rest.pop(0)

    # Anything before the first recognised heading (e.g. an untitled summary)
    # goes into an untitled section rather than being lost or mislabelled.
    sections: List[Section] = []
    body = rest
    for i, l in enumerate(body):
        if _is_section_title(l):
            sections.append(Section(title=l.strip().rstrip(":").strip()))
            continue
        if not sections:
            sections.append(Section(title=""))
        sec = sections[-1]
        prev = sec.items[-1] if sec.items else None
        nxt = body[i + 1] if i + 1 < len(body) else ""

        if BULLET_RE.match(l):
            sec.items.append(Item("bullet", BULLET_RE.sub("", l).strip()))
            continue
        left, right = _split_tab(l)
        if _is_skills(sec.title):
            # Skill grids/tables come through as columns; keep them as one
            # "Label: values" line instead of mistaking them for job headers.
            if right:
                left = f"{left} {right}" if ":" in left else f"{left}: {right}"
            sec.items.append(Item("text", left))
        # The short line right under an entry header is its subtitle (role, degree).
        elif prev is not None and prev.kind == "header" and len(left.split()) <= 14 and (right or not LABEL_RE.match(left)):
            sec.items.append(Item("sub", left, right))
        elif right:
            sec.items.append(Item("header", left, right))
        # A line with bullets under it is an entry title, even with a colon
        # in it ("QIS: Quality Inspection System | FastAPI").
        elif BULLET_RE.match(nxt):
            sec.items.append(Item("header", left))
        elif LABEL_RE.match(left):
            sec.items.append(Item("text", left))
        elif prev is not None and prev.kind == "text" and len(prev.left) > 60 and prev.left[-1:] not in ".!?:":
            prev.left = join_wrapped(prev.left, left)  # a wrapped paragraph line
        else:
            sec.items.append(Item("text", left))

    # Drop exact duplicate lines within a section (a common copy/paste slip).
    for sec in sections:
        seen, kept = set(), []
        for it in sec.items:
            key = (it.kind, it.left.lower(), it.right.lower())
            if it.kind in ("text", "bullet") and key in seen:
                continue
            seen.add(key)
            kept.append(it)
        sec.items = kept

    return Resume(name=name, contact=contact, sections=sections, headline=headline)


def _match_bullet_style(resume: Resume, bullets: List[str]) -> List[str]:
    """New bullets follow the resume's own convention: capitalised, and
    ending with a full stop only if most existing bullets do."""
    existing = [it.left for s in resume.sections for it in s.items if it.kind == "bullet"]
    with_period = sum(b.rstrip().endswith(".") for b in existing)
    use_period = with_period * 2 >= len(existing) if existing else True
    styled = []
    for b in bullets:
        b = b.strip().rstrip(".").strip()
        b = b[:1].upper() + b[1:]
        styled.append(b + "." if use_period else b)
    return styled


def place_bullets(resume: Resume, bullets: List[str]) -> None:
    """Puts each new bullet under the experience/project entry it is most
    similar to, instead of tacking it onto the end of the document."""
    if not bullets:
        return
    bullets = _match_bullet_style(resume, bullets)
    entries = []  # (section, index to insert after, entry text)
    for sec in resume.sections:
        if not any(h in _norm(sec.title) for h in ENTRY_SECTION_HINTS):
            continue
        for start, it in enumerate(sec.items):
            if it.kind != "header":
                continue
            # An entry is its header, subtitle and bullets; a plain text line
            # (e.g. "Other internships: ...") ends it, so new bullets never
            # land under a note line.
            end = start
            while end + 1 < len(sec.items) and sec.items[end + 1].kind in ("sub", "bullet"):
                end += 1
            text = " ".join(x.left for x in sec.items[start:end + 1])
            entries.append((sec, end, text))
    if not entries:
        sec = Section("Highlights")
        resume.sections.append(sec)
        sec.items.extend(Item("bullet", b) for b in bullets)
        return

    from matcher import embed

    sims = embed(bullets) @ embed([e[2] for e in entries]).T
    targets = [(int(row.argmax()), b) for row, b in zip(sims, bullets)]
    # Insert from the bottom entry up so earlier insertion indexes stay valid.
    for entry_idx in sorted({t[0] for t in targets}, reverse=True):
        sec, last_idx, _ = entries[entry_idx]
        new_items = [Item("bullet", b) for i, b in targets if i == entry_idx]
        sec.items[last_idx + 1:last_idx + 1] = new_items


PLACEHOLDER_RE = re.compile(r"\[[^\[\]]{2,}\](?!\()")  # "[Month Year]", not "[link](url)"


def strip_placeholders(resume: Resume) -> List[str]:
    """Leaves unfilled template placeholders out of the final resume instead
    of printing "[Month Year]" or "[What you built…]". Returns a short
    description of what was left out, so the user knows what to fill in."""
    missing: List[str] = []

    def clean(s: str) -> str:
        s = PLACEHOLDER_RE.sub("", s)
        s = re.sub(r"^[\s|,–—-]+|[\s|,–—-]+$", "", s)  # separators left dangling
        return re.sub(r"\s{2,}", " ", s)

    resume.contact = [c for c in resume.contact if not PLACEHOLDER_RE.fullmatch(c.strip())]
    for sec in resume.sections:
        kept: List[Item] = []
        entry = sec.title
        gaps: List[str] = []
        for it in sec.items:
            if it.kind == "header":
                if gaps:
                    missing.append(f"{entry}: {', '.join(gaps)}")
                entry, gaps = it.left, []
            has_ph = PLACEHOLDER_RE.search(it.left) or PLACEHOLDER_RE.search(it.right)
            if not has_ph:
                kept.append(it)
                continue
            if PLACEHOLDER_RE.search(it.right):
                it.right = clean(it.right)
                gaps.append("dates")
            if PLACEHOLDER_RE.search(it.left):
                it.left = clean(it.left)
                if it.kind in ("bullet", "text") and len(it.left.split()) < 3:
                    gaps.append("a bullet")
                    continue
            kept.append(it)
        if gaps:
            missing.append(f"{entry}: {', '.join(gaps)}")
        sec.items = kept
    # "a bullet, a bullet" -> "2 bullets"
    return [re.sub(r"(a bullet(, )?){2,}", lambda m: f"{m.group(0).count('a bullet')} bullets", m).rstrip(", ")
            for m in missing]


# ---------------------------------------------------------------- rendering

_FONTS = {"regular": "Times-Roman", "bold": "Times-Bold", "italic": "Times-Italic", "bolditalic": "Times-BoldItalic"}


def _register_fonts() -> None:
    """Prefers a TrueType serif (full Unicode) over the built-in Times, which
    can't draw characters outside Latin-1."""
    candidates = [
        ("C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/timesbd.ttf", "C:/Windows/Fonts/timesi.ttf", "C:/Windows/Fonts/timesbi.ttf"),
        tuple(f"/usr/share/fonts/truetype/liberation/LiberationSerif-{s}.ttf" for s in ("Regular", "Bold", "Italic", "BoldItalic")),
    ]
    for paths in candidates:
        if all(os.path.exists(p) for p in paths):
            names = ("ResumeSerif", "ResumeSerif-Bold", "ResumeSerif-Italic", "ResumeSerif-BoldItalic")
            for n, p in zip(names, paths):
                pdfmetrics.registerFont(TTFont(n, p))
            pdfmetrics.registerFontFamily(names[0], normal=names[0], bold=names[1], italic=names[2], boldItalic=names[3])
            _FONTS.update(zip(("regular", "bold", "italic", "bolditalic"), names))
            return


_register_fonts()


def _inline(text: str) -> str:
    return escape(text)


def _label_markup(text: str) -> str:
    m = LABEL_RE.match(text)
    if m:
        return f"<b>{escape(m.group(1))}:</b> {escape(m.group(2))}"
    return escape(text)


def _header_markup(text: str) -> str:
    # "Project Name | Python, React" -> bold name, italic stack
    name, sep, rest = text.partition(" | ")
    if sep:
        return f"<b>{escape(name)}</b> | <i>{escape(rest)}</i>"
    return f"<b>{escape(text)}</b>"


def _build(resume: Resume, links: dict, size: float, page_size=LETTER) -> bytes:
    lead = size * 1.16
    gap = size * 0.3
    F = _FONTS
    name_st = ParagraphStyle("name", fontName=F["bold"], fontSize=size * 2.25, leading=size * 2.6, alignment=1)
    contact_st = ParagraphStyle("contact", fontName=F["regular"], fontSize=size * 1.05, leading=size * 1.3, alignment=1)
    sec_st = ParagraphStyle("sec", fontName=F["regular"], fontSize=size * 1.3, leading=size * 1.5, spaceBefore=gap)
    body = ParagraphStyle("body", fontName=F["regular"], fontSize=size, leading=lead)
    head_l = ParagraphStyle("hl", parent=body, fontSize=size * 1.1, leading=lead * 1.1)
    head_r = ParagraphStyle("hr", parent=head_l, alignment=2)
    sub_l = ParagraphStyle("sl", parent=body, fontName=F["italic"])
    sub_r = ParagraphStyle("sr", parent=sub_l, alignment=2)

    width = page_size[0] - 1.0 * inch - 12  # minus the frame's 6pt side paddings

    def row(left_markup, right_text, ls, rs):
        if not right_text:
            return Paragraph(left_markup, ls)
        text_w = pdfmetrics.stringWidth(right_text, rs.fontName, rs.fontSize) + 4
        right_w = min(0.45, max(0.15, text_w / width))
        t = Table([[Paragraph(left_markup, ls), Paragraph(escape(right_text), rs)]],
                  colWidths=[width * (1 - right_w), width * right_w])
        t.setStyle(TableStyle([(k, (0, 0), (-1, -1), 0) for k in
                               ("LEFTPADDING", "RIGHTPADDING", "TOPPADDING", "BOTTOMPADDING")] +
                              [("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return t

    def bullet_list(texts):
        return ListFlowable(
            [ListItem(Paragraph(_label_markup(t), body), leftIndent=size * 1.2) for t in texts],
            bulletType="bullet", start="\u2022", bulletFontSize=size * 0.9, bulletOffsetY=-size * 0.1,
            leftIndent=size * 1.2, spaceAfter=gap * 0.5,
        )

    contact_parts = []
    for c in resume.contact:
        url = links.get(c) or (f"mailto:{c}" if re.fullmatch(r"[^@\s]+@[^@\s]+\.\w+", c) else None)
        contact_parts.append(f'<a href="{escape(url)}">{escape(c)}</a>' if url else escape(c))

    story = [Paragraph(escape(resume.name), name_st)]
    for h in resume.headline:
        story.append(Paragraph(escape(h), contact_st))
    if contact_parts:
        story += [Spacer(1, 2), Paragraph(" | ".join(contact_parts), contact_st)]

    for sec in resume.sections:
        title = sec.title
        if title:
            story.append(Paragraph(
                f'{escape(title[:1].upper())}<font size="{size * 1.05:.1f}">{escape(title[1:].upper())}</font>', sec_st))
            story.append(HRFlowable(width="100%", thickness=0.6, color="black", spaceBefore=1, spaceAfter=gap))
        else:
            story.append(Spacer(1, gap * 2))
        pending: List[str] = []
        for it in sec.items:
            if it.kind == "bullet":
                pending.append(it.left)
                continue
            if pending:
                story.append(bullet_list(pending))
                pending = []
            if it.kind == "header":
                story.append(row(_header_markup(it.left), it.right, head_l, head_r))
            elif it.kind == "sub":
                story.append(row(_inline(it.left), it.right, sub_l, sub_r))
            else:
                story.append(Paragraph(_label_markup(it.left), body))
        if pending:
            story.append(bullet_list(pending))

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=page_size, leftMargin=0.5 * inch, rightMargin=0.5 * inch,
                      topMargin=0.4 * inch, bottomMargin=0.4 * inch,
                      title=f"{resume.name} - Resume", author=resume.name).build(story)
    return buf.getvalue()


def render_pdf(resume: Resume, links: Optional[dict], page_limit: int) -> Tuple[bytes, int, bool]:
    """Renders at the largest readable size that fits `page_limit` pages.
    Returns (pdf bytes, page count, fits). Content is never dropped; if it
    can't fit even at the smallest size, it's returned longer and fits=False."""
    links = links or {}
    out, pages = b"", 0
    for size in (10.5, 10.2, 10.0, 9.7, 9.4, 9.1, 8.8):
        out = _build(resume, links, size)
        with fitz.open(stream=out, filetype="pdf") as d:
            pages = len(d)
        if pages <= page_limit:
            return out, pages, True
    return out, pages, False
