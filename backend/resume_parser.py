import io
import math
import re
import zipfile
from typing import List, Tuple

import pymupdf as fitz
from docx import Document

MAX_FILE_BYTES = 8 * 1024 * 1024  # 8 MB

# Rough estimate for formats where page count isn't stored in the file
# (DOCX/TXT render page count depends on the viewer's fonts/margins).
WORDS_PER_PAGE = 550


class UnsupportedFileType(ValueError):
    pass


def detect_format(filename: str) -> str:
    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()
    if ext not in ("pdf", "docx", "txt"):
        raise UnsupportedFileType(f"Unsupported file type '.{ext}'. Please upload a PDF, DOCX, or TXT file.")
    return ext


def estimate_pages_from_words(text: str) -> int:
    words = len(text.split())
    return max(1, math.ceil(words / WORDS_PER_PAGE))


BULLET_CHARS = "•‣▪●○◦■□➢►–-*"
# A horizontal gap this wide between two pieces of text on the same row means
# the right piece is a separately aligned column (dates, locations).
COLUMN_GAP = 36
# Icon fonts (phone/mail/LinkedIn glyphs) extract as junk letters like "Ó" or "W".
ICON_FONT_RE = re.compile(r"awesome|icon|symbol|dingbat|wingding|marvosym|academicons|fontello", re.I)
# Icon-font glyphs (private-use area), emoji and dingbats like 📞 ✉ 🔗 ▸ ✔.
ICON_CHAR_RE = re.compile("[-\U0001f000-\U0001faff☀-➿■-◿⬀-⯿]️?")
ZERO_WIDTH_RE = re.compile("[​-‍⁠﻿]")


def clean_icons(text: str) -> str:
    """An icon leading a line is a bullet; an icon mid-line separates items
    (e.g. "📞 phone ✉ email" -> "phone | email")."""
    text = ZERO_WIDTH_RE.sub("", text).strip()
    if ICON_CHAR_RE.match(text):
        text = "• " + text[ICON_CHAR_RE.match(text).end():].strip()
    if ICON_CHAR_RE.search(text):
        text = ICON_CHAR_RE.sub(" | ", text).strip(" |")
    return re.sub(r"\s{2,}", " ", text)


def extract_pdf_rows(content: bytes) -> List[str]:
    """Text in *visual* order, one string per printed row. PyMuPDF's block
    order isn't reading order (LaTeX puts right-aligned dates in their own
    blocks), so text is regrouped by vertical position. A right-aligned column
    on the same row is kept after a tab, e.g. "Innoknowvex\\tJan 2026 – Mar 2026",
    and wrapped bullet continuation lines are joined back onto their bullet."""
    rows_out: List[str] = []
    with fitz.open(stream=content, filetype="pdf") as doc:
        for page in doc:
            pieces = []
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = [s for s in line.get("spans", []) if not ICON_FONT_RE.search(s.get("font", ""))]
                    text = clean_icons("".join(s["text"] for s in spans))
                    if text:
                        x0, y0, x1, y1 = line["bbox"]
                        pieces.append({"text": text, "x0": x0, "x1": x1, "y0": y0, "y1": y1})

            pieces.sort(key=lambda p: (p["y0"] + p["y1"]) / 2)
            rows: List[List[dict]] = []
            for p in pieces:
                mid = (p["y0"] + p["y1"]) / 2
                if rows:
                    last = rows[-1]
                    top = min(q["y0"] for q in last)
                    bottom = max(q["y1"] for q in last)
                    if top - 1 <= mid <= bottom + 1:
                        last.append(p)
                        continue
                rows.append([p])

            bullet_x = None  # x of the bullet glyph on the previous row, if it was a bullet
            for row in rows:
                row.sort(key=lambda p: p["x0"])
                left, right = [row[0]], []
                for a, b in zip(row, row[1:]):
                    (right if right or b["x0"] - a["x1"] > COLUMN_GAP else left).append(b)
                left_text = " ".join(p["text"] for p in left)
                text = left_text + ("\t" + " ".join(p["text"] for p in right) if right else "")

                # A wrapped bullet line is indented past its bullet glyph.
                if bullet_x is not None and not right and row[0]["x0"] > bullet_x + 2:
                    rows_out[-1] = join_wrapped(rows_out[-1], text)
                    continue
                rows_out.append(text)
                is_bullet = left_text[:1] in BULLET_CHARS and (len(left) > 1 or left_text[1:2] == " ")
                bullet_x = row[0]["x0"] if is_bullet else None
    return rows_out


def join_wrapped(prev: str, nxt: str) -> str:
    """Joins a line wrapped across rows, undoing end-of-line hyphenation
    ("Active Direc-" + "tory" -> "Active Directory")."""
    if prev.endswith("-") and nxt[:1].islower():
        return prev[:-1] + nxt
    if prev.endswith("-"):
        return prev + nxt
    return prev + " " + nxt


def extract_pdf_links(content: bytes) -> dict:
    """Maps visible link text (e.g. "LinkedIn") to its URL, so a regenerated
    resume keeps the original's clickable links."""
    links = {}
    with fitz.open(stream=content, filetype="pdf") as doc:
        for page in doc:
            for link in page.get_links():
                uri = link.get("uri")
                if not uri:
                    continue
                text = page.get_textbox(link["from"]).strip().strip("|").strip()
                if text and len(text) <= 60:
                    links[text] = uri
    return links


def _docx_paragraph_line(p) -> str:
    """Word bullets are list formatting, not a "•" character, so list
    paragraphs get one; tab-aligned dates ("Company<TAB>Jan 2025") keep a
    single tab as the left/right column split."""
    text = re.sub(r"\t+", "\t", p.text).strip()
    if not text:
        return ""
    ppr = p._p.pPr
    is_list = (ppr is not None and ppr.numPr is not None) or "list" in (p.style.name or "").lower()
    if is_list and text[:1] not in BULLET_CHARS:
        text = "• " + text
    return text


def extract_docx_lines(content: bytes) -> List[str]:
    """Paragraphs and tables in document order. A table row laid out as
    columns (e.g. company/role on the left, dates/location on the right)
    becomes "left<TAB>right" lines, matching what PDFs produce."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(io.BytesIO(content))
    lines: List[str] = []
    for el in doc.element.body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            lines.append(_docx_paragraph_line(Paragraph(el, doc)))
        elif tag == "tbl":
            for row in Table(el, doc).rows:
                cells, seen = [], set()
                for cell in row.cells:  # merged cells repeat; keep each once
                    if id(cell._tc) not in seen:
                        seen.add(id(cell._tc))
                        cells.append([l for l in (_docx_paragraph_line(p) for p in cell.paragraphs) if l])
                cells = [c for c in cells if c]
                if len(cells) == 1:
                    lines += cells[0]
                    continue
                for i in range(max((len(c) for c in cells), default=0)):
                    left = cells[0][i] if i < len(cells[0]) else ""
                    right = " ".join(c[i] for c in cells[1:] if i < len(c))
                    lines.append(f"{left}\t{right}" if left and right else left or right)
    return [l for l in lines if l]


def extract_docx_links(content: bytes) -> dict:
    """Visible hyperlink text -> URL, like extract_pdf_links."""
    doc = Document(io.BytesIO(content))
    rels = {r.rId: r.target_ref for r in doc.part.rels.values() if r.reltype.endswith("/hyperlink")}
    links = {}
    for h in doc.element.body.iter():
        if h.tag.endswith("}hyperlink"):
            rid = h.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            text = "".join(t.text or "" for t in h.iter() if t.tag.endswith("}t")).strip()
            if rid in rels and text and len(text) <= 60:
                links[text] = rels[rid]
    return links


def _docx_page_count(content: bytes) -> int:
    """Word stores the page count it last laid out in docProps/app.xml."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            m = re.search(rb"<Pages>(\d+)</Pages>", z.read("docProps/app.xml"))
            return int(m.group(1)) if m else 0
    except (KeyError, zipfile.BadZipFile):
        return 0


def extract_text_and_pages(filename: str, content: bytes) -> Tuple[str, str, int]:
    """Returns (text, format, page_count). `text` is one resume line per
    visual line, in reading order, with right-aligned columns after a tab."""
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("File is too large (max 8 MB).")

    fmt = detect_format(filename)

    if fmt == "pdf":
        text = "\n".join(extract_pdf_rows(content))
        with fitz.open(stream=content, filetype="pdf") as doc:
            page_count = len(doc)
    elif fmt == "docx":
        text = "\n".join(extract_docx_lines(content))
        page_count = _docx_page_count(content) or estimate_pages_from_words(text)
    else:
        text = content.decode("utf-8", errors="ignore")
        page_count = estimate_pages_from_words(text)

    text = text.strip()
    if not text:
        raise ValueError(
            "Couldn't find any readable text in that file. If it's a scanned/image PDF, "
            "try exporting a text-based version instead."
        )
    return text, fmt, max(page_count, 1)
