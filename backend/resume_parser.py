import io
import math
import re
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


def extract_text_and_pages(filename: str, content: bytes) -> Tuple[str, str, int]:
    """Returns (text, format, page_count). `text` is one resume line per line,
    in the same order as the source document's paragraphs/pages, so line
    positions line up with the original file for later editing."""
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("File is too large (max 8 MB).")

    fmt = detect_format(filename)

    if fmt == "pdf":
        text = "\n".join(extract_pdf_rows(content))
        with fitz.open(stream=content, filetype="pdf") as doc:
            page_count = len(doc)
    elif fmt == "docx":
        doc = Document(io.BytesIO(content))
        text = "\n".join(p.text for p in doc.paragraphs)
        page_count = estimate_pages_from_words(text)
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
