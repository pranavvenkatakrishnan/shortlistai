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
    if ext not in ("pdf", "docx", "tex", "txt"):
        raise UnsupportedFileType(f"Unsupported file type '.{ext}'. Please upload a PDF, DOCX, TEX, or TXT file.")
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


# ---------------------------------------------------------------- LaTeX

# Commands whose argument is printed as-is (formatting only).
TEX_STYLE = {"textbf", "textit", "emph", "underline", "textsc", "texttt", "textrm", "textsf", "textmd", "textup",
             "mbox", "text", "uline", "small", "footnotesize", "large", "Large", "LARGE", "huge", "Huge", "scshape",
             "bfseries", "itshape", "normalsize", "scriptsize", "tiny", "centering", "makebox", "fbox", "raisebox"}
# Commands dropped together with their arguments (spacing, setup, icons).
TEX_DROP = {"vspace", "hspace", "setlength", "addtolength", "pagestyle", "thispagestyle", "fontsize", "color",
            "titleformat", "titlespacing", "label", "includegraphics", "input", "usepackage", "renewcommand",
            "newcommand", "definecolor", "hypersetup", "pdfgentounicode", "linespread", "setlist", "rule",
            "phantom", "vfill", "faicon", "includesvg", "extracolsep", "arraystretch", "cline", "hline"}
TEX_SYMBOLS = {"&": "&", "%": "%", "$": "$", "#": "#", "_": "_", "{": "{", "}": "}", "textbar": "|",
               "textbullet": "•", "cdot": "·", "bullet": "•", "ldots": "...", "dots": "...", "textendash": "–",
               "textemdash": "—", "quad": " ", "qquad": " ", "enspace": " ", "space": " ", "hfill": "\t",
               "hfil": "\t", "newline": "\n", "linebreak": "\n", "par": "\n", "item": "\n• ", "LaTeX": "LaTeX",
               "TeX": "TeX", "textasciitilde": "~", "slash": "/", "textpipe": "|", "vert": "|", "mid": "|"}


def _tex_group(src: str, i: int, open_c: str = "{", close_c: str = "}") -> Tuple[str, int]:
    """The balanced group starting at src[i] (an opening brace) -> (inside, index after it)."""
    depth = 0
    for j in range(i, len(src)):
        c = src[j]
        if c == "\\":
            continue
        if c == open_c and (j == 0 or src[j - 1] != "\\"):
            depth += 1
        elif c == close_c and src[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return src[i + 1:j], j + 1
    return src[i + 1:], len(src)


def _tex_args(src: str, i: int, limit: int = 9) -> Tuple[List[str], int]:
    """Up to `limit` brace arguments after a command (optional [..] args skipped)."""
    args = []
    while len(args) < limit:
        j = i
        while j < len(src) and src[j] in " \t\n" and src[j:j + 2] != "\n\n":
            j += 1
        if j < len(src) and src[j] == "[" and not args:
            _, j = _tex_group(src, j, "[", "]")
            i = j
            continue
        if j < len(src) and src[j] == "{":
            arg, i = _tex_group(src, j)
            args.append(arg)
        else:
            break
    return args, i


def _tex_macro_arities(src: str) -> dict:
    """User macros from the preamble -> argument count (\\newcommand{\\resumeItem}[1]{...})."""
    return {m.group(1): int(m.group(2) or 0) for m in
            re.finditer(r"\\(?:re)?newcommand\*?\s*\{?\\(\w+)\}?\s*(?:\[(\d)\])?", src)}


def _tex_convert(src: str, arity: dict) -> str:
    out: List[str] = []
    i = 0
    while i < len(src):
        c = src[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+\*?|.?)", src[i:], re.S)
            name = m.group(1).rstrip("*")
            i += len(m.group(0))
            if name in ("", " ", "\n", ",", ";", ":", "!", "/"):  # control spaces and thin spaces
                out.append(" ")
            elif name == "\\":  # line break, maybe with [2pt]
                i = _tex_group(src, i, "[", "]")[1] if src[i:i + 1] == "[" else i
                out.append("\n")
            elif name in TEX_SYMBOLS:
                out.append(TEX_SYMBOLS[name])
            elif name in ("begin", "end"):
                args, i = _tex_args(src, i, 1)
                if src[i:i + 1] == "[":  # \begin{itemize}[leftmargin=...]
                    i = _tex_group(src, i, "[", "]")[1]
                if args and args[0].startswith("tabular"):  # its column spec
                    i = _tex_args(src, i, 2 if args[0] == "tabular*" else 1)[1]
                out.append("\n")
            elif name in ("section", "subsection", "subsubsection", "chapter", "cvsection"):
                args, i = _tex_args(src, i, 1)
                out.append("\n" + _tex_convert(args[0], arity) + "\n" if args else "\n")
            elif name == "href":
                args, i = _tex_args(src, i, 2)
                out.append(_tex_convert(args[-1], arity) if args else "")
            elif name in TEX_DROP:
                i = _tex_args(src, i, arity.get(name, 2) if name not in ("vspace", "hspace") else 1)[1]
            elif name in TEX_STYLE:
                args, i = _tex_args(src, i, 1)
                out.append(_tex_convert(args[0], arity) if args else "")
            elif name.startswith("fa") or name.startswith("ai"):
                i = _tex_args(src, i, 1)[1]  # fontawesome/academicons icons
            else:
                args, i = _tex_args(src, i, arity.get(name, 9))
                parts = [_tex_convert(a, arity).strip() for a in args]
                low = name.lower()
                if "item" in low and parts:
                    out.append("\n• " + ": ".join(p for p in parts if p) + "\n")
                elif len(parts) == 4:  # \resumeSubheading{Company}{Dates}{Role}{Location}
                    out.append(f"\n{parts[0]}\t{parts[1]}\n{parts[2]}\t{parts[3]}\n")
                elif len(parts) == 2 and "heading" in low:
                    out.append(f"\n{parts[0]}\t{parts[1]}\n")
                elif parts:
                    out.append("\n" + " ".join(p for p in parts if p) + "\n" if "heading" in low else " ".join(parts))
                else:
                    out.append(" ")
        elif c == "%":
            nl = src.find("\n", i)
            i = len(src) if nl < 0 else nl + 1
        elif c == "$":
            j = src.find("$", i + 1)
            inner = src[i + 1:j] if j > 0 else ""
            out.append(_tex_convert(inner, arity) if inner.strip() not in ("", "|") else " | " if inner.strip() else "")
            i = j + 1 if j > 0 else len(src)
        elif c == "&":
            out.append("\t")
            i += 1
        elif c in "{}":
            i += 1
        elif c == "~":
            out.append(" ")
            i += 1
        elif c == "\n":  # a single newline is a space; a blank line ends the paragraph
            j = i + 1
            while j < len(src) and src[j] in " \t":
                j += 1
            out.append("\n" if src[j:j + 1] == "\n" else " ")
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def latex_to_text(src: str) -> str:
    """Plain text of a LaTeX resume body, laid out like a PDF extraction:
    one line per printed line, "• " bullets, right-aligned parts after a tab."""
    m = re.search(r"\\begin\{document\}(.*?)(\\end\{document\}|$)", src, re.S)
    body = m.group(1) if m else src
    text = _tex_convert(body, _tex_macro_arities(src))
    text = text.replace("---", "—").replace("--", "–").replace("``", "\u201c").replace("''", "\u201d")
    lines = []
    for line in text.split("\n"):
        cells = [re.sub(r"\s+", " ", c).strip() for c in line.split("\t")]
        cells = [c for c in cells if c]
        if cells:
            lines.append("\t".join(cells))
    return "\n".join(l for l in lines if l.strip("•| "))


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
    elif fmt == "tex":
        text = latex_to_text(content.decode("utf-8", errors="ignore"))
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
