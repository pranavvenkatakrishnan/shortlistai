"""Applies keyword edits to the original resume file, in its own format.

Each edit is a find/replace op from matcher.edit_ops: {line (the text it
sits in, for context), old, new, append}. Only the changed words are touched,
so fonts, alignment, spacing and page count stay as they were:

- PDF: the changed word and the rest of its line are re-drawn in place, in the
  original embedded font where it has the glyphs; the line's word spacing
  absorbs any change in length, so justified lines stay flush and nothing
  runs into a right-hand column.
- DOCX: the text of the existing runs is changed, keeping their formatting.
- LaTeX: the source is changed, with the new words escaped.

New bullet sentences (insert ops) are added after the bullet they relate to,
copying its style, in DOCX, LaTeX and text. A PDF can't take a new line
without moving everything below it, so there they are left for the user to copy.
"""
import copy
import io
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import pymupdf as fitz
from docx import Document
from docx.oxml.ns import qn

from matcher import find_word
from resume_parser import _tex_group

MIN_GAP_SHARE = 0.7  # word gaps may shrink to this share of their width
MIN_FONT_SCALE = 0.9  # after that, the redrawn words may shrink this much
MIN_CONTEXT = 0.5  # share of a line's words that must match the edit's context


def _words(text: str) -> set:
    return set(re.findall(r"[a-z0-9+#]+", text.lower()))


def _context_score(candidate: str, context: str) -> float:
    cand, ctx = _words(candidate), _words(context)
    return len(cand & ctx) / len(cand) if cand else 0.0


def _locate(text: str, op: dict) -> Optional[Tuple[int, int]]:
    """(start, end) of op["old"] in `text`: the last occurrence for an
    append (the end of a Skills line), else the first whole-word match."""
    if op["append"]:
        pos = text.rfind(op["old"])
        return (pos, pos + len(op["old"])) if pos >= 0 else None
    m = find_word(text, op["old"]) or find_word(text, op["old"], re.I)
    return (m.start(), m.end()) if m else None


# ---------------------------------------------------------------- PDF

BASE14 = {
    "helvetica": "helv", "helvetica-bold": "hebo", "helvetica-oblique": "heit", "helvetica-boldoblique": "hebi",
    "arial": "helv", "arial-bold": "hebo", "arial,bold": "hebo", "arialmt": "helv", "arial-boldmt": "hebo",
    "times-roman": "tiro", "times-bold": "tibo", "times-italic": "tiit", "times-bolditalic": "tibi",
    "timesnewroman": "tiro", "timesnewromanpsmt": "tiro", "timesnewromanps-boldmt": "tibo",
    "courier": "cour", "courier-bold": "cobo", "courier-oblique": "coit", "courier-boldoblique": "cobi",
}
SERIF_RE = re.compile(r"times|serif(?!.*sans)|roman|georgia|garamond|cambria|book|cmr|cmbx|cmti|lmroman|"
                      r"palatino|baskerville|charter|minion|caslon|libertine|tinos|merriweather", re.I)
MONO_RE = re.compile(r"mono|courier|consol|cmtt|code|inconsolata", re.I)


SANS_RE = re.compile(r"sans|arial|helvet|calibri|carlito|arimo|verdana|tahoma|segoe|roboto|lato|open|"
                     r"source ?sans|noto ?sans|inter|montserrat|poppins|trebuchet|century gothic|gill", re.I)


def _weight(name: str, flags: int) -> Tuple[bool, bool]:
    bold = bool(flags & 16) or bool(re.search(r"bold|black|heavy|semibold|demi|cmbx", name, re.I))
    italic = bool(flags & 2) or bool(re.search(r"italic|oblique|cmti|cmsl", name, re.I))
    return bold, italic


def _fallback_font(name: str, flags: int) -> str:
    """The closest built-in font: same family type, weight and slant."""
    exact = BASE14.get(name.lower().split("+")[-1])
    if exact:
        return exact
    bold, italic = _weight(name, flags)
    if flags & 8 or MONO_RE.search(name):
        fam = ("cour", "cobo", "coit", "cobi")
    elif not SANS_RE.search(name) and (flags & 4 or SERIF_RE.search(name)):
        fam = ("tiro", "tibo", "tiit", "tibi")
    else:
        fam = ("helv", "hebo", "heit", "hebi")
    return fam[bold + 2 * italic]


# Installed fonts that match a PDF font: the same family, or a metric-
# compatible twin (Carlito is drawn exactly like Calibri). Files are
# (regular, bold, italic, bold italic) without extension.
SYSTEM_FONT_FILES = {
    "calibri": ("calibri", "calibrib", "calibrii", "calibriz"),
    "arial": ("arial", "arialbd", "ariali", "arialbi"),
    "times": ("times", "timesbd", "timesi", "timesbi"),
    "georgia": ("georgia", "georgiab", "georgiai", "georgiaz"),
    "verdana": ("verdana", "verdanab", "verdanai", "verdanaz"),
    "tahoma": ("tahoma", "tahomabd", "tahoma", "tahomabd"),
    "segoeui": ("segoeui", "segoeuib", "segoeuii", "segoeuiz"),
    "trebuchet": ("trebuc", "trebucbd", "trebucit", "trebucbi"),
    "garamond": ("GARA", "GARABD", "GARAIT", "GARABD"),
    "bookantiqua": ("BKANT", "ANTQUAB", "ANTQUAI", "ANTQUABI"),
    "consolas": ("consola", "consolab", "consolai", "consolaz"),
    "courier": ("cour", "courbd", "couri", "courbi"),
    "carlito": ("Carlito-Regular", "Carlito-Bold", "Carlito-Italic", "Carlito-BoldItalic"),
    "liberationsans": ("LiberationSans-Regular", "LiberationSans-Bold", "LiberationSans-Italic", "LiberationSans-BoldItalic"),
    "liberationserif": ("LiberationSerif-Regular", "LiberationSerif-Bold", "LiberationSerif-Italic", "LiberationSerif-BoldItalic"),
}
FONT_TWINS = {"carlito": ["carlito", "calibri"], "calibri": ["calibri", "carlito"], "arimo": ["arial", "liberationsans"],
              "liberationsans": ["liberationsans", "arial"], "helvetica": ["arial", "liberationsans"],
              "arial": ["arial", "liberationsans"], "arialmt": ["arial", "liberationsans"],
              "tinos": ["times", "liberationserif"], "liberationserif": ["liberationserif", "times"],
              "timesnewroman": ["times", "liberationserif"], "timesnewromanpsmt": ["times", "liberationserif"],
              "cousine": ["courier"], "couriernew": ["courier"], "trebuchetms": ["trebuchet"],
              "palatinolinotype": ["bookantiqua"]}
FONT_DIRS = ["C:/Windows/Fonts", os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts"),
             "/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
             "/Library/Fonts", "/System/Library/Fonts"]
_system_cache: Dict[Tuple[str, bool, bool], Optional[fitz.Font]] = {}


def _system_font(name: str, flags: int) -> Optional[fitz.Font]:
    """The installed font matching a PDF font name ("ABCDEF+Carlito-Bold"),
    or None."""
    base = name.split("+")[-1]
    family = re.split(r"[-,]", base)[0]
    family = re.sub(r"(PS|MT|PSMT)$", "", family).replace(" ", "").lower()
    bold, italic = _weight(base, flags)
    k = (family, bold, italic)
    if k in _system_cache:
        return _system_cache[k]
    font = None
    for fam in FONT_TWINS.get(family, [family]):
        files = SYSTEM_FONT_FILES.get(fam)
        stems = [files[bold + 2 * italic]] if files else []
        for d in FONT_DIRS:
            if font is not None or not os.path.isdir(d):
                continue
            for root, _, names in os.walk(d):
                lower = {n.lower(): n for n in names}
                for stem in stems:
                    for ext in (".ttf", ".otf"):
                        if stem.lower() + ext in lower:
                            try:
                                font = fitz.Font(fontfile=os.path.join(root, lower[stem.lower() + ext]))
                            except Exception:
                                font = None
                            break
                    if font is not None:
                        break
                if font is not None:
                    break
        if font is not None:
            break
    _system_cache[k] = font
    return font


class _PdfFonts:
    """Fonts to draw with, per original font name. The original embedded
    font is used when every character needed already appears in text set in
    that font (so the glyph is in its subset); otherwise the closest built-in."""

    def __init__(self, doc: fitz.Document):
        self.doc = doc
        self.used: Dict[str, set] = {}
        self.xref: Dict[str, int] = {}
        self.cache: Dict[str, Optional[fitz.Font]] = {}
        for page in doc:
            for f in page.get_fonts(full=True):
                self.xref.setdefault(f[3].split("+")[-1], f[0])
            for b in page.get_text("rawdict")["blocks"]:
                for l in b.get("lines", []):
                    for s in l["spans"]:
                        self.used.setdefault(s["font"], set()).update(c["c"] for c in s["chars"])

    def _embedded(self, name: str) -> Optional[fitz.Font]:
        if name not in self.cache:
            font = None
            xref = self.xref.get(name.split("+")[-1])
            if xref:
                try:
                    _, ext, ftype, buf = self.doc.extract_font(xref)
                    if buf and ext != "n/a" and "Type3" not in ftype:
                        font = fitz.Font(fontbuffer=buf)
                except Exception:
                    font = None
            self.cache[name] = font
        return self.cache[name]

    def get(self, name: str, flags: int, text: str) -> fitz.Font:
        chars = {c for c in text if not c.isspace()}
        font = self._embedded(name)
        if font is not None and chars <= self.used.get(name, set()) and all(font.has_glyph(ord(c)) for c in chars):
            return font
        system = _system_font(name, flags)
        if system is not None and all(system.has_glyph(ord(c)) for c in chars):
            return system
        return fitz.Font(_fallback_font(name, flags))

    def runs(self, name: str, flags: int, text: str) -> List[Tuple[str, fitz.Font]]:
        """`text` split into runs by font: the original embedded font for
        every character it has, else the same font installed on this machine
        (or its metric twin), else the closest built-in, so a new line looks
        like its neighbours even if a letter is missing from the subset."""
        font = self._embedded(name)
        used = self.used.get(name, set())
        system = _system_font(name, flags)
        fallback = fitz.Font(_fallback_font(name, flags))
        out: List[Tuple[str, fitz.Font]] = []
        for c in text:
            if font is not None and (c.isspace() or (c in used and font.has_glyph(ord(c)))) and not _ambiguous(font, c):
                f = font
            elif system is not None and system.has_glyph(ord(c)) and not _ambiguous(system, c):
                f = system
            else:
                f = fallback
            if out and out[-1][1] is f:
                out[-1] = (out[-1][0] + c, f)
            else:
                out.append((c, f))
        return out


# Characters a font may draw with the same glyph as a look-alike; text
# written with that glyph reads back as the look-alike (a soft hyphen, a
# Greek question mark), which an ATS may not understand.
LOOKALIKES = {"-": "­", ";": ";", " ": " "}


def _ambiguous(font: fitz.Font, c: str) -> bool:
    alt = LOOKALIKES.get(c)
    return bool(alt) and font.has_glyph(ord(c)) != 0 and font.has_glyph(ord(c)) == font.has_glyph(ord(alt))


def _runs_width(runs: List[Tuple[str, fitz.Font]], size: float) -> float:
    return sum(f.text_length(t, fontsize=size) for t, f in runs)


def _write_runs(tw: fitz.TextWriter, x: float, y: float, runs: List[Tuple[str, fitz.Font]], size: float) -> float:
    for t, f in runs:
        tw.append((x, y), t, font=f, fontsize=size)
        x += f.text_length(t, fontsize=size)
    return x


def _pdf_lines(page: fitz.Page) -> List[dict]:
    """Visual lines, each {text, chars, block_x1, last_in_block}; text is
    NFKC-normalised (ligatures split) with one char record per text char."""
    out = []
    for b in page.get_text("rawdict")["blocks"]:
        if b.get("type") != 0:
            continue
        lines = b.get("lines", [])
        block_x1 = max((l["bbox"][2] for l in lines), default=0)
        for n, l in enumerate(lines):
            chars = []
            for s in l["spans"]:
                style = (s["font"], s["size"], s["color"], s["flags"])
                for c in s["chars"]:
                    for ch in unicodedata.normalize("NFKC", c["c"]) or " ":
                        chars.append({"c": ch, "bbox": c["bbox"], "origin": c["origin"], "style": style})
            if chars and l["dir"] == (1.0, 0.0):  # horizontal text only
                out.append({"text": "".join(c["c"] for c in chars), "chars": chars, "bbox": l["bbox"],
                            "block_x1": block_x1, "last_in_block": n == len(lines) - 1})
    return out


def _right_bound(line: dict, lines: List[dict], page: fitz.Page) -> float:
    """How far right this line may run: up to the next text on the same row
    (a right-aligned date column), else the page's text margin."""
    x1 = max(c["bbox"][2] for c in line["chars"])
    y0, y1 = line["bbox"][1], line["bbox"][3]
    mid = (y0 + y1) / 2
    right = [l["bbox"][0] for l in lines if l is not line and l["bbox"][0] >= x1 - 1
             and l["bbox"][1] <= mid <= l["bbox"][3]]
    if right:
        return min(right) - 4
    margin = max((l["bbox"][2] for l in lines), default=page.rect.width - 36)
    return max(margin, x1)


def _tail_words(chars: List[dict]) -> List[dict]:
    """Runs of non-space characters that share a style, with their original x0/x1."""
    words: List[dict] = []
    for c in chars:
        if c["c"].isspace():
            if words:
                words[-1]["closed"] = True
            continue
        w = words[-1] if words else None
        if w and not w.get("closed") and w["style"] == c["style"]:
            w["text"] += c["c"]
            w["x1"] = c["bbox"][2]
        else:
            words.append({"text": c["c"], "style": c["style"], "x0": c["bbox"][0], "x1": c["bbox"][2],
                          "baseline": c["origin"][1]})
    return words


def _fit_line(page: fitz.Page, line: dict, lines: List[dict], fonts: _PdfFonts, s: int, e: int,
              new_text: str, shrink: bool = True) -> Optional[dict]:
    """How the line looks with chars[s:e] replaced by `new_text`, or None if
    it can't fit. A justified line is re-spaced as a whole so its gaps stay
    even; otherwise only the changed word and what follows it are redrawn.
    The change in length is absorbed by the word gaps, then by shrinking the
    redrawn words slightly."""
    chars = line["chars"]
    line_end = max(c["bbox"][2] for c in chars)
    bound = _right_bound(line, lines, page)
    justified = (not line["last_in_block"] and line_end >= min(line["block_x1"], bound) - 1.5
                 and line_end >= bound - 0.03 * page.rect.width and any(c["c"] == " " for c in chars))
    head = _tail_words(chars[:s]) if justified else []
    first = chars[s]

    # The new text in the replaced word's style, then the rest of the line
    # in its own styles, keeping each original gap between words.
    pieces = head + [{"text": new_text, "style": first["style"], "baseline": first["origin"][1],
                      "x0": first["bbox"][0], "x1": chars[e - 1]["bbox"][2]}]
    pieces += _tail_words(chars[e:])
    start = pieces[0]["x0"]
    prev_x1 = start
    for p in pieces:
        p["gap"] = max(p["x0"] - prev_x1, 0.0)
        prev_x1 = p["x1"]
        name, size, _, flags = p["style"]
        p["runs"] = fonts.runs(name, flags, p["text"])
        p["width"] = _runs_width(p["runs"], size)

    natural_end = start + sum(p["gap"] + p["width"] for p in pieces)
    space_gaps = [p for p in pieces[1:] if p["gap"] > 0.5]
    target = line_end if justified and space_gaps else min(natural_end, bound)
    scale = 1.0
    excess = natural_end - target
    if space_gaps and excess:
        room = sum(p["gap"] * (1 - MIN_GAP_SHARE) for p in space_gaps)
        take = min(excess, room) if excess > 0 else excess
        total_gap = sum(p["gap"] for p in space_gaps)
        for p in space_gaps:
            p["gap"] -= take * p["gap"] / total_gap
        excess -= take
    if excess > 0.5:
        width = sum(p["gap"] + p["width"] for p in pieces)
        scale = (width - excess) / width
        if scale < MIN_FONT_SCALE or not shrink:
            return None
    return {"pieces": pieces, "scale": scale, "start": start, "line_end": line_end,
            "redrawn": chars if head else chars[s:], "lead": chars[0] if head else first}


def _draw_fit(page: fitz.Page, fit: dict) -> None:
    """Removes the old words (only this line's band, so lines above and
    below are untouched) and draws the new ones."""
    redrawn = fit["redrawn"]
    y0, y1 = min(c["bbox"][1] for c in redrawn), max(c["bbox"][3] for c in redrawn)
    h = y1 - y0
    w0 = fit["lead"]["bbox"][2] - fit["lead"]["bbox"][0]
    rect = fitz.Rect(fit["start"] + 0.3 * w0, y0 + 0.3 * h, fit["line_end"] + 0.5, y1 - 0.3 * h)
    page.add_redact_annot(rect, fill=False)
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                          text=fitz.PDF_REDACT_TEXT_REMOVE)
    writers: Dict[int, fitz.TextWriter] = {}
    x, scale = fit["start"], fit["scale"]
    for p in fit["pieces"]:
        x += p["gap"] * scale
        _, size, color, _ = p["style"]
        tw = writers.setdefault(color, fitz.TextWriter(page.rect))
        _write_runs(tw, x, p["baseline"], p["runs"], size * scale)
        x += p["width"] * scale
    for color, tw in writers.items():
        tw.write_text(page, color=fitz.sRGB_to_pdf(color))


def _pdf_find(doc: fitz.Document, op: dict):
    """(page, line, (start, end), lines) of op["old"] on the line that best
    matches the op's context, or None."""
    best = None
    for page in doc:
        lines = _pdf_lines(page)
        for line in lines:
            span = _locate(line["text"], op)
            if span is None:
                continue
            if op["append"] and line["text"][span[1]:].strip(" .,;") != "":
                continue  # an addition goes at the end of the list
            score = _context_score(line["text"], op["line"])
            if score >= MIN_CONTEXT and (best is None or score > best[0]):
                best = (score, page, line, span, lines)
    return best[1:] if best else None


def _pdf_apply(doc: fitz.Document, fonts: _PdfFonts, op: dict, wraps: List[dict]) -> Optional[str]:
    """Applies one in-line op; returns None on success or the reason it
    couldn't. Skills additions that don't all fit on the line keep as many
    as fit there, and the rest are queued in `wraps` as a continuation line."""
    found = _pdf_find(doc, op)
    if found is None:
        return f"Couldn't find “{op['old']}” in the PDF (it may be split across two lines)."
    page, line, (s, e), lines = found
    terms, sep = op.get("terms") or [], op.get("sep") or ", "
    if op["append"] and terms:
        for k in range(len(terms), -1, -1):
            new = op["old"] + "".join(sep + t for t in terms[:k]) + ("" if k == len(terms) else sep.rstrip())
            fit = _fit_line(page, line, lines, fonts, s, e, new, shrink=k == 0)  # wrap before shrinking
            if fit:
                _draw_fit(page, fit)
                if k < len(terms):
                    wraps.append(_wrap_request(page, line, lines, sep.join(terms[k:])))
                return None
        return f"Not enough room on the line for “{op['new']}”."
    fit = _fit_line(page, line, lines, fonts, s, e, op["new"])
    if fit is None:
        return f"Not enough room on the line for “{op['new']}”."
    _draw_fit(page, fit)
    return None


# ---------------------------------------------------------------- PDF: new lines

BULLET_GLYPHS = set("•●○◦▪■□‣∙·➢►➤–-*�")
MIN_BOTTOM_MARGIN = 18.0  # a new line may use the bottom margin down to this
GAP_SHARE = 0.35  # share of a section/paragraph gap that may be given up to fit new lines
GAP_SHARE_MAX = 0.6  # ... when that isn't enough
ROW_TIGHTEN = 0.08  # share of the line height any line may move up, when that isn't enough either
MIN_SLACK_GAP = 3.0  # whitespace (pt) between lines that counts as a section/paragraph gap


def _center(bbox) -> float:
    return (bbox[1] + bbox[3]) / 2


def _baseline(line: dict) -> float:
    return max(c["origin"][1] for c in line["chars"])


def _bullet_start(line: dict, lines: List[dict]) -> Tuple[Optional[dict], int]:
    """(bullet glyph info, index of the first text char) for a line that
    starts a bullet; the glyph may be in the line or a separate line beside it."""
    chars = line["chars"]
    i = 0
    while i < len(chars) and chars[i]["c"].isspace():
        i += 1
    if i < len(chars) and chars[i]["c"] in BULLET_GLYPHS and (
            i + 1 >= len(chars) or chars[i + 1]["c"].isspace() or chars[i + 1]["bbox"][0] - chars[i]["bbox"][2] > 1.5):
        j = i + 1
        while j < len(chars) and chars[j]["c"].isspace():
            j += 1
        return {"bbox": fitz.Rect(chars[i]["bbox"]), "baseline": chars[i]["origin"][1], "line": line}, j
    x0 = line["bbox"][0]
    for other in lines:
        t = other["text"].strip()
        if (other is not line and len(t) == 1 and t in BULLET_GLYPHS and other["bbox"][2] <= x0 + 1
                and x0 - other["bbox"][2] < 30 and abs(_center(other["bbox"]) - _center(line["bbox"])) < 3):
            ch = next(c for c in other["chars"] if not c["c"].isspace())
            return {"bbox": fitz.Rect(ch["bbox"]), "baseline": ch["origin"][1], "line": other}, i
    return None, i


def _rows(lines: List[dict]) -> List[List[dict]]:
    """Lines grouped into printed rows (a bullet and its right-aligned date,
    or a line whose edited tail became a separate line), top to bottom."""
    rows: List[List[dict]] = []
    for l in sorted(lines, key=lambda l: _center(l["bbox"])):
        size = l["chars"][0]["style"][1]
        if rows and abs(_center(l["bbox"]) - _center(rows[-1][0]["bbox"])) < 0.4 * size:
            rows[-1].append(l)
        else:
            rows.append([l])
    return rows


def _column_lines(line: dict, lines: List[dict]) -> List[dict]:
    """Lines in the same column, top to bottom (horizontal overlap)."""
    x0, x1 = line["bbox"][0], line["bbox"][2]
    return sorted((l for l in lines if l["bbox"][0] < x1 - 2 and l["bbox"][2] > x0 + 2), key=lambda l: _center(l["bbox"]))


def _typical_spacing(lines: List[dict], size: float) -> float:
    """The usual baseline-to-baseline distance inside a paragraph."""
    base = sorted({round(_baseline(l), 1) for l in lines})
    diffs = sorted(b - a for a, b in zip(base, base[1:]) if 0.95 * size < b - a < 1.6 * size)
    return diffs[len(diffs) // 2] if diffs else 1.2 * size


def _anchor(line: dict) -> dict:
    """Where a line is, so it can be found again after its text changes."""
    return {"anchor_center": _center(line["bbox"]), "anchor_x0": line["bbox"][0]}


def _wrap_request(page: fitz.Page, line: dict, lines: List[dict], text: str) -> dict:
    """A continuation line under a Skills line, aligned with its first item
    ("Tools: Git, ..." -> under "Git"). An earlier edit may have split the
    printed row into pieces, so the label is looked for across the row."""
    style_chars = line["chars"]
    row = sorted((l for l in lines if abs(_center(l["bbox"]) - _center(line["bbox"])) < 2), key=lambda l: l["bbox"][0])
    line = next((l for l in row if ":" in l["text"]), row[0] if row else line)
    chars = line["chars"]
    colon = line["text"].find(":")
    i = colon + 1 if 0 <= colon < len(chars) - 1 else 0
    while i < len(chars) - 1 and chars[i]["c"].isspace():
        i += 1
    style = next((c["style"] for c in reversed(style_chars) if not c["c"].isspace()), chars[i]["style"])
    return dict(_anchor(line), pno=page.number, x=chars[i]["bbox"][0], style=style, text=text, bullet=False)


def _sentence_request(doc: fitz.Document, op: dict) -> Optional[dict]:
    """Where a new bullet goes: right after the bullet it relates to (all of
    its wrapped lines), matching that bullet's glyph, indent and font."""
    best = None
    for page in doc:
        lines = _pdf_lines(page)
        for line in lines:
            score = _context_score(line["text"], op["line"])
            if score >= MIN_CONTEXT and (best is None or score > best[0]):
                best = (score, page, line, lines)
    if best is None:
        return None
    _, page, line, lines = best
    col = _column_lines(line, lines)
    k = next(i for i, l in enumerate(col) if l is line)
    size = line["chars"][0]["style"][1]
    while k > 0 and _bullet_start(col[k], lines)[0] is None and _baseline(col[k]) - _baseline(col[k - 1]) < 1.6 * size:
        k -= 1  # back to the line the bullet starts on
    start = col[k]
    glyph, ti = _bullet_start(start, lines)
    if glyph is None:
        return None
    ti = min(ti, len(start["chars"]) - 1)
    return dict(_anchor(start), pno=page.number, x=start["chars"][ti]["bbox"][0],
                style=start["chars"][ti]["style"], text=op["new"], bullet=True)


def _find_anchor(req: dict, lines: List[dict]) -> Optional[dict]:
    near = [l for l in lines if abs(_center(l["bbox"]) - req["anchor_center"]) < 2.5]
    return min(near, key=lambda l: abs(l["bbox"][0] - req["anchor_x0"]), default=None)


def _resolve(req: dict, lines: List[dict], fonts: _PdfFonts) -> Optional[dict]:
    """Fills in a request's geometry on the page as it is now: the anchor's
    last row, the row after it, line spacing, and the wrapped rows of text."""
    anchor = _find_anchor(req, lines)
    if anchor is None:
        return None
    name, size, _, flags = req["style"]
    col = _column_lines(anchor, lines)
    rows = _rows(col)
    k = next(i for i, r in enumerate(rows) if any(l is anchor for l in r))
    glyph = _bullet_start(anchor, lines)[0] if req["bullet"] else None
    if req["bullet"]:
        # Include the bullet's wrapped lines: indented past the glyph, no glyph of their own.
        while (k + 1 < len(rows) and all(_bullet_start(l, lines)[0] is None for l in rows[k + 1])
               and min(l["bbox"][0] for l in rows[k + 1]) >= req["x"] - 2
               and min(map(_baseline, rows[k + 1])) - max(map(_baseline, rows[k])) < 1.6 * size):
            k += 1
    end, nxt = rows[k], rows[k + 1] if k + 1 < len(rows) else None
    end_base = max(map(_baseline, end))
    line_h = _typical_spacing(col, size)
    gap = line_h
    if req["bullet"]:
        # The bullet-to-bullet distance: to the next bullet, or the usual one on the page.
        starts = [r for r in rows if any(_bullet_start(l, lines)[0] for l in r)]
        if nxt is not None and any(_bullet_start(l, lines)[0] for l in nxt):
            gap = min(map(_baseline, nxt)) - end_base
        else:
            gaps = sorted(min(map(_baseline, b)) - max(map(_baseline, a)) for a, b in zip(rows, rows[1:])
                          if b in starts and min(map(_baseline, b)) - max(map(_baseline, a)) < 2.2 * size)
            gap = gaps[len(gaps) // 2] if gaps else line_h
    right = max(l["bbox"][2] for l in col)
    rows_out, cur = [], ""
    for word in req["text"].split():
        trial = f"{cur} {word}".strip()
        if cur and _runs_width(fonts.runs(name, flags, trial), size) > right - req["x"]:
            rows_out.append(cur)
            cur = word
        else:
            cur = trial
    rows_out.append(cur)
    bottom = max(l["bbox"][3] for l in end)
    if nxt is None:
        split = bottom + 1
    else:
        top = min(l["bbox"][1] for l in nxt)
        split = (bottom + top) / 2 if top >= bottom else (_center(end[0]["bbox"]) + _center(nxt[0]["bbox"])) / 2
    return dict(req, end_base=end_base, split=split, gap=gap, line_h=line_h, rows=rows_out, glyph=glyph,
                dy=gap + (len(rows_out) - 1) * line_h,
                col=(min(l["bbox"][0] for l in col) - 2, max(l["bbox"][2] for l in col) + 2))


def _shrunk(bbox, share: float = 0.3) -> fitz.Rect:
    """The middle band of a line: redacting it removes this line's text and
    never a neighbour's, even when line boxes overlap."""
    r = fitz.Rect(bbox)
    h = r.height
    return fitz.Rect(r.x0 - 0.5, r.y0 + share * h, r.x1 + 0.5, r.y1 - share * h)


def _redact_lines(page: fitz.Page, lines: List[dict]) -> None:
    for l in lines:
        page.add_redact_annot(_shrunk(l["bbox"]), fill=False)
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                          text=fitz.PDF_REDACT_TEXT_REMOVE)


def _redact_graphics(page: fitz.Page, rects: List[fitz.Rect]) -> None:
    for r in rects:
        if r.height > 0 and r.width > 0:
            page.add_redact_annot(r, fill=False)
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                          graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED, text=fitz.PDF_REDACT_TEXT_NONE)


def _clear_of_drawings(y: float, lo: float, hi: float, drawings: List[fitz.Rect]) -> Optional[float]:
    """A split height near `y`, between `lo` and `hi`, that doesn't cut
    through a drawn rule or box; None if there isn't one."""
    for cand in (y, *(d.y1 + 1.2 for d in drawings), *(d.y0 - 1.2 for d in drawings)):
        if lo < cand < hi and not any(d.y0 - 1 < cand < d.y1 + 1 for d in drawings):
            return cand
    return None


def _row_gaps(rows: List[List[dict]], drawings: List[fitz.Rect]) -> List[dict]:
    """The boundary between each pair of rows: where to split, and how much
    room it can give. Section/paragraph gaps give a share of their
    whitespace; any row may also move up by a sliver of its line height."""
    out = []
    for a, b in zip(rows, rows[1:]):
        lo, hi = max(l["bbox"][3] for l in a), min(l["bbox"][1] for l in b)
        size = a[0]["chars"][0]["style"][1]
        if hi > lo:
            y = _clear_of_drawings((lo + hi) / 2, lo, hi, drawings)
        else:  # line boxes overlap (tight leading): split between the line centres
            y = (_center(a[0]["bbox"]) + _center(b[0]["bbox"])) / 2
        if y is None:
            continue
        w = max(hi - lo, 0.0)
        slack = w >= MIN_SLACK_GAP
        out.append({"y": y, "w": w, "slack": slack,
                    "soft": w * GAP_SHARE if slack else 0.0,
                    "firm": max(w * GAP_SHARE_MAX if slack else 0.0, ROW_TIGHTEN * size)})
    return out


def _reflow(doc: fitz.Document, pno: int, reqs: List[dict], fonts: _PdfFonts) -> Dict[int, str]:
    """Opens room for the requested lines on one page and draws them.

    The column below each insertion point moves down as vector content, so
    text stays selectable and links keep working; nothing is resized. Room
    comes, in order, from: the bottom margin and a share of the larger gaps
    between sections and entries; then a little more of those gaps and a
    sliver of line spacing, spread evenly; and if the page is still too full,
    the content from the last section/entry break that doesn't fit moves to a
    new page right after this one. Returns {request index: reason} for lines
    that couldn't be placed at all."""
    page = doc[pno]
    W, H = page.rect.width, page.rect.height
    lines = _pdf_lines(page)
    failed: Dict[int, str] = {}
    resolved = []
    for i, r in enumerate(reqs):
        rr = _resolve(r, lines, fonts)
        if rr is None:
            failed[i] = "Couldn't find where this line goes in the PDF."
        else:
            resolved.append((i, rr))
    if not resolved:
        return failed

    # One column per page: the first request's column decides.
    col = resolved[0][1]["col"]
    same = []
    for i, r in resolved:
        if r["col"][0] < col[1] and r["col"][1] > col[0]:
            same.append((i, r))
        else:
            failed[i] = "This line is in a different column of the page."
    same.sort(key=lambda ir: ir[1]["split"])
    y_first = same[0][1]["split"]
    below = [l for l in lines if _center(l["bbox"]) > y_first]
    full_width = all(l["bbox"][0] < col[1] and l["bbox"][2] > col[0] for l in below)
    cx0, cx1 = (0.0, W) if full_width else col

    def in_column(bbox) -> bool:
        return cx0 <= (bbox[0] + bbox[2]) / 2 <= cx1

    region = [l for l in lines if _center(l["bbox"]) > y_first and in_column(l["bbox"])]
    drawings = [fitz.Rect(d["rect"]) for d in page.get_drawings() if in_column(d["rect"])]
    content_bottom = max([l["bbox"][3] for l in region] + [d.y1 for d in drawings if d.y0 >= y_first] + [y_first])
    limit = H - min(MIN_BOTTOM_MARGIN, H - content_bottom)
    rows = _rows(region)
    gaps = [g for g in _row_gaps(rows, drawings)
            if not any(abs(g["y"] - r["split"]) < 0.5 for _, r in same)]
    total = sum(r["dy"] for _, r in same)
    deficit = content_bottom + total - limit

    # Room: soft (section gaps), then firm (more of them plus a sliver of
    # every line gap, spread evenly), then a new page.
    takes: List[Tuple[float, float]] = []
    cut = None
    if deficit > 0:
        soft = sum(g["soft"] for g in gaps)
        firm = sum(g["firm"] for g in gaps)
        if deficit <= soft:
            left = deficit
            for g in sorted(gaps, key=lambda g: -g["soft"]):
                if left <= 0:
                    break
                t = min(g["soft"], left)
                takes.append((g["y"], t))
                left -= t
        elif deficit <= firm:
            takes = [(g["y"], g["firm"] * deficit / firm) for g in gaps]
        else:
            # Too much for one page: break before the first row that would
            # run past the bottom, moved up to a section/entry gap nearby.
            def shifted_bottom(row):
                c = _center(row[0]["bbox"])
                return max(l["bbox"][3] for l in row) + sum(r["dy"] for _, r in same if r["split"] <= c)
            over = next((k for k, row in enumerate(rows) if shifted_bottom(row) > limit), None)
            if over is not None and over > 0:
                k = over
                for j in range(over, max(over - 6, 0), -1):
                    g = next((g for g in gaps if max(l["bbox"][3] for l in rows[j - 1]) <= g["y"]
                              <= min(l["bbox"][1] for l in rows[j])), None)
                    if g is not None and g["slack"]:
                        k = j
                        break
                cut = (max(l["bbox"][3] for l in rows[k - 1]) + min(l["bbox"][1] for l in rows[k])) / 2
            else:
                cut = y_first + 0.01  # everything below the first insertion moves on

    bounds = sorted([(r["split"], r["dy"]) for _, r in same] + [(y, -t) for y, t in takes]
                    + ([(cut, 0.0)] if cut is not None else []))
    ys = [b[0] for b in bounds]
    shifts, acc = [], 0.0
    for _, d in bounds:
        acc += d
        shifts.append(acc)

    def band_of(y: float) -> int:
        return max((k for k, s in enumerate(ys) if s <= y), default=-1)

    src = doc.tobytes()
    links = [l for l in page.get_links() if _center(l["from"]) > y_first and in_column(l["from"])]
    top_margin = min((l["bbox"][1] for l in lines), default=36.0)
    cut_at = None
    if cut is not None:
        cut_at = cut + shifts[ys.index(cut)]  # where the break would sit on this page
        page2 = doc.new_page(pno + 1, width=W, height=H)
        page = doc[pno]  # adding a page invalidates the old page handle

    def place(y_shifted: float) -> Tuple[fitz.Page, float]:
        """Final page and height for a point already moved down on this page."""
        if cut_at is not None and y_shifted >= cut_at - 0.01:
            return page2, top_margin + (y_shifted - cut_at)
        return page, y_shifted

    # The page keeps everything outside the moved region; the region is drawn
    # again band by band from copies holding only that band's text. Clips are
    # padded so no ascender or descender is cut; other bands' text and
    # graphics are already removed from each copy.
    _redact_lines(page, region)
    _redact_graphics(page, [fitz.Rect(cx0, y_first, cx1, H)])
    for k, top in enumerate(ys):
        bottom = ys[k + 1] if k + 1 < len(ys) else H
        copy = fitz.open("pdf", src)
        cp = copy[pno]
        _redact_lines(cp, [l for l in _pdf_lines(cp)
                           if not (_center(l["bbox"]) > y_first and in_column(l["bbox"]) and band_of(_center(l["bbox"])) == k)])
        _redact_graphics(cp, [fitz.Rect(0, 0, W, top), fitz.Rect(0, bottom, W, H),
                              fitz.Rect(0, top, cx0, bottom), fitz.Rect(cx1, top, W, bottom)])
        pad = 6.0
        clip = fitz.Rect(cx0, top - (pad if k else 0.0), cx1, bottom + (pad if k + 1 < len(ys) else 0.0))
        target_page, y0 = place(top + shifts[k])
        dy = y0 - top
        target_page.show_pdf_page(clip + (0, dy, 0, dy), copy, pno, clip=clip)

    # The new lines, in the gap each insertion opened; several after the
    # same line stack in the order they were asked for.
    stacked: Dict[float, float] = {}
    for _, r in same:
        k = band_of(r["split"] - 0.01)
        before = shifts[k] if k >= 0 else 0.0
        key = round(r["split"], 1)
        base = r["end_base"] + before + stacked.get(key, 0.0)
        stacked[key] = stacked.get(key, 0.0) + r["dy"]
        name, size, color, flags = r["style"]
        writers: Dict[int, fitz.TextWriter] = {}
        for j, row in enumerate(r["rows"]):
            target_page, y = place(base + r["gap"] + j * r["line_h"])
            tw = writers.setdefault(target_page.number, fitz.TextWriter(target_page.rect))
            _write_runs(tw, r["x"], y, fonts.runs(name, flags, row), size)
        for n, tw in writers.items():
            tw.write_text(doc[n], color=fitz.sRGB_to_pdf(color))
        if r["glyph"]:
            # A copy of the anchor's own bullet glyph, on the new line.
            g = r["glyph"]
            copy = fitz.open("pdf", src)
            cp = copy[pno]
            keep = g["line"]
            _redact_lines(cp, [l for l in _pdf_lines(cp) if abs(_center(l["bbox"]) - _center(keep["bbox"])) > 1
                               or abs(l["bbox"][0] - keep["bbox"][0]) > 1])
            rest = fitz.Rect(g["bbox"].x1 + 0.3, keep["bbox"][1], keep["bbox"][2], keep["bbox"][3])
            if rest.width > 1:
                cp.add_redact_annot(_shrunk(rest), fill=False)
                cp.apply_redactions(images=0, graphics=0, text=0)
            clip = g["bbox"] + (-1, -1, 1, 1)
            target_page, y = place(base + r["gap"])
            dy = y - g["baseline"]  # the copy is the page before reflow
            target_page.show_pdf_page(clip + (0, dy, 0, dy), copy, pno, clip=clip)

    for link in links:
        k = band_of(_center(link["from"]))
        if k >= 0:
            r = fitz.Rect(link["from"])
            target_page, y0 = place(r.y0 + shifts[k])
            page.delete_link(link)
            link["from"] = fitz.Rect(r.x0, y0, r.x1, y0 + r.height)
            target_page.insert_link(link)
    return failed


def edit_pdf(content: bytes, ops: List[dict]) -> Tuple[bytes, List[Optional[str]]]:
    """Word changes first, in place; then new lines (wrapped Skills items and
    new bullets), each page reflowed once."""
    with fitz.open(stream=content, filetype="pdf") as doc:
        fonts = _PdfFonts(doc)
        errors: List[Optional[str]] = [None] * len(ops)
        requests: List[Tuple[Optional[int], dict]] = []  # (op index, request)
        for i, op in enumerate(ops):
            if op["insert"]:
                continue
            wraps: List[dict] = []
            errors[i] = _pdf_apply(doc, fonts, op, wraps)
            requests += [(i, w) for w in wraps]
        for i, op in enumerate(ops):
            if op["insert"]:
                req = _sentence_request(doc, op)
                if req is None:
                    errors[i] = "Couldn't find the bullet this line goes under in the PDF."
                else:
                    requests.append((i, req))
        fonts = _PdfFonts(doc)  # the word changes may have added fonts
        # Last page first: a page added for overflow doesn't renumber pages still to do.
        for pno in sorted({r["pno"] for _, r in requests}, reverse=True):
            on_page = [(i, r) for i, r in requests if r["pno"] == pno]
            for k, reason in _reflow(doc, pno, [r for _, r in on_page], fonts).items():
                errors[on_page[k][0]] = reason
        return doc.tobytes(garbage=4, deflate=True), errors


# ---------------------------------------------------------------- DOCX

def _docx_paragraphs(doc) -> list:
    """Every paragraph: body (with tables and text boxes), headers, footers."""
    parts = [doc.element.body]
    for sec in doc.sections:
        for hf in (sec.header, sec.footer, sec.first_page_header, sec.first_page_footer):
            if hf is not None and hf._element is not None:
                parts.append(hf._element)
    return [p for part in parts for p in part.iter(qn("w:p"))]


def _replace_in_runs(nodes: list, start: int, end: int, new: str) -> None:
    """Replaces text[start:end] across the paragraph's w:t nodes; the new
    text goes into the first node, so it takes that run's formatting."""
    pos, placed = 0, False
    for t in nodes:
        text = t.text or ""
        a, b = pos, pos + len(text)
        pos = b
        if b <= start or a >= end:
            continue
        s0, e0 = max(start, a) - a, min(end, b) - a
        t.text = text[:s0] + (new if not placed else "") + text[e0:]
        t.set(qn("xml:space"), "preserve")
        placed = True


def _docx_text(p) -> str:
    return "".join(t.text or "" for t in p.iter(qn("w:t")))


def _docx_insert(paragraphs: list, op: dict) -> Optional[str]:
    """Adds op["new"] as a new paragraph after the one it relates to, cloned
    from it so it keeps the same bullet, indent and font. The text goes into
    the run holding most of the text (the body, not a bold lead-in)."""
    scored = [(_context_score(_docx_text(p), op["line"]), i) for i, p in enumerate(paragraphs) if _docx_text(p).strip()]
    score, i = max(scored, default=(0.0, -1))
    if score < MIN_CONTEXT:
        return "Couldn't find where this bullet belongs in the document."
    anchor = paragraphs[i]
    clone = copy.deepcopy(anchor)
    for el in list(clone.iter()):  # ids must stay unique in the document
        if el.tag in (qn("w:bookmarkStart"), qn("w:bookmarkEnd"), qn("w:commentRangeStart"),
                      qn("w:commentRangeEnd"), qn("w:commentReference")):
            el.getparent().remove(el)
    for attr in [a for a in clone.attrib if a.endswith("}paraId") or a.endswith("}textId")]:
        del clone.attrib[attr]
    nodes = list(clone.iter(qn("w:t")))
    main = max(nodes, key=lambda t: len(t.text or ""))
    for t in nodes:
        t.text = op["new"] if t is main else ""
    main.set(qn("xml:space"), "preserve")
    anchor.addnext(clone)
    paragraphs.insert(i + 1, clone)
    return None


def edit_docx(content: bytes, ops: List[dict]) -> Tuple[bytes, List[Optional[str]]]:
    doc = Document(io.BytesIO(content))
    paragraphs = _docx_paragraphs(doc)
    errors: List[Optional[str]] = []
    for op in ops:
        if op["insert"]:
            errors.append(_docx_insert(paragraphs, op))
            continue
        best = None
        for p in paragraphs:
            nodes = list(p.iter(qn("w:t")))
            text = "".join(t.text or "" for t in nodes)
            span = _locate(text, op)
            if span is None:
                continue
            score = _context_score(text, op["line"])
            if score >= MIN_CONTEXT and (best is None or score > best[0]):
                best = (score, nodes, span)
        if best is None:
            errors.append(f"Couldn't find “{op['old']}” in the document.")
            continue
        _, nodes, (s, e) = best
        _replace_in_runs(nodes, s, e, op["new"])
        errors.append(None)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue(), errors


# ---------------------------------------------------------------- LaTeX / plain text

TEX_SPECIAL = {"#": r"\#", "$": r"\$", "%": r"\%", "&": r"\&", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def tex_escape(text: str) -> str:
    return "".join(TEX_SPECIAL.get(c, c) for c in text)


def _tex_pattern(old: str) -> re.Pattern:
    """`old` as it may be written in LaTeX source: "C#" as "C\\#", a space as
    a line break or "~", and never part of a command name (\\item)."""
    parts = []
    for c in old:
        if c.isspace():
            parts.append(r"(?:\s+|~)")
        elif c in TEX_SPECIAL and c != "\\":
            parts.append(r"\\?" + re.escape(c))
        elif c in "–—":
            parts.append({"–": "(?:–|--)", "—": "(?:—|---)"}[c])
        else:
            parts.append(re.escape(c))
    return re.compile(r"(?<![\w\\+#])" + "".join(parts) + r"(?![\w+#])")


def _in_comment_or_url(src: str, pos: int) -> bool:
    line_start = src.rfind("\n", 0, pos) + 1
    if re.search(r"(?<!\\)%", src[line_start:pos]):
        return True
    href = src.rfind(r"\href{", 0, pos)
    return href >= 0 and "}" not in src[href:pos]


# \item, \resumeItem, \cvitem ... but not \resumeItemListStart.
TEX_ITEM = re.compile(r"\\(\w*[Ii]tem)(?![A-Za-z])")


def _tex_insert(src: str, body: int, op: dict) -> Tuple[str, Optional[str]]:
    """Adds op["new"] as a new item after the item it relates to, using the
    same command: \\resumeItem{...} stays \\resumeItem, \\item stays \\item."""
    best = None
    items = [m for m in TEX_ITEM.finditer(src, body) if not _in_comment_or_url(src, m.start())]
    for n, m in enumerate(items):
        rest = src[m.end():]
        braced = rest.lstrip(" \t").startswith("{")
        if braced:
            start = m.end() + len(rest) - len(rest.lstrip(" \t"))
            text, end = _tex_group(src, start)
        else:
            nxt = re.compile(r"\\(?:\w*[Ii]tem(?![A-Za-z])|end\{)").search(src, m.end())
            end = nxt.start() if nxt else len(src)
            text = src[m.end():end]
        score = _context_score(re.sub(r"\\[A-Za-z]+", " ", text), op["line"])
        if best is None or score > best[0]:
            best = (score, m, braced, end)
    if best is None or best[0] < MIN_CONTEXT:
        return src, "Couldn't find where this bullet belongs in the LaTeX source."
    _, m, braced, end = best
    line_start = src.rfind("\n", 0, m.start()) + 1
    indent = re.match(r"[ \t]*", src[line_start:]).group()
    if braced:
        return src[:end] + f"\n{indent}\\{m.group(1)}{{{tex_escape(op['new'])}}}" + src[end:], None
    # A plain \item runs until the next \item or \end: put the new one there.
    return src[:end] + f"\\{m.group(1)} {tex_escape(op['new'])}\n{indent}" + src[end:], None


def edit_tex(content: bytes, ops: List[dict]) -> Tuple[bytes, List[Optional[str]]]:
    src = content.decode("utf-8", errors="replace")
    errors: List[Optional[str]] = []
    for op in ops:
        body = src.find(r"\begin{document}")
        body = body if body >= 0 else 0
        if op["insert"]:
            src, err = _tex_insert(src, body, op)
            errors.append(err)
            continue
        best = None
        matches = [m for m in _tex_pattern(op["old"]).finditer(src, body) if not _in_comment_or_url(src, m.start())]
        for m in matches:
            # Context: the source around the match, minus commands.
            window = re.sub(r"\\[A-Za-z]+", " ", src[max(body, m.start() - 160):m.end() + 160])
            score = len(_words(window) & _words(op["line"])) / max(len(_words(op["line"])), 1)
            if best is None or score > best[0] or (op["append"] and score == best[0]):
                best = (score, m)
        if best is None or best[0] < MIN_CONTEXT:
            errors.append(f"Couldn't find “{op['old']}” in the LaTeX source.")
            continue
        m = best[1]
        if op["append"]:  # keep the original (already escaped) text, add the new items after it
            src = src[:m.end()] + tex_escape(op["new"][len(op["old"]):]) + src[m.end():]
        else:
            src = src[:m.start()] + tex_escape(op["new"]) + src[m.end():]
        errors.append(None)
    return src.encode("utf-8"), errors


def edit_txt(content: bytes, ops: List[dict]) -> Tuple[bytes, List[Optional[str]]]:
    text = content.decode("utf-8", errors="replace")
    lines = text.split("\n")
    errors: List[Optional[str]] = []
    for op in ops:
        if op["insert"]:
            score, i = max(((_context_score(l, op["line"]), i) for i, l in enumerate(lines)), default=(0.0, -1))
            if score < MIN_CONTEXT:
                errors.append("Couldn't find where this bullet belongs in the text.")
                continue
            mark = re.match(r"\s*(?:[-*•‣▪●○◦■□➢►–]|\d+[.)])\s*", lines[i])
            lines.insert(i + 1, (mark.group() if mark else "- ") + op["new"])
            errors.append(None)
            continue
        best = None
        for i, line in enumerate(lines):
            span = _locate(line, op)
            score = _context_score(line, op["line"]) if span else 0.0
            if span and score >= MIN_CONTEXT and (best is None or score > best[0]):
                best = (score, i, span)
        if best is None:
            errors.append(f"Couldn't find “{op['old']}” in the text.")
            continue
        _, i, (s, e) = best
        lines[i] = lines[i][:s] + op["new"] + lines[i][e:]
        errors.append(None)
    return "\n".join(lines).encode("utf-8"), errors


EDITORS = {"pdf": edit_pdf, "docx": edit_docx, "tex": edit_tex, "txt": edit_txt}
MIME = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "tex": "application/x-tex",
    "txt": "text/plain",
}


def apply_edits(fmt: str, content: bytes, ops: List[dict]) -> Tuple[bytes, List[Optional[str]]]:
    """(edited file, per-op error or None) in the file's own format."""
    # Each new bullet goes right after its anchor line, so several for the
    # same anchor are applied last-first to come out in their given order.
    order = list(range(len(ops)))
    groups: Dict[int, List[int]] = {}
    for i, op in enumerate(ops):
        if op["insert"]:
            groups.setdefault(op["line_index"], []).append(i)
    for idxs in groups.values():
        for pos, i in zip(idxs, reversed(idxs)):
            order[pos] = i
    out, errors = EDITORS[fmt](content, [ops[i] for i in order])
    by_op: List[Optional[str]] = [None] * len(ops)
    for i, err in zip(order, errors):
        by_op[i] = err
    return out, by_op
