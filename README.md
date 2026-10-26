# ShortlistAI

Compares a resume against a job description, scores how well it matches, suggests improvements, and generates an improved resume as a clean, consistently formatted PDF.

## Features

- Upload a resume as PDF, Word (DOCX), LaTeX (TEX), or TXT, or paste the text
- ATS match score: each requirement gets smooth 0-1 credit from sentence embeddings (`all-MiniLM-L6-v2`) plus keyword coverage; the same resume and JD always give the same score
- Three kinds of fixes, each with a plain "Why" and its score gain:
  - **Swap a word**: your word for the JD's word for the same thing ("Postgres" -> "PostgreSQL", "developers" -> "engineers"); one tool is never swapped for another
  - **Add to your skills**: a missing JD keyword appended to the best-fitting Skills line
  - **Add a short bullet**: a one-line bullet in the JD's words for a requirement the resume doesn't cover ("Used Docker for containerized deployments"), editable in place
- **Apply best fixes** picks the fewest fixes that reach 75% (keyword fixes first), checked by re-scoring
- **SWOT**: strengths, weaknesses, opportunities (fixes worth the most points) and threats (what could get the resume filtered out)
- One-click **Download** in the original format, layout kept:
  - **PDF**: words are changed in place in the original embedded font (or the same font installed on the machine, e.g. Calibri for Carlito); a Skills line that runs out of room wraps onto a continuation line aligned under its first item; new bullets are inserted after the related bullet with the same glyph, indent, font and spacing. Content below an insertion moves down as vector content (still selectable, links kept) using the bottom margin and a share of the larger section gaps, then a sliver of line spacing; nothing is rescaled. If the page is still full, the content from the last section break moves to a new page, so every selected bullet is always added.
  - **Word**: runs keep their formatting; new bullets copy the related bullet's paragraph style
  - **LaTeX**: source edited and escaped; new bullets use the same `\resumeItem` or `\item` command
- Skills additions are checked with the embedding model against a skills vocabulary, so JD wording ("quality input", "stay updated") is never added as a skill
- **Semantic analysis**: semantic match, exact-keyword and requirements-met meters; a keyword map (as written / by meaning / missing); and each requirement with meaning and keyword bars, the closest resume line with matching words highlighted, and what's missing
- **Check another job** keeps the resume loaded for the next JD

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r backend/requirements.txt
uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000.

## Project layout

```
backend/
  main.py             FastAPI app and API endpoints
  matcher.py          Scoring, semantic keyword analysis, and keyword edits
  doc_editor.py       Applies keyword edits inside the original PDF/DOCX/TEX/TXT file
  resume_parser.py    Text extraction from PDF/DOCX/TEX/TXT (PDFs read in visual order)
  resume_template.py  Parses the resume into sections; clean PDF for pasted text
frontend/
  index.html          Single-page UI
```
