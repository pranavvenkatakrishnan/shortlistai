# ShortlistAI

Compares a resume against a job description, scores how well it matches, suggests improvements, and generates an improved resume as a clean, consistently formatted PDF.

## Features

- Upload a resume as PDF, DOCX, or TXT, or paste the text
- Match score per job requirement using sentence embeddings (`all-MiniLM-L6-v2`)
- Suggestions for missing or weakly covered requirements
- Regenerates the resume into one clean template, fitted to the original page count; new bullets are placed under the most relevant job or project

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
  matcher.py          Resume vs. job description scoring and suggestions
  resume_parser.py    Text extraction from PDF/DOCX/TXT (PDFs read in visual order)
  resume_template.py  Parses the resume into sections and renders the PDF template
frontend/
  index.html          Single-page UI
```
