import base64
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from matcher import analyze, craft_bullet_from_answer, get_model
from resume_parser import UnsupportedFileType, estimate_pages_from_words, extract_pdf_links, extract_text_and_pages
from resume_template import parse_resume, place_bullets, render_pdf

app = FastAPI(title="ShortlistAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store of uploaded originals, keyed by a per-upload id, so the final
# resume keeps the original's page count and links. Prototype-scale only:
# lost on server restart, and not shared across processes.
_RESUME_CACHE: dict[str, dict] = {}


class AnalyzeRequest(BaseModel):
    resume_text: str
    jd_text: str


class ClarifyRequest(BaseModel):
    requirement: str
    answer: str


class GenerateResumeRequest(BaseModel):
    resume_id: Optional[str] = None
    final_text: str
    # Bullets the user accepted that had no existing line to replace; they're
    # placed under the most relevant entry instead of at the end.
    added_bullets: List[str] = []


@app.on_event("startup")
def warm_model():
    get_model()


@app.post("/api/analyze")
def api_analyze(req: AnalyzeRequest):
    if not req.resume_text.strip() or not req.jd_text.strip():
        raise HTTPException(400, "Both resume and job description text are required.")
    try:
        return analyze(req.resume_text, req.jd_text)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/extract-resume")
async def api_extract_resume(file: UploadFile = File(...)):
    content = await file.read()
    try:
        text, fmt, page_count = extract_text_and_pages(file.filename or "", content)
    except UnsupportedFileType as e:
        raise HTTPException(400, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))

    resume_id = uuid.uuid4().hex
    _RESUME_CACHE[resume_id] = {
        "format": fmt,
        "filename": file.filename or "resume",
        "page_count": page_count,
        "links": extract_pdf_links(content) if fmt == "pdf" else {},
    }
    return {"resume_id": resume_id, "filename": file.filename, "text": text, "format": fmt, "page_count": page_count}


@app.post("/api/generate-resume")
def api_generate_resume(req: GenerateResumeRequest):
    """Re-typesets the resume into one clean template (any input format ->
    PDF), so edits and additions are always properly aligned."""
    entry = _RESUME_CACHE.get(req.resume_id) if req.resume_id else None

    added = [b.strip() for b in req.added_bullets if b.strip()]
    lines = req.final_text.split("\n")
    for b in added:  # the frontend also appended these as "- text" for scoring
        if f"- {b}" in lines:
            lines.remove(f"- {b}")

    resume = parse_resume("\n".join(lines))
    if not resume.name:
        raise HTTPException(400, "The resume text is empty.")
    place_bullets(resume, added)

    if entry is not None:
        page_limit = entry["page_count"]
        links = entry["links"]
        base_name = entry["filename"].rsplit(".", 1)[0] if "." in entry["filename"] else entry["filename"]
        note = f"Kept to {page_limit} page(s) to match your original resume."
    else:
        page_limit = estimate_pages_from_words(req.final_text)
        links = {}
        base_name = "resume"
        note = f"Fitted to {page_limit} page(s)."

    file_bytes, pages, fits = render_pdf(resume, links, page_limit)
    if not fits:
        note = (
            f"Your content runs to {pages} page(s) even at the smallest readable size (limit: {page_limit}). "
            "Shorten or remove a few bullets to bring it back within the limit."
        )

    return {
        "format": "pdf",
        "filename": f"{base_name}-improved.pdf",
        "page_count_estimate": pages,
        "truncated": not fits,
        "note": note,
        "file_base64": base64.b64encode(file_bytes).decode("ascii"),
    }


@app.post("/api/clarify")
def api_clarify(req: ClarifyRequest):
    if not req.answer.strip():
        raise HTTPException(400, "Please describe your experience first.")
    bullet = craft_bullet_from_answer(req.requirement, req.answer)
    return {"suggested_bullet": bullet}


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
