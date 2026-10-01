import re
from dataclasses import dataclass
from typing import List, Optional

from sentence_transformers import SentenceTransformer, util

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


REQUIRED_HINTS = re.compile(r"\b(required|must[- ]have|minimum|at least|essential|mandatory)\b", re.I)
PREFERRED_HINTS = re.compile(r"\b(preferred|nice[- ]to[- ]have|plus|bonus|desirable|is a plus)\b", re.I)

BULLET_PREFIX = re.compile(r"^\s*[-*•‣▪●○•]+\s*|^\s*\d+[.)]\s*")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

STOPWORDS = set(
    "a an the and or of to in for with on at by from as is are be will you your our we "
    "experience years year knowledge ability strong excellent working proven demonstrated "
    "including etc this that have has having skills skill role team work good great".split()
)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.]{1,}")

FOCUS_STRIP = re.compile(
    r"^(\d+\+?\s*years?(\s+of)?\s+experience( with| in)?|experience( with| in)?|proficiency in|"
    r"ability to|strong knowledge of|knowledge of|"
    r"familiarity with|skilled in|proven ability to|demonstrated experience( with| in)?|"
    r"hands[- ]on experience( with| in)?)\s+",
    re.I,
)
FOCUS_SUFFIX_STRIP = re.compile(
    r"\s*(is a plus|is a bonus|is preferred|is desired|is a must|is required|is mandatory|is essential)\.?$",
    re.I,
)
FOCUS_LABEL_STRIP = re.compile(
    r"^(required|preferred|must[- ]have|nice[- ]to[- ]have|essential|mandatory|bonus)\s*:\s*",
    re.I,
)


# General guideline threshold for "likely to clear an ATS keyword screen / get shortlisted".
# Not any specific employer's real cutoff — just a reasonable, visible bar to improve against.
TARGET_PERCENT = 75


def _clean_line(line: str) -> str:
    return BULLET_PREFIX.sub("", line.strip()).strip(" \t-•")


def split_into_fragments(text: str, min_words: int = 3) -> List[str]:
    text = text.replace("\r\n", "\n")
    raw_lines = [l for l in text.split("\n") if l.strip()]
    first_bullet_idx = next((i for i, l in enumerate(raw_lines) if BULLET_PREFIX.match(l)), None)
    fragments: List[str] = []
    for i, line in enumerate(raw_lines):
        # skip leading prose/intro paragraphs that precede the first bulleted item
        if first_bullet_idx is not None and i < first_bullet_idx and not BULLET_PREFIX.match(line):
            continue
        cleaned = _clean_line(line)
        if not cleaned:
            continue
        if len(cleaned.split()) <= 3 and (cleaned.endswith(":") or cleaned.isupper()):
            continue
        parts = SENTENCE_SPLIT.split(cleaned) if len(cleaned.split()) > 25 else [cleaned]
        for p in parts:
            p = p.strip()
            if len(p.split()) >= min_words:
                fragments.append(p)
    seen = set()
    unique = []
    for f in fragments:
        key = f.lower()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def requirement_weight(text: str) -> float:
    if PREFERRED_HINTS.search(text):
        return 1.0
    if REQUIRED_HINTS.search(text):
        return 2.0
    return 1.5


def importance_label(weight: float) -> str:
    if weight == 2.0:
        return "Required"
    if weight == 1.0:
        return "Preferred"
    return "Standard"


def keywords(text: str) -> set:
    return {w.lower() for w in WORD_RE.findall(text) if w.lower() not in STOPWORDS and len(w) > 1}


def keyword_overlap(req: str, resume_text: str) -> float:
    req_kw = keywords(req)
    if not req_kw:
        return 0.0
    resume_kw = keywords(resume_text)
    hit = sum(1 for k in req_kw if k in resume_kw)
    return hit / len(req_kw)


def extract_focus_phrase(requirement: str) -> str:
    """Returns the requirement's core phrase with leading filler (e.g. '3+ years of
    experience with') stripped, preserving original casing so acronyms like AWS or
    REST stay readable when the phrase is inlined mid-sentence."""
    req = requirement.strip().rstrip(".")
    req = FOCUS_LABEL_STRIP.sub("", req, count=1).strip()
    req = FOCUS_STRIP.sub("", req, count=1).strip()
    req = FOCUS_SUFFIX_STRIP.sub("", req).strip()
    return req if req else requirement.strip().rstrip(".")


@dataclass
class RequirementResult:
    requirement: str
    status: str
    evidence: Optional[str]
    similarity: float
    weight: float
    importance: str


def classify(combined: float) -> str:
    if combined >= 0.60:
        return "Match"
    if combined >= 0.35:
        return "Unclear"
    return "Not found"


def analyze(resume_text: str, jd_text: str) -> dict:
    model = get_model()
    requirements = split_into_fragments(jd_text, min_words=3)
    resume_sentences = split_into_fragments(resume_text, min_words=3)

    if not requirements:
        raise ValueError("Could not find any requirements in the job description. Try adding more detail.")
    if not resume_sentences:
        raise ValueError("Could not find any content in the resume. Try adding more detail.")

    req_emb = model.encode(requirements, convert_to_tensor=True, normalize_embeddings=True)
    res_emb = model.encode(resume_sentences, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(req_emb, res_emb)

    results: List[RequirementResult] = []
    for i, req in enumerate(requirements):
        row = sims[i]
        best_idx = int(row.argmax())
        cos = max(float(row[best_idx]), 0.0)
        evidence = resume_sentences[best_idx]
        kw = keyword_overlap(req, resume_text)
        combined = round(0.7 * cos + 0.3 * kw, 3)
        status = classify(combined)
        if status == "Not found" and cos < 0.2:
            evidence = None
        weight = requirement_weight(req)
        results.append(RequirementResult(req, status, evidence, combined, weight, importance_label(weight)))

    score = _compute_score(results)
    swot = _build_swot(results)
    suggestions = [_build_suggestion(r) for r in results if r.status != "Match"]

    return {
        "score": score,
        "requirements": [
            {
                "requirement": r.requirement,
                "status": r.status,
                "evidence": r.evidence,
                "similarity": r.similarity,
                "importance": r.importance,
            }
            for r in results
        ],
        "swot": swot,
        "suggestions": suggestions,
    }


def _compute_score(results: List[RequirementResult]) -> dict:
    value_map = {"Match": 1.0, "Unclear": 0.5, "Not found": 0.0}
    total_weight = sum(r.weight for r in results) or 1.0
    weighted = sum(r.weight * value_map[r.status] for r in results)
    pct = round((weighted / total_weight) * 100)
    counts = {"Match": 0, "Unclear": 0, "Not found": 0}
    for r in results:
        counts[r.status] += 1
    return {
        "percent": pct,
        "counts": counts,
        "total_requirements": len(results),
        "target_percent": TARGET_PERCENT,
        "meets_target": pct >= TARGET_PERCENT,
        "methodology": (
            "Each requirement is scored Match (1.0), Unclear (0.5) or Not found (0.0) based on semantic "
            "similarity (MiniLM sentence embeddings) plus keyword overlap with your resume. Requirements "
            "flagged as 'required/must-have' count double, 'preferred/nice-to-have' count at normal weight, "
            "and everything else counts at 1.5x. This is a heuristic estimate to guide edits — it is not the "
            "same as any specific employer's real ATS scoring system."
        ),
        "target_note": (
            f"{TARGET_PERCENT}% is a general guideline for a resume that's likely to clear an ATS keyword "
            "screen — not any specific employer's actual cutoff. Aim to get your projected score above it "
            "as you accept suggestions below."
        ),
    }


def _fmt(r: RequirementResult) -> dict:
    return {
        "requirement": r.requirement,
        "evidence": r.evidence,
        "similarity": r.similarity,
        "importance": r.importance,
    }


def _build_swot(results: List[RequirementResult]) -> dict:
    strengths = [r for r in results if r.status == "Match"]
    weaknesses = [r for r in results if r.status in ("Unclear", "Not found")]
    opportunities = [r for r in results if r.status == "Unclear"]
    threats = [r for r in results if r.status == "Not found" and r.importance == "Required"]
    return {
        "strengths": [_fmt(r) for r in strengths],
        "weaknesses": [_fmt(r) for r in weaknesses],
        "opportunities": [_fmt(r) for r in opportunities],
        "threats": [_fmt(r) for r in threats],
    }


def _build_suggestion(r: RequirementResult) -> dict:
    focus = extract_focus_phrase(r.requirement)
    if r.status == "Not found":
        explain = (
            f"This requirement wasn't found anywhere in your resume"
            + (f" (closest content was only {int(r.similarity * 100)}% related)." if r.evidence is None else ".")
            + " Either you haven't mentioned this experience, it's a genuine skill gap, or your resume "
            "describes it with different wording."
        )
        ask = f"Have you worked with {focus} in any job, project, coursework, or personal project? Describe what you did and the result."
        improve = None
    else:
        explain = (
            f'Your resume has related content ("{r.evidence}") but it does not clearly call out '
            f"{focus}. An ATS keyword scan may read this as only a partial match."
        )
        ask = f"Can you confirm hands-on experience with {focus}? Add specifics (tools, scale, outcome) if so."
        improve = {
            "original": r.evidence,
            "suggested": f"{r.evidence.rstrip('.')}, applying {focus} to deliver measurable results.",
        }
    return {
        "requirement": r.requirement,
        "status": r.status,
        "importance": r.importance,
        "explain_gap": explain,
        "ask_me": ask,
        "improve_wording": improve,
    }


VAGUE_ANSWER = re.compile(
    r"^(yes|yeah|yep|y|no|nope|ok|okay|sure|done|i have|i did|i know( it)?|familiar|some|a bit|little|true)\W*$", re.I
)
MIN_ANSWER_WORDS = 4


class VagueAnswer(ValueError):
    pass


def craft_bullet_from_answer(requirement: str, user_answer: str) -> str:
    """Turns the user's own description into a resume bullet. Their words are
    the content, so a bare "yes" can't become a bullet: gluing it onto the
    job-description phrase produced lines like "Built exposure to X — yes."."""
    answer = " ".join(user_answer.split()).strip(" .-•")
    if VAGUE_ANSWER.match(answer) or len(answer.split()) < MIN_ANSWER_WORDS:
        focus = extract_focus_phrase(requirement)
        raise VagueAnswer(
            f"Describe what you actually did with {focus}: what you built, the tools, and the result "
            f"(e.g. \"Split the order service into 3 Spring Boot microservices behind an API gateway\")."
        )
    answer = re.sub(r"^(i|we)\s+", "", answer, flags=re.I)
    return answer[0].upper() + answer[1:] + ("" if answer.endswith((".", "!", "?")) else ".")
