"""Scores a resume against a job description and suggests keyword-only edits.

Each JD requirement gets 0-1 credit from its best-matching resume line
(semantic similarity) plus the share of its keywords the resume contains.
Credit is a smooth ramp, not Match/Partial/Missing steps, so a small wording
change nudges the score instead of jumping it, and the same inputs always
give the same score.

Suggestions never rewrite a sentence. Each one either swaps a resume word for
the JD's word with the same meaning ("developers" -> "engineers") or adds a
missing JD keyword to a Skills line, so it can be applied in place to the
original PDF/DOCX/LaTeX file without disturbing its layout.
"""
import difflib
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# A general "likely to clear an ATS keyword screen" bar, not any employer's real cutoff.
TARGET_PERCENT = 75
MATCH, PARTIAL = 0.60, 0.40  # combined-score cutoffs for the status label
CREDIT_FLOOR = 0.25  # combined score below this earns no credit
WEIGHTS = {"Required": 2.0, "Standard": 1.5, "Preferred": 1.0}
SWAP_SIMILARITY = 0.8  # least embedding similarity for a meaning-based word swap

# ---------------------------------------------------------------- model

_model: Optional[SentenceTransformer] = None
_vectors: Dict[str, np.ndarray] = {}


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(texts: List[str]) -> np.ndarray:
    """Unit vectors, cached per text: a line always gets the same vector, and
    re-scoring an edited resume only encodes the lines that changed."""
    new = [t for t in dict.fromkeys(texts) if t not in _vectors]
    if new:
        if len(_vectors) > 20000:
            _vectors.clear()
        _vectors.update(zip(new, get_model().encode(new, normalize_embeddings=True, convert_to_numpy=True)))
    return np.stack([_vectors[t] for t in texts])


# ---------------------------------------------------------------- text

BULLET = re.compile(r"^\s*(?:[-*•‣▪●○◦■□➢►–]+|\d+[.)])\s*")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9])")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-]*")

def key(word: str) -> str:
    """Comparable form of a word: lowercase, simple plural folded (APIs -> api, mentored -> mentor)."""
    w = word.lower().strip(".-")
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for suffix in ("ing", "ed"):  # mentoring / mentored -> mentor
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            return w[:-len(suffix)]
    return w


# Words that never make a keyword and break a keyword phrase.
FILLER = set("""
    a an the and or of to in for with on at by from as is are be will can you your our we us their
    such like similar etc including include e.g i.e also any other both either within across into via per
    experience experienced year knowledge ability able strong excellent good great solid proven demonstrated
    familiarity familiar understanding proficiency proficient skill hands-on required preferred must nice
    have has having bonus desirable minimum least essential mandatory plus exposure
    specific existing relevant latest large various new updated tailored overall multiple diverse emerging
    innovative cutting-edge state-of-the-art efficient effective appropriate
""".split())
# Words that aren't keywords on their own but can sit inside one ("REST APIs").
GENERIC = set("""
    team environment service tool using use build building develop developing write writing maintain work
    working related deliver delivering ensure support challenge technique principle advancement guidance
    subfield input technology production practice concept area field
""".split())
# Verbs that open responsibility lines ("Design and maintain pipelines").
LEAD_VERBS = set("""
    design build develop own lead manage drive collaborate partner work create implement maintain support write
    deliver ensure define help contribute participate analyze analyse monitor improve optimize architect deploy
    operate run troubleshoot review communicate coordinate establish identify perform provide integrate
    apply stay incorporate preprocess conduct leverage utilize use execute evaluate research translate
""".split())
FILLER |= {key(w) for w in FILLER}
GENERIC |= {key(w) for w in GENERIC}

FOCUS_PREFIX = re.compile(
    r"^((required|preferred|must[- ]have|nice[- ]to[- ]have|essential|mandatory|bonus)\s*:\s*)?"
    r"(\d+\+?\s*years?(\s+of)?\s+)?"
    r"((hands[- ]on|demonstrated|proven|strong|solid|deep)\s+)?"
    r"(experience|proficiency|knowledge|familiarity|expertise|ability|skilled)?\s*(with|in|of|to)?\s+",
    re.I,
)
FOCUS_SUFFIX = re.compile(r"\s*\bis (a plus|a bonus|preferred|desired|a must|required|mandatory|essential)\.?$", re.I)


def normalize(text: str) -> str:
    """One canonical form per input, so copy/paste differences (non-breaking
    spaces, ligatures, doubled spaces, blank lines) can't change the score."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = (re.sub(r" +", " ", l.replace("​", "")).strip() for l in text.split("\n"))
    return "\n".join(l for l in lines if l)


def keywords(text: str) -> set:
    return {k for k in map(key, WORD_RE.findall(text)) if len(k) > 1 and k not in FILLER and k not in GENERIC}


def focus_phrase(requirement: str) -> str:
    """The requirement minus its lead-in: "3+ years of experience with
    Python" -> "Python". Casing is kept so AWS/REST stay readable."""
    req = requirement.strip().rstrip(".")
    core = FOCUS_SUFFIX.sub("", FOCUS_PREFIX.sub("", req, count=1)).strip()
    return core or req


def _fragments(lines: List[str], min_words: int) -> List[str]:
    """Long lines are split into sentences; short ones are kept whole."""
    out = []
    for line in lines:
        parts = SENTENCE_SPLIT.split(line) if len(line.split()) > 25 else [line]
        out += [p.strip(" ;") for p in parts if len(p.split()) >= min_words]
    return list(dict.fromkeys(out))


def resume_lines(resume: str) -> List[str]:
    return _fragments([BULLET.sub("", l) for l in resume.split("\n")], min_words=2)


# ---------------------------------------------------------------- job description

HEAD_SKIP = re.compile(
    r"^(about(?! you| the role)|who we are|our (company|story|mission|culture)|benefits|perks|what we offer|"
    r"why (join|us|work)|compensation|salary|pay|how to apply|equal (opportunity|employment)|eeo|location|"
    r"job type|schedule|work hours)\b", re.I)
# A heading without a trailing colon must be one of these whole phrases, so a
# short requirement such as "Experience with Docker" is never taken for one.
HEAD_KNOWN = re.compile(
    r"^((key|core|basic|minimum|required|preferred|desired|technical|job) )*"
    r"(requirements|qualifications|responsibilities|skills|duties|must[- ]haves?|nice[- ]to[- ]haves?|"
    r"bonus( points)?|about you|about the role|the role|who you are|what you('ll)? (do|bring|need|have)|"
    r"you (will|have)|experience)$", re.I)
HINT_PREFERRED = re.compile(r"\b(preferred|nice[- ]to[- ]have|plus|bonus|desirable|desired)\b", re.I)
HINT_REQUIRED = re.compile(r"\b(required|must[- ]have|must|minimum|at least|essential|mandatory)\b", re.I)


def _is_heading(line: str) -> bool:
    if line.endswith(":"):
        return len(line.split()) <= 6
    return ":" not in line and len(line.split()) <= 5 and bool(HEAD_KNOWN.match(line) or HEAD_SKIP.match(line))


def importance(line: str, heading: str = "") -> str:
    for text in (line, heading):
        if HINT_PREFERRED.search(text):
            return "Preferred"
        if HINT_REQUIRED.search(text) and "requirement" not in text.lower():
            return "Required"
    return "Standard"


def jd_requirements(jd: str) -> List[Tuple[str, str]]:
    """(requirement, importance) pairs. Skips the intro before the first
    heading or bullet, and sections that aren't requirements (About us,
    Benefits, How to apply), whether or not the bullets survived the paste."""
    lines = jd.split("\n")
    structured = any(BULLET.match(l) or _is_heading(l) for l in lines)
    started, skipping, heading = not structured, False, ""
    picked: List[Tuple[str, str]] = []
    for raw in lines:
        bulleted = bool(BULLET.match(raw))
        line = BULLET.sub("", raw)
        if not bulleted and _is_heading(line):
            started, heading = True, line.rstrip(":")
            skipping = bool(HEAD_SKIP.match(line))
            continue
        started = started or bulleted
        label = line.split(":", 1)[0]
        if not started or skipping or (":" in line and HEAD_SKIP.match(label)):
            continue
        for frag in _fragments([line], min_words=1):  # "Terraform" alone is a requirement
            if keywords(frag):
                picked.append((frag, importance(frag, heading)))
    return list(dict.fromkeys(picked))




# ---------------------------------------------------------------- keyword terms

# A word, or a short slash compound kept whole ("CI/CD", "TCP/IP"); a long one
# ("Python/Django") is two words.
TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-]*(?:/[A-Za-z][A-Za-z0-9+#.\-]{0,3}(?![A-Za-z0-9]))*")


def _word_keys(word: str) -> List[str]:
    return [key(p) for p in word.split("/") if p.strip(".-")]


def term_keys(phrase: str) -> set:
    return {k for w in TERM_RE.findall(phrase) for k in _word_keys(w)}


def phrase_key(phrase: str) -> str:
    return " ".join(key(w) for w in TERM_RE.findall(phrase))


def is_technical(phrase: str) -> bool:
    """A tool or acronym ("Docker", "REST", "C++"), not a plain English word."""
    return any(w[:1].isupper() or re.search(r"[A-Z].*[A-Z]|\d|[+#/]|\.\w", w) for w in TERM_RE.findall(phrase))


def _present(phrase: str, have: set) -> bool:
    ks = [k for k in term_keys(phrase) if k not in GENERIC]
    return bool(ks) and all(k in have for k in ks)


def missing_terms(requirement: str, have: set) -> List[Tuple[str, bool]]:
    """Keyword phrases from the requirement that `have` lacks, as
    (phrase, is_technical). Phrases stay whole ("REST APIs", not "REST",
    "APIs"); technical means a tool or acronym rather than a soft skill.
    With an empty `have` this lists every keyword phrase of the requirement."""
    focus = focus_phrase(requirement)
    spans: List[List[Tuple[str, bool]]] = [[]]
    last_end = 0
    for i, m in enumerate(TERM_RE.finditer(focus)):
        word = m.group().strip(".-")
        # A verb opening a phrase isn't part of the keyword: "Integrate AI
        # solutions" -> "AI solutions", "deploying AI models" -> "AI models".
        opens_phrase = not spans[-1] or bool(focus[last_end:m.start()].strip())
        filler = (key(word) in FILLER or word.lower() in FILLER
                  or (opens_phrase and _verb_base(word) is not None and not word.isupper())
                  or (len(word) == 1 and not word.isupper()))  # the "s" of "Bachelor's"
        if focus[last_end:m.start()].strip() or filler:
            spans.append([])  # punctuation or a filler word ends the phrase
        if not filler:
            # A capital on the first word of a sentence is just sentence case.
            sentence_case = i == 0 and requirement.strip().startswith(focus) and " " in focus
            technical = (word[:1].isupper() and not sentence_case) or bool(re.search(r"[A-Z].*[A-Z]|\d|[+#/]", word))
            spans[-1].append((word, technical))
        last_end = m.end()

    terms = []
    for span in spans:
        while span and key(span[0][0]) in GENERIC:
            span = span[1:]
        while span and key(span[-1][0]) in GENERIC:
            span = span[:-1]
        if span and any(k not in have and k not in GENERIC for w, _ in span for k in _word_keys(w)):
            phrase = " ".join(w for w, _ in span)
            if phrase.lower() not in (t.lower() for t, _ in terms):
                terms.append((phrase, any(t for _, t in span)))
    return terms


# Same meaning, different wording. Swapping one of these for the JD's word is
# always safe; anything else is judged by embedding similarity.
SYNONYMS = [
    "javascript|js", "typescript|ts", "postgresql|postgres|psql", "kubernetes|k8s",
    "aws|amazon web services", "gcp|google cloud|google cloud platform", "azure|microsoft azure",
    "machine learning|ml", "artificial intelligence|ai", "natural language processing|nlp", "deep learning|dl",
    "large language models|large language model|llms|llm", "generative ai|genai|gen ai",
    "node.js|nodejs|node", "react|react.js|reactjs", "vue|vue.js|vuejs", "next.js|nextjs", "express|express.js",
    "angular|angularjs|angular.js", "mongodb|mongo", "c#|csharp", "c++|cpp", ".net|dotnet",
    "object-oriented programming|oop", "object-oriented design|ood", "ci/cd|cicd|ci-cd",
    "continuous integration|ci", "rest|restful", "rest api|restful api|rest apis|restful apis",
    "user interface|ui", "user experience|ux", "quality assurance|qa", "engineer|developer|programmer",
    "software engineer|software developer|sde|swe", "database|db", "sql server|mssql|microsoft sql server",
    "version control|source control", "automated testing|test automation|automated tests",
    "unit testing|unit tests", "teamwork|collaboration", "mentoring|coaching", "microservices|micro-services",
    "html|html5", "css|css3", "scikit-learn|sklearn", "github actions|gh actions",
    "data structures and algorithms|dsa", "power bi|powerbi", "full stack|full-stack|fullstack",
    "front end|front-end|frontend", "back end|back-end|backend",
]
_SYNONYM_GROUP: Dict[str, int] = {}
for _i, _group in enumerate(SYNONYMS):
    for _form in _group.split("|"):
        _SYNONYM_GROUP[phrase_key(_form) or _form] = _i


def _synonym_group(phrase: str) -> Optional[int]:
    return _SYNONYM_GROUP.get(phrase_key(phrase) or phrase.lower())


CONTACT_LINE = re.compile(r"@|https?://|www\.|linkedin|github\.com|\+?\d[\d ()-]{7,}\d", re.I)


def resume_terms(lines: List[str]) -> Dict[str, List[int]]:
    """Every 1-3 word keyword phrase in the resume, as written -> the line
    indexes it appears on. These are the words a swap may replace."""
    out: Dict[str, List[int]] = {}
    for idx, line in enumerate(lines):
        if idx == 0 or CONTACT_LINE.search(line):
            continue  # the name and contact details are never swapped
        toks = []
        for m in TERM_RE.finditer(line):
            w = m.group().rstrip(".-")
            toks.append((w, m.start(), m.start() + len(w)))
        for i in range(len(toks)):
            for n in (1, 2, 3):
                span = toks[i:i + n]
                if len(span) < n:
                    break
                words = [w for w, _, _ in span]
                if any(key(w) in FILLER or w.lower() in FILLER for w in words):
                    break
                if any(line[a[2]:b[1]].strip() for a, b in zip(span, span[1:])):
                    break  # punctuation between the words
                if key(words[0]) in GENERIC or key(words[-1]) in GENERIC or len(words[-1]) < 2:
                    continue
                phrase = line[span[0][1]:span[-1][2]]
                if idx not in out.setdefault(phrase, []):
                    out[phrase].append(idx)
    return out


def skills_lines(lines: List[str]) -> List[int]:
    """Indexes of the lines under a Skills-type heading."""
    from resume_template import _is_section_title, _is_skills

    out, in_skills = [], False
    for i, line in enumerate(lines):
        bare = BULLET.sub("", line).strip()
        if i and _is_section_title(bare):
            in_skills = _is_skills(bare)
        elif in_skills and (":" in bare or sum(bare.count(s) for s in LIST_SEPARATORS) >= 2):
            out.append(i)  # a list ("Tools: Git, Jira"), not a sentence under the heading
    return out


# Reference skills for judging whether a JD phrase is a real, listable skill
# ("model evaluation", "TensorFlow") rather than wording ("quality input",
# "specific business challenges"): a phrase must be one of these or close to
# one in meaning (sentence embeddings).
SKILL_VOCAB = """
python java javascript typescript c c++ c# go rust kotlin swift scala r sql nosql bash html css php ruby matlab
react angular vue node.js express next.js django flask fastapi spring boot spring .net graphql rest apis
microservices system design software design software engineering object-oriented programming data structures
algorithms design patterns unit testing automated testing test automation tdd ci/cd devops git github gitlab
docker kubernetes terraform ansible jenkins aws azure gcp cloud computing serverless linux networking
postgresql mysql mongodb redis elasticsearch kafka spark hadoop airflow etl data pipelines data warehousing
snowflake bigquery tableau power bi excel data analysis data visualization statistics a/b testing
machine learning deep learning artificial intelligence natural language processing computer vision
large language models generative ai prompt engineering rag transformers neural networks reinforcement learning
model training model evaluation model deployment mlops feature engineering data preprocessing data cleaning
hyperparameter tuning tensorflow pytorch keras scikit-learn pandas numpy opencv hugging face langchain
llm fine-tuning time series forecasting recommendation systems anomaly detection classification regression
cybersecurity penetration testing vulnerability assessment siem soc network security cryptography owasp
ui/ux design figma wireframing prototyping user research accessibility responsive design
agile scrum kanban jira project management product management stakeholder management requirements gathering
technical documentation code review mentoring leadership communication teamwork problem solving
""".strip()
SKILL_SIMILARITY = 0.72
_skill_vecs: Optional[np.ndarray] = None


def _skill_list() -> List[str]:
    """The vocabulary as phrases: multi-word skills are kept together by
    matching the longest known phrase at each position."""
    words = SKILL_VOCAB.split()
    out, i = [], 0
    known = {"spring boot", "system design", "software design", "software engineering", "object-oriented programming",
             "data structures", "design patterns", "unit testing", "automated testing", "test automation",
             "cloud computing", "data pipelines", "data warehousing", "power bi", "data analysis", "data visualization",
             "a/b testing", "machine learning", "deep learning", "artificial intelligence", "natural language processing",
             "computer vision", "large language models", "generative ai", "prompt engineering", "neural networks",
             "reinforcement learning", "model training", "model evaluation", "model deployment", "feature engineering",
             "data preprocessing", "data cleaning", "hyperparameter tuning", "hugging face", "llm fine-tuning",
             "time series forecasting", "recommendation systems", "anomaly detection", "penetration testing",
             "vulnerability assessment", "network security", "ui/ux design", "user research", "responsive design",
             "project management", "product management", "stakeholder management", "requirements gathering",
             "technical documentation", "code review", "problem solving", "rest apis"}
    while i < len(words):
        for n in (3, 2, 1):
            phrase = " ".join(words[i:i + n])
            if n == 1 or phrase in known:
                out.append(phrase)
                i += n
                break
    return out


SKILL_PHRASES = _skill_list()
_SKILL_KEYS = {phrase_key(p) for p in SKILL_PHRASES}


def is_skill(term: str) -> bool:
    """Whether `term` is something you'd list under Skills: a known skill, a
    tool-shaped name (acronym, "Node.js", "YOLOv8"), or close in meaning to a
    known skill. Phrases containing a verb are wording, not skills."""
    global _skill_vecs
    if phrase_key(term) in _SKILL_KEYS:
        return True
    words = term.split()
    if not words or (_verb_base(words[0]) and not words[0].isupper()):
        return False
    if len(words) == 1 and re.search(r"[A-Z].*[A-Z]|\d|[+#./]", term):
        return True  # AWS, PyTorch, YOLOv8, Node.js, C++
    if _skill_vecs is None:
        _skill_vecs = embed(SKILL_PHRASES)
    return float((_skill_vecs @ embed([term])[0]).max()) >= SKILL_SIMILARITY


SOFT_SKILLS_LABEL = re.compile(r"soft|interpersonal|personal|competenc|strength", re.I)


def _list_case(term: str, line: str) -> str:
    """"relational databases" -> "Relational databases" on a list whose items are capitalised."""
    items = [w for w in re.split(r"[,;|·•:]\s*", line) if w.strip()][1:]
    if term[:1].islower() and items and sum(i.strip()[:1].isupper() for i in items) * 2 > len(items):
        return term[:1].upper() + term[1:]
    return term

def find_word(text: str, word: str, flags=0) -> Optional[re.Match]:
    """`word` as a whole word ("C++" in "C++, Java", but not "C" in "C++")."""
    return re.search(r"(?<![\w+#])" + re.escape(word) + r"(?![\w+#])", text, flags)


# ---------------------------------------------------------------- semantic keyword analysis

def _missing_part(term: str, have: set) -> str:
    """The words of a partly-present phrase that are actually missing, when
    they sit together: "mentoring junior engineers" with "mentored" and
    "junior" in the resume -> "engineers". Otherwise the whole phrase."""
    words = term.split()
    missing = [i for i, w in enumerate(words) if any(k not in have and k not in GENERIC for k in _word_keys(w))]
    if missing and len(missing) < len(words) and missing == list(range(missing[0], missing[-1] + 1)):
        return " ".join(words[missing[0]:missing[-1] + 1])
    return term


def _swap_for(term: str, technical: bool, cands: List[str], cand_vecs: Optional[np.ndarray], jd_keys: set):
    """Resume phrases that already mean `term`, as [(phrase, similarity,
    how)]: every synonym found, else the closest safe match. Never trades one tool for another ("MySQL" -> "PostgreSQL" would
    misstate the resume), and never removes a word the JD also asks for."""
    tkeys = term_keys(term)
    group = _synonym_group(term)
    if group is not None:
        found = [(c, 1.0, "synonym") for c in cands if _synonym_group(c) == group and phrase_key(c) != phrase_key(term)]
        if found:
            return found
    if cand_vecs is None:
        return []
    sims = cand_vecs @ embed([term])[0]
    for idx in np.argsort(-sims)[:15]:
        c, s = cands[idx], float(sims[idx])
        ckeys = term_keys(c)
        # "CI" -> "CI/CD", "APIs" -> "REST APIs": the JD's fuller form of the same word.
        variant = ckeys < tkeys or (term.lower().startswith(c.lower()) and not term[len(c):len(c) + 1].isalpha())
        if s < (0.45 if variant else SWAP_SIMILARITY):
            continue
        if ckeys & (jd_keys - tkeys) or all(k in GENERIC for k in ckeys):
            continue
        if len(c.split()) > len(term.split()) + 1:
            continue
        # A loose match swaps like for like ("unit tests" -> "automated
        # tests"), never a short word for a long phrase that won't read right.
        if not variant and (len(c.split()) != len(term.split()) or len(term.split()) > 2):
            continue
        # Only a variant may be swapped for a tool name: a plain word
        # ("containers") doesn't prove the tool ("Docker"), and one tool
        # ("MySQL") is never another.
        if not variant and (technical or is_technical(c)):
            continue
        return [(c, s, "variant" if variant else "similar")]
    return []


def _swap_text(old: str, term: str) -> str:
    """The JD's word, fitted to where it goes: plural only if the replaced
    word was ("developers" -> "engineers", "Developer" -> "Engineer"), and
    capitalised like it."""
    new = term
    if " " not in new and not is_technical(new) and not is_technical(old):
        old_plural, new_plural = key(old) != old.lower().strip(".-"), key(new) != new.lower()
        if old_plural and not new_plural:
            new += "s"
        elif new_plural and not old_plural and new.endswith("s"):
            new = new[:-1]
    if new[:1].islower() and old[:1].isupper():
        new = new[:1].upper() + new[1:]
    return new


# Plain-language "why" for each swap; {old} is your word, {new} the JD's.
SWAP_REASONS = {
    "synonym": "Same thing, different word. The JD says “{new}” and ATS filters count exact words, "
               "so “{old}” may not be matched.",
    "variant": "The JD uses the fuller name “{new}”. ATS filters match exact words, so "
               "“{old}” alone may not count.",
    "similar": "Closest wording in your resume to the JD's “{new}”. Swap only if it still describes "
               "what you did.",
}


def keyword_analysis(lines: List[str], reqs: List[Tuple[str, str]], results: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Maps every JD keyword to the resume: found as written, found by
    meaning (with a swap to the JD's word), or missing (with an addition to a
    Skills line for tools). Returns (keyword map, keyword edits)."""
    have = keywords("\n".join(lines))
    jd_keys = keywords(" ".join(r for r, _ in reqs))
    cand_lines = resume_terms(lines)
    cands = list(cand_lines)
    cand_vecs = embed(cands) if cands else None
    line_vecs = embed([BULLET.sub("", l) or l for l in lines])
    skills = skills_lines(lines)

    terms: Dict[str, dict] = {}
    for r in results:
        for phrase, tech in missing_terms(r["requirement"], set()):
            t = terms.setdefault(phrase.lower(), {"term": phrase, "technical": tech, "requirements": [],
                                                   "importance": r["importance"]})
            t["requirements"].append(r["requirement"])
            if WEIGHTS[r["importance"]] > WEIGHTS[t["importance"]]:
                t["importance"] = r["importance"]

    kmap, edits = [], []
    for t in terms.values():
        entry = {k: t[k] for k in ("term", "technical", "importance", "requirements")}
        entry.update(status="exact", resume_term=None, similarity=1.0)
        kmap.append(entry)
        if _present(t["term"], have):
            continue
        part = _missing_part(t["term"], have)
        swap = _swap_for(part, is_technical(part) if part != t["term"] else t["technical"], cands, cand_vecs, jd_keys)
        if swap:
            # The occurrence on the line closest to the requirement: the
            # "junior developers" bullet, not the "Software Developer" title.
            req_vec = embed([t["requirements"][0]])[0]
            (old, sim, how), idx = max(((c, i) for c in swap for i in cand_lines[c[0]]),
                                       key=lambda ci: float(line_vecs[ci[1]] @ req_vec))
            m = find_word(lines[idx], old)
            if m is not None:
                entry.update(status="semantic", resume_term=old, similarity=round(sim, 3))
                edits.append({
                    "kind": "replace", "term": part, "old": old,
                    "new": _swap_text(old, part),
                    "line_index": idx, "line": lines[idx], "confidence": "high" if how != "similar" else "medium",
                    "reason": SWAP_REASONS[how].format(old=old, new=part), "similarity": round(sim, 3),
                    "requirements": t["requirements"], "importance": t["importance"],
                })
                continue
        entry.update(status="missing", similarity=0.0)
        # Tools and concepts ("relational databases") go on the Skills line
        # they fit best, judged with the requirement as context so "Lambda"
        # lands with the cloud tools; a lone soft word only on a soft-skills line.
        soft = [i for i in skills if SOFT_SKILLS_LABEL.search(lines[i].split(":", 1)[0])]
        targets = skills if t["technical"] or len(t["term"].split()) > 1 else soft
        if len(t["term"].split()) <= 3 and targets and is_skill(t["term"]):
            term_vec = embed([f"{t['term']} ({t['requirements'][0]})"])[0]
            idx = max(targets, key=lambda i: float(line_vecs[i] @ term_vec))
            edits.append({
                "kind": "add", "term": t["term"], "old": "", "new": _list_case(t["term"], lines[idx]), "line_index": idx,
                "line": lines[idx], "confidence": "check",
                "reason": f"“{t['term']}” is in the JD but nowhere in your resume. Listing it in your "
                          "skills lets the ATS find it. Add it only if you have used it.",
                "similarity": 0.0, "requirements": t["requirements"], "importance": t["importance"],
            })

    # Two JD terms can pick the same resume word; keep the first swap only.
    seen, unique = set(), []
    for e in edits:
        k = (e["line_index"], (e["old"] if e["kind"] == "replace" else e["term"]).lower())
        if k not in seen:
            seen.add(k)
            unique.append(e)
    for i, e in enumerate(unique):
        e["id"] = i
    return kmap, unique


# ---------------------------------------------------------------- short sentences

VERBS = LEAD_VERBS | set("""
    mentor coach train teach test automate integrate migrate scale ship launch document debug research
    present plan communicate write build run use apply configure containerize orchestrate refactor secure
    publish visualize visualise tune measure scope prototype negotiate
""".split())
IRREGULAR_PAST = {
    "build": "built", "write": "wrote", "lead": "led", "run": "ran", "drive": "drove", "make": "made",
    "take": "took", "teach": "taught", "bring": "brought", "hold": "held", "keep": "kept", "set": "set",
    "put": "put", "ship": "shipped", "plan": "planned", "own": "owned", "spend": "spent", "win": "won",
}
# Templates for a requirement that is only nouns ("CI/CD pipelines"):
# the first pattern found in it picks the verb.
NOUN_TEMPLATES = [
    (r"\btest", "Wrote {n}"),
    (r"pipeline|ci/cd|\bci\b|workflow", "Built {n}"),
    (r"architect|system design|distributed", "Designed {n}"),
    (r"cach", "Implemented {n}"),
    (r"database|sql|postgres|mysql|mongo|nosql|redis", "Designed and queried {n}"),
    (r"kubernetes|k8s|docker|container|cloud|aws|azure|gcp|serverless|terraform|lambda", "Deployed services on {n}"),
    (r"\bapi|microservice|backend|frontend|web", "Built {n}"),
    (r"framework|librar", "Built models with {n}"),
    (r"machine learning|deep learning|natural language|computer vision|\bnlp\b|\bai\b|\bml\b|llm|neural",
     "Applied {n} in projects"),
    (r"dashboard|report|visuali", "Built {n} for stakeholders"),
    (r"\bdata\b|analytic|statistic", "Analyzed data using {n}"),
    (r"agile|scrum|kanban", "Delivered features in {n} sprints"),
]
PART_LEAD = re.compile(
    r"^(?:(?:the )?ability to|able to|exposure to|background in|track record of|(?:(?:hands[- ]on|proven|strong|solid|demonstrated|excellent|good|deep)\s+)*"
    r"(?:experience|knowledge|proficiency|familiarity|expertise|understanding|skills?)\s+(?:with|in|of|using|to)"
    r"|(?:hands[- ]on|proven|strong|solid|demonstrated|excellent|good|deep))\s+", re.I)
MAX_SENTENCE_WORDS = 22
# Requirements a bullet can't claim: degrees, credentials, eligibility.
NOT_A_BULLET = re.compile(r"\b(degree|bachelor'?s?|master'?s?|ph\.?d|diploma|certifi\w*|licen[cs]e|citizen\w*|"
                          r"clearance|authori[sz]ed to work|visa|relocat\w*|travel)\b", re.I)


def _past(verb: str) -> str:
    v = verb.lower()
    if v in IRREGULAR_PAST:
        return IRREGULAR_PAST[v]
    if v.endswith("e"):
        return v + "d"
    if v.endswith("y") and v[-2:-1] not in "aeiou":
        return v[:-1] + "ied"
    return v + "ed"


def _verb_base(word: str) -> Optional[str]:
    """"build" -> "build", "building" -> "build", "writing" -> "write"; None for a noun."""
    w = word.lower()
    if w in VERBS:
        return w
    if w.endswith("ing") and len(w) > 5:
        stem = w[:-3]
        for cand in (stem, stem + "e", stem[:-1] if stem[-1:] == stem[-2:-1] else ""):
            if cand in VERBS:
                return cand
    return None


def _split_parts(text: str) -> List[str]:
    """Splits on commas and "and"/"or", but not inside parentheses."""
    groups: List[str] = []

    def protect(m: re.Match) -> str:
        groups.append(m.group())
        return f"\x00{len(groups) - 1}\x00"

    protected = re.sub(r"\([^()]*\)", protect, text)
    parts = re.split(r",\s*(?:and\s+|or\s+)?|\s+(?:and|or|&)\s+", protected)
    return [re.sub(r"\x00(\d+)\x00", lambda m: groups[int(m.group(1))], p).strip(" .;") for p in parts if p.strip(" .;")]


def suggest_sentence(requirement: str) -> Optional[str]:
    """A short past-tense bullet that states the requirement in the JD's own
    words: "Python and building REST APIs" -> "Built REST APIs with Python.";
    "Docker and containerized deployments" -> "Used Docker for containerized
    deployments."; "Design and maintain data pipelines" -> "Designed and
    maintained data pipelines." None if the requirement doesn't fit a bullet."""
    if NOT_A_BULLET.search(requirement):
        return None
    focus = focus_phrase(requirement)
    focus = re.sub(r",?\s*(?:or|and)\s+(?:similar|equivalent|other|related|comparable)\b[^)]*", "", focus, flags=re.I)
    focus = re.sub(r"\s+(?:such as|e\.g\.,?|including|like)\s+(.+)$", r" (\1)", focus, flags=re.I)
    focus = re.sub(r"\(\s*\)", "", focus).strip(" .")

    # Each clause is a verb phrase plus the nouns listed after it, which share
    # its verb: "Apply data preprocessing, model training and model
    # evaluation" -> one clause with three objects.
    clauses: List[List[str]] = []
    nouns: List[str] = []
    pending_verb = None
    for part in _split_parts(focus):
        while PART_LEAD.match(part):
            part = PART_LEAD.sub("", part, count=1)
        if part.lower().startswith("to "):
            part = part[3:]
        words = part.split()
        if not words:
            continue
        base = _verb_base(words[0])
        if base is None:
            if clauses:
                clauses[-1].append(part)
            else:
                nouns.append(part)
        elif len(words) == 1:
            pending_verb = _past(base)  # "Design and maintain X": the first verb waits for the second
        else:
            clause = _past(base) + " " + " ".join(words[1:])
            clauses.append([f"{pending_verb} and {clause}" if pending_verb else clause])
            pending_verb = None

    if clauses:
        sentence = " and ".join(join_terms(c) for c in clauses)
        tech = [n for n in nouns if is_technical(n)]
        soft = [n for n in nouns if not is_technical(n)]
        if tech:
            sentence += " with " + join_terms(tech)
        if soft:
            sentence += ", applying " + join_terms(soft)
    elif not nouns:
        return None
    elif len(nouns) == 2 and is_technical(nouns[0]) and not is_technical(nouns[1]):
        sentence = f"Used {nouns[0]} for {nouns[1]}"
    elif len(nouns) == 2 and not is_technical(nouns[0]) and is_technical(nouns[1]):
        sentence = f"Built {nouns[0]} with {nouns[1]}"  # "microservices architecture and Kafka"
    else:
        joined = join_terms(nouns)
        template = next((t for pat, t in NOUN_TEMPLATES if re.search(pat, joined, re.I)),
                        "Built features with {n}" if is_technical(joined) else "Demonstrated {n} in team projects")
        sentence = template.format(n=joined)
    sentence = sentence[:1].upper() + sentence[1:] + "."
    return sentence if len(sentence.split()) <= MAX_SENTENCE_WORDS else None


def join_terms(terms: List[str]) -> str:
    return terms[0] if len(terms) == 1 else ", ".join(terms[:-1]) + " and " + terms[-1]


def sentence_edits(lines: List[str], results: List[dict], first_id: int) -> List[dict]:
    """A short bullet for each requirement the resume doesn't meet, placed
    after the existing bullet it relates to most."""
    from resume_template import ENTRY_SECTION_HINTS, _is_section_title

    # Only bullets under Experience/Projects-type headings: a new achievement
    # never goes under Certifications or Skills.
    skills = set(skills_lines(lines))
    bullets, in_entries = [], False
    for i, l in enumerate(lines):
        bare = BULLET.sub("", l).strip()
        if i and _is_section_title(bare):
            in_entries = any(h in re.sub(r"[^a-z]", "", bare.lower()) for h in ENTRY_SECTION_HINTS)
        elif in_entries and BULLET.match(l) and i not in skills and not CONTACT_LINE.search(l):
            bullets.append(i)
    if not bullets:  # no recognised sections: any bullet that isn't a skill list
        bullets = [i for i, l in enumerate(lines) if BULLET.match(l) and i not in skills and not CONTACT_LINE.search(l)]
    # Follow the resume's own convention for a closing full stop.
    periods = sum(lines[i].rstrip().endswith(".") for i in bullets)
    use_period = periods * 2 >= len(bullets) if bullets else True
    out, seen = [], set()
    for r in results:
        if r["status"] == "Match":
            continue
        sentence = suggest_sentence(r["requirement"])
        if not sentence or sentence.lower() in seen:
            continue
        seen.add(sentence.lower())
        if not use_period:
            sentence = sentence.rstrip(".")
        if bullets:
            sims = embed([BULLET.sub("", lines[i]) for i in bullets]) @ embed([sentence])[0]
            anchor = bullets[int(sims.argmax())]
        else:
            anchor = len(lines) - 1
        out.append({
            "kind": "sentence", "term": focus_phrase(r["requirement"]), "old": "", "new": sentence,
            "line_index": anchor, "line": lines[anchor], "confidence": "check",
            "reason": f"The JD asks for {join_terms([f'“{t}”' for t in r['missing_terms'][:3]] or [focus_phrase(r['requirement'])])} "
                      "and no line in your resume says it. "
                      "This bullet uses the JD's exact words. Edit it to match what you really did.",
            "similarity": 0.0, "requirements": [r["requirement"]], "importance": r["importance"],
            "id": first_id + len(out),
        })
    return out


# ---------------------------------------------------------------- SWOT

def _short(text: str, n: int = 60) -> str:
    return text if len(text) <= n else text[:n - 1].rsplit(" ", 1)[0] + "…"


def swot(score: dict, results: List[dict], kmap: List[dict], edits: List[dict], plan_pct: int) -> dict:
    """Strengths, weaknesses, opportunities and threats of this resume for
    this JD, each a few short plain-language points."""
    rank = {"Required": 0, "Standard": 1, "Preferred": 2}
    by_imp = sorted(results, key=lambda r: rank[r["importance"]])
    exact = [k for k in kmap if k["status"] == "exact"]
    fixable = {req for e in edits for req in e["requirements"]}

    strengths = [f"Clearly shows: {_short(focus_phrase(r['requirement']))}" for r in by_imp if r["status"] == "Match"][:4]
    if exact:
        strengths.append(f"{len(exact)} of {len(kmap)} JD keywords already appear word for word")
    if score["semantic_percent"] >= 50:
        strengths.append(f"Your experience is close in meaning to the role ({score['semantic_percent']}% semantic match)")

    weaknesses = [f"No sign of {', '.join(r['missing_terms'][:3]) or _short(focus_phrase(r['requirement']), 40)}"
                  f"{' (required)' if r['importance'] == 'Required' else ''}"
                  for r in by_imp if r["status"] == "Missing"][:4]
    semantic_only = [k for k in kmap if k["status"] == "semantic"]
    if semantic_only:
        weaknesses.append("Uses different words than the JD for: " + ", ".join(k["term"] for k in semantic_only[:4]))

    gains = sorted((e for e in edits if e["gain"] > 0), key=lambda e: -e["gain"])
    opportunities = []
    for e in gains[:4]:
        if e["kind"] == "replace":
            opportunities.append(f"Say “{e['new']}” instead of “{e['old']}” (+{e['gain']} pts)")
        elif e["kind"] == "add":
            opportunities.append(f"Add {e['new']} to your skills (+{e['gain']} pts)")
        else:
            opportunities.append(f"Add the bullet “{e['new']}” (+{e['gain']} pts)")
    if gains and plan_pct > score["percent"]:
        opportunities.append(f"The suggested fixes can lift your score from {score['percent']}% to about {plan_pct}%")

    threats = []
    if not score["meets_target"]:
        threats.append(f"At {score['percent']}% you're below the {score['target_percent']}% bar many ATS filters use")
    missing_req = [k["term"] for k in kmap if k["status"] == "missing" and k["importance"] == "Required"]
    if missing_req:
        threats.append("Missing required keywords can auto-reject a resume: " + ", ".join(missing_req[:4]))
    if score["keyword_percent"] < 50 <= score["semantic_percent"]:
        threats.append("ATS reads exact words, not meaning: your experience fits but the wording doesn't")
    unfixable = [r for r in results if r["status"] == "Missing" and r["requirement"] not in fixable]
    if unfixable:
        threats.append(f"{len(unfixable)} requirement(s) need real experience no edit can add")
    return {"strengths": strengths, "weaknesses": weaknesses, "opportunities": opportunities, "threats": threats}


# ---------------------------------------------------------------- applying edits

LIST_SEPARATORS = (", ", " | ", " · ", " • ", "; ")


def _add_op(line: str, terms: List[str]) -> Optional[Tuple[str, str, str]]:
    """(old, new, separator) that appends `terms` to a Skills line using the
    line's own separator: "Tools: Git, Jira" -> "Jira", "Jira, Docker", ", "."""
    body = line.split(":", 1)[1] if ":" in line else BULLET.sub("", line)
    sep = max(LIST_SEPARATORS, key=body.count) if any(s in body for s in LIST_SEPARATORS) else ", "
    last = None
    for last in re.finditer(r"[^,;|·•:\s][^,;|·•:]*", line.rstrip(" .")):
        pass
    if last is None:
        return None
    old = last.group().rstrip()
    return old, old + sep + sep.join(terms), sep


def edit_ops(lines: List[str], edits: List[dict]) -> List[dict]:
    """Turns the chosen edits into ordered operations, each {line_index,
    line (its text before the op), old, new, append, insert, ids}.
    `append` ops replace the *last* occurrence of `old` (the end of a Skills
    line); additions to the same line become one op, after any swaps.
    `insert` ops add `new` as a bullet line after line `line_index`."""
    cur = list(lines)
    ops = []
    by_line: Dict[int, List[dict]] = {}
    for e in edits:
        by_line.setdefault(e["line_index"], []).append(e)
    for idx in sorted(by_line):
        group = by_line[idx]
        if not 0 <= idx < len(cur):
            continue
        for e in (e for e in group if e["kind"] == "replace"):
            m = find_word(cur[idx], e["old"])
            if m:
                ops.append({"line_index": idx, "line": cur[idx], "old": e["old"], "new": e["new"],
                            "append": False, "insert": False, "ids": [e["id"]]})
                cur[idx] = cur[idx][:m.start()] + e["new"] + cur[idx][m.end():]
        adds = [e for e in group if e["kind"] == "add" and not find_word(cur[idx], e["term"], re.I)]
        op = _add_op(cur[idx], [e["new"] for e in adds]) if adds else None
        if op:
            # terms/sep let a PDF wrap the added terms onto a new line when they don't fit.
            ops.append({"line_index": idx, "line": cur[idx], "old": op[0], "new": op[1],
                        "append": True, "insert": False, "ids": [e["id"] for e in adds],
                        "terms": [e["new"] for e in adds], "sep": op[2]})
            cur[idx] = apply_op(cur[idx], ops[-1])
        for e in (e for e in group if e["kind"] == "sentence" and e["new"].strip()):
            ops.append({"line_index": idx, "line": cur[idx], "old": "", "new": e["new"].strip(),
                        "append": False, "insert": True, "ids": [e["id"]]})
    return ops


def apply_op(text: str, op: dict) -> str:
    if op["append"]:
        pos = text.rfind(op["old"])
    else:
        m = find_word(text, op["old"])
        pos = m.start() if m else -1
    return text if pos < 0 else text[:pos] + op["new"] + text[pos + len(op["old"]):]


def apply_text_ops(lines: List[str], ops: List[dict]) -> List[Tuple[Optional[str], str]]:
    """(line before, line after) pairs; an inserted line has None before.
    A new bullet uses the same bullet mark as the line it follows."""
    out = list(lines)
    inserts: Dict[int, List[str]] = {}
    for op in ops:
        if op["insert"]:
            mark = BULLET.match(lines[op["line_index"]])
            inserts.setdefault(op["line_index"], []).append(
                (mark.group().strip() + " " if mark else "- ") + op["new"])
        else:
            out[op["line_index"]] = apply_op(out[op["line_index"]], op)
    pairs: List[Tuple[Optional[str], str]] = []
    for i, (before, after) in enumerate(zip(lines, out)):
        pairs.append((before, after))
        pairs += [(None, l) for l in inserts.get(i, [])]
    return pairs


def edited_text(lines: List[str], edits: List[dict]) -> str:
    return "\n".join(after for _, after in apply_text_ops(lines, edit_ops(lines, edits)))


def diff_segments(before: Optional[str], after: str) -> List[dict]:
    """A line as segments for display: {"t": unchanged} or {"del": old, "ins": new}."""
    if before is None:
        return [{"del": "", "ins": after}]
    if before == after:
        return [{"t": after}]
    a, b = re.split(r"(\s+)", before), re.split(r"(\s+)", after)
    segs = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            segs.append({"t": "".join(a[i1:i2])})
        else:
            segs.append({"del": "".join(a[i1:i2]), "ins": "".join(b[j1:j2])})
    return segs


# ---------------------------------------------------------------- scoring

def credit(combined: float) -> float:
    return min(max((combined - CREDIT_FLOOR) / (MATCH - CREDIT_FLOOR), 0.0), 1.0)


def status(combined: float) -> str:
    return "Match" if combined >= MATCH else "Partial" if combined >= PARTIAL else "Missing"


def _evaluate(resume: str, reqs: List[Tuple[str, str]]) -> Tuple[List[dict], float, float]:
    """(per-requirement results, score 0-100 unrounded, weighted mean semantic similarity 0-1)."""
    lines = resume_lines(resume)
    if not lines:
        raise ValueError("Couldn't find any content in the resume.")
    have = keywords(resume)
    sims = embed([r for r, _ in reqs]) @ embed(lines).T
    results = []
    for (req, imp), row in zip(reqs, sims):
        best = int(row.argmax())
        cos = max(float(row[best]), 0.0)
        req_kw = keywords(req)
        kw = len(req_kw & have) / len(req_kw) if req_kw else 0.0
        combined = round(0.6 * cos + 0.4 * kw, 3)
        results.append({
            "requirement": req,
            "importance": imp,
            "status": status(combined),
            "similarity": combined,
            "semantic": round(cos, 3),
            "keyword_share": round(kw, 3),
            "evidence": lines[best] if cos >= 0.3 else None,
            "missing_terms": [t for t, _ in missing_terms(req, have)],
            "matched_terms": [t for t, _ in missing_terms(req, set()) if _present(t, have)],
        })
    total = sum(WEIGHTS[r["importance"]] for r in results)
    pct = 100 * sum(WEIGHTS[r["importance"]] * credit(r["similarity"]) for r in results) / total
    semantic = sum(WEIGHTS[r["importance"]] * r["semantic"] for r in results) / total
    return results, pct, semantic


def _keyword_percent(resume: str, results: List[dict]) -> int:
    have = keywords(resume)
    terms = {t.lower(): t for r in results for t, _ in missing_terms(r["requirement"], set())}
    return round(100 * sum(_present(t, have) for t in terms.values()) / len(terms)) if terms else 100


def _score_summary(resume: str, results: List[dict], pct: float, semantic: float) -> dict:
    return {
        "percent": round(pct),
        "counts": {s: sum(r["status"] == s for r in results) for s in ("Match", "Partial", "Missing")},
        "total_requirements": len(results),
        "target_percent": TARGET_PERCENT,
        "meets_target": round(pct) >= TARGET_PERCENT,
        "semantic_percent": round(100 * semantic),
        "keyword_percent": _keyword_percent(resume, results),
    }


def _prepare(resume_text: str, jd_text: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    resume, jd = normalize(resume_text), normalize(jd_text)
    reqs = jd_requirements(jd)
    if not reqs:
        raise ValueError("Couldn't find any requirements in the job description. Try pasting the full posting.")
    if not resume:
        raise ValueError("Couldn't find any content in the resume.")
    return resume.split("\n"), reqs


def _score_edits(lines: List[str], reqs: List[Tuple[str, str]], edits: List[dict]) -> float:
    return _evaluate(edited_text(lines, edits), reqs)[1]


def analyze(resume_text: str, jd_text: str) -> dict:
    lines, reqs = _prepare(resume_text, jd_text)
    resume = "\n".join(lines)
    results, pct, semantic = _evaluate(resume, reqs)
    kmap, edits = keyword_analysis(lines, reqs, results)
    edits += sentence_edits(lines, results, first_id=len(edits))

    # What each edit is worth on its own, then the fewest edits that reach
    # the target, checked by re-scoring: keyword edits first (biggest gain
    # first), new sentences only if keywords alone fall short.
    for e in edits:
        e["gain"] = round(_score_edits(lines, reqs, [e]) - pct, 1)
    # A new line that doesn't raise the score isn't worth its space.
    edits = [e for e in edits if e["kind"] != "sentence" or e["gain"] > 0]
    plan, plan_pct = [], pct
    for e in sorted((e for e in edits if e["gain"] > 0), key=lambda e: (e["kind"] == "sentence", -e["gain"])):
        if round(plan_pct) >= TARGET_PERCENT:
            break
        plan.append(e)
        plan_pct = _score_edits(lines, reqs, plan)
    for e in edits:
        e["selected"] = e["kind"] == "replace" and e["confidence"] == "high" and e["gain"] >= 0

    score = _score_summary(resume, results, pct, semantic)
    return {
        "resume_text": resume,
        "score": score,
        "requirements": results,
        "keywords": kmap,
        "edits": edits,
        "plan": {"edit_ids": [e["id"] for e in plan], "percent": round(plan_pct),
                 "reaches_target": round(plan_pct) >= TARGET_PERCENT},
        "swot": swot(score, results, kmap, edits, round(plan_pct)),
    }


def preview_edits(resume_text: str, jd_text: str, edits: List[dict]) -> dict:
    """The resume with the chosen edits applied: its score, the operations,
    and each line as diff segments so changed words and new lines can be
    highlighted."""
    lines, reqs = _prepare(resume_text, jd_text)
    ops = edit_ops(lines, edits)
    pairs = apply_text_ops(lines, ops)
    edited = "\n".join(after for _, after in pairs)
    results, pct, semantic = _evaluate(edited, reqs)
    return {
        "score": _score_summary(edited, results, pct, semantic),
        "edited_text": edited,
        "lines": [diff_segments(before, after) for before, after in pairs],
        "ops": ops,
    }
