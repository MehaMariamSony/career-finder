import io
import re
import html as html_lib
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote_plus

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

app = FastAPI(title="Job & Internship Aggregator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expanded skill vocabulary for higher extraction accuracy from CVs
KNOWN_SKILLS = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang", "rust", "php", "ruby", "kotlin", "swift", "r", "matlab", "scala",
    # Databases
    "sql", "mysql", "postgresql", "postgres", "mongodb", "redis", "nosql", "sqlite", "oracle", "cassandra", "dynamodb",
    # Web & Frameworks
    "html", "html5", "css", "css3", "react", "react.js", "reactjs", "angular", "vue", "vue.js", "next.js", "nextjs", "node.js", "nodejs",
    "express", "express.js", "django", "flask", "fastapi", "spring", "spring boot", "laravel", ".net", "asp.net", "tailwind", "bootstrap",
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s", "terraform", "ci/cd", "jenkins", "github actions", "ansible",
    "git", "github", "gitlab", "linux", "bash", "shell scripting", "devops", "sysadmin",
    # Data & AI
    "machine learning", "deep learning", "nlp", "natural language processing", "computer vision", "tensorflow", "pytorch",
    "pandas", "numpy", "scikit-learn", "keras", "opencv", "data analysis", "data visualization", "power bi", "tableau", "excel", "advanced excel",
    # Design & Product
    "figma", "adobe xd", "photoshop", "illustrator", "ui/ux", "user experience", "user interface", "wireframing", "prototyping",
    # Management & Methodologies
    "project management", "agile", "scrum", "jira", "confluence", "kanban", "communication", "problem solving",
    # Digital Marketing & Content
    "digital marketing", "seo", "sem", "content writing", "social media marketing", "google analytics", "copywriting",
    # Mobile
    "android", "ios", "flutter", "react native", "mobile development",
    # Security & Networks
    "cybersecurity", "networking", "ethical hacking", "pen testing", "information security"
]

HEADERS = {"User-Agent": "Mozilla/5.0 (job-aggregator/1.0)"}
REQUEST_TIMEOUT = 6

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_description(raw: str) -> str:
    if not raw:
        return ""
    text = html_lib.unescape(raw)
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


_ENGLISH_MARKERS = {
    "the", "and", "you", "with", "for", "our", "your", "team", "work",
    "experience", "will", "have", "are", "role", "job", "we", "to", "in",
    "of", "a", "is", "this",
}


def is_english(text: str) -> bool:
    if not text or len(text) < 25:
        return True
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 8:
        return True
    hits = sum(1 for w in words if w in _ENGLISH_MARKERS)
    return (hits / len(words)) >= 0.12


KNOWN_COUNTRIES = [
    "india", "united states", "usa", "u.s.", "uk", "united kingdom", "canada",
    "germany", "france", "spain", "netherlands", "poland", "portugal", "italy",
    "ireland", "australia", "singapore", "philippines", "pakistan", "bangladesh",
    "sri lanka", "nepal", "uae", "united arab emirates", "saudi arabia", "china",
    "japan", "brazil", "mexico", "argentina", "south africa", "nigeria", "egypt",
    "turkey", "russia", "sweden", "norway", "denmark", "finland", "belgium",
    "switzerland", "austria", "romania", "ukraine", "vietnam", "indonesia",
    "malaysia", "thailand", "new zealand", "israel", "greece", "czech republic",
]


class JobSearchRequest(BaseModel):
    role: str
    skills: List[str] = []
    experience: Optional[str] = ""
    is_internship: bool = False
    is_remote: bool = False
    work_mode: Optional[str] = "any"
    country: Optional[str] = "India"
    state: Optional[str] = ""
    district: Optional[str] = ""


def calculate_match_score(job_desc: str, title: str, role: str, user_skills: List[str]) -> int:
    text = f"{title} {job_desc}".lower()
    role_hit = 1 if role and role.lower() in text else 0
    matched_skills = [s for s in user_skills if s.lower() in text]
    skill_ratio = (len(matched_skills) / len(user_skills)) if user_skills else 0.4
    score = int(35 + role_hit * 25 + skill_ratio * 38)
    return min(max(score, 30), 98)


REMOTE_MARKERS = ["remote", "anywhere", "worldwide", "global", "work from home", "wfh"]


def _text_mentions(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    needle = needle.lower().strip()
    if needle in haystack:
        return True
    first_word = needle.split(",")[0].split()[0] if needle.split() else needle
    return len(first_word) > 3 and first_word in haystack


DISTRICT_ALIASES = {
    "thiruvananthapuram": ["trivandrum", "tvm"],
    "ernakulam": ["kochi", "cochin"],
    "kozhikode": ["calicut"],
    "thrissur": ["trichur"],
    "kollam": ["quilon"],
    "alappuzha": ["alleppey"],
    "kannur": ["cannanore"],
    "bengaluru urban": ["bangalore", "bengaluru"],
    "chennai": ["madras"],
    "mumbai city": ["bombay"],
    "pune": ["poona"],
    "kolkata": ["calcutta"],
    "visakhapatnam": ["vizag", "vishakhapatnam"],
    "vijayawada (ntr)": ["vijayawada"],
}


def _district_mentions(loc: str, district: str) -> bool:
    if not district:
        return False
    if _text_mentions(loc, district):
        return True
    key = district.lower().strip()
    for alias in DISTRICT_ALIASES.get(key, []):
        if _text_mentions(loc, alias):
            return True
    return False


def _mentions_other_country(loc: str, requested_country: str) -> bool:
    req = (requested_country or "").lower()
    for c in KNOWN_COUNTRIES:
        if c in loc and c not in req and req not in c:
            return True
    return False


def location_match_tier(job_location: str, country: str, state: str, district: str, work_mode: str) -> int:
    loc = (job_location or "").lower()
    is_remote_listing = any(k in loc for k in REMOTE_MARKERS)

    place_tier = 0
    if district and _district_mentions(loc, district):
        place_tier = 3
    elif state and _text_mentions(loc, state):
        place_tier = 2
    elif country and _text_mentions(loc, country):
        place_tier = 1

    mentions_other_country = bool(country and _mentions_other_country(loc, country))
    mismatched_country = not place_tier and not is_remote_listing and mentions_other_country

    if work_mode == "remote":
        if is_remote_listing:
            return 5 if place_tier else 4
        return place_tier

    if work_mode == "onsite":
        if is_remote_listing:
            return -2
        if mismatched_country:
            return -1
        return place_tier

    if work_mode == "hybrid":
        if place_tier:
            return place_tier + 1
        if mismatched_country:
            return -1
        return 1 if is_remote_listing else 0

    if place_tier:
        return place_tier
    if mismatched_country:
        return -1
    return 1 if is_remote_listing else 0


def is_relevant(title: str, desc: str, role: str, skills: List[str]) -> bool:
    if not role:
        return True
    text = f"{title} {desc}".lower()
    role_lower = role.lower().strip()
    if role_lower in text:
        return True
    role_terms = [t for t in role_lower.split() if len(t) > 2]
    if role_terms and all(t in text for t in role_terms):
        return True
    return any(s.lower() in text for s in skills)


# Live source fetchers
def fetch_remotive(role: str) -> List[dict]:
    try:
        r = requests.get("https://remotive.com/api/remote-jobs", params={"search": role, "limit": 20}, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return [{"title": j.get("title"), "company": j.get("company_name"), "location": j.get("candidate_required_location") or "Remote", "publisher": "Remotive", "apply_link": j.get("url"), "description": j.get("description", ""), "source_type": "live"} for j in r.json().get("jobs", [])]
    except Exception:
        return []

def fetch_arbeitnow(role: str) -> List[dict]:
    try:
        r = requests.get("https://www.arbeitnow.com/api/job-board-api", headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return [{"title": j.get("title"), "company": j.get("company_name"), "location": j.get("location") or ("Remote" if j.get("remote") else ""), "publisher": "Arbeitnow", "apply_link": j.get("url"), "description": j.get("description", ""), "source_type": "live"} for j in r.json().get("data", [])]
    except Exception:
        return []

def fetch_remoteok(role: str) -> List[dict]:
    try:
        r = requests.get("https://remoteok.com/api", headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        jobs = [j for j in r.json() if isinstance(j, dict) and j.get("id")]
        return [{"title": j.get("position"), "company": j.get("company"), "location": j.get("location") or "Remote", "publisher": "RemoteOK", "apply_link": f"https://remoteok.com{j.get('url')}" if j.get("url", "").startswith("/") else j.get("url"), "description": j.get("description", ""), "source_type": "live"} for j in jobs]
    except Exception:
        return []

def fetch_themuse(role: str) -> List[dict]:
    try:
        r = requests.get("https://www.themuse.com/api/public/jobs", params={"page": 0}, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        out = []
        for j in r.json().get("results", []):
            locs = ", ".join(loc.get("name", "") for loc in j.get("locations", [])) or "Various"
            out.append({"title": j.get("name"), "company": (j.get("company") or {}).get("name"), "location": locs, "publisher": "The Muse", "apply_link": j.get("refs", {}).get("landing_page"), "description": j.get("contents", ""), "source_type": "live"})
        return out
    except Exception:
        return []

LIVE_SOURCES = [fetch_remotive, fetch_arbeitnow, fetch_remoteok, fetch_themuse]


def build_deep_links(role: str, is_internship: bool, location_query: str, country: str = "") -> List[dict]:
    q_role = quote_plus(role)
    loc_parts = [p.strip() for p in location_query.split(",") if p.strip()]
    if country and loc_parts and loc_parts[-1].lower() == country.lower():
        loc_parts = loc_parts[:-1]
    place = ", ".join(loc_parts)
    q_loc = quote_plus(place) if place else ""
    indeed_domain = "in.indeed.com" if country.lower() == "india" else "www.indeed.com"

    links = [
        {"platform": "LinkedIn", "note": "Search directly on LinkedIn for corporate, MNC, and hiring manager postings.", "url": f"https://www.linkedin.com/jobs/search/?keywords={q_role}&location={q_loc}"},
        {"platform": "Naukri", "note": "Explore nationwide openings on Naukri.", "url": f"https://www.naukri.com/{quote_plus(role.replace(' ', '-'))}-jobs" + (f"-in-{quote_plus(loc_parts[0])}" if loc_parts else "")},
        {"platform": "Indeed", "note": "Search aggregated listings on Indeed.", "url": f"https://{indeed_domain}/jobs?q={q_role}&l={q_loc}"},
    ]

    if is_internship:
        links.insert(1, {"platform": "Internshala", "note": "Search student and early-career internship listings directly.", "url": f"https://internshala.com/internships/keywords-{quote_plus(role.replace(' ', '%20'))}"})

    return links


@app.post("/api/resume/parse")
async def parse_resume(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    try:
        raw = await file.read()
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Couldn't read that PDF: {e}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No readable text found in PDF.")

    text_lower = text.lower()
    matched_skills = set()
    
    # Enhanced extraction matching using word boundaries
    for skill in KNOWN_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            matched_skills.add(skill.title() if len(skill) > 3 else skill.upper())

    return {
        "success": True,
        "matched_skills": sorted(list(matched_skills)),
        "preview": text.strip()[:400],
    }


@app.post("/api/jobs/search")
def search_jobs(req: JobSearchRequest):
    work_mode = (req.work_mode or "any").lower()
    if work_mode not in ("any", "remote", "onsite", "hybrid"):
        work_mode = "any"

    if work_mode == "remote":
        location_query = "Remote"
    else:
        loc_parts = [p for p in [req.district, req.state, req.country] if p]
        location_query = ", ".join(loc_parts)

    try:
        raw_jobs: List[dict] = []
        with ThreadPoolExecutor(max_workers=len(LIVE_SOURCES)) as pool:
            futures = [pool.submit(fn, req.role) for fn in LIVE_SOURCES]
            for f in as_completed(futures):
                raw_jobs.extend(f.result())

        results = []
        for job in raw_jobs:
            title = job.get("title") or ""
            desc = clean_description(job.get("description") or "")

            if not is_relevant(title, desc, req.role, req.skills):
                continue
            if not (is_english(title) and is_english(desc)):
                continue

            match_score = calculate_match_score(desc, title, req.role, req.skills)
            job_loc = job.get("location") or ("Remote" if work_mode == "remote" else "Location not listed")
            loc_tier = location_match_tier(job.get("location") or "", req.country or "", req.state or "", req.district or "", work_mode)

            results.append({
                "title": title,
                "company": job.get("company") or "Unknown company",
                "location": job_loc,
                "publisher": job.get("publisher"),
                "type": "INTERNSHIP" if req.is_internship else "FULL-TIME",
                "apply_link": job.get("apply_link"),
                "match_score": match_score,
                "location_tier": loc_tier,
                "description": (desc[:180] + "...") if len(desc) > 180 else desc,
            })

        if work_mode in ("onsite", "hybrid", "any") and (req.district or req.state or req.country):
            results = [r for r in results if r["location_tier"] >= 1]

        results.sort(key=lambda j: (j["location_tier"], j["match_score"]), reverse=True)
        results = results[:24]
        for r in results:
            r.pop("location_tier", None)

        deep_links = build_deep_links(req.role, req.is_internship, location_query, req.country or "")

        return {
            "success": True,
            "count": len(results),
            "jobs": results,
            "deep_links": deep_links,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
