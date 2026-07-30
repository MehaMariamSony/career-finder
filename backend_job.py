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

# A reasonably broad skill vocabulary to match resume text against.
# Not exhaustive -- extend this list any time for skills you expect to see.
KNOWN_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "php", "ruby",
    "sql", "mysql", "postgresql", "mongodb", "redis", "nosql",
    "html", "css", "react", "angular", "vue", "next.js", "node.js", "express",
    "django", "flask", "fastapi", "spring", "laravel", ".net",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd", "jenkins",
    "git", "linux", "bash", "shell scripting",
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch",
    "pandas", "numpy", "scikit-learn", "data analysis", "data visualization", "power bi", "tableau",
    "excel", "r", "matlab",
    "figma", "adobe xd", "photoshop", "illustrator", "ui/ux",
    "project management", "agile", "scrum", "jira",
    "digital marketing", "seo", "content writing", "social media marketing",
    "android", "ios", "swift", "kotlin", "flutter", "react native",
    "cybersecurity", "networking", "devops", "salesforce", "sap",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (job-aggregator/1.0)"}
REQUEST_TIMEOUT = 6

# Experience-level vocabulary used to line a job posting's stated seniority
# up against what the candidate selected in the search form.
EXPERIENCE_KEYWORDS = {
    "fresher": ["fresher", "entry level", "entry-level", "no experience", "0-1 year",
                "graduate", "trainee", "campus hire", "0 to 1 year"],
    "junior": ["junior", "1-3 years", "1-2 years", "associate", "1+ year"],
    "mid": ["mid level", "mid-level", "3-5 years", "3+ years", "intermediate"],
    "senior": ["senior", "5+ years", "lead", "principal", "staff engineer", "8+ years"],
}

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_description(raw: str) -> str:
    """Strip HTML tags/entities out of a posting's description so the UI
    never shows raw markup like '<br><br>Company:'.

    Order matters here: some sources (Arbeitnow in particular) store the
    description with its HTML double-encoded -- the tags themselves appear
    as literal text like '&lt;h1&gt;'. Stripping tags before unescaping
    entities misses those, and unescaping afterward turns them into real,
    un-stripped tags that leak straight into the UI. Unescape first so any
    encoded tags become real tags, then strip everything in one pass.
    """
    if not raw:
        return ""
    text = html_lib.unescape(raw)
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# A rough but effective English detector: real English postings are loaded
# with short, extremely common function words. Non-English text (German,
# French, Spanish postings that slip through Arbeitnow, etc.) essentially
# never uses these at any real frequency.
_ENGLISH_MARKERS = {
    "the", "and", "you", "with", "for", "our", "your", "team", "work",
    "experience", "will", "have", "are", "role", "job", "we", "to", "in",
    "of", "a", "is", "this",
}


def is_english(text: str) -> bool:
    if not text or len(text) < 25:
        return True  # too short to judge reliably -- don't over-filter
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 8:
        return True
    hits = sum(1 for w in words if w in _ENGLISH_MARKERS)
    return (hits / len(words)) >= 0.12


# Common country names used to penalise live listings that are explicitly
# tagged to a *different* country than the one the candidate searched for.
# City-level detail isn't available from these free feeds, but an explicit
# country mention is a strong, cheap signal.
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
    experience: Optional[str] = ""   # "fresher" | "junior" | "mid" | "senior" | "" (any)
    is_internship: bool = False
    is_remote: bool = False
    work_mode: Optional[str] = "any"  # "any" | "remote" | "onsite" | "hybrid"
    country: Optional[str] = "India"
    state: Optional[str] = ""
    district: Optional[str] = ""


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def calculate_experience_fit(text: str, experience: str) -> int:
    """0-15 bonus points for how well the posting's stated seniority lines
    up with what the candidate selected. Neutral (10) when either side is
    unspecified, since most postings don't name a level explicitly."""
    if not experience or experience == "any":
        return 10
    keywords = EXPERIENCE_KEYWORDS.get(experience, [])
    if any(k in text for k in keywords):
        return 15
    # Check whether the posting explicitly asks for a different level --
    # if so it's a weaker fit, but not disqualifying.
    for level, kws in EXPERIENCE_KEYWORDS.items():
        if level != experience and any(k in text for k in kws):
            return 4
    return 10


def calculate_match_score(job_desc: str, title: str, role: str, user_skills: List[str], experience: str = "") -> int:
    text = f"{title} {job_desc}".lower()
    role_hit = 1 if role and role.lower() in text else 0
    matched_skills = [s for s in user_skills if s.lower() in text]
    skill_ratio = (len(matched_skills) / len(user_skills)) if user_skills else 0.4
    exp_score = calculate_experience_fit(text, experience)
    score = int(25 + role_hit * 22 + skill_ratio * 36 + exp_score)
    return min(max(score, 30), 98)


REMOTE_MARKERS = ["remote", "anywhere", "worldwide", "global", "work from home", "wfh"]


def _text_mentions(haystack: str, needle: str) -> bool:
    """Loose containment check: matches the full phrase, or -- for multi-word
    places like 'Thiruvananthapuram' vs a listing that only says 'Kerala' --
    falls back to matching on the first significant word."""
    if not needle:
        return False
    needle = needle.lower().strip()
    if needle in haystack:
        return True
    first_word = needle.split(",")[0].split()[0] if needle.split() else needle
    return len(first_word) > 3 and first_word in haystack


# Common alternate/older/colloquial names for Indian districts and cities.
# Job listings frequently use the informal name (e.g. "Bangalore", "Trivandrum")
# rather than the official district name a person picks from the dropdown
# (e.g. "Bengaluru Urban", "Thiruvananthapuram") -- this keeps those matching.
# Keyed by the canonical district name (lowercase) as selected in the UI.
DISTRICT_ALIASES = {
    "thiruvananthapuram": ["trivandrum", "tvm"],
    "ernakulam": ["kochi", "cochin"],
    "kozhikode": ["calicut"],
    "thrissur": ["trichur"],
    "kollam": ["quilon"],
    "alappuzha": ["alleppey"],
    "kannur": ["cannanore"],
    "kottayam": ["kottayam"],
    "palakkad": ["palghat"],
    "bengaluru urban": ["bangalore", "bengaluru"],
    "bengaluru rural": ["bangalore rural"],
    "mysuru": ["mysore"],
    "dakshina kannada": ["mangalore", "mangaluru"],
    "belagavi": ["belgaum"],
    "chennai": ["madras"],
    "tiruchirappalli": ["trichy", "tiruchi"],
    "thanjavur": ["tanjore"],
    "puducherry": ["pondicherry"],
    "vadodara": ["baroda"],
    "mumbai city": ["bombay"],
    "mumbai suburban": ["bombay"],
    "pune": ["poona"],
    "kolkata": ["calcutta"],
    "prayagraj": ["allahabad"],
    "kanpur nagar": ["kanpur"],
    "varanasi": ["banaras", "kashi"],
    "gurugram": ["gurgaon"],
    "visakhapatnam": ["vizag", "vishakhapatnam"],
    "kamrup metropolitan": ["guwahati", "gauhati"],
    "hyderabad": ["hyderabad"],
}


def _district_mentions(loc: str, district: str) -> bool:
    """District match that also checks known aliases in both directions --
    the person may have picked the formal name while the listing uses the
    informal one, or vice versa."""
    if not district:
        return False
    if _text_mentions(loc, district):
        return True
    key = district.lower().strip()
    for alias in DISTRICT_ALIASES.get(key, []):
        if _text_mentions(loc, alias):
            return True
    # Reverse lookup: person typed an alias (free-text fallback for
    # countries without a district dropdown) that maps to a canonical name.
    for canonical, aliases in DISTRICT_ALIASES.items():
        if key in aliases and _text_mentions(loc, canonical):
            return True
    return False


def _mentions_other_country(loc: str, requested_country: str) -> bool:
    """True if the listing text names a country that clearly isn't the one
    the candidate asked for (e.g. 'Madrid, Spain' when searching India)."""
    req = (requested_country or "").lower()
    for c in KNOWN_COUNTRIES:
        if c in loc and c not in req and req not in c:
            return True
    return False


def location_match_tier(job_location: str, country: str, state: str, district: str, work_mode: str) -> int:
    """Higher tier = closer match to what the candidate asked for.
    Tiers are shaped by work_mode so 'sort by place' means something
    different for a remote search vs. an on-site one:

      remote  -> remote-tagged listings always rank top; place is secondary
      onsite  -> remote-tagged listings sink to the bottom; a real
                 district/state/country mention ranks top; listings
                 explicitly tagged to a different country sink too
      hybrid  -> place match ranks top, remote-tagged listings rank middle
      any     -> a specified place is treated as a soft on-site preference
    """
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
            if place_tier:
                return 5  # remote AND explicitly open to the requested place
            if mentions_other_country:
                return 3  # remote but restricted to a different country
            return 4  # remote, no place restriction mentioned
        return place_tier

    if work_mode == "onsite":
        if is_remote_listing:
            return -2
        if mismatched_country:
            return -1
        return place_tier  # 0 = unlabeled location, still above the two negatives

    if work_mode == "hybrid":
        if place_tier:
            return place_tier + 1
        if mismatched_country:
            return -1
        return 1 if is_remote_listing else 0

    # "any" -- a chosen place is still treated as a soft preference
    if place_tier:
        return place_tier
    if mismatched_country:
        return -1
    return 1 if is_remote_listing else 0


def _term_matches(text: str, term: str) -> bool:
    """Word match with light stemming tolerance: 'developer' should still
    match a posting that says 'development' or 'developing'."""
    if term in text:
        return True
    prefix = term[:7] if len(term) > 7 else term
    return re.search(rf"\b{re.escape(prefix)}\w*", text) is not None


def is_relevant(title: str, desc: str, role: str, skills: List[str]) -> bool:
    """Relevance filter so unrelated jobs from broad feeds get dropped.

    Deliberately stricter than 'any word matches': for a multi-word role
    like 'business developer', matching on just 'business' alone lets in
    completely unrelated postings (an admin assistant job that happens to
    say 'our business' somewhere). Require the whole phrase, or -- failing
    that -- ALL of the role's significant words (with light stemming, so
    'developer' vs. 'development' still counts), before falling back to a
    skills match.
    """
    if not role:
        return True
    text = f"{title} {desc}".lower()
    role_lower = role.lower().strip()
    if role_lower in text:
        return True
    role_terms = [t for t in role_lower.split() if len(t) > 2]
    if role_terms and all(_term_matches(text, t) for t in role_terms):
        return True
    return any(s.lower() in text for s in skills)


# ---------------------------------------------------------------------------
# Live source fetchers -- each returns a list of normalized job dicts.
# Every fetcher is defensive: any failure returns [] rather than raising,
# so one dead API never takes the whole search down.
# ---------------------------------------------------------------------------

def fetch_remotive(role: str) -> List[dict]:
    try:
        r = requests.get(
            "https://remotive.com/api/remote-jobs",
            params={"search": role, "limit": 20},
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        jobs = r.json().get("jobs", [])
        return [
            {
                "title": j.get("title"),
                "company": j.get("company_name"),
                "location": j.get("candidate_required_location") or "Remote",
                "publisher": "Remotive",
                "apply_link": j.get("url"),
                "description": j.get("description", ""),
                "source_type": "live",
            }
            for j in jobs
        ]
    except Exception:
        return []


def fetch_arbeitnow(role: str) -> List[dict]:
    try:
        r = requests.get(
            "https://www.arbeitnow.com/api/job-board-api",
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        jobs = r.json().get("data", [])
        return [
            {
                "title": j.get("title"),
                "company": j.get("company_name"),
                "location": j.get("location") or ("Remote" if j.get("remote") else ""),
                "publisher": "Arbeitnow",
                "apply_link": j.get("url"),
                "description": j.get("description", ""),
                "source_type": "live",
            }
            for j in jobs
        ]
    except Exception:
        return []


def fetch_remoteok(role: str) -> List[dict]:
    try:
        r = requests.get("https://remoteok.com/api", headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        jobs = r.json()
        # First element is a legal/metadata notice, not a job.
        jobs = [j for j in jobs if isinstance(j, dict) and j.get("id")]
        return [
            {
                "title": j.get("position"),
                "company": j.get("company"),
                "location": j.get("location") or "Remote",
                "publisher": "RemoteOK",
                "apply_link": f"https://remoteok.com{j.get('url')}" if j.get("url", "").startswith("/") else j.get("url"),
                "description": j.get("description", ""),
                "source_type": "live",
            }
            for j in jobs
        ]
    except Exception:
        return []


def fetch_themuse(role: str) -> List[dict]:
    try:
        r = requests.get(
            "https://www.themuse.com/api/public/jobs",
            params={"page": 0},
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        jobs = r.json().get("results", [])
        out = []
        for j in jobs:
            locations = ", ".join(loc.get("name", "") for loc in j.get("locations", [])) or "Various"
            out.append(
                {
                    "title": j.get("name"),
                    "company": (j.get("company") or {}).get("name"),
                    "location": locations,
                    "publisher": "The Muse",
                    "apply_link": j.get("refs", {}).get("landing_page"),
                    "description": j.get("contents", ""),
                    "source_type": "live",
                }
            )
        return out
    except Exception:
        return []


LIVE_SOURCES = [fetch_remotive, fetch_arbeitnow, fetch_remoteok, fetch_themuse]


# ---------------------------------------------------------------------------
# Deep-link cards for platforms that don't expose a public search API
# (LinkedIn, Naukri, Internshala, Indeed). These are pre-filled search URLs,
# not live listings -- clicking through takes the user to a real search on
# that platform.
# ---------------------------------------------------------------------------

def build_deep_links(role: str, is_internship: bool, location_query: str, country: str = "") -> List[dict]:
    q_role = quote_plus(role)

    # Country tacked onto the location string (e.g. "Thiruvananthapuram, Kerala,
    # India") confuses these platforms' location geocoding/sorting more than it
    # helps -- city+state is what they actually key off. Strip it here so the
    # place filter on the destination site actually applies instead of being
    # silently dropped.
    loc_parts = [p.strip() for p in location_query.split(",") if p.strip()]
    if country and loc_parts and loc_parts[-1].lower() == country.lower():
        loc_parts = loc_parts[:-1]
    place = ", ".join(loc_parts)
    q_loc = quote_plus(place) if place else ""

    # Indeed runs separate country-specific sites (in.indeed.com, indeed.co.uk,
    # ...) -- the generic www.indeed.com / .com domain often won't apply an
    # Indian city filter correctly (or redirects and drops it). Route to the
    # right one when we know the country.
    indeed_domain = "in.indeed.com" if country.lower() == "india" else "www.indeed.com"

    links = [
        {
            "platform": "LinkedIn",
            "note": "The largest professional network -- best for corporate, MNC, and referral-driven roles.",
            "url": f"https://www.linkedin.com/jobs/search/?keywords={q_role}&location={q_loc}",
        },
        {
            "platform": "Naukri",
            "note": "India's largest job board, with broad coverage across every experience level.",
            "url": f"https://www.naukri.com/{quote_plus(role.replace(' ', '-'))}-jobs" + (f"-in-{quote_plus(loc_parts[0])}" if loc_parts else ""),
        },
        {
            "platform": "Indeed",
            "note": "Aggregates listings from thousands of company career pages in one search.",
            "url": f"https://{indeed_domain}/jobs?q={q_role}&l={q_loc}",
        },
    ]

    if is_internship:
        links.insert(
            1,
            {
                "platform": "Internshala",
                "note": "India's leading internship-specific platform -- purpose-built for students and early-career talent.",
                "url": f"https://internshala.com/internships/keywords-{quote_plus(role.replace(' ', '%20'))}",
            },
        )

    return links


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

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
        raise HTTPException(
            status_code=422,
            detail="No readable text found in that PDF (it may be a scanned image rather than text).",
        )

    text_lower = text.lower()
    matched_skills = sorted({s for s in KNOWN_SKILLS if s in text_lower})

    return {
        "success": True,
        "matched_skills": matched_skills,
        "preview": text.strip()[:400],
    }


@app.post("/api/jobs/search")
def search_jobs(req: JobSearchRequest):
    work_mode = (req.work_mode or "any").lower()
    if work_mode not in ("any", "remote", "onsite", "hybrid"):
        work_mode = "any"
    if req.is_remote and work_mode == "any":
        work_mode = "remote"  # backward compatibility with the old checkbox

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

            match_score = calculate_match_score(desc, title, req.role, req.skills, req.experience)
            # Never invent a location -- falling back to the searched place
            # made jobs with no real location data (or a different real
            # location buried in the description) look like confirmed local
            # matches, which is actively misleading rather than just vague.
            job_loc = job.get("location") or ("Remote" if work_mode == "remote" else "Location not listed")
            loc_tier = location_match_tier(job.get("location") or "", req.country or "", req.state or "", req.district or "", work_mode)

            results.append(
                {
                    "title": title,
                    "company": job.get("company") or "Unknown company",
                    "location": job_loc,
                    "publisher": job.get("publisher"),
                    "type": "INTERNSHIP" if req.is_internship else "FULL-TIME",
                    "apply_link": job.get("apply_link"),
                    "match_score": match_score,
                    "location_tier": loc_tier,
                    "description": (desc[:180] + "...") if len(desc) > 180 else desc,
                }
            )

        # When a place is specified, only show jobs that actually match it --
        # no padding the list with unrelated jobs just because nothing local
        # was found. If nothing real matches, the result is an honest empty
        # list (with location_note explaining why) rather than a wall of
        # jobs from everywhere that happen to have no location data.
        confirmed_local = sum(1 for r in results if r["location_tier"] >= 1)
        if work_mode in ("onsite", "hybrid", "any") and (req.district or req.state or req.country):
            results = [r for r in results if r["location_tier"] >= 1]

        # Closest location match first, then best overall match within that tier.
        results.sort(key=lambda j: (j["location_tier"], j["match_score"]), reverse=True)
        results = results[:24]
        for r in results:
            r.pop("location_tier", None)

        # Live feeds (Remotive, Arbeitnow, RemoteOK, The Muse) skew heavily
        # toward Western remote roles and rarely carry Indian city/state
        # detail -- be upfront about that instead of implying a precision
        # the underlying data can't back up.
        location_note = None
        if work_mode != "remote" and location_query and confirmed_local < 3:
            location_note = (
                f"The live feeds have very few confirmed listings in {location_query} -- "
                "they lean toward global remote roles. For local, on-site openings, "
                "the platforms below are the more reliable source."
            )

        deep_links = build_deep_links(req.role, req.is_internship, location_query, req.country or "")

        return {
            "success": True,
            "count": len(results),
            "jobs": results,
            "deep_links": deep_links,
            "location_note": location_note,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
