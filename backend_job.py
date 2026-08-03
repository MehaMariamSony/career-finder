import io
import re
from urllib.parse import quote_plus

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

app = FastAPI(title="Job & Internship Platform Finder API")

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


def build_deep_links(role: str, is_internship: bool, location_query: str, country: str = "") -> List[dict]:
    q_role = quote_plus(role)
    loc_parts = [p.strip() for p in location_query.split(",") if p.strip()]
    if country and loc_parts and loc_parts[-1].lower() == country.lower():
        loc_parts = loc_parts[:-1]
    place = ", ".join(loc_parts)
    q_loc = quote_plus(place) if place else ""
    is_india = country.lower() == "india"
    indeed_domain = "in.indeed.com" if is_india else "www.indeed.com"

    links = [
        {"platform": "LinkedIn", "note": "Search directly on LinkedIn for corporate, MNC, and hiring manager postings.", "url": f"https://www.linkedin.com/jobs/search/?keywords={q_role}&location={q_loc}"},
    ]

    if is_india:
        links.append({"platform": "Naukri", "note": "Explore nationwide openings on Naukri.", "url": f"https://www.naukri.com/{quote_plus(role.replace(' ', '-'))}-jobs" + (f"-in-{quote_plus(loc_parts[0])}" if loc_parts else "")})
    else:
        links.append({"platform": "Glassdoor", "note": "Search company-reviewed listings on Glassdoor.", "url": f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={q_role}&locT=C&locKeyword={q_loc}"})

    links.append({"platform": "Indeed", "note": "Search aggregated listings on Indeed.", "url": f"https://{indeed_domain}/jobs?q={q_role}&l={q_loc}"})

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
        deep_links = build_deep_links(req.role, req.is_internship, location_query, req.country or "")

        return {
            "success": True,
            "deep_links": deep_links,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
