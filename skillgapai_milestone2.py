# skillgap_milestone2_extended.py
"""
Extended SkillGapAI - Milestone 2
New features:
 - spaCy PhraseMatcher + alias mapping (better detection)
 - Resume upload (txt, pdf)
 - Chart toggle (donut / bar)
 - CSV, JSON and PDF report export
 - Download highlighted text
 - Skill normalization & suggestions
"""

import streamlit as st
import spacy
from spacy.matcher import PhraseMatcher
import re
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO, StringIO
import pandas as pd
import json
import base64
import tempfile

# For PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# For PDF text extraction (basic)
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="SkillGapAI — Milestone 2 ",
                   page_icon="🧭",
                   layout="wide")

# ----------------------------
# Utilities & CSS
# ----------------------------
def local_css():
    st.markdown(
        """
        <style>
        .skill-chip { display:inline-block; padding:6px 10px; margin:6px 8px 8px 0; border-radius:999px; background:#E8F6F3; color:#117A65; font-weight:600; }
        .skill-chip-soft { background:#EEF2FF; color:#4338CA; }
        .chip-small { display:inline-block; padding:5px 10px; margin:6px 8px 6px 0; border-radius:999px; font-size:0.78rem; font-weight:600; border:1px solid rgba(148,163,184,0.45); background: rgba(15,23,42,0.9); color: #e5e7eb; }
        .chip-missing { border-color: rgba(248,113,113,0.7); background: rgba(127,29,29,0.8); color: #fee2e2; }
        .chip-extra { border-color: rgba(52,211,153,0.7); background: rgba(6,78,59,0.85); color: #d1fae5; }
        .highlight-box { background-color:#020617; border-radius:8px; padding:12px; border:1px solid #1f2937; color:#e5e7eb; line-height:1.6; }
        .highlight { background-color:#65a30d; padding:1px 4px; border-radius:4px; color:#020617; font-weight:700; }
        </style>
        """, unsafe_allow_html=True)

local_css()

# ----------------------------
# Load spaCy model & PhraseMatcher (cached)
# ----------------------------
@st.cache_resource
def load_nlp_and_matcher(skill_map):
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = []
    for canonical, aliases in skill_map.items():
        for alias in aliases:
            patterns.append(nlp.make_doc(alias))
        matcher.add(canonical, patterns)
        patterns = []
    return nlp, matcher

# ----------------------------
# Skill lists + alias mapping (add synonyms/abbrev here)
# ----------------------------
# Map canonical -> list of alias strings (lowercased)
SKILL_ALIASES = {
    # technical
    "python": ["python", "py"],
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "sql": ["sql", "structured query language", "postgres", "postgresql", "mysql"],
    "html": ["html", "html5"],
    "css": ["css", "cascading style sheets"],
    "javascript": ["javascript", "js", "node js", "node.js"],
    "react": ["react", "reactjs", "react.js"],
    "node.js": ["node.js", "node", "nodejs"],
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch", "torch"],
    "machine learning": ["machine learning", "ml"],
    "data analysis": ["data analysis", "data analytics", "analytics"],
    "data visualization": ["data visualization", "dataviz", "visualization"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "django": ["django"],
    "flask": ["flask"],
    "scikit-learn": ["scikit-learn", "scikitlearn", "sklearn"],
    "nlp": ["nlp", "natural language processing"],
    # soft skills
    "communication": ["communication", "communicate"],
    "leadership": ["leadership", "lead"],
    "teamwork": ["teamwork", "team work", "team-player"],
    "problem solving": ["problem solving", "problem-solving", "problem solving skills"],
    "time management": ["time management", "time-management"],
    "adaptability": ["adaptability", "adaptable"],
    "critical thinking": ["critical thinking", "critical-thinking"],
    "creativity": ["creativity", "creative"],
    "collaboration": ["collaboration", "collaborate"],
    "decision making": ["decision making", "decision-making"]
}

# Prepare lists for UI (technical vs soft canonical)
TECH_CANONICAL = [
    "python","java","c++","sql","html","css","javascript","react","node.js",
    "tensorflow","pytorch","machine learning","data analysis","data visualization",
    "aws","azure","gcp","power bi","tableau","django","flask","scikit-learn","nlp"
]
SOFT_CANONICAL = [
    "communication","leadership","teamwork","problem solving","time management",
    "adaptability","critical thinking","creativity","collaboration","decision making"
]

nlp, matcher = load_nlp_and_matcher(SKILL_ALIASES)

# ----------------------------
# Helper functions
# ----------------------------
def extract_text_from_pdf(uploaded_file):
    if PyPDF2 is None:
        return ""
    reader = PyPDF2.PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text())
        except Exception:
            pages.append("")
    return "\n".join([p for p in pages if p])

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def phrase_match_skills(text):
    doc = nlp(text)
    matches = matcher(doc)
    found = set()
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]  # canonical name used when adding
        found.add(label)
    # also fallback substring search for things missing
    lc = text.lower()
    for canonical, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lc:
                found.add(canonical)
                break
    return sorted(found)

def categorize_skills(skills):
    tech = [s.title() for s in skills if s in TECH_CANONICAL]
    soft = [s.title() for s in skills if s in SOFT_CANONICAL]
    # unknown go into technical by default (safe)
    others = [s for s in skills if s not in TECH_CANONICAL + SOFT_CANONICAL]
    tech += [s.title() for s in others]
    # unique & sorted
    return sorted(set(tech)), sorted(set(soft))

def confidences_for(skills):
    # synthetic confidence: longer-known skills get higher score, just for UI
    if not skills:
        return {}
    base = 96
    step = 10 / max(len(skills)-1, 1)
    conf = {}
    for i, s in enumerate(sorted(skills, key=str.lower)):
        conf[s] = max(70, round(base - i*step))
    return conf

def highlight_text_html(text, skills):
    if not text:
        return ""
    highlighted = text
    # replace longer first
    for s in sorted(skills, key=len, reverse=True):
        try:
            pattern = re.compile(re.escape(s), flags=re.IGNORECASE)
            highlighted = pattern.sub(lambda m: f"<span class='highlight'>{m.group(0)}</span>", highlighted)
        except Exception:
            continue
    highlighted = highlighted.replace("\n", "<br>")
    return highlighted

def create_csv_bytes(resume_skills, jd_skills, missing, extra):
    df = pd.DataFrame({
        "Resume Skills": list(resume_skills) + [""] * max(0, len(jd_skills)-len(resume_skills)),
        "Job Description Skills": list(jd_skills) + [""] * max(0, len(resume_skills)-len(jd_skills))
    })
    summary = {
        "overlap": list(sorted(set(resume_skills) & set(jd_skills))),
        "missing_in_resume": list(missing),
        "extra_in_resume": list(extra),
        "generated_at": datetime.now().isoformat()
    }
    csv_buf = StringIO()
    df.to_csv(csv_buf, index=False)
    return csv_buf.getvalue(), summary

def create_json_bytes(resume_skills, jd_skills, missing, extra):
    payload = {
        "resume_skills": list(resume_skills),
        "jd_skills": list(jd_skills),
        "missing_in_resume": list(missing),
        "extra_in_resume": list(extra),
        "generated_at": datetime.now().isoformat()
    }
    return json.dumps(payload, indent=2)

def create_pdf_report(resume_text, jd_text, resume_skills, jd_skills, missing, extra, filename="skillgap_report.pdf"):
    # create a simple PDF using reportlab
    bio = BytesIO()
    c = canvas.Canvas(bio, pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "SkillGapAI - Analysis Report")
    c.setFont("Helvetica", 10)
    y -= 22
    c.drawString(margin, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 18

    def write_block(title, lines, font_size=10):
        nonlocal y
        if y < 120:
            c.showPage()
            y = height - margin
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y, title)
        y -= 14
        c.setFont("Helvetica", font_size)
        for line in lines:
            if y < 80:
                c.showPage()
                y = height - margin
            # wrap if necessary (very naive)
            if len(line) > 95:
                parts = [line[i:i+95] for i in range(0, len(line), 95)]
                for p in parts:
                    c.drawString(margin+6, y, p)
                    y -= 12
            else:
                c.drawString(margin+6, y, line)
                y -= 12
        y -= 6

    write_block("Resume (excerpt)", [l.strip() for l in resume_text.splitlines()[:8] if l.strip()])
    write_block("Job Description (excerpt)", [l.strip() for l in jd_text.splitlines()[:8] if l.strip()])
    write_block("Resume Skills (detected)", resume_skills or ["None"])
    write_block("Job Description Skills (detected)", jd_skills or ["None"])
    write_block("Missing in Resume (from JD)", missing or ["None"])
    write_block("Extra in Resume", extra or ["None"])

    c.save()
    bio.seek(0)
    return bio.read()

# ----------------------------
# Inputs: Paste or Upload
# ----------------------------
st.header("AI Skill Gap Analyzer — Milestone 2 (Extended)")
with st.expander("Input: Paste Resume & Job Description or Upload Files", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Resume Text")
        resume_text = st.text_area("Paste Resume content here", value="", height=220)
        uploaded_resume = st.file_uploader("Or upload resume (.txt or .pdf)", type=['txt','pdf'], key="resume_upload")
        if uploaded_resume is not None:
            if uploaded_resume.type == "application/pdf" and PyPDF2 is not None:
                try:
                    extracted = extract_text_from_pdf(uploaded_resume)
                    if extracted.strip():
                        resume_text = extracted
                        st.success("Extracted text from PDF (basic).")
                    else:
                        st.warning("PDF uploaded but could not extract text. Try a .txt file.")
                except Exception:
                    st.error("Failed to extract PDF text. Ensure PyPDF2 is installed.")
            elif uploaded_resume.type.startswith("text"):
                resume_text = uploaded_resume.getvalue().decode('utf-8', errors='ignore')
    with col2:
        st.subheader("Job Description Text")
        jd_text = st.text_area("Paste Job Description content here", value="", height=220)
        uploaded_jd = st.file_uploader("Or upload JD (.txt or .pdf)", type=['txt','pdf'], key="jd_upload")
        if uploaded_jd is not None:
            if uploaded_jd.type == "application/pdf" and PyPDF2 is not None:
                try:
                    extracted_jd = extract_text_from_pdf(uploaded_jd)
                    if extracted_jd.strip():
                        jd_text = extracted_jd
                        st.success("Extracted text from JD PDF (basic).")
                except Exception:
                    st.error("Failed to extract JD PDF text.")
            elif uploaded_jd.type.startswith("text"):
                jd_text = uploaded_jd.getvalue().decode('utf-8', errors='ignore')

# If no text at all, show info
if not resume_text and not jd_text:
    st.info("Paste or upload a resume or job description to begin skill extraction.")
    st.stop()

# ----------------------------
# Skill extraction using PhraseMatcher + alias mapping
# ----------------------------
resume_text_clean = clean_text(resume_text or "")
jd_text_clean = clean_text(jd_text or "")

resume_matches = phrase_match_skills(resume_text_clean) if resume_text_clean else []
jd_matches = phrase_match_skills(jd_text_clean) if jd_text_clean else []

tech_resume, soft_resume = categorize_skills(resume_matches)
tech_jd, soft_jd = categorize_skills(jd_matches)

resume_skills_set = set([s.title() for s in resume_matches])
jd_skills_set = set([s.title() for s in jd_matches])

common_skills = sorted(resume_skills_set & jd_skills_set)
missing_in_resume = sorted(jd_skills_set - resume_skills_set)
extra_in_resume = sorted(resume_skills_set - jd_skills_set)

# synthetic confidences
conf_resume = confidences_for(list(resume_skills_set))
conf_jd = confidences_for(list(jd_skills_set))

# ----------------------------
# UI: top summary & chart options
# ----------------------------
colA, colB = st.columns([1.2, 1])
with colA:
    st.subheader("Resume Skills")
    # show chips
    if resume_matches:
        chips_html = ""
        for s in sorted(resume_matches, key=str.lower):
            label = s.title()
            style = "skill-chip" if s in TECH_CANONICAL else "skill-chip-soft"
            score = conf_resume.get(s.title(), 85)
            chips_html += f"<span class='{style}'>{label} {score}%</span>"
        st.markdown(chips_html, unsafe_allow_html=True)
    else:
        st.info("No skills detected in resume (based on configured aliases).")

with colB:
    st.subheader("Skill Distribution")
    chart_type = st.selectbox("Chart Type", ["Donut", "Bar"], index=0)
    # compute counts
    tech_count = len([s for s in resume_matches if s in TECH_CANONICAL])
    soft_count = len([s for s in resume_matches if s in SOFT_CANONICAL])
    other_count = len([s for s in resume_matches if s not in TECH_CANONICAL + SOFT_CANONICAL])
    labels = ["Technical", "Soft", "Other"]
    sizes = [tech_count, soft_count, other_count]
    fig, ax = plt.subplots(figsize=(4,3.4))
    colors = ["#1F77B4", "#FF7F0E", "#2ECC71"]
    if chart_type == "Donut":
        wedges, texts = ax.pie(sizes, startangle=90, wedgeprops=dict(width=0.45, edgecolor='white'), colors=colors)
        ax.axis("equal")
        ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        ax.bar(labels, sizes, color=colors)
        ax.set_ylabel("Count")
    st.pyplot(fig, use_container_width=True)

# ----------------------------
# Highlighted text and download
# ----------------------------
st.markdown("### ✨ Highlighted Resume Text")
if resume_text_clean:
    highlighted_html = highlight_text_html(resume_text, [s.title() for s in resume_matches])
    st.markdown(f"<div class='highlight-box'>{highlighted_html}</div>", unsafe_allow_html=True)
    # download highlighted as HTML
    html_bytes = highlighted_html.encode("utf-8")
    st.download_button("Download highlighted HTML", data=html_bytes, file_name="resume_highlighted.html", mime="text/html")
else:
    st.write("No resume text to highlight.")

# ----------------------------
# Detailed skills with progress + suggestion box
# ----------------------------
st.markdown("### 📋 Detailed Skills & Suggestions")
left_col, right_col = st.columns([2,1])
with left_col:
    if resume_matches:
        for s in sorted(resume_matches, key=str.lower):
            score = conf_resume.get(s.title(), 82)
            is_tech = s in TECH_CANONICAL
            st.markdown(f"**{s.title()}** — {'Technical' if is_tech else 'Soft'} • {score}%")
            st.progress(int(score))
    else:
        st.info("No detected skills to display progress bars.")
with right_col:
    st.markdown("#### 🔍 Resume vs JD Summary")
    st.write(f"Overlap: **{len(common_skills)}**")
    st.write(f"Missing in Resume (from JD): **{len(missing_in_resume)}**")
    st.write(f"Extra in Resume: **{len(extra_in_resume)}**")
    st.markdown("**Missing skill suggestions**")
    if missing_in_resume:
        for m in missing_in_resume:
            st.button(f"Add suggestion: {m}", disabled=True)  # placeholder for manual add
            st.markdown(f"- {m}")
    else:
        st.markdown("_No missing skills detected. Good match!_")

# ----------------------------
# Export: CSV / JSON / PDF downloads
# ----------------------------
st.markdown("### ⤓ Export Results")
csv_str, summary = create_csv_bytes(sorted(resume_skills_set), sorted(jd_skills_set), missing_in_resume, extra_in_resume)
json_str = create_json_bytes(sorted(resume_skills_set), sorted(jd_skills_set), missing_in_resume, extra_in_resume)
pdf_bytes = create_pdf_report(resume_text, jd_text, sorted(resume_skills_set), sorted(jd_skills_set), missing_in_resume, extra_in_resume)

col1, col2, col3 = st.columns(3)
with col1:
    st.download_button("Download CSV", data=csv_str, file_name="skillgap_export.csv", mime="text/csv")
with col2:
    st.download_button("Download JSON", data=json_str, file_name="skillgap_export.json", mime="application/json")
with col3:
    st.download_button("Download PDF Report", data=pdf_bytes, file_name="skillgap_report.pdf", mime="application/pdf")

# Also show summary JSON on page
with st.expander("Show JSON summary", expanded=False):
    st.json(json.loads(json_str))

# ----------------------------
# Footer & tweak options
# ----------------------------
st.markdown("---")
st.caption(f"App Version: 1.0.1 • Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.info("Notes: Skill matching uses a PhraseMatcher + alias dictionary. For improved detection you can add more aliases to SKILL_ALIASES at the top of the script.")

# End of file


