# ==========================================
# SkillGapAI - Fully Integrated Project
# Milestone 1–4 in a Single Application
# ==========================================

import streamlit as st
import spacy
from spacy.matcher import PhraseMatcher
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ------------------------------------------
# PAGE CONFIG
# ------------------------------------------
st.set_page_config(
    page_title="SkillGapAI – Integrated System",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------------------
# UI THEME
# ------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom right, #e0f2fe, #bae6fd);
}
h1,h2,h3,h4,h5,h6,p {
    color:#0f172a;
}
.card {
    background:#ffffff;
    padding:16px;
    border-radius:14px;
    border:1px solid #93c5fd;
    box-shadow:0 8px 20px rgba(0,0,0,0.08);
}
.tag {
    display:inline-block;
    padding:6px 14px;
    margin:6px 6px 6px 0;
    border-radius:999px;
    font-weight:600;
    font-size:0.85rem;
}
.match { background:#bbf7d0; color:#065f46; }
.partial { background:#fde68a; color:#92400e; }
.missing { background:#fecaca; color:#7f1d1d; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# HEADER
# ------------------------------------------
st.markdown("""
<h2 style="background:#38bdf8;color:#020617;padding:18px;border-radius:18px">
🧠 SkillGapAI – Integrated Resume & Job Skill Gap Analysis
</h2>
<p>
<b>Integrated Milestones 1–4:</b> Input → Skill Extraction → Semantic Matching → Dashboard & Recommendations
</p>
""", unsafe_allow_html=True)

# =====================================================
# MILESTONE 1 – INPUT
# =====================================================
st.subheader("📥 Milestone 1: Resume & Job Description Input")

col1, col2 = st.columns(2)
with col1:
    resume_text = st.text_area(
        "📄 Resume Text",
        height=220,
        placeholder="Paste resume content here"
    )

with col2:
    jd_text = st.text_area(
        "🏢 Job Description Text",
        height=220,
        placeholder="Paste job description here"
    )

if not resume_text or not jd_text:
    st.info("Enter both Resume and Job Description to proceed.")
    st.stop()

# =====================================================
# MILESTONE 2 – SKILL EXTRACTION
# =====================================================
st.subheader("🧠 Milestone 2: Skill Extraction")

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

SKILLS = [
    "Python", "SQL", "Machine Learning", "Data Analysis", "Deep Learning",
    "Statistics", "Power BI", "Tableau", "Communication", "Teamwork"
]

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
for skill in SKILLS:
    matcher.add(skill, [nlp(skill)])

def extract_skills(text):
    doc = nlp(text)
    matches = matcher(doc)
    return sorted(set(nlp.vocab.strings[m[0]] for m in matches))

resume_skills = extract_skills(resume_text)
jd_skills = extract_skills(jd_text)

c1, c2 = st.columns(2)
with c1:
    st.markdown("### 📄 Resume Skills")
    for s in resume_skills:
        st.markdown(f"<span class='tag match'>{s}</span>", unsafe_allow_html=True)

with c2:
    st.markdown("### 🏢 Job Description Skills")
    for s in jd_skills:
        st.markdown(f"<span class='tag match'>{s}</span>", unsafe_allow_html=True)

if not resume_skills or not jd_skills:
    st.warning("Skills not detected properly. Try richer text.")
    st.stop()

# =====================================================
# MILESTONE 3 – SEMANTIC SKILL GAP ANALYSIS
# =====================================================
st.subheader("📊 Milestone 3: Semantic Skill Gap Analysis")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

res_emb = model.encode(resume_skills)
jd_emb = model.encode(jd_skills)
sim_matrix = cosine_similarity(res_emb, jd_emb)

MATCH_T = 0.70
PARTIAL_T = 0.50

matched, partial, missing = [], [], []

for j, skill in enumerate(jd_skills):
    score = sim_matrix[:, j].max()
    if score >= MATCH_T:
        matched.append((skill, score))
    elif score >= PARTIAL_T:
        partial.append((skill, score))
    else:
        missing.append((skill, score))

overall_score = int(((len(matched) + len(partial)) / len(jd_skills)) * 100)

# ------------------------------------------
# SKILL TAG VIEW
# ------------------------------------------
colA, colB, colC = st.columns(3)

with colA:
    st.markdown("### ✅ Matched Skills")
    for s, sc in matched:
        st.markdown(f"<span class='tag match'>{s} ({sc:.2f})</span>", unsafe_allow_html=True)

with colB:
    st.markdown("### ⚠️ Partial Skills")
    for s, sc in partial:
        st.markdown(f"<span class='tag partial'>{s} ({sc:.2f})</span>", unsafe_allow_html=True)

with colC:
    st.markdown("### ❌ Missing Skills")
    for s, _ in missing:
        st.markdown(f"<span class='tag missing'>{s}</span>", unsafe_allow_html=True)

st.progress(overall_score / 100)
st.success(f"Overall Skill Match: {overall_score}%")

# ------------------------------------------
# RADAR + HEATMAP (COMPACT)
# ------------------------------------------
st.subheader("📈 Skill Distribution Visuals")

col1, col2 = st.columns(2)

with col1:
    def radar_chart(skills, scores):
        scores = scores + scores[:1]
        angles = np.linspace(0, 2*np.pi, len(skills), endpoint=False).tolist()
        angles += angles[:1]
        fig = plt.figure(figsize=(4.5,4.5))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, scores, linewidth=2)
        ax.fill(angles, scores, alpha=0.3)
        ax.set_thetagrids(np.degrees(angles[:-1]), skills)
        ax.set_ylim(0,1)
        return fig

    radar_scores = [sim_matrix[:, j].max() for j in range(len(jd_skills))]
    st.pyplot(radar_chart(jd_skills, radar_scores))

with col2:
    fig, ax = plt.subplots(figsize=(5,3))
    im = ax.imshow(sim_matrix, cmap="viridis")
    ax.set_xticks(range(len(jd_skills)))
    ax.set_xticklabels(jd_skills, rotation=30, ha="right")
    ax.set_yticks(range(len(resume_skills)))
    ax.set_yticklabels(resume_skills)
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

# =====================================================
# MILESTONE 4 – DASHBOARD & RECOMMENDATIONS
# =====================================================
st.subheader("🚀 Milestone 4: Final Dashboard & Recommendations")

st.markdown(f"""
<div class='card'>
<b>Career Readiness:</b><br>
Overall Match Score: <b>{overall_score}%</b>
</div>
""", unsafe_allow_html=True)

if overall_score >= 75:
    st.success("You are well-aligned for this role.")
elif overall_score >= 50:
    st.warning("Moderate alignment. Improve missing skills.")
else:
    st.error("Low alignment. Significant upskilling required.")

st.markdown("### 📚 Recommended Skills to Learn")
if missing:
    for s, _ in missing:
        st.write(f"• {s}")
else:
    st.success("No skill gaps identified!")

# ------------------------------------------
# FINAL REPORT EXPORT
# ------------------------------------------
st.subheader("⤓ Download Final Report")

df = pd.DataFrame({
    "Skill": [s for s,_ in matched + partial + missing],
    "Status": (
        ["Matched"]*len(matched) +
        ["Partial"]*len(partial) +
        ["Missing"]*len(missing)
    )
})

st.download_button(
    "⬇️ Download SkillGapAI Report (CSV)",
    df.to_csv(index=False),
    file_name="SkillGapAI_Final_Report.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption(f"SkillGapAI Integrated System • Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

