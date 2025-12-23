# ==========================================
# SkillGapAI - Milestone 3
# Skill Gap Analysis & Similarity Matching
# Final Polished UI (Compact Radar + Heatmap)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ------------------------------------------
# PAGE CONFIG
# ------------------------------------------
st.set_page_config(
    page_title="SkillGapAI – Milestone 3",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------------------
# CUSTOM CSS (SKY BLUE THEME)
# ------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom right, #e0f2fe, #bae6fd);
}
h1,h2,h3,h4,h5,h6,p,label {
    color:#0f172a;
}
.metric-box {
    background:#ffffff;
    padding:16px;
    border-radius:14px;
    text-align:center;
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
<h2 style="background:#38bdf8;color:#020617;padding:18px;border-radius:16px">
🧠 SkillGapAI – Milestone 3: Skill Gap Analysis
</h2>
<p>
<b>Objective:</b> Perform semantic skill gap analysis by comparing resume and
job description skills using Sentence-BERT embeddings and cosine similarity.
</p>
""", unsafe_allow_html=True)

# ------------------------------------------
# LOAD MODEL
# ------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ------------------------------------------
# INPUT SECTION
# ------------------------------------------
st.subheader("📥 Input Skills (from Milestone 2)")

c1, c2 = st.columns(2)
with c1:
    resume_skills = st.text_area(
        "📄 Resume Skills (comma separated)",
        "Python, SQL, Data Analysis, Data Visualization, Pandas, NumPy"
    )
with c2:
    jd_skills = st.text_area(
        "🏢 Job Description Skills (comma separated)",
        "Python, SQL, Machine Learning, Data Analysis, Deep Learning, Statistics"
    )

resume_skills = [s.strip() for s in resume_skills.split(",") if s.strip()]
jd_skills = [s.strip() for s in jd_skills.split(",") if s.strip()]

if not resume_skills or not jd_skills:
    st.stop()

# ------------------------------------------
# EMBEDDINGS & SIMILARITY
# ------------------------------------------
resume_emb = model.encode(resume_skills)
jd_emb = model.encode(jd_skills)
sim_matrix = cosine_similarity(resume_emb, jd_emb)

# ------------------------------------------
# SKILL GAP LOGIC
# ------------------------------------------
MATCH_T = 0.75
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

overall_score = int((len(matched) / len(jd_skills)) * 100)

# ------------------------------------------
# SUMMARY METRICS
# ------------------------------------------
st.subheader("📊 Summary")

a, b, c, d = st.columns(4)
a.markdown(f"<div class='metric-box'>Resume Skills<br><h2>{len(resume_skills)}</h2></div>", unsafe_allow_html=True)
b.markdown(f"<div class='metric-box'>JD Skills<br><h2>{len(jd_skills)}</h2></div>", unsafe_allow_html=True)
c.markdown(f"<div class='metric-box'>Matched<br><h2>{len(matched)}</h2></div>", unsafe_allow_html=True)
d.markdown(f"<div class='metric-box'>Overall Match<br><h2>{overall_score}%</h2></div>", unsafe_allow_html=True)

st.progress(overall_score / 100)

# ------------------------------------------
# SKILL TAG RESULTS
# ------------------------------------------
st.subheader("🧩 Skill Gap Results")

x, y, z = st.columns(3)
with x:
    st.markdown("### ✅ Matched")
    for s, sc in matched:
        st.markdown(f"<span class='tag match'>{s} ({sc:.2f})</span>", unsafe_allow_html=True)

with y:
    st.markdown("### ⚠️ Partial")
    for s, sc in partial:
        st.markdown(f"<span class='tag partial'>{s} ({sc:.2f})</span>", unsafe_allow_html=True)

with z:
    st.markdown("### ❌ Missing")
    for s, _ in missing:
        st.markdown(f"<span class='tag missing'>{s}</span>", unsafe_allow_html=True)

# ------------------------------------------
# SWEET RADAR CHART (COMPACT)
# ------------------------------------------
def plot_sweet_radar(skills, scores):
    scores = scores + scores[:1]
    angles = np.linspace(0, 2*np.pi, len(skills), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(4.8, 4.8))
    ax = plt.subplot(111, polar=True)

    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#020617")
    ax.grid(color="#38bdf8", linestyle="dotted", linewidth=1, alpha=0.6)

    ax.set_thetagrids(np.degrees(angles[:-1]), skills, fontsize=9, color="#e0f2fe")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25","0.5","0.75","1.0"], color="#93c5fd", fontsize=8)
    ax.set_ylim(0, 1)

    ax.plot(
        angles, scores,
        linewidth=2.5,
        color="#22d3ee",
        marker="o",
        markersize=6,
        markerfacecolor="#facc15"
    )
    ax.fill(angles, scores, color="#38bdf8", alpha=0.35)

    ax.set_title(
        "Skill Match Radar",
        size=12,
        color="#e0f2fe",
        pad=15,
        fontweight="bold"
    )
    return fig

# ------------------------------------------
# VISUAL DISTRIBUTION (SIDE BY SIDE)
# ------------------------------------------
st.subheader("📊 Visual Skill Distribution")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🕸️ Radar View")
    radar_scores = [sim_matrix[:, j].max() for j in range(len(jd_skills))]
    st.pyplot(plot_sweet_radar(jd_skills, radar_scores))

with col2:
    st.markdown("#### 🔥 Heatmap View")

    fig, ax = plt.subplots(figsize=(5.5, 3))
    im = ax.imshow(sim_matrix, cmap="viridis")

    ax.set_xticks(range(len(jd_skills)))
    ax.set_xticklabels(jd_skills, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(resume_skills)))
    ax.set_yticklabels(resume_skills, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig)

# ------------------------------------------
# PRIORITY RANKING
# ------------------------------------------
st.subheader("🎯 Missing Skill Priority")

if missing:
    ranked = sorted(missing, key=lambda x: x[1])
    for i, (s, sc) in enumerate(ranked, 1):
        st.write(f"{i}. **{s}** (Similarity: {sc:.2f})")
else:
    st.success("No critical skill gaps detected!")

# ------------------------------------------
# EXPORT CSV
# ------------------------------------------
st.subheader("⤓ Download Report")

df = pd.DataFrame({
    "JD Skill": [s for s,_ in matched + partial + missing],
    "Status": ["Matched"]*len(matched) + ["Partial"]*len(partial) + ["Missing"]*len(missing),
    "Similarity Score": [round(sc,2) for _,sc in matched+partial+missing]
})

st.download_button(
    "⬇️ Download Skill Gap Report (CSV)",
    df.to_csv(index=False),
    file_name="skillgap_milestone3_report.csv",
    mime="text/csv"
)

# ------------------------------------------
# FOOTER
# ------------------------------------------
st.markdown("---")
st.caption(
    f"Milestone 3 • Sentence-BERT • Cosine Similarity • Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

