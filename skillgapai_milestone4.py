# ==========================================
# SkillGapAI - Milestone 4
# Final Dashboard & Recommendations
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ------------------------------------------
# PAGE CONFIG
# ------------------------------------------
st.set_page_config(
    page_title="SkillGapAI – Milestone 4",
    page_icon="🚀",
    layout="wide"
)

# ------------------------------------------
# CUSTOM CSS
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
    padding:18px;
    border-radius:16px;
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
.good { background:#bbf7d0; color:#065f46; }
.warn { background:#fde68a; color:#92400e; }
.bad { background:#fecaca; color:#7f1d1d; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# HEADER
# ------------------------------------------
st.markdown("""
<h2 style="background:#38bdf8;color:#020617;padding:18px;border-radius:18px">
🚀 SkillGapAI – Milestone 4: Final Dashboard & Recommendations
</h2>
<p>
<b>Objective:</b> Provide a consolidated skill gap summary, insights, and learning
recommendations based on earlier milestones.
</p>
""", unsafe_allow_html=True)

# ------------------------------------------
# INPUT SECTION (FROM MILESTONE 3 OUTPUT)
# ------------------------------------------
st.subheader("📥 Skill Gap Summary Input")

col1, col2, col3 = st.columns(3)

with col1:
    matched_skills = st.text_area(
        "✅ Matched Skills",
        "Python, SQL, Data Analysis"
    )

with col2:
    partial_skills = st.text_area(
        "⚠️ Partially Matched Skills",
        "Data Visualization"
    )

with col3:
    missing_skills = st.text_area(
        "❌ Missing Skills",
        "Machine Learning, Deep Learning, Statistics"
    )

matched = [s.strip() for s in matched_skills.split(",") if s.strip()]
partial = [s.strip() for s in partial_skills.split(",") if s.strip()]
missing = [s.strip() for s in missing_skills.split(",") if s.strip()]

total_skills = len(matched) + len(partial) + len(missing)
match_percentage = int((len(matched) / total_skills) * 100) if total_skills else 0

# ------------------------------------------
# DASHBOARD METRICS
# ------------------------------------------
st.subheader("📊 Final Evaluation Summary")

a, b, c, d = st.columns(4)

a.markdown(f"<div class='card'>Matched Skills<br><h2>{len(matched)}</h2></div>", unsafe_allow_html=True)
b.markdown(f"<div class='card'>Partial Skills<br><h2>{len(partial)}</h2></div>", unsafe_allow_html=True)
c.markdown(f"<div class='card'>Missing Skills<br><h2>{len(missing)}</h2></div>", unsafe_allow_html=True)
d.markdown(f"<div class='card'>Overall Match<br><h2>{match_percentage}%</h2></div>", unsafe_allow_html=True)

st.progress(match_percentage / 100)

# ------------------------------------------
# SKILL STATUS VIEW
# ------------------------------------------
st.subheader("🧩 Skill Status Overview")

colA, colB, colC = st.columns(3)

with colA:
    st.markdown("### ✅ Strengths")
    for s in matched:
        st.markdown(f"<span class='tag good'>{s}</span>", unsafe_allow_html=True)

with colB:
    st.markdown("### ⚠️ Needs Improvement")
    for s in partial:
        st.markdown(f"<span class='tag warn'>{s}</span>", unsafe_allow_html=True)

with colC:
    st.markdown("### ❌ Skill Gaps")
    for s in missing:
        st.markdown(f"<span class='tag bad'>{s}</span>", unsafe_allow_html=True)

# ------------------------------------------
# LEARNING RECOMMENDATIONS
# ------------------------------------------
st.subheader("📚 Learning Recommendations")

if missing:
    for s in missing:
        st.markdown(f"""
        <div class='card'>
        <b>{s}</b><br>
        Recommended Action:
        <ul>
            <li>Enroll in beginner-to-advanced online courses</li>
            <li>Practice with hands-on projects</li>
            <li>Apply concepts in real-world scenarios</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
else:
    st.success("No learning recommendations required!")

# ------------------------------------------
# CAREER INSIGHT
# ------------------------------------------
st.subheader("🎯 Career Readiness Insight")

if match_percentage >= 75:
    st.success("You are well-aligned with the job role and ready to apply.")
elif match_percentage >= 50:
    st.warning("You are moderately aligned. Upskilling is recommended.")
else:
    st.error("Significant skill gaps detected. Focus on learning before applying.")

# ------------------------------------------
# FINAL REPORT PREVIEW
# ------------------------------------------
st.subheader("📄 Final Report Preview")

report_df = pd.DataFrame({
    "Skill": matched + partial + missing,
    "Status": (
        ["Matched"] * len(matched) +
        ["Partial"] * len(partial) +
        ["Missing"] * len(missing)
    )
})

st.dataframe(report_df, use_container_width=True)

# ------------------------------------------
# EXPORT FINAL REPORT
# ------------------------------------------
st.subheader("⤓ Download Final Report")

csv_data = report_df.to_csv(index=False)
st.download_button(
    "⬇️ Download Final Skill Gap Report (CSV)",
    csv_data,
    file_name="SkillGapAI_Final_Report.csv",
    mime="text/csv"
)

# ------------------------------------------
# FOOTER
# ------------------------------------------
st.markdown("---")
st.caption(
    f"Milestone 4 • Final Dashboard • Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

