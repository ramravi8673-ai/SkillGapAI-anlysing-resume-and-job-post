# ==========================================
# SkillGapAI - Milestone 1: Data Ingestion & Parsing
# Weeks 1–2
# ==========================================

import streamlit as st
import docx2txt
import PyPDF2
import re
from io import BytesIO
import tempfile
import os

# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------
st.set_page_config(page_title="SkillGapAI - Milestone 1", layout="wide")

st.markdown(
    """
    <h2 style='color:white; background-color:#1E3D59; padding:15px; border-radius:10px'>
        🧠 SkillGapAI - Milestone 1: Data Ingestion & Parsing
    </h2>
    <p><b>Objective:</b> Build a system to upload resumes and job descriptions, extract and clean text, preview parsed content, and download the cleaned data.</p>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------
# FUNCTIONS
# ------------------------------------------

def clean_text(text: str) -> str:
    """Normalize text by removing extra spaces and line breaks"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_text(uploaded_file) -> str:
    """Extract plain text from PDF, DOCX, or TXT"""
    text = ""
    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".pdf"):
            # ensure pointer at start
            uploaded_file.seek(0)
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"

        elif file_name.endswith(".docx"):
            # docx2txt.process expects a path, so write a temp file
            uploaded_file.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                text = docx2txt.process(tmp_path)
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        elif file_name.endswith(".txt"):
            # reset file pointer if needed
            uploaded_file.seek(0)
            text = uploaded_file.read().decode("utf-8", errors="ignore")

        else:
            st.error("❌ Unsupported file format. Please upload PDF, DOCX, or TXT.")
            return ""

        return clean_text(text)

    except Exception as e:
        st.error(f"⚠ Error extracting text: {e}")
        return ""
    



# ------------------------------------------
# LAYOUT
# ------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📤 Upload Resume or Job Description File")
    uploaded_file = st.file_uploader(
        "Choose a file (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"]
    )
    st.info("Supported formats: PDF • DOCX • TXT")

with col2:
    st.markdown("### 🧾 Parsed Document Preview")

    if uploaded_file:
        with st.spinner("🔍 Extracting and cleaning text..."):
            extracted_text = extract_text(uploaded_file)

        if extracted_text:
            st.success(f"✅ Successfully parsed: {uploaded_file.name}")
            st.text_area("Extracted & Cleaned Text", extracted_text[:4000], height=350)
            st.caption(f"Characters: {len(extracted_text)} | Words: {len(extracted_text.split())}")

            # --- Download Parsed Data ---
            st.download_button(
                label="💾 Download Parsed Text",
                data=extracted_text,
                file_name=f"parsed_{uploaded_file.name.rsplit('.', 1)[0]}.txt",
                mime="text/plain"
            )
        else:
            st.warning("⚠ No text extracted from the file. Try another format or ensure it contains readable text.")

    else:
        st.warning("Upload a file to see and download the parsed text preview here.")


# ------------------------------------------
# MANUAL JOB DESCRIPTION SECTION
# ------------------------------------------
st.markdown("---")
st.subheader("📋 Paste Job Description (Optional)")

jd_text = st.text_area("Paste Job Description here:", "", height=200)

if jd_text.strip():
    cleaned_jd = clean_text(jd_text)
    st.text_area("Cleaned Job Description Output", cleaned_jd, height=200)
    st.caption(f"Characters: {len(cleaned_jd)} | Words: {len(cleaned_jd.split())}")

    st.download_button(
        label="💾 Download Cleaned Job Description",
        data=cleaned_jd,
        file_name="cleaned_job_description.txt",
        mime="text/plain"
    )

# ------------------------------------------
# FOOTER
# ------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Milestone 1 • Data Ingestion & Parsing • SkillGapAI Project • Developed by P K RAVI </p>",
    unsafe_allow_html=True
)
