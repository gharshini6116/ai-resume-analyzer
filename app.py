import streamlit as st
import re
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Utility Functions ----------

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return normalize_text(text)

def extract_skills(text, skill_list):
    found = []
    for skill in skill_list:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            found.append(skill)
    return sorted(set(found))

def calculate_match(resume, job):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume, job])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)

# ---------- Skill Database ----------
skills = [
    "python", "java", "machine learning", "deep learning", "data science",
    "sql", "html", "css", "javascript", "django", "flask",
    "excel", "power bi", "communication", "problem solving"
]

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Resume Analyzer", layout="centered")
st.title("📄 AI Resume Analyzer & Job Matcher")

st.write("Upload your resume PDF and paste the job description.")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste Job Description")

if st.button("Analyze Resume"):
    if resume_file and job_desc:
        resume_text = extract_text_from_pdf(resume_file)
        job_text = normalize_text(job_desc)

        resume_skills = extract_skills(resume_text, skills)
        job_skills = extract_skills(job_text, skills)
        match_score = calculate_match(resume_text, job_text)
        missing_skills = list(set(job_skills) - set(resume_skills))

        st.subheader("📊 Match Results")
        st.metric("Match Score", f"{match_score}%")

        st.subheader("✅ Skills Found")
        st.write(", ".join(resume_skills) if resume_skills else "None")

        st.subheader("❌ Missing Skills")
        st.write(", ".join(missing_skills) if missing_skills else "None")

        if match_score >= 75:
            st.success("Strong Match – Apply with confidence")
        elif match_score >= 55:
            st.warning("Moderate Match – Upskill recommended")
        else:
            st.error("Weak Match – Resume improvement needed")
    else:
        st.warning("Please upload resume and paste job description")
