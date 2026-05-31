import streamlit as st
import PyPDF2
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# Extract text from PDF
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    words = text.split()
    try:
        from nltk.corpus import stopwords
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

# Calculate similarity
def calculate_score(resume, job_desc):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([resume, job_desc])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(score[0][0] * 100, 2)

# UI
st.title("📄 AI Resume Analyzer")

st.write("Upload your resume and paste job description")

resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
job_description = st.text_area("Paste Job Description")

if resume_file and job_description:
    resume_text = extract_text(resume_file)

    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(job_description)

    score = calculate_score(cleaned_resume, cleaned_jd)

    st.subheader("🔍 Match Score")
    st.success(f"{score}% match")

    if score > 75:
        st.write("✅ Strong match!")
    elif score > 50:
        st.write("⚠️ Moderate match.")
    else:
        st.write("❌ Low match.")