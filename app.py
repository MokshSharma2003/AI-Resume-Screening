import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

# 🎨 Custom CSS for UI Styling
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
        color: #1e3a8a;
    }
    .main-title {
        font-size: 24px !important;
        font-weight: bold;
        text-align: center;
        color: #4169E1;
    }
    .sub-header {
        font-size: 18px !important;
        font-weight: bold;
        color: #3b82f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 📌 Function to extract text from PDFs
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""

    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip() if text.strip() else "No readable text found."

# 📌 Optimized Matching Function (Fuzzy + Cosine)
def final_optimized_match(job_description, applications):
    job_keywords = job_description.lower().split()
    scores = []

    for app_text in applications:
        resume_keywords = app_text.lower().split()

        # Fuzzy Matching Score (Now 90% weight for stronger impact)
        fuzzy_score = sum([fuzz.partial_ratio(word, " ".join(resume_keywords)) for word in job_keywords]) / len(job_keywords)
        fuzzy_score = (fuzzy_score / 100) ** 0.85  # Adjusted scaling to favor higher scores

        # Cosine Similarity Score (Reduced to 40% weight)
        vectorizer = TfidfVectorizer().fit_transform([job_description] + [app_text])
        vectors = vectorizer.toarray()
        cosine_score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

        # **Final Score Calculation**
        final_score = (0.90 * fuzzy_score) + (0.40 * cosine_score)  # Higher fuzzy match weight
        final_score = max(0, min(1, final_score))  # Keep score between 0-1

        scores.append(final_score)

    return scores

# 📌 Function to classify scores
def classify_score(score):
    if score >= 0.8:
        return "⭐⭐⭐⭐⭐ Excellent Fit"
    elif score >= 0.6:
        return "⭐⭐⭐⭐ Good Fit"
    elif score >= 0.4:
        return "⭐⭐⭐ Average Fit"
    elif score >= 0.2:
        return "⭐⭐ Needs Improvement"
    else:
        return "⭐ Not Relevant"

# 🎯 UI Components
st.markdown('<p class="main-title">🚀 AI Resume Screening & Ranking System</p>', unsafe_allow_html=True)

job_description = st.text_area("📝 Enter the Job Description", height=150)

uploaded_files = st.file_uploader("📂 Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    applications = [extract_text_from_pdf(file) for file in uploaded_files]
    valid_applications = [app for app in applications if app != "No readable text found."]
    
    if valid_applications:
        metrics = final_optimized_match(job_description, valid_applications)
        metriced_applications = sorted(zip(uploaded_files, metrics), key=lambda x: x[1], reverse=True)

        st.markdown('<p class="sub-header">🏆 Ranked Resumes</p>', unsafe_allow_html=True)
        for i, (file, metric) in enumerate(metriced_applications, start=1):
            classification = classify_score(metric)
            st.write(f"**{i}. {file.name}** - 🎯 Score: **{metric:.2f}** - {classification}")
    else:
        st.error("⚠️ No readable text was found in the uploaded resumes. Please check the files and try again.")
