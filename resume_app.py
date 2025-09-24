import io
import streamlit as st
import pandas as pd
import pdfplumber
from docx import Document
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy English model with auto-download if not present
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="AI-Powered Resume Screening Tool", layout="wide")

st.title("AI-Powered Resume Screening Tool")
st.write("Upload resumes (PDF/DOCX/TXT) and paste a job description to rank candidates.")

def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        # Assume plain text or txt file
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return str(data)

def extract_skills(text, topn=10):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE", "LANGUAGE"]]
    unique_skills = list(set(skills))
    return ", ".join(unique_skills[:topn]) if unique_skills else "-"

job_desc = st.text_area("Paste Job Description Here", height=200)
uploaded_files = st.file_uploader("Upload Resumes (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if st.button("Analyze"):
    if not job_desc or not uploaded_files:
        st.error("Please enter a job description and upload at least one resume file.")
    else:
        resumes = []
        names = []
        skills_list = []
        progress_text = st.empty()
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)
        for idx, file in enumerate(uploaded_files):
            progress_text.text(f"Processing file {idx+1} of {total_files}: {file.name}")
            text = extract_text(file)
            resumes.append(text)
            names.append(file.name)
            skills_list.append(extract_skills(text))
            progress_bar.progress((idx+1)/total_files)

        # Vectorize resumes and job description
        vectorizer = TfidfVectorizer(stop_words="english")
        docs = [job_desc] + resumes
        tfidf_matrix = vectorizer.fit_transform(docs)

        jd_vec = tfidf_matrix[0:1]
        resume_vecs = tfidf_matrix[1:]
        similarities = cosine_similarity(jd_vec, resume_vecs).flatten()

        df = pd.DataFrame({
            "Resume": names,
            "Similarity_Score": similarities,
            "Extracted_Skills": skills_list
        })
        df["Similarity_Score (%)"] = (df["Similarity_Score"] * 100).round(2)
        df = df.sort_values(by="Similarity_Score", ascending=False).reset_index(drop=True)

        st.subheader("Ranked Resumes by Similarity")
        st.dataframe(df[["Resume", "Similarity_Score (%)", "Extracted_Skills"]])

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results as CSV", data=csv, file_name="ranked_resumes.csv", mime="text/csv")



