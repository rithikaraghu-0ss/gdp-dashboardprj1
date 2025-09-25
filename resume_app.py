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
    spacy.cli.download("en_core_web_sm", "--user")
    nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="AI-Powered Resume Screening Tool", layout="wide")
st.title("AI-Powered Resume Screening Tool")
st.write("Upload resumes (PDF/DOCX/TXT) and paste a job description to rank candidates.")

def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages)
        return text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif uploaded_file.type == "text/plain":
        return str(uploaded_file.read(), "utf-8")
    else:
        return ""

def main():
    uploaded_files = st.file_uploader("Upload resumes", type=["pdf","docx","txt"], accept_multiple_files=True)
    job_description = st.text_area("Paste the job description here")

    if uploaded_files and job_description:
        resumes_text = []
        filenames = []
        for uploaded_file in uploaded_files:
            text = extract_text(uploaded_file)
            resumes_text.append(text)
            filenames.append(uploaded_file.name)

        # Vectorize resumes and job description for similarity
        vectorizer = TfidfVectorizer(stop_words="english")
        resumes_vectors = vectorizer.fit_transform(resumes_text)
        job_desc_vector = vectorizer.transform([job_description])

        # Calculate cosine similarity
        similarity_scores = cosine_similarity(job_desc_vector, resumes_vectors).flatten() * 100
        # Prepare results dataframe
        df = pd.DataFrame({
            "Resume": filenames,
            "Similarity_Score (%)": similarity_scores
        })
        df = df.sort_values(by="Similarity_Score (%)", ascending=False)

        st.subheader("Ranked Resumes by Similarity")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results as CSV", data=csv, file_name="results.csv", mime="text/csv")

if __name__ == "__main__":
    main()
