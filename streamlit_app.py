import streamlit as st
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import io
from transformers import pipelineimport streamlit as st
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns  # For improved visual aesthetics
import io
from transformers import pipeline

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from Word documents
def extract_text_from_docx(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text

# Function to preprocess text
def preprocess_text(text):
    return text.lower().replace('\n', ' ')

# Function to compute relevance score
def compute_relevance_score(resumes, job_description):
    vectorizer = TfidfVectorizer()
    documents = resumes + [job_description]
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return cosine_sim.flatten()

# Function to rank candidates and justify selections
def rank_candidates(resumes, job_description):
    scores = compute_relevance_score(resumes, job_description)
    ranked_indices = np.argsort(scores)[::-1]
    return ranked_indices, scores

# Streamlit application
def main():
    st.title("Kavin's AI Resume Analyser")

    # Upload job description
    job_description_file = st.file_uploader("Upload Job Description (Text File)", type="txt", key="job_description_uploader")
    if job_description_file:
        job_description = job_description_file.read().decode("utf-8")
        job_description = preprocess_text(job_description)
        
        # Upload resumes
        resume_files = st.file_uploader("Upload Resumes (PDF/Word)", type=["pdf", "docx"], accept_multiple_files=True, key="resume_uploader")
        
        if resume_files:
            resumes = []
            for idx, file in enumerate(resume_files):
                if file.type == "application/pdf":
                    resumes.append(preprocess_text(extract_text_from_pdf(file)))
                elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    resumes.append(preprocess_text(extract_text_from_docx(file)))
            
            if resumes:
                # Rank candidates
                ranked_indices, scores = rank_candidates(resumes, job_description)
                
                # Options to select top-N candidates
                num_candidates = st.selectbox("Select number of top candidates to display", [1, 5, 10, 15, 20], key="num_candidates_selectbox")
                
                # Display results
                st.write(f"Top {num_candidates} Candidates:")
                for i in range(min(num_candidates, len(ranked_indices))):
                    st.write(f"**Candidate {i+1}**:")
                    st.write(f"Score: {scores[ranked_indices[i]]:.2f}")
                    
                    # Display resume snippet with unique key
                    st.text_area(f"Resume Snippet {i+1}", resumes[ranked_indices[i]][:1000], height=200, key=f"resume_snippet_{i}")
                
                # Generate and display charts
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=np.arange(len(scores)), y=scores[ranked_indices], palette="viridis", ax=ax)
                ax.set_xlabel('Candidates')
                ax.set_ylabel('Scores')
                ax.set_title('Resume Scores')
                ax.set_xticks(np.arange(len(scores)))
                ax.set_xticklabels([f"Candidate {i+1}" for i in ranked_indices], rotation=45, ha='right')
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                # Generate assessment justification
                st.write("Assessment Justification:")
                nlp = get_text_generator()
                if nlp:
                    justification_input = (
                        f"Justify why these candidates are ranked high based on their resumes and job description:\n"
                        f"Job Description: {job_description}\n"
                        f"Resumes: {' | '.join(resumes[ranked_indices[i]][:500] for i in range(min(num_candidates, len(ranked_indices))))}"
                    )
                    try:
                        justification = nlp(justification_input, max_length=500, truncation=True)
                        st.write(justification[0]['generated_text'])
                    except Exception as e:
                        st.error(f"Error generating text: {e}")
            else:
                st.write("No resumes uploaded.")
        else:
            st.write("Please upload resumes.")
    else:
        st.write("Please upload a job description.")

def get_text_generator():
    try:
        return pipeline("text-generation", model="gpt2")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    main()


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from Word documents
def extract_text_from_docx(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text

# Function to preprocess text
def preprocess_text(text):
    return text.lower().replace('\n', ' ')

# Function to compute relevance score
def compute_relevance_score(resumes, job_description):
    vectorizer = TfidfVectorizer()
    documents = resumes + [job_description]
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return cosine_sim.flatten()

# Function to rank candidates and justify selections
def rank_candidates(resumes, job_description):
    scores = compute_relevance_score(resumes, job_description)
    ranked_indices = np.argsort(scores)[::-1]
    return ranked_indices, scores

# Streamlit application
def main():
    st.title("Kavin's AI Resume Analyser")

    # Upload job description
    job_description_file = st.file_uploader("Upload Job Description (Text File)", type="txt", key="job_description_uploader")
    if job_description_file:
        job_description = job_description_file.read().decode("utf-8")
        job_description = preprocess_text(job_description)
        
        # Upload resumes
        resume_files = st.file_uploader("Upload Resumes (PDF/Word)", type=["pdf", "docx"], accept_multiple_files=True, key="resume_uploader")
        
        if resume_files:
            resumes = []
            for idx, file in enumerate(resume_files):
                if file.type == "application/pdf":
                    resumes.append(preprocess_text(extract_text_from_pdf(file)))
                elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    resumes.append(preprocess_text(extract_text_from_docx(file)))
            
            if resumes:
                # Rank candidates
                ranked_indices, scores = rank_candidates(resumes, job_description)
                
                # Options to select top-N candidates
                num_candidates = st.selectbox("Select number of top candidates to display", [1, 5, 10, 15, 20], key="num_candidates_selectbox")
                
                # Display results
                st.write(f"Top {num_candidates} Candidates:")
                for i in range(min(num_candidates, len(ranked_indices))):
                    st.write(f"**Candidate {i+1}**:")
                    st.write(f"Score: {scores[ranked_indices[i]]:.2f}")
                    
                    # Display resume snippet with unique key
                    st.text_area(f"Resume Snippet {i+1}", resumes[ranked_indices[i]][:1000], height=200, key=f"resume_snippet_{i}")
                
                # Generate and display charts
                fig, ax = plt.subplots()
                ax.bar(range(len(scores)), scores[ranked_indices], color='blue')
                ax.set_xlabel('Candidates')
                ax.set_ylabel('Scores')
                ax.set_title('Resume Scores')
                plt.xticks(range(len(scores)), [f"Candidate {i+1}" for i in ranked_indices])
                st.pyplot(fig)
                
                # Generate assessment justification
                st.write("Assessment Justification:")
                nlp = get_text_generator()
                if nlp:
                    justification_input = (
                        f"Justify why these candidates are ranked high based on their resumes and job description:\n"
                        f"Job Description: {job_description}\n"
                        f"Resumes: {' | '.join(resumes[ranked_indices[i]][:500] for i in range(min(num_candidates, len(ranked_indices))))}"
                    )
                    try:
                        justification = nlp(justification_input, max_length=500, truncation=True)
                        st.write(justification[0]['generated_text'])
                    except Exception as e:
                        st.error(f"Error generating text: {e}")
            else:
                st.write("No resumes uploaded.")
        else:
            st.write("Please upload resumes.")
    else:
        st.write("Please upload a job description.")

def get_text_generator():
    try:
        return pipeline("text-generation", model="gpt2")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    main()
