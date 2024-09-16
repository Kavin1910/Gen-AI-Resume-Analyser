import streamlit as st
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import io
from transformers import pipeline

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Handle cases where extract_text() might return None
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
    st.title("Kavin's AI Resume Analyzer and Ranking")
    
    # Upload job description
    job_description_file = st.file_uploader("Upload Job Description (Text File)", type="txt")
    if job_description_file:
        job_description = job_description_file.read().decode("utf-8")
        job_description = preprocess_text(job_description)
        
        # Upload resumes
        resume_files = st.file_uploader("Upload Resumes (PDF/Word)", type=["pdf", "docx"], accept_multiple_files=True)
        
        if resume_files:
            resumes = []
            for file in resume_files:
                if file.type == "application/pdf":
                    resumes.append(preprocess_text(extract_text_from_pdf(file)))
                elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    resumes.append(preprocess_text(extract_text_from_docx(file)))
            
            if resumes:
                # Rank candidates
                ranked_indices, scores = rank_candidates(resumes, job_description)
                
                # Options to select top-N candidates
                num_candidates = st.selectbox("Select number of top candidates to display", [1, 5, 10, 15, 20])
                
                # Display results
                st.write(f"Top {num_candidates} Candidates:")
                for i in range(min(num_candidates, len(ranked_indices))):
                    candidate_name = f"Candidate {i+1}"
                    st.write(f"### {candidate_name}")
                    st.write(f"**Score: {scores[ranked_indices[i]] * 100:.2f}%**")  # Score in percentage
                    
                    # Display resume snippet in bullet points
                    resume_snippet = resumes[ranked_indices[i]].split('. ')
                    st.write(f"**Resume Snippet {i+1}:**")
                    for bullet in resume_snippet[:10]:  # Limiting to first 10 sentences
                        st.write(f"- {bullet.strip()}")
                
                # Generate and display charts
                fig, ax = plt.subplots()
                top_scores = [scores[idx] for idx in ranked_indices[:num_candidates]]
                ax.bar(range(len(top_scores)), [s * 100 for s in top_scores], color='blue')  # Scores in percentage
                ax.set_xlabel('Candidates')
                ax.set_ylabel('Scores (%)')
                ax.set_title('Resume Scores')

                # Rotate the x-axis labels for better readability
                plt.xticks(range(len(top_scores)), [f"Candidate {i+1}" for i in range(num_candidates)], rotation=45, ha="right")

                st.pyplot(fig)
                
                # Generate assessment justification
                st.write("Assessment Justification:")
                try:
                    nlp = pipeline("text-generation", model="gpt2", tokenizer="gpt2", 
                                    truncation=True, max_length=50)  # Explicit truncation and shorter max_length
                    justification_input = (
                        f"Provide a short justification for high ranking based on resumes and job description:\n"
                        f"Job Description: {job_description}\n"
                        f"Resumes: {', '.join([resumes[idx][:500] for idx in ranked_indices[:num_candidates]])}"
                    )
                    justification = nlp(justification_input, truncation=True)
                    
                    # Truncate to 200 characters to fit within two lines
                    st.write(justification[0]['generated_text'][:200])
                except Exception as e:
                    st.error(f"Error generating justification: {e}")
            else:
                st.write("No resumes uploaded.")
        else:
            st.write("Please upload resumes.")
    else:
        st.write("Please upload a job description.")

if __name__ == "__main__":
    main()
