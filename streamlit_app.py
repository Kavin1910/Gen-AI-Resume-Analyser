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

def main():
    st.title("AI Resume Analyzer and Ranking")
    
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
                    st.write(f"**Candidate {i+1}**:")
                    st.write(f"Score: {scores[ranked_indices[i]]:.2f}")
                    
                    # Display resume snippet
                    st.text_area("Resume Snippet", resumes[ranked_indices[i]][:1000], height=200)
                
                # Generate and display charts
                fig, ax = plt.subplots()
                top_scores = [scores[idx] for idx in ranked_indices[:num_candidates]]
                ax.bar(range(len(top_scores)), top_scores, color='blue')
                ax.set_xlabel('Candidates')
                ax.set_ylabel('Scores')
                ax.set_title('Resume Scores')
                plt.xticks(range(len(top_scores)), [f"Candidate {i+1}" for i in range(num_candidates)])
                st.pyplot(fig)
                
                # Generate assessment justification
                st.write("Assessment Justification:")
                try:
                    nlp = pipeline("text-generation", model="gpt2")  # Updated model name
                    justification = nlp(f"Reason for high ranking based on their resumes and job description:\nJob Description: {job_description}\nResumes: {', '.join([resumes[idx][:500] for idx in ranked_indices[:num_candidates]])}")  # Justification for top-N candidates
                    st.write(justification[0]['generated_text'])
                except Exception as e:
                    st.error(f"Error generating justification: {e}")
            else:
                st.write("No resumes uploaded.")
        else:
            st.write("Please upload resumes.")
    else:
        st.write("Please upload a job description.")

