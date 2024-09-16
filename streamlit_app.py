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

# Function to format resume snippet as bullet points (max 3 lines)
def format_resume_snippet(snippet):
    snippet_lines = snippet.split('. ')
    return "\n".join([f"• {line.strip()}" for line in snippet_lines[:3]])

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
                    st.write(f"**Candidate {i+1}**:")
                    st.write(f"Score: {scores[ranked_indices[i]]:.2f}%")
                    
                    # Display resume snippet
                    st.text_area(f"Resume Snippet {i+1}", format_resume_snippet(resumes[ranked_indices[i]]), height=150)
                
                # Generate and display charts
                fig, ax = plt.subplots(figsize=(10, 5))
                top_scores = [scores[idx] for idx in ranked_indices[:num_candidates]]
                ax.bar(range(len(top_scores)), top_scores, color='blue')
                ax.set_xlabel('Candidates')
                ax.set_ylabel('Scores (%)')
                ax.set_title('Resume Scores')
                plt.xticks(range(len(top_scores)), [f"Candidate {i+1}" for i in range(num_candidates)], rotation=45)
                st.pyplot(fig)
                
                # Generate assessment justification
                st.write("Assessment Justification:")
                try:
                    nlp = pipeline("text-generation", model="gpt2")
                    justification_prompt = (
                        f"Generate a 2-line summary of why the following resumes are a good match for the job description:\n\n"
                        f"Job Description: {job_description}\n\n"
                        f"Resumes: {', '.join([resumes[idx][:500] for idx in ranked_indices[:num_candidates]])}"
                    )
                    justification = nlp(justification_prompt, max_new_tokens=100)  # Adjust max_new_tokens as needed
                    summary = justification[0]['generated_text']
                    # Limit to 2 lines and format as bullet points
                    lines = summary.split('\n')[:2]
                    st.markdown("• " + "\n• ".join(lines))
                except Exception as e:
                    st.error(f"Error generating justification: {e}")
            
            else:
                st.write("No resumes uploaded.")
        else:
            st.write("Please upload resumes.")
        
        # Add "Thank You" image        
        st.image("https://images.pexels.com/photos/1887992/pexels-photo-1887992.jpeg?auto=compress&cs=tinysrgb&w=600") 
    else:
        st.write("Please upload a job description.")

if __name__ == "__main__":
    main()
