import streamlit as st
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import io
from fpdf import FPDF  # Import FPDF for PDF generation
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
    return text.lower().strip().replace('\n', ' ')

# Function to compute relevance score
def compute_relevance_score(resumes, job_description):
    vectorizer = TfidfVectorizer()
    documents = resumes + [job_description]
    
    # Compute TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return cosine_sim.flatten()

# Function to rank candidates and justify selections
def rank_candidates(resumes, job_description):
    scores = compute_relevance_score(resumes, job_description)
    ranked_indices = np.argsort(scores)[::-1]
    return ranked_indices, scores

# Function to create a PDF report
def create_pdf_report(candidate_data, num_candidates):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="AI Resume Analyzer Report", ln=True, align='C')

    # Content
    pdf.set_font("Arial", size=12)
    
    for i in range(min(num_candidates, len(candidate_data))):
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Candidate {i+1}:", ln=True)
        pdf.cell(200, 10, txt=f"Score: {candidate_data[i]['score']}%", ln=True)
        
        # Label the resume snippet section
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, txt="Resume Snippet", ln=True)  # Label for the resume snippet box
        
        # Resume Snippet content inside a bordered box
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=f"{candidate_data[i]['snippet']}", border=1)

        # Justification
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, txt="Justification", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=f"{candidate_data[i]['justification']}", border=1)
    
    # Save PDF to BytesIO buffer
    pdf_output = io.BytesIO()
    pdf.output(pdf_output, 'S')  # Save PDF to buffer as string
    pdf_output.seek(0)
    
    return pdf_output

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
                
                # Collect candidate data for PDF report
                candidate_data = []
                for i in range(min(num_candidates, len(ranked_indices))):
                    st.write(f"**Candidate {i+1}**:")
                    score = int(scores[ranked_indices[i]] * 100)
                    st.write(f"Score: {score}%")
                    
                    # Display resume snippet in a box-like structure
                    resume_snippet = resumes[ranked_indices[i]]
                    snippets = resume_snippet.split('. ')
                    
                    # HTML for box-like structure
                    snippet_html = f"""
                    <div style="border: 1px solid #ddd; border-radius: 10px; padding: 10px; margin-bottom: 10px;">
                    <p style="font-size: 14px; font-family: 'Arial', sans-serif;">
                    {'<br>'.join(f'• {snippet.strip()}' for snippet in snippets[:3])}
                    </p>
                    </div>
                    """
                    st.markdown(snippet_html, unsafe_allow_html=True)

                    # Save data for PDF report
                    candidate_data.append({
                        'score': score,
                        'snippet': ' '.join(snippets[:3])
                    })
                
                # Generate and display charts
                fig, ax = plt.subplots(figsize=(10, 5))
                top_scores = [scores[idx] for idx in ranked_indices[:num_candidates]]
                ax.bar(range(len(top_scores)), [score * 100 for score in top_scores], color='blue')
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
                    justification = nlp(justification_prompt, max_length=150, truncation=True)
                    summary = justification[0]['generated_text']
                    # Limit to 2 lines and format as bullet points
                    lines = summary.split('\n')[:2]
                    st.markdown("• " + "\n• ".join(lines))
                except Exception as e:
                    st.error(f"Error generating justification: {e}")

                # Add "Thank You" image        
                st.image("https://images.pexels.com/photos/1887992/pexels-photo-1887992.jpeg?auto=compress&cs=tinysrgb&w=600")

                # Button to download report as PDF
                pdf_output = create_pdf_report(candidate_data, num_candidates)
                st.download_button(
                    label="Download Report as PDF",
                    data=pdf_output,
                    file_name="AI_Resume_Analyzer_Report.pdf",
                    mime="application/pdf"
                )
            else:
                st.write("No resumes uploaded.")
        else:
            st.write("Please upload resumes.")
    
    else:
        st.write("Please upload a job description.")

if __name__ == "__main__":
    main()
