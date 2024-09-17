import streamlit as st
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import io
from fpdf import FPDF
import openai
import os

# Set your OpenAI API key
openai.api_key = os.getenv("sk-P4MhtrIxMbYiw9RVFBBa0imlqTGC9p8KcXy9I5L8u9T3BlbkFJz9A8BOMEPf47JP-c18pZjkEJgZ6DYyGMLbvsDl7L8A")  # You can replace with your key for testing

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
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
    tfidf_matrix = vectorizer.fit_transform(documents)
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
        pdf.cell(200, 10, txt="Resume Snippet", ln=True)
        
        # Resume Snippet content inside a bordered box
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=f"{candidate_data[i]['snippet']}", border=1)

        # Justification (only include if available)
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, txt="Justification", ln=True)
        pdf.set_font("Arial", size=12)
        
        justification = candidate_data[i].get('justification', 'No justification provided')
        pdf.multi_cell(0, 10, txt=justification, border=1)
    
    # Save PDF to BytesIO buffer
    pdf_output = io.BytesIO()
    pdf.output(pdf_output, 'S')  # Save PDF to buffer as string
    pdf_output.seek(0)
    
    return pdf_output

# Function to generate assessment justification with the updated OpenAI API
def generate_chatgpt_justification(job_description, resumes, ranked_indices, num_candidates):
    justifications = []
    try:
        # Iterate through the top candidates
        for i in range(min(num_candidates, len(ranked_indices))):
            resume_snippet = resumes[ranked_indices[i]][:500]  # Take a portion of the resume
            prompt = (
                f"The following is a job description and a resume snippet. "
                f"Please provide a concise, two-line justification for why this candidate is a good fit for the job.\n\n"
                f"Job Description:\n{job_description}\n\n"
                f"Resume Snippet:\n{resume_snippet}\n\n"
                f"Justification:"
            )

            # Call OpenAI ChatGPT model using the updated API method
            response = openai.chat_completions.create(
                model="gpt-4",  # Replace with "gpt-3.5-turbo" if needed
                messages=[
                    {"role": "system", "content": "You are an expert in HR resume analysis."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
                temperature=0.7,
            )
            
            # Extract the generated justification from ChatGPT's response
            justification = response['choices'][0]['message']['content'].strip()
            justifications.append(justification)

    except Exception as e:
        st.error(f"Error generating justifications: {e}")
        justifications = ["No justification provided"] * num_candidates

    return justifications


# Streamlit application
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
                        'snippet': ' '.join(snippets[:3]),
                        'justification': ''  # Placeholder for justification
                    })

                # Generate justifications using ChatGPT
                justifications = generate_chatgpt_justification(job_description, resumes, ranked_indices, num_candidates)
                
                # Add justifications to candidate data
                for i in range(min(num_candidates, len(ranked_indices))):
                    candidate_data[i]['justification'] = justifications[i]
                
                # Debugging output
                st.write("Debugging Information:")
                st.write(f"Candidate Data: {candidate_data}")

                # Generate and display charts
                fig, ax = plt.subplots(figsize=(10, 5))
                top_scores = [scores[idx] for idx in ranked_indices[:num_candidates]]
                ax.bar(range(len(top_scores)), [score * 100 for score in top_scores], color='blue')
                ax.set_xlabel('Candidates')
                ax.set_ylabel('Scores (%)')
                ax.set_title('Resume Scores')
                plt.xticks(range(len(top_scores)), [f"Candidate {i+1}" for i in range(num_candidates)], rotation=45)
                st.pyplot(fig)

                # Button to download report as PDF
                if candidate_data:
                    pdf_output = create_pdf_report(candidate_data, num_candidates)
                    st.download_button(
                        label="Download Report as PDF",
                        data=pdf_output,
                        file_name="AI_Resume_Analyzer_Report.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.write("No candidate data available to generate PDF.")
            else:
                st.write("No resumes uploaded.")
        else:
            st.write("Please upload resumes.")
    
    else:
        st.write("Please upload a job description.")

if __name__ == "__main__":
    main()
