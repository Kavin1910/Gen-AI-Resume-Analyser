import streamlit as st
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import io
import pdfkit
import openai
import os

# Directly set your OpenAI API key
openai.api_key = "sk-OK8DS4e_lJAMjEq8sHIOP_SsOX5Tl_ow1smg04TpA4T3BlbkFJvL8gmfDukHuHuuaZJhwhSqqWVah_eSkBL9dRyx6rYA"  # Replace with your actual API key

# Ensure your API key is correctly set
if not openai.api_key:
    st.error("OpenAI API key is not set. Please check the API key configuration.")

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
    # Create a HTML template for the report
    html_template = """
    <html>
    <head>
    <title>AI Resume Analyzer Report</title>
    <style>
    body {
        font-family: Arial, sans-serif;
    }
    </style>
    </head>
    <body>
    <h1>AI Resume Analyzer Report</h1>
    {content}
    </body>
    </html>
    """

    # Generate the content for the report
    content = ""
    for i in range(min(num_candidates, len(candidate_data))):
        content += f"<h2>Candidate {i+1}</h2>"
        content += f"<p>Score: {candidate_data[i]['score']}%</p>"
        content += f"<h3>Resume Snippet</h3>"
        content += f"<p>{candidate_data[i]['snippet']}</p>"
        content += f"<h3>Justification</h3>"
        content += f"<p>{candidate_data[i]['justification']}</p>"

    # Render the HTML template with the content
    html_report = html_template.format(content=content)

    # Use pdfkit to generate the PDF report
    options = {
        'page-size': 'A4',
        'margin-top': '0.75in',
        'margin-right': '0.75in',
        'margin-bottom': '0.75in',
        'margin-left': '0.75in',
        'encoding': "UTF-8",
        'no-outline': None
    }
    pdf_output = pdfkit.from_string(html_report, False, options=options)

    return pdf_output

# Function to generate assessment justification using the OpenAI API
def generate_chatgpt_justification(job_description, resumes, ranked_indices, num_candidates):
    justifications = []
    try:
        # Ensure the API key is correctly set
        if not openai.api_key:
            raise ValueError("OpenAI API key is not set.")

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

            # Call OpenAI ChatGPT model using the correct method
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Ensure you are using the correct model name
                messages=[
                    {"role": "system", "content": "You are an expert in resume analysis."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.7,
            )
            
            # Extract the generated justification from the response
            justification = response.choices[0].message['content'].strip()
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
