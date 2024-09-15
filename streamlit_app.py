import streamlit as st
import requests
from PyPDF2 import PdfReader
from docx import Document
import os
import pandas as pd

# Hugging Face API details (consider a larger model for better analysis)
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-base"
headers = {"Authorization": f"Bearer {os.getenv('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')}"}

# Function to extract text from PDF
def extract_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from Word Doc
def extract_doc_text(doc_file):
    doc = Document(doc_file)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

# Function to query Hugging Face API with better error handling
def query_huggingface(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Check for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Hugging Face API Error: {e}")
        return {"error": str(e)}

# Function to analyze resume using Hugging Face API
def analyze_resume_hf(resume_text, job_description):
    prompt = f"Analyze this resume: {resume_text}. Based on the job description: {job_description}, is this candidate a good fit for the role? Why or why not? Provide a score (0-100)."
    payload = {"inputs": prompt}
    response = query_huggingface(payload)

    if response and 'error' not in response:
        return response[0].get('generated_text', "No text generated")
    else:
        return "Error in generating response from Hugging Face AI."

# Main Streamlit app
st.title("Illama HR Resume Analysis Tool (Hugging Face)")
st.write("Upload resumes in PDF or Word format, and we'll analyze and rank the candidates for a specific role.")

# Job Description Input
job_description = st.text_area("Enter the job description for the role (e.g., Software Developer):", height=150)

# Uploading resumes
uploaded_files = st.file_uploader("Upload resumes (PDF or Word)", accept_multiple_files=True)

if uploaded_files and job_description:
    st.write("Analyzing resumes...")

    # Store resume analysis results
    results = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.pdf'):
            resume_text = extract_pdf_text(uploaded_file)
        elif uploaded_file.name.endswith('.docx'):
            resume_text = extract_doc_text(uploaded_file)
        else:
            st.write(f"Unsupported file type: {uploaded_file.name}")
            continue

        # Analyze resume using Hugging Face API
        analysis_result = analyze_resume_hf(resume_text, job_description)

        # Extract score and key skills (optional)
        try:
            score = float(analysis_result.split('Score:')[1].split()[0])
            key_skills = analysis_result.split('Key Skills:')[1].split('\n')[0]  # Extract top skill
        except:
            score = 0
            key_skills = ""

        results.append({
            'file_name': uploaded_file.name,
            'analysis': analysis_result,
            'score': score,
            'key_skills': key_skills
        })

    # Sort candidates based on the score
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

    # Display top 5 candidates
    st.write("**Top Candidates:**")

    for idx, result in enumerate(sorted_results[:5]):
        st.write(f"**{idx+1}. {result['file_name']} (Score: {result['score']})**")
        st.write(f"**Justification:** {result['analysis']}")
        st.write(f"**Key Skills:** {result['key_skills']}")  # Display extracted key skills

    # Prepare data for chart
    chart_data = {
        'Candidate': [result['file_name'] for result in sorted_results[:5]],
        'Score': [result['score'] for result in sorted_results[:5]]
    }

    df = pd.DataFrame(chart_data)

    # Display results and chart
    st.write("Candidate Scores:")
    st.bar_chart(df.set_index('Candidate'))
