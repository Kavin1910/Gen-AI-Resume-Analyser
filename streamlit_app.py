import streamlit as st
import requests
from PyPDF2 import PdfReader
from docx import Document
import os
import pandas as pd

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/distilgpt2"  # Correct URL for DistilGPT-2
HUGGING_FACE_API_KEY = os.getenv("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")  # Ensure this environment variable is set
headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}

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

# Function to query Hugging Face API
def query_huggingface(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Check for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return {"error": str(e)}

# Function to analyze resume using Hugging Face API
def analyze_resume_hf(resume_text, job_description):
    prompt = f"Analyze this resume: {resume_text}. Based on the job description: {job_description}, provide a detailed score (0-100) and reasoning for why this candidate is a good fit or not for the role."
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
        
        # Extract score from analysis_result
        try:
            # Improved extraction of score
            score_line = [line for line in analysis_result.split('\n') if 'Score' in line]
            score = float(score_line[0].split(':')[1].strip()) if score_line else 0
        except Exception as e:
            st.error(f"Error parsing score: {e}")
            score = 0  # Default to 0 if score extraction fails

        results.append({
            'file_name': uploaded_file.name,
            'analysis': analysis_result,
            'score': score
        })
    
    # Sort candidates based on the score
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

    # Display top 5 candidates
    st.write("Top 5 candidates:")
    
    # Prepare data for chart
    chart_data = {
        'Candidate': [result['file_name'] for result in sorted_results[:5]],
        'Score': [result['score'] for result in sorted_results[:5]]
    }
    
    df = pd.DataFrame(chart_data)
    
    # Display results and chart
    for idx, result in enumerate(sorted_results[:5]):
        st.write(f"**Candidate {idx+1}: {result['file_name']}**")
        st.write(f"**Justification:** {result['analysis']}")
    
    st.write("Candidate Scores:")
    st.bar_chart(df.set_index('Candidate'))
