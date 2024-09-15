import streamlit as st
import requests
from PyPDF2 import PdfReader
from docx import Document

# Hugging Face API details for GPT-2
API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"
HUGGING_FACE_API_KEY = "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"
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

# Function to query Hugging Face API for GPT-2 with input truncation
def query_huggingface_gpt2(prompt):
    # Truncate the prompt to a reasonable length to avoid API issues
    max_token_length = 1000
    prompt = prompt[:max_token_length]  # Truncate to avoid exceeding token limit
    
    payload = {"inputs": prompt}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Function to analyze resume using GPT-2 with error handling
def analyze_resume_gpt2(resume_text, job_description):
    prompt = f"Analyze this resume: {resume_text}. Based on the job description: {job_description}, provide a detailed justification."
    response = query_huggingface_gpt2(prompt)
    
    if 'error' not in response:
        if isinstance(response, list) and len(response) > 0:
            return response[0].get('generated_text', 'No analysis generated.')
        else:
            return "Unexpected response format from GPT-2."
    else:
        return f"Error: {response['error']}"

# Main Streamlit app
st.title("Illama HR Resume Analysis Tool (GPT-2)")
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
        
        # Analyze resume using GPT-2
        analysis_result = analyze_resume_gpt2(resume_text, job_description)
        results.append({
            'file_name': uploaded_file.name,
            'analysis': analysis_result
        })
    
    # Sort candidates based on the analysis (Here, it's a placeholder to sort by length of response)
    sorted_results = sorted(results, key=lambda x: len(x['analysis']), reverse=True)

    # Display top 5 candidates
    st.write("Top 5 candidates:")
    for idx, result in enumerate(sorted_results[:5]):
        st.write(f"**Candidate {idx+1}: {result['file_name']}**")
        st.write(f"**Justification:** {result['analysis']}")
