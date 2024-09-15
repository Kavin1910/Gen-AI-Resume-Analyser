import streamlit as st
import os
from groq import Groq
from PyPDF2 import PdfReader
from docx import Document
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize Groq client with API key from Streamlit secrets
GROQ_API_KEY = st.secrets["groq"]["api_key"]
client = Groq(api_key=GROQ_API_KEY)

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

# Function to query Groq API for LLaMA with input truncation
def query_groq_llama(prompt):
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Function to analyze resume using Groq LLaMA with error handling
def analyze_resume_llama(resume_text, job_description):
    prompt = f"Analyze this resume: {resume_text}. Based on the job description: {job_description}, provide a detailed justification (50 words only)."
    response = query_groq_llama(prompt)
    return response[:500]  # Truncate response to 50 words

# Main Streamlit app
st.title("Kavin's Resume Analysis Tool (Groq & LLaMA)")
st.write("Upload resumes in PDF or Word format, and we'll analyze and rank the candidates for a specific role.")

# Job Description Input
job_description = st.text_area("Enter the job description for the role (e.g., Human Resources Manager):", height=150)

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

        # Analyze resume using Groq LLaMA
        analysis_result = analyze_resume_llama(resume_text, job_description)
        results.append({
            'file_name': uploaded_file.name,
            'analysis': analysis_result
        })

    # Display results and rankings
    st.write("Top 5 candidates:")

    # Create a DataFrame for visualization
    import pandas as pd

    # Assuming analysis result length as a score for ranking
    df = pd.DataFrame({
        'Candidate': [result['file_name'] for result in results],
        'Justification Length': [len(result['analysis']) for result in results]
    })

    # Sort candidates based on the analysis length
    sorted_results = df.sort_values(by='Justification Length', ascending=False).head(5)

    # Plot rankings with seaborn
    st.write("Candidate Rankings")
    sns.barplot(data=sorted_results, x='Justification Length', y='Candidate')
    st.pyplot()
