import os
import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from docx import Document
from groq import Groq

# Initialize Groq client
client = Groq(api_key=os.environ.get("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"))

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

# Function to query Groq API for LLaMA
def query_groq_llama(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Function to analyze resume using LLaMA with error handling
def analyze_resume_llama(resume_text, job_description):
    prompt = f"Analyze this resume: {resume_text}. Based on the job description: {job_description}, provide a detailed justification in 50 words."
    analysis_result = query_groq_llama(prompt)
    
    # Truncate result to 50 words
    analysis_words = analysis_result.split()
    if len(analysis_words) > 50:
        analysis_result = ' '.join(analysis_words[:50]) + '...'
    
    return analysis_result

# Main Streamlit app
st.title("Kavin's Application Resume Analysis (LLaMA)")
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
        
        # Analyze resume using LLaMA
        analysis_result = analyze_resume_llama(resume_text, job_description)
        results.append({
            'file_name': uploaded_file.name,
            'analysis': analysis_result,
            'score': len(analysis_result)  # Example scoring based on length of justification
        })
    
    # Create DataFrame for visualization
    df_results = pd.DataFrame(results)
    
    # Sort candidates based on the score
    sorted_results = df_results.sort_values(by='score', ascending=False)
    
    # Display top 5 candidates
    st.write("Top 5 candidates:")
    for idx, result in enumerate(sorted_results.head(5).itertuples()):
        st.write(f"**Candidate {idx+1}: {result.file_name}**")
        st.write(f"**Justification:** {result.analysis}")
    
    # Visualize results with a bar chart using Seaborn
    st.write("Candidate Rankings:")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='score', y='file_name', data=sorted_results.head(5))
    plt.xlabel('Score')
    plt.ylabel('Candidate')
    plt.title('Top 5 Candidates by Score')
    st.pyplot()
