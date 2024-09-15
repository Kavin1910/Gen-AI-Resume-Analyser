import streamlit as st
import requests
from PyPDF2 import PdfReader
from docx import Document
import os
import pandas as pd

# Check for Hugging Face API Key (replace with your actual check)
if not os.getenv("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"):
    st.error("Missing Hugging Face API Key! Set the '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' environment variable.")
    exit()

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
uploaded_files = st.file_uploader("Upload resumes (PDF or
