import streamlit as st
import requests
from PyPDF2 import PdfReader
from docx import Document
import matplotlib.pyplot as plt

# Hugging Face API details for GPT-2
API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"
HUGGING_FACE_API_KEY = "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"  # Replace with your actual key
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
    # Truncate the prompt to avoid exceeding token limit
    max_token_length = 1000
    prompt = prompt[:max_token_length]  # Limit the prompt length
    
    payload = {"inputs": prompt}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Function to truncate the generated text to 50 words
def truncate_to_50_words(text):
    words = text.split()
    return ' '.join(words[:50]) + ('...' if len(words) > 50 else '')

# Function to analyze resume using GPT-2 with error handling
def analyze_resume_gpt2(resume_text, job_description):
    prompt = f"Analyze this resume: {resume_text}. Based on the job description: {job_description}, provide a detailed justification."
    response = query_huggingface_gpt2(prompt)
    
    if 'error' not in response:
        if isinstance(response, list) and len(response) > 0:
            generated_text = response[0].get('generated_text', 'No analysis generated.')
            return truncate_to_50_words(generated_text)  # Limit to 50 words
        else:
            return "Unexpected response format from GPT-2."
    else:
        return f"Error: {response['error']}"

# Main Streamlit app
st.title("Kavin's Application Resume Analysis(GPT-2)")
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
        score = len(analysis_result.split())  # Score based on the word count of analysis
        results.append({
            'file_name': uploaded_file.name,
            'analysis': analysis_result,
            'score': score
        })
    
    # Sort candidates based on the score (longer justification gets higher score)
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

    # Display top 5 candidates
    st.write("Top 5 candidates:")
    for idx, result in enumerate(sorted_results[:5]):
        st.write(f"**Candidate {idx+1}: {result['file_name']}**")
        st.write(f"**Justification (50 words):** {result['analysis']}")
    
    # Display a bar chart for the top candidates
    top_5 = sorted_results[:5]
    candidate_names = [res['file_name'] for res in top_5]
    candidate_scores = [res['score'] for res in top_5]
    
    # Plotting the bar chart
    fig, ax = plt.subplots()
    ax.barh(candidate_names, candidate_scores, color='skyblue')
    ax.set_xlabel('Score (Based on Justification Length)')
    ax.set_title('Top 5 Candidates Based on Resume Analysis')
    ax.invert_yaxis()  # Invert y-axis to show highest score at the top
    st.pyplot(fig)
