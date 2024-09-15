import streamlit as st
import pandas as pd
from io import BytesIO
from transformers import pipeline
import fitz  # PyMuPDF
from docx import Document

# Initialize Hugging Face text generation pipeline
generator = pipeline('text-generation', model='gpt-4')

# Function to generate justification using GPT-based model
def generate_gpt_justification(resume_text, description):
    prompt = f"Given the following job description: {description}\nAnd the resume: {resume_text}\n\nProvide a detailed justification of how well the resume fits the job description."
    response = generator(prompt, max_length=300, num_return_sequences=1)
    return response[0]['generated_text']

# Function to generate justification based on keyword matching (as a fallback)
def generate_keyword_justification(resume_text, description):
    # Extract key skills or keywords from the job description
    skills_required = set(description.lower().split())  # Faster than regex
    resume_words = set(resume_text.lower().split())
    skills_matched = skills_required.intersection(resume_words)
    
    if skills_matched:
        return f"The resume includes several key skills required for the job, such as {', '.join(skills_matched)}."
    else:
        return "The resume does not match the key skills required for the job."

# Placeholder function for resume analysis
def analyze_resume(resume_text, description, use_gpt=True):
    if use_gpt:
        justification = generate_gpt_justification(resume_text, description)
    else:
        justification = generate_keyword_justification(resume_text, description)
    
    fit_score = 0.8 if use_gpt else 0.5  # Placeholder for fit score logic
    return {
        'fit_score': fit_score,
        'justification': justification
    }

# Function to rank resumes
def rank_resumes(resume_data, description, use_gpt=True):
    results = []
    for _, row in resume_data.iterrows():
        analysis = analyze_resume(row['resume_text'], description, use_gpt=use_gpt)
        results.append({
            'name': row['name'],
            'fit_score': analysis['fit_score'],
            'justification': analysis['justification']
        })
    
    return pd.DataFrame(results).sort_values(by='fit_score', ascending=False).reset_index(drop=True)

# Streamlit UI
st.title("Kavin's Find")

# Collect job description
description = st.text_area("Enter the job description:", "")

# Option to select justification method (GPT vs Keyword Matching)
use_gpt = st.checkbox("Use GPT for Justification", value=True)

# Upload resumes
uploaded_files = st.file_uploader("Upload resumes (PDF or Word)", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    # Convert uploaded files to text
    resume_data = []
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_type = uploaded_file.type
        file_content = uploaded_file.read()
        
        # Extract text from the uploaded file
        resume_text = ""
        if file_type == "application/pdf":
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            resume_text = " ".join([page.get_text() for page in pdf_document])
            pdf_document.close()
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(BytesIO(file_content))
            resume_text = "\n".join([para.text for para in doc.paragraphs])
        
        resume_data.append({'name': file_name, 'resume_text': resume_text})
    
    if description:
        # Rank resumes
        resume_df = pd.DataFrame(resume_data)
        ranked_resumes = rank_resumes(resume_df, description, use_gpt=use_gpt)
        
        # Select number of top candidates
        num_top_candidates = st.selectbox("Select number of top candidates to display:", [5, 10])
        top_candidates = ranked_resumes.head(num_top_candidates)
        
        st.write("### Top Candidates")
        st.dataframe(top_candidates[['name', 'fit_score', 'justification']])
        
        st.write("### Detailed Justifications")
        for _, row in top_candidates.iterrows():
            st.write(f"**{row['name']}**")
            st.write(f"Fit Score: {row['fit_score']:.2f}")
            st.write(f"Justification: {row['justification']}")
            st.write("---")
else:
    st.write("Please upload resumes to start analysis.")
