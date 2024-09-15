import streamlit as st
import pandas as pd
from io import BytesIO
from transformers import pipeline

# Initialize Hugging Face text generation pipeline
generator = pipeline('text-generation', model='gpt-4')

# Function to generate justification
def generate_justification(resume_text, description):
    prompt = f"Given the following job description: {description}\nAnd the resume: {resume_text}\n\nProvide a detailed justification of how well the resume fits the job description."
    response = generator(prompt, max_length=300, num_return_sequences=1)
    return response[0]['generated_text']

# Placeholder function for resume analysis
def analyze_resume(resume_text, description):
    justification = generate_justification(resume_text, description)
    # Dummy fit score, replace with actual model logic
    fit_score = 0.8
    return {
        'fit_score': fit_score,
        'justification': justification
    }

# Function to rank resumes
def rank_resumes(resume_data, description):
    results = []
    for index, row in resume_data.iterrows():
        analysis = analyze_resume(row['resume_text'], description)
        results.append({
            'name': row['name'],
            'fit_score': analysis['fit_score'],
            'justification': analysis['justification']
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='fit_score', ascending=False).reset_index(drop=True)
    return results_df

# Streamlit UI
st.title("Kavin's Find")

# Collect job description
description = st.text_area("Enter the job description:", "")

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
        if file_type == "application/pdf":
            import fitz  # PyMuPDF
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            resume_text = ""
            for page in pdf_document:
                resume_text += page.get_text()
            pdf_document.close()
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            from docx import Document
            doc = Document(BytesIO(file_content))
            resume_text = "\n".join([para.text for para in doc.paragraphs])
        else:
            resume_text = "Unsupported file type"
        
        resume_data.append({'name': file_name, 'resume_text': resume_text})
    
    if description:
        # Rank resumes
        resume_df = pd.DataFrame(resume_data)
        ranked_resumes = rank_resumes(resume_df, description)
        
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
