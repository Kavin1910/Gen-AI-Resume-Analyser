import streamlit as st
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import io
from transformers import pipeline

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Handle cases where extract_text() might return None
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
    return text.lower().replace('\n', ' ')

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

# Streamlit application
def main():
    st.title("Kavin's AI Resume Analyzer and Ranking")
    
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
                
                # Display results
                st.write(f"Top {num_candidates} Candidates:")
                for i in range(min(num_candidates, len(ranked_indices))):
                    st.write(f"**Candidate {i+1}**:")
                    st.write(f"Score: {scores[ranked_indices[i]]:.2f}%")
                    
                    # Display resume snippet
                    st.text_area(f"Resume Snippet {i+1}", resumes[ranked_indices[i]][:1000], height=100)
                
                # Generate and display charts
                fig, ax = plt.subplots(figsize=(10, 5))
                top_scores = [scores[idx] for idx in ranked_indices[:num_candidates]]
                ax.bar(range(len(top_scores)), top_scores, color='blue')
                ax.set_xlabel('Candidates')
                ax.set_ylabel('Scores (%)')
                ax.set_title('Resume Scores')
                plt.xticks(range(len(top_scores)), [f"Candidate {i+1}" for i in range(num_candidates)], rotation=45)
                st.pyplot(fig)
                
                # Generate assessment justification
                st.write("Assessment Justification:")
                try:
                    nlp = pipeline("text-generation", model="gpt2")
                    justification_prompt = (
                        f"Generate a 2-line summary of why the following resumes are a good match for the job description:\n\n"
                        f"Job Description: {job_description}\n\n"
                        f"Resumes: {', '.join([resumes[idx][:500] for idx in ranked_indices[:num_candidates]])}"
                    )
                    justification = nlp(justification_prompt, max_length=100, truncation=True)
                    summary = justification[0]['generated_text']
                    # Limit to 2 lines and format as bullet points
                    lines = summary.split('\n')[:2]
                    st.markdown("• " + "\n• ".join(lines))
                except Exception as e:
                    st.error(f"Error generating justification: {e}")
            
            else:
                st.write("No resumes uploaded.")
        else:
            st.write("Please upload resumes.")
        
        # Add "Thank You" image        
        st.image("data:image/jpeg;base64,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PMhdjOdMBW9hRnSpYmt+SqvqkalXMeGDNqqOOp8sYONHDitb3ZtC8XtrDXBo4U6sqfAwXWh3RMCPVVDL7NzzmNVfp/klK8XIuMtEIAM7jIBru4wFNtfs/RpPVUer04L5t/HHocql1KXDcZoEckZYzddGRjA1Cv3TTWGiNnVvNDXbOvDy+ke0tP6jtCPaqK42RLU3Re7kDD+QrhvY5EY698Y71E/C7nOMfEHtuyRqwG3GYNg4vihJy/0u6uxTrPZbg1Kq/wCxjBo7lZa2iqHxsgfJHnmOa3Ix0L0KaKOrb1IS3LcSbY61z0Mc89SCx84biM8WgZ4+9ayeSfZUJU05S8ySLQnFHZ3TjjjRayzpeOJlFqFsgYOVcHEdIXC2jXjTSqyTfoZk03uPeOjKkGuSobhZBUoDXXmCsnhY2ik3CHc7BwVXX9K4qRj2LwDMpWyNgY2Z29IG4cetTaSlGCUuILuF0AwgMO6B4pvzee3Cptuqt90fZc9+OR3t8a95i2flgXb29uHhlV/2f7fVPVnTu4na607sG1xniV6lMhDdWcmMFQMLBkEZQANQHh72xtc5zgGtGST0BAk28FKaeOohZLC9r43jLXNOQUNpRlCTjJYZdQ1CAx6ylbV074XSPjDsc6M4I9qytxrOOpYLjRuNa3eJ3RjXpWDKWFgqCOlN5k9Y6kBVAUwgLFfTmqop6cPLDLG5gcOLcjGVtCWmSlyOdWHaU3DmjV7M2V9lgliknEvKP3tBo1drm4VeSeMEeytHbRcc5N3hRsEwEJgGLX1sNDCJZ87pcBzRnUlazmoLLOlKnKq9MTIbzgCNAdVuc9+d56AwgKoAgKEacVgFieB0ksTmyvYGOyQ06O7CudSEpSTTxj4m8JaU1jOS/hdTQqgKYTAGEBVAEBQjKxjIKEY4YRLAMOSrqWXWGlbRPdTPjc51UHDDCODSOOq2SWMmUlgzQsGCqAIAgPEkbZGlrxkEYIPSEC3PKPNNTxUsLIYGNjiYMNa0YACGZSlJ5k8suoYCAIDW36jnr7TVUtJNyE0rCGydRW1OSjNNm0WlLLMHY+1VtntYpbjVeMS75Ockho6Bqtq04zl7JtVlGUvZJAuZzCAIAgKYQFUAQHl0bX+UAfSECbXAHmgkDJ6kQMa21M9VTCSqpXUshJHJucHHAOAcjr4rLOlaEIT0wllczLWDmEAQBAEAQBAEAQBAEAOqApjXKAqgCAIAgCAIAgCAIChGUAAwgKoAgCAIAgCAIAgBGUBQDCAqgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAIAgCAID//2Q==")
    
    else:
        st.write("Please upload a job description.")

if __name__ == "__main__":
    main()
