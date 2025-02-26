import streamlit as st
import tempfile
import os
from document_processor import DocumentProcessor
from keyword_extractor import KeywordExtractor
from resume_tailor import ResumeTailor
import docx
from docx.shared import RGBColor

def main():
    st.set_page_config(
        page_title="Resume Tailoring Tool",
        page_icon="📝",
        layout="wide"
    )

    st.title("Resume Tailoring Tool")
    st.write("""
    This tool helps you tailor your resume to match job descriptions.
    Upload your resume and a job description to get started.
    """)

    # Initialize processors
    doc_processor = DocumentProcessor()
    keyword_extractor = KeywordExtractor()
    resume_tailor = ResumeTailor()

    # File upload section
    st.header("1. Upload Documents")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Your Resume")
        resume_file = st.file_uploader("Choose your resume file", type=["pdf", "docx"])

    with col2:
        st.subheader("Upload Job Description")
        job_desc_file = st.file_uploader("Choose the job description file", type=["pdf", "docx"])

    # Process documents if uploaded
    if resume_file and job_desc_file:
        with st.spinner("Analyzing documents..."):
            # Process resume
            resume_type = 'pdf' if resume_file.name.endswith('.pdf') else 'docx'
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{resume_type}") as tmp_resume:
                tmp_resume.write(resume_file.getvalue())
                tmp_resume_path = tmp_resume.name

            resume_text = doc_processor.extract_text(tmp_resume_path, resume_type)
            preprocessed_resume = doc_processor.preprocess_text(resume_text)

            # Process job description
            job_type = 'pdf' if job_desc_file.name.endswith('.pdf') else 'docx'
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{job_type}") as tmp_job:
                tmp_job.write(job_desc_file.getvalue())
                tmp_job_path = tmp_job.name

            job_text = doc_processor.extract_text(tmp_job_path, job_type)
            preprocessed_job = doc_processor.preprocess_text(job_text)

            # Extract keywords and calculate similarity
            keyword_results = keyword_extractor.get_keyword_suggestions(
                resume_text, job_text
            )

            # Generate tailoring suggestions
            suggestions = resume_tailor.generate_tailoring_suggestions(
                keyword_results, resume_text
            )

            # Clean up temporary files
            os.unlink(tmp_resume_path)
            os.unlink(tmp_job_path)

        # Display results
        st.header("2. Analysis Results")

        # Display similarity score
        similarity = keyword_results['similarity_score']
        st.subheader("Resume-Job Description Similarity")

        # Create a color for the similarity score
        if similarity < 0.5:
            color = "red"
        elif similarity < 0.7:
            color = "orange"
        else:
            color = "green"

        st.markdown(f"<h3 style='color: {color};'>{similarity:.2f}/1.00</h3>", unsafe_allow_html=True)

        # Display keywords
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Keywords in Your Resume")
            for keyword in keyword_results['resume_keywords']:
                st.write(f"- {keyword}")

        with col2:
            st.subheader("Keywords in Job Description")
            for keyword in keyword_results['job_keywords']:
                if keyword in keyword_results['missing_keywords']:
                    st.markdown(f"- <span style='color: red;'>{keyword} (missing)</span>", unsafe_allow_html=True)
                else:
                    st.write(f"- {keyword}")

        # Display tailoring suggestions
        st.header("3. Tailoring Suggestions")

        for suggestion in suggestions:
            if suggestion['keyword']:
                st.write(f"🔹 **{suggestion['suggestion']}** (Missing keyword: '{suggestion['keyword']}')")
            else:
                st.write(f"🔸 **{suggestion['suggestion']}**")

        # Create tailored resume
        st.header("4. Create Tailored Resume")

        if st.button("Generate Tailored Resume"):
            with st.spinner("Creating tailored resume..."):
                # Only works with DOCX files
                if resume_type == 'docx':
                    # Load the original resume
                    doc = docx.Document(tmp_resume_path)

                    # Add a section for suggested modifications
                    doc.add_paragraph().add_run("SUGGESTED MODIFICATIONS:").bold = True

                    # Add each suggestion
                    for suggestion in suggestions:
                        if suggestion['keyword']:
                            p = doc.add_paragraph("• ")
                            run = p.add_run(suggestion['suggestion'])
                            run.font.color.rgb = RGBColor(255, 0, 0)  # Red text for emphasis

                    # Save the modified document
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_modified:
                        tmp_modified_path = tmp_modified.name

                    doc.save(tmp_modified_path)

                    # Provide download link
                    with open(tmp_modified_path, "rb") as file:
                        st.download_button(
                            label="Download Tailored Resume",
                            data=file,
                            file_name="tailored_resume.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )

                    # Clean up
                    os.unlink(tmp_modified_path)
                else:
                    st.error("Tailoring feature currently only works with DOCX files. Please upload a DOCX resume.")

if __name__ == "__main__":
    main() 
