import streamlit as st
import tempfile
import os
import torch
from document_processor import DocumentProcessor
from keyword_extractor import KeywordExtractor
from resume_tailor import ResumeTailor
from sbert import SemanticMatcher
from gpt2 import ResumeBulletGenerator
import docx
from docx.shared import RGBColor

def main():
    st.set_page_config(
        page_title="AI-Enhanced Resume Tailoring Tool",
        page_icon="📝",
        layout="wide"
    )

    st.title("AKPs AI-Enhanced Resume Tailoring Tool")
    st.write("""
    This tool uses advanced AI models (BERT and GPT-2) to help you tailor your resume
    to match job descriptions. Upload your resume and a job description to get started.
    """)

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.info(f"Running on: {device.upper()}")

    # Model selection
    st.sidebar.header("Model Configuration")
    sbert_model = st.sidebar.selectbox(
        "Select SBERT Model",
        ["paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2"],
        help="Smaller models are faster, larger models are more accurate"
    )

    gpt2_model = st.sidebar.selectbox(
        "Select GPT-2 Model",
        ["gpt2", "gpt2-medium"],
        help="Smaller models require less resources, larger models generate better suggestions"
    )

    matching_threshold = st.sidebar.slider(
        "Matching Threshold",
        min_value=0.5,
        max_value=0.9,
        value=0.7,
        help="Lower values find more potential matches, higher values are more strict"
    )

    # Initialize processors
    doc_processor = DocumentProcessor()
    keyword_extractor = KeywordExtractor()
    resume_tailor = ResumeTailor()

    # Initialize AI models with loading state indicators
    sbert_load_state = st.sidebar.text("SBERT: Not loaded")
    gpt2_load_state = st.sidebar.text("GPT-2: Not loaded")

    # Create session state for storing data
    if 'tmp_resume_path' not in st.session_state:
        st.session_state.tmp_resume_path = None
    if 'tmp_job_path' not in st.session_state:
        st.session_state.tmp_job_path = None
    if 'resume_type' not in st.session_state:
        st.session_state.resume_type = None
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = None
    if 'job_text' not in st.session_state:
        st.session_state.job_text = None
    if 'keyword_results' not in st.session_state:
        st.session_state.keyword_results = None
    if 'semantic_results' not in st.session_state:
        st.session_state.semantic_results = None
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = None
    if 'gpt2_suggestions' not in st.session_state:
        st.session_state.gpt2_suggestions = None
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False

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
        # First, load the AI models if not already loaded
        if not st.session_state.models_loaded:
            with st.spinner("Loading AI models..."):
                try:
                    # Initialize models
                    sbert_load_state.text("SBERT: Loading...")
                    st.session_state.semantic_matcher = SemanticMatcher(model_name=sbert_model)
                    sbert_load_state.text("SBERT: ✅ Loaded")

                    gpt2_load_state.text("GPT-2: Loading...")
                    st.session_state.bullet_generator = ResumeBulletGenerator(model_name=gpt2_model)
                    gpt2_load_state.text("GPT-2: ✅ Loaded")

                    st.session_state.models_loaded = True
                except Exception as e:
                    st.error(f"Error loading AI models: {str(e)}")
                    st.stop()

        with st.spinner("Analyzing documents with AI..."):
            # Process resume
            resume_type = 'pdf' if resume_file.name.endswith('.pdf') else 'docx'
            st.session_state.resume_type = resume_type

            # Create temporary files
            tmp_resume = tempfile.NamedTemporaryFile(delete=False, suffix=f".{resume_type}")
            tmp_resume.write(resume_file.getvalue())
            tmp_resume.close()
            st.session_state.tmp_resume_path = tmp_resume.name

            # Process job description
            job_type = 'pdf' if job_desc_file.name.endswith('.pdf') else 'docx'
            tmp_job = tempfile.NamedTemporaryFile(delete=False, suffix=f".{job_type}")
            tmp_job.write(job_desc_file.getvalue())
            tmp_job.close()
            st.session_state.tmp_job_path = tmp_job.name

            # Extract text
            resume_text = doc_processor.extract_text(st.session_state.tmp_resume_path, resume_type)
            st.session_state.resume_text = resume_text

            job_text = doc_processor.extract_text(st.session_state.tmp_job_path, job_type)
            st.session_state.job_text = job_text

            # 1. Simple keyword extraction (original functionality)
            st.session_state.keyword_results = keyword_extractor.get_keyword_suggestions(
                resume_text, job_text, similarity_threshold=matching_threshold
            )

            # 2. Semantic matching using SBERT
            st.session_state.semantic_results = st.session_state.semantic_matcher.find_missing_skills(
                resume_text, job_text, similarity_threshold=matching_threshold
            )

            # 3. Generate tailored suggestions using ResumeTailor
            st.session_state.suggestions = resume_tailor.generate_tailoring_suggestions(
                st.session_state.keyword_results, resume_text
            )

            # 4. Generate resume bullet points using GPT-2
            st.session_state.gpt2_suggestions = st.session_state.bullet_generator.generate_suggestions(
                st.session_state.semantic_results, job_text, resume_text
            )

        # Display results
        st.header("2. Analysis Results")

        # Create tabs for different analysis methods
        tab1, tab2 = st.tabs(["Keyword Analysis", "Semantic Analysis (BERT)"])

        with tab1:
            # Display similarity score
            keyword_similarity = st.session_state.keyword_results['similarity_score']
            st.subheader("Keyword-Based Similarity")

            # Create a color for the similarity score
            if keyword_similarity < 0.5:
                color = "red"
            elif keyword_similarity < 0.7:
                color = "orange"
            else:
                color = "green"

            st.markdown(f"<h3 style='color: {color};'>{keyword_similarity:.2f}/1.00</h3>", unsafe_allow_html=True)

            # Display keywords
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Keywords in Your Resume")
                for keyword in st.session_state.keyword_results['resume_keywords']:
                    st.write(f"- {keyword}")

            with col2:
                st.subheader("Keywords in Job Description")
                for keyword in st.session_state.keyword_results['job_keywords']:
                    if keyword in st.session_state.keyword_results['missing_keywords']:
                        st.markdown(f"- <span style='color: red;'>{keyword} (missing)</span>", unsafe_allow_html=True)
                    else:
                        st.write(f"- {keyword}")

        with tab2:
            # Display BERT semantic matching results
            semantic_similarity = st.session_state.semantic_results['similarity_score']
            st.subheader("BERT Semantic Similarity")

            # Create a color for the semantic similarity score
            if semantic_similarity < 0.5:
                color = "red"
            elif semantic_similarity < 0.7:
                color = "orange"
            else:
                color = "green"

            st.markdown(f"<h3 style='color: {color};'>{semantic_similarity:.2f}/1.00</h3>", unsafe_allow_html=True)

            # Display semantically matched skills
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Missing Skills (Semantic)")
                if st.session_state.semantic_results['missing_skills']:
                    for skill in st.session_state.semantic_results['missing_skills']:
                        confidence = skill['confidence'] * 100
                        st.markdown(f"- **{skill['skill']}** (Confidence: {confidence:.1f}%)")
                else:
                    st.write("No missing skills detected!")

            with col2:
                st.subheader("Weak Skills (Need Improvement)")
                if st.session_state.semantic_results['weak_skills']:
                    for skill in st.session_state.semantic_results['weak_skills']:
                        confidence = skill['confidence'] * 100
                        st.markdown(f"- **{skill['skill']}** (Confidence: {confidence:.1f}%)")
                        with st.expander("Current Context"):
                            st.write(skill['context'])
                else:
                    st.write("No weak skills detected!")

        # Display tailoring suggestions
        st.header("3. Tailoring Suggestions")

        # Create tabs for different suggestion methods
        tab1, tab2 = st.tabs(["Basic Suggestions", "AI-Generated Bullet Points (GPT-2)"])

        with tab1:
            for suggestion in st.session_state.suggestions:
                if suggestion['keyword']:
                    st.write(f"🔹 **{suggestion['suggestion']}** (Missing keyword: '{suggestion['keyword']}')")
                else:
                    st.write(f"🔸 **{suggestion['suggestion']}**")

        with tab2:
            # Display GPT-2 generated bullet points
            st.subheader("Missing Skills - Add These to Your Resume")
            for skill_suggestion in st.session_state.gpt2_suggestions['missing_skills']:
                skill = skill_suggestion['skill']
                st.markdown(f"#### {skill.title()}")

                for i, bullet in enumerate(skill_suggestion['bullet_points']):
                    st.markdown(f"• {bullet}")

            st.subheader("Weak Skills - Replace with These Improved Versions")
            for skill_suggestion in st.session_state.gpt2_suggestions['weak_skills']:
                skill = skill_suggestion['skill']
                st.markdown(f"#### {skill.title()}")

                with st.expander("Current Content"):
                    st.write(skill_suggestion['current_context'])

                st.markdown("**Improved Bullet Points:**")
                for bullet in skill_suggestion['improved_bullet_points']:
                    st.markdown(f"• {bullet}")

        # Create tailored resume
        st.header("4. Create Tailored Resume")

        if st.button("Generate AI-Enhanced Resume"):
            with st.spinner("Creating tailored resume with AI suggestions..."):
                # Only works with DOCX files
                if st.session_state.resume_type == 'docx':
                    try:
                        # Load the original resume
                        doc = docx.Document(st.session_state.tmp_resume_path)

                        # Add a section for AI suggestions
                        doc.add_paragraph().add_run("AI-GENERATED SUGGESTIONS:").bold = True
                        doc.add_paragraph("The following are AI-generated suggestions to improve your resume:")

                        # Add missing skills with GPT-2 bullet points
                        p = doc.add_paragraph()
                        run = p.add_run("MISSING SKILLS - Consider adding these bullet points:")
                        run.bold = True
                        run.font.color.rgb = RGBColor(255, 0, 0)  # Red

                        for skill_suggestion in st.session_state.gpt2_suggestions['missing_skills']:
                            skill = skill_suggestion['skill']
                            doc.add_paragraph(f"{skill.title()}:").bold = True

                            for bullet in skill_suggestion['bullet_points']:
                                doc.add_paragraph(f"• {bullet}", style='ListBullet')

                        # Add weak skills with improved bullet points
                        p = doc.add_paragraph()
                        run = p.add_run("WEAK SKILLS - Consider replacing with these improved versions:")
                        run.bold = True
                        run.font.color.rgb = RGBColor(255, 128, 0)  # Orange

                        for skill_suggestion in st.session_state.gpt2_suggestions['weak_skills']:
                            skill = skill_suggestion['skill']
                            doc.add_paragraph(f"{skill.title()}:").bold = True

                            for bullet in skill_suggestion['improved_bullet_points']:
                                doc.add_paragraph(f"• {bullet}", style='ListBullet')

                        # Add original suggestions
                        p = doc.add_paragraph()
                        run = p.add_run("ADDITIONAL SUGGESTIONS:")
                        run.bold = True

                        for suggestion in st.session_state.suggestions:
                            if suggestion['keyword']:
                                p = doc.add_paragraph(f"• {suggestion['suggestion']} (Missing keyword: '{suggestion['keyword']}')")
                            else:
                                p = doc.add_paragraph(f"• {suggestion['suggestion']}")

                        # Save the modified document
                        tmp_modified = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
                        tmp_modified_path = tmp_modified.name
                        tmp_modified.close()

                        doc.save(tmp_modified_path)

                        # Provide download link
                        with open(tmp_modified_path, "rb") as file:
                            st.download_button(
                                label="Download AI-Enhanced Resume",
                                data=file,
                                file_name="ai_enhanced_resume.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )

                        # Clean up modified file (keep original files for potential reuse)
                        os.unlink(tmp_modified_path)
                    except Exception as e:
                        st.error(f"Error creating tailored resume: {str(e)}")
                else:
                    st.error("Tailoring feature currently only works with DOCX files. Please upload a DOCX resume.")

    # Clean up temporary files when the session ends
    if st.button("Clear All Data"):
        if st.session_state.tmp_resume_path and os.path.exists(st.session_state.tmp_resume_path):
            try:
                os.unlink(st.session_state.tmp_resume_path)
            except:
                pass

        if st.session_state.tmp_job_path and os.path.exists(st.session_state.tmp_job_path):
            try:
                os.unlink(st.session_state.tmp_job_path)
            except:
                pass

        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.experimental_rerun()

if __name__ == "__main__":
    main()
