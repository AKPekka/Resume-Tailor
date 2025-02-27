# Resume Tailoring Assistant

An advanced NLP-powered tool that helps job seekers optimize their resumes for specific job descriptions using semantic matching, keyword extraction, and AI-driven content generation.

## 🌟 Features

### 1. Job Description Analysis
- **Keyword Extraction**: Automatically identifies key skills, qualifications, and requirements from job descriptions using NLTK and custom NER models
- **Semantic Understanding**: Processes job descriptions to understand context and importance of different requirements
- **Priority Ranking**: Assigns weights to different job requirements based on frequency and semantic importance

### 2. Resume Analysis
- **Content Parsing**: Supports multiple resume formats (PDF and DOCX) with intelligent section detection
- **Skills Identification**: Extracts candidate skills using a comprehensive skills taxonomy
- **Experience Mapping**: Maps candidate experience to relevant job requirements using semantic similarity
- **Gap Analysis**: Identifies discrepancies between resume content and job requirements

### 3. AI-Driven Optimization
- **Content Generation**: Suggests improvements for weak sections using fine-tuned GPT-2 models
- **Phrasing Enhancements**: Recommends stronger action verbs and industry-specific terminology
- **Tailored Bullet Points**: Generates achievement-focused bullet points aligned with job requirements
- **Language Refinement**: Improves clarity, conciseness, and impact of existing content

### 4. Semantic Matching
- **Sentence-BERT Integration**: Calculates semantic similarity between resume sections and job requirements
- **Contextual Understanding**: Recognizes when different terminology describes similar skills or experiences
- **Transfer Skills Detection**: Identifies transferable skills relevant to target positions


## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

```bash
# Clone the repository
git clone https://github.com/AKPekka/resume-tailor.git
cd resume-tailor

# Set up virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger

```

### Configuration

Edit `config.yaml` to customize:
- Model parameters
- Industry-specific settings
- Output preferences
- API keys (if using cloud services)

## 💻 Usage

### Command Line Interface

```bash
# Basic usage
python tailor.py --resume path/to/resume.pdf --job path/to/job_description.txt

# With advanced options
python tailor.py --resume path/to/resume.pdf --job path/to/job_description.txt --industry tech --output-format docx --enhancement-level comprehensive
```

### Python API

```python
from resume_tailor import ResumeTailoringAssistant

# Initialize the assistant
assistant = ResumeTailoringAssistant(
    industry="technology",
    enhancement_level="balanced"
)

# Process resume and job description
results = assistant.process(
    resume_path="path/to/resume.pdf",
    job_description_path="path/to/job_description.txt"
)

# Get optimization suggestions
suggestions = results.get_suggestions()

# Generate optimized resume
optimized_resume = results.generate_optimized_resume()
optimized_resume.save("optimized_resume.docx")
```

### Web Interface

Run the web server:
```bash
python app.py
```
Then access the web interface at `http://localhost:5000`

## 🔧 Core Modules

### `document_parser.py`
Handles parsing of different document formats:
- `ResumeParser`: Extracts structured data from resumes
- `JobDescriptionParser`: Processes job descriptions into analyzable components

### `nlp_engine.py`
Core NLP functionality:
- `KeywordExtractor`: Identifies important keywords and phrases
- `SemanticAnalyzer`: Performs semantic similarity calculations
- `EntityRecognizer`: Custom NER for resume and job description entities

### `optimization_engine.py`
AI-driven content optimization:
- `ContentGenerator`: GPT-2 based text generation
- `PhraseEnhancer`: Improves existing content
- `BulletPointGenerator`: Creates achievement-focused bullet points

### `semantic_matcher.py`
Matching algorithms:
- `SkillMatcher`: Matches skills between resume and job description
- `ExperienceMatcher`: Aligns experience with job requirements
- `TransferableSkillDetector`: Identifies applicable transferable skills

### `visualization.py`
Reporting and visualization:
- `MatchScoreVisualizer`: Creates visual representations of match scores
- `ReportGenerator`: Produces detailed analysis reports

## 🛠️ Technology Stack

### NLP & Machine Learning
- **NLTK**: Natural language processing for tokenization, POS tagging, and basic NER
- **Sentence-BERT**: State-of-the-art sentence embeddings for semantic matching
- **Hugging Face Transformers**: Pre-trained transformer models for advanced NLP tasks
- **GPT-2**: Fine-tuned models for context-aware content generation
- **spaCy**: Industrial-strength NLP for entity recognition and dependency parsing

### Data Processing
- **PyPDF2**: PDF processing
- **python-docx**: Word document handling
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations


## 🤝 Contributing

Contributions are welcome! Please check out our [contributing guidelines](CONTRIBUTING.md).

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [NLTK Team](https://www.nltk.org/)
- [Sentence-BERT](https://www.sbert.net/)
- [Hugging Face](https://huggingface.co/)
- [OpenAI](https://openai.com/) for GPT-2
- All contributors and beta testers
