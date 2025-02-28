# Resume Tailoring Assistant

An advanced NLP-powered tool that helps job seekers optimize their resumes for specific job descriptions using semantic matching, keyword extraction, and AI-driven text generation.

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
- Required packages listed in `simple_requirements.txt`

### Installation

```bash
# Clone the repository
git clone https://github.com/AKPekka/resume-tailor.git
cd resume-tailor

# Set up virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r simple_requirements.txt

# Download NLTK data
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger

```



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
- **NumPy**: Numerical operations


## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 🙏 Acknowledgements

- [NLTK Team](https://www.nltk.org/)
- [Sentence-BERT](https://www.sbert.net/)
- [Hugging Face](https://huggingface.co/)
- [OpenAI](https://openai.com/) for GPT-2
- All contributors
