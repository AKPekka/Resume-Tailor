import pdfplumber
import docx
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

class DocumentProcessor:
    def __init__(self):
        # Download necessary NLTK resources if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')

        self.stop_words = set(stopwords.words('english'))

    def extract_text_from_pdf(self, pdf_file_path):
        """Extract text from PDF file using pdfplumber"""
        text = ""
        try:
            with pdfplumber.open(pdf_file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text: # Ensure text was extracted
                        text += page_text + "\n" # Add newline between pages
        except Exception as e:
            # Handle potential errors like encrypted PDFs or corrupted files
            print(f"Error reading PDF {pdf_file_path}: {e}")
            # Optionally, re-raise or return a specific error message / empty string
            # For now, return empty string if an error occurs
            return ""
        return text

    def extract_text_from_docx(self, docx_file):
        """Extract text from DOCX file"""
        doc = docx.Document(docx_file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)

    def extract_text(self, file, file_type):
        """Extract text based on file type"""
        if file_type == 'pdf':
            return self.extract_text_from_pdf(file)
        elif file_type == 'docx':
            return self.extract_text_from_docx(file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def preprocess_text(self, text):
        """Preprocess the extracted text"""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_text(self, text):
        """Tokenize text into words and sentences"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        # Remove stopwords
        filtered_words = [word for word in words if word not in self.stop_words]

        return {
            'sentences': sentences,
            'words': filtered_words
        }
