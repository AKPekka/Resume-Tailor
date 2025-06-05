import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keybert import KeyBERT

# Initialize KeyBERT model outside the class or pass model name for flexibility
# Using a generic sentence transformer model that's good for keywords
# Ensure 'sentence-transformers' is installed
try:
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')
except Exception as e:
    # This can happen if model files are not found or internet is off during first download
    print(f"Error initializing KeyBERT: {e}. Keyword extraction might be impaired.")
    kw_model = None

class KeywordExtractor:
    def __init__(self):
        # Ensure NLTK resources are downloaded (mainly for stopwords if still needed by TF-IDF or other parts)
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        # self.tech_skills = set([...]) # Removed hardcoded skills

    # _preprocess_text and _extract_ngrams are no longer needed for KeyBERT
    # but TF-IDF might need some preprocessing if we were to change it.
    # For now, TF-IDF uses raw text.

    def extract_keywords(self, text, top_n=20, keyphrase_ngram_range=(1, 2)):
        """Extract keywords from text using KeyBERT"""
        if kw_model is None:
            print("KeyBERT model failed to initialize. Keyword extraction is not available.")
            return []
        
        # KeyBERT can use custom stop words if needed, but its underlying model often handles this well.
        # For explicit control, one can pass 'stop_words=self.stop_words' or 'stop_words='english''
        # to kw_model.extract_keywords
        try:
            keywords_with_scores = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=keyphrase_ngram_range,
                stop_words='english', # Using KeyBERT's internal or a common stop word list
                top_n=top_n,
                use_mmr=True, # Use Maximal Marginal Relevance for diverse keywords
                diversity=0.5 # Adjust diversity for MMR
            )
            # Extract just the keywords
            keywords = [kw for kw, score in keywords_with_scores]
            return keywords
        except Exception as e:
            print(f"Error during KeyBERT keyword extraction: {e}")
            # Fallback or return empty: For now, returning empty to signal failure.
            return []

    def calculate_similarity(self, resume_text, job_text):
        """Calculate similarity between resume and job description using TF-IDF.
        This uses the full text, not just keywords.
        NLTK's word_tokenize and stopwords can be used if we want to preprocess for TF-IDF.
        """
        # Simple tokenizer and stopword removal for TF-IDF
        def preprocess_for_tfidf(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
            tokens = word_tokenize(text)
            return " ".join([word for word in tokens if word not in self.stop_words])

        processed_resume = preprocess_for_tfidf(resume_text)
        processed_job = preprocess_for_tfidf(job_text)

        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([processed_resume, processed_job])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except ValueError: # Handles empty vocabulary
            similarity = 0.0
        return similarity

    def find_missing_keywords(self, resume_keywords, job_keywords):
        """Find keywords in job description that are missing from resume"""
        return [keyword for keyword in job_keywords if keyword not in resume_keywords]

    def get_keyword_suggestions(self, resume_text, job_text, similarity_threshold=0.7):
        """Generate keyword suggestions based on comparison"""
        # Extract keywords
        resume_keywords = self.extract_keywords(resume_text)
        job_keywords = self.extract_keywords(job_text)

        # Calculate similarity
        similarity = self.calculate_similarity(resume_text, job_text)

        # Find missing keywords
        missing_keywords = self.find_missing_keywords(resume_keywords, job_keywords)

        return {
            'resume_keywords': resume_keywords,
            'job_keywords': job_keywords,
            'similarity_score': similarity,
            'missing_keywords': missing_keywords,
            'needs_improvement': similarity < similarity_threshold
        }
