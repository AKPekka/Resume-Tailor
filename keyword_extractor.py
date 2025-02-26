import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

class KeywordExtractor:
    def __init__(self):
        # Ensure NLTK resources are downloaded
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')

        self.stop_words = set(stopwords.words('english'))

        # Common technical skills and keywords
        self.tech_skills = set([
            "python", "java", "javascript", "html", "css", "react", "angular", "vue",
            "node.js", "express", "django", "flask", "spring", "hibernate", "sql",
            "mysql", "postgresql", "mongodb", "nosql", "aws", "azure", "gcp",
            "docker", "kubernetes", "jenkins", "git", "github", "ci/cd", "agile",
            "scrum", "devops", "machine learning", "deep learning", "ai", "nlp",
            "data science", "data analysis", "data visualization", "tableau", "power bi",
            "excel", "word", "powerpoint", "project management", "leadership",
            "communication", "teamwork", "problem solving", "critical thinking"
        ])

    def _preprocess_text(self, text):
        """Preprocess text for keyword extraction"""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)

        # Tokenize
        words = word_tokenize(text)

        # Remove stopwords
        words = [word for word in words if word not in self.stop_words and len(word) > 1]

        return words

    def _extract_ngrams(self, words, n=2):
        """Extract n-grams from a list of words"""
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        return ngrams

    def extract_keywords(self, text, top_n=20):
        """Extract keywords from text using frequency and predefined skills"""
        # Preprocess text
        words = self._preprocess_text(text)

        # Extract single words and bigrams
        single_words = words
        bigrams = self._extract_ngrams(words, 2)

        # Combine all potential keywords
        all_keywords = single_words + bigrams

        # Count occurrences
        keyword_counts = Counter(all_keywords)

        # Filter by potential technical skills or job-related terms
        filtered_keywords = []
        for keyword, count in keyword_counts.items():
            # Check if the keyword is in our predefined tech skills
            if keyword in self.tech_skills:
                filtered_keywords.append((keyword, count * 2))  # Give extra weight to tech skills
                continue

            # Check if the keyword contains any tech skills
            for skill in self.tech_skills:
                if skill in keyword:
                    filtered_keywords.append((keyword, count))
                    break

        # If we don't have enough filtered keywords, add more from the original counts
        if len(filtered_keywords) < top_n:
            additional_keywords = [(k, c) for k, c in keyword_counts.most_common(top_n * 2)
                                  if not any(k == existing[0] for existing in filtered_keywords)]
            filtered_keywords.extend(additional_keywords[:top_n - len(filtered_keywords)])

        # Sort by count and return top N
        sorted_keywords = sorted(filtered_keywords, key=lambda x: x[1], reverse=True)
        return [keyword for keyword, _ in sorted_keywords[:top_n]]

    def calculate_similarity(self, resume_text, job_text):
        """Calculate similarity between resume and job description using TF-IDF"""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])

        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
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
