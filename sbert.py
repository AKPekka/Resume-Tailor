from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import re
from typing import List, Dict, Tuple, Any

class SemanticMatcher:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        """Initialize the semantic matcher with a specific SBERT model.

        Args:
            model_name: Name of the SBERT model to use. Default is a smaller, faster model.
                        For higher accuracy, consider 'all-mpnet-base-v2'
        """
        self.model = SentenceTransformer(model_name)
        # Cache for document embeddings to avoid recomputing
        self.embedding_cache = {}

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for better embedding quality."""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        return text

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get the embedding for a text, using cache if available."""
        # Create a hash of the text to use as cache key
        text_hash = hash(text)
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]

        # Compute the embedding
        embedding = self.model.encode(text, convert_to_tensor=True)
        # Store in cache
        self.embedding_cache[text_hash] = embedding
        return embedding

    def _segment_document(self, document: str) -> List[str]:
        """Split document into sections or paragraphs for more granular analysis."""
        # Simple split by double newlines (paragraphs)
        segments = [s.strip() for s in re.split(r'\n\s*\n', document) if s.strip()]

        # If the document doesn't have clear paragraph breaks, try to split by sentences
        if len(segments) <= 1:
            from nltk.tokenize import sent_tokenize
            try:
                segments = sent_tokenize(document)
            except:
                # Fallback to a simple split by periods
                segments = [s.strip() + '.' for s in document.split('.') if s.strip()]

        return segments

    def calculate_similarity(self, resume_text: str, job_text: str) -> float:
        """Calculate semantic similarity between resume and job description."""
        # Preprocess texts
        resume_text = self._preprocess_text(resume_text)
        job_text = self._preprocess_text(job_text)

        # Get embeddings
        resume_embedding = self._get_embedding(resume_text)
        job_embedding = self._get_embedding(job_text)

        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
        return similarity

    def extract_key_skills(self, text: str, top_n: int = 25) -> List[str]:
        """Extract key skills or phrases from text using SBERT and clustering."""
        # Preprocess text
        clean_text = self._preprocess_text(text)

        # Extract candidate keyphrases using n-grams
        unigrams = clean_text.split()
        bigrams = [' '.join(unigrams[i:i+2]) for i in range(len(unigrams)-1)]
        trigrams = [' '.join(unigrams[i:i+3]) for i in range(len(unigrams)-2)]

        # Combine all candidates
        candidates = unigrams + bigrams + trigrams

        # Remove duplicates and very short candidates
        candidates = list(set([c for c in candidates if len(c) > 3]))

        # Limit to a manageable number
        candidates = candidates[:500]  # Practical limit

        if not candidates:
            return []

        # Get embeddings for all candidates
        candidate_embeddings = self.model.encode(candidates, convert_to_tensor=True)

        # Use clustering to find diverse key skills
        from sklearn.cluster import KMeans

        # Determine number of clusters (smaller of top_n or candidates length)
        n_clusters = min(top_n, len(candidates))

        # Convert embeddings to numpy for KMeans
        embeddings_np = candidate_embeddings.cpu().numpy()

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings_np)

        # Get closest candidates to centroids
        closest_indices = []

        for i in range(n_clusters):
            # Find candidates in this cluster
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            if len(cluster_indices) == 0:
                continue

            # Calculate distances to centroid
            distances = np.linalg.norm(
                embeddings_np[cluster_indices] - kmeans.cluster_centers_[i].reshape(1, -1),
                axis=1
            )

            # Get the index of the closest candidate
            closest_idx = cluster_indices[np.argmin(distances)]
            closest_indices.append(closest_idx)

        # Return the key skills
        return [candidates[idx] for idx in closest_indices]

    def find_missing_skills(self, resume_text: str, job_text: str,
                           similarity_threshold: float = 0.75) -> Dict[str, Any]:
        """Identify skills from job description missing in resume using semantic matching."""
        # Extract key skills
        job_skills = self.extract_key_skills(job_text)

        # Segment the resume
        resume_segments = self._segment_document(resume_text)

        # Get embeddings
        job_skill_embeddings = self.model.encode(job_skills, convert_to_tensor=True)
        resume_segment_embeddings = self.model.encode(resume_segments, convert_to_tensor=True)

        # For each job skill, find the best matching resume segment
        missing_skills = []
        weak_skills = []
        present_skills = []

        for i, skill in enumerate(job_skills):
            # Calculate similarity with each resume segment
            skill_embedding = job_skill_embeddings[i].reshape(1, -1)
            similarities = util.pytorch_cos_sim(skill_embedding, resume_segment_embeddings)

            # Get the maximum similarity
            max_similarity = torch.max(similarities).item()
            best_segment_idx = torch.argmax(similarities).item()

            # Classify as missing, weak, or present
            if max_similarity < 0.3:
                missing_skills.append({
                    'skill': skill,
                    'confidence': 1.0 - max_similarity
                })
            elif max_similarity < similarity_threshold:
                weak_skills.append({
                    'skill': skill,
                    'confidence': 1.0 - max_similarity,
                    'context': resume_segments[best_segment_idx]
                })
            else:
                present_skills.append({
                    'skill': skill,
                    'confidence': max_similarity,
                    'context': resume_segments[best_segment_idx]
                })

        # Sort by confidence
        missing_skills.sort(key=lambda x: x['confidence'], reverse=True)
        weak_skills.sort(key=lambda x: x['confidence'], reverse=True)
        present_skills.sort(key=lambda x: x['confidence'], reverse=True)

        # Calculate overall similarity
        overall_similarity = self.calculate_similarity(resume_text, job_text)

        return {
            'missing_skills': missing_skills,
            'weak_skills': weak_skills,
            'present_skills': present_skills,
            'similarity_score': overall_similarity
        }
