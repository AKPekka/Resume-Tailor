from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import re
from typing import List, Dict, Tuple, Any
import nltk
from nltk.tokenize import sent_tokenize

# Download required nltk data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

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

        # Common technical terms and skills that should be preserved during preprocessing
        self.technical_terms = set([
            "c++", "c#", ".net", "node.js", "react.js", "vue.js", "angular.js",
            "aws", "azure", "gcp", "api", "rest", "graphql", "sql", "nosql",
            "mongodb", "postgresql", "mysql", "redis", "docker", "kubernetes", "k8s",
            "ci/cd", "jenkins", "github", "gitlab", "bitbucket", "agile", "scrum",
            "kanban", "jira", "confluence", "aws/azure", "java/python", "html/css",
            "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn", "keras",
            "nlp", "ml", "ai", "ux/ui", "a/b", "seo/sem", "ios/android"
        ])

        # Common resume skills - helps with better skill extraction
        self.common_skills = set([
            "python", "java", "javascript", "typescript", "ruby", "php", "golang", "scala",
            "swift", "kotlin", "rust", "html", "css", "sql", "nosql", "react", "angular",
            "vue", "node", "django", "flask", "spring", "express", "tensorflow", "pytorch",
            "machine learning", "data science", "data analysis", "statistics", "algorithm",
            "optimization", "cloud computing", "aws", "azure", "gcp", "devops", "ci/cd",
            "docker", "kubernetes", "jenkins", "git", "agile", "scrum", "project management",
            "product management", "leadership", "communication", "presentation", "teamwork",
            "problem solving", "critical thinking", "creativity", "time management",
            "customer service", "sales", "marketing", "seo", "content creation", "editing",
            "copywriting", "public speaking", "negotiation", "conflict resolution",
            "financial analysis", "budgeting", "forecasting", "accounting", "auditing",
            "compliance", "risk management", "regulatory", "legal", "research", "analytics",
            "business intelligence", "data visualization", "tableau", "power bi", "excel",
            "microsoft office", "linux", "windows", "macos", "networking", "security",
            "cryptography", "blockchain", "mobile development", "ios", "android", "react native",
            "flutter", "ui/ux design", "graphic design", "wireframing", "prototyping",
            "user research", "testing", "qa", "automation", "manual testing", "performance testing",
            "database design", "data modeling", "etl", "data warehousing", "big data", "hadoop",
            "spark", "kafka", "api development", "microservices", "serverless", "rest api",
            "graphql", "oauth", "authentication", "authorization", "identity management"
        ])

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for better embedding quality while preserving key terms."""
        # Lowercase but preserve the original text for technical term matching
        original_text = text
        text = text.lower()

        # Replace common technical punctuation patterns before general cleanup
        for term in self.technical_terms:
            if term in text:
                # Create a temporary placeholder (e.g., c++ becomes cPLUSPLUS)
                placeholder = term.replace('+', 'PLUS').replace('#', 'SHARP').replace('.', 'DOT')
                placeholder = re.sub(r'[^\w]', 'X', placeholder)  # Replace remaining special chars
                text = text.replace(term, placeholder)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # More gentle special character removal - keep hyphens in compound words
        text = re.sub(r'[^\w\s\-]', ' ', text)

        # Restore technical terms from placeholders
        for term in self.technical_terms:
            placeholder = term.replace('+', 'PLUS').replace('#', 'SHARP').replace('.', 'DOT')
            placeholder = re.sub(r'[^\w]', 'X', placeholder)
            text = text.replace(placeholder, term)

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
        # Try to identify resume sections using common headers
        section_headers = [
            "education", "experience", "work experience", "employment", "skills",
            "technical skills", "projects", "professional experience", "certifications",
            "achievements", "publications", "languages", "interests", "summary",
            "objective", "profile", "volunteer", "activities", "references"
        ]

        # Try to split by resume sections first
        sections = []
        lines = document.split('\n')
        current_section = []

        for line in lines:
            line_lower = line.lower().strip()

            # Check if this line might be a section header
            if any(header in line_lower for header in section_headers) and len(line.strip()) < 50:
                # Save the previous section if it exists
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line.strip()]
            else:
                current_section.append(line.strip())

        # Add the last section
        if current_section:
            sections.append('\n'.join(current_section))

        # If we couldn't identify clear sections, fall back to paragraph splitting
        if len(sections) <= 1:
            sections = [s.strip() for s in re.split(r'\n\s*\n', document) if s.strip()]

            # If still no clear paragraphs, try sentence splitting
            if len(sections) <= 1:
                try:
                    sections = sent_tokenize(document)
                except:
                    # Fallback to a simple split by periods
                    sections = [s.strip() + '.' for s in document.split('.') if s.strip()]

        return sections

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
        """Extract key skills or phrases from text using SBERT and improved methods."""
        # Preprocess text
        clean_text = self._preprocess_text(text)

        # First attempt: Try to identify skills from predefined common skills
        identified_skills = []
        for skill in self.common_skills:
            if skill in clean_text:
                identified_skills.append(skill)

        # Look for technical terms
        for term in self.technical_terms:
            if term in clean_text and term not in identified_skills:
                identified_skills.append(term)

        # If we found enough skills from predefined lists, use those
        if len(identified_skills) >= top_n * 0.7:  # If we found at least 70% of desired skills
            return identified_skills[:top_n]

        # Otherwise, use more sophisticated extraction with SBERT

        # Extract candidate keyphrases using n-grams
        words = clean_text.split()
        unigrams = words
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]

        # Combine all candidates
        candidates = unigrams + bigrams + trigrams

        # Remove duplicates and very short candidates
        candidates = list(set([c for c in candidates if len(c) > 3]))

        # Prioritize candidates that might be skills
        prioritized_candidates = []
        other_candidates = []

        for c in candidates:
            if any(skill in c for skill in self.common_skills) or any(term in c for term in self.technical_terms):
                prioritized_candidates.append(c)
            else:
                other_candidates.append(c)

        # Combine, with prioritized candidates first
        candidates = prioritized_candidates + other_candidates

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
        result_skills = [candidates[idx] for idx in closest_indices]

        # Combine with identified skills and remove duplicates
        all_skills = list(set(identified_skills + result_skills))
        return all_skills[:top_n]

    def find_missing_skills(self, resume_text: str, job_text: str,
                           similarity_threshold: float = 0.75) -> Dict[str, Any]:
        """Identify skills from job description missing in resume using semantic matching."""
        # Extract key skills with improved method
        job_skills = self.extract_key_skills(job_text)

        # Segment the resume with improved method
        resume_segments = self._segment_document(resume_text)

        # Get embeddings
        job_skill_embeddings = self.model.encode(job_skills, convert_to_tensor=True)
        resume_segment_embeddings = self.model.encode(resume_segments, convert_to_tensor=True)

        # For each job skill, find the best matching resume segment
        missing_skills = []
        weak_skills = []
        present_skills = []

        # Adjust thresholds based on model and document length
        missing_threshold = 0.35  # Slightly higher to reduce false positives

        # Adjust similarity threshold based on resume length
        # Shorter resumes need a more lenient threshold
        if len(resume_text) < 1000:  # Short resume
            similarity_threshold = max(0.6, similarity_threshold - 0.1)
        elif len(resume_text) > 5000:  # Very detailed resume
            similarity_threshold = min(0.85, similarity_threshold + 0.05)

        for i, skill in enumerate(job_skills):
            # Calculate similarity with each resume segment
            skill_embedding = job_skill_embeddings[i].reshape(1, -1)
            similarities = util.pytorch_cos_sim(skill_embedding, resume_segment_embeddings)

            # Get the maximum similarity
            max_similarity = torch.max(similarities).item()
            best_segment_idx = torch.argmax(similarities).item()

            # Skip very generic skills that might cause false positives
            if skill in ['experienced', 'skills', 'knowledge', 'ability', 'proficient']:
                continue

            # Skip very short skills that might cause false positives
            if len(skill) < 4:
                continue

            # Classify as missing, weak, or present
            if max_similarity < missing_threshold:
                missing_skills.append({
                    'skill': skill,
                    'confidence': 1.0 - max_similarity
                })
            elif max_similarity < similarity_threshold:
                # For weak skills, get context from the best matching segment
                context = resume_segments[best_segment_idx]

                # Highlight the part of the context most relevant to the skill
                sentences = sent_tokenize(context)
                best_sentence = ""
                best_score = 0

                for sentence in sentences:
                    score = util.pytorch_cos_sim(
                        self.model.encode(skill, convert_to_tensor=True),
                        self.model.encode(sentence, convert_to_tensor=True)
                    ).item()

                    if score > best_score:
                        best_score = score
                        best_sentence = sentence

                # Use the best sentence as context if it's a good match
                if best_score > 0.5:
                    context = best_sentence

                weak_skills.append({
                    'skill': skill,
                    'confidence': 1.0 - max_similarity,
                    'context': context
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

        # Remove potential duplicate skills (e.g., "python" and "python programming")
        missing_skills = self._remove_duplicate_skills(missing_skills)
        weak_skills = self._remove_duplicate_skills(weak_skills)
        present_skills = self._remove_duplicate_skills(present_skills)

        # Calculate overall similarity
        overall_similarity = self.calculate_similarity(resume_text, job_text)

        return {
            'missing_skills': missing_skills,
            'weak_skills': weak_skills,
            'present_skills': present_skills,
            'similarity_score': overall_similarity
        }

    def _remove_duplicate_skills(self, skills_list: List[Dict]) -> List[Dict]:
        """Remove duplicate or highly similar skills from a list."""
        if not skills_list:
            return []

        # Sort by confidence first
        sorted_skills = sorted(skills_list, key=lambda x: x['confidence'], reverse=True)

        # Keep track of processed skills
        unique_skills = []
        processed_skills = set()

        for skill_info in sorted_skills:
            skill = skill_info['skill'].lower()

            # Check if this skill is a subset or superset of already processed skills
            is_duplicate = False
            for processed in processed_skills:
                # If one skill is contained within another
                if skill in processed or processed in skill:
                    # Only consider as duplicate if they're very similar in length
                    if abs(len(skill) - len(processed)) < 5:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_skills.append(skill_info)
                processed_skills.add(skill)

        return unique_skills
