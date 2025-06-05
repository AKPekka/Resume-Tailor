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

        # Common technical terms that should be preserved during preprocessing
        self.technical_terms = set([
            "c++", "c#", ".net", "node.js", "react.js", "vue.js", "angular.js",
            "aws", "azure", "gcp", "api", "rest", "graphql", "sql", "nosql",
            "mongodb", "postgresql", "mysql", "redis", "docker", "kubernetes", "k8s",
            "ci/cd", "jenkins", "github", "gitlab", "bitbucket", "agile", "scrum",
            "kanban", "jira", "confluence", "aws/azure", "java/python", "html/css",
            "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn", "keras",
            "nlp", "ml", "ai", "ux/ui", "a/b", "seo/sem", "ios/android"
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

    def find_missing_skills(self, resume_text: str, job_text: str,
                           job_keywords: List[str],
                           similarity_threshold: float = 0.75) -> Dict[str, Any]:
        """Identify skills from provided job_keywords missing in resume using semantic matching."""
        # Segment the resume with improved method
        resume_segments = self._segment_document(resume_text)
        if not resume_segments: # Handle empty resume or segmentation failure
            resume_segments = [self._preprocess_text(resume_text)] if resume_text.strip() else [""]

        # Get embeddings
        if not job_keywords: # No keywords to search for
             # Calculate overall similarity and return early
            overall_similarity = self.calculate_similarity(resume_text, job_text)
            return {
                'missing_skills': [],
                'weak_skills': [],
                'present_skills': [],
                'similarity_score': overall_similarity
            }

        job_skill_embeddings = self.model.encode(job_keywords, convert_to_tensor=True)
        
        # Ensure resume_segments are not empty before encoding
        if not any(s.strip() for s in resume_segments):
            resume_segment_embeddings = self.model.encode([""], convert_to_tensor=True) # Avoid error with empty list
        else:
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

        for i, skill in enumerate(job_keywords):
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
