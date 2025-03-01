import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
from typing import List, Dict, Any

class ResumeBulletGenerator:
    def __init__(self, model_name='gpt2-medium'):
        """Initialize the resume bullet point generator with GPT-2.

        Args:
            model_name: The GPT-2 model to use. Options include:
                        'gpt2' (small), 'gpt2-medium', 'gpt2-large'
        """
        # Load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        # Set padding token to be the same as the EOS token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Action verbs commonly used in resumes
        self.action_verbs = [
            "Achieved", "Developed", "Implemented", "Managed", "Led", "Created",
            "Designed", "Improved", "Reduced", "Increased", "Executed", "Coordinated",
            "Delivered", "Transformed", "Streamlined", "Optimized", "Generated",
            "Negotiated", "Facilitated", "Launched", "Pioneered", "Spearheaded",
            "Analyzed", "Engineered", "Established", "Boosted", "Automated",
            "Resolved", "Cultivated", "Supervised", "Administered", "Mentored"
        ]

        # Mapping of common skills to specific accomplishments for more targeted bullet points
        self.skill_examples = {
            "python": [
                "Developed a Python-based data pipeline that processed over 10,000 records daily, reducing manual work by 75%",
                "Built custom Python scripts to automate reporting, saving 15 hours per week of analyst time",
                "Engineered a Python application that integrated with REST APIs to pull and analyze customer data"
            ],
            "java": [
                "Developed a Java microservice that handled 5,000+ concurrent transactions with 99.9% uptime",
                "Created Java applications that reduced database query times by 40% through optimized algorithms",
                "Built scalable Java backends supporting mission-critical applications with 100K+ daily users"
            ],
            "sql": [
                "Optimized SQL queries that improved database performance by 60% and reduced server load",
                "Designed SQL database schemas that supported scalable growth from 10K to 1M+ records",
                "Implemented complex SQL stored procedures that automated reporting and reduced run times by 75%"
            ],
            "javascript": [
                "Developed responsive JavaScript front-end components that improved user engagement by 35%",
                "Built JavaScript applications that reduced page load times by 45% through optimized rendering",
                "Created interactive JavaScript data visualizations that increased dashboard utility for executive team"
            ],
            "project management": [
                "Led cross-functional team of 12 engineers to deliver project 2 weeks ahead of schedule and 10% under budget",
                "Managed 5 concurrent projects with combined budget of $1.2M while maintaining stakeholder satisfaction",
                "Implemented Agile methodology that increased team velocity by 40% and improved release predictability"
            ],
            "data analysis": [
                "Analyzed customer data to identify patterns that led to 25% increase in retention rate",
                "Conducted statistical analysis of market trends that informed strategic decisions resulting in $500K revenue increase",
                "Delivered actionable insights from complex datasets that guided product development priorities"
            ],
            "communication": [
                "Presented technical concepts to non-technical stakeholders, securing buy-in for $1.5M project",
                "Facilitated weekly meetings between engineering and business teams that improved cross-department collaboration",
                "Created documentation that reduced onboarding time for new team members by 30%"
            ],
            "leadership": [
                "Led team of 8 engineers to successful product launch that exceeded revenue targets by 30%",
                "Mentored 5 junior developers who were subsequently promoted within 18 months",
                "Directed reorganization of department structure that improved efficiency by 25%"
            ]
        }

    def generate_bullet_points(self, skill: str, job_context: str,
                              resume_context: str = "", num_bullets: int = 3,
                              max_length: int = 50) -> List[str]:
        """Generate resume bullet points for a specific skill.

        Args:
            skill: The skill to generate bullet points for
            job_context: Text from the job description for context
            resume_context: Optional existing text from the resume for context
            num_bullets: Number of bullet points to generate
            max_length: Maximum length of each bullet point in tokens

        Returns:
            List of generated bullet point strings
        """
        # Select a few random action verbs
        import random
        selected_verbs = random.sample(self.action_verbs, min(5, len(self.action_verbs)))

        # Extract key phrases from job context
        key_phrases = self._extract_key_phrases(job_context)

        # Find example bullets for the skill (or related skills)
        skill_examples = self._find_skill_examples(skill)

        # Create a more structured prompt with better examples
        prompt = f"""
Write professional resume bullet points for a skilled professional with expertise in {skill.upper()}.

JOB REQUIREMENTS:
{self._extract_relevant_sections(job_context, skill, 150)}

BULLET POINT GUIDELINES:
1. Start with powerful action verbs like: {', '.join(selected_verbs)}
2. Focus specifically on {skill.upper()} skills and achievements
3. Include specific metrics, percentages, or numbers to quantify achievements
4. Highlight outcomes and business impacts, not just responsibilities
5. Use industry terminology from the job description: {', '.join(key_phrases[:5])}
6. Keep each bullet concise and impactful (under 25 words)

PREVIOUS RESUME CONTEXT:
{self._extract_relevant_sections(resume_context, skill, 100)}

EXCELLENT EXAMPLES:
{self._format_examples(skill_examples)}

NOW GENERATE PROFESSIONAL {skill.upper()} BULLET POINTS:
• """

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)

        # Generate bullet points
        bullet_points = []

        for _ in range(num_bullets):
            # Generate text with more controlled parameters
            output = self.model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.6,  # Lower temperature for more focused output
                top_p=0.85,       # Slightly more focused sampling
                top_k=40,         # Limit vocabulary to top 40 tokens at each step
                do_sample=True,
                no_repeat_ngram_size=3,  # Avoid repeating 3-grams
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2   # Penalize repetition
            )

            # Decode the generated text
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract just the new bullet point
            bullet_text = generated_text[len(prompt):]

            # Clean up the bullet point
            bullet_text = self._clean_bullet_point(bullet_text)

            # Add to list if it's valid
            if bullet_text and not self._is_duplicate(bullet_text, bullet_points):
                bullet_points.append(bullet_text)

                # Update the prompt to include this bullet point for better coherence
                prompt += f"{bullet_text}\n• "
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)

        return bullet_points

    def _extract_relevant_sections(self, text: str, keyword: str, max_chars: int = 150) -> str:
        """Extract sections of text most relevant to the skill/keyword."""
        if not text:
            return ""

        # Find paragraphs containing the keyword or related terms
        paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]

        # Search for keyword (case insensitive)
        keyword_lower = keyword.lower()
        relevant_paragraphs = []

        for p in paragraphs:
            if keyword_lower in p.lower():
                relevant_paragraphs.append(p)

        # If no direct matches, get a sample of text
        if not relevant_paragraphs and paragraphs:
            return ' '.join(paragraphs[:2])[:max_chars]

        # Return concatenated relevant text, respecting max length
        return ' '.join(relevant_paragraphs)[:max_chars]

    def _extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract potential key phrases from job description text."""
        if not text:
            return []

        # Simple approach: look for capitalized terms or terms in quotes
        potential_terms = re.findall(r'\b[A-Z][A-Za-z0-9]+\b', text)  # Capitalized terms
        quoted_terms = re.findall(r'\"(.*?)\"', text)  # Terms in quotes

        # Combine and limit
        all_terms = list(set(potential_terms + quoted_terms))
        return all_terms[:max_phrases]

    def _find_skill_examples(self, skill: str) -> List[str]:
        """Find example bullets for this skill or similar skills."""
        skill_lower = skill.lower()

        # Direct match
        if skill_lower in self.skill_examples:
            return self.skill_examples[skill_lower]

        # Partial match
        for k, examples in self.skill_examples.items():
            if k in skill_lower or skill_lower in k:
                return examples

        # Default to a random set of examples if no match
        import random
        random_key = random.choice(list(self.skill_examples.keys()))
        return self.skill_examples[random_key]

    def _format_examples(self, examples: List[str]) -> str:
        """Format example bullets nicely for the prompt."""
        formatted = ""
        for i, example in enumerate(examples[:3], 1):  # Limit to 3 examples
            formatted += f"{i}. • {example}\n"
        return formatted

    def _clean_bullet_point(self, text: str) -> str:
        """Clean up a generated bullet point."""
        # Stop at the first newline or bullet point
        text = text.split('\n')[0].split('•')[0]

        # Remove any trailing punctuation or whitespace
        text = text.strip().rstrip(',.:;')

        # Remove any additional bullets that might have been generated
        text = re.sub(r'^[•\-\*]\s*', '', text)

        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)

        # Ensure proper capitalization
        if text and len(text) > 0:
            text = text[0].upper() + text[1:]

        # Ensure it's a complete sentence (has a verb)
        if len(text.split()) < 3 or not any(verb.lower() in text.lower() for verb in self.action_verbs):
            return ""

        return text

    def _is_duplicate(self, text: str, existing_bullets: List[str]) -> bool:
        """Check if a bullet point is too similar to existing ones."""
        # Simple duplicate check based on word overlap
        text_words = set(text.lower().split())

        for bullet in existing_bullets:
            bullet_words = set(bullet.lower().split())
            # If more than 50% of words overlap, consider it a duplicate
            if len(text_words.intersection(bullet_words)) > 0.5 * min(len(text_words), len(bullet_words)):
                return True

        # Check if it's too short
        if len(text.split()) < 5:
            return True

        return False

    def generate_suggestions(self, semantic_results: Dict[str, Any],
                            job_text: str, resume_text: str) -> Dict[str, Any]:
        """Generate tailored suggestions based on semantic matching results."""
        suggestions = {
            'missing_skills': [],
            'weak_skills': []
        }

        # Safety check for missing_skills
        missing_skills = semantic_results.get('missing_skills', [])
        if not missing_skills:
            # Return empty suggestions if no missing skills
            return suggestions

        # Generate bullet points for missing skills (safely)
        missing_skills_to_process = min(5, len(missing_skills))  # Process at most 5 missing skills
        for i in range(missing_skills_to_process):
            skill_info = missing_skills[i]
            skill = skill_info['skill']

            # Generate bullet points
            bullets = self.generate_bullet_points(
                skill=skill,
                job_context=job_text,
                resume_context=resume_text,
                num_bullets=2
            )

            suggestions['missing_skills'].append({
                'skill': skill,
                'confidence': skill_info['confidence'],
                'bullet_points': bullets
            })

        # Safety check for weak_skills
        weak_skills = semantic_results.get('weak_skills', [])
        if not weak_skills:
            # Return with only missing skills suggestions
            return suggestions

        # Generate improvements for weak skills (safely)
        weak_skills_to_process = min(3, len(weak_skills))  # Process at most 3 weak skills
        for i in range(weak_skills_to_process):
            skill_info = weak_skills[i]
            skill = skill_info['skill']
            context = skill_info.get('context', '')  # Safely get context

            # Generate bullet points
            bullets = self.generate_bullet_points(
                skill=skill,
                job_context=job_text,
                resume_context=context,
                num_bullets=2
            )

            suggestions['weak_skills'].append({
                'skill': skill,
                'confidence': skill_info['confidence'],
                'current_context': context,
                'improved_bullet_points': bullets
            })

        return suggestions
