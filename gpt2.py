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

        # Create the prompt
        prompt = f"""
Generate professional resume bullet points that showcase {skill} skills.

Job Description Context:
{job_context}

Resume Context:
{resume_context}

Strong bullet points should:
- Start with action verbs like {', '.join(selected_verbs)}
- Be concise and quantify achievements
- Focus on the skill: {skill}

Examples:
• Developed a Python data analysis pipeline that reduced processing time by 45%
• Led a cross-functional team of 8 engineers to implement new cloud infrastructure

Resume Bullet Points:
• """

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)

        # Generate bullet points
        bullet_points = []

        for _ in range(num_bullets):
            # Generate text
            output = self.model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
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

                # Update the prompt to include this bullet point
                prompt += f"{bullet_text}\n• "
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)

        return bullet_points

    def _clean_bullet_point(self, text: str) -> str:
        """Clean up a generated bullet point."""
        # Stop at the first newline or bullet point
        text = text.split('\n')[0].split('•')[0]

        # Remove any trailing punctuation or whitespace
        text = text.strip().rstrip(',.:;')

        # Ensure proper capitalization
        if text and len(text) > 0:
            text = text[0].upper() + text[1:]

        return text

    def _is_duplicate(self, text: str, existing_bullets: List[str]) -> bool:
        """Check if a bullet point is too similar to existing ones."""
        # Simple duplicate check based on word overlap
        text_words = set(text.lower().split())

        for bullet in existing_bullets:
            bullet_words = set(bullet.lower().split())
            # If more than 60% of words overlap, consider it a duplicate
            if len(text_words.intersection(bullet_words)) > 0.6 * min(len(text_words), len(bullet_words)):
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
