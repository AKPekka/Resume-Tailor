import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
import re
from typing import List, Dict, Any
import random # Added for random.sample in generate_bullet_points
# Assuming KeywordExtractor is in a module named keyword_extractor
# from keyword_extractor import KeywordExtractor # If direct import is preferred and path is set up

class ResumeBulletGenerator:
    def __init__(self, keyword_extractor_instance, model_name='gpt2-medium'):
        """Initialize the resume bullet point generator.

        Args:
            keyword_extractor_instance: An instance of KeywordExtractor.
            model_name: The model to use (e.g., 'gpt2-medium', 'deepseek-ai/deepseek-coder-1.3b-instruct').
        """
        self.keyword_extractor = keyword_extractor_instance
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # MPS (Apple Silicon GPU) check - might be useful for future optimization
        # if torch.backends.mps.is_available() and self.device == "cpu":
        #     self.device = "mps"
        
        print(f"[ResumeBulletGenerator] Initializing with model: {self.model_name} on device: {self.device}")

        try:
            if "deepseek" in self.model_name.lower():
                print(f"[ResumeBulletGenerator] Loading DeepSeek Tokenizer: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                print(f"[ResumeBulletGenerator] Loading DeepSeek Model: {self.model_name}")
                if self.device == "cuda":
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
                        ).to(self.device)
                        print(f"[ResumeBulletGenerator] DeepSeek model loaded on CUDA with bfloat16.")
                    except RuntimeError as e:
                        print(f"[ResumeBulletGenerator] Failed to load {self.model_name} with bfloat16 on CUDA: {e}. Trying default precision.")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name, trust_remote_code=True
                        ).to(self.device)
                        print(f"[ResumeBulletGenerator] DeepSeek model loaded on CUDA with default precision.")
                else: # CPU or MPS (though MPS not explicitly handled for dtype yet)
                     self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name, trust_remote_code=True
                    ).to(self.device) # For CPU, bfloat16 might not be optimal or supported by default ops
                     print(f"[ResumeBulletGenerator] DeepSeek model loaded on {self.device}.")
            else: # Assuming GPT-2 family
                print(f"[ResumeBulletGenerator] Loading GPT-2 Tokenizer: {self.model_name}")
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
                print(f"[ResumeBulletGenerator] Loading GPT-2 Model: {self.model_name}")
                self.model = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.device)
                print(f"[ResumeBulletGenerator] GPT-2 model loaded on {self.device}.")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            print(f"[ResumeBulletGenerator] Tokenizer pad token set.")

        except Exception as e:
            print(f"[ResumeBulletGenerator] FATAL Error loading model {self.model_name}: {e}")
            raise

        self.action_verbs = [
            "Achieved", "Developed", "Implemented", "Managed", "Led", "Created",
            "Designed", "Improved", "Reduced", "Increased", "Executed", "Coordinated",
            "Delivered", "Transformed", "Streamlined", "Optimized", "Generated",
            "Negotiated", "Facilitated", "Launched", "Pioneered", "Spearheaded",
            "Analyzed", "Engineered", "Established", "Boosted", "Automated",
            "Resolved", "Cultivated", "Supervised", "Administered", "Mentored"
        ]

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
                              resume_context: str = "", num_bullets: int = 1, # Reduced for debugging
                              max_length: int = 60) -> List[str]:
        print(f"[ResumeBulletGenerator] Generating {num_bullets} bullet(s) for skill: '{skill}', model: {self.model_name}")
        selected_verbs = random.sample(self.action_verbs, min(len(self.action_verbs), 3))
        key_phrases = self._extract_key_phrases(job_context)
        skill_examples_for_prompt = self._find_skill_examples(skill)

        prompt_content = f"""
JOB REQUIREMENTS:
{self._extract_relevant_sections(job_context, skill, 100)}

BULLET POINT GUIDELINES:
1. Start with action verbs like: {', '.join(selected_verbs)}
2. Focus on {skill.upper()} skills.
3. Quantify achievements.
4. Use terminology: {', '.join(key_phrases[:3])}
PREVIOUS RESUME CONTEXT:
{self._extract_relevant_sections(resume_context, skill, 70)}
EXCELLENT EXAMPLES:
{self._format_examples(skill_examples_for_prompt)}
"""

        if "deepseek" in self.model_name.lower():
            full_prompt = f"""You are an expert resume writing assistant.
### Instruction:
Write {num_bullets} professional resume bullet points for {skill.upper()}.
{prompt_content}
Generate {num_bullets} bullet points starting with "• ".

### Response:
• """
            generation_prefix_for_stripping = "• "
        else: # GPT-2 style prompt
            full_prompt = f"""
Write professional resume bullet points for {skill.upper()}. {prompt_content} BULLET POINTS:
• """
            generation_prefix_for_stripping = "• "
        
        print(f"[ResumeBulletGenerator] Prompt length: {len(full_prompt)} chars")
        inputs = self.tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)
        
        bullet_points = []
        current_generated_text_for_prompt = ""

        for i in range(num_bullets):
            active_prompt_for_generation = full_prompt # Default to full prompt
            if i > 0 and len(bullet_points) > 0:
                # Simplified iterative prompting for now
                current_generated_text_for_prompt += bullet_points[-1] + "\n• "
                if "deepseek" in self.model_name.lower():
                    active_prompt_for_generation = f"""You are an expert resume writing assistant.
### Instruction:
Write {num_bullets - len(bullet_points)} more bullet points for {skill.upper()}.
{prompt_content}
Previously generated:
{current_generated_text_for_prompt}
Generate next bullet point starting with "• ".

### Response:
• """
                else:
                    active_prompt_for_generation = full_prompt.replace("BULLET POINTS:", f"PREVIOUSLY GENERATED:\n{current_generated_text_for_prompt}\nMORE BULLET POINTS:")
                
                print(f"[ResumeBulletGenerator] Iteration {i+1} - Iterative prompt length: {len(active_prompt_for_generation)} chars")
                current_inputs = self.tokenizer(active_prompt_for_generation, return_tensors="pt", padding=True, truncation=True, max_length=1500).to(self.device)
                input_ids = current_inputs.input_ids
                attention_mask = current_inputs.attention_mask
            else:
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
            
            print(f"[ResumeBulletGenerator] Iteration {i+1} - Calling model.generate() for '{skill}'...")
            try:
                output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_length,
                    num_return_sequences=1, temperature=0.7, top_p=0.9, top_k=50, do_sample=True,
                    no_repeat_ngram_size=3, pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id, repetition_penalty=1.2
                )
                print(f"[ResumeBulletGenerator] model.generate() completed for '{skill}'. Output length: {len(output[0])}")
                generated_token_ids = output[0][input_ids.shape[-1]:]
                generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                print(f"[ResumeBulletGenerator] Decoded text: '{generated_text[:100]}...'")
                
                bullet_text = self._clean_bullet_point(generated_text, generation_prefix_for_stripping if i == 0 and not current_generated_text_for_prompt else "")
                print(f"[ResumeBulletGenerator] Cleaned bullet: '{bullet_text}'")

                if bullet_text and not self._is_duplicate(bullet_text, bullet_points):
                    bullet_points.append(bullet_text)
                    if len(bullet_points) == num_bullets: break 
            except Exception as e:
                print(f"[ResumeBulletGenerator] ERROR during model.generate() or processing for skill '{skill}': {e}")
                # Optionally, `st.exception(e)` if running in Streamlit context and want it in UI
                break # Stop trying for this skill if an error occurs

        print(f"[ResumeBulletGenerator] Finished generating bullets for skill '{skill}'. Got {len(bullet_points)} bullets.")
        return bullet_points

    def _extract_relevant_sections(self, text: str, keyword: str, max_chars: int = 70) -> str:
        """Extract sections of text most relevant to the skill/keyword."""
        if not text:
            return ""
        paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
        keyword_lower = keyword.lower()
        relevant_paragraphs = [p for p in paragraphs if keyword_lower in p.lower()]
        if not relevant_paragraphs and paragraphs:
            return ' '.join(paragraphs[:2])[:max_chars]
        return ' '.join(relevant_paragraphs)[:max_chars]

    def _extract_key_phrases(self, text: str, max_phrases: int = 3) -> List[str]:
        """Extract potential key phrases from job description text using KeyBERT."""
        if not text or self.keyword_extractor is None:
            return []
        try:
            key_phrases_list = self.keyword_extractor.extract_keywords(
                text,
                top_n=max_phrases,
                keyphrase_ngram_range=(1, 2) 
            )
            return key_phrases_list
        except Exception as e:
            print(f"[RBG KeywordError]: {e}")
            return []

    def _find_skill_examples(self, skill: str) -> List[str]:
        """Find example bullets for this skill or similar skills."""
        skill_lower = skill.lower()
        if skill_lower in self.skill_examples:
            return self.skill_examples[skill_lower]
        for k, examples in self.skill_examples.items():
            if k in skill_lower or skill_lower in k:
                return examples
        return []

    def _format_examples(self, examples: List[str]) -> str:
        """Format example bullets nicely for the prompt."""
        if not examples:
            return "None specific."
        formatted = ""
        for i, example in enumerate(examples[:1], 1):
            formatted += f"{i}. • {example}\n"
        return formatted.strip()

    def _clean_bullet_point(self, text: str, prefix_to_remove: str = "") -> str:
        """Clean up a generated bullet point."""
        if prefix_to_remove and text.startswith(prefix_to_remove):
            text = text[len(prefix_to_remove):]

        # Stop at the first true newline, or a new bullet if model generates multiple
        if '\n' in text:
            text = text.split('\n')[0]
        if '•' in text: # check after initial prefix removal
            text = text.split('•')[0]
        
        text = text.strip().rstrip(',.:;').replace("\"","")
        text = re.sub(r'^[•\-\*\s]*', '', text) # Remove any leading bullet characters or whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if not text: return ""

        # Ensure proper capitalization
        text = text[0].upper() + text[1:]

        # Basic quality check (e.g., too short, no verb) - can be expanded
        # For now, keeping it simple, as the prompt guides verb usage.
        if len(text.split()) < 3: # Minimum words for a decent bullet
             # Check if it contains any action verb from the list as a loose proxy for being a valid statement
            if not any(verb.lower() in text.lower() for verb in self.action_verbs[:15]): # check against a subset
                # print(f"Filtered out short/non-action bullet: {text}")
                return ""
        
        return text

    def _is_duplicate(self, text: str, existing_bullets: List[str], threshold=0.7) -> bool:
        """Check if a bullet point is too similar to existing ones using simple word overlap."""
        if not text: return True # Empty text is a duplicate of nothing useful
        
        text_words = set(text.lower().split())
        if len(text_words) < 3 : return True # Too short to be a unique, useful bullet

        for bullet in existing_bullets:
            bullet_words = set(bullet.lower().split())
            if not bullet_words: continue

            common_words = text_words.intersection(bullet_words)
            # Jaccard similarity:
            # similarity = len(common_words) / (len(text_words) + len(bullet_words) - len(common_words))
            # Simpler overlap:
            overlap_ratio = len(common_words) / min(len(text_words), len(bullet_words)) if min(len(text_words), len(bullet_words)) > 0 else 0
            
            if overlap_ratio > threshold:
                # print(f"Duplicate detected (ratio {overlap_ratio:.2f}): '{text}' vs '{bullet}'")
                return True
        return False

    def generate_suggestions(self, semantic_results: Dict[str, Any],
                            job_text: str, resume_text: str) -> Dict[str, Any]:
        """Generate tailored suggestions based on semantic matching results."""
        print(f"[ResumeBulletGenerator] Starting generate_suggestions with model {self.model_name}.")
        suggestions = {
            'missing_skills': [],
            'weak_skills': []
        }

        missing_skills_data = semantic_results.get('missing_skills', [])
        weak_skills_data = semantic_results.get('weak_skills', [])

        # DEBUG: Reduce workload
        # Process only 1 missing skill
        print(f"[ResumeBulletGenerator] Processing {len(missing_skills_data)} missing skills (will take max 1 for debug).")
        for skill_info in missing_skills_data[:1]: # DEBUG: Process only first missing skill
            skill = skill_info['skill']
            print(f"[ResumeBulletGenerator] Generating bullets for MISSING skill: {skill}")
            bullets = self.generate_bullet_points(
                skill=skill,
                job_context=job_text,
                resume_context=resume_text, # Provide general resume context
                num_bullets=1 
            )
            if bullets: # Only add if bullets were generated
                suggestions['missing_skills'].append({
                    'skill': skill,
                    'confidence': skill_info.get('confidence', 0.0),
                    'bullet_points': bullets
                })

        # DEBUG: Reduce workload
        # Process only 1 weak skill
        print(f"[ResumeBulletGenerator] Processing {len(weak_skills_data)} weak skills (will take max 1 for debug).")
        for skill_info in weak_skills_data[:1]: # DEBUG: Process only first weak skill
            skill = skill_info['skill']
            context = skill_info.get('context', resume_text) # Use specific context if available, else full resume
            
            bullets = self.generate_bullet_points(
                skill=skill,
                job_context=job_text,
                resume_context=context, # Provide specific context for weak skill
                num_bullets=1
            )
            if bullets: # Only add if bullets were generated
                suggestions['weak_skills'].append({
                    'skill': skill,
                    'confidence': skill_info.get('confidence', 0.0),
                    'current_context': skill_info.get('context', 'N/A'),
                    'improved_bullet_points': bullets
                })
        print(f"[ResumeBulletGenerator] Finished generate_suggestions. Output: {str(suggestions)[:200]}...")
        return suggestions
