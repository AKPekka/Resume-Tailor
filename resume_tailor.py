class ResumeTailor:
    def __init__(self):
        # Common resume sections
        self.resume_sections = ["skills", "experience", "education", "projects", "summary"]

    def identify_resume_sections(self, resume_text):
        """Identify different sections in the resume"""
        sections = {}

        # Convert to lowercase for case-insensitive matching
        resume_lower = resume_text.lower()

        for section in self.resume_sections:
            # Find the starting index of the section
            section_index = resume_lower.find(section)

            if section_index != -1:
                sections[section] = section_index

        # Sort sections by their appearance in the resume
        sorted_sections = sorted(sections.items(), key=lambda x: x[1])

        # Extract the text for each section
        section_texts = {}
        for i, (section_name, start_idx) in enumerate(sorted_sections):
            # If this is the last section, extract until the end
            if i == len(sorted_sections) - 1:
                section_texts[section_name] = resume_text[start_idx:]
            else:
                # Otherwise, extract until the next section
                next_section_name, next_idx = sorted_sections[i + 1]
                section_texts[section_name] = resume_text[start_idx:next_idx]

        return section_texts

    def generate_tailoring_suggestions(self, keyword_results, resume_text):
        """Generate actionable suggestions for resume tailoring"""
        missing_keywords = keyword_results['missing_keywords']
        similarity_score = keyword_results['similarity_score']

        # Identify resume sections
        resume_sections = self.identify_resume_sections(resume_text)

        suggestions = []

        # Generate suggestions for each missing keyword
        for keyword in missing_keywords:
            # Determine the best section to add this keyword
            if self._is_skill(keyword):
                section = "skills"
                suggestion = f"Add '{keyword}' to your Skills section"
            elif self._is_education_related(keyword):
                section = "education"
                suggestion = f"Mention '{keyword}' in your Education section"
            else:
                # Default to Experience section for most keywords
                section = "experience"
                suggestion = f"Incorporate '{keyword}' when describing your work experience"

            suggestions.append({
                'keyword': keyword,
                'suggestion': suggestion,
                'section': section
            })

        # Generate general suggestions based on similarity score
        if similarity_score < 0.5:
            suggestions.append({
                'keyword': None,
                'suggestion': "Your resume needs significant tailoring to match this job description",
                'section': 'general'
            })
        elif similarity_score < 0.7:
            suggestions.append({
                'keyword': None,
                'suggestion': "Consider reorganizing your resume to emphasize the skills mentioned in the job description",
                'section': 'general'
            })

        # Add word choice suggestions
        action_verbs = self._suggest_action_verbs()
        suggestions.append({
            'keyword': None,
            'suggestion': f"Use strong action verbs like: {', '.join(action_verbs[:5])}",
            'section': 'language'
        })

        return suggestions

    def _is_skill(self, keyword):
        """Determine if a keyword is likely a skill"""
        common_skills = [
            # Technical Skills
            "python", "java", "javascript", "c++", "c#", "php", "ruby", "swift", "kotlin", "golang", "rust", "scala",
            "html", "css", "sql", "nosql", "mongodb", "postgresql", "mysql", "oracle", "database",
            "react", "angular", "vue", "node.js", "django", "flask", "spring", "asp.net", ".net", "jquery",
            "aws", "azure", "gcp", "cloud", "docker", "kubernetes", "terraform", "ansible", "ci/cd", "jenkins", "git",
            "linux", "unix", "windows", "macos", "bash", "powershell",
            "machine learning", "deep learning", "artificial intelligence", "ai", "nlp", "data science", "data analysis",
            "statistics", "big data", "spark", "hadoop", "kafka", "tableau", "power bi", "excel", "api", "rest", "graphql",
            "microservices", "serverless", "cybersecurity", "networking", "penetration testing", "encryption",
            # Soft Skills & Business Skills
            "leadership", "communication", "project management", "product management", "agile", "scrum", "kanban",
            "problem solving", "critical thinking", "analytical skills", "research", "design", "ux/ui", "user experience",
            "user interface", "prototyping", "wireframing", "testing", "qa", "automation", "devops",
            "strategy", "marketing", "sales", "business development", "customer service", "operations", "finance",
            "budgeting", "forecasting", "negotiation", "presentation", "collaboration", "teamwork", "mentoring"
        ]

        for skill in common_skills:
            if skill in keyword.lower():
                return True

        return False

    def _is_education_related(self, keyword):
        """Determine if a keyword is related to education"""
        education_terms = [
            "degree", "bachelor", "b.s.", "b.a.", "master", "m.s.", "m.a.", "mba", "phd", "doctorate",
            "associate", "diploma", "certificate", "certification", "licensed", "cpa", "pmp", "cfa",
            "course", "training", "program", "studies", "research", "thesis", "dissertation",
            "university", "college", "school", "institute", "academy", "education", "graduate", "undergraduate",
            "major", "minor", "gpa"
        ]

        for term in education_terms:
            if term in keyword.lower():
                return True

        return False

    def _suggest_action_verbs(self):
        """Suggest strong action verbs for resume bullet points"""
        return [
            "Achieved", "Developed", "Implemented", "Managed", "Led", "Created",
            "Designed", "Improved", "Reduced", "Increased", "Executed", "Coordinated",
            "Delivered", "Transformed", "Streamlined", "Optimized", "Generated",
            "Negotiated", "Facilitated", "Launched", "Pioneered", "Spearheaded"
        ]
