"""
Reviewer service implementation.

This module implements the reviewer service interface, providing functionality for
analyzing reviewer comments and extracting structured feedback.
"""

import logging
import re
from typing import Dict, Any, List, Optional

from src.core.context import RevisionContext
from src.domain.reviewer_comment import ReviewerComment, Comment, AssessmentType
from src.services.interfaces import ReviewerServiceInterface
from src.core.constants import SYSTEM_PROMPTS
from src.core.json_utils import parse_json_safely


class ReviewerService(ReviewerServiceInterface):
    """
    Service for reviewer comment analysis.
    
    This service is responsible for analyzing reviewer comments and extracting
    structured feedback, as well as processing editor requirements.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the reviewer service.
        
        Args:
            context: The shared revision context
        """
        self.context = context
        self.logger = context.logger or logging.getLogger(__name__)
        self.reviewer_comments = []
        self.editor_requirements = {}
    
    def analyze_reviewer_comments(self) -> List[ReviewerComment]:
        """
        Analyze reviewer comments and extract structured feedback.
        
        Returns:
            List of ReviewerComment domain entities
        """
        self.logger.info("Analyzing reviewer comments")
        
        reviewer_files = self.context.reviewer_comment_files
        if not reviewer_files:
            self.logger.warning("No reviewer comment files found")
            return []
        
        reviewer_comments = []
        for i, file_path in enumerate(reviewer_files, 1):
            try:
                # Read the reviewer comment file
                with open(file_path, 'r', encoding='utf-8') as f:
                    comment_text = f.read()
                
                # Create prompt for reviewer comment analysis
                prompt = f"""
                I'm analyzing reviewer comments for a scientific paper. Extract the key feedback in a structured format.
                
                REVIEWER COMMENT TEXT:
                ```
                {comment_text[:8000] if self.context.optimize_costs else comment_text}
                ```
                
                Based on this reviewer feedback, provide a comprehensive analysis including:
                1. Overall assessment (positive, neutral, or negative)
                2. Main concerns
                3. Required changes
                4. Suggested changes (but not required)
                5. Comments about methodology
                6. Comments about results and findings
                7. Comments about writing and clarity
                8. Comments about references and citations
                
                IMPORTANT: Format the response as a valid JSON object with these fields. Do not include any explanatory text before or after the JSON. The response should begin with '{{' and end with '}}'.
                
                Example format:
                {{
                  "overall_assessment": "negative",
                  "main_concerns": ["Concern 1", "Concern 2"],
                  "required_changes": ["Change 1", "Change 2"],
                  "suggested_changes": ["Suggestion 1", "Suggestion 2"],
                  "methodology_comments": ["Comment about methodology"],
                  "results_comments": ["Comment about results"],
                  "writing_comments": ["Comment about writing"],
                  "references_comments": ["Comment about references"]
                }}
                """
                
                # Get analysis from LLM
                analysis_json = self._get_completion(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPTS.get("reviewer_analysis", "You are an expert academic reviewer"),
                    max_tokens=2000 if self.context.optimize_costs else 3000
                )
                
                # Parse the LLM response
                analysis = parse_json_safely(analysis_json)
                
                # Extract detailed comments
                detailed_comments = self._extract_detailed_comments(comment_text)
                
                # Map the assessment string to enum
                assessment_str = analysis.get("overall_assessment", "neutral").lower()
                if "positive" in assessment_str:
                    assessment = AssessmentType.POSITIVE
                elif "negative" in assessment_str:
                    assessment = AssessmentType.NEGATIVE
                else:
                    assessment = AssessmentType.NEUTRAL
                
                # Create the ReviewerComment domain entity
                reviewer_comment = ReviewerComment(
                    reviewer_number=i,
                    overall_assessment=assessment,
                    main_concerns=analysis.get("main_concerns", []),
                    required_changes=analysis.get("required_changes", []),
                    suggested_changes=analysis.get("suggested_changes", []),
                    methodology_comments=analysis.get("methodology_comments", []),
                    results_comments=analysis.get("results_comments", []),
                    writing_comments=analysis.get("writing_comments", []),
                    references_comments=analysis.get("references_comments", []),
                    detailed_comments=detailed_comments,
                    full_text=comment_text,
                    text_preview=comment_text[:200] + "..." if len(comment_text) > 200 else comment_text
                )
                
                reviewer_comments.append(reviewer_comment)
                self.logger.info(f"Reviewer {i} analysis completed successfully")
                
            except Exception as e:
                self.logger.error(f"Error analyzing reviewer {i} comments: {e}")
                
                # Create a minimal fallback entity
                reviewer_comment = ReviewerComment(
                    reviewer_number=i,
                    overall_assessment=AssessmentType.NEUTRAL,
                    full_text=comment_text if 'comment_text' in locals() else "",
                    text_preview="Error analyzing reviewer comments"
                )
                reviewer_comments.append(reviewer_comment)
        
        self.reviewer_comments = reviewer_comments
        return reviewer_comments
    
    def analyze_editor_requirements(self) -> Dict[str, Any]:
        """
        Process editor letter and requirements.
        
        Returns:
            Dictionary with editor requirements
        """
        self.logger.info("Analyzing editor requirements")
        
        editor_file = self.context.editor_letter_path
        if not editor_file:
            self.logger.warning("No editor letter file found")
            return {}
        
        try:
            # Read the editor letter file
            with open(editor_file, 'r', encoding='utf-8') as f:
                editor_text = f.read()
            
            # Create prompt for editor letter analysis
            prompt = f"""
            I'm analyzing an editor's letter for a scientific paper revision. Extract the key requirements in a structured format.
            
            EDITOR LETTER TEXT:
            ```
            {editor_text[:5000] if self.context.optimize_costs else editor_text}
            ```
            
            Based on this editor's letter, provide a comprehensive analysis including:
            1. Decision (accept, minor revision, major revision, reject)
            2. Key requirements for revision
            3. Deadline for revision (if mentioned)
            4. Specific reviewer feedback to address
            5. Additional editor comments
            
            IMPORTANT: Format the response as a valid JSON object with these fields. Do not include any explanatory text before or after the JSON. The response should begin with '{{' and end with '}}'.
            
            Example format:
            {{
              "decision": "major revision",
              "key_requirements": ["Requirement 1", "Requirement 2"],
              "deadline": "2023-12-31",
              "reviewer_feedback_to_address": ["Address all comments from Reviewer 1", "Focus on methodology concerns from Reviewer 2"],
              "additional_comments": ["The editor suggests focusing on improving the clarity of the methodology section"]
            }}
            """
            
            # Get analysis from LLM
            analysis_json = self._get_completion(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPTS.get("editor_analysis", "You are an expert academic editor"),
                max_tokens=1500 if self.context.optimize_costs else 2000
            )
            
            # Parse the LLM response
            requirements = parse_json_safely(analysis_json)
            
            self.editor_requirements = requirements
            self.logger.info("Editor requirements analysis completed successfully")
            
            return requirements
            
        except Exception as e:
            self.logger.error(f"Error analyzing editor requirements: {e}")
            return {}
    
    def _extract_detailed_comments(self, comment_text: str) -> List[Comment]:
        """
        Extract detailed comments from reviewer text.
        
        Args:
            comment_text: The full reviewer comment text
            
        Returns:
            List of Comment objects
        """
        detailed_comments = []
        
        # Define patterns for different comment types
        patterns = {
            "concern": r"(concern|issue|problem|limitation|drawback)s?",
            "suggestion": r"(suggest|recommend|advise|propose)s?",
            "question": r"\b(why|how|what|when|where|who)\b|(\?)"
        }
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', comment_text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            # Determine comment type
            comment_type = "general"
            for ctype, pattern in patterns.items():
                if re.search(pattern, sentence, re.IGNORECASE):
                    comment_type = ctype
                    break
            
            # Determine importance (simple heuristic)
            importance = 1  # Default low importance
            
            # Check for importance indicators
            if re.search(r"(major|critical|significant|important|serious|main|key)", sentence, re.IGNORECASE):
                importance = 4  # High importance
            elif re.search(r"(minor|small|slight|little)", sentence, re.IGNORECASE):
                importance = 2  # Low importance
            elif re.search(r"(must|should|need to|require)", sentence, re.IGNORECASE):
                importance = 3  # Medium-high importance
            
            # Add to detailed comments
            detailed_comments.append(Comment(
                text=sentence,
                type=comment_type,
                importance=importance
            ))
        
        return detailed_comments
    
    def _get_completion(self, prompt: str, system_prompt: str, max_tokens: int) -> str:
        """
        Get a completion from the LLM with appropriate error handling.
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: The system prompt to use
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The LLM response text
            
        Raises:
            Exception: If there was an error getting the completion
        """
        try:
            # Ensure LLM client is initialized
            if not self.context.llm_client:
                self.context.setup_llm_client()
                
            # Get completion from LLM
            response = self.context.llm_client.get_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens
            )
            
            # Update budget
            tokens_used = self.context.llm_client.total_tokens_used
            cost = self.context.llm_client.total_cost
            self.context.update_budget(tokens_used, cost)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting completion: {e}")
            raise