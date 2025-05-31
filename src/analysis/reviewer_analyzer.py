"""
Reviewer analyzer module for extracting structured information from reviewer comments.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json

from src.core.context import RevisionContext
from src.utils.pdf_processor import PDFProcessor
from src.core.json_utils import parse_json_safely
from src.core.constants import SYSTEM_PROMPTS


class ReviewerAnalyzer:
    """
    Analyzes reviewer comments and extracts structured feedback.
    
    This class is responsible for extracting structured information from 
    reviewer comments, including assessment, concerns, and required changes.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the reviewer analyzer.
        
        Args:
            context: The shared revision context
        """
        self.context = context
        self.logger = context.logger or logging.getLogger(__name__)
    
    def analyze_reviewer_comments(self) -> List[Dict[str, Any]]:
        """
        Analyze the reviewer comments and extract structured feedback.
        
        Returns:
            List of dictionaries with reviewer comment analyses
        """
        reviewer_paths = [
            self.context.reviewer1_path,
            self.context.reviewer2_path,
            self.context.reviewer3_path
        ]
        reviewer_comments = []
        
        for i, path in enumerate(reviewer_paths, 1):
            self.logger.info(f"Analyzing reviewer {i} comments")
            
            # Process the PDF
            pdf_processor = PDFProcessor(path)
            self.context.process_statistics["files_processed"] = self.context.process_statistics.get("files_processed", 0) + 1
            
            # Extract text
            text = pdf_processor.text
            
            # Optimize token usage by focusing on key parts of the review
            # This uses the heuristic that most important comments are often in the first third and last third
            text_length = len(text)
            if self.context.optimize_costs and text_length > 3000:
                # For long reviews, take first 1000 chars + last 1000 chars
                trimmed_text = text[:1000] + "\n...[content trimmed]...\n" + text[-1000:]
                self.logger.debug(f"Trimmed reviewer {i} text from {text_length} to {len(trimmed_text)} chars")
            else:
                # For shorter reviews or when not optimizing, take up to 3000 chars
                max_length = 3000 if self.context.optimize_costs else 10000
                trimmed_text = text[:max_length] + ("..." if len(text) > max_length else "")
            
            # Create prompt for reviewer analysis
            prompt = f"""
            I'm analyzing reviewer comments for a scientific paper. Please extract only the most critical feedback.
            
            Here are the reviewer {i} comments:
            
            {trimmed_text}
            
            Provide a concise structured analysis with just these key points:
            1. Overall assessment (positive, neutral, negative)
            2. Main concerns (3-5 bullet points)
            3. Required changes (3-5 most important changes that must be addressed)
            
            IMPORTANT: Format the response as a valid JSON object with ONLY these fields. Do not include any explanatory text before or after the JSON.
            
            The JSON should look exactly like this:
            {{
                "overall_assessment": "positive|neutral|negative",
                "main_concerns": ["concern 1", "concern 2", "concern 3"],
                "required_changes": ["change 1", "change 2", "change 3"]
            }}
            """
            
            # Use the LLM to analyze the reviewer comments
            reviewer_analysis_json = self._get_completion(
                prompt=prompt,
                system_prompt=f"You are a scientific reviewer analysis assistant. Extract only the most critical feedback from reviewer {i}'s comments as JSON. You must return ONLY valid JSON in your response, no additional text or explanation.",
                max_tokens=1000  # Limit response size
            )
            
            try:
                # Parse the LLM response
                reviewer_analysis = parse_json_safely(reviewer_analysis_json)
                
                # Add default empty fields for optional analysis components
                reviewer_analysis.setdefault("suggested_changes", [])
                reviewer_analysis.setdefault("methodology_comments", [])
                reviewer_analysis.setdefault("results_comments", [])
                reviewer_analysis.setdefault("writing_comments", [])
                reviewer_analysis.setdefault("references_comments", [])
                
            except ValueError as e:
                # Fallback if LLM didn't return valid JSON
                self.logger.warning(f"LLM didn't return valid JSON for reviewer {i}. Error: {e}. Using basic analysis.")
                reviewer_analysis = {
                    "overall_assessment": "Unknown",
                    "main_concerns": ["Unknown concerns"],
                    "required_changes": ["Unknown required changes"],
                    "suggested_changes": [],
                    "methodology_comments": [],
                    "results_comments": [],
                    "writing_comments": [],
                    "references_comments": []
                }
            
            # To save memory/tokens, don't include full text when optimizing costs
            if not self.context.optimize_costs:
                reviewer_analysis["full_text"] = text
            else:
                # Just save the first 100 chars as a reference
                reviewer_analysis["text_preview"] = text[:100] + "..."
                
            reviewer_analysis["reviewer_number"] = i
            
            reviewer_comments.append(reviewer_analysis)
            pdf_processor.close()
        
        self.logger.info("Reviewer comment analysis completed")
        return reviewer_comments
    
    def analyze_editor_requirements(self) -> Dict[str, Any]:
        """
        Process editor letter and PRISMA requirements.
        
        Returns:
            Dictionary with editor requirements
        """
        # Process editor letter
        self.logger.info("Processing editor letter")
        editor_pdf = PDFProcessor(self.context.editor_letter_path)
        self.context.process_statistics["files_processed"] = self.context.process_statistics.get("files_processed", 0) + 1
        editor_text = editor_pdf.text
        editor_pdf.close()
        
        # Process PRISMA requirements
        self.logger.info("Processing PRISMA requirements")
        prisma_pdf = PDFProcessor(self.context.prisma_requirements_path)
        self.context.process_statistics["files_processed"] = self.context.process_statistics.get("files_processed", 0) + 1
        prisma_text = prisma_pdf.text
        prisma_pdf.close()
        
        # Optimize token usage
        if self.context.optimize_costs:
            # Editor text: Focus on first part of letter (usually contains key decisions)
            editor_text_trimmed = editor_text[:1500]
            
            # For PRISMA: Extract just the main checklist items
            import re
            prisma_lines = prisma_text.split('\n')
            prisma_checklist = []
            for line in prisma_lines:
                # Look for lines that might be checklist items (often numbered or bulleted)
                if re.search(r'^\s*(\d+[\.\)]|\*|\-|\•)', line) and len(line) < 200:
                    prisma_checklist.append(line.strip())
            
            # If we found checklist items, use them; otherwise take the beginning
            if prisma_checklist:
                prisma_text_trimmed = "\n".join(prisma_checklist)
            else:
                prisma_text_trimmed = prisma_text[:1500]
        else:
            # Use more text when not optimizing
            editor_text_trimmed = editor_text[:3000]
            prisma_text_trimmed = prisma_text[:3000]
        
        # Create prompt for editor/PRISMA requirements
        prompt = f"""
        I'm analyzing editor requirements and PRISMA guidelines for a scientific paper revision.
        
        EDITOR LETTER:
        ```
        {editor_text_trimmed}
        ```
        
        PRISMA REQUIREMENTS:
        ```
        {prisma_text_trimmed}
        ```
        
        Extract the key requirements into a structured format, focusing on:
        1. Editor's overall decision (accept, minor revision, major revision, reject)
        2. Specific editor requirements
        3. Key PRISMA checklist items that must be addressed
        
        IMPORTANT: Format the response as a valid JSON object with these fields. Do not include any explanatory text before or after the JSON.
        
        Example format:
        {{
          "editor_decision": "major revision",
          "editor_requirements": ["requirement 1", "requirement 2"],
          "prisma_requirements": ["item 1", "item 2"],
          "revision_deadline": "if mentioned in the text"
        }}
        """
        
        # Use the LLM to analyze the requirements
        requirements_json = self._get_completion(
            prompt=prompt,
            system_prompt="You are a scientific publication requirements analyzer. Extract the key requirements from editor letters and PRISMA guidelines.",
            max_tokens=1500 if self.context.optimize_costs else 2000
        )
        
        try:
            # Parse the LLM response
            requirements = parse_json_safely(requirements_json)
            
            # Add metadata
            requirements["editor_text_length"] = len(editor_text)
            requirements["prisma_text_length"] = len(prisma_text)
            
            return requirements
            
        except ValueError as e:
            self.logger.warning(f"Error parsing editor requirements: {e}. Using basic requirements.")
            
            # Create fallback requirements
            fallback_requirements = {
                "editor_decision": "unknown",
                "editor_requirements": ["Address reviewer comments"],
                "prisma_requirements": ["Follow PRISMA guidelines"],
                "revision_deadline": "unknown",
                "editor_text_length": len(editor_text),
                "prisma_text_length": len(prisma_text),
                "error": str(e)
            }
            
            return fallback_requirements
    
    def _get_completion(self, prompt: str, system_prompt: str, max_tokens: int) -> str:
        """
        Get a completion from the LLM with appropriate error handling.
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: The system prompt to use
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The LLM response text
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