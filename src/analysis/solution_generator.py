"""
Solution generator module for creating solutions to address reviewer comments.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json

from src.core.context import RevisionContext
from src.core.json_utils import parse_json_safely
from src.core.constants import SYSTEM_PROMPTS


class SolutionGenerator:
    """
    Generates solutions to address reviewer comments and paper issues.
    
    This class is responsible for analyzing paper issues and reviewer feedback,
    then generating appropriate solutions to address them.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the solution generator.
        
        Args:
            context: The shared revision context
        """
        self.context = context
        self.logger = context.logger or logging.getLogger(__name__)
    
    def identify_issues(
        self, 
        paper_analysis: Dict[str, Any],
        reviewer_comments: List[Dict[str, Any]],
        editor_requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify issues and concerns from paper analysis and reviewer comments.
        
        Args:
            paper_analysis: Analysis of the original paper
            reviewer_comments: Analysis of reviewer comments
            editor_requirements: Editor requirements and PRISMA guidelines
            
        Returns:
            List of identified issues
        """
        self.logger.info("Identifying issues from reviewer comments and paper analysis")
        
        # Extract key information for issue identification
        paper_summary = {
            "title": paper_analysis.get("title", "Unknown Title"),
            "objectives": paper_analysis.get("objectives", ["Unknown Objectives"]),
            "methodology": paper_analysis.get("methodology", "Unknown Methodology"),
            "findings": paper_analysis.get("findings", ["Unknown Findings"]),
            "limitations": paper_analysis.get("limitations", ["Unknown Limitations"])
        }
        
        # Extract reviewer concerns
        all_concerns = []
        all_required_changes = []
        
        for reviewer in reviewer_comments:
            reviewer_num = reviewer.get("reviewer_number", 0)
            concerns = reviewer.get("main_concerns", [])
            required = reviewer.get("required_changes", [])
            
            # Tag concerns with reviewer number
            for concern in concerns:
                all_concerns.append({
                    "concern": concern,
                    "reviewer": reviewer_num,
                    "assessment": reviewer.get("overall_assessment", "Unknown")
                })
            
            # Tag required changes with reviewer number
            for change in required:
                all_required_changes.append({
                    "change": change,
                    "reviewer": reviewer_num,
                    "assessment": reviewer.get("overall_assessment", "Unknown")
                })
        
        # Create prompt for issue identification
        prompt = f"""
        I'm identifying issues in a scientific paper based on reviewer comments and analysis.
        
        PAPER SUMMARY:
        {json.dumps(paper_summary, indent=2)}
        
        REVIEWER CONCERNS:
        {json.dumps(all_concerns, indent=2)}
        
        REQUIRED CHANGES:
        {json.dumps(all_required_changes, indent=2)}
        
        EDITOR REQUIREMENTS:
        {json.dumps(editor_requirements.get("editor_requirements", []), indent=2)}
        
        Based on this information, identify the key issues that need to be addressed in the revision.
        For each issue, provide:
        1. Issue description
        2. Severity (high, medium, low)
        3. Source (which reviewer or analysis)
        4. Type (methodology, results, writing, references, etc.)
        
        IMPORTANT: Format the response as a valid JSON array of objects with these fields. Do not include any explanatory text before or after the JSON. The response should begin with '[' and end with ']'.
        
        Example format:
        [
          {{
            "description": "Issue description here",
            "severity": "high",
            "source": "Reviewer 1",
            "type": "methodology"
          }},
          ...
        ]
        """
        
        # Use the LLM to identify issues
        issues_json = self._get_completion(
            prompt=prompt,
            system_prompt="You are a scientific paper issue analyzer. Identify key issues that need to be addressed in a paper revision.",
            max_tokens=2000 if self.context.optimize_costs else 3000
        )
        
        try:
            # Parse the LLM response
            issues = parse_json_safely(issues_json)
            
            # Ensure issues is a list
            if not isinstance(issues, list):
                if isinstance(issues, dict):
                    issues = [issues]
                else:
                    raise ValueError("Issues response is not a list or dictionary")
            
            self.logger.info(f"Identified {len(issues)} issues")
            return issues
            
        except ValueError as e:
            self.logger.warning(f"Error parsing identified issues: {e}. Using basic issues.")
            
            # Create fallback issues based on reviewer comments
            fallback_issues = []
            for i, reviewer in enumerate(reviewer_comments, 1):
                for concern in reviewer.get("main_concerns", [])[:2]:  # Take top 2 concerns
                    fallback_issues.append({
                        "description": concern,
                        "severity": "medium",
                        "source": f"Reviewer {i}",
                        "type": "unknown"
                    })
            
            self.logger.info(f"Created {len(fallback_issues)} fallback issues")
            return fallback_issues
    
    def generate_solutions(
        self,
        paper_analysis: Dict[str, Any],
        issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate solutions for identified issues.
        
        Args:
            paper_analysis: Analysis of the original paper
            issues: List of identified issues
            
        Returns:
            List of solution dictionaries
        """
        self.logger.info("Generating solutions for identified issues")
        
        # Extract key information for solution generation
        paper_summary = {
            "title": paper_analysis.get("title", "Unknown Title"),
            "objectives": paper_analysis.get("objectives", ["Unknown Objectives"]),
            "methodology": paper_analysis.get("methodology", "Unknown Methodology"),
            "findings": paper_analysis.get("findings", ["Unknown Findings"]),
            "limitations": paper_analysis.get("limitations", ["Unknown Limitations"])
        }
        
        # Create prompt for solution generation
        prompt = f"""
        I'm generating solutions for issues identified in a scientific paper.
        
        PAPER SUMMARY:
        {json.dumps(paper_summary, indent=2)}
        
        IDENTIFIED ISSUES:
        {json.dumps(issues, indent=2)}
        
        For each issue, generate a concrete solution that addresses the concern. 
        For each solution, provide:
        1. Title (brief description)
        2. Detailed implementation steps
        3. Expected impact on the paper
        4. Complexity of implementation (high, medium, low)
        5. Which issue(s) it addresses (by description)
        
        IMPORTANT: Format the response as a valid JSON array of objects with these fields. Do not include any explanatory text before or after the JSON. The response should begin with '[' and end with ']'.
        
        Example format:
        [
          {{
            "title": "Solution title here",
            "implementation": "Detailed implementation steps",
            "impact": "Expected impact on the paper",
            "complexity": "medium",
            "addresses": ["Issue description 1", "Issue description 2"]
          }},
          ...
        ]
        """
        
        # Use the LLM to generate solutions
        solutions_json = self._get_completion(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPTS["solution_generation"],
            max_tokens=2500 if self.context.optimize_costs else 4000
        )
        
        try:
            # Parse the LLM response
            solutions = parse_json_safely(solutions_json)
            
            # Ensure solutions is a list
            if not isinstance(solutions, list):
                if isinstance(solutions, dict):
                    solutions = [solutions]
                else:
                    raise ValueError("Solutions response is not a list or dictionary")
            
            self.logger.info(f"Generated {len(solutions)} solutions")
            return solutions
            
        except ValueError as e:
            self.logger.warning(f"Error parsing generated solutions: {e}. Using basic solutions.")
            
            # Create fallback solutions based on issues
            fallback_solutions = []
            for i, issue in enumerate(issues[:5]):  # Take top 5 issues
                fallback_solutions.append({
                    "title": f"Address {issue.get('type', 'issue')} - {i+1}",
                    "implementation": f"Revise the paper to address: {issue.get('description', 'Unknown issue')}",
                    "impact": "Improve paper quality and address reviewer concerns",
                    "complexity": issue.get("severity", "medium"),
                    "addresses": [issue.get("description", "Unknown issue")]
                })
            
            self.logger.info(f"Created {len(fallback_solutions)} fallback solutions")
            return fallback_solutions
    
    def generate_specific_changes(
        self,
        paper_analysis: Dict[str, Any],
        solutions: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, str, Optional[int]]]:
        """
        Generate specific text changes to implement solutions.
        
        Args:
            paper_analysis: Analysis of the original paper
            solutions: List of solutions to implement
            
        Returns:
            List of tuples (old_text, new_text, reason, line_number)
        """
        self.logger.info("Generating specific text changes")
        
        # Extract sections for context
        sections = paper_analysis.get("sections", {})
        section_samples = {}
        
        # Get sample content from each section for context
        for section_name, content in sections.items():
            section_samples[section_name] = content[:1000] + "..." if len(content) > 1000 else content
        
        # Summarize solutions for prompt
        solutions_summary = []
        for solution in solutions:
            summary = {
                "title": solution.get("title", "Unknown solution"),
                "implementation": solution.get("implementation", "Unknown implementation"),
                "complexity": solution.get("complexity", "medium")
            }
            solutions_summary.append(summary)
        
        # Create prompt for generating specific changes
        prompt = f"""
        I'm generating specific text changes for a scientific paper revision.
        
        Paper Sections:
        {json.dumps(section_samples, indent=2)}
        
        Solutions to Implement:
        {json.dumps(solutions_summary, indent=2)}
        
        Based on this information, generate a list of specific text changes that should be made to implement the solutions.
        For each change, provide:
        1. The original text to be replaced
        2. The new text to replace it with
        3. The reason for the change
        4. The approximate line number (can be an estimate)
        
        Focus on substantive changes that address the revision requirements. Include:
        - Changes to improve methodology description
        - Changes to address limitations
        - Changes to improve the presentation of results
        - Changes to improve the structure and flow
        - Changes to address specific reviewer concerns
        
        Generate at least 5 but no more than 15 specific changes.
        
        IMPORTANT: Format the response as a valid JSON array of objects with "old_text", "new_text", "reason", and "line_number" fields. Do not include any explanatory text before or after the JSON. The response should begin with '[' and end with ']'.
        
        Example format:
        [
          {{
            "old_text": "original text here",
            "new_text": "revised text here",
            "reason": "Improve clarity of methodology",
            "line_number": 42
          }},
          ...
        ]
        """
        
        # Use the LLM to generate specific changes
        changes_json = self._get_completion(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPTS["changes_generation"],
            max_tokens=3000 if self.context.optimize_costs else 4000
        )
        
        try:
            # Parse the LLM response
            changes_data = parse_json_safely(changes_json)
            
            # Convert to list of tuples
            changes = [(
                item.get("old_text", ""),
                item.get("new_text", ""),
                item.get("reason", ""),
                item.get("line_number")
            ) for item in changes_data]
            
            self.logger.info(f"Generated {len(changes)} specific text changes")
            return changes
            
        except ValueError as e:
            # Fallback if LLM didn't return valid JSON
            self.logger.warning(f"LLM didn't return valid JSON for changes. Error: {e}. Using basic changes.")
            changes = []
            
            # Create basic changes based on sections
            for section_name, content in list(section_samples.items())[:3]:  # Use first 3 sections
                if len(content) > 100:
                    changes.append((
                        content[:100],
                        f"[REVISED] {content[:100]}",
                        f"Improve {section_name} section based on reviewer comments",
                        None
                    ))
            
            self.logger.info(f"Created {len(changes)} fallback text changes")
            return changes
    
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