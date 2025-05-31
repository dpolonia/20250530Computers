"""
Solution service implementation.

This module implements the solution service interface, providing functionality for
identifying issues and generating solutions.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from src.core.context import RevisionContext
from src.domain.paper import Paper
from src.domain.reviewer_comment import ReviewerComment
from src.domain.issue import Issue, SeverityLevel, IssueType
from src.domain.solution import Solution, ComplexityLevel
from src.domain.change import Change
from src.services.interfaces import SolutionServiceInterface
from src.core.constants import SYSTEM_PROMPTS
from src.core.json_utils import parse_json_safely


class SolutionService(SolutionServiceInterface):
    """
    Service for solution generation.
    
    This service is responsible for identifying issues in the paper based on
    reviewer comments and generating solutions to address them.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the solution service.
        
        Args:
            context: The shared revision context
        """
        self.context = context
        self.logger = context.logger or logging.getLogger(__name__)
        self.issues = []
        self.solutions = []
        self.changes = []
    
    def identify_issues(
        self, 
        paper: Paper,
        reviewer_comments: List[ReviewerComment],
        editor_requirements: Dict[str, Any]
    ) -> List[Issue]:
        """
        Identify issues and concerns from paper analysis and reviewer comments.
        
        Args:
            paper: Paper domain entity
            reviewer_comments: List of ReviewerComment domain entities
            editor_requirements: Dictionary with editor requirements
            
        Returns:
            List of Issue domain entities
        """
        self.logger.info("Identifying issues from reviewer comments and paper analysis")
        
        issues = []
        
        # 1. First, process high-priority issues from the editor
        if editor_requirements:
            editor_issues = self._process_editor_issues(editor_requirements)
            issues.extend(editor_issues)
        
        # 2. Process issues from reviewer comments
        reviewer_issues = self._process_reviewer_issues(reviewer_comments)
        issues.extend(reviewer_issues)
        
        # 3. Analyze the paper itself for additional issues
        paper_issues = self._analyze_paper_issues(paper)
        issues.extend(paper_issues)
        
        # 4. Deduplicate and merge similar issues
        unique_issues = self._deduplicate_issues(issues)
        
        self.issues = unique_issues
        self.logger.info(f"Identified {len(unique_issues)} unique issues")
        
        return unique_issues
    
    def generate_solutions(
        self,
        paper: Paper,
        issues: List[Issue]
    ) -> List[Solution]:
        """
        Generate solutions for identified issues.
        
        Args:
            paper: Paper domain entity
            issues: List of Issue domain entities
            
        Returns:
            List of Solution domain entities
        """
        self.logger.info("Generating solutions for identified issues")
        
        if not issues:
            self.logger.warning("No issues to generate solutions for")
            return []
        
        # Group related issues for more coherent solutions
        issue_groups = self._group_related_issues(issues)
        
        solutions = []
        for issue_group in issue_groups:
            try:
                # Generate solution for this group of issues
                solution = self._generate_solution_for_issues(paper, issue_group)
                solutions.append(solution)
            except Exception as e:
                self.logger.error(f"Error generating solution for issue group: {e}")
        
        self.solutions = solutions
        self.logger.info(f"Generated {len(solutions)} solutions")
        
        return solutions
    
    def generate_specific_changes(
        self,
        paper: Paper,
        solutions: List[Solution]
    ) -> List[Change]:
        """
        Generate specific text changes to implement solutions.
        
        Args:
            paper: Paper domain entity
            solutions: List of Solution domain entities
            
        Returns:
            List of Change domain entities
        """
        self.logger.info("Generating specific text changes for solutions")
        
        if not solutions:
            self.logger.warning("No solutions to generate changes for")
            return []
        
        changes = []
        for solution in solutions:
            try:
                # Generate changes for this solution
                solution_changes = self._generate_changes_for_solution(paper, solution)
                changes.extend(solution_changes)
            except Exception as e:
                self.logger.error(f"Error generating changes for solution '{solution.title}': {e}")
        
        self.changes = changes
        self.logger.info(f"Generated {len(changes)} specific text changes")
        
        return changes
    
    def _process_editor_issues(self, editor_requirements: Dict[str, Any]) -> List[Issue]:
        """
        Process editor requirements into issues.
        
        Args:
            editor_requirements: Dictionary with editor requirements
            
        Returns:
            List of Issue domain entities
        """
        issues = []
        
        # Extract key requirements
        key_requirements = editor_requirements.get("key_requirements", [])
        for req in key_requirements:
            # Determine issue type based on text content
            issue_type = IssueType.OTHER
            if any(term in req.lower() for term in ["method", "approach", "experiment", "analysis", "data"]):
                issue_type = IssueType.METHODOLOGY
            elif any(term in req.lower() for term in ["result", "finding", "outcome", "conclusion"]):
                issue_type = IssueType.RESULTS
            elif any(term in req.lower() for term in ["write", "clarity", "structure", "organization", "explain", "describe"]):
                issue_type = IssueType.WRITING
            elif any(term in req.lower() for term in ["reference", "citation", "literature", "prior work"]):
                issue_type = IssueType.REFERENCES
            
            # Add as high severity issue since it comes from the editor
            issues.append(Issue(
                description=req,
                severity=SeverityLevel.HIGH,
                type=issue_type,
                source="Editor"
            ))
        
        # Extract reviewer feedback to address
        reviewer_feedback = editor_requirements.get("reviewer_feedback_to_address", [])
        for feedback in reviewer_feedback:
            issues.append(Issue(
                description=feedback,
                severity=SeverityLevel.HIGH,
                type=IssueType.OTHER,
                source="Editor (Reviewer Feedback)"
            ))
        
        return issues
    
    def _process_reviewer_issues(self, reviewer_comments: List[ReviewerComment]) -> List[Issue]:
        """
        Process reviewer comments into issues.
        
        Args:
            reviewer_comments: List of ReviewerComment domain entities
            
        Returns:
            List of Issue domain entities
        """
        issues = []
        
        for reviewer in reviewer_comments:
            # Process main concerns as high severity issues
            for concern in reviewer.main_concerns:
                # Determine issue type based on text content
                issue_type = IssueType.OTHER
                if any(term in concern.lower() for term in ["method", "approach", "experiment", "analysis", "data"]):
                    issue_type = IssueType.METHODOLOGY
                elif any(term in concern.lower() for term in ["result", "finding", "outcome", "conclusion"]):
                    issue_type = IssueType.RESULTS
                elif any(term in concern.lower() for term in ["write", "clarity", "structure", "organization", "explain", "describe"]):
                    issue_type = IssueType.WRITING
                elif any(term in concern.lower() for term in ["reference", "citation", "literature", "prior work"]):
                    issue_type = IssueType.REFERENCES
                
                issues.append(Issue(
                    description=concern,
                    severity=SeverityLevel.HIGH,
                    type=issue_type,
                    source=f"Reviewer {reviewer.reviewer_number}",
                    reviewer_number=reviewer.reviewer_number
                ))
            
            # Process required changes as medium severity issues
            for change in reviewer.required_changes:
                # Determine issue type based on text content
                issue_type = IssueType.OTHER
                if any(term in change.lower() for term in ["method", "approach", "experiment", "analysis", "data"]):
                    issue_type = IssueType.METHODOLOGY
                elif any(term in change.lower() for term in ["result", "finding", "outcome", "conclusion"]):
                    issue_type = IssueType.RESULTS
                elif any(term in change.lower() for term in ["write", "clarity", "structure", "organization", "explain", "describe"]):
                    issue_type = IssueType.WRITING
                elif any(term in change.lower() for term in ["reference", "citation", "literature", "prior work"]):
                    issue_type = IssueType.REFERENCES
                
                issues.append(Issue(
                    description=change,
                    severity=SeverityLevel.MEDIUM,
                    type=issue_type,
                    source=f"Reviewer {reviewer.reviewer_number}",
                    reviewer_number=reviewer.reviewer_number
                ))
            
            # Process specific comment types
            for comment in reviewer.methodology_comments:
                issues.append(Issue(
                    description=comment,
                    severity=SeverityLevel.MEDIUM,
                    type=IssueType.METHODOLOGY,
                    source=f"Reviewer {reviewer.reviewer_number}",
                    reviewer_number=reviewer.reviewer_number
                ))
            
            for comment in reviewer.results_comments:
                issues.append(Issue(
                    description=comment,
                    severity=SeverityLevel.MEDIUM,
                    type=IssueType.RESULTS,
                    source=f"Reviewer {reviewer.reviewer_number}",
                    reviewer_number=reviewer.reviewer_number
                ))
            
            for comment in reviewer.writing_comments:
                issues.append(Issue(
                    description=comment,
                    severity=SeverityLevel.MEDIUM,
                    type=IssueType.WRITING,
                    source=f"Reviewer {reviewer.reviewer_number}",
                    reviewer_number=reviewer.reviewer_number
                ))
            
            for comment in reviewer.references_comments:
                issues.append(Issue(
                    description=comment,
                    severity=SeverityLevel.MEDIUM,
                    type=IssueType.REFERENCES,
                    source=f"Reviewer {reviewer.reviewer_number}",
                    reviewer_number=reviewer.reviewer_number
                ))
            
            # Process suggested changes as low severity issues
            for suggestion in reviewer.suggested_changes:
                # Determine issue type based on text content
                issue_type = IssueType.OTHER
                if any(term in suggestion.lower() for term in ["method", "approach", "experiment", "analysis", "data"]):
                    issue_type = IssueType.METHODOLOGY
                elif any(term in suggestion.lower() for term in ["result", "finding", "outcome", "conclusion"]):
                    issue_type = IssueType.RESULTS
                elif any(term in suggestion.lower() for term in ["write", "clarity", "structure", "organization", "explain", "describe"]):
                    issue_type = IssueType.WRITING
                elif any(term in suggestion.lower() for term in ["reference", "citation", "literature", "prior work"]):
                    issue_type = IssueType.REFERENCES
                
                issues.append(Issue(
                    description=suggestion,
                    severity=SeverityLevel.LOW,
                    type=issue_type,
                    source=f"Reviewer {reviewer.reviewer_number}",
                    reviewer_number=reviewer.reviewer_number
                ))
        
        return issues
    
    def _analyze_paper_issues(self, paper: Paper) -> List[Issue]:
        """
        Analyze the paper itself for additional issues.
        
        Args:
            paper: Paper domain entity
            
        Returns:
            List of Issue domain entities
        """
        issues = []
        
        # Create prompt for paper issue analysis
        sections_preview = "\n".join([f"- {section.name}" for section in paper.sections[:10]])
        references_preview = f"{len(paper.references)} references"
        
        prompt = f"""
        I'm analyzing a scientific paper to identify potential issues that might need to be addressed. The paper has the following properties:

        Title: {paper.title}
        Abstract: {paper.abstract[:200]}...
        Sections: {sections_preview}
        References: {references_preview}
        
        Based on this information, identify potential issues with the paper that might need to be addressed. Consider:
        1. Structure and organization issues
        2. Potential clarity or writing issues
        3. Methodology issues that might exist
        4. Results presentation issues
        5. Reference and citation issues
        
        IMPORTANT: Format the response as a valid JSON array of issue objects. Do not include any explanatory text before or after the JSON. The response should begin with '[' and end with ']'.
        
        Example format:
        [
          {{
            "description": "The methodology section lacks detail on sample selection",
            "type": "methodology",
            "severity": "medium"
          }},
          {{
            "description": "The abstract is too long and lacks focus",
            "type": "writing",
            "severity": "low"
          }}
        ]
        """
        
        try:
            # Get analysis from LLM
            analysis_json = self._get_completion(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPTS.get("paper_analysis", "You are an expert academic reviewer"),
                max_tokens=1500 if self.context.optimize_costs else 2000
            )
            
            # Parse the LLM response
            paper_issues = parse_json_safely(analysis_json)
            
            # Convert to Issue domain entities
            for issue_data in paper_issues:
                # Map type string to enum
                type_str = issue_data.get("type", "other").lower()
                if "method" in type_str:
                    issue_type = IssueType.METHODOLOGY
                elif "result" in type_str:
                    issue_type = IssueType.RESULTS
                elif "writ" in type_str or "structure" in type_str:
                    issue_type = IssueType.WRITING
                elif "reference" in type_str or "citation" in type_str:
                    issue_type = IssueType.REFERENCES
                else:
                    issue_type = IssueType.OTHER
                
                # Map severity string to enum
                severity_str = issue_data.get("severity", "medium").lower()
                if "high" in severity_str:
                    severity = SeverityLevel.HIGH
                elif "low" in severity_str:
                    severity = SeverityLevel.LOW
                else:
                    severity = SeverityLevel.MEDIUM
                
                issues.append(Issue(
                    description=issue_data.get("description", ""),
                    severity=severity,
                    type=issue_type,
                    source="Paper Analysis"
                ))
        
        except Exception as e:
            self.logger.error(f"Error analyzing paper for issues: {e}")
        
        return issues
    
    def _deduplicate_issues(self, issues: List[Issue]) -> List[Issue]:
        """
        Deduplicate and merge similar issues.
        
        Args:
            issues: List of Issue domain entities
            
        Returns:
            List of deduplicated Issue domain entities
        """
        if not issues:
            return []
        
        # Group similar issues by text similarity
        issue_groups = []
        for issue in issues:
            # Check if this issue is similar to any existing group
            found_match = False
            for group in issue_groups:
                # Simple text similarity check (could be improved with embeddings or more sophisticated NLP)
                for existing_issue in group:
                    if self._text_similarity(issue.description, existing_issue.description) > 0.7:
                        group.append(issue)
                        found_match = True
                        break
                
                if found_match:
                    break
            
            # If no match found, create a new group
            if not found_match:
                issue_groups.append([issue])
        
        # Merge issues in each group
        merged_issues = []
        for group in issue_groups:
            if len(group) == 1:
                # No need to merge, just add the single issue
                merged_issues.append(group[0])
            else:
                # Merge similar issues
                merged_issue = self._merge_issues(group)
                merged_issues.append(merged_issue)
        
        return merged_issues
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity between two strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Simple word overlap similarity (could be improved with embeddings)
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _merge_issues(self, issues: List[Issue]) -> Issue:
        """
        Merge similar issues into a single issue.
        
        Args:
            issues: List of similar Issue domain entities
            
        Returns:
            Merged Issue domain entity
        """
        # Get the most severe issue to use as a base
        base_issue = max(issues, key=lambda x: {
            SeverityLevel.HIGH: 3,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.LOW: 1
        }[x.severity])
        
        # Combine descriptions
        descriptions = [issue.description for issue in issues]
        unique_descriptions = []
        for desc in descriptions:
            if not any(self._text_similarity(desc, existing) > 0.8 for existing in unique_descriptions):
                unique_descriptions.append(desc)
        
        # Build a more comprehensive description
        if len(unique_descriptions) == 1:
            merged_description = unique_descriptions[0]
        else:
            sources = [issue.source for issue in issues]
            unique_sources = list(set(sources))
            
            merged_description = f"{base_issue.description} (Identified by {', '.join(unique_sources)})"
        
        # Create the merged issue
        return Issue(
            description=merged_description,
            severity=base_issue.severity,
            type=base_issue.type,
            source=", ".join(set(issue.source for issue in issues)),
            reviewer_number=base_issue.reviewer_number,
            section=base_issue.section,
            line_number=base_issue.line_number
        )
    
    def _group_related_issues(self, issues: List[Issue]) -> List[List[Issue]]:
        """
        Group related issues for more coherent solutions.
        
        Args:
            issues: List of Issue domain entities
            
        Returns:
            List of lists of related Issue domain entities
        """
        # Group by type first
        type_groups = {}
        for issue in issues:
            if issue.type not in type_groups:
                type_groups[issue.type] = []
            type_groups[issue.type].append(issue)
        
        # For each type group, further group by similarity
        result_groups = []
        for type_issues in type_groups.values():
            # Skip if only one issue of this type
            if len(type_issues) == 1:
                result_groups.append(type_issues)
                continue
            
            # Group by similarity within this type
            subgroups = []
            for issue in type_issues:
                # Check if this issue is similar to any existing subgroup
                found_match = False
                for group in subgroups:
                    # Check similarity with first issue in group (representative)
                    if self._text_similarity(issue.description, group[0].description) > 0.5:
                        group.append(issue)
                        found_match = True
                        break
                
                # If no match found, create a new subgroup
                if not found_match:
                    subgroups.append([issue])
            
            # Add all subgroups to result
            result_groups.extend(subgroups)
        
        return result_groups
    
    def _generate_solution_for_issues(self, paper: Paper, issues: List[Issue]) -> Solution:
        """
        Generate a solution for a group of related issues.
        
        Args:
            paper: Paper domain entity
            issues: List of related Issue domain entities
            
        Returns:
            Solution domain entity
        """
        # Extract relevant paper sections based on issue type
        relevant_sections = []
        issue_type = issues[0].type  # Use the type of the first issue in the group
        
        for section in paper.sections:
            section_name = section.name.lower()
            
            if issue_type == IssueType.METHODOLOGY and any(term in section_name for term in ["method", "approach", "experiment", "data"]):
                relevant_sections.append(f"{section.name}: {section.content[:500]}...")
            elif issue_type == IssueType.RESULTS and any(term in section_name for term in ["result", "finding", "discussion", "analysis"]):
                relevant_sections.append(f"{section.name}: {section.content[:500]}...")
            elif issue_type == IssueType.WRITING and any(term in section_name for term in ["introduction", "background", "conclusion"]):
                relevant_sections.append(f"{section.name}: {section.content[:500]}...")
            elif issue_type == IssueType.REFERENCES and "reference" in section_name:
                relevant_sections.append(f"{section.name}: {section.content[:500]}...")
        
        # Prepare issue descriptions
        issue_descriptions = []
        for i, issue in enumerate(issues, 1):
            source = f" (Source: {issue.source})" if issue.source else ""
            issue_descriptions.append(f"Issue {i}: {issue.description}{source}")
        
        # Create prompt for solution generation
        prompt = f"""
        I'm generating solutions for issues identified in a scientific paper. Generate a comprehensive solution to address the following issues:
        
        ISSUES TO ADDRESS:
        {chr(10).join(issue_descriptions)}
        
        PAPER TITLE: {paper.title}
        
        RELEVANT SECTIONS:
        {chr(10).join(relevant_sections[:3])}
        
        Based on these issues and paper context, generate a detailed solution that addresses all the identified issues. Include:
        1. A concise title for the solution
        2. Detailed implementation steps
        3. The expected impact of implementing this solution
        4. The complexity level of implementing this solution (low, medium, high)
        
        IMPORTANT: Format the response as a valid JSON object with these fields. Do not include any explanatory text before or after the JSON. The response should begin with '{{' and end with '}}'.
        
        Example format:
        {{
          "title": "Improve methodology description with detailed protocol",
          "implementation": "1. Add a detailed step-by-step protocol...\n2. Include equipment specifications...\n3. Add a methodology flowchart...",
          "impact": "This will address the concerns about methodology transparency and reproducibility...",
          "complexity": "medium"
        }}
        """
        
        # Get solution from LLM
        solution_json = self._get_completion(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPTS.get("solution_generation", "You are an expert academic researcher"),
            max_tokens=2000 if self.context.optimize_costs else 3000
        )
        
        # Parse the LLM response
        solution_data = parse_json_safely(solution_json)
        
        # Map complexity string to enum
        complexity_str = solution_data.get("complexity", "medium").lower()
        if "high" in complexity_str:
            complexity = ComplexityLevel.HIGH
        elif "low" in complexity_str:
            complexity = ComplexityLevel.LOW
        else:
            complexity = ComplexityLevel.MEDIUM
        
        # Create the Solution domain entity
        return Solution(
            title=solution_data.get("title", f"Solution for {issue_type.value} issues"),
            implementation=solution_data.get("implementation", ""),
            complexity=complexity,
            impact=solution_data.get("impact", ""),
            addresses=[issue.description for issue in issues],
            issues=issues
        )
    
    def _generate_changes_for_solution(self, paper: Paper, solution: Solution) -> List[Change]:
        """
        Generate specific text changes to implement a solution.
        
        Args:
            paper: Paper domain entity
            solution: Solution domain entity
            
        Returns:
            List of Change domain entities
        """
        changes = []
        
        # Determine which sections are likely to be affected by this solution
        affected_sections = []
        primary_issue_type = solution.issues[0].type if solution.issues else None
        
        for section in paper.sections:
            section_name = section.name.lower()
            
            if primary_issue_type == IssueType.METHODOLOGY and any(term in section_name for term in ["method", "approach", "experiment", "data"]):
                affected_sections.append(section)
            elif primary_issue_type == IssueType.RESULTS and any(term in section_name for term in ["result", "finding", "discussion", "analysis"]):
                affected_sections.append(section)
            elif primary_issue_type == IssueType.WRITING and any(term in section_name for term in ["introduction", "background", "conclusion"]):
                affected_sections.append(section)
            elif primary_issue_type == IssueType.REFERENCES and "reference" in section_name:
                affected_sections.append(section)
            elif primary_issue_type == IssueType.STRUCTURE:
                # For structure issues, consider most sections
                affected_sections.append(section)
        
        # If no specifically affected sections found, default to all sections
        if not affected_sections:
            affected_sections = paper.sections
        
        # Limit to 3 most relevant sections to keep token usage reasonable
        affected_sections = affected_sections[:3]
        
        # Create prompt for generating specific changes
        sections_content = []
        for section in affected_sections:
            sections_content.append(f"SECTION: {section.name}\n{section.content}")
        
        # Extract issue descriptions for context
        issue_descriptions = [issue.description for issue in solution.issues]
        
        prompt = f"""
        I'm generating specific text changes to implement a solution in a scientific paper. Generate concrete text changes based on the solution and affected sections.
        
        SOLUTION:
        Title: {solution.title}
        Implementation: {solution.implementation}
        
        ISSUES ADDRESSED:
        {chr(10).join(issue_descriptions)}
        
        AFFECTED SECTIONS:
        {chr(10).join(sections_content)}
        
        Based on this solution and the current text, generate specific text changes that would implement the solution. For each change, specify:
        1. The exact text to be replaced (old_text)
        2. The new text to replace it with (new_text)
        3. The reason for this specific change
        
        IMPORTANT: Format the response as a valid JSON array of change objects. Do not include any explanatory text before or after the JSON. The response should begin with '[' and end with ']'.
        
        Example format:
        [
          {{
            "old_text": "We collected samples from 30 participants.",
            "new_text": "We collected samples from 30 participants (15 male, 15 female) using stratified random sampling to ensure representativeness.",
            "reason": "Adding detail about the sampling strategy to address reviewer concerns about methodology",
            "section": "Methods"
          }},
          {{
            "old_text": "",
            "new_text": "Figure 2 illustrates the experimental setup, showing the key components and their arrangement.",
            "reason": "Adding a new figure reference to improve clarity of the experimental setup",
            "section": "Methods"
          }}
        ]
        """
        
        # Get changes from LLM
        changes_json = self._get_completion(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPTS.get("change_generation", "You are an expert academic editor"),
            max_tokens=3000 if self.context.optimize_costs else 5000
        )
        
        # Parse the LLM response
        try:
            changes_data = parse_json_safely(changes_json)
            
            # Convert to Change domain entities
            for change_data in changes_data:
                changes.append(Change(
                    old_text=change_data.get("old_text", ""),
                    new_text=change_data.get("new_text", ""),
                    reason=change_data.get("reason", ""),
                    section=change_data.get("section", affected_sections[0].name if affected_sections else None),
                    solution=solution
                ))
        
        except Exception as e:
            self.logger.error(f"Error generating changes for solution '{solution.title}': {e}")
            
            # Add a fallback change if parsing fails
            changes.append(Change(
                old_text="",
                new_text=f"[Implement {solution.title}]",
                reason=f"Implementation of solution: {solution.title}",
                section=affected_sections[0].name if affected_sections else None,
                solution=solution
            ))
        
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