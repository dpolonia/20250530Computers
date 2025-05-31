"""
Paper revision tool implementation.

This module provides the main PaperRevisionTool class, which coordinates the paper
revision process using the service layer.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING

from src.utils.path_utils import (
    ensure_directory_exists, get_current_timestamp, construct_output_path
)

from src.core.context import RevisionContext
from src.domain.paper import Paper
from src.domain.reviewer_comment import ReviewerComment
from src.domain.issue import Issue
from src.domain.solution import Solution
from src.domain.change import Change
from src.domain.reference import Reference
from src.services.factory import ServiceFactory

# Import types only when type checking to avoid circular imports
if TYPE_CHECKING:
    from src.config import AppConfig


class PaperRevisionTool:
    """
    Main class for the paper revision tool.
    
    This class coordinates the paper revision process, using the service layer
    to perform the various tasks involved in revising a paper based on reviewer
    comments.
    """
    
    def __init__(
        self,
        original_paper_path: str,
        reviewer_comment_files: List[str],
        editor_letter_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        provider: str = "anthropic",
        model_name: str = "claude-3-opus-20240229",
        optimize_costs: bool = False,
        verbose: bool = False,
        config: Optional['AppConfig'] = None
    ):
        """
        Initialize the paper revision tool.
        
        Args:
            original_paper_path: Path to the original paper PDF
            reviewer_comment_files: List of paths to reviewer comment files
            editor_letter_path: Path to editor letter file (optional)
            output_dir: Directory for output files (optional)
            provider: LLM provider (default: "anthropic")
            model_name: LLM model name (default: "claude-3-opus-20240229")
            optimize_costs: Whether to optimize for lower costs (default: False)
            verbose: Whether to enable verbose logging (default: False)
            config: Optional AppConfig instance (overrides other parameters if provided)
        """
        # Import AppConfig type only when needed to avoid circular imports
        if config is not None:
            from src.config import AppConfig
            self.config = config
        else:
            self.config = None
            
        # Set up logging
        log_level = logging.INFO if not verbose else logging.DEBUG
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("PaperRevisionTool")
        
        # Set up context
        if self.config is not None:
            # Create context from config
            self.context = RevisionContext.from_config(self.config, logger=self.logger)
        else:
            # Create context from individual parameters
            self.context = RevisionContext(
                original_paper_path=original_paper_path,
                reviewer_comment_files=reviewer_comment_files,
                editor_letter_path=editor_letter_path,
                output_dir=output_dir,
                provider=provider,
                model_name=model_name,
                optimize_costs=optimize_costs,
                logger=self.logger
            )
        
        # Create service factory
        self.service_factory = ServiceFactory(self.context)
        
        # Initialize state
        self.paper = None
        self.reviewer_comments = []
        self.editor_requirements = {}
        self.issues = []
        self.solutions = []
        self.changes = []
        self.new_references = []
        
        self.logger.info("PaperRevisionTool initialized")
    
    def analyze_paper(self) -> Paper:
        """
        Analyze the paper and extract structured information.
        
        Returns:
            Paper domain entity
        """
        self.logger.info("Analyzing paper")
        
        paper_service = self.service_factory.get_paper_service()
        self.paper = paper_service.analyze_paper()
        
        # Store paper metadata in context
        self.context.paper_title = self.paper.title
        self.context.paper_authors = self.paper.authors
        
        return self.paper
    
    def analyze_reviewer_comments(self) -> List[ReviewerComment]:
        """
        Analyze reviewer comments and extract structured feedback.
        
        Returns:
            List of ReviewerComment domain entities
        """
        self.logger.info("Analyzing reviewer comments")
        
        reviewer_service = self.service_factory.get_reviewer_service()
        self.reviewer_comments = reviewer_service.analyze_reviewer_comments()
        
        return self.reviewer_comments
    
    def analyze_editor_requirements(self) -> Dict[str, Any]:
        """
        Process editor letter and requirements.
        
        Returns:
            Dictionary with editor requirements
        """
        self.logger.info("Analyzing editor requirements")
        
        reviewer_service = self.service_factory.get_reviewer_service()
        self.editor_requirements = reviewer_service.analyze_editor_requirements()
        
        return self.editor_requirements
    
    def identify_issues(self) -> List[Issue]:
        """
        Identify issues and concerns from paper analysis and reviewer comments.
        
        Returns:
            List of Issue domain entities
        """
        self.logger.info("Identifying issues")
        
        # Ensure paper and reviewer comments are analyzed
        if not self.paper:
            self.analyze_paper()
        
        if not self.reviewer_comments:
            self.analyze_reviewer_comments()
        
        if not self.editor_requirements and self.context.editor_letter_path:
            self.analyze_editor_requirements()
        
        # Identify issues
        solution_service = self.service_factory.get_solution_service()
        self.issues = solution_service.identify_issues(
            paper=self.paper,
            reviewer_comments=self.reviewer_comments,
            editor_requirements=self.editor_requirements
        )
        
        return self.issues
    
    def generate_solutions(self) -> List[Solution]:
        """
        Generate solutions for identified issues.
        
        Returns:
            List of Solution domain entities
        """
        self.logger.info("Generating solutions")
        
        # Ensure issues are identified
        if not self.issues:
            self.identify_issues()
        
        # Generate solutions
        solution_service = self.service_factory.get_solution_service()
        self.solutions = solution_service.generate_solutions(
            paper=self.paper,
            issues=self.issues
        )
        
        return self.solutions
    
    def generate_specific_changes(self) -> List[Change]:
        """
        Generate specific text changes to implement solutions.
        
        Returns:
            List of Change domain entities
        """
        self.logger.info("Generating specific text changes")
        
        # Ensure solutions are generated
        if not self.solutions:
            self.generate_solutions()
        
        # Generate specific changes
        solution_service = self.service_factory.get_solution_service()
        self.changes = solution_service.generate_specific_changes(
            paper=self.paper,
            solutions=self.solutions
        )
        
        return self.changes
    
    def validate_and_update_references(self) -> List[Reference]:
        """
        Validate existing references and add new ones based on reviewer suggestions.
        
        Returns:
            List of new Reference domain entities
        """
        self.logger.info("Validating and updating references")
        
        # Ensure paper and reviewer comments are analyzed
        if not self.paper:
            self.analyze_paper()
        
        if not self.reviewer_comments:
            self.analyze_reviewer_comments()
        
        # Validate and update references
        reference_service = self.service_factory.get_reference_service()
        self.new_references = reference_service.validate_and_update_references(
            paper=self.paper,
            reviewer_comments=self.reviewer_comments
        )
        
        return self.new_references
    
    def create_changes_document(self, output_path: Optional[str] = None) -> str:
        """
        Create a document detailing all changes.
        
        Args:
            output_path: Path where the document should be saved (optional)
            
        Returns:
            Path to the created document
        """
        self.logger.info("Creating changes document")
        
        # Ensure changes are generated
        if not self.changes:
            self.generate_specific_changes()
        
        # Create changes document
        document_service = self.service_factory.get_document_service()
        return document_service.create_changes_document(
            changes=self.changes,
            output_path=output_path
        )
    
    def create_revised_paper(self, output_path: Optional[str] = None) -> str:
        """
        Create revised paper with track changes.
        
        Args:
            output_path: Path where the document should be saved (optional)
            
        Returns:
            Path to the created document
        """
        self.logger.info("Creating revised paper")
        
        # Ensure changes are generated
        if not self.changes:
            self.generate_specific_changes()
        
        # Create revised paper
        document_service = self.service_factory.get_document_service()
        return document_service.create_revised_paper(
            changes=self.changes,
            output_path=output_path
        )
    
    def create_assessment(self, output_path: Optional[str] = None) -> str:
        """
        Create assessment document.
        
        Args:
            output_path: Path where the document should be saved (optional)
            
        Returns:
            Path to the created document
        """
        self.logger.info("Creating assessment document")
        
        # Ensure paper and changes are generated
        if not self.paper:
            self.analyze_paper()
        
        if not self.changes:
            self.generate_specific_changes()
        
        # Create assessment document
        document_service = self.service_factory.get_document_service()
        return document_service.create_assessment(
            changes=self.changes,
            paper=self.paper,
            output_path=output_path
        )
    
    def create_editor_letter(self, output_path: Optional[str] = None) -> str:
        """
        Create letter to the editor.
        
        Args:
            output_path: Path where the document should be saved (optional)
            
        Returns:
            Path to the created document
        """
        self.logger.info("Creating letter to the editor")
        
        # Ensure reviewer comments and changes are generated
        if not self.reviewer_comments:
            self.analyze_reviewer_comments()
        
        if not self.changes:
            self.generate_specific_changes()
        
        # Create editor letter
        document_service = self.service_factory.get_document_service()
        return document_service.create_editor_letter(
            reviewer_comments=self.reviewer_comments,
            changes=self.changes,
            output_path=output_path
        )
    
    def run_full_process(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Run the full paper revision process.
        
        Args:
            output_dir: Directory for output files (optional)
            
        Returns:
            Dictionary with paths to created documents
        """
        self.logger.info("Running full paper revision process")
        
        # Override output directory if provided
        if output_dir:
            self.context.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if self.context.output_dir:
            ensure_directory_exists(self.context.output_dir)
        
        # Run all steps
        self.analyze_paper()
        self.analyze_reviewer_comments()
        self.analyze_editor_requirements()
        self.identify_issues()
        self.generate_solutions()
        self.generate_specific_changes()
        self.validate_and_update_references()
        
        # Generate output paths
        output_dir = self.context.output_dir or os.path.dirname(self.context.original_paper_path)
        
        changes_path = construct_output_path(
            "changes", output_dir, self.context.original_paper_path, ".docx"
        )
        revised_path = construct_output_path(
            "revised_paper", output_dir, self.context.original_paper_path, ".docx"
        )
        assessment_path = construct_output_path(
            "assessment", output_dir, self.context.original_paper_path, ".docx"
        )
        editor_letter_path = construct_output_path(
            "editor_letter", output_dir, self.context.original_paper_path, ".docx"
        )
        
        # Create documents
        created_docs = {}
        created_docs["changes"] = self.create_changes_document(changes_path)
        created_docs["revised_paper"] = self.create_revised_paper(revised_path)
        created_docs["assessment"] = self.create_assessment(assessment_path)
        created_docs["editor_letter"] = self.create_editor_letter(editor_letter_path)
        
        # Log process statistics
        self.logger.info("Paper revision process completed")
        self.logger.info(f"Total tokens used: {self.context.process_statistics.get('total_tokens', 0)}")
        self.logger.info(f"Total cost: ${self.context.process_statistics.get('total_cost', 0.0):.4f}")
        
        return created_docs
    
    def get_llm_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        task_type: str = "general"
    ) -> str:
        """
        Get a completion from the language model.
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: The system prompt to use
            max_tokens: Maximum number of tokens to generate
            task_type: The type of task (analysis, generation, editing, etc.)
            
        Returns:
            The LLM response text
        """
        llm_service = self.service_factory.get_llm_service()
        return llm_service.get_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            task_type=task_type
        )