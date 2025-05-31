"""
Service interfaces for the paper revision tool.

This module defines the abstract base classes for services, providing a
consistent interface for business logic operations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

from src.domain.paper import Paper
from src.domain.reviewer_comment import ReviewerComment
from src.domain.issue import Issue
from src.domain.solution import Solution
from src.domain.change import Change
from src.domain.reference import Reference
from src.domain.assessment import Assessment
from src.core.context import RevisionContext


class ServiceInterface(ABC):
    """
    Base interface for all services.
    
    Services are responsible for coordinating business logic operations and
    providing a clean API for higher-level components.
    """
    
    @abstractmethod
    def __init__(self, context: RevisionContext):
        """
        Initialize the service with a revision context.
        
        Args:
            context: The shared revision context
        """
        pass


class PaperServiceInterface(ServiceInterface):
    """Interface for paper services."""
    
    @abstractmethod
    def analyze_paper(self) -> Paper:
        """
        Analyze the paper and extract structured information.
        
        Returns:
            Paper domain entity
        """
        pass
    
    @abstractmethod
    def get_section_by_name(self, name: str) -> Optional[str]:
        """
        Get a section by name.
        
        Args:
            name: The name of the section to find
            
        Returns:
            The section content if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_paper_metadata(self) -> Dict[str, Any]:
        """
        Get paper metadata.
        
        Returns:
            Dictionary with paper metadata
        """
        pass


class ReviewerServiceInterface(ServiceInterface):
    """Interface for reviewer services."""
    
    @abstractmethod
    def analyze_reviewer_comments(self) -> List[ReviewerComment]:
        """
        Analyze reviewer comments and extract structured feedback.
        
        Returns:
            List of ReviewerComment domain entities
        """
        pass
    
    @abstractmethod
    def analyze_editor_requirements(self) -> Dict[str, Any]:
        """
        Process editor letter and requirements.
        
        Returns:
            Dictionary with editor requirements
        """
        pass


class SolutionServiceInterface(ServiceInterface):
    """Interface for solution services."""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass


class DocumentServiceInterface(ServiceInterface):
    """Interface for document services."""
    
    @abstractmethod
    def create_changes_document(
        self,
        changes: List[Change],
        output_path: Optional[str] = None
    ) -> str:
        """
        Create a document detailing all changes.
        
        Args:
            changes: List of Change domain entities
            output_path: Path where the document should be saved (optional)
            
        Returns:
            Path to the created document
        """
        pass
    
    @abstractmethod
    def create_revised_paper(
        self,
        changes: List[Change],
        output_path: Optional[str] = None
    ) -> str:
        """
        Create revised paper with track changes.
        
        Args:
            changes: List of Change domain entities
            output_path: Path where the document should be saved (optional)
            
        Returns:
            Path to the created document
        """
        pass
    
    @abstractmethod
    def create_assessment(
        self,
        changes: List[Change],
        paper: Paper,
        output_path: Optional[str] = None
    ) -> str:
        """
        Create assessment document.
        
        Args:
            changes: List of Change domain entities
            paper: Paper domain entity
            output_path: Path where the document should be saved (optional)
            
        Returns:
            Path to the created document
        """
        pass
    
    @abstractmethod
    def create_editor_letter(
        self,
        reviewer_comments: List[ReviewerComment],
        changes: List[Change],
        output_path: Optional[str] = None
    ) -> str:
        """
        Create letter to the editor.
        
        Args:
            reviewer_comments: List of ReviewerComment domain entities
            changes: List of Change domain entities
            output_path: Path where the document should be saved (optional)
            
        Returns:
            Path to the created document
        """
        pass


class ReferenceServiceInterface(ServiceInterface):
    """Interface for reference services."""
    
    @abstractmethod
    def validate_and_update_references(
        self,
        paper: Paper,
        reviewer_comments: List[ReviewerComment],
        output_path: Optional[str] = None
    ) -> List[Reference]:
        """
        Validate existing references and add new ones based on reviewer suggestions.
        
        Args:
            paper: Paper domain entity
            reviewer_comments: List of ReviewerComment domain entities
            output_path: Path where the new BibTeX file should be saved (optional)
            
        Returns:
            List of new Reference domain entities
        """
        pass


class LLMServiceInterface(ServiceInterface):
    """Interface for language model services."""
    
    @abstractmethod
    def get_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        task_type: str = "general"
    ) -> str:
        """
        Get a completion from the language model with appropriate error handling.
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: The system prompt to use
            max_tokens: Maximum number of tokens to generate
            task_type: The type of task (analysis, generation, editing, etc.)
            
        Returns:
            The LLM response text
        """
        pass
    
    @abstractmethod
    def evaluate_response_quality(
        self, 
        prompt: str, 
        response: str, 
        task_type: str = "general",
        use_competitor: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a model response.
        
        Args:
            prompt: The original prompt sent to the model
            response: The model's response
            task_type: The type of task (analysis, generation, editing, etc.)
            use_competitor: Whether to use a competing model for evaluation
            
        Returns:
            Dictionary with quality metrics and feedback
        """
        pass