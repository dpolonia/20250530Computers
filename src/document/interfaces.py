"""
Interface definitions for document generation components.

This module defines abstract base classes that serve as interfaces for the
document generation components, ensuring consistent design patterns across the codebase.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

from src.core.context import RevisionContext


class DocumentGenerator(ABC):
    """
    Base interface for all document generator components.
    
    Document generators are responsible for creating various document outputs
    based on analysis results and suggested changes.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the document generator with a revision context.
        
        Args:
            context: The shared revision context
        """
        self.context = context
    
    @abstractmethod
    def generate(self, *args, **kwargs) -> str:
        """
        Generate a document and return its path.
        
        Returns:
            Path to the generated document
        """
        pass


class ChangesDocumentGeneratorInterface(DocumentGenerator):
    """Interface for changes document generation components."""
    
    @abstractmethod
    def create_changes_document(
        self,
        changes: List[Tuple[str, str, str, Optional[int]]],
        output_path: Optional[str] = None
    ) -> str:
        """
        Create a document detailing all changes.
        
        Args:
            changes: List of tuples (old_text, new_text, reason, line_number)
            output_path: Path where the document should be saved (optional)
            
        Returns:
            Path to the created document
        """
        pass


class RevisedPaperGeneratorInterface(DocumentGenerator):
    """Interface for revised paper generation components."""
    
    @abstractmethod
    def create_revised_paper(
        self,
        changes: List[Tuple[str, str, str, Optional[int]]],
        output_path: Optional[str] = None
    ) -> str:
        """
        Create revised paper with track changes.
        
        Args:
            changes: List of tuples (old_text, new_text, reason, line_number)
            output_path: Path where the document should be saved (optional)
            
        Returns:
            Path to the created document
        """
        pass


class AssessmentGeneratorInterface(DocumentGenerator):
    """Interface for assessment document generation components."""
    
    @abstractmethod
    def create_assessment(
        self,
        changes: List[Tuple[str, str, str, Optional[int]]],
        paper_analysis: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Create assessment document.
        
        Args:
            changes: List of tuples (old_text, new_text, reason, line_number)
            paper_analysis: Analysis of the original paper
            output_path: Path where the document should be saved (optional)
            
        Returns:
            Path to the created document
        """
        pass


class EditorLetterGeneratorInterface(DocumentGenerator):
    """Interface for editor letter generation components."""
    
    @abstractmethod
    def create_editor_letter(
        self,
        reviewer_comments: List[Dict[str, Any]],
        changes: List[Tuple[str, str, str, Optional[int]]],
        output_path: Optional[str] = None
    ) -> str:
        """
        Create letter to the editor.
        
        Args:
            reviewer_comments: Analysis of reviewer comments
            changes: List of tuples (old_text, new_text, reason, line_number)
            output_path: Path where the document should be saved (optional)
            
        Returns:
            Path to the created document
        """
        pass