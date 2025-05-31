"""
Interface definitions for reference management components.

This module defines abstract base classes that serve as interfaces for the
reference management components, ensuring consistent design patterns across the codebase.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Set

from src.core.context import RevisionContext


class ReferenceManagerInterface(ABC):
    """
    Base interface for reference management components.
    
    Reference managers are responsible for validating, updating, and managing
    bibliographic references in academic papers.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the reference manager with a revision context.
        
        Args:
            context: The shared revision context
        """
        self.context = context
    
    @abstractmethod
    def validate_and_update_references(
        self,
        paper_analysis: Dict[str, Any],
        reviewer_comments: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Validate existing references and add new ones based on reviewer suggestions.
        
        Args:
            paper_analysis: Analysis of the original paper
            reviewer_comments: Analysis of reviewer comments
            output_path: Path where the new BibTeX file should be saved (optional)
            
        Returns:
            List of new references added
        """
        pass
    
    @abstractmethod
    def validate_references(self, references: List[str]) -> Tuple[Set[str], Set[str]]:
        """
        Validate a list of references and return valid and invalid ones.
        
        Args:
            references: List of reference identifiers to validate
            
        Returns:
            Tuple of (valid_references, invalid_references) as sets of strings
        """
        pass
    
    @abstractmethod
    def add_reference(self, reference_data: Dict[str, str]) -> str:
        """
        Add a new reference to the database.
        
        Args:
            reference_data: Dictionary with reference data
            
        Returns:
            Reference identifier
        """
        pass
    
    @abstractmethod
    def export_references(self, references: Set[str], output_path: str) -> str:
        """
        Export references to a file.
        
        Args:
            references: Set of reference identifiers to export
            output_path: Path where the references should be saved
            
        Returns:
            Path to the created file
        """
        pass