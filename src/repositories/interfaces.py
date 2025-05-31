"""
Repository interfaces for the paper revision tool.

This module defines the abstract base classes for repositories, providing a
consistent interface for data access regardless of the underlying storage.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generic, TypeVar, Optional, Set, Tuple

# Type variable for repository generics
T = TypeVar('T')


class Repository(Generic[T], ABC):
    """
    Base interface for all repositories.
    
    Repositories are responsible for data access and persistence, providing
    a clean separation between domain logic and data storage.
    """
    
    @abstractmethod
    def get(self, id: str) -> Optional[T]:
        """
        Get an entity by ID.
        
        Args:
            id: Entity ID
            
        Returns:
            Entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_all(self) -> List[T]:
        """
        Get all entities.
        
        Returns:
            List of all entities
        """
        pass
    
    @abstractmethod
    def add(self, entity: T) -> T:
        """
        Add a new entity.
        
        Args:
            entity: Entity to add
            
        Returns:
            Added entity
        """
        pass
    
    @abstractmethod
    def update(self, entity: T) -> T:
        """
        Update an existing entity.
        
        Args:
            entity: Entity to update
            
        Returns:
            Updated entity
        """
        pass
    
    @abstractmethod
    def remove(self, id: str) -> bool:
        """
        Remove an entity by ID.
        
        Args:
            id: Entity ID
            
        Returns:
            True if removed, False otherwise
        """
        pass


class PaperRepositoryInterface(Repository[T], ABC):
    """Interface for paper repositories."""
    
    @abstractmethod
    def get_by_title(self, title: str) -> Optional[T]:
        """
        Get a paper by title.
        
        Args:
            title: Paper title
            
        Returns:
            Paper if found, None otherwise
        """
        pass
    
    @abstractmethod
    def extract_sections(self, paper_id: str) -> Dict[str, str]:
        """
        Extract sections from a paper.
        
        Args:
            paper_id: Paper ID
            
        Returns:
            Dictionary mapping section names to content
        """
        pass
    
    @abstractmethod
    def extract_tables(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        Extract tables from a paper.
        
        Args:
            paper_id: Paper ID
            
        Returns:
            List of tables
        """
        pass
    
    @abstractmethod
    def extract_figures(self, paper_id: str) -> List[Tuple[str, Optional[str]]]:
        """
        Extract figures from a paper.
        
        Args:
            paper_id: Paper ID
            
        Returns:
            List of tuples (caption, path)
        """
        pass
    
    @abstractmethod
    def extract_references(self, paper_id: str) -> List[str]:
        """
        Extract references from a paper.
        
        Args:
            paper_id: Paper ID
            
        Returns:
            List of reference strings
        """
        pass


class ReviewerRepositoryInterface(Repository[T], ABC):
    """Interface for reviewer repositories."""
    
    @abstractmethod
    def get_by_reviewer_number(self, reviewer_number: int) -> Optional[T]:
        """
        Get reviewer comments by reviewer number.
        
        Args:
            reviewer_number: Reviewer number
            
        Returns:
            Reviewer comments if found, None otherwise
        """
        pass
    
    @abstractmethod
    def extract_text(self, reviewer_id: str) -> str:
        """
        Extract text from reviewer comments.
        
        Args:
            reviewer_id: Reviewer ID
            
        Returns:
            Extracted text
        """
        pass


class ReferenceRepositoryInterface(Repository[T], ABC):
    """Interface for reference repositories."""
    
    @abstractmethod
    def validate_references(self, references: List[str]) -> Tuple[Set[str], Set[str]]:
        """
        Validate references.
        
        Args:
            references: List of reference IDs to validate
            
        Returns:
            Tuple of (valid_references, invalid_references)
        """
        pass
    
    @abstractmethod
    def export_references(self, references: Set[str], output_path: str) -> str:
        """
        Export references to a file.
        
        Args:
            references: Set of reference IDs to export
            output_path: Path where the references should be saved
            
        Returns:
            Path to the created file
        """
        pass


class DocumentRepositoryInterface(Repository[T], ABC):
    """Interface for document repositories."""
    
    @abstractmethod
    def create_changes_document(
        self,
        changes: List[Tuple[str, str, str, Optional[int]]],
        output_path: str
    ) -> str:
        """
        Create a document detailing changes.
        
        Args:
            changes: List of tuples (old_text, new_text, reason, line_number)
            output_path: Path where the document should be saved
            
        Returns:
            Path to the created document
        """
        pass
    
    @abstractmethod
    def create_revised_paper(
        self,
        changes: List[Tuple[str, str, str, Optional[int]]],
        output_path: str
    ) -> str:
        """
        Create revised paper with track changes.
        
        Args:
            changes: List of tuples (old_text, new_text, reason, line_number)
            output_path: Path where the document should be saved
            
        Returns:
            Path to the created document
        """
        pass
    
    @abstractmethod
    def create_assessment_document(
        self,
        assessment: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Create an assessment document.
        
        Args:
            assessment: Assessment data
            output_path: Path where the document should be saved
            
        Returns:
            Path to the created document
        """
        pass
    
    @abstractmethod
    def create_editor_letter(
        self,
        reviewer_responses: List[Dict[str, Any]],
        output_path: str,
        process_summary: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a letter to the editor.
        
        Args:
            reviewer_responses: Reviewer responses data
            output_path: Path where the document should be saved
            process_summary: Optional process summary data
            
        Returns:
            Path to the created document
        """
        pass


class ChangeRepositoryInterface(Repository[T], ABC):
    """Interface for change repositories."""
    
    @abstractmethod
    def get_by_section(self, section: str) -> List[T]:
        """
        Get changes by section.
        
        Args:
            section: Section name
            
        Returns:
            List of changes for the section
        """
        pass
    
    @abstractmethod
    def get_by_reason(self, reason: str) -> List[T]:
        """
        Get changes by reason.
        
        Args:
            reason: Change reason
            
        Returns:
            List of changes with the given reason
        """
        pass