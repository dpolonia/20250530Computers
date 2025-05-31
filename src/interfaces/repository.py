"""
Repository interfaces for the Paper Revision System.

This module defines interfaces for data access repositories,
enabling a clean separation between domain logic and data storage.
"""

import abc
from typing import Dict, List, Optional, Any, Union, TypeVar, Generic

# Type variable for entity types
T = TypeVar('T')


class RepositoryInterface(Generic[T], abc.ABC):
    """Generic interface for repositories."""
    
    @abc.abstractmethod
    def add(self, entity: T) -> T:
        """Add an entity to the repository.
        
        Args:
            entity: The entity to add
            
        Returns:
            The added entity
        """
        pass
    
    @abc.abstractmethod
    def get(self, id_: Any) -> Optional[T]:
        """Get an entity by ID.
        
        Args:
            id_: The entity ID
            
        Returns:
            The entity, or None if not found
        """
        pass
    
    @abc.abstractmethod
    def get_all(self) -> List[T]:
        """Get all entities.
        
        Returns:
            List of all entities
        """
        pass
    
    @abc.abstractmethod
    def update(self, entity: T) -> T:
        """Update an entity.
        
        Args:
            entity: The entity to update
            
        Returns:
            The updated entity
        """
        pass
    
    @abc.abstractmethod
    def delete(self, id_: Any) -> bool:
        """Delete an entity by ID.
        
        Args:
            id_: The entity ID
            
        Returns:
            True if the entity was deleted, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def exists(self, id_: Any) -> bool:
        """Check if an entity exists.
        
        Args:
            id_: The entity ID
            
        Returns:
            True if the entity exists, False otherwise
        """
        pass


class PaperRepositoryInterface(RepositoryInterface[T], abc.ABC):
    """Interface for paper repositories."""
    
    @abc.abstractmethod
    def get_by_title(self, title: str) -> List[T]:
        """Get papers by title.
        
        Args:
            title: The paper title (exact or partial match)
            
        Returns:
            List of matching papers
        """
        pass
    
    @abc.abstractmethod
    def get_by_author(self, author: str) -> List[T]:
        """Get papers by author.
        
        Args:
            author: The author name (exact or partial match)
            
        Returns:
            List of matching papers
        """
        pass
    
    @abc.abstractmethod
    def get_latest(self, limit: int = 10) -> List[T]:
        """Get the latest papers.
        
        Args:
            limit: Maximum number of papers to return
            
        Returns:
            List of papers
        """
        pass


class RevisionRepositoryInterface(RepositoryInterface[T], abc.ABC):
    """Interface for revision repositories."""
    
    @abc.abstractmethod
    def get_by_paper_id(self, paper_id: Any) -> List[T]:
        """Get revisions by paper ID.
        
        Args:
            paper_id: The paper ID
            
        Returns:
            List of matching revisions
        """
        pass
    
    @abc.abstractmethod
    def get_latest_by_paper_id(self, paper_id: Any) -> Optional[T]:
        """Get the latest revision for a paper.
        
        Args:
            paper_id: The paper ID
            
        Returns:
            The latest revision, or None if not found
        """
        pass


class ReviewRepositoryInterface(RepositoryInterface[T], abc.ABC):
    """Interface for review repositories."""
    
    @abc.abstractmethod
    def get_by_paper_id(self, paper_id: Any) -> List[T]:
        """Get reviews by paper ID.
        
        Args:
            paper_id: The paper ID
            
        Returns:
            List of matching reviews
        """
        pass
    
    @abc.abstractmethod
    def get_by_reviewer_id(self, reviewer_id: Any) -> List[T]:
        """Get reviews by reviewer ID.
        
        Args:
            reviewer_id: The reviewer ID
            
        Returns:
            List of matching reviews
        """
        pass


class JournalRepositoryInterface(RepositoryInterface[T], abc.ABC):
    """Interface for journal repositories."""
    
    @abc.abstractmethod
    def get_by_name(self, name: str) -> List[T]:
        """Get journals by name.
        
        Args:
            name: The journal name (exact or partial match)
            
        Returns:
            List of matching journals
        """
        pass
    
    @abc.abstractmethod
    def get_by_issn(self, issn: str) -> Optional[T]:
        """Get a journal by ISSN.
        
        Args:
            issn: The journal ISSN
            
        Returns:
            The journal, or None if not found
        """
        pass
    
    @abc.abstractmethod
    def get_by_category(self, category: str) -> List[T]:
        """Get journals by category.
        
        Args:
            category: The journal category
            
        Returns:
            List of matching journals
        """
        pass


class WorkflowRepositoryInterface(RepositoryInterface[T], abc.ABC):
    """Interface for workflow repositories."""
    
    @abc.abstractmethod
    def get_by_status(self, status: str) -> List[T]:
        """Get workflows by status.
        
        Args:
            status: The workflow status
            
        Returns:
            List of matching workflows
        """
        pass
    
    @abc.abstractmethod
    def get_by_user_id(self, user_id: Any) -> List[T]:
        """Get workflows by user ID.
        
        Args:
            user_id: The user ID
            
        Returns:
            List of matching workflows
        """
        pass
    
    @abc.abstractmethod
    def get_by_paper_id(self, paper_id: Any) -> List[T]:
        """Get workflows by paper ID.
        
        Args:
            paper_id: The paper ID
            
        Returns:
            List of matching workflows
        """
        pass