"""
Change domain entity.

This module defines the Change entity, which represents a specific text change
to be applied to a paper.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .solution import Solution


@dataclass
class Change:
    """
    Domain entity representing a specific text change.
    
    This entity encapsulates information about a specific text change to be
    applied to the paper, including the old text, new text, reason, and location.
    """
    
    old_text: str
    new_text: str
    reason: str
    line_number: Optional[int] = None
    section: Optional[str] = None
    solution: Optional[Solution] = None  # The solution this change implements
    
    @property
    def is_addition(self) -> bool:
        """Check if this change is an addition (empty old text)."""
        return not self.old_text and self.new_text
    
    @property
    def is_deletion(self) -> bool:
        """Check if this change is a deletion (empty new text)."""
        return self.old_text and not self.new_text
    
    @property
    def is_modification(self) -> bool:
        """Check if this change is a modification (both old and new text)."""
        return self.old_text and self.new_text
    
    @property
    def word_count_delta(self) -> int:
        """Get the change in word count (positive for additions, negative for deletions)."""
        old_words = len(self.old_text.split()) if self.old_text else 0
        new_words = len(self.new_text.split()) if self.new_text else 0
        return new_words - old_words
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the change to a dictionary.
        
        Returns:
            Dictionary representation of the change
        """
        return {
            "old_text": self.old_text,
            "new_text": self.new_text,
            "reason": self.reason,
            "line_number": self.line_number,
            "section": self.section,
            "solution": self.solution.to_dict() if self.solution else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Change':
        """
        Create a change from a dictionary.
        
        Args:
            data: Dictionary with change data
            
        Returns:
            Change instance
        """
        # Convert solution data if present
        solution = None
        if data.get("solution") and isinstance(data["solution"], dict):
            from .solution import Solution
            solution = Solution.from_dict(data["solution"])
        
        return cls(
            old_text=data.get("old_text", ""),
            new_text=data.get("new_text", ""),
            reason=data.get("reason", ""),
            line_number=data.get("line_number"),
            section=data.get("section"),
            solution=solution
        )
    
    @classmethod
    def from_tuple(cls, change_tuple: tuple) -> 'Change':
        """
        Create a change from a tuple (old_text, new_text, reason, line_number).
        
        Args:
            change_tuple: Tuple with change data
            
        Returns:
            Change instance
        """
        old_text, new_text, reason, line_number = change_tuple
        return cls(
            old_text=old_text,
            new_text=new_text,
            reason=reason,
            line_number=line_number
        )
    
    def to_tuple(self) -> tuple:
        """
        Convert the change to a tuple (old_text, new_text, reason, line_number).
        
        Returns:
            Tuple representation of the change
        """
        return (self.old_text, self.new_text, self.reason, self.line_number)