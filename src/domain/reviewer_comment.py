"""
Reviewer comment domain entity.

This module defines the ReviewerComment entity, which represents reviewer feedback
in the domain model.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from enum import Enum


class AssessmentType(str, Enum):
    """Type of reviewer assessment."""
    
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


@dataclass
class Comment:
    """Individual comment from a reviewer."""
    
    text: str
    type: str  # E.g., "concern", "suggestion", "question"
    importance: int = 1  # 1 = low, 5 = high


@dataclass
class ReviewerComment:
    """
    Domain entity representing feedback from a reviewer.
    
    This entity encapsulates all the information about a reviewer's comments,
    including their assessment, concerns, and required changes.
    """
    
    reviewer_number: int
    overall_assessment: AssessmentType
    main_concerns: List[str] = field(default_factory=list)
    required_changes: List[str] = field(default_factory=list)
    suggested_changes: List[str] = field(default_factory=list)
    methodology_comments: List[str] = field(default_factory=list)
    results_comments: List[str] = field(default_factory=list)
    writing_comments: List[str] = field(default_factory=list)
    references_comments: List[str] = field(default_factory=list)
    detailed_comments: List[Comment] = field(default_factory=list)
    full_text: Optional[str] = None
    text_preview: Optional[str] = None
    
    @property
    def is_positive(self) -> bool:
        """Check if the overall assessment is positive."""
        return self.overall_assessment == AssessmentType.POSITIVE
    
    @property
    def is_negative(self) -> bool:
        """Check if the overall assessment is negative."""
        return self.overall_assessment == AssessmentType.NEGATIVE
    
    @property
    def concern_count(self) -> int:
        """Get the number of main concerns."""
        return len(self.main_concerns)
    
    @property
    def change_count(self) -> int:
        """Get the total number of required and suggested changes."""
        return len(self.required_changes) + len(self.suggested_changes)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the reviewer comment to a dictionary.
        
        Returns:
            Dictionary representation of the reviewer comment
        """
        return {
            "reviewer_number": self.reviewer_number,
            "overall_assessment": self.overall_assessment.value,
            "main_concerns": self.main_concerns,
            "required_changes": self.required_changes,
            "suggested_changes": self.suggested_changes,
            "methodology_comments": self.methodology_comments,
            "results_comments": self.results_comments,
            "writing_comments": self.writing_comments,
            "references_comments": self.references_comments,
            "detailed_comments": [
                {"text": c.text, "type": c.type, "importance": c.importance}
                for c in self.detailed_comments
            ],
            "text_preview": self.text_preview
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewerComment':
        """
        Create a reviewer comment from a dictionary.
        
        Args:
            data: Dictionary with reviewer comment data
            
        Returns:
            ReviewerComment instance
        """
        # Convert assessment string to enum
        assessment_str = data.get("overall_assessment", "neutral")
        if isinstance(assessment_str, str):
            try:
                assessment = AssessmentType(assessment_str.lower())
            except ValueError:
                assessment = AssessmentType.NEUTRAL
        else:
            assessment = AssessmentType.NEUTRAL
        
        # Convert detailed comments
        detailed_comments = []
        for comment_data in data.get("detailed_comments", []):
            detailed_comments.append(Comment(
                text=comment_data["text"],
                type=comment_data["type"],
                importance=comment_data.get("importance", 1)
            ))
        
        return cls(
            reviewer_number=data.get("reviewer_number", 0),
            overall_assessment=assessment,
            main_concerns=data.get("main_concerns", []),
            required_changes=data.get("required_changes", []),
            suggested_changes=data.get("suggested_changes", []),
            methodology_comments=data.get("methodology_comments", []),
            results_comments=data.get("results_comments", []),
            writing_comments=data.get("writing_comments", []),
            references_comments=data.get("references_comments", []),
            detailed_comments=detailed_comments,
            full_text=data.get("full_text"),
            text_preview=data.get("text_preview")
        )