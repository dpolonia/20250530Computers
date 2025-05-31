"""
Issue domain entity.

This module defines the Issue entity, which represents a problem or concern
identified in a paper.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class SeverityLevel(str, Enum):
    """Severity level of an issue."""
    
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueType(str, Enum):
    """Type of issue."""
    
    METHODOLOGY = "methodology"
    RESULTS = "results"
    WRITING = "writing"
    REFERENCES = "references"
    STRUCTURE = "structure"
    OTHER = "other"


@dataclass
class Issue:
    """
    Domain entity representing an issue identified in a paper.
    
    This entity encapsulates information about a problem or concern that needs
    to be addressed in the paper revision.
    """
    
    description: str
    severity: SeverityLevel
    type: IssueType
    source: str  # E.g., "Reviewer 1", "Analysis"
    section: Optional[str] = None  # Section where the issue was found
    reviewer_number: Optional[int] = None  # Reviewer who identified the issue
    line_number: Optional[int] = None  # Line number where the issue was found
    
    @property
    def is_high_severity(self) -> bool:
        """Check if the issue has high severity."""
        return self.severity == SeverityLevel.HIGH
    
    @property
    def is_methodology_issue(self) -> bool:
        """Check if the issue is related to methodology."""
        return self.type == IssueType.METHODOLOGY
    
    @property
    def is_results_issue(self) -> bool:
        """Check if the issue is related to results."""
        return self.type == IssueType.RESULTS
    
    @property
    def is_writing_issue(self) -> bool:
        """Check if the issue is related to writing."""
        return self.type == IssueType.WRITING
    
    @property
    def is_references_issue(self) -> bool:
        """Check if the issue is related to references."""
        return self.type == IssueType.REFERENCES
    
    @property
    def is_structure_issue(self) -> bool:
        """Check if the issue is related to structure."""
        return self.type == IssueType.STRUCTURE
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the issue to a dictionary.
        
        Returns:
            Dictionary representation of the issue
        """
        return {
            "description": self.description,
            "severity": self.severity.value,
            "type": self.type.value,
            "source": self.source,
            "section": self.section,
            "reviewer_number": self.reviewer_number,
            "line_number": self.line_number
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Issue':
        """
        Create an issue from a dictionary.
        
        Args:
            data: Dictionary with issue data
            
        Returns:
            Issue instance
        """
        # Convert severity string to enum
        severity_str = data.get("severity", "medium")
        if isinstance(severity_str, str):
            try:
                severity = SeverityLevel(severity_str.lower())
            except ValueError:
                severity = SeverityLevel.MEDIUM
        else:
            severity = SeverityLevel.MEDIUM
        
        # Convert type string to enum
        type_str = data.get("type", "other")
        if isinstance(type_str, str):
            try:
                issue_type = IssueType(type_str.lower())
            except ValueError:
                issue_type = IssueType.OTHER
        else:
            issue_type = IssueType.OTHER
        
        return cls(
            description=data.get("description", ""),
            severity=severity,
            type=issue_type,
            source=data.get("source", ""),
            section=data.get("section"),
            reviewer_number=data.get("reviewer_number"),
            line_number=data.get("line_number")
        )