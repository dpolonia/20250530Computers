"""
Solution domain entity.

This module defines the Solution entity, which represents a proposed solution
to address an issue in a paper.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from .issue import Issue


class ComplexityLevel(str, Enum):
    """Complexity level of a solution."""
    
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Solution:
    """
    Domain entity representing a solution to an issue.
    
    This entity encapsulates information about a proposed solution to address
    an issue identified in the paper.
    """
    
    title: str
    implementation: str
    complexity: ComplexityLevel
    impact: str
    addresses: List[str]  # List of issue descriptions this solution addresses
    issues: List[Issue] = field(default_factory=list)  # Linked issues
    
    @property
    def is_high_complexity(self) -> bool:
        """Check if the solution has high complexity."""
        return self.complexity == ComplexityLevel.HIGH
    
    @property
    def issue_count(self) -> int:
        """Get the number of issues this solution addresses."""
        return len(self.issues) if self.issues else len(self.addresses)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the solution to a dictionary.
        
        Returns:
            Dictionary representation of the solution
        """
        return {
            "title": self.title,
            "implementation": self.implementation,
            "complexity": self.complexity.value,
            "impact": self.impact,
            "addresses": self.addresses,
            "issues": [issue.to_dict() for issue in self.issues] if self.issues else []
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Solution':
        """
        Create a solution from a dictionary.
        
        Args:
            data: Dictionary with solution data
            
        Returns:
            Solution instance
        """
        # Convert complexity string to enum
        complexity_str = data.get("complexity", "medium")
        if isinstance(complexity_str, str):
            try:
                complexity = ComplexityLevel(complexity_str.lower())
            except ValueError:
                complexity = ComplexityLevel.MEDIUM
        else:
            complexity = ComplexityLevel.MEDIUM
        
        # Convert issues
        issues = []
        for issue_data in data.get("issues", []):
            if isinstance(issue_data, dict):
                issues.append(Issue.from_dict(issue_data))
        
        return cls(
            title=data.get("title", ""),
            implementation=data.get("implementation", ""),
            complexity=complexity,
            impact=data.get("impact", ""),
            addresses=data.get("addresses", []),
            issues=issues
        )