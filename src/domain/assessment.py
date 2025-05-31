"""
Assessment domain entity.

This module defines the Assessment entity, which represents an evaluation
of paper changes and their impact.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class ImpactLevel(str, Enum):
    """Level of impact."""
    
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Assessment:
    """
    Domain entity representing an assessment of paper changes.
    
    This entity encapsulates information about the impact of changes made to
    a paper, including overall impact, reviewer concerns addressed, remaining
    issues, and recommendations.
    """
    
    overall_impact: str
    reviewer_concerns: str
    remaining_issues: List[str]
    strengthened_areas: List[str]
    recommendations: List[str]
    impact_level: ImpactLevel = ImpactLevel.MEDIUM
    created_at: Optional[str] = None
    
    @property
    def is_high_impact(self) -> bool:
        """Check if the assessment indicates high impact."""
        return self.impact_level == ImpactLevel.HIGH
    
    @property
    def is_low_impact(self) -> bool:
        """Check if the assessment indicates low impact."""
        return self.impact_level == ImpactLevel.LOW
    
    @property
    def has_remaining_issues(self) -> bool:
        """Check if there are remaining issues."""
        return len(self.remaining_issues) > 0
    
    @property
    def issue_count(self) -> int:
        """Get the number of remaining issues."""
        return len(self.remaining_issues)
    
    @property
    def recommendation_count(self) -> int:
        """Get the number of recommendations."""
        return len(self.recommendations)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the assessment to a dictionary.
        
        Returns:
            Dictionary representation of the assessment
        """
        return {
            "overall_impact": self.overall_impact,
            "reviewer_concerns": self.reviewer_concerns,
            "remaining_issues": self.remaining_issues,
            "strengthened_areas": self.strengthened_areas,
            "recommendations": self.recommendations,
            "impact_level": self.impact_level.value,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Assessment':
        """
        Create an assessment from a dictionary.
        
        Args:
            data: Dictionary with assessment data
            
        Returns:
            Assessment instance
        """
        # Convert impact level string to enum
        impact_str = data.get("impact_level", "medium")
        if isinstance(impact_str, str):
            try:
                impact_level = ImpactLevel(impact_str.lower())
            except ValueError:
                impact_level = ImpactLevel.MEDIUM
        else:
            impact_level = ImpactLevel.MEDIUM
        
        # Handle remaining issues
        remaining_issues = data.get("remaining_issues", [])
        if isinstance(remaining_issues, str):
            remaining_issues = [remaining_issues]
        
        # Handle strengthened areas
        strengthened_areas = data.get("strengthened_areas", [])
        if isinstance(strengthened_areas, str):
            strengthened_areas = [strengthened_areas]
        
        # Handle recommendations
        recommendations = data.get("recommendations", [])
        if isinstance(recommendations, str):
            recommendations = [recommendations]
        
        return cls(
            overall_impact=data.get("overall_impact", ""),
            reviewer_concerns=data.get("reviewer_concerns", ""),
            remaining_issues=remaining_issues,
            strengthened_areas=strengthened_areas,
            recommendations=recommendations,
            impact_level=impact_level,
            created_at=data.get("created_at")
        )