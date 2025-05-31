"""
Interface definitions for analysis components.

This module defines abstract base classes that serve as interfaces for the
analysis components, ensuring consistent design patterns across the codebase.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

from src.core.context import RevisionContext


class Analyzer(ABC):
    """
    Base interface for all analyzer components.
    
    Analyzers are responsible for extracting structured information from
    unstructured data sources like papers and reviewer comments.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the analyzer with a revision context.
        
        Args:
            context: The shared revision context
        """
        self.context = context
    
    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """
        Perform analysis and return structured results.
        
        Returns:
            Dictionary with analysis results
        """
        pass


class PaperAnalyzerInterface(Analyzer):
    """Interface for paper analysis components."""
    
    @abstractmethod
    def analyze_paper(self) -> Dict[str, Any]:
        """
        Analyze the original paper and extract structured information.
        
        Returns:
            Dictionary with paper analysis results
        """
        pass


class ReviewerAnalyzerInterface(Analyzer):
    """Interface for reviewer comment analysis components."""
    
    @abstractmethod
    def analyze_reviewer_comments(self) -> List[Dict[str, Any]]:
        """
        Analyze reviewer comments and extract structured feedback.
        
        Returns:
            List of dictionaries with reviewer comment analyses
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


class SolutionGeneratorInterface(Analyzer):
    """Interface for solution generation components."""
    
    @abstractmethod
    def identify_issues(
        self, 
        paper_analysis: Dict[str, Any],
        reviewer_comments: List[Dict[str, Any]],
        editor_requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify issues and concerns from paper analysis and reviewer comments.
        
        Args:
            paper_analysis: Analysis of the original paper
            reviewer_comments: Analysis of reviewer comments
            editor_requirements: Editor requirements and PRISMA guidelines
            
        Returns:
            List of identified issues
        """
        pass
    
    @abstractmethod
    def generate_solutions(
        self,
        paper_analysis: Dict[str, Any],
        issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate solutions for identified issues.
        
        Args:
            paper_analysis: Analysis of the original paper
            issues: List of identified issues
            
        Returns:
            List of solution dictionaries
        """
        pass
    
    @abstractmethod
    def generate_specific_changes(
        self,
        paper_analysis: Dict[str, Any],
        solutions: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, str, Optional[int]]]:
        """
        Generate specific text changes to implement solutions.
        
        Args:
            paper_analysis: Analysis of the original paper
            solutions: List of solutions to implement
            
        Returns:
            List of tuples (old_text, new_text, reason, line_number)
        """
        pass