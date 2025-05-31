"""Analysis module for the paper revision tool."""

from .paper_analyzer import PaperAnalyzer
from .reviewer_analyzer import ReviewerAnalyzer
from .solution_generator import SolutionGenerator

__all__ = [
    'PaperAnalyzer',
    'ReviewerAnalyzer',
    'SolutionGenerator'
]
