"""
Domain model for the paper revision tool.

This module contains domain entities and value objects that represent the core
concepts of the paper revision domain.
"""

from .paper import Paper
from .reviewer_comment import ReviewerComment
from .issue import Issue
from .solution import Solution
from .change import Change
from .reference import Reference
from .assessment import Assessment

__all__ = [
    'Paper',
    'ReviewerComment',
    'Issue',
    'Solution',
    'Change',
    'Reference',
    'Assessment'
]