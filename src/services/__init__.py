"""
Service layer for the paper revision tool.

This module contains service classes that coordinate business logic and provide
a clean separation between domain logic and external dependencies.
"""

from .paper_service import PaperService
from .reviewer_service import ReviewerService
from .solution_service import SolutionService
from .document_service import DocumentService
from .reference_service import ReferenceService
from .llm_service import LLMService

__all__ = [
    'PaperService',
    'ReviewerService',
    'SolutionService',
    'DocumentService',
    'ReferenceService',
    'LLMService'
]