"""
Repository layer for the paper revision tool.

This module contains repository classes that provide an abstraction over data
access, whether from databases, files, or external services.
"""

from .paper_repository import PaperRepository
from .reviewer_repository import ReviewerRepository
from .reference_repository import ReferenceRepository
from .document_repository import DocumentRepository
from .change_repository import ChangeRepository

__all__ = [
    'PaperRepository',
    'ReviewerRepository',
    'ReferenceRepository',
    'DocumentRepository',
    'ChangeRepository'
]