"""Document generation module for the paper revision tool."""

from .changes_document import ChangesDocumentGenerator
from .revised_paper import RevisedPaperGenerator
from .assessment import AssessmentGenerator
from .editor_letter import EditorLetterGenerator

__all__ = [
    'ChangesDocumentGenerator',
    'RevisedPaperGenerator',
    'AssessmentGenerator',
    'EditorLetterGenerator'
]
