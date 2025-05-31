"""
Adapter layer for the paper revision tool.

This module contains adapter classes that provide an interface to external
systems and APIs, isolating the application from external dependencies.
"""

from .pdf_adapter import PDFAdapter
from .docx_adapter import DocxAdapter
from .llm_adapter import LLMAdapter
from .scopus_adapter import ScopusAdapter
from .bibtex_adapter import BibtexAdapter

__all__ = [
    'PDFAdapter',
    'DocxAdapter',
    'LLMAdapter',
    'ScopusAdapter',
    'BibtexAdapter'
]