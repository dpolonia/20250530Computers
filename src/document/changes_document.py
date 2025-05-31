"""
Changes document generator for creating a document detailing proposed changes.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Tuple

from src.core.context import RevisionContext
from src.utils.document_processor import DocumentProcessor
from src.utils.pdf_processor import PDFProcessor


class ChangesDocumentGenerator:
    """
    Creates a document detailing all proposed changes to the paper.
    
    This class is responsible for creating a document that outlines all the 
    proposed changes to be made to the original paper, including the old text,
    new text, and the reason for each change.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the changes document generator.
        
        Args:
            context: The shared revision context
        """
        self.context = context
        self.logger = context.logger or logging.getLogger(__name__)
    
    def create_changes_document(
        self,
        changes: List[Tuple[str, str, str, Optional[int]]],
        output_path: Optional[str] = None
    ) -> str:
        """
        Create a document detailing all changes.
        
        Args:
            changes: List of tuples (old_text, new_text, reason, line_number)
            output_path: Path where the document should be saved (optional)
            
        Returns:
            Path to the created document
        """
        self.logger.info("Creating changes document")
        
        # Determine output path if not provided
        if output_path is None:
            output_path = self.context.get_output_path("changes_document.docx")
        
        # Load the original docx if it exists, otherwise use a new document
        if os.path.exists(self.context.original_docx_path):
            doc_processor = DocumentProcessor(self.context.original_docx_path)
        else:
            # Create a new document from the PDF
            pdf_processor = PDFProcessor(self.context.original_paper_path)
            temp_docx_path = pdf_processor.pdf_to_docx()
            pdf_processor.close()
            doc_processor = DocumentProcessor(temp_docx_path)
        
        # Create the changes document
        changes_path = doc_processor.create_changes_document(changes, output_path)
        self.context.process_statistics["files_created"] = self.context.process_statistics.get("files_created", 0) + 1
        
        self.logger.info(f"Changes document created at {changes_path}")
        return changes_path