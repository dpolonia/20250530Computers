"""
Revised paper generator for creating the revised paper with track changes.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Tuple

from src.core.context import RevisionContext
from src.utils.document_processor import DocumentProcessor
from src.utils.pdf_processor import PDFProcessor


class RevisedPaperGenerator:
    """
    Creates the revised paper with track changes.
    
    This class is responsible for applying the proposed changes to the original
    paper and creating a new document with track changes enabled, allowing the
    user to see what was changed.
    """
    
    def __init__(self, context: RevisionContext):
        """
        Initialize the revised paper generator.
        
        Args:
            context: The shared revision context
        """
        self.context = context
        self.logger = context.logger or logging.getLogger(__name__)
    
    def create_revised_paper(
        self,
        changes: List[Tuple[str, str, str, Optional[int]]],
        output_path: Optional[str] = None
    ) -> str:
        """
        Create revised paper with track changes.
        
        Args:
            changes: List of tuples (old_text, new_text, reason, line_number)
            output_path: Path where the document should be saved (optional)
            
        Returns:
            Path to the created document
        """
        self.logger.info("Creating revised paper with track changes")
        
        # Determine output path if not provided
        if output_path is None:
            output_path = self.context.get_output_path("revised_paper.docx")
        
        # Load the original docx
        if os.path.exists(self.context.original_docx_path):
            doc_processor = DocumentProcessor(self.context.original_docx_path)
        else:
            # Create a new document from the PDF
            pdf_processor = PDFProcessor(self.context.original_paper_path)
            temp_docx_path = pdf_processor.pdf_to_docx()
            pdf_processor.close()
            doc_processor = DocumentProcessor(temp_docx_path)
        
        # Apply changes with track changes
        changes_applied = 0
        for old_text, new_text, reason, _ in changes:
            if doc_processor.add_tracked_change(old_text, new_text, reason):
                changes_applied += 1
        
        # Save the revised document
        doc_processor.save(output_path)
        self.context.process_statistics["files_created"] = self.context.process_statistics.get("files_created", 0) + 1
        
        self.logger.info(f"Applied {changes_applied} changes to the paper")
        self.logger.info(f"Revised paper created at {output_path}")
        return output_path