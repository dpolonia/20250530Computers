"""
Document processing interfaces for the Paper Revision System.

This module defines interfaces for document processing components,
including PDF and DOCX document processors.
"""

import abc
from typing import Dict, List, Optional, Any, Union


class DocumentProcessorInterface(abc.ABC):
    """Interface for document processors."""
    
    @abc.abstractmethod
    def load_document(self, file_path: str) -> bool:
        """Load a document from a file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if the document was loaded successfully, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def extract_text(self, output_path: Optional[str] = None) -> str:
        """Extract text from the document.
        
        Args:
            output_path: Optional path to save the extracted text
            
        Returns:
            Extracted text
        """
        pass
    
    @abc.abstractmethod
    def extract_sections(self) -> Dict[str, str]:
        """Extract sections from the document.
        
        Returns:
            Dictionary mapping section names to section contents
        """
        pass
    
    @abc.abstractmethod
    def extract_metadata(self) -> Dict[str, Any]:
        """Extract metadata from the document.
        
        Returns:
            Dictionary of metadata
        """
        pass
    
    @abc.abstractmethod
    def extract_references(self) -> List[Dict[str, Any]]:
        """Extract references from the document.
        
        Returns:
            List of reference dictionaries
        """
        pass
    
    @abc.abstractmethod
    def get_document_info(self) -> Dict[str, Any]:
        """Get information about the document.
        
        Returns:
            Dictionary with document information
        """
        pass


class DocumentWriterInterface(abc.ABC):
    """Interface for document writers."""
    
    @abc.abstractmethod
    def create_document(self, template_path: Optional[str] = None) -> bool:
        """Create a new document.
        
        Args:
            template_path: Optional path to a template document
            
        Returns:
            True if the document was created successfully, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def add_text(self, text: str) -> bool:
        """Add text to the document.
        
        Args:
            text: Text to add
            
        Returns:
            True if the text was added successfully, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def add_heading(self, text: str, level: int = 1) -> bool:
        """Add a heading to the document.
        
        Args:
            text: Heading text
            level: Heading level (1-9)
            
        Returns:
            True if the heading was added successfully, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def add_paragraph(self, text: str, style: Optional[str] = None) -> bool:
        """Add a paragraph to the document.
        
        Args:
            text: Paragraph text
            style: Optional style name
            
        Returns:
            True if the paragraph was added successfully, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def add_table(self, data: List[List[str]], headers: bool = False) -> bool:
        """Add a table to the document.
        
        Args:
            data: Table data as a list of rows, each row being a list of cells
            headers: Whether the first row contains headers
            
        Returns:
            True if the table was added successfully, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def add_image(self, image_path: str, width: Optional[int] = None, 
                height: Optional[int] = None, caption: Optional[str] = None) -> bool:
        """Add an image to the document.
        
        Args:
            image_path: Path to the image
            width: Optional width in pixels
            height: Optional height in pixels
            caption: Optional caption
            
        Returns:
            True if the image was added successfully, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def add_page_break(self) -> bool:
        """Add a page break to the document.
        
        Returns:
            True if the page break was added successfully, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def save(self, output_path: str) -> bool:
        """Save the document to a file.
        
        Args:
            output_path: Path to save the document
            
        Returns:
            True if the document was saved successfully, False otherwise
        """
        pass


class DocumentConverterInterface(abc.ABC):
    """Interface for document converters."""
    
    @abc.abstractmethod
    def convert(self, input_path: str, output_path: str, 
              output_format: str) -> bool:
        """Convert a document to a different format.
        
        Args:
            input_path: Path to the input document
            output_path: Path to save the output document
            output_format: Format to convert to (e.g., 'pdf', 'docx', 'html')
            
        Returns:
            True if the conversion was successful, False otherwise
        """
        pass


class DocumentFactoryInterface(abc.ABC):
    """Interface for document processor factories."""
    
    @abc.abstractmethod
    def create_processor(self, file_path: str) -> DocumentProcessorInterface:
        """Create a document processor for the given file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document processor instance
            
        Raises:
            ValueError: If the file format is not supported
        """
        pass
    
    @abc.abstractmethod
    def create_writer(self, format_type: str) -> DocumentWriterInterface:
        """Create a document writer for the given format.
        
        Args:
            format_type: Document format (e.g., 'pdf', 'docx', 'html')
            
        Returns:
            Document writer instance
            
        Raises:
            ValueError: If the format is not supported
        """
        pass
    
    @abc.abstractmethod
    def create_converter(self) -> DocumentConverterInterface:
        """Create a document converter.
        
        Returns:
            Document converter instance
        """
        pass