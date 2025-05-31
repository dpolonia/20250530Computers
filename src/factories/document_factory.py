"""
Document factory for the Paper Revision System.

This module provides factories for creating document processors, writers, and converters.
"""

import os
import logging
from typing import Dict, Any, Optional, Type

from src.interfaces.document import (
    DocumentProcessorInterface,
    DocumentWriterInterface,
    DocumentConverterInterface,
    DocumentFactoryInterface
)

# Import concrete implementations
from src.utils.pdf_processor import PDFProcessor
from src.utils.document_processor import DocumentProcessor

# Configure logging
logger = logging.getLogger(__name__)


class DocumentFactory(DocumentFactoryInterface):
    """Factory for document-related components."""
    
    def __init__(self):
        """Initialize the factory."""
        self._processor_registry = {}
        self._writer_registry = {}
        self._converter_registry = {}
        
        # Register default processor implementations
        self.register_processor('.pdf', PDFProcessor)
        self.register_processor('.docx', DocumentProcessor)
    
    def register_processor(self, extension: str, processor_class: Type[DocumentProcessorInterface]):
        """Register a document processor for a file extension.
        
        Args:
            extension: File extension (including the dot)
            processor_class: Document processor class
        """
        self._processor_registry[extension.lower()] = processor_class
        logger.debug(f"Registered processor for {extension}: {processor_class.__name__}")
    
    def register_writer(self, format_type: str, writer_class: Type[DocumentWriterInterface]):
        """Register a document writer for a format.
        
        Args:
            format_type: Document format
            writer_class: Document writer class
        """
        self._writer_registry[format_type.lower()] = writer_class
        logger.debug(f"Registered writer for {format_type}: {writer_class.__name__}")
    
    def register_converter(self, converter_class: Type[DocumentConverterInterface]):
        """Register a document converter.
        
        Args:
            converter_class: Document converter class
        """
        self._converter_registry[converter_class.__name__] = converter_class
        logger.debug(f"Registered converter: {converter_class.__name__}")
    
    def create_processor(self, file_path: str) -> DocumentProcessorInterface:
        """Create a document processor for the given file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document processor instance
            
        Raises:
            ValueError: If the file format is not supported
        """
        # Get file extension
        _, ext = os.path.splitext(file_path.lower())
        
        # Check if processor exists for this extension
        if ext not in self._processor_registry:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Create processor
        processor_class = self._processor_registry[ext]
        logger.debug(f"Creating processor for {ext}: {processor_class.__name__}")
        
        try:
            processor = processor_class(file_path)
            return processor
        except Exception as e:
            logger.error(f"Failed to create processor for {file_path}: {e}")
            raise
    
    def create_writer(self, format_type: str) -> DocumentWriterInterface:
        """Create a document writer for the given format.
        
        Args:
            format_type: Document format (e.g., 'pdf', 'docx', 'html')
            
        Returns:
            Document writer instance
            
        Raises:
            ValueError: If the format is not supported
        """
        # Normalize format
        format_type = format_type.lower()
        
        # Check if writer exists for this format
        if format_type not in self._writer_registry:
            raise ValueError(f"Unsupported format for writer: {format_type}")
        
        # Create writer
        writer_class = self._writer_registry[format_type]
        logger.debug(f"Creating writer for {format_type}: {writer_class.__name__}")
        
        try:
            writer = writer_class()
            return writer
        except Exception as e:
            logger.error(f"Failed to create writer for {format_type}: {e}")
            raise
    
    def create_converter(self) -> DocumentConverterInterface:
        """Create a document converter.
        
        Returns:
            Document converter instance
            
        Raises:
            ValueError: If no converter is registered
        """
        # Check if any converters are registered
        if not self._converter_registry:
            raise ValueError("No document converters registered")
        
        # Use the first registered converter
        converter_name = list(self._converter_registry.keys())[0]
        converter_class = self._converter_registry[converter_name]
        logger.debug(f"Creating converter: {converter_name}")
        
        try:
            converter = converter_class()
            return converter
        except Exception as e:
            logger.error(f"Failed to create converter: {e}")
            raise


# Create a singleton instance
_document_factory = None

def get_document_factory() -> DocumentFactory:
    """Get the document factory singleton.
    
    Returns:
        Document factory instance
    """
    global _document_factory
    
    if _document_factory is None:
        _document_factory = DocumentFactory()
        
    return _document_factory