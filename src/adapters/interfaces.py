"""
Adapter interfaces for the paper revision tool.

This module defines the abstract base classes for adapters, providing a
consistent interface for interacting with external systems.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, BinaryIO


class FileAdapter(ABC):
    """
    Base interface for file adapters.
    
    File adapters are responsible for reading and writing files in specific
    formats, isolating the application from file format details.
    """
    
    @abstractmethod
    def read(self, file_path: str) -> Any:
        """
        Read a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Parsed file content
        """
        pass
    
    @abstractmethod
    def write(self, content: Any, file_path: str) -> str:
        """
        Write content to a file.
        
        Args:
            content: Content to write
            file_path: Path where the file should be saved
            
        Returns:
            Path to the written file
        """
        pass


class PDFAdapterInterface(FileAdapter):
    """Interface for PDF adapters."""
    
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        pass
    
    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted metadata
        """
        pass
    
    @abstractmethod
    def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of extracted tables
        """
        pass
    
    @abstractmethod
    def extract_figures(self, file_path: str) -> List[Tuple[str, Optional[str]]]:
        """
        Extract figures from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of tuples (caption, path)
        """
        pass
    
    @abstractmethod
    def extract_sections(self, file_path: str) -> Dict[str, str]:
        """
        Extract sections from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary mapping section names to content
        """
        pass
    
    @abstractmethod
    def pdf_to_docx(self, file_path: str) -> str:
        """
        Convert a PDF file to DOCX format.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Path to the created DOCX file
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass


class DocxAdapterInterface(FileAdapter):
    """Interface for DOCX adapters."""
    
    @abstractmethod
    def create_changes_document(
        self,
        changes: List[Tuple[str, str, str, Optional[int]]],
        output_path: str
    ) -> str:
        """
        Create a document detailing changes.
        
        Args:
            changes: List of tuples (old_text, new_text, reason, line_number)
            output_path: Path where the document should be saved
            
        Returns:
            Path to the created document
        """
        pass
    
    @abstractmethod
    def add_tracked_change(
        self,
        old_text: str,
        new_text: str,
        reason: str
    ) -> bool:
        """
        Add a tracked change to the document.
        
        Args:
            old_text: Text to be replaced
            new_text: Text to replace with
            reason: Reason for the change
            
        Returns:
            True if the change was added, False otherwise
        """
        pass
    
    @abstractmethod
    def create_editor_letter(
        self,
        reviewer_responses: List[Dict[str, Any]],
        output_path: str,
        process_summary: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a letter to the editor.
        
        Args:
            reviewer_responses: Reviewer responses data
            output_path: Path where the document should be saved
            process_summary: Optional process summary data
            
        Returns:
            Path to the created document
        """
        pass
    
    @abstractmethod
    def save(self, output_path: str) -> str:
        """
        Save the document.
        
        Args:
            output_path: Path where the document should be saved
            
        Returns:
            Path to the saved document
        """
        pass


class LLMAdapterInterface(ABC):
    """Interface for language model adapters."""
    
    @abstractmethod
    def get_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Get a completion from the language model.
        
        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The model's response
        """
        pass
    
    @abstractmethod
    def get_token_estimate(self, text: str) -> int:
        """
        Get an estimate of the number of tokens in a text.
        
        Args:
            text: The text to estimate
            
        Returns:
            Estimated number of tokens
        """
        pass


class BibtexAdapterInterface(FileAdapter):
    """Interface for BibTeX adapters."""
    
    @abstractmethod
    def validate_references(self, references: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate references.
        
        Args:
            references: List of reference IDs to validate
            
        Returns:
            Tuple of (valid_references, invalid_references)
        """
        pass
    
    @abstractmethod
    def add_reference(self, reference_data: Dict[str, str]) -> str:
        """
        Add a new reference.
        
        Args:
            reference_data: Reference data
            
        Returns:
            Reference ID
        """
        pass
    
    @abstractmethod
    def export_references(
        self,
        references: List[str],
        output_path: str
    ) -> str:
        """
        Export references to a BibTeX file.
        
        Args:
            references: List of reference IDs to export
            output_path: Path where the BibTeX file should be saved
            
        Returns:
            Path to the created BibTeX file
        """
        pass


class ScopusAdapterInterface(ABC):
    """Interface for Scopus API adapters."""
    
    @abstractmethod
    def search_by_title(self, title: str) -> List[Dict[str, Any]]:
        """
        Search for papers by title.
        
        Args:
            title: Paper title
            
        Returns:
            List of matching papers
        """
        pass
    
    @abstractmethod
    def search_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """
        Search for a paper by DOI.
        
        Args:
            doi: DOI to search for
            
        Returns:
            Paper if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_citation_count(self, doi: str) -> int:
        """
        Get the citation count for a paper.
        
        Args:
            doi: DOI of the paper
            
        Returns:
            Citation count
        """
        pass