"""
Reference domain entity.

This module defines the Reference entity, which represents a bibliographic
reference in a paper.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class ReferenceType(str, Enum):
    """Type of reference."""
    
    ARTICLE = "article"
    BOOK = "book"
    CONFERENCE = "conference"
    THESIS = "thesis"
    REPORT = "report"
    WEBSITE = "website"
    OTHER = "other"


@dataclass
class Reference:
    """
    Domain entity representing a bibliographic reference.
    
    This entity encapsulates information about a bibliographic reference,
    including title, authors, venue, and other metadata.
    """
    
    id: str  # BibTeX key or other identifier
    title: str
    authors: List[str]
    year: int
    venue: str  # Journal or conference name
    reference_type: ReferenceType = ReferenceType.ARTICLE
    doi: Optional[str] = None
    url: Optional[str] = None
    pages: Optional[str] = None
    volume: Optional[str] = None
    number: Optional[str] = None
    publisher: Optional[str] = None
    abstract: Optional[str] = None
    valid: bool = True
    reason: Optional[str] = None  # Reason for adding this reference
    
    @property
    def author_string(self) -> str:
        """Get a formatted string of authors."""
        if not self.authors:
            return "Unknown Author"
        
        if len(self.authors) == 1:
            return self.authors[0]
        elif len(self.authors) == 2:
            return f"{self.authors[0]} and {self.authors[1]}"
        else:
            return f"{self.authors[0]} et al."
    
    @property
    def citation(self) -> str:
        """Get a formatted citation string."""
        return f"{self.author_string}, {self.year}. {self.title}. {self.venue}."
    
    @property
    def is_recent(self) -> bool:
        """Check if the reference is recent (published in the last 5 years)."""
        import datetime
        current_year = datetime.datetime.now().year
        return current_year - self.year <= 5
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the reference to a dictionary.
        
        Returns:
            Dictionary representation of the reference
        """
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "reference_type": self.reference_type.value,
            "doi": self.doi,
            "url": self.url,
            "pages": self.pages,
            "volume": self.volume,
            "number": self.number,
            "publisher": self.publisher,
            "abstract": self.abstract,
            "valid": self.valid,
            "reason": self.reason
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Reference':
        """
        Create a reference from a dictionary.
        
        Args:
            data: Dictionary with reference data
            
        Returns:
            Reference instance
        """
        # Convert type string to enum
        type_str = data.get("reference_type", "article")
        if isinstance(type_str, str):
            try:
                reference_type = ReferenceType(type_str.lower())
            except ValueError:
                reference_type = ReferenceType.ARTICLE
        else:
            reference_type = ReferenceType.ARTICLE
        
        # Handle authors as string or list
        authors = data.get("authors", [])
        if isinstance(authors, str):
            # Split authors string by common separators
            authors = [author.strip() for author in authors.replace(" and ", ", ").split(",")]
        
        # Ensure year is an integer
        year = data.get("year", 0)
        if isinstance(year, str):
            try:
                year = int(year)
            except ValueError:
                year = 0
        
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            authors=authors,
            year=year,
            venue=data.get("venue", data.get("journal", "")),
            reference_type=reference_type,
            doi=data.get("doi"),
            url=data.get("url"),
            pages=data.get("pages"),
            volume=data.get("volume"),
            number=data.get("number"),
            publisher=data.get("publisher"),
            abstract=data.get("abstract"),
            valid=data.get("valid", True),
            reason=data.get("reason")
        )
    
    def to_bibtex(self) -> str:
        """
        Convert the reference to BibTeX format.
        
        Returns:
            BibTeX string representation of the reference
        """
        # Map reference types to BibTeX entry types
        entry_type_map = {
            ReferenceType.ARTICLE: "article",
            ReferenceType.BOOK: "book",
            ReferenceType.CONFERENCE: "inproceedings",
            ReferenceType.THESIS: "phdthesis",
            ReferenceType.REPORT: "techreport",
            ReferenceType.WEBSITE: "misc",
            ReferenceType.OTHER: "misc"
        }
        
        entry_type = entry_type_map.get(self.reference_type, "article")
        
        # Start building the BibTeX entry
        bibtex = f"@{entry_type}{{{self.id},\n"
        
        # Add required fields
        bibtex += f"  title = {{{self.title}}},\n"
        bibtex += f"  author = {{{', '.join(self.authors)}}},\n"
        bibtex += f"  year = {{{self.year}}},\n"
        
        # Add venue field based on reference type
        if self.reference_type == ReferenceType.ARTICLE:
            bibtex += f"  journal = {{{self.venue}}},\n"
        elif self.reference_type == ReferenceType.CONFERENCE:
            bibtex += f"  booktitle = {{{self.venue}}},\n"
        elif self.reference_type == ReferenceType.BOOK:
            bibtex += f"  publisher = {{{self.publisher or self.venue}}},\n"
        else:
            bibtex += f"  howpublished = {{{self.venue}}},\n"
        
        # Add optional fields if they exist
        if self.doi:
            bibtex += f"  doi = {{{self.doi}}},\n"
        if self.url:
            bibtex += f"  url = {{{self.url}}},\n"
        if self.pages:
            bibtex += f"  pages = {{{self.pages}}},\n"
        if self.volume:
            bibtex += f"  volume = {{{self.volume}}},\n"
        if self.number:
            bibtex += f"  number = {{{self.number}}},\n"
        if self.publisher and self.reference_type != ReferenceType.BOOK:
            bibtex += f"  publisher = {{{self.publisher}}},\n"
        
        # Close the BibTeX entry
        bibtex += "}\n"
        
        return bibtex
    
    @classmethod
    def from_bibtex_entry(cls, entry: Dict[str, Any]) -> 'Reference':
        """
        Create a reference from a BibTeX entry.
        
        Args:
            entry: Dictionary with BibTeX entry data
            
        Returns:
            Reference instance
        """
        # Map BibTeX entry types to reference types
        reference_type_map = {
            "article": ReferenceType.ARTICLE,
            "book": ReferenceType.BOOK,
            "inproceedings": ReferenceType.CONFERENCE,
            "conference": ReferenceType.CONFERENCE,
            "phdthesis": ReferenceType.THESIS,
            "mastersthesis": ReferenceType.THESIS,
            "techreport": ReferenceType.REPORT,
            "misc": ReferenceType.OTHER
        }
        
        # Get entry type and convert to reference type
        entry_type = entry.get("ENTRYTYPE", "").lower()
        reference_type = reference_type_map.get(entry_type, ReferenceType.OTHER)
        
        # Parse authors
        authors = []
        if "author" in entry:
            # Simple split by "and" (would need more sophisticated parsing for a real app)
            authors = [author.strip() for author in entry["author"].split(" and ")]
        
        # Get venue based on reference type
        venue = ""
        if reference_type == ReferenceType.ARTICLE:
            venue = entry.get("journal", "")
        elif reference_type == ReferenceType.CONFERENCE:
            venue = entry.get("booktitle", "")
        elif reference_type == ReferenceType.BOOK:
            venue = entry.get("publisher", "")
        else:
            venue = entry.get("howpublished", "")
        
        # Get year
        year = 0
        if "year" in entry:
            try:
                year = int(entry["year"])
            except ValueError:
                year = 0
        
        return cls(
            id=entry.get("ID", ""),
            title=entry.get("title", ""),
            authors=authors,
            year=year,
            venue=venue,
            reference_type=reference_type,
            doi=entry.get("doi"),
            url=entry.get("url"),
            pages=entry.get("pages"),
            volume=entry.get("volume"),
            number=entry.get("number"),
            publisher=entry.get("publisher"),
            valid=True
        )