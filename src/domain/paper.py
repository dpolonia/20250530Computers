"""
Paper domain entity.

This module defines the Paper entity, which represents a scientific paper in the
domain model.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import datetime


@dataclass
class Section:
    """Section of a paper."""
    
    name: str
    content: str
    level: int = 1  # Heading level


@dataclass
class Table:
    """Table in a paper."""
    
    caption: str
    content: List[List[str]]  # Rows and columns
    number: int = 0


@dataclass
class Figure:
    """Figure in a paper."""
    
    caption: str
    path: Optional[str] = None  # Path to the figure file
    number: int = 0


@dataclass
class Reference:
    """Bibliographic reference in a paper."""
    
    key: str
    title: str
    authors: List[str]
    year: int
    venue: str  # Journal or conference
    doi: Optional[str] = None
    valid: bool = True


@dataclass
class Paper:
    """
    Domain entity representing a scientific paper.
    
    This entity encapsulates all the information about a scientific paper,
    including metadata, content, and structure.
    """
    
    title: str
    authors: List[str]
    abstract: str
    sections: List[Section] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    figures: List[Figure] = field(default_factory=list)
    references: List[Reference] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    publication_date: Optional[datetime.date] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    objectives: List[str] = field(default_factory=list)
    methodology: str = ""
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize derived attributes."""
        # Initialize section numbers
        for i, table in enumerate(self.tables, 1):
            if table.number == 0:
                table.number = i
        
        # Initialize figure numbers
        for i, figure in enumerate(self.figures, 1):
            if figure.number == 0:
                figure.number = i
    
    @property
    def word_count(self) -> int:
        """Get the total word count of the paper."""
        content = " ".join([section.content for section in self.sections])
        return len(content.split())
    
    @property
    def section_count(self) -> int:
        """Get the number of sections in the paper."""
        return len(self.sections)
    
    @property
    def reference_count(self) -> int:
        """Get the number of references in the paper."""
        return len(self.references)
    
    def get_section_by_name(self, name: str) -> Optional[Section]:
        """
        Get a section by name.
        
        Args:
            name: The name of the section to find
            
        Returns:
            The section if found, None otherwise
        """
        for section in self.sections:
            if section.name.lower() == name.lower():
                return section
        return None
    
    def get_section_content(self, name: str) -> str:
        """
        Get the content of a section by name.
        
        Args:
            name: The name of the section to find
            
        Returns:
            The content of the section if found, empty string otherwise
        """
        section = self.get_section_by_name(name)
        return section.content if section else ""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the paper to a dictionary.
        
        Returns:
            Dictionary representation of the paper
        """
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "sections": [{"name": s.name, "content": s.content, "level": s.level} for s in self.sections],
            "tables": [{"caption": t.caption, "number": t.number} for t in self.tables],
            "figures": [{"caption": f.caption, "number": f.number} for f in self.figures],
            "references": [{"key": r.key, "title": r.title, "authors": r.authors, "year": r.year, "venue": r.venue} 
                           for r in self.references],
            "keywords": self.keywords,
            "objectives": self.objectives,
            "methodology": self.methodology,
            "findings": self.findings,
            "limitations": self.limitations,
            "conclusions": self.conclusions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Paper':
        """
        Create a paper from a dictionary.
        
        Args:
            data: Dictionary with paper data
            
        Returns:
            Paper instance
        """
        sections = [Section(name=s["name"], content=s["content"], level=s.get("level", 1)) 
                   for s in data.get("sections", [])]
        
        tables = [Table(caption=t["caption"], content=[], number=t.get("number", 0)) 
                 for t in data.get("tables", [])]
        
        figures = [Figure(caption=f["caption"], number=f.get("number", 0)) 
                  for f in data.get("figures", [])]
        
        references = [Reference(
            key=r["key"],
            title=r["title"],
            authors=r["authors"],
            year=r["year"],
            venue=r["venue"],
            doi=r.get("doi")
        ) for r in data.get("references", [])]
        
        return cls(
            title=data.get("title", ""),
            authors=data.get("authors", []),
            abstract=data.get("abstract", ""),
            sections=sections,
            tables=tables,
            figures=figures,
            references=references,
            keywords=data.get("keywords", []),
            objectives=data.get("objectives", []),
            methodology=data.get("methodology", ""),
            findings=data.get("findings", []),
            limitations=data.get("limitations", []),
            conclusions=data.get("conclusions", [])
        )