"""
Unit tests for the Paper domain entity.
"""

import pytest
from src.domain.paper import Paper
from tests.fixtures.factories import create_paper, create_reference


class TestPaper:
    """Tests for the Paper domain entity."""
    
    def test_paper_creation(self):
        """Test creating a Paper object with basic attributes."""
        # Arrange
        title = "Test Paper Title"
        authors = ["Author One", "Author Two"]
        abstract = "Test abstract"
        
        # Act
        paper = Paper(title=title, authors=authors, abstract=abstract)
        
        # Assert
        assert paper.title == title
        assert paper.authors == authors
        assert paper.abstract == abstract
        assert paper.sections == {}
        assert paper.references == []
    
    def test_paper_with_sections(self):
        """Test creating a Paper object with sections."""
        # Arrange
        sections = {
            "Introduction": "Test introduction",
            "Methods": "Test methods",
            "Results": "Test results",
            "Discussion": "Test discussion"
        }
        
        # Act
        paper = Paper(title="Test", authors=["Author"], sections=sections)
        
        # Assert
        assert paper.sections == sections
        assert len(paper.sections) == 4
        assert "Introduction" in paper.sections
        assert paper.sections["Introduction"] == "Test introduction"
    
    def test_paper_with_references(self):
        """Test creating a Paper object with references."""
        # Arrange
        references = [
            create_reference(key="ref1", title="Reference 1"),
            create_reference(key="ref2", title="Reference 2")
        ]
        
        # Act
        paper = Paper(title="Test", authors=["Author"], references=references)
        
        # Assert
        assert paper.references == references
        assert len(paper.references) == 2
        assert paper.references[0].key == "ref1"
        assert paper.references[1].title == "Reference 2"
    
    def test_paper_to_dict(self):
        """Test converting a Paper object to a dictionary."""
        # Arrange
        paper = create_paper(
            title="Test Paper",
            authors=["Author One", "Author Two"],
            abstract="Test abstract",
            sections={"Introduction": "Test intro"}
        )
        
        # Act
        paper_dict = paper.to_dict()
        
        # Assert
        assert isinstance(paper_dict, dict)
        assert paper_dict["title"] == "Test Paper"
        assert paper_dict["authors"] == ["Author One", "Author Two"]
        assert paper_dict["abstract"] == "Test abstract"
        assert paper_dict["sections"] == {"Introduction": "Test intro"}
        assert "references" in paper_dict
    
    def test_paper_from_dict(self):
        """Test creating a Paper object from a dictionary."""
        # Arrange
        paper_dict = {
            "title": "Test Paper",
            "authors": ["Author One", "Author Two"],
            "abstract": "Test abstract",
            "sections": {"Introduction": "Test intro"},
            "references": [
                {
                    "key": "ref1",
                    "title": "Reference 1",
                    "authors": ["Author A"],
                    "year": 2020
                }
            ]
        }
        
        # Act
        paper = Paper.from_dict(paper_dict)
        
        # Assert
        assert paper.title == "Test Paper"
        assert paper.authors == ["Author One", "Author Two"]
        assert paper.abstract == "Test abstract"
        assert paper.sections == {"Introduction": "Test intro"}
        assert len(paper.references) == 1
        assert paper.references[0].key == "ref1"
        assert paper.references[0].title == "Reference 1"
    
    def test_paper_str_representation(self):
        """Test the string representation of a Paper object."""
        # Arrange
        paper = create_paper(title="Test Paper", authors=["Author One", "Author Two"])
        
        # Act
        paper_str = str(paper)
        
        # Assert
        assert "Test Paper" in paper_str
        assert "Author One" in paper_str or "Author Two" in paper_str
    
    def test_paper_with_empty_values(self):
        """Test creating a Paper object with empty values."""
        # Arrange & Act
        paper = Paper(title="", authors=[])
        
        # Assert
        assert paper.title == ""
        assert paper.authors == []
        assert paper.abstract is None
        assert paper.sections == {}
        assert paper.references == []
    
    def test_paper_equality(self):
        """Test Paper equality comparison."""
        # Arrange
        paper1 = create_paper(title="Test Paper", authors=["Author"])
        paper2 = create_paper(title="Test Paper", authors=["Author"])
        paper3 = create_paper(title="Different Paper", authors=["Author"])
        
        # Assert
        assert paper1 == paper2
        assert paper1 != paper3
        assert paper1 != "Not a Paper object"