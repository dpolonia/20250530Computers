"""
Unit tests for the PaperService.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.services.paper_service import PaperService
from src.domain.paper import Paper
from tests.fixtures.factories import create_paper


class TestPaperService:
    """Tests for the PaperService."""
    
    @pytest.fixture
    def mock_pdf_adapter(self):
        """Fixture for a mock PDFAdapter."""
        mock = Mock()
        mock.extract_text.return_value = "Extracted text from PDF"
        mock.extract_metadata.return_value = {
            "title": "Test Paper",
            "authors": ["Author One", "Author Two"]
        }
        mock.extract_sections.return_value = {
            "Introduction": "Test introduction",
            "Methods": "Test methods",
            "Results": "Test results",
            "Discussion": "Test discussion"
        }
        mock.extract_references.return_value = [
            {
                "key": "ref1",
                "title": "Reference 1",
                "authors": ["Author A"],
                "year": 2020
            }
        ]
        return mock
    
    @pytest.fixture
    def mock_llm_service(self):
        """Fixture for a mock LLMService."""
        mock = Mock()
        mock.get_completion.return_value = """
        {
            "title": "Enhanced Test Paper",
            "abstract": "This is an extracted abstract.",
            "keywords": ["test", "paper", "academic"]
        }
        """
        return mock
    
    @pytest.fixture
    def paper_service(self, test_context, mock_pdf_adapter, mock_llm_service):
        """Fixture for a PaperService with mock dependencies."""
        service = PaperService(test_context)
        
        # Inject mock dependencies
        service._pdf_adapter = mock_pdf_adapter
        service._llm_service = mock_llm_service
        
        return service
    
    def test_analyze_paper(self, paper_service, mock_pdf_adapter, mock_llm_service):
        """Test analyzing a paper."""
        # Arrange
        # Setup is done in fixtures
        
        # Act
        paper = paper_service.analyze_paper()
        
        # Assert
        assert isinstance(paper, Paper)
        assert paper.title == "Test Paper"  # From mock_pdf_adapter.extract_metadata
        assert len(paper.authors) == 2
        assert len(paper.sections) == 4
        assert "Introduction" in paper.sections
        assert len(paper.references) == 1
        
        # Verify interactions with mocks
        mock_pdf_adapter.extract_text.assert_called_once()
        mock_pdf_adapter.extract_metadata.assert_called_once()
        mock_pdf_adapter.extract_sections.assert_called_once()
        mock_pdf_adapter.extract_references.assert_called_once()
        mock_llm_service.get_completion.assert_called()
    
    def test_extract_abstract(self, paper_service, mock_llm_service):
        """Test extracting abstract from paper."""
        # Arrange
        paper_text = "This is the full text of the paper."
        
        # Act
        abstract = paper_service._extract_abstract(paper_text)
        
        # Assert
        assert abstract == "This is an extracted abstract."
        mock_llm_service.get_completion.assert_called_once()
    
    def test_paper_caching(self, paper_service):
        """Test that paper is cached after first analysis."""
        # Arrange
        # First call to analyze_paper
        paper1 = paper_service.analyze_paper()
        
        # Act
        # Second call should return cached result
        paper2 = paper_service.analyze_paper()
        
        # Assert
        assert paper1 is paper2  # Same instance
        
        # Verify that the adapters are only called once
        assert paper_service._pdf_adapter.extract_text.call_count == 1
        assert paper_service._pdf_adapter.extract_metadata.call_count == 1
    
    @patch('src.services.paper_service.PDFAdapter')
    def test_adapter_creation(self, mock_pdf_adapter_class, test_context):
        """Test that the PDF adapter is created correctly."""
        # Arrange
        mock_pdf_adapter_instance = Mock()
        mock_pdf_adapter_class.return_value = mock_pdf_adapter_instance
        
        # Act
        service = PaperService(test_context)
        
        # Access the adapter to trigger creation
        adapter = service._get_pdf_adapter()
        
        # Assert
        assert adapter is mock_pdf_adapter_instance
        mock_pdf_adapter_class.assert_called_once_with(test_context)
    
    def test_get_paper_metadata(self, paper_service):
        """Test getting paper metadata."""
        # Arrange - already setup in fixtures
        
        # Act
        metadata = paper_service.get_paper_metadata()
        
        # Assert
        assert isinstance(metadata, dict)
        assert "title" in metadata
        assert "authors" in metadata
        assert metadata["title"] == "Test Paper"
        assert len(metadata["authors"]) == 2
    
    def test_get_paper_sections(self, paper_service):
        """Test getting paper sections."""
        # Arrange - analyze paper first
        paper_service.analyze_paper()
        
        # Act
        sections = paper_service.get_paper_sections()
        
        # Assert
        assert isinstance(sections, dict)
        assert len(sections) == 4
        assert "Introduction" in sections
        assert "Methods" in sections
        assert "Results" in sections
        assert "Discussion" in sections