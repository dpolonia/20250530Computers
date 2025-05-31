"""
Integration tests for the paper analysis flow.

These tests verify that the paper analysis components work together correctly.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.services.paper_service import PaperService
from src.services.reviewer_service import ReviewerService
from src.services.solution_service import SolutionService
from src.core.context import RevisionContext
from src.adapters.pdf_adapter import PDFAdapter


class TestPaperAnalysisFlow:
    """Tests for the paper analysis workflow."""
    
    @pytest.fixture
    def mock_adapters(self):
        """Setup mock adapters."""
        # Mock PDF adapter
        mock_pdf = Mock(spec=PDFAdapter)
        mock_pdf.extract_text.return_value = "Extracted text from PDF"
        mock_pdf.extract_metadata.return_value = {
            "title": "Test Paper",
            "authors": ["Author One", "Author Two"]
        }
        mock_pdf.extract_sections.return_value = {
            "Introduction": "Test introduction",
            "Methods": "Test methods",
            "Results": "Test results",
            "Discussion": "Test discussion"
        }
        mock_pdf.extract_references.return_value = [
            {
                "key": "ref1",
                "title": "Reference 1",
                "authors": ["Author A"],
                "year": 2020
            }
        ]
        
        # Mock other adapters as needed
        
        return {
            "pdf_adapter": mock_pdf
        }
    
    @pytest.fixture
    def mock_llm_service(self):
        """Setup mock LLM service."""
        mock = Mock()
        
        # Setup responses for different prompts
        def get_completion_side_effect(prompt, system_prompt=None, max_tokens=None, force_verification=False):
            if "analyze paper" in prompt.lower():
                return """
                {
                    "title": "Enhanced Test Paper",
                    "abstract": "This is an extracted abstract.",
                    "keywords": ["test", "paper", "academic"]
                }
                """
            elif "analyze reviewer" in prompt.lower():
                return """
                {
                    "reviewer_number": 1,
                    "overall_assessment": "minor_revision",
                    "main_concerns": ["Methods lack clarity", "Missing references"],
                    "methodology_comments": ["The methods need more detail"],
                    "results_comments": ["Results are well presented"],
                    "writing_comments": ["The writing is clear"],
                    "references_comments": ["Missing some recent work"]
                }
                """
            elif "identify issues" in prompt.lower():
                return """
                [
                    {
                        "description": "Methods section lacks clarity",
                        "type": "methodology",
                        "severity": "high",
                        "reviewer_number": 1,
                        "section": "Methods"
                    },
                    {
                        "description": "Missing recent references",
                        "type": "references",
                        "severity": "medium",
                        "reviewer_number": 1,
                        "section": "References"
                    }
                ]
                """
            else:
                return "Default mock response"
        
        mock.get_completion.side_effect = get_completion_side_effect
        return mock
    
    @pytest.fixture
    def services(self, test_context, mock_adapters, mock_llm_service):
        """Setup services with mock dependencies."""
        # Create the services
        paper_service = PaperService(test_context)
        reviewer_service = ReviewerService(test_context)
        solution_service = SolutionService(test_context)
        
        # Inject mock dependencies
        paper_service._pdf_adapter = mock_adapters["pdf_adapter"]
        paper_service._llm_service = mock_llm_service
        reviewer_service._llm_service = mock_llm_service
        solution_service._llm_service = mock_llm_service
        
        return {
            "paper_service": paper_service,
            "reviewer_service": reviewer_service,
            "solution_service": solution_service
        }
    
    def test_complete_analysis_flow(self, services):
        """Test the complete paper analysis flow."""
        # Arrange
        paper_service = services["paper_service"]
        reviewer_service = services["reviewer_service"]
        solution_service = services["solution_service"]
        
        # Act - Step 1: Analyze paper
        paper = paper_service.analyze_paper()
        
        # Assert
        assert paper.title == "Test Paper"
        assert len(paper.sections) == 4
        assert paper.abstract == "This is an extracted abstract."
        
        # Act - Step 2: Analyze reviewer comments
        reviewer_comments = reviewer_service.analyze_reviewer_comments()
        
        # Assert
        assert len(reviewer_comments) >= 1
        assert reviewer_comments[0].reviewer_number == 1
        assert reviewer_comments[0].overall_assessment == "minor_revision"
        assert len(reviewer_comments[0].main_concerns) >= 1
        
        # Act - Step 3: Identify issues
        issues = solution_service.identify_issues(
            paper=paper,
            reviewer_comments=reviewer_comments
        )
        
        # Assert
        assert len(issues) >= 1
        assert issues[0].description == "Methods section lacks clarity"
        assert issues[0].type == "methodology"
        assert issues[0].severity == "high"
        
        # Act - Step 4: Generate solutions
        solutions = solution_service.generate_solutions(
            paper=paper,
            issues=issues
        )
        
        # Assert
        assert len(solutions) >= 1
        assert solutions[0].description is not None
        assert solutions[0].issue is issues[0]
    
    @patch('src.services.solution_service.SolutionService._generate_solutions_from_llm')
    def test_error_handling_in_flow(self, mock_generate_solutions, services):
        """Test error handling in the paper analysis flow."""
        # Arrange
        paper_service = services["paper_service"]
        reviewer_service = services["reviewer_service"]
        solution_service = services["solution_service"]
        
        # Setup mock to raise an exception
        mock_generate_solutions.side_effect = Exception("LLM API error")
        
        # Act - Steps 1-2: Analyze paper and reviewer comments
        paper = paper_service.analyze_paper()
        reviewer_comments = reviewer_service.analyze_reviewer_comments()
        issues = solution_service.identify_issues(
            paper=paper,
            reviewer_comments=reviewer_comments
        )
        
        # Act/Assert - Step 3: Generate solutions (should handle error)
        with pytest.raises(Exception) as excinfo:
            solutions = solution_service.generate_solutions(
                paper=paper,
                issues=issues
            )
        
        assert "LLM API error" in str(excinfo.value)
    
    def test_data_consistency_across_services(self, services):
        """Test that data is consistent across services."""
        # Arrange
        paper_service = services["paper_service"]
        reviewer_service = services["reviewer_service"]
        solution_service = services["solution_service"]
        
        # Act - Get data from multiple services
        paper = paper_service.analyze_paper()
        reviewer_comments = reviewer_service.analyze_reviewer_comments()
        issues = solution_service.identify_issues(
            paper=paper,
            reviewer_comments=reviewer_comments
        )
        
        # Assert - Verify references are maintained
        assert issues[0].paper is paper
        assert issues[0].reviewer_number == reviewer_comments[0].reviewer_number